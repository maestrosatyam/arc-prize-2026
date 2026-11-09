"""Causal sequence models for grid transduction: Mamba-2 (SSD), Transformer, GRU and a Mamba/attention hybrid.

All models share the input embedding (token + row + column + role) and a tied output head, and expose
``forward`` (parallel, optional final state) and ``step`` (one token, recurrent state) for cached decoding.
The Mamba-2 scan is plain PyTorch: ``ssd_chunked`` (block decomposition, matmul form) is checked against the
sequential recurrence ``ssd_reference`` in tests/test_scan.py.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

VOCAB = 16
N_ROW = 32
N_COL = 32
N_ROLE = 4


# ----------------------------------------------------------------------------------------------
# Selective scan (Mamba-2 state-space duality form)
# ----------------------------------------------------------------------------------------------


def segsum(x: torch.Tensor) -> torch.Tensor:
    """out[..., i, j] = sum_{k=j+1..i} x[..., k] for j <= i, -inf for j > i (stable masked cumsum)."""
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)  # [..., i, j] = x_i
    below = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~below, 0.0)
    out = torch.cumsum(x, dim=-2)
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~keep, float("-inf"))


def ssd_reference(X, A, B, C, h0=None):
    """Sequential recurrence h_t = exp(A_t) h_{t-1} + X_t B_t^T, y_t = h_t C_t.

    X (b, L, H, P), A (b, L, H) log-decay, B and C (b, L, H, N). Returns y (b, L, H, P), h_L (b, H, P, N).
    """
    b, L, H, P = X.shape
    N = B.shape[-1]
    h = X.new_zeros(b, H, P, N) if h0 is None else h0.clone()
    ys = []
    for t in range(L):
        h = h * torch.exp(A[:, t])[:, :, None, None] + X[:, t, :, :, None] * B[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", h, C[:, t]))
    return torch.stack(ys, dim=1), h


def ssd_chunked(X, A, B, C, chunk: int = 64, h0=None):
    """Chunked SSD scan, same contract as ssd_reference; cost O(L * chunk) instead of a length-L loop.

    B and C may also be (b, L, N), shared by all heads (one group), which skips the per-head copies.
    """
    b, L, H, P = X.shape
    N = B.shape[-1]
    shared = B.dim() == 3
    pad = (-L) % chunk
    if pad:  # zero input and zero log-decay at the tail leave earlier outputs and the final state unchanged
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        bc_pad = (0, 0, 0, pad) if shared else (0, 0, 0, 0, 0, pad)
        B = F.pad(B, bc_pad)
        C = F.pad(C, bc_pad)
    nc = (L + pad) // chunk
    X = X.reshape(b, nc, chunk, H, P)
    bc_shape = (b, nc, chunk, N) if shared else (b, nc, chunk, H, N)
    B = B.reshape(bc_shape)
    C = C.reshape(bc_shape)
    A = A.reshape(b, nc, chunk, H).permute(0, 3, 1, 2)  # b h c l
    A_cum = torch.cumsum(A, dim=-1)

    # 1. within-chunk outputs
    Lmat = torch.exp(segsum(A))  # b h c l s
    if shared:
        CB = torch.einsum("bcln,bcsn->bcls", C, B)[:, None]
    else:
        CB = torch.einsum("bclhn,bcshn->bhcls", C, B)
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", CB * Lmat, X)
    # 2. state at the end of each chunk from its own inputs
    decay_states = torch.exp(A_cum[..., -1:] - A_cum).permute(0, 2, 3, 1)  # b c l h
    Xd = X * decay_states[..., None]
    if shared:
        states = torch.einsum("bcln,bclhp->bchpn", B, Xd)
    else:
        states = torch.einsum("bclhn,bclhp->bchpn", B, Xd)
    # 3. recurrence across chunks
    init = X.new_zeros(b, 1, H, P, N) if h0 is None else h0[:, None]
    states = torch.cat([init, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))  # b h (c+1) (c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final = new_states[:, :-1], new_states[:, -1]
    # 4. contribution of the carried state to each output
    decay_out = torch.exp(A_cum).permute(0, 2, 3, 1)  # b c l h
    if shared:
        Y_off = torch.einsum("bcln,bchpn->bclhp", C, states)
    else:
        Y_off = torch.einsum("bclhn,bchpn->bclhp", C, states)
    Y = (Y_diag + Y_off * decay_out[..., None]).reshape(b, nc * chunk, H, P)[:, :L]
    return Y, final


# ----------------------------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _cdt(t: torch.Tensor) -> torch.dtype:
    """Scan compute dtype: float64 stays float64 (tests), everything else runs in float32."""
    return torch.float64 if t.dtype == torch.float64 else torch.float32


class Mamba2Mixer(nn.Module):
    """Mamba-2 mixer: in_proj -> causal depthwise conv on (x, B, C) -> SSD scan -> gated RMSNorm -> out_proj."""

    def __init__(self, d_model: int, d_state: int = 32, expand: int = 2, headdim: int = 32, d_conv: int = 4,
                 chunk: int = 64, scan: str = "chunked"):
        super().__init__()
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.H = self.d_inner // headdim
        self.P = headdim
        self.N = d_state
        self.d_conv = d_conv
        self.chunk = chunk
        self.scan = scan
        self.conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.H, bias=False)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, groups=self.conv_dim, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _split(self, zxbcdt):
        return torch.split(zxbcdt, [self.d_inner, self.conv_dim, self.H], dim=-1)

    def forward(self, u, return_state: bool = False):
        b, L, _ = u.shape
        z, xBC_raw, dt = self._split(self.in_proj(u))
        xBC = F.silu(self.conv(xBC_raw.transpose(1, 2))[..., :L].transpose(1, 2))
        # the scan runs in float32 even under mixed precision (cumulative log-decays and exp)
        with torch.autocast(device_type=u.device.type, enabled=False):
            cdt = _cdt(self.A_log)
            x, Bm, Cm = torch.split(xBC.to(cdt), [self.d_inner, self.N, self.N], dim=-1)
            dt = F.softplus(dt.to(cdt) + self.dt_bias)  # b L H
            A = -torch.exp(self.A_log)
            x = x.reshape(b, L, self.H, self.P)
            if self.scan == "chunked":
                y, h = ssd_chunked(x * dt[..., None], A * dt, Bm, Cm, chunk=self.chunk)
            else:
                Bh = Bm[:, :, None, :].expand(b, L, self.H, self.N)
                Ch = Cm[:, :, None, :].expand(b, L, self.H, self.N)
                y, h = ssd_reference(x * dt[..., None], A * dt, Bh, Ch)
            y = (y + x * self.D[:, None]).reshape(b, L, self.d_inner)
        out = self.out_proj(self.norm(y.to(z.dtype) * F.silu(z)))
        if not return_state:
            return out
        k = self.d_conv - 1
        tail = xBC_raw[:, -k:].transpose(1, 2).to(_cdt(self.A_log))
        if tail.shape[-1] < k:
            tail = F.pad(tail, (k - tail.shape[-1], 0))
        return out, (tail.contiguous(), h)

    def step(self, u, state):
        """u (b, d_model); state = (conv window (b, conv_dim, d_conv-1), h (b, H, P, N))."""
        conv_win, h = state
        z, xBC_raw, dt = self._split(self.in_proj(u))
        with torch.autocast(device_type=u.device.type, enabled=False):
            cdt = _cdt(self.A_log)
            win = torch.cat([conv_win.to(cdt), xBC_raw[:, :, None].to(cdt)], dim=-1)  # b conv_dim d_conv
            xBC = F.silu((win * self.conv.weight[:, 0, :].to(cdt)).sum(-1) + self.conv.bias.to(cdt))
            x, Bm, Cm = torch.split(xBC, [self.d_inner, self.N, self.N], dim=-1)
            dt = F.softplus(dt.to(cdt) + self.dt_bias)  # b H
            A = -torch.exp(self.A_log)
            x = x.reshape(-1, self.H, self.P)
            h = h * torch.exp(A * dt)[:, :, None, None] + (x * dt[..., None])[..., None] * Bm[:, None, None, :]
            y = torch.einsum("bhpn,bn->bhp", h, Cm) + x * self.D[:, None]
        out = self.out_proj(self.norm(y.reshape(-1, self.d_inner).to(z.dtype) * F.silu(z)))
        return out, (win[:, :, 1:], h)


def rope_tables(L: int, dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.arange(L).float()[:, None] * inv[None]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):  # x (b, h, L, dh); cos/sin (L, dh/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 4096):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        cos, sin = rope_tables(max_len, self.dh)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _qkv(self, x, pos0: int):
        b, L, d = x.shape
        q, k, v = self.qkv(x).view(b, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        cos, sin = self.cos[pos0:pos0 + L], self.sin[pos0:pos0 + L]
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, x, return_state: bool = False):
        b, L, d = x.shape
        q, k, v = self._qkv(x, 0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = self.o(y.transpose(1, 2).reshape(b, L, d))
        return (out, (k, v)) if return_state else out

    def step(self, u, state):
        k_c, v_c = state
        x = u[:, None]
        q, k, v = self._qkv(x, k_c.shape[2])
        k_c = torch.cat([k_c, k], dim=2)
        v_c = torch.cat([v_c, v], dim=2)
        y = F.scaled_dot_product_attention(q, k_c, v_c)
        return self.o(y.transpose(1, 2).reshape(u.shape[0], -1)), (k_c, v_c)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        a, g = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * a)


class GRUMixer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)

    def forward(self, x, return_state: bool = False):
        y, h = self.rnn(x)
        return (y, h) if return_state else y

    def step(self, u, state):
        y, h = self.rnn(u[:, None], state)
        return y[:, 0], h


class Block(nn.Module):
    """Pre-norm residual block: mixer, then an optional SwiGLU MLP."""

    def __init__(self, kind: str, d_model: int, cfg: "ModelConfig"):
        super().__init__()
        self.kind = kind
        self.n1 = RMSNorm(d_model)
        if kind == "mamba":
            self.mixer = Mamba2Mixer(d_model, cfg.d_state, cfg.expand, cfg.headdim, cfg.d_conv, cfg.chunk, cfg.scan)
        elif kind == "attn":
            self.mixer = Attention(d_model, cfg.n_heads, cfg.max_len)
        elif kind == "gru":
            self.mixer = GRUMixer(d_model)
        else:
            raise ValueError(kind)
        self.mlp = None
        if cfg.mlp_mult > 0 and kind in cfg.mlp_kinds:
            self.n2 = RMSNorm(d_model)
            self.mlp = SwiGLU(d_model, int(cfg.mlp_mult * d_model))

    def forward(self, x, return_state: bool = False):
        if return_state:
            m, st = self.mixer(self.n1(x), return_state=True)
        else:
            m, st = self.mixer(self.n1(x)), None
        x = x + m
        if self.mlp is not None:
            x = x + self.mlp(self.n2(x))
        return (x, st) if return_state else x

    def step(self, x, state):
        m, st = self.mixer.step(self.n1(x), state)
        x = x + m
        if self.mlp is not None:
            x = x + self.mlp(self.n2(x))
        return x, st


@dataclass
class ModelConfig:
    arch: str = "mamba"  # mamba | transformer | gru | hybrid
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 32
    expand: int = 2
    headdim: int = 32
    d_conv: int = 4
    chunk: int = 64
    scan: str = "chunked"
    n_heads: int = 4
    mlp_mult: float = 0.0
    mlp_kinds: tuple = ("attn",)
    attn_layers: tuple = ()  # hybrid: indices of attention layers
    max_len: int = 4096

    def layer_kinds(self) -> list[str]:
        if self.arch == "mamba":
            return ["mamba"] * self.n_layers
        if self.arch == "transformer":
            return ["attn"] * self.n_layers
        if self.arch == "gru":
            return ["gru"] * self.n_layers
        if self.arch == "hybrid":
            return ["attn" if i in self.attn_layers else "mamba" for i in range(self.n_layers)]
        raise ValueError(self.arch)

    def to_dict(self):
        d = asdict(self)
        d["mlp_kinds"] = list(self.mlp_kinds)
        d["attn_layers"] = list(self.attn_layers)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["mlp_kinds"] = tuple(d.get("mlp_kinds", ("attn",)))
        d["attn_layers"] = tuple(d.get("attn_layers", ()))
        return cls(**d)


class SeqModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(VOCAB, d)
        self.row = nn.Embedding(N_ROW, d)
        self.col = nn.Embedding(N_COL, d)
        self.role = nn.Embedding(N_ROLE, d)
        for e in (self.tok, self.row, self.col, self.role):
            nn.init.normal_(e.weight, std=0.02)
        self.layers = nn.ModuleList(Block(k, d, cfg) for k in cfg.layer_kinds())
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.grad_ckpt = False  # recompute each block in backward (memory; same math)

    def embed(self, tok, row, col, role):
        return self.tok(tok) + self.row(row) + self.col(col) + self.role(role)

    def forward(self, tok, row, col, role, return_state: bool = False):
        x = self.embed(tok, row, col, role)
        states = []
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled() and not return_state
        for layer in self.layers:
            if return_state:
                x, st = layer(x, return_state=True)
                states.append(st)
            elif ckpt:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        logits = self.head(self.norm_f(x))
        return (logits, states) if return_state else logits

    def step(self, tok, row, col, role, states):
        x = self.embed(tok, row, col, role)
        new = []
        for layer, st in zip(self.layers, states):
            x, st = layer.step(x, st)
            new.append(st)
        return self.head(self.norm_f(x)), new


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(cfg: ModelConfig | dict) -> SeqModel:
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return SeqModel(cfg)
