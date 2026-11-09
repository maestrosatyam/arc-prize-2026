"""Chunked SSD scan against the sequential reference; parallel forward against step-by-step decoding."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcssm.models import ModelConfig, build, segsum, ssd_chunked, ssd_reference  # noqa: E402

torch.set_num_threads(2)


def _inputs(b=2, L=150, H=3, P=5, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, L, H, P, generator=g, dtype=torch.float64)
    A = -torch.rand(b, L, H, generator=g, dtype=torch.float64) * 0.5
    B = torch.randn(b, L, H, N, generator=g, dtype=torch.float64)
    C = torch.randn(b, L, H, N, generator=g, dtype=torch.float64)
    return X, A, B, C


def test_segsum_matches_definition():
    x = torch.randn(3, 6, dtype=torch.float64)
    s = segsum(x)
    for i in range(6):
        for j in range(6):
            want = x[:, j + 1:i + 1].sum(-1) if j <= i else torch.full((3,), float("-inf"), dtype=torch.float64)
            assert torch.allclose(s[:, i, j], want)


def test_chunked_matches_reference():
    X, A, B, C = _inputs()
    y_ref, h_ref = ssd_reference(X, A, B, C)
    for chunk in (1, 7, 16, 64, 256):
        y, h = ssd_chunked(X, A, B, C, chunk=chunk)
        assert torch.allclose(y, y_ref, atol=1e-9, rtol=1e-9), chunk
        assert torch.allclose(h, h_ref, atol=1e-9, rtol=1e-9), chunk


def test_chunked_shared_bc_matches_reference():
    X, A, B, C = _inputs(L=77, seed=4)
    Bs, Cs = B[:, :, 0], C[:, :, 0]  # one group shared by all heads
    H = X.shape[2]
    y_ref, h_ref = ssd_reference(X, A, Bs[:, :, None].expand(-1, -1, H, -1), Cs[:, :, None].expand(-1, -1, H, -1))
    for chunk in (8, 32, 128):
        y, h = ssd_chunked(X, A, Bs, Cs, chunk=chunk)
        assert torch.allclose(y, y_ref, atol=1e-9), chunk
        assert torch.allclose(h, h_ref, atol=1e-9), chunk


def test_chunked_with_initial_state_and_split():
    X, A, B, C = _inputs(L=97, seed=1)
    y_full, h_full = ssd_reference(X, A, B, C)
    y1, h1 = ssd_chunked(X[:, :40], A[:, :40], B[:, :40], C[:, :40], chunk=16)
    y2, h2 = ssd_chunked(X[:, 40:], A[:, 40:], B[:, 40:], C[:, 40:], chunk=16, h0=h1)
    assert torch.allclose(torch.cat([y1, y2], 1), y_full, atol=1e-9)
    assert torch.allclose(h2, h_full, atol=1e-9)


def test_chunked_gradients_match_reference():
    X, A, B, C = _inputs(L=40, seed=2)
    grads = []
    for fn in (lambda *a: ssd_reference(*a), lambda *a: ssd_chunked(*a, chunk=16)):
        ins = [t.clone().requires_grad_(True) for t in (X, A, B, C)]
        y, h = fn(*ins)
        (y.pow(2).sum() + h.sum()).backward()
        grads.append([t.grad for t in ins])
    for g1, g2 in zip(*grads):
        assert torch.allclose(g1, g2, atol=1e-8)


def _seq(b, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, 16, (b, L), generator=g)
    row = torch.randint(0, 32, (b, L), generator=g)
    col = torch.randint(0, 32, (b, L), generator=g)
    role = torch.randint(0, 4, (b, L), generator=g)
    return tok, row, col, role


def test_step_matches_forward_all_archs():
    cfgs = [
        ModelConfig(arch="mamba", d_model=32, n_layers=2, d_state=8, headdim=16, chunk=8),
        ModelConfig(arch="transformer", d_model=32, n_layers=2, n_heads=4, mlp_mult=2.0),
        ModelConfig(arch="gru", d_model=32, n_layers=2),
        ModelConfig(arch="hybrid", d_model=32, n_layers=3, d_state=8, headdim=16, chunk=8, attn_layers=(1,),
                    n_heads=4, mlp_mult=2.0),
    ]
    for cfg in cfgs:
        torch.manual_seed(0)
        m = build(cfg).double().eval()
        tok, row, col, role = _seq(2, 30)
        with torch.no_grad():
            full = m(tok, row, col, role)
            # prefill 17 tokens, then step the remaining ones
            logits, states = m(tok[:, :17], row[:, :17], col[:, :17], role[:, :17], return_state=True)
            assert torch.allclose(logits, full[:, :17], atol=1e-9), cfg.arch
            outs = []
            for t in range(17, 30):
                lt, states = m.step(tok[:, t], row[:, t], col[:, t], role[:, t], states)
                outs.append(lt)
            assert torch.allclose(torch.stack(outs, 1), full[:, 17:], atol=1e-8), cfg.arch


def test_reference_scan_option_matches_chunked_model():
    cfg = ModelConfig(arch="mamba", d_model=32, n_layers=2, d_state=8, headdim=16, chunk=8)
    torch.manual_seed(0)
    m = build(cfg).double().eval()
    tok, row, col, role = _seq(1, 21, seed=3)
    with torch.no_grad():
        a = m(tok, row, col, role)
        for layer in m.layers:
            layer.mixer.scan = "reference"
        b = m(tok, row, col, role)
    assert torch.allclose(a, b, atol=1e-9)


def test_grad_checkpoint_matches_plain_backward():
    for cfg in (ModelConfig(arch="mamba", d_model=32, n_layers=2, d_state=8, headdim=16, chunk=8),
                ModelConfig(arch="hybrid", d_model=32, n_layers=3, d_state=8, headdim=16, chunk=8, attn_layers=(1,),
                            n_heads=4, mlp_mult=2.0)):
        grads = []
        for ckpt in (False, True):
            torch.manual_seed(0)
            m = build(cfg).double().train()
            m.grad_ckpt = ckpt
            tok, row, col, role = _seq(2, 25, seed=5)
            m(tok, row, col, role).pow(2).mean().backward()
            grads.append([p.grad.clone() for p in m.parameters()])
        assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(*grads)), cfg.arch


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
