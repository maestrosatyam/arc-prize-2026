"""Constrained greedy decoding, augmentation voting (two attempts) and test-time training."""
from __future__ import annotations

import contextlib
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from arcssm.encoding import (EOS, IN, NL, OUT, PAD, ROLE_OUT, IDENTITY, collate, encode, encoded_len, grid_len,
                             random_aug)

MAX_LEN = int(os.environ.get("ARC2_MAMBA_MAX_LEN", "2048"))


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def amp(device: torch.device):
    """bfloat16 autocast on CUDA; a no-op on CPU (the CPU runs are float32)."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_dev(arrays, device):
    return [(a if torch.is_tensor(a) else torch.from_numpy(a)).to(device, non_blocking=True) for a in arrays]


# ----------------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------------


@torch.no_grad()
def decode_group(model, encs, max_rows: int = 30, max_cols: int = 30):
    """Greedy decoding for prompts of equal length; the grammar forces a rectangular grid of at most 30x30.

    Returns [(grid | None, mean log-prob)] in input order.
    """
    dev = device_of(model)
    tok, row, col, role = to_dev(collate(encs, with_targets=False)[:4], dev)
    with amp(dev):
        logits, states = model(tok, row, col, role, return_state=True)
    last = logits[:, -1]
    b = len(encs)
    width = [None] * b
    r_cur = [0] * b
    c_cur = [0] * b
    done = [False] * b
    cells = [[] for _ in range(b)]
    lp_sum = [0.0] * b
    n_tok = [0] * b
    neg = float("-inf")
    for _ in range(max_rows * (max_cols + 1) + 1):
        logp = F.log_softmax(last.float(), dim=-1)
        mask = torch.full(logp.shape, neg)
        for i in range(b):
            if done[i]:
                mask[i, EOS] = 0.0
                continue
            r, c, w = r_cur[i], c_cur[i], width[i]
            if r >= max_rows:
                mask[i, EOS] = 0.0
                continue
            if c == 0 and r >= 1:
                mask[i, EOS] = 0.0
            if (w is None and c < max_cols) or (w is not None and c < w):
                mask[i, :10] = 0.0
            if (w is None and c >= 1) or (w is not None and c == w):
                mask[i, NL] = 0.0
        nxt_d = (logp + mask.to(logp.device)).argmax(-1)
        chosen = logp.gather(1, nxt_d[:, None])[:, 0]
        nxt, chosen = nxt_d.tolist(), chosen.tolist()  # one host sync per step
        t_tok = torch.tensor(nxt, dtype=torch.long)
        t_row = torch.zeros(b, dtype=torch.long)
        t_col = torch.zeros(b, dtype=torch.long)
        for i in range(b):
            if done[i]:
                t_tok[i] = EOS
                t_row[i] = r_cur[i]
                continue
            t = int(nxt[i])
            lp_sum[i] += float(chosen[i])
            n_tok[i] += 1
            if t < 10:
                t_row[i], t_col[i] = r_cur[i], c_cur[i]
                cells[i].append(t)
                c_cur[i] += 1
            elif t == NL:
                t_row[i], t_col[i] = r_cur[i], c_cur[i]
                if width[i] is None:
                    width[i] = c_cur[i]
                r_cur[i] += 1
                c_cur[i] = 0
            else:  # EOS
                t_row[i], t_col[i] = r_cur[i], 0
                done[i] = True
        if all(done):
            break
        with amp(dev):
            last, states = model.step(t_tok.to(dev), t_row.to(dev), t_col.to(dev),
                                      torch.full((b,), ROLE_OUT, dtype=torch.long, device=dev), states)
    out = []
    for i in range(b):
        if width[i] is None or r_cur[i] == 0:
            out.append((None, lp_sum[i] / max(n_tok[i], 1)))
            continue
        g = np.asarray(cells[i][: r_cur[i] * width[i]], dtype=np.int8).reshape(r_cur[i], width[i])
        out.append((g, lp_sum[i] / max(n_tok[i], 1)))
    return out


def decode_many(model, encs, max_batch: int = 8):
    """Group prompts by length (needed for exact recurrent state) and decode each group."""
    order = {}
    for i, e in enumerate(encs):
        order.setdefault(len(e), []).append(i)
    res = [None] * len(encs)
    for _, idx in sorted(order.items()):
        for j in range(0, len(idx), max_batch):
            part = idx[j:j + max_batch]
            for i, r in zip(part, decode_group(model, [encs[k] for k in part])):
                res[i] = r
    return res


# ----------------------------------------------------------------------------------------------
# Prompts, voting, metrics
# ----------------------------------------------------------------------------------------------


def est_out_len(train_pairs, test_in) -> int:
    if all(i.shape == o.shape for i, o in train_pairs):
        return grid_len(test_in.shape) + 1
    return max(grid_len(o.shape) for _, o in train_pairs) + 1


def build_prompt(train_pairs, test_in, aug, max_len: int = MAX_LEN, order=None):
    pairs = [(aug.fwd(i), aug.fwd(o)) for i, o in train_pairs]
    if order is not None:
        pairs = [pairs[k] for k in order]
    q = aug.fwd(test_in)
    need = est_out_len(pairs, q)
    while pairs and encoded_len(pairs, q, None) + need > max_len:
        pairs.pop()
    if not pairs:
        return None
    return encode(pairs, q)


def augs_for(task_key: str, n_aug: int, n_pairs: int):
    """Deterministic list of (aug, pair order); the first is the identity in the original order."""
    seed = int.from_bytes(task_key.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
    rng = np.random.default_rng(seed)
    out = [(IDENTITY, list(range(n_pairs)))]
    while len(out) < n_aug:
        out.append((random_aug(rng), [int(x) for x in rng.permutation(n_pairs)]))
    return out


def vote(preds):
    """preds: [(grid | None, mean_lp)] already in the original frame -> ranked [(grid, count, mean_lp)]."""
    table = {}
    for g, lp in preds:
        if g is None:
            continue
        key = (g.shape, g.tobytes())
        cnt, best, grid = table.get(key, (0, -1e9, g))
        table[key] = (cnt + 1, max(best, lp), grid)
    ranked = sorted(table.values(), key=lambda x: (-x[0], -x[1]))
    return [(g, c, lp) for c, lp, g in ranked]


def predict(model, train_pairs, test_in, task_key: str, n_aug: int = 8, max_len: int = MAX_LEN):
    """Two attempts for one test input by voting over augmented prompts; None when no prompt fits."""
    encs, augs = [], []
    for aug, order in augs_for(task_key, n_aug, len(train_pairs)):
        e = build_prompt(train_pairs, test_in, aug, max_len, order)
        if e is not None:
            encs.append(e)
            augs.append(aug)
    if not encs:
        return None
    raw = decode_many(model, encs)
    preds = [(aug.inv(g) if g is not None else None, lp) for (g, lp), aug in zip(raw, augs)]
    ranked = vote(preds)
    return {"top": [g for g, _, _ in ranked[:2]], "votes": [c for _, c, _ in ranked[:2]], "n_prompts": len(encs),
            "n_distinct": len(ranked)}


def score(top, truth):
    """exact@1, exact@2, shape@1, cell accuracy of attempt 1 (cells equal at the same position / truth cells)."""
    if not top:
        return {"exact1": 0, "exact2": 0, "shape1": 0, "shape2": 0, "cells_ok": 0, "cells": int(truth.size)}
    eq = [g.shape == truth.shape and np.array_equal(g, truth) for g in top]
    g = top[0]
    h, w = min(g.shape[0], truth.shape[0]), min(g.shape[1], truth.shape[1])
    ok = int((g[:h, :w] == truth[:h, :w]).sum())
    return {"exact1": int(eq[0]), "exact2": int(any(eq)), "shape1": int(g.shape == truth.shape),
            "shape2": int(any(x.shape == truth.shape for x in top)), "cells_ok": ok, "cells": int(truth.size)}


# ----------------------------------------------------------------------------------------------
# Test-time training
# ----------------------------------------------------------------------------------------------


def ttt_example(rng, pairs, max_len: int = MAX_LEN):
    a = random_aug(rng)
    ps = [(a.fwd(i), a.fwd(o)) for i, o in pairs]
    ps = [ps[k] for k in rng.permutation(len(ps))]
    q = ps[-1]
    ctx = ps[:-1]
    while ctx and encoded_len(ctx, q[0], q[1].shape) > max_len:
        ctx.pop()
    if encoded_len(ctx, q[0], q[1].shape) > max_len:
        return None
    return encode(ctx, q[0], q[1])


def _loss(model, encs):
    dev = device_of(model)
    tok, row, col, role, tgt, mask = to_dev(collate(encs), dev)
    with amp(dev):
        logits = model(tok[:, :-1], row[:, :-1], col[:, :-1], role[:, :-1])
    lp = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
    m = mask.float()
    return (lp * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def query_nll(model, ctx_pairs, q_in, q_out, max_len: int = MAX_LEN):
    """Teacher-forced mean NLL of the query output given the context pairs (identity frame)."""
    ctx = list(ctx_pairs)
    while ctx and encoded_len(ctx, q_in, q_out.shape) > max_len:
        ctx.pop()
    e = encode(ctx, q_in, q_out)
    dev = device_of(model)
    tok, row, col, role, tgt, _ = to_dev(collate([e]), dev)
    with amp(dev):
        logits = model(tok[:, :-1], row[:, :-1], col[:, :-1], role[:, :-1])
    sl = slice(e.prefix_len - 1, len(e) - 1)
    return float(F.cross_entropy(logits[0, sl].float(), tgt[0, sl]))


def finetune(model, pairs, steps: int, rng, lr: float = 1e-3, bs: int = 2, checkpoints=(), on_checkpoint=None,
             max_len: int = MAX_LEN):
    m = copy.deepcopy(model)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
    if 0 in checkpoints and on_checkpoint is not None:
        m.eval()
        on_checkpoint(0, m)
        m.train()
    for s in range(1, steps + 1):
        encs = [e for e in (ttt_example(rng, pairs, max_len) for _ in range(bs)) if e is not None]
        if not encs:
            break
        loss = _loss(m, encs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s in checkpoints and on_checkpoint is not None:
            m.eval()
            on_checkpoint(s, m)
            m.train()
    m.eval()
    return m


def choose_steps(model, pairs, rng, grid=(0, 4, 8, 16), max_folds: int = 2, lr: float = 1e-3, bs: int = 2,
                 default: int = 8):
    """Leave-one-pair-out inside the train pairs: fine-tune on the others, read the held-out pair's NLL at each
    checkpoint, return the step count with the lowest mean NLL (ties -> fewer steps) and the per-fold table."""
    n = len(pairs)
    if n < 2:
        return default, {}
    folds = list(range(n - 1, -1, -1))[:max_folds]
    table = {s: [] for s in grid}
    for i in folds:
        rest = [p for j, p in enumerate(pairs) if j != i]
        q_in, q_out = pairs[i]
        finetune(model, rest, max(grid), rng, lr, bs, checkpoints=grid,
                 on_checkpoint=lambda s, m: table[s].append(query_nll(m, rest, q_in, q_out)))
    mean = {s: float(np.mean(v)) for s, v in table.items() if v}
    best = min(mean, key=lambda s: (mean[s], s))
    return best, mean


def ttt_predict(model, train_pairs, test_inputs, task_key: str, rng, n_aug: int = 8, grid=(0, 4, 8, 16),
                max_folds: int = 2, lr: float = 1e-3, bs: int = 2):
    t0 = time.process_time()
    steps, table = choose_steps(model, train_pairs, rng, grid, max_folds, lr, bs)
    t_sel = time.process_time() - t0
    m = finetune(model, train_pairs, steps, rng, lr, bs) if steps > 0 else model
    preds = [predict(m, train_pairs, x, f"{task_key}:{k}", n_aug) for k, x in enumerate(test_inputs)]
    return preds, {"steps": steps, "loo_nll": table, "cpu_select_s": t_sel, "cpu_total_s": time.process_time() - t0}
