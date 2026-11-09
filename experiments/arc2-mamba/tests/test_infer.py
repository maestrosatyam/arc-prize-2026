"""Decoding grammar, voting, scoring and TTT isolation on tiny untrained models."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcssm.encoding import encode  # noqa: E402
from arcssm.infer import decode_group, finetune, predict, score, vote  # noqa: E402
from arcssm.models import ModelConfig, build  # noqa: E402

torch.set_num_threads(2)


def _pairs(rng, n=3, h=3, w=4):
    out = []
    for _ in range(n):
        g = rng.integers(0, 10, (h, w)).astype(np.int8)
        out.append((g, g[:, ::-1].copy()))
    return out


def test_decode_returns_rectangular_grids():
    rng = np.random.default_rng(0)
    for arch in ("mamba", "transformer", "gru"):
        torch.manual_seed(0)
        m = build(ModelConfig(arch=arch, d_model=32, n_layers=2, d_state=8, headdim=16, chunk=8, n_heads=4,
                              mlp_mult=2.0)).eval()
        pairs = _pairs(rng)
        encs = [encode(pairs[:2], pairs[2][0]) for _ in range(2)]
        for g, lp in decode_group(m, encs, max_rows=6, max_cols=6):
            assert g is not None and g.ndim == 2 and 1 <= g.shape[0] <= 6 and 1 <= g.shape[1] <= 6
            assert g.min() >= 0 and g.max() <= 9 and np.isfinite(lp)


def test_predict_two_attempts_distinct():
    rng = np.random.default_rng(1)
    torch.manual_seed(0)
    m = build(ModelConfig(arch="mamba", d_model=32, n_layers=2, d_state=8, headdim=16, chunk=8)).eval()
    pairs = _pairs(rng)
    p = predict(m, pairs[:2], pairs[2][0], "t:0", n_aug=4)
    assert p is not None and 1 <= len(p["top"]) <= 2 and p["n_prompts"] == 4
    if len(p["top"]) == 2:
        a, b = p["top"]
        assert a.shape != b.shape or not np.array_equal(a, b)


def test_vote_orders_by_count_then_logprob():
    a = np.zeros((2, 2), dtype=np.int8)
    b = np.ones((2, 2), dtype=np.int8)
    ranked = vote([(a, -1.0), (b, -0.1), (a.copy(), -2.0), (None, 0.0)])
    assert np.array_equal(ranked[0][0], a) and ranked[0][1] == 2
    ranked = vote([(a, -1.0), (b, -0.1)])
    assert np.array_equal(ranked[0][0], b)


def test_score_counts():
    t = np.arange(6, dtype=np.int8).reshape(2, 3)
    s = score([t.copy()], t)
    assert s == {"exact1": 1, "exact2": 1, "shape1": 1, "shape2": 1, "cells_ok": 6, "cells": 6}
    wrong = t.copy()
    wrong[0, 0] = 9
    s = score([wrong, t], t)
    assert s["exact1"] == 0 and s["exact2"] == 1 and s["cells_ok"] == 5


def test_finetune_leaves_base_model_unchanged():
    rng = np.random.default_rng(2)
    torch.manual_seed(0)
    m = build(ModelConfig(arch="gru", d_model=32, n_layers=1))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ft = finetune(m, _pairs(rng), 2, rng, lr=1e-2, bs=2)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in ft.state_dict().items())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
