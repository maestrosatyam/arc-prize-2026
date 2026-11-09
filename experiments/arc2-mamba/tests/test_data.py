"""Encoding round trips, augmentation inverses, generator validity and determinism, split properties."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcssm.encoding import (BOS, EOS, NL, OUT, ROLE_OUT, Aug, collate, encode, encoded_len, loss_targets,  # noqa: E402
                             random_aug)
from arcssm.generators import FAMILIES, make_task, sample_task, valid  # noqa: E402


def test_encode_layout_and_length():
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    b = np.array([[7]], dtype=np.int8)
    e = encode([(a, b)], a, b)
    assert e.tok[0] == BOS
    assert len(e) == encoded_len([(a, b)], a, b.shape)
    # query output follows the prefix: 7, NL, EOS
    assert list(e.tok[e.prefix_len:]) == [7, NL, EOS]
    assert e.tok[e.prefix_len - 1] == OUT
    tgt, mask = loss_targets(e)
    assert mask.sum() == 2 * 3  # two output grids of 1 cell + NL + EOS each
    assert all(e.role[1:][mask] == ROLE_OUT)
    # row/col of the input grid cells
    i0 = 2
    assert list(e.row[i0:i0 + 9]) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert list(e.col[i0:i0 + 9]) == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_aug_inverse():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 10, (4, 7)).astype(np.int8)
    for _ in range(50):
        a = random_aug(rng, keep_bg=bool(rng.random() < 0.5))
        assert np.array_equal(a.inv(a.fwd(g)), g)
    for k in range(8):
        a = Aug(k)
        assert np.array_equal(a.inv(a.fwd(g)), g)
    assert len({Aug(k).fwd(g).tobytes() + bytes(Aug(k).fwd(g).shape) for k in range(8)}) == 8


def test_generators_valid_and_deterministic():
    for fam in FAMILIES:
        made = 0
        for seed in range(15):
            p1 = make_task(np.random.default_rng(seed), fam, 4, 10)
            p2 = make_task(np.random.default_rng(seed), fam, 4, 10)
            if p1 is None:
                assert p2 is None
                continue
            made += 1
            assert len(p1) == 4
            for (i1, o1), (i2, o2) in zip(p1, p2):
                assert np.array_equal(i1, i2) and np.array_equal(o1, o2)
                assert valid(i1) and valid(o1)
                assert i1.min() >= 0 and i1.max() <= 9 and o1.min() >= 0 and o1.max() <= 9
            assert not all(i.shape == o.shape and np.array_equal(i, o) for i, o in p1)
        assert made >= 10, (fam, made)


def test_sample_task_mix():
    rng = np.random.default_rng(1)
    fams = {sample_task(rng, 10)[0] for _ in range(300)}
    assert len(fams) >= 20


def test_collate_padding():
    a = np.array([[1]], dtype=np.int8)
    e1 = encode([(a, a)], a, a)
    e2 = encode([(a, a), (a, a)], a, a)
    tok, row, col, role, tgt, mask = collate([e1, e2])
    assert tok.shape == (2, len(e2))
    assert mask[0].sum() == loss_targets(e1)[1].sum() and mask[1].sum() == loss_targets(e2)[1].sum()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
