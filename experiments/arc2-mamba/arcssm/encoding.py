"""Task serialisation and augmentation.

Sequence: BOS, then per pair IN <input grid> OUT <output grid> EOS, then IN <query input> OUT and the query
output grid followed by EOS. A grid is its rows in order, each row followed by NL. Every token carries its
row, column and role (marker, input grid, output grid) as extra embedding indices. The training loss covers
every output-grid token (all pairs, including NL and EOS).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NL, IN, OUT, EOS, BOS, PAD = 10, 11, 12, 13, 14, 15
ROLE_MARK, ROLE_IN, ROLE_OUT = 0, 1, 2
MARK_POS = 31
MAX_SIDE = 30


def grid_tokens(g: np.ndarray, role: int):
    h, w = g.shape
    tok = np.empty((h, w + 1), dtype=np.int64)
    tok[:, :w] = g
    tok[:, w] = NL
    rows = np.repeat(np.arange(h), w + 1).reshape(h, w + 1)
    cols = np.tile(np.arange(w + 1), (h, 1))
    return tok.ravel(), rows.ravel(), cols.ravel(), np.full(h * (w + 1), role, dtype=np.int64)


def grid_len(shape) -> int:
    h, w = shape
    return h * (w + 1)


@dataclass
class Encoded:
    tok: np.ndarray
    row: np.ndarray
    col: np.ndarray
    role: np.ndarray
    prefix_len: int  # tokens up to and including the query OUT marker

    def __len__(self):
        return len(self.tok)


def encode(pairs: list[tuple[np.ndarray, np.ndarray]], query_in: np.ndarray, query_out: np.ndarray | None = None) -> Encoded:
    parts = [([BOS], [MARK_POS], [MARK_POS], [ROLE_MARK])]

    def mark(t):
        parts.append(([t], [MARK_POS], [MARK_POS], [ROLE_MARK]))

    def out_end(g):
        parts.append(([EOS], [g.shape[0]], [0], [ROLE_OUT]))

    for gi, go in pairs:
        mark(IN)
        parts.append(grid_tokens(gi, ROLE_IN))
        mark(OUT)
        parts.append(grid_tokens(go, ROLE_OUT))
        out_end(go)
    mark(IN)
    parts.append(grid_tokens(query_in, ROLE_IN))
    mark(OUT)
    prefix_len = sum(len(p[0]) for p in parts)
    if query_out is not None:
        parts.append(grid_tokens(query_out, ROLE_OUT))
        out_end(query_out)
    cat = [np.concatenate([np.asarray(p[i], dtype=np.int64) for p in parts]) for i in range(4)]
    return Encoded(cat[0], cat[1], cat[2], cat[3], prefix_len)


def encoded_len(pairs, query_in, query_out_shape) -> int:
    n = 1
    for gi, go in pairs:
        n += 2 + grid_len(gi.shape) + grid_len(go.shape) + 1
    n += 2 + grid_len(query_in.shape)
    if query_out_shape is not None:
        n += grid_len(query_out_shape) + 1
    return n


def loss_targets(e: Encoded):
    """Inputs are positions 0..L-2; target at position t is token t+1, counted when it is an output-grid token."""
    tgt = e.tok[1:]
    mask = e.role[1:] == ROLE_OUT
    return tgt, mask


# ----------------------------------------------------------------------------------------------
# Augmentation: 8 dihedral transforms, colour permutations, pair order
# ----------------------------------------------------------------------------------------------

DIHEDRAL_INV = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral(g: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return g
    if k == 1:
        return np.rot90(g, 1)
    if k == 2:
        return np.rot90(g, 2)
    if k == 3:
        return np.rot90(g, 3)
    if k == 4:
        return g[:, ::-1]
    if k == 5:
        return g[::-1, :]
    if k == 6:
        return g.T
    if k == 7:
        return np.rot90(g, 2).T
    raise ValueError(k)


@dataclass(frozen=True)
class Aug:
    k: int = 0
    perm: tuple = tuple(range(10))

    def fwd(self, g: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(dihedral(np.asarray(self.perm, dtype=g.dtype)[g], self.k))

    def inv(self, g: np.ndarray) -> np.ndarray:
        inv_perm = np.argsort(np.asarray(self.perm)).astype(g.dtype)
        return np.ascontiguousarray(inv_perm[dihedral(g, DIHEDRAL_INV[self.k])])


IDENTITY = Aug()


def random_aug(rng: np.random.Generator, keep_bg: bool = True) -> Aug:
    k = int(rng.integers(8))
    if keep_bg:
        perm = (0,) + tuple(int(x) for x in rng.permutation(np.arange(1, 10)))
    else:
        perm = tuple(int(x) for x in rng.permutation(10))
    return Aug(k, perm)


def collate(encs: list[Encoded], with_targets: bool = True):
    """Right-pad a list of encodings; returns int64 arrays (b, L) and, with targets, (tgt, mask) of (b, L-1)."""
    L = max(len(e) for e in encs)
    b = len(encs)
    tok = np.full((b, L), PAD, dtype=np.int64)
    row = np.full((b, L), MARK_POS, dtype=np.int64)
    col = np.full((b, L), MARK_POS, dtype=np.int64)
    role = np.zeros((b, L), dtype=np.int64)
    tgt = np.zeros((b, L - 1), dtype=np.int64)
    mask = np.zeros((b, L - 1), dtype=bool)
    for i, e in enumerate(encs):
        n = len(e)
        tok[i, :n], row[i, :n], col[i, :n], role[i, :n] = e.tok, e.row, e.col, e.role
        if with_targets:
            t, m = loss_targets(e)
            tgt[i, :n - 1], mask[i, :n - 1] = t, m
    return tok, row, col, role, tgt, mask
