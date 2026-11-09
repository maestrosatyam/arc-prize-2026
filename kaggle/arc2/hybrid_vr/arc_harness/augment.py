"""Task augmentations for independent second samples: a dihedral transform plus a colour permutation (0 fixed).

A sample on an augmented view is solved in the transformed frame; its grids and program outputs are mapped back
with the inverse, so every candidate lives in the original frame and votes across views are over the same grids.
The program is verified on the augmented demonstrations (the frame it was written for). The same trick is what
ARChitects/NVARC use for augmentation voting; here it buys diversity for the few second samples the budget allows.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from .grids import Grid


def _rot90(g: Grid) -> Grid:          # counter-clockwise
    return [list(r) for r in zip(*g)][::-1]


def _rot(g: Grid, k: int) -> Grid:
    for _ in range(k % 4):
        g = _rot90(g)
    return g


def _flip(g: Grid) -> Grid:           # left-right
    return [r[::-1] for r in g]


@dataclass(frozen=True)
class Aug:
    rot: int = 0                      # quarter turns, counter-clockwise, applied after the flip
    flip: bool = False
    perm: tuple = tuple(range(10))    # colour map, perm[0] == 0

    @property
    def inv_perm(self) -> tuple:
        inv = [0] * 10
        for a, b in enumerate(self.perm):
            inv[b] = a
        return tuple(inv)

    def apply(self, g: Grid) -> Grid:
        g = [[self.perm[c] for c in r] for r in g]
        if self.flip:
            g = _flip(g)
        return _rot(g, self.rot)

    def invert(self, g: Optional[Grid]) -> Optional[Grid]:
        if g is None:
            return None
        g = _rot(g, -self.rot)
        if self.flip:
            g = _flip(g)
        inv = self.inv_perm
        return [[inv[c] for c in r] for r in g]

    def apply_task(self, task: dict) -> dict:
        return {"id": task["id"], "aug": self.key(),
                "train": [{"input": self.apply(p["input"]), "output": self.apply(p["output"])} for p in task["train"]],
                "test": [{"input": self.apply(t["input"])} for t in task["test"]]}

    def key(self) -> str:
        return f"r{self.rot}{'f' if self.flip else ''}p{''.join(map(str, self.perm))}"

    @property
    def is_identity(self) -> bool:
        return self.rot == 0 and not self.flip and self.perm == tuple(range(10))


IDENTITY = Aug()


def random_aug(seed: int, geometry: bool = True, colours: bool = True) -> Aug:
    rng = random.Random(seed)
    rot, flip = (rng.randrange(4), rng.random() < 0.5) if geometry else (0, False)
    perm = list(range(10))
    if colours:
        tail = perm[1:]
        rng.shuffle(tail)
        perm = [0] + tail
    a = Aug(rot, flip, tuple(perm))
    return a if not a.is_identity else Aug(1, False, tuple(perm))


def invert_sample(aug: Aug, sample: dict) -> dict:
    """Map a sample's grids and program outputs back to the original frame (in place)."""
    if aug.is_identity:
        return sample
    sample["grids"] = [aug.invert(g) for g in sample["grids"]]
    v = sample.get("verify") or {}
    if v.get("program_outputs"):
        v["program_outputs"] = [aug.invert(g) for g in v["program_outputs"]]
    sample["aug"] = aug.key()
    return sample
