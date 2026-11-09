"""Rank the candidate outputs of a task into attempt_1 / attempt_2, and merge with an external pool.

A Sample (see solvers.py) carries for each test input a written grid and, if it had a program, the program's
output on that test input, plus the program's verification result on the demonstrations.

Policies
  grid_only       votes over written grids (the direct baseline's selection)
  poetiq          Poetiq's ranking: passers (by votes) before failures (by votes, then demo score)
  vr              verified programs first (by votes), then the model's written grids before outputs that only an
                  unverified program produced, then self-consistency, validator, demo score, finish state, stage order
Merge with an external pool (e.g. the NVARC phase's top-2):
  R0              external attempt_1 kept; the harness's best differing answer takes attempt_2
  verified_first  a verified harness answer takes attempt_1, the external attempt_1 moves to attempt_2; else R0
  harness_first   harness top-1, then the external attempt_1
  verified_then_margin  verified harness answer first; otherwise the external attempt_1 first only when its
                  confidence margin (external[tid] may be {"attempts": [...], "margin": x}) is >= margin_threshold,
                  else the harness answer first (the 4B's margin predicts its correctness, AUC 0.94 [brief])
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .grids import Grid, hardcoded_risk, is_grid
from .validator import check

STAGE_ORDER = {"s1": 0, "rescue": 0, "repair": 1, "rescue_repair": 1, "s2": 2, "rescue_s2": 2}


@dataclass
class Group:
    grid: Grid
    verified_votes: int = 0        # samples whose program passed every demonstration and outputs this grid
    verified_clean: int = 0        # ...and whose program carries no long literal lists (not a lookup table)
    support: int = 0               # samples whose written grid or program output is this grid
    written: int = 0               # samples whose written grid (not program output) is this grid
    double: int = 0                # samples where both the written grid and the program output are this grid
    own_finish: int = 0            # supporting samples that finished on their own (not cut, not rescued)
    best_score: float = 0.0        # best demonstration cell-accuracy of a supporting program
    best_npass: int = 0
    hard_ok: bool = True
    earliest_stage: int = 9
    sources: List[str] = field(default_factory=list)

    def key_vr(self):
        # v7-vr2: an unverified program's output no longer outranks a written grid (it won on the demo-score
        # tie-breakers, which only program candidates carry); verified programs keep first place
        return (self.verified_clean > 0, self.verified_votes > 0, self.verified_votes,
                self.verified_votes > 0 or self.written > 0, self.hard_ok, self.support,
                self.double, self.best_npass, self.best_score, self.own_finish, -self.earliest_stage)

    def key_grid_only(self):
        return (self.support, self.own_finish, self.hard_ok, -self.earliest_stage)

    def key_poetiq(self):
        return (self.verified_votes > 0, self.support, self.best_score)


def _k(g) -> str:
    return json.dumps(g)


def group_candidates(task: dict, samples: List[dict], j: int, use_validator: bool = True) -> Dict[str, Group]:
    """Groups of identical candidate grids for test input j."""
    groups: Dict[str, Group] = {}
    test_in = task["test"][j]["input"]
    for s in samples:
        g_written = (s.get("grids") or [None] * (j + 1))[j] if j < len(s.get("grids") or []) else None
        v = s.get("verify") or {}
        prog_outs = v.get("program_outputs") or []
        g_prog = prog_outs[j] if j < len(prog_outs) else None
        if g_written is not None and not is_grid(g_written):
            g_written = None
        if g_prog is not None and not is_grid(g_prog):
            g_prog = None
        seen = set()
        for g, kind in ((g_written, "grid"), (g_prog, "program")):
            if g is None:
                continue
            k = _k(g)
            grp = groups.get(k)
            if grp is None:
                grp = Group(grid=g)
                if use_validator:
                    grp.hard_ok = check(task["train"], test_in, g)["hard_ok"]
                groups[k] = grp
            if k not in seen:
                grp.support += 1
                seen.add(k)
                if s.get("finish") == "own":
                    grp.own_finish += 1
                grp.earliest_stage = min(grp.earliest_stage, STAGE_ORDER.get(s.get("stage", ""), 5))
                grp.sources.append(f"{s.get('sample_id')}:{kind}")
            if kind == "grid":
                grp.written += 1
            if kind == "program" and v.get("ran"):
                grp.best_score = max(grp.best_score, float(v.get("score", 0.0)))
                grp.best_npass = max(grp.best_npass, int(v.get("n_pass", 0)))
                if v.get("passed"):
                    grp.verified_votes += 1
                    if not hardcoded_risk(s.get("program") or ""):
                        grp.verified_clean += 1
        if g_written is not None and g_prog is not None and _k(g_written) == _k(g_prog):
            groups[_k(g_written)].double += 1
    return groups


def rank(task: dict, samples: List[dict], j: int, policy: str = "vr", use_validator: bool = True) -> List[Group]:
    groups = group_candidates(task, samples, j, use_validator)
    keyf = {"vr": Group.key_vr, "grid_only": Group.key_grid_only, "poetiq": Group.key_poetiq}[policy]
    return sorted(groups.values(), key=keyf, reverse=True)


def fallback_grid(task: dict, j: int) -> Grid:
    """A valid grid when nothing else exists: the test input itself (identity is right on ~0% but valid)."""
    return [row[:] for row in task["test"][j]["input"]]


def two_attempts(task: dict, samples: List[dict], policy: str = "vr", external=None,
                 merge: str = "R0", use_validator: bool = True, margin_threshold: float = 0.5) -> List[dict]:
    """attempt_1 / attempt_2 for every test input of the task.

    external: [[attempt_1, attempt_2] per test input] or {"attempts": that, "margin": float | [float per test]}."""
    out = []
    ext_margin = None
    if isinstance(external, dict):
        ext_margin = external.get("margin")
        external = external.get("attempts")
    for j in range(len(task["test"])):
        ranked = rank(task, samples, j, policy, use_validator)
        ours = [g.grid for g in ranked]
        top_verified = bool(ranked) and ranked[0].verified_clean > 0
        ext = (external[j] if external and j < len(external) else []) or []
        ext = [g for g in ext if is_grid(g)]
        picks: List[Grid] = []

        def add(g):
            if g is not None and is_grid(g) and all(g != p for p in picks) and len(picks) < 2:
                picks.append(g)

        if not ext:
            for g in ours:
                add(g)
        elif merge == "R0":
            add(ext[0])
            for g in ours:
                add(g)
            for g in ext[1:]:
                add(g)
        elif merge == "verified_first":
            if top_verified:
                add(ours[0])
                add(ext[0])
                for g in ours[1:]:
                    add(g)
                for g in ext[1:]:
                    add(g)
            else:
                add(ext[0])
                for g in ours:
                    add(g)
                for g in ext[1:]:
                    add(g)
        elif merge == "harness_first":
            for g in ours[:1]:
                add(g)
            for g in ext:
                add(g)
            for g in ours[1:]:
                add(g)
        elif merge == "verified_then_margin":
            m = ext_margin[j] if isinstance(ext_margin, list) else ext_margin
            ext_confident = m is not None and float(m) >= margin_threshold
            if top_verified or (ours and not ext_confident):
                for g in ours[:1]:
                    add(g)
                add(ext[0])
                for g in ours[1:]:
                    add(g)
                for g in ext[1:]:
                    add(g)
            else:
                add(ext[0])
                for g in ours:
                    add(g)
                for g in ext[1:]:
                    add(g)
        else:
            raise ValueError(merge)
        while len(picks) < 2:
            fb = fallback_grid(task, j)
            if picks and picks[0] == fb:
                fb = [[0]]
            picks.append(fb)
        out.append({"attempt_1": picks[0], "attempt_2": picks[1]})
    return out
