"""RankSelector: the public baseline's selection rule, extended with a documented insertion rule for symbolic
candidates (this is the B2 selector; B3 = DiverseSelector differs only in the choice of attempt_2).

Baseline rule (arc_decoder.py ``score_kgmon``): distinct grids ranked by ``n_hits - mean(score_aug)`` descending;
``attempt_1``/``attempt_2`` = first two. In Candidate form that score is ``evidence['model_score']`` as produced by
``arc2.decoding.pool.neural_to_candidate``.

Insertion rule for symbolic candidates (tiers, highest first):
  tier 0: ``demo_consistent is True`` and ``loo_consistent is True``  (program reproduces every demo and survives
          leave-one-demo-out re-fit)
  tier 1: ``demo_consistent is True`` (loo None or False)
  tier 2: everything else, ordered by ``model_score`` descending (neural ranking); candidates without a model
          score go last.
Within a tier the input order is kept (symbolic candidates arrive in program-rank order from
``build_candidates``; neural candidates arrive ranked by the pool reader). Ties are therefore deterministic given
a deterministic pool.

Fallbacks when fewer than two distinct valid grids exist: the configured ``fallbacks`` grids in order (default
``[[0]]``, the baseline's placeholder), skipping any equal to attempt_1; as a last resort attempt_1 is duplicated,
which is schema-safe and scores identically to a single attempt.
"""
from __future__ import annotations

from collections.abc import Sequence

from arc2.io.types import Candidate, Grid, is_valid_grid

__all__ = ["RankSelector", "rank_candidates", "tier", "two_valid_grids"]

DEFAULT_FALLBACKS: tuple[Grid, ...] = ([[0]],)


def tier(c: Candidate) -> int:
    ev = c.evidence or {}
    if ev.get("demo_consistent") is True:
        return 0 if ev.get("loo_consistent") is True else 1
    return 2


def _model_score(c: Candidate) -> float:
    s = (c.evidence or {}).get("model_score")
    try:
        return float(s) if s is not None else float("-inf")
    except (TypeError, ValueError):
        return float("-inf")


def rank_candidates(cands: Sequence[Candidate], test_index: int | None = None) -> list[Candidate]:
    """Valid candidates of ``test_index`` (all if None) in tier order, model_score descending within tier 2,
    input order otherwise. Exact duplicate grids keep the first occurrence."""
    pool = [c for c in cands if (test_index is None or c.test_index == test_index) and is_valid_grid(c.grid)]
    indexed = list(enumerate(pool))
    indexed.sort(key=lambda ic: (tier(ic[1]), -_model_score(ic[1]) if tier(ic[1]) == 2 else 0.0, ic[0]))
    out: list[Candidate] = []
    seen: set[tuple] = set()
    for _, c in indexed:
        gk = tuple(tuple(r) for r in c.grid)
        if gk in seen:
            continue
        seen.add(gk)
        out.append(c)
    return out


def two_valid_grids(chosen: list[Grid], fallbacks: Sequence[Grid]) -> tuple[Grid, Grid]:
    """Pad ``chosen`` (0..2 distinct valid grids) to exactly two valid grids."""
    out = [[list(r) for r in g] for g in chosen[:2] if is_valid_grid(g)]
    for fb in fallbacks:
        if len(out) >= 2:
            break
        if is_valid_grid(fb) and not any(fb == g for g in out):
            out.append([list(r) for r in fb])
    if not out:
        out.append([[0]])
    while len(out) < 2:
        out.append([list(r) for r in out[0]])
    return out[0], out[1]


class RankSelector:
    name = "rank_kgmon_symbolic_first"

    def __init__(self, fallbacks: Sequence[Grid] = DEFAULT_FALLBACKS) -> None:
        self.fallbacks = tuple([list(r) for r in g] for g in fallbacks)

    def ranked(self, candidates: Sequence[Candidate], test_index: int) -> list[Candidate]:
        return rank_candidates(candidates, test_index)

    def select(self, candidates: Sequence[Candidate], test_index: int) -> tuple[Grid, Grid]:
        ranked = self.ranked(candidates, test_index)
        return two_valid_grids([c.grid for c in ranked[:2]], self.fallbacks)
