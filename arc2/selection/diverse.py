"""DiverseSelector (B3): attempt_1 = top-ranked candidate; attempt_2 = the highest-ranked candidate that is
*structurally* different from attempt_1 (arc2/selection/structure.py), falling back to the highest-ranked merely
pixel-different candidate, then the configured fallback grids, then a duplicate (schema safety only).

Ranking score (higher first):
    symbolic (demo_consistent True):  w_symbolic + (w_loo if loo_consistent is True else 0) + rank_bonus
    neural / other:                   w_neural * model_score                                  (model_score = kgmon)
    candidates with neither:          -inf (after everything else, input order)
``rank_bonus`` = -1e-6 * input position, so within equal scores the input order (program rank / pool rank) decides.
With the defaults (w_symbolic=1e3, w_loo=1, w_neural=1) the ordering equals RankSelector's tiers, because kgmon
scores are bounded by the number of decoded beams (tens) and symbolic candidates always come first; calibration
may lower w_symbolic so that strongly supported neural grids outrank unverified-by-LOO programs.

Parameters are plain floats so ``fit_weights`` can grid-search them on a labelled development set and store them
in a versioned JSON (``Calibration``). ``select`` takes candidates and a test index only; it has no label input.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from arc2.io.types import Candidate, Grid, is_valid_grid
from arc2.selection.rank import DEFAULT_FALLBACKS, two_valid_grids
from arc2.selection.structure import structurally_different

__all__ = ["DEFAULT_PARAMS", "DiverseSelector"]

DEFAULT_PARAMS: dict[str, float] = {"tau": 0.1, "w_symbolic": 1000.0, "w_loo": 1.0, "w_neural": 1.0}


class DiverseSelector:
    name = "diverse_structural_second"

    def __init__(
        self,
        tau: float = DEFAULT_PARAMS["tau"],
        w_symbolic: float = DEFAULT_PARAMS["w_symbolic"],
        w_loo: float = DEFAULT_PARAMS["w_loo"],
        w_neural: float = DEFAULT_PARAMS["w_neural"],
        fallbacks: Sequence[Grid] = DEFAULT_FALLBACKS,
        calibration_version: str | None = None,
    ) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        self.tau = float(tau)
        self.w_symbolic = float(w_symbolic)
        self.w_loo = float(w_loo)
        self.w_neural = float(w_neural)
        self.fallbacks = tuple([list(r) for r in g] for g in fallbacks)
        self.calibration_version = calibration_version

    @property
    def params(self) -> dict[str, float]:
        return {"tau": self.tau, "w_symbolic": self.w_symbolic, "w_loo": self.w_loo, "w_neural": self.w_neural}

    @classmethod
    def from_params(cls, params: Mapping[str, Any], **kw: Any) -> DiverseSelector:
        p = {k: float(params[k]) for k in DEFAULT_PARAMS if k in params}
        return cls(**{**DEFAULT_PARAMS, **p}, **kw)

    def score(self, c: Candidate, position: int) -> float:
        ev = c.evidence or {}
        bonus = -1e-6 * position
        if ev.get("demo_consistent") is True:
            return self.w_symbolic + (self.w_loo if ev.get("loo_consistent") is True else 0.0) + bonus
        ms = ev.get("model_score")
        if ms is not None:
            try:
                return self.w_neural * float(ms) + bonus
            except (TypeError, ValueError):
                pass
        return -1e18 + bonus

    def ranked(self, candidates: Sequence[Candidate], test_index: int) -> list[Candidate]:
        pool = [c for c in candidates if c.test_index == test_index and is_valid_grid(c.grid)]
        scored = sorted(((self.score(c, i), i, c) for i, c in enumerate(pool)), key=lambda t: (-t[0], t[1]))
        out: list[Candidate] = []
        seen: set[tuple] = set()
        for _, _, c in scored:
            gk = tuple(tuple(r) for r in c.grid)
            if gk in seen:
                continue
            seen.add(gk)
            out.append(c)
        return out

    def choose(self, ranked: Sequence[Candidate]) -> tuple[list[Grid], str]:
        """Return (chosen grids, reason) where reason is 'structural' | 'pixel' | 'single' | 'empty'."""
        if not ranked:
            return [], "empty"
        a1 = ranked[0].grid
        for c in ranked[1:]:
            if structurally_different(a1, c.grid, self.tau):
                return [a1, c.grid], "structural"
        if len(ranked) > 1:
            return [a1, ranked[1].grid], "pixel"
        return [a1], "single"

    def select(self, candidates: Sequence[Candidate], test_index: int) -> tuple[Grid, Grid]:
        chosen, _ = self.choose(self.ranked(candidates, test_index))
        return two_valid_grids(chosen, self.fallbacks)
