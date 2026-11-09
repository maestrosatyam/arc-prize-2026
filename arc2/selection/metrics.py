"""Selection metrics (protocol: oracle best-two coverage and marginal gain of attempt_2 are reported for every run).

Per test output (``output_metrics``):
  correct_1   attempt_1 equals the label
  correct_2   attempt_2 equals the label
  pass_at_1   = correct_1
  pass_at_2   = correct_1 or correct_2  (the official metric)
  oracle      the label is anywhere in the candidate pool (an oracle choosing the best two would score it)
Aggregate (``aggregate``), all with explicit denominators:
  pass_at_1, pass_at_2, oracle_best_two (= oracle coverage), marginal_gain_attempt_2 = pass_at_2 - pass_at_1,
  selection_gap = oracle_best_two - pass_at_2 (what better selection could still recover), n_outputs.
Labels enter only here; selectors never see them.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from arc2.io.types import Grid, is_valid_grid

__all__ = ["aggregate", "output_metrics", "selection_gap"]


def _eq(a: Any, b: Grid) -> bool:
    return is_valid_grid(a) and len(a) == len(b) and len(a[0]) == len(b[0]) and all(ra == rb for ra, rb in zip(a, b))


def output_metrics(attempts: Sequence[Grid], pool: Iterable[Grid], label: Grid) -> dict[str, Any]:
    a1 = attempts[0] if len(attempts) > 0 else None
    a2 = attempts[1] if len(attempts) > 1 else None
    c1 = _eq(a1, label)
    c2 = _eq(a2, label)
    pool_list = list(pool)
    return {
        "correct_1": c1,
        "correct_2": c2,
        "pass_at_1": c1,
        "pass_at_2": c1 or c2,
        "oracle": any(_eq(g, label) for g in pool_list),
        "pool_size": len(pool_list),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_outputs": 0, "pass_at_1": 0.0, "pass_at_2": 0.0, "oracle_best_two": 0.0, "marginal_gain_attempt_2": 0.0, "selection_gap": 0.0,
                "n_pass_at_1": 0, "n_pass_at_2": 0, "n_oracle": 0}
    n1 = sum(1 for r in rows if r["pass_at_1"])
    n2 = sum(1 for r in rows if r["pass_at_2"])
    no = sum(1 for r in rows if r["oracle"])
    return {
        "n_outputs": n,
        "n_pass_at_1": n1,
        "n_pass_at_2": n2,
        "n_oracle": no,
        "pass_at_1": n1 / n,
        "pass_at_2": n2 / n,
        "oracle_best_two": no / n,
        "marginal_gain_attempt_2": (n2 - n1) / n,
        "selection_gap": (no - n2) / n,
    }


def selection_gap(oracle_best_two: float, pass_at_2: float) -> float:
    return oracle_best_two - pass_at_2
