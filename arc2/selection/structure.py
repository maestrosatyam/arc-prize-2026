"""Structural difference between two grids (the B3 second-answer rule).

Two grids are *structurally different* when at least one holds:
  1. different shape;
  2. different colour histogram (multiset of cell values);
  3. same shape and histogram but the fraction of disagreeing cells is >= tau.
Otherwise they are at most a small local edit of each other (``tau`` controls "small"), and spending attempt_2 on
such a near-duplicate rarely helps: if attempt_1 is wrong in a way the model is unsure about, the second guess
should cover a different hypothesis, not a variant of the same one.
"""
from __future__ import annotations

from collections import Counter

from arc2.io.types import Grid

__all__ = ["cell_disagreement", "colour_histogram", "structurally_different"]


def colour_histogram(grid: Grid) -> Counter:
    return Counter(v for row in grid for v in row)


def cell_disagreement(a: Grid, b: Grid) -> float:
    """Fraction of cells that differ; 1.0 when the shapes differ."""
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return 1.0
    n = len(a) * len(a[0])
    diff = sum(1 for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x != y)
    return diff / n


def structurally_different(a: Grid, b: Grid, tau: float) -> bool:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return True
    if colour_histogram(a) != colour_histogram(b):
        return True
    return cell_disagreement(a, b) >= tau
