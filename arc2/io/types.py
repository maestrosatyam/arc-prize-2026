"""Frozen ARC-AGI-2 contract types (architecture/CONTRACTS.md v1). """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

Grid = list[list[int]]


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class TaskView:
    """Solver-visible view of a task. INVARIANT: no test outputs are ever present."""

    task_id: str
    train: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]


@dataclass(frozen=True)
class Candidate:
    task_id: str
    test_index: int
    grid: Grid
    source: str
    evidence: dict[str, Any] = field(default_factory=dict)
    cost_s: float = 0.0
    lineage: dict[str, Any] = field(default_factory=dict)


class Solver(Protocol):
    name: str

    def propose(self, task: TaskView, deadline_s: float) -> list[Candidate]:
        """Return candidates before the monotonic deadline (time.monotonic() value)."""
        ...


class Selector(Protocol):
    name: str

    def select(self, candidates: list[Candidate], test_index: int) -> tuple[Grid, Grid]:
        ...


def is_valid_grid(g: Any) -> bool:
    if not isinstance(g, list) or not g or not all(isinstance(r, list) and r for r in g):
        return False
    w = len(g[0])
    if w > 30 or len(g) > 30:
        return False
    return all(len(r) == w and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for v in r) for r in g)
