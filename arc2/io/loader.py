"""ARC-AGI-2 task loading and format conversion (A2-01).

Two on-disk layouts are supported:

* Kaggle layout (compliance/sources.json K5): one ``arc-agi_<split>-challenges.json`` file
  ``{task_id: {"train": [{"input", "output"}], "test": [{"input"}]}}`` and a separate
  ``arc-agi_<split>-solutions.json`` file ``{task_id: [output_grid, ...]}`` in test order.
* Repository layout (third_party/ARC-AGI-2/data/<split>/<task_id>.json): one file per task,
  ``{"train": [...], "test": [{"input", "output"}]}`` with test outputs inline.

Solver-facing entry points (``load_kaggle_challenges``, ``load_task_dir``, ``task_from_dict``)
return :class:`TaskView` objects and strip every ``output`` key found under ``test`` before the
view is constructed, so test labels never reach solver code. Every grid is validated
(rectangular, ints 0..9, 1x1..30x30) and malformed input raises :class:`TaskFormatError`.

The converters ``repo_to_kaggle_format`` / ``kaggle_to_repo_format`` are evaluator-side helpers
that carry labels; they are used by ``scripts/arc2_build_eval_data.py`` to produce the files
under ``eval/data/`` and must not be imported by solver code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from arc2.io.types import Grid, Pair, TaskView

MAX_SIDE = 30
MIN_COLOR = 0
MAX_COLOR = 9

__all__ = [
    "TaskFormatError",
    "validate_grid",
    "copy_grid",
    "task_from_dict",
    "load_kaggle_challenges",
    "load_task_dir",
    "iter_task_files",
    "load_raw_task_dir",
    "load_kaggle_solutions",
    "repo_to_kaggle_format",
    "kaggle_to_repo_format",
    "strip_test_outputs",
]


class TaskFormatError(ValueError):
    """Raised for any structurally invalid task or grid."""


def validate_grid(grid: Any, where: str = "grid") -> None:
    """Raise TaskFormatError unless ``grid`` is a rectangular list of int rows, values 0..9, 1x1..30x30.

    ``bool`` values are rejected (``True == 1`` in Python would otherwise pass silently).
    """
    if not isinstance(grid, list):
        raise TaskFormatError(f"{where}: grid must be a list, got {type(grid).__name__}")
    if not grid:
        raise TaskFormatError(f"{where}: grid has no rows")
    h = len(grid)
    if h > MAX_SIDE:
        raise TaskFormatError(f"{where}: height {h} exceeds {MAX_SIDE}")
    first = grid[0]
    if not isinstance(first, list):
        raise TaskFormatError(f"{where}: row 0 must be a list, got {type(first).__name__}")
    w = len(first)
    if w == 0:
        raise TaskFormatError(f"{where}: row 0 is empty")
    if w > MAX_SIDE:
        raise TaskFormatError(f"{where}: width {w} exceeds {MAX_SIDE}")
    for r, row in enumerate(grid):
        if not isinstance(row, list):
            raise TaskFormatError(f"{where}: row {r} must be a list, got {type(row).__name__}")
        if len(row) != w:
            raise TaskFormatError(f"{where}: row {r} has width {len(row)}, expected {w} (not rectangular)")
        for c, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, int):
                raise TaskFormatError(f"{where}: cell ({r},{c}) must be int, got {type(v).__name__}")
            if v < MIN_COLOR or v > MAX_COLOR:
                raise TaskFormatError(f"{where}: cell ({r},{c}) value {v} outside {MIN_COLOR}..{MAX_COLOR}")


def copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _validated_copy(grid: Any, where: str) -> Grid:
    validate_grid(grid, where)
    return copy_grid(grid)


def strip_test_outputs(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied raw task dict whose test pairs contain only ``input``.

    Used to turn repository-format (labelled) tasks into Kaggle challenge entries. Does not validate.
    """
    out: dict[str, Any] = {
        "train": [
            {"input": copy_grid(p["input"]), "output": copy_grid(p["output"])} for p in raw.get("train", [])
        ],
        "test": [{"input": copy_grid(p["input"])} for p in raw.get("test", [])],
    }
    return out


def task_from_dict(task_id: str, raw: Mapping[str, Any]) -> TaskView:
    """Build a validated, label-free TaskView from a raw task mapping.

    Accepts both Kaggle challenge entries and repository entries. Any ``output`` (or other extra key)
    inside ``test`` pairs is discarded before the TaskView exists; only ``test[i]["input"]`` is read.
    """
    if not isinstance(task_id, str) or not task_id:
        raise TaskFormatError(f"task_id must be a non-empty string, got {task_id!r}")
    if not isinstance(raw, Mapping):
        raise TaskFormatError(f"{task_id}: task must be a mapping, got {type(raw).__name__}")
    train_raw = raw.get("train")
    test_raw = raw.get("test")
    if not isinstance(train_raw, list) or not train_raw:
        raise TaskFormatError(f"{task_id}: 'train' must be a non-empty list")
    if not isinstance(test_raw, list) or not test_raw:
        raise TaskFormatError(f"{task_id}: 'test' must be a non-empty list")

    train: list[Pair] = []
    for i, pair in enumerate(train_raw):
        if not isinstance(pair, Mapping) or "input" not in pair or "output" not in pair:
            raise TaskFormatError(f"{task_id}: train[{i}] must have 'input' and 'output'")
        train.append(
            Pair(
                input=_validated_copy(pair["input"], f"{task_id}.train[{i}].input"),
                output=_validated_copy(pair["output"], f"{task_id}.train[{i}].output"),
            )
        )

    test_inputs: list[Grid] = []
    for i, pair in enumerate(test_raw):
        if not isinstance(pair, Mapping) or "input" not in pair:
            raise TaskFormatError(f"{task_id}: test[{i}] must have 'input'")
        # Deliberately read only 'input'; 'output' and any other key are never touched.
        test_inputs.append(_validated_copy(pair["input"], f"{task_id}.test[{i}].input"))

    return TaskView(task_id=task_id, train=tuple(train), test_inputs=tuple(test_inputs))


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise TaskFormatError(f"{path}: invalid JSON: {exc}") from exc


def load_kaggle_challenges(path: str | os.PathLike[str]) -> dict[str, TaskView]:
    """Load a Kaggle ``arc-agi_<split>-challenges.json`` file into ``{task_id: TaskView}``.

    Task ids are returned in sorted order. Test outputs, if present, are stripped.
    """
    p = Path(path)
    data = _read_json(p)
    if not isinstance(data, dict):
        raise TaskFormatError(f"{p}: challenges file must be a JSON object keyed by task_id")
    return {tid: task_from_dict(tid, data[tid]) for tid in sorted(data)}


def iter_task_files(directory: str | os.PathLike[str]) -> Iterator[tuple[str, Path]]:
    """Yield ``(task_id, path)`` for every ``*.json`` in ``directory`` in sorted order."""
    d = Path(directory)
    if not d.is_dir():
        raise TaskFormatError(f"{d}: not a directory")
    for p in sorted(d.glob("*.json")):
        yield p.stem, p


def load_task_dir(directory: str | os.PathLike[str]) -> dict[str, TaskView]:
    """Load a directory of per-task JSON files (ARC-AGI-2 repo layout) into ``{task_id: TaskView}``.

    Test outputs present in the files are stripped before the TaskView is constructed.
    """
    views: dict[str, TaskView] = {}
    for tid, p in iter_task_files(directory):
        views[tid] = task_from_dict(tid, _read_json(p))
    if not views:
        raise TaskFormatError(f"{directory}: no *.json task files found")
    return views


# --------------------------------------------------------------------------------------
# Evaluator-side helpers (carry labels). Not for solver code.
# --------------------------------------------------------------------------------------


def load_raw_task_dir(directory: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Load repo-layout task files as raw dicts, validated, labels included. Evaluator-side only."""
    raw: dict[str, dict[str, Any]] = {}
    for tid, p in iter_task_files(directory):
        task = _read_json(p)
        _validate_raw_labelled_task(tid, task)
        raw[tid] = task
    if not raw:
        raise TaskFormatError(f"{directory}: no *.json task files found")
    return raw


def _validate_raw_labelled_task(task_id: str, task: Any) -> None:
    # Validates structure and all grids including test outputs.
    task_from_dict(task_id, task)  # validates train + test inputs
    for i, pair in enumerate(task["test"]):
        if "output" not in pair:
            raise TaskFormatError(f"{task_id}: test[{i}] has no 'output' (labelled task required)")
        validate_grid(pair["output"], f"{task_id}.test[{i}].output")


def load_kaggle_solutions(path: str | os.PathLike[str]) -> dict[str, list[Grid]]:
    """Load ``arc-agi_<split>-solutions.json`` (``{task_id: [grid, ...]}``). Evaluator-side only."""
    p = Path(path)
    data = _read_json(p)
    if not isinstance(data, dict):
        raise TaskFormatError(f"{p}: solutions file must be a JSON object keyed by task_id")
    out: dict[str, list[Grid]] = {}
    for tid in sorted(data):
        grids = data[tid]
        if not isinstance(grids, list) or not grids:
            raise TaskFormatError(f"{p}: {tid}: solutions must be a non-empty list of grids")
        for i, g in enumerate(grids):
            validate_grid(g, f"{tid}.solution[{i}]")
        out[tid] = [copy_grid(g) for g in grids]
    return out


def repo_to_kaggle_format(
    tasks: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Grid]]]:
    """Convert raw repo-format tasks into ``(challenges, solutions)`` with Kaggle's schema.

    ``challenges[tid] = {"train": [{"input","output"}...], "test": [{"input"}...]}``
    ``solutions[tid] = [test_output_grid, ...]`` in test order.
    Keys are emitted in sorted task_id order. Every grid is validated.
    """
    challenges: dict[str, dict[str, Any]] = {}
    solutions: dict[str, list[Grid]] = {}
    for tid in sorted(tasks):
        task = tasks[tid]
        _validate_raw_labelled_task(tid, task)
        challenges[tid] = strip_test_outputs(task)
        solutions[tid] = [copy_grid(p["output"]) for p in task["test"]]
    return challenges, solutions


def kaggle_to_repo_format(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Grid]],
) -> dict[str, dict[str, Any]]:
    """Inverse of :func:`repo_to_kaggle_format`: merge solutions back into per-task dicts.

    Raises TaskFormatError if a task has no solutions entry or the solution count does not match
    the number of test inputs.
    """
    out: dict[str, dict[str, Any]] = {}
    for tid in sorted(challenges):
        ch = challenges[tid]
        task_from_dict(tid, ch)  # validate structure and grids
        if tid not in solutions:
            raise TaskFormatError(f"{tid}: missing from solutions")
        sols = solutions[tid]
        if len(sols) != len(ch["test"]):
            raise TaskFormatError(f"{tid}: {len(sols)} solutions for {len(ch['test'])} test inputs")
        test = []
        for i, (pair, sol) in enumerate(zip(ch["test"], sols)):
            validate_grid(sol, f"{tid}.solution[{i}]")
            test.append({"input": copy_grid(pair["input"]), "output": copy_grid(sol)})
        out[tid] = {
            "train": [{"input": copy_grid(p["input"]), "output": copy_grid(p["output"])} for p in ch["train"]],
            "test": test,
        }
    return out
