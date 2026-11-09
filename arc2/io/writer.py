"""Submission writer and validator (A2-01), per architecture/CONTRACTS.md.

``write_submission`` produces ``{task_id: [{"attempt_1": Grid, "attempt_2": Grid}, ...]}`` with every
requested task id present, one entry per test input in order, both attempts always present and
valid, written atomically (tmp file in the same directory + ``os.replace``).

Predictions are never trusted: each attempt is validated with ``is_valid_grid`` and replaced by the
fallback grid when missing or invalid. The writer has no access to labels.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from arc2.io.types import Grid, is_valid_grid

DEFAULT_FALLBACK: Grid = [[0]]

__all__ = [
    "write_submission",
    "build_submission",
    "validate_submission",
    "atomic_write_json",
    "DEFAULT_FALLBACK",
]


def _copy(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def atomic_write_json(path: str | os.PathLike[str], obj: Any, *, indent: int | None = None) -> None:
    """Serialize ``obj`` to ``path`` atomically: write to a temp file in the same directory, fsync, replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, separators=(",", ":") if indent is None else None)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _coerce_attempt(value: Any, fallback: Grid) -> tuple[Grid, bool]:
    """Return (grid, was_fallback). Any non-valid value becomes a copy of the fallback."""
    if is_valid_grid(value):
        return _copy(value), False
    return _copy(fallback), True


def build_submission(
    predictions: Mapping[str, list[Mapping[str, Any]]],
    task_ids: Iterable[str],
    fallback: Grid | None = None,
    n_tests: Mapping[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Grid]]], dict[str, int]]:
    """Build the submission object in memory. Returns ``(submission, stats)``.

    * ``predictions[task_id]`` is a list (one item per test input, in order) of mappings with
      ``attempt_1`` / ``attempt_2`` grids. Extra keys are ignored.
    * ``task_ids`` are the ids that must be present. Ids present in ``predictions`` but not in
      ``task_ids`` are dropped (a submission with unknown ids is not what the scorer expects).
    * ``n_tests`` (optional) gives the number of test inputs per task; lists are padded with fallback
      entries or truncated to that length. Without ``n_tests`` a task with no prediction gets one
      fallback entry (the writer cannot know the test count otherwise) and lists are used as given.
    * ``fallback`` must itself be a valid grid; it defaults to ``[[0]]``.

    ``stats`` counts ``fallback_attempts`` (attempt slots replaced), ``missing_tasks`` and
    ``padded_entries`` for run records.
    """
    fb: Grid = DEFAULT_FALLBACK if fallback is None else fallback
    if not is_valid_grid(fb):
        raise ValueError("fallback is not a valid grid")
    ids = list(task_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("task_ids contains duplicates")

    submission: dict[str, list[dict[str, Grid]]] = {}
    stats = {"fallback_attempts": 0, "missing_tasks": 0, "padded_entries": 0}
    for tid in ids:
        entries = predictions.get(tid)
        if entries is None:
            stats["missing_tasks"] += 1
            entries = []
        if not isinstance(entries, list):
            raise ValueError(f"{tid}: predictions must be a list of per-test mappings")
        want = n_tests.get(tid) if n_tests is not None else None
        if want is not None:
            if want < 1:
                raise ValueError(f"{tid}: n_tests must be >= 1")
            if len(entries) < want:
                stats["padded_entries"] += want - len(entries)
            entries = list(entries[:want]) + [{}] * max(0, want - len(entries))
        elif not entries:
            stats["padded_entries"] += 1
            entries = [{}]
        out_entries: list[dict[str, Grid]] = []
        for entry in entries:
            e = entry if isinstance(entry, Mapping) else {}
            a1, f1 = _coerce_attempt(e.get("attempt_1"), fb)
            a2, f2 = _coerce_attempt(e.get("attempt_2"), fb)
            stats["fallback_attempts"] += int(f1) + int(f2)
            out_entries.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = out_entries
    return submission, stats


def write_submission(
    path: str | os.PathLike[str],
    predictions: Mapping[str, list[Mapping[str, Any]]],
    task_ids: Iterable[str],
    fallback: Grid | None = None,
    n_tests: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Write the submission JSON atomically. See :func:`build_submission` for semantics.

    Returns the stats dict. The file on disk is either the previous content or the complete new
    content, never a partial write.
    """
    submission, stats = build_submission(predictions, task_ids, fallback=fallback, n_tests=n_tests)
    atomic_write_json(path, submission)
    return stats


def validate_submission(submission: Any, challenges: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return a list of human-readable problems; empty list means the submission is well-formed.

    Checks: top-level object; every challenge task id present; no unknown ids; entry count equals
    the number of test inputs; each entry is an object with valid ``attempt_1`` and ``attempt_2``.
    """
    problems: list[str] = []
    if not isinstance(submission, Mapping):
        return ["submission must be a JSON object keyed by task_id"]
    for tid in sorted(challenges):
        if tid not in submission:
            problems.append(f"{tid}: missing task id")
    for tid in sorted(submission):
        if tid not in challenges:
            problems.append(f"{tid}: unknown task id (not in challenges)")
    for tid in sorted(challenges):
        if tid not in submission:
            continue
        entries = submission[tid]
        expected = len(challenges[tid].get("test", []))
        if not isinstance(entries, list):
            problems.append(f"{tid}: entries must be a list, got {type(entries).__name__}")
            continue
        if len(entries) != expected:
            problems.append(f"{tid}: {len(entries)} entries for {expected} test inputs")
        for i, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                problems.append(f"{tid}[{i}]: entry must be an object")
                continue
            for key in ("attempt_1", "attempt_2"):
                if key not in entry:
                    problems.append(f"{tid}[{i}]: missing {key}")
                elif not is_valid_grid(entry[key]):
                    problems.append(f"{tid}[{i}]: {key} is not a valid grid")
    return problems
