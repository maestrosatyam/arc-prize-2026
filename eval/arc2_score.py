"""Official ARC Prize 2026 (ARC-AGI-2) metric, evaluator-only.

Source: compliance/sources.json K3 — "per test output, any of 2 attempts exact => 1; averaged over
all test outputs". The denominator is the number of test OUTPUTS in the solutions file, not the
number of tasks: a two-output task where one output is solved contributes 1 of N outputs.

Documented behaviour for defective submissions (choice: count 0, never raise, so a partially
broken submission still gets a number and the defect is reported):

* task id missing from the submission        -> every output of that task scores 0; id listed in ``missing_tasks``
* fewer entries than test outputs            -> the missing outputs score 0; counted in ``missing_outputs``
* more entries than test outputs             -> extra entries ignored; counted in ``extra_entries``
* attempt missing / not a valid grid         -> that attempt scores 0; counted in ``invalid_attempts``
* wrong shape                                -> 0 (and ``shape_ok`` False for that attempt)

Grid equality is exact: same height, same width, same integer values. ``bool`` cells are invalid,
not equal to ints (``is_valid_grid`` is applied before comparison).

Solver code must never import this module (see arc2/__init__.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # allow `python eval/arc2_score.py` from anywhere
    sys.path.insert(0, str(_ROOT))

from arc2.io.types import Grid, is_valid_grid  # noqa: E402

__all__ = ["grids_equal", "grid_shape", "score_submission", "main"]


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def grids_equal(a: Any, b: Grid) -> bool:
    """Exact equality: ``a`` must be a valid grid with the same shape and values as ``b``."""
    if not is_valid_grid(a):
        return False
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return False
    return all(ra == rb for ra, rb in zip(a, b))


def _attempt(entry: Any, key: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key)
    return None


def score_submission(submission: Any, solutions: Mapping[str, list[Grid]]) -> dict[str, Any]:
    """Score ``submission`` against ``solutions`` (``{task_id: [grid, ...]}`` in test order).

    Returns a dict with:
      score (official, = pass_at_2), pass_at_1, pass_at_2, n_outputs, n_solved, n_tasks,
      n_tasks_all_outputs (tasks with every output solved), task_all_outputs_rate,
      shape_acc (fraction of outputs where attempt_1 or attempt_2 has the right shape),
      invalid_attempts (attempt slots missing or not a valid grid), invalid_rate (per attempt slot),
      missing_tasks, missing_outputs, extra_entries, unknown_tasks, per_output (list of dicts).
    """
    if not isinstance(submission, Mapping):
        submission = {}
    if not solutions:
        raise ValueError("solutions is empty; nothing to score")

    per_output: list[dict[str, Any]] = []
    missing_tasks: list[str] = []
    unknown_tasks = sorted(set(submission) - set(solutions)) if isinstance(submission, Mapping) else []
    missing_outputs = 0
    extra_entries = 0
    invalid_attempts = 0
    n_tasks_all = 0

    for tid in sorted(solutions):
        truths = solutions[tid]
        if not isinstance(truths, list) or not truths:
            raise ValueError(f"{tid}: solutions must be a non-empty list of grids")
        for i, g in enumerate(truths):
            if not is_valid_grid(g):
                raise ValueError(f"{tid}: solution[{i}] is not a valid grid")
        entries = submission.get(tid)
        if entries is None:
            missing_tasks.append(tid)
            entries = []
        elif not isinstance(entries, list):
            entries = []
        if len(entries) > len(truths):
            extra_entries += len(entries) - len(truths)
        task_all = True
        for i, truth in enumerate(truths):
            entry = entries[i] if i < len(entries) else None
            if entry is None:
                missing_outputs += 1
            a1 = _attempt(entry, "attempt_1")
            a2 = _attempt(entry, "attempt_2")
            v1, v2 = is_valid_grid(a1), is_valid_grid(a2)
            invalid_attempts += int(not v1) + int(not v2)
            c1 = grids_equal(a1, truth)
            c2 = grids_equal(a2, truth)
            shape = grid_shape(truth)
            s1 = v1 and grid_shape(a1) == shape
            s2 = v2 and grid_shape(a2) == shape
            correct = c1 or c2
            task_all = task_all and correct
            per_output.append(
                {
                    "task_id": tid,
                    "test_index": i,
                    "correct": correct,
                    "correct_1": c1,
                    "correct_2": c2,
                    "valid_1": v1,
                    "valid_2": v2,
                    "shape_ok": s1 or s2,
                    "truth_shape": list(shape),
                }
            )
        if task_all:
            n_tasks_all += 1

    n_outputs = len(per_output)
    n_solved = sum(1 for o in per_output if o["correct"])
    n_pass1 = sum(1 for o in per_output if o["correct_1"])
    n_shape = sum(1 for o in per_output if o["shape_ok"])
    n_tasks = len(solutions)
    return {
        "score": n_solved / n_outputs,
        "pass_at_2": n_solved / n_outputs,
        "pass_at_1": n_pass1 / n_outputs,
        "n_outputs": n_outputs,
        "n_solved": n_solved,
        "n_tasks": n_tasks,
        "n_tasks_all_outputs": n_tasks_all,
        "task_all_outputs_rate": n_tasks_all / n_tasks,
        "shape_acc": n_shape / n_outputs,
        "invalid_attempts": invalid_attempts,
        "invalid_rate": invalid_attempts / (2 * n_outputs),
        "missing_tasks": missing_tasks,
        "missing_outputs": missing_outputs,
        "extra_entries": extra_entries,
        "unknown_tasks": unknown_tasks,
        "per_output": per_output,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score an ARC-AGI-2 submission.json against a solutions file.")
    ap.add_argument("submission")
    ap.add_argument("solutions")
    ap.add_argument("--per-output", action="store_true", help="print per-output rows")
    args = ap.parse_args(argv)
    with open(args.submission, "r", encoding="utf-8") as fh:
        sub = json.load(fh)
    with open(args.solutions, "r", encoding="utf-8") as fh:
        sol = json.load(fh)
    res = score_submission(sub, sol)
    summary = {k: v for k, v in res.items() if k != "per_output"}
    print(json.dumps(summary, indent=1))
    if args.per_output:
        for row in res["per_output"]:
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
