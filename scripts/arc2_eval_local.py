"""Local ARC-AGI-2 evaluation harness (S03, evaluator-side).

    python scripts/arc2_eval_local.py --solver arc2.solvers.identity:IdentitySolver --split dev \
        --per-task-seconds 5 --out experiments/runs/<run_id> --seed 0

Flow: load the split's tasks label-free through arc2.io.loader -> for each task call
``solver.propose(task, deadline)`` with a monotonic deadline -> select two attempts per test input
with the default selector -> write submission.json through arc2.io.writer -> score against the
solutions file under eval/data with eval.arc2_score -> write run_record.json
(experiments/_control/run_schema.json) and per_task.json.

Labels are read only in the scoring step, after the submission file is complete on disk.
``--split holdout`` is refused unless ``--confirm-holdout`` is given (splits/holdout_policy.md).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc2.io.loader import load_kaggle_challenges, load_kaggle_solutions  # noqa: E402
from arc2.io.types import Candidate, Grid, TaskView, is_valid_grid  # noqa: E402
from arc2.io.writer import atomic_write_json, validate_submission, write_submission  # noqa: E402
from eval.arc2_score import grids_equal, score_submission  # noqa: E402

EVAL_DATA = _ROOT / "eval" / "data"
SPLIT_MANIFEST = _ROOT / "splits" / "arc2_eval_split_v1.json"
SPLIT_SOURCE = {
    "dev": "evaluation",
    "tune": "evaluation",
    "holdout": "evaluation",
    "eval-all": "evaluation",
    "train": "training",
}


# ----------------------------------------------------------------------------------------------
# Default selection (first two distinct valid candidates in order; documented fallbacks)
# ----------------------------------------------------------------------------------------------


def most_common_color(grid: Grid) -> int:
    counts = Counter(v for row in grid for v in row)
    # deterministic tie-break: lowest color among the most frequent
    best = max(counts.values())
    return min(c for c, n in counts.items() if n == best)


def fallback_grids(task: TaskView, test_index: int) -> tuple[Grid, Grid]:
    """Fallback 1: first train output's shape filled with the most common test-input color.
    Fallback 2: the test input itself."""
    test_input = task.test_inputs[test_index]
    shape_src = task.train[0].output
    fill = most_common_color(test_input)
    fb1 = [[fill] * len(shape_src[0]) for _ in range(len(shape_src))]
    fb2 = [list(r) for r in test_input]
    return fb1, fb2


class DefaultSelector:
    """Selector.select: first two distinct valid candidate grids by candidate order, padded with fallbacks."""

    name = "default_first_two_distinct"

    def __init__(self, task: TaskView) -> None:
        self.task = task

    def select(self, candidates: list[Candidate], test_index: int) -> tuple[Grid, Grid]:
        chosen: list[Grid] = []
        for c in candidates:
            if c.test_index != test_index or not is_valid_grid(c.grid):
                continue
            if any(grids_equal(c.grid, g) for g in chosen):
                continue
            chosen.append([list(r) for r in c.grid])
            if len(chosen) == 2:
                break
        for fb in fallback_grids(self.task, test_index):
            if len(chosen) == 2:
                break
            if not any(grids_equal(fb, g) for g in chosen):
                chosen.append(fb)
        while len(chosen) < 2:  # only if fallbacks collide with chosen grids; duplicate is still valid
            chosen.append([list(r) for r in chosen[0]])
        return chosen[0], chosen[1]


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def load_solver(spec: str, seed: int) -> Any:
    if ":" not in spec:
        raise SystemExit("--solver must be dotted.module.path:ClassName")
    mod_name, cls_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def git_info(ignore_dir: Path) -> tuple[str | None, bool | None]:
    """HEAD sha and whether anything (tracked or untracked) other than this run's output dir differs."""
    try:
        sha = subprocess.check_output(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(_ROOT), "status", "--porcelain", "--untracked-files=all"], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    try:
        rel = str(ignore_dir.resolve().relative_to(_ROOT)) + "/"
    except ValueError:
        rel = None
    dirty = False
    for line in status.splitlines():
        path = line[3:].strip()
        if rel is not None and path.startswith(rel):
            continue
        dirty = True
    return sha, dirty


def machine_info() -> dict[str, Any]:
    ram_gb = None
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    versions: dict[str, str] = {}
    for name in ("numpy", "torch"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            pass
    return {
        "hostname": socket.gethostname(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "gpu": gpu,
        "python": platform.python_version(),
        "package_versions": versions,
    }


def select_task_ids(split: str, all_ids: list[str]) -> tuple[list[str], str | None]:
    if split in ("train", "eval-all"):
        return sorted(all_ids), None
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    ids = manifest["splits"][split]
    unknown = sorted(set(ids) - set(all_ids))
    if unknown:
        raise SystemExit(f"split {split} references ids not in the challenges file: {unknown[:5]}")
    return sorted(ids), manifest["manifest_sha256"]


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True, help="dotted.module.path:ClassName")
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_SOURCE))
    ap.add_argument("--per-task-seconds", type=float, required=True)
    ap.add_argument("--out", required=True, help="output directory (run_record.json, per_task.json, submission.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N task ids (debugging)")
    ap.add_argument("--confirm-holdout", action="store_true", help="required for --split holdout")
    ap.add_argument("--fallback", default="[[0]]", help="JSON grid used by the writer for missing attempts")
    args = ap.parse_args(argv)

    if args.split == "holdout" and not args.confirm_holdout:
        raise SystemExit("refusing to evaluate the sealed holdout without --confirm-holdout (splits/holdout_policy.md)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = SPLIT_SOURCE[args.split]
    challenges_path = EVAL_DATA / f"arc-agi_{source}-challenges.json"
    solutions_path = EVAL_DATA / f"arc-agi_{source}-solutions.json"

    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    tasks = load_kaggle_challenges(challenges_path)  # label-free TaskViews
    task_ids, split_hash = select_task_ids(args.split, list(tasks))
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    solver = load_solver(args.solver, args.seed)
    solver_name = getattr(solver, "name", args.solver)
    fallback = json.loads(args.fallback)

    predictions: dict[str, list[dict[str, Grid]]] = {}
    pools: dict[str, dict[int, list[Grid]]] = {}
    per_task: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for tid in task_ids:
        task = tasks[tid]
        t0 = time.monotonic()
        deadline = t0 + args.per_task_seconds
        cands: list[Candidate] = []
        error: str | None = None
        try:
            cands = list(solver.propose(task, deadline))
        except Exception as exc:  # noqa: BLE001 - a crashing solver must not kill the run
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"task_id": tid, "error": error, "traceback": traceback.format_exc()})
        elapsed = time.monotonic() - t0
        selector = DefaultSelector(task)
        entries: list[dict[str, Grid]] = []
        pool: dict[int, list[Grid]] = {}
        for i in range(len(task.test_inputs)):
            a1, a2 = selector.select(cands, i)
            entries.append({"attempt_1": a1, "attempt_2": a2})
            pool[i] = [c.grid for c in cands if c.test_index == i and is_valid_grid(c.grid)]
        predictions[tid] = entries
        pools[tid] = pool
        per_task[tid] = {
            "time_s": elapsed,
            "overran_deadline": elapsed > args.per_task_seconds,
            "candidates_n": len(cands),
            "error": error,
            "outputs": [{"test_index": i, "candidates_n": len(pool[i])} for i in range(len(task.test_inputs))],
        }

    n_tests = {tid: len(tasks[tid].test_inputs) for tid in task_ids}
    submission_path = out_dir / "submission.json"
    writer_stats = write_submission(submission_path, predictions, task_ids, fallback=fallback, n_tests=n_tests)

    # Labels enter here only, after the submission is sealed on disk.
    with open(submission_path, "r", encoding="utf-8") as fh:
        submission = json.load(fh)
    challenges_subset = {tid: {"test": [{} for _ in range(n_tests[tid])]} for tid in task_ids}
    problems = validate_submission(submission, challenges_subset)
    all_solutions = load_kaggle_solutions(solutions_path)
    solutions = {tid: all_solutions[tid] for tid in task_ids}
    result = score_submission(submission, solutions)

    oracle_hits = 0
    for row in result["per_output"]:
        tid, i = row["task_id"], row["test_index"]
        truth = solutions[tid][i]
        hit = any(grids_equal(g, truth) for g in pools[tid].get(i, []))
        oracle_hits += int(hit)
        per_task[tid]["outputs"][i].update(
            {"correct_1": row["correct_1"], "correct_2": row["correct_2"], "shape_ok": row["shape_ok"], "oracle_pool_hit": hit}
        )
        per_task[tid]["outputs"][i]["time_s"] = per_task[tid]["time_s"] / len(per_task[tid]["outputs"])

    ended = time.time()
    git_sha, dirty = git_info(out_dir)
    config = {
        "solver": args.solver,
        "solver_name": solver_name,
        "selector": DefaultSelector.name,
        "split": args.split,
        "per_task_seconds": args.per_task_seconds,
        "seed": args.seed,
        "limit": args.limit,
        "fallback": fallback,
        "challenges_file": str(challenges_path.relative_to(_ROOT)),
    }
    run_id = out_dir.name
    record = {
        "schema": "run_record.v1",
        "run_id": run_id,
        "track": "arc2",
        "git_sha": git_sha,
        "dirty": dirty,
        "config": config,
        "config_hash": sha256_obj(config),
        "data_hash": sha256_file(challenges_path),
        "split_id": args.split if split_hash else f"{args.split}:{source}",
        "split_hash": split_hash,
        "ckpt_hash": getattr(solver, "ckpt_hash", None),
        "seed": args.seed,
        "machine": machine_info(),
        "budget": {"per_task_seconds": args.per_task_seconds, "n_tasks": len(task_ids), "total_task_seconds": args.per_task_seconds * len(task_ids)},
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": ended - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "metric": {
            "official_score": result["score"],
            "pass_at_1": result["pass_at_1"],
            "pass_at_2": result["pass_at_2"],
            "n_outputs": result["n_outputs"],
            "n_solved": result["n_solved"],
            "n_tasks": result["n_tasks"],
            "n_tasks_all_outputs": result["n_tasks_all_outputs"],
            "task_all_outputs_rate": result["task_all_outputs_rate"],
            "invalid_rate": result["invalid_rate"],
            "shape_acc": result["shape_acc"],
            "oracle_pool_pass_at_2": oracle_hits / result["n_outputs"],
            "submission_problems": problems,
            "writer_stats": writer_stats,
            "solver_time_s": sum(r["time_s"] for r in per_task.values()),
            "n_overran_deadline": sum(1 for r in per_task.values() if r["overran_deadline"]),
        },
        "per_case": {tid: {"correct_outputs": sum(1 for o in r["outputs"] if o["correct_1"] or o["correct_2"]), "n_outputs": len(r["outputs"]), "time_s": r["time_s"]} for tid, r in per_task.items()},
        "failures": failures,
        "command": " ".join([sys.executable] + sys.argv if argv is None else [sys.executable, __file__] + list(argv)),
    }
    atomic_write_json(out_dir / "run_record.json", record, indent=1)
    atomic_write_json(out_dir / "per_task.json", per_task, indent=1)
    summary = {k: record["metric"][k] for k in ("official_score", "pass_at_1", "n_outputs", "n_solved", "n_tasks", "shape_acc", "invalid_rate", "oracle_pool_pass_at_2", "n_overran_deadline")}
    summary.update({"run_id": run_id, "split": args.split, "wall_s": round(record["wall_s"], 3), "failures": len(failures), "submission_problems": len(problems)})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
