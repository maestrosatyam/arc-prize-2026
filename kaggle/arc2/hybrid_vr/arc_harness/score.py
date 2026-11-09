"""Local scorer: Kaggle's rule (1 if either attempt matches exactly; mean over all test outputs)."""
from __future__ import annotations

from typing import Dict, List, Optional

from .grids import is_grid


def score_submission(sub: dict, tasks: Dict[str, dict]) -> dict:
    rows = []
    for tid, t in tasks.items():
        for j, te in enumerate(t["test"]):
            if "output" not in te:
                continue
            truth = te["output"]
            e = (sub.get(tid) or [{}] * (j + 1))[j] if tid in sub and j < len(sub[tid]) else {}
            a1, a2 = e.get("attempt_1"), e.get("attempt_2")
            hit1 = is_grid(a1) and a1 == truth
            hit2 = is_grid(a2) and a2 == truth
            rows.append({"task_id": tid, "test": j, "hit": hit1 or hit2, "hit1": hit1, "hit2": hit2})
    n = len(rows)
    right = sum(r["hit"] for r in rows)
    tasks_full = {}
    for r in rows:
        tasks_full.setdefault(r["task_id"], True)
        tasks_full[r["task_id"]] &= r["hit"]
    return {"n_outputs": n, "right": right, "score": right / n if n else 0.0,
            "attempt1_right": sum(r["hit1"] for r in rows), "n_tasks": len(tasks_full),
            "tasks_fully_right": sum(tasks_full.values()), "rows": rows}


def candidate_is_right(grid: Optional[list], truth: list) -> bool:
    return grid is not None and is_grid(grid) and grid == truth


def per_output_table(sub: dict, tasks: Dict[str, dict]) -> List[dict]:
    return score_submission(sub, tasks)["rows"]
