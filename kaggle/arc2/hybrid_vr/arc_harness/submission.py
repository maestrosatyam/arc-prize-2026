"""An always-valid submission.json.

validate_submission() enforces the Kaggle shape: every task id, one entry per test input, attempt_1 and attempt_2
both present and valid grids. write_submission() writes atomically (temp file + rename), so a crash mid-write
never leaves a half file. The harness writes an identity submission first thing, then rewrites after every task.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

from .grids import is_grid


def identity_submission(tasks: Dict[str, dict]) -> Dict[str, List[dict]]:
    sub = {}
    for tid, t in tasks.items():
        sub[tid] = [{"attempt_1": [r[:] for r in te["input"]], "attempt_2": [[0]]} for te in t["test"]]
    return sub


def validate_submission(sub: dict, tasks: Dict[str, dict]) -> List[str]:
    problems = []
    for tid, t in tasks.items():
        if tid not in sub:
            problems.append(f"{tid}: missing")
            continue
        ent = sub[tid]
        if not isinstance(ent, list) or len(ent) != len(t["test"]):
            problems.append(f"{tid}: expected {len(t['test'])} test entries")
            continue
        for j, e in enumerate(ent):
            for a in ("attempt_1", "attempt_2"):
                if a not in e or not is_grid(e[a]):
                    problems.append(f"{tid}[{j}].{a}: not a valid grid")
    extra = set(sub) - set(tasks)
    if extra:
        problems.append(f"extra ids: {sorted(extra)[:5]}")
    return problems


def repair_submission(sub: dict, tasks: Dict[str, dict]) -> dict:
    """Fill anything missing or invalid with the identity fallback, never raising."""
    fixed = identity_submission(tasks)
    for tid, t in tasks.items():
        ent = sub.get(tid)
        if not isinstance(ent, list):
            continue
        for j in range(min(len(ent), len(t["test"]))):
            e = ent[j] if isinstance(ent[j], dict) else {}
            for a in ("attempt_1", "attempt_2"):
                if is_grid(e.get(a)):
                    fixed[tid][j][a] = e[a]
    return fixed


def write_submission(sub: dict, tasks: Dict[str, dict], path: str) -> dict:
    sub = repair_submission(sub, tasks)
    assert not validate_submission(sub, tasks)
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".sub-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(sub, f)
    os.replace(tmp, path)
    return sub
