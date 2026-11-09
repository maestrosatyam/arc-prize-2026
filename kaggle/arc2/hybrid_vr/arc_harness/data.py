"""Task loading and splits.

A task is {"id": str, "train": [{"input": grid, "output": grid}, ...], "test": [{"input": grid, "output": grid?}, ...]}.
Grids are lists of lists of ints 0-9.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

Grid = List[List[int]]

SPLIT_SALT = "arc2-split-v1"
HOLDOUT_SIZE = 40


def load_dir(path: str) -> Dict[str, dict]:
    """Load every *.json task file in a directory (GitHub ARC-AGI / ARC-AGI-2 layout)."""
    tasks = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".json"):
            continue
        tid = name[:-5]
        with open(os.path.join(path, name)) as f:
            t = json.load(f)
        tasks[tid] = {"id": tid, "train": t["train"], "test": t["test"]}
    return tasks


def load_kaggle(challenges_path: str, solutions_path: Optional[str] = None) -> Dict[str, dict]:
    """Load the Kaggle challenge file (and optionally the solutions file) into tasks."""
    with open(challenges_path) as f:
        ch = json.load(f)
    sol = {}
    if solutions_path and os.path.exists(solutions_path):
        with open(solutions_path) as f:
            sol = json.load(f)
    tasks = {}
    for tid, t in ch.items():
        tests = [{"input": x["input"]} for x in t["test"]]
        if tid in sol:
            for j, out in enumerate(sol[tid]):
                tests[j]["output"] = out
        tasks[tid] = {"id": tid, "train": t["train"], "test": tests}
    return tasks


def load_conceptarc(root: str) -> Dict[str, dict]:
    """ConceptARC corpus: corpus/<Concept>/<Concept><n>.json, same task format."""
    tasks = {}
    corpus = os.path.join(root, "corpus")
    if not os.path.isdir(corpus):
        return tasks
    for concept in sorted(os.listdir(corpus)):
        d = os.path.join(corpus, concept)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(d, name)) as f:
                t = json.load(f)
            tid = f"conceptarc/{name[:-5]}"
            tasks[tid] = {"id": tid, "train": t["train"], "test": t["test"]}
    return tasks


def public_sets(data_root: str) -> Dict[str, Dict[str, dict]]:
    """All public task sets under data/: arc2_train, arc2_eval, arc1_train, arc1_eval, conceptarc."""
    out = {}
    pairs = {
        "arc2_train": "ARC-AGI-2/data/training",
        "arc2_eval": "ARC-AGI-2/data/evaluation",
        "arc1_train": "ARC-AGI/data/training",
        "arc1_eval": "ARC-AGI/data/evaluation",
    }
    for k, rel in pairs.items():
        p = os.path.join(data_root, rel)
        if os.path.isdir(p):
            out[k] = load_dir(p)
    ca = os.path.join(data_root, "ConceptARC")
    if os.path.isdir(ca):
        out["conceptarc"] = load_conceptarc(ca)
    return out


def split_key(task_id: str, salt: str = SPLIT_SALT) -> str:
    return hashlib.sha256((task_id + salt).encode()).hexdigest()


def holdout_split(task_ids: Iterable[str], n_holdout: int = HOLDOUT_SIZE, salt: str = SPLIT_SALT):
    """Sort ids by sha256(task_id + salt); the last n are the untouched holdout.

    Returns (dev_ids, holdout_ids), each in the sorted order.
    """
    ordered = sorted(task_ids, key=lambda t: split_key(t, salt))
    return ordered[:-n_holdout], ordered[-n_holdout:]


def task_size(task: dict) -> int:
    """Cells in the task text: the proxy for prompt length used for shortest-first ordering."""
    n = 0
    for p in task["train"]:
        n += _cells(p["input"]) + _cells(p["output"])
    for t in task["test"]:
        n += _cells(t["input"])
    return n


def _cells(g: Grid) -> int:
    return sum(len(r) for r in g)


def n_test_outputs(tasks: Dict[str, dict]) -> int:
    return sum(len(t["test"]) for t in tasks.values())
