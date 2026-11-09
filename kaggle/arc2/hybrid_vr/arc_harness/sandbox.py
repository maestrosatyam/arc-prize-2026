"""Run the model's transform(grid) in a subprocess and grade it on the demonstrations.

Feedback follows Poetiq's format (MIT, Copyright 2025 Poetiq, Inc.): per example correct/incorrect, a shape
message, a diff grid with wrong cells as 'prediction/correct', the cell accuracy, and any error text.
All grids of one program run in one subprocess (one interpreter start per program instead of per grid).
The subprocess is not a security sandbox.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from .grids import Grid, cell_accuracy, grid_shape, is_grid

PER_GRID_TIMEOUT_S = 5.0
MAX_TIMEOUT_S = 40.0
# The interpreter that runs the model's programs. In the hybrid notebook the harness runs under the serving Python
# (3.12, no numpy outside the vLLM site, which -I hides), so the notebook points this at the image's own Python.
SANDBOX_PYTHON_ENV = "ARC_SANDBOX_PYTHON"


def sandbox_python() -> str:
    return os.environ.get(SANDBOX_PYTHON_ENV) or sys.executable

RUNNER = r'''
import json, sys, signal
import numpy as np
try:
    import scipy  # noqa: F401
except Exception:
    pass
code = sys.stdin.readline()
grids = json.loads(sys.stdin.readline())
ns = {}
try:
    exec(compile(json.loads(code), "<model>", "exec"), ns)
    fn = ns["transform"]
except Exception as e:
    print(json.dumps([{"out": None, "err": "definition failed: %s: %s" % (type(e).__name__, e)}] * len(grids)))
    sys.exit(0)
res = []
class _TO(Exception):
    pass
def _alarm(*a):
    raise _TO()
signal.signal(signal.SIGALRM, _alarm)
for g in grids:
    signal.alarm(__ALARM__)
    try:
        out = fn(np.array(g))
        signal.alarm(0)
        if hasattr(out, "tolist"):
            out = out.tolist()
        out = [[int(c) for c in row] for row in out]
        res.append({"out": out, "err": None})
    except _TO:
        res.append({"out": None, "err": "timeout"})
    except Exception as e:
        signal.alarm(0)
        res.append({"out": None, "err": "%s: %s" % (type(e).__name__, str(e)[:300])})
print(json.dumps(res))
'''.replace("__ALARM__", str(int(PER_GRID_TIMEOUT_S)))


def run_program(code: str, grids: List[Grid]) -> List[dict]:
    """Run transform on each grid. Returns [{"out": grid|None, "err": str|None}, ...]."""
    if not grids:
        return []
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "runner.py")
        with open(path, "w") as f:
            f.write(RUNNER)
        stdin = json.dumps(code) + "\n" + json.dumps(grids) + "\n"
        timeout = min(MAX_TIMEOUT_S, PER_GRID_TIMEOUT_S * len(grids) + 3)
        try:
            p = subprocess.run([sandbox_python(), "-I", path], input=stdin, capture_output=True, text=True,
                               timeout=timeout, cwd=td)
        except subprocess.TimeoutExpired:
            return [{"out": None, "err": "timeout"}] * len(grids)
    if p.returncode != 0:
        return [{"out": None, "err": (p.stderr or p.stdout).strip()[-500:]}] * len(grids)
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return [{"out": None, "err": "runner produced no result"}] * len(grids)
    for r in res:
        if r["out"] is not None and not is_grid(r["out"]):
            r["err"] = (r["err"] or "") + " output is not a valid grid (rectangular, 1-30 per side, values 0-9)"
            r["bad"] = r["out"]
            r["out"] = None
    return res


def diff_text(pred: Grid, truth: Grid) -> str:
    """Right cells as-is, wrong cells as 'prediction/correct'."""
    return "\n".join(" ".join(str(p) if p == t else f"{p}/{t}" for p, t in zip(pr, tr))
                     for pr, tr in zip(pred, truth))


@dataclass
class Grade:
    passed: bool                     # right on every demonstration
    score: float                     # mean cell accuracy over demonstrations
    n_pass: int                      # demonstrations matched exactly
    feedback: str                    # Poetiq-style text for the model
    test_outputs: List[Optional[Grid]]
    errors: List[Optional[str]] = field(default_factory=list)


def grade(code: str, train: List[dict], test: List[dict]) -> Grade:
    grids = [p["input"] for p in train] + [t["input"] for t in test]
    res = run_program(code, grids)
    scores, notes, errs = [], [], []
    for i, (p, r) in enumerate(zip(train, res[:len(train)]), 1):
        truth = p["output"]
        pred = r["out"]
        errs.append(r["err"])
        if pred is not None and pred == truth:
            scores.append(1.0)
            notes.append(f"Example #{i}: correct.")
            continue
        note = f"Example #{i}: incorrect."
        if pred is None:
            scores.append(0.0)
            note += "\nThe output has to be a rectangular grid of integers 0-9."
        elif grid_shape(pred) != grid_shape(truth):
            scores.append(0.0)
            note += f"\nShape mismatch: your output shape was {grid_shape(pred)}, the correct shape is {grid_shape(truth)}."
        else:
            acc = cell_accuracy(pred, truth)
            scores.append(acc)
            note += (f"\nWrong cells shown as 'prediction/correct':\n```\n{diff_text(pred, truth)}\n```"
                     f"\nCell accuracy: {acc:.2f} (0 worst, 1 best).")
        if r["err"]:
            note += f"\nYour code produced this error:\n{r['err']}"
        notes.append(note)
    test_outputs = [r["out"] for r in res[len(train):]]
    n_pass = sum(1 for s in scores if s == 1.0)
    return Grade(passed=bool(scores) and n_pass == len(scores), score=float(sum(scores) / max(1, len(scores))),
                 n_pass=n_pass, feedback="\n\n".join(notes), test_outputs=test_outputs, errors=errs)
