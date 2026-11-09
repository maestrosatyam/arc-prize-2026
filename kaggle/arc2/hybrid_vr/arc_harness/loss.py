"""Where each missed test output was lost.

For every test output that the final attempts miss, the first matching category (in this order):
  parse_failure            a sample finished (own / rescued) but produced no candidate grid at all for this output
  not_in_top_two           a right grid exists among the candidates (written or program output) but was not picked
  verified_but_wrong       a program passed every demonstration and its test output is wrong; nothing right exists
  right_rule_wrong_output  a near miss: a candidate has the right shape and >= 90% right cells (or <= 2 wrong
                           cells), or a program
                           matched all but one demonstration (the rule was essentially found, the execution was off)
  ran_out_of_budget        no sample for the task finished on its own (every sample was cut; rescues do not count)
  no_right_rule            samples finished on their own and nothing came close
Categories are heuristics: 'right rule' is read from the artefacts, not from the reasoning.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .grids import cell_accuracy, grid_shape, is_grid

CATEGORIES = ["parse_failure", "not_in_top_two", "verified_but_wrong", "right_rule_wrong_output",
              "ran_out_of_budget", "no_right_rule"]


def _cands(samples: List[dict], j: int):
    for s in samples:
        g = s["grids"][j] if j < len(s["grids"]) else None
        if is_grid(g):
            yield g, s, "grid"
        po = (s.get("verify") or {}).get("program_outputs") or []
        if j < len(po) and is_grid(po[j]):
            yield po[j], s, "program"


def classify_output(task: dict, samples: List[dict], j: int, attempts: dict) -> dict:
    truth = task["test"][j]["output"]
    a1, a2 = attempts.get("attempt_1"), attempts.get("attempt_2")
    hit = (is_grid(a1) and a1 == truth) or (is_grid(a2) and a2 == truth)
    cands = list(_cands(samples, j))
    finished = [s for s in samples if s["finish"] in ("own", "rescued", "length")]
    own = [s for s in samples if s["finish"] == "own"]
    row = {"task_id": task["id"], "test": j, "hit": hit, "n_samples": len(samples), "n_candidates": len(cands),
           "n_own_finish": len(own), "category": None}
    if hit:
        return row
    if not cands:
        row["category"] = "parse_failure" if finished else "ran_out_of_budget"
        return row
    if any(g == truth for g, _, _ in cands):
        row["category"] = "not_in_top_two"
        return row
    verified = [s for s in samples if (s.get("verify") or {}).get("passed")]
    if verified:
        row["category"] = "verified_but_wrong"
        return row
    best_acc = max(cell_accuracy(g, truth) if grid_shape(g) == grid_shape(truth) else 0.0 for g, _, _ in cands)
    n_cells = sum(len(r) for r in truth)
    min_wrong = min((round((1 - cell_accuracy(g, truth)) * n_cells) if grid_shape(g) == grid_shape(truth) else n_cells)
                    for g, _, _ in cands)
    n_train = len(task["train"])
    near_prog = any((s.get("verify") or {}).get("ran") and n_train >= 2 and s["verify"]["n_pass"] >= n_train - 1
                    for s in samples)
    row["best_cell_accuracy"] = round(best_acc, 3)
    row["min_wrong_cells"] = min_wrong
    if best_acc >= 0.9 or min_wrong <= 2 or near_prog:
        row["category"] = "right_rule_wrong_output"
        return row
    if not own:
        row["category"] = "ran_out_of_budget"
        return row
    row["category"] = "no_right_rule"
    return row


def attribute(tasks: Dict[str, dict], samples: Dict[str, List[dict]], submission: dict) -> dict:
    rows = []
    for tid, task in tasks.items():
        for j, te in enumerate(task["test"]):
            if "output" not in te:
                continue
            att = (submission.get(tid) or [{}] * (j + 1))[j] if tid in submission and j < len(submission[tid]) else {}
            rows.append(classify_output(task, samples.get(tid, []), j, att))
    counts = Counter(r["category"] for r in rows if r["category"])
    n = len(rows)
    right = sum(r["hit"] for r in rows)
    return {"n_outputs": n, "right": right, "missed": n - right,
            "by_category": {c: counts.get(c, 0) for c in CATEGORIES}, "rows": rows}


def format_report(rep: dict) -> str:
    lines = [f"right {rep['right']}/{rep['n_outputs']} ({rep['right']/max(1,rep['n_outputs']):.1%}); missed {rep['missed']}:"]
    for c in CATEGORIES:
        k = rep["by_category"][c]
        lines.append(f"  {c:26s} {k:4d}  ({k/max(1,rep['missed']):.0%} of misses)")
    return "\n".join(lines)
