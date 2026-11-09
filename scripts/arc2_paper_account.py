"""Where each output ends up on the ARC-AGI-2 output bank, by group (CPU only, no GPU).

For the ARC-AGI-2 paper (reports/paper/arc2/PAPER_ARC2.md, Section 1). Per output of the 101 unsealed bank tasks
(145 outputs; `eval/arc2_bank.py`, the 19 sealed tasks are refused by the loader), for B0 alone (rule `b0`, B0's own
pool) and for v7's direct merge (rule `v7_direct`, B0's pool plus the 27B's samples):
- right: submitted and right (pass@2 of the rule's two attempts);
- pool: the right grid is in the rule's candidates but was not submitted;
- never: the right grid is in no candidate.
Groups: all outputs; the four task-length quartiles (`eval.arc2_bank.quartiles_by_size`, q1 = the shortest quarter,
the hybrid's band); the player difficulty labels 1-5 of the team's ARC-AGI-2 play release (`arc2_tasks`, evaluation
rows, field `difficulty`; one label per task, so a task's outputs share it). Only derived counts are written.

Usage: envs/kaggle-sim/bin/python -I scripts/arc2_paper_account.py --derived <release>/data/play/derived
       [--out experiments/arc2-bank/account.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.arc2_bank import _q27_grids, grids_equal, load_bank, quartiles_by_size, replay  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", required=True)
    p.add_argument("--out", default=str(ROOT / "experiments/arc2-bank/account.json"))
    args = p.parse_args()
    rows = [json.loads(line) for line in open(Path(args.derived) / "arc2_tasks.jsonl")]
    diff = {r["task_id"]: r["difficulty"] for r in rows if r["set"] == "evaluation"}
    bank = load_bank()
    assert len(bank.task_ids) == 101 and bank.n_outputs == 145 and set(bank.task_ids) <= set(diff)
    right = {r: replay(bank, r).right_keys() for r in ("b0", "v7_direct")}

    in_pool: dict[str, dict[str, bool]] = {"b0": {}, "v7_direct": {}}
    for t in bank.task_ids:
        task = bank.tasks[t]
        for j, o in enumerate(task.outputs):
            truth = bank.truth(o)
            b0 = truth is not None and any(grids_equal(c["grid"], truth) for c in o.candidates)
            q27 = truth is not None and any(grids_equal(g, truth) for _, g in _q27_grids(bank, task, j))
            in_pool["b0"][o.key] = b0
            in_pool["v7_direct"][o.key] = b0 or q27

    groups: dict[str, list[str]] = {"all": list(bank.task_ids)}
    groups.update({q: list(ts) for q, ts in quartiles_by_size(bank).items()})
    for d in sorted({diff[t] for t in bank.task_ids}):
        groups[f"difficulty_{d}"] = [t for t in bank.task_ids if diff[t] == d]

    out = {"bank": {"fingerprint": bank.fingerprint(), "n_tasks": len(bank.task_ids), "n_outputs": bank.n_outputs},
           "labels": "team ARC-AGI-2 play release, arc2_tasks evaluation rows, field difficulty (1-5, one per task)",
           "groups": {}}
    for name, tasks in groups.items():
        keys = [o.key for t in tasks for o in bank.tasks[t].outputs]
        g = {"n_tasks": len(tasks), "n_outputs": len(keys)}
        for r in ("b0", "v7_direct"):
            n_right = sum(k in right[r] for k in keys)
            n_pool = sum(in_pool[r][k] for k in keys)
            assert all(in_pool[r][k] for k in keys if k in right[r])
            g[r] = {"right": n_right, "pool": n_pool - n_right, "never": len(keys) - n_pool}
        out["groups"][name] = g
    a = out["groups"]["all"]
    assert (a["b0"]["right"], a["b0"]["pool"], a["b0"]["never"]) == (40, 6, 99), a
    assert (a["v7_direct"]["right"], a["v7_direct"]["pool"], a["v7_direct"]["never"]) == (45, 10, 90), a
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for name, g in out["groups"].items():
        print(f"{name:14s} tasks {g['n_tasks']:3d} outputs {g['n_outputs']:3d}  b0 {g['b0']}  merge {g['v7_direct']}")


if __name__ == "__main__":
    main()
