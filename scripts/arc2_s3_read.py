#!/usr/bin/env python3
"""S3 (experiments/arc2-bank/S3_RULE.md) section 4: B0 with colour 0 fixed against the bank's B0 on the same tasks.
CPU only.

    python scripts/arc2_s3_read.py          # archives experiments/runs/kaggle-A2-s3-k<i>-v1 -> experiments/arc2-bank/s3_result.json

Per kernel of kaggle/arc2/bank/s3/S3_PLAN.json, the floor (section 6): pool files only for its listed tasks; every
beam key's colour permutation starts with 'permute0'; pool/_run.json's run info (variant s3-k<i>, 64 sequences,
floor 0.2, seed 0, s3.colour0_fixed) and the solver's torch / transformers / unsloth / peft / triton equal to the
bank's; rank_check_ok and kgmon equal to the submitted pair on every output. A failed item makes the kernel's tasks
"report only" (read apart). A listed task without a pool file or with end_time_reached is "not run" and leaves both
arms; a kernel without an archive leaves all its tasks.

Lines on the N outputs of the tasks run, against the bank's B0 on the same outputs (Kaggle's solutions copy as truth;
pass@2 of each arm's own attempts; exact one-sided McNemar = the bank tools' sign test):
  1. B0 alone: gain at net >= ceil(5 N / 145) and p <= 0.05; no gain at net <= floor(N / 145); otherwise report.
  2. Entry line: v7_direct with S3's pool against v7_direct on the bank (the bank's 27B samples), gate v2 (net >= 3,
     lost <= 1), unscaled.
Recorded: the B0-pool oracle in each arm, gained / lost outputs and tasks, the time ratio (task seconds, sum and
median; above 1.1 the entry line is marked "time not neutral"), cap stops per arm, the board-point bootstrap of both
lines, and the kernel of every output.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.arc2_bank import (BANK_PARTS, CHALLENGES, Q27_BANK, RUNS, b0r_check, bootstrap_difference,  # noqa: E402
                            compare, grids_equal, load_bank, replay)

SCHEMA = "arc2_s3_result.v1"
PLAN = ROOT / "kaggle" / "arc2" / "bank" / "s3" / "S3_PLAN.json"
OUT = ROOT / "experiments" / "arc2-bank" / "s3_result.json"
ARCHIVE = "kaggle-A2-s3-{k}-v1"
ENV_KEYS = ("torch", "transformers", "unsloth", "peft", "triton")
PERMUTE = re.compile(r"permute(\d{10})")
TIME_NEUTRAL_MAX = 1.1


def bank_environment(pools=BANK_PARTS) -> dict:
    run = json.loads((Path(pools[0]) / "_run.json").read_text(encoding="utf-8"))["run"]
    solver = run["environment"]["solver"]
    return {k: solver.get(k) for k in ENV_KEYS}


def check_kernel(k: str, spec: dict, pool: Path, bank_env: dict) -> dict:
    """Section 6 on one kernel's pool directory: {status: valid | report_only | missing, run, not_run, problems}."""
    listed = list(spec["task_ids"])
    if not pool.is_dir():
        return {"kernel": k, "status": "missing", "run": [], "not_run": listed, "problems": [f"no archive at {pool}"],
                "beams_checked": 0}
    problems, run, not_run, n_beams = [], [], [], 0
    files = {p.stem: p for p in pool.glob("*.json") if not p.name.startswith("_")}
    extra = sorted(set(files) - set(listed))
    if extra:
        problems.append(f"pool files for unlisted tasks: {extra}")
    for t in listed:
        if t not in files:
            not_run.append(t)
            continue
        doc = json.loads(files[t].read_text(encoding="utf-8"))
        info = doc.get("run") or {}
        if info.get("end_time_reached"):
            not_run.append(t)
            continue
        run.append(t)
        if info.get("colour0_fixed") is not True:
            problems.append(f"{t}: run info colour0_fixed {info.get('colour0_fixed')}")
        for o in doc.get("outputs") or []:
            if o.get("rank_check_ok") is False:
                problems.append(f"{t} output {o.get('test_index')}: rank_check_ok false")
            sub = o.get("submitted") or {}
            sel = o.get("selected") or []
            if sub and sel and not (grids_equal(sub.get("attempt_1"), sel[0])
                                    and (len(sel) < 2 or grids_equal(sub.get("attempt_2"), sel[1]))):
                problems.append(f"{t} output {o.get('test_index')}: submitted differs from kgmon's pair")
            for c in o.get("candidates") or []:
                bad = b0r_check(c)
                if bad:
                    problems.append(f"{t} output {o.get('test_index')}: candidate fields {bad}")
                for b in c.get("beams") or []:
                    n_beams += 1
                    perms = PERMUTE.findall(b.get("key", ""))
                    if len(perms) != 1 or not perms[0].startswith("0"):
                        problems.append(f"{t}: beam key {b.get('key')!r} without colour 0 fixed")
    meta_path = pool / "_run.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("run") or {}
        want = {"variant": f"s3-{k}", "ttt_sequences": 64, "dfs_min_prob": 0.2, "bank_seed": 0}
        got = {key: meta.get(key) for key in want}
        if got != want:
            problems.append(f"_run.json run {got} != {want}")
        if (meta.get("s3") or {}).get("colour0_fixed") is not True:
            problems.append(f"_run.json s3 {meta.get('s3')}")
        solver = ((meta.get("environment") or {}).get("solver") or {})
        env = {key: solver.get(key) for key in ENV_KEYS}
        if env != bank_env:
            problems.append(f"environment {env} != the bank's {bank_env}")
    else:
        problems.append("no pool/_run.json")
    return {"kernel": k, "status": "report_only" if problems else "valid", "run": run, "not_run": not_run,
            "problems": problems[:50], "n_problems": len(problems), "beams_checked": n_beams}


def _pool_oracle(bank, keys: set[str]) -> set[str]:
    return {o.key for o in bank.outputs() if o.key in keys
            and any(bank.truth(o) is not None and grids_equal(c["grid"], bank.truth(o)) for c in o.candidates)}


def _task_seconds(bank, tasks: list[str]) -> dict[str, float]:
    return {t: float(bank.tasks[t].run.get("elapsed_s") or 0.0) for t in tasks if t in bank.tasks}


def read_s3(plan: dict, archives: dict[str, Path], *, bank_pools=BANK_PARTS, q27_dir=Q27_BANK,
            challenges=CHALLENGES, solutions=None, n_boot: int = 4000, seed: int = 0) -> dict:
    bank_env = bank_environment(bank_pools)
    kernels = {k: check_kernel(k, spec, archives[k] / "pool", bank_env) for k, spec in plan["kernels"].items()}
    valid = [k for k in kernels if kernels[k]["status"] == "valid"]
    tasks = sorted(t for k in valid for t in kernels[k]["run"])
    kernel_of = {t: k for k in kernels for t in plan["kernels"][k]["task_ids"]}
    res = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule": plan.get("rule"),
           "kernels": kernels, "n_tasks_run": len(tasks),
           "report_only_tasks": sorted(t for k in kernels if kernels[k]["status"] == "report_only" for t in kernels[k]["run"]),
           "not_run_tasks": sorted(t for k in kernels for t in kernels[k]["not_run"])}
    if not tasks:
        res["outcome"] = "no task run in a valid kernel: nothing to read"
        return res
    base = load_bank(list(bank_pools), q27_dir, challenges=challenges, solutions=solutions, tasks=tasks)
    new = load_bank([archives[k] / "pool" for k in valid], q27_dir, challenges=challenges, solutions=solutions,
                    tasks=tasks)
    assert sorted(base.task_ids) == sorted(new.task_ids) == tasks
    b0_b, b0_n = replay(base, "b0"), replay(new, "b0")
    v7_b, v7_n = replay(base, "v7_direct"), replay(new, "v7_direct")
    l1 = compare(b0_b, b0_n, n_boot=0)
    l2 = compare(v7_b, v7_n, n_boot=0)
    n = l1["n_shared_outputs"]
    need = math.ceil(5 * n / 145)
    p = l1["sign_test_p"]
    if l1["net"] >= need and p is not None and p <= 0.05:
        verdict1 = "gain"
    elif l1["net"] <= math.floor(n / 145):
        verdict1 = "no gain"
    else:
        verdict1 = "report"
    entry = l2["gate_v2"]["pass"]
    sb, sn = _task_seconds(base, tasks), _task_seconds(new, tasks)
    ratio_sum = sum(sn.values()) / sum(sb.values()) if sum(sb.values()) else None
    ratio_med = statistics.median(sn[t] / sb[t] for t in tasks if sb.get(t)) if any(sb.values()) else None
    neutral = ratio_sum is not None and ratio_sum <= TIME_NEUTRAL_MAX
    if verdict1 == "gain" and entry:
        outcome = ("entry candidate: a v7-vr2 + S3 build under its own rule and commit run, only by a separate "
                   "decision")
    elif verdict1 == "gain":
        outcome = "S3 is a B0 training lever for Stage 1; no entry on its own"
    elif verdict1 == "no gain":
        outcome = "S3 is not a lever on this evidence; recorded"
    else:
        outcome = "reported; no decision"
    if verdict1 == "gain" and entry and not neutral:
        outcome += " (time not neutral: ratio above 1.1)"
    keys = set(b0_b.per_output)
    cap = lambda bank: sorted(t for t in tasks if bank.tasks[t].run.get("stop_reason") == "per_task_cap")  # noqa: E731
    res.update({
        "n_outputs": n, "tasks": tasks,
        "line1": {"base_right": l1["base_right"], "s3_right": l1["new_right"], "gained": l1["gained"],
                  "lost": l1["lost"], "net": l1["net"], "sign_test_p_one_sided": p, "net_needed": need,
                  "no_gain_at_or_below": math.floor(n / 145), "verdict": verdict1,
                  "gained_keys": l1["gained_keys"], "lost_keys": l1["lost_keys"],
                  "board": bootstrap_difference(b0_b, b0_n, n_boot=n_boot, seed=seed)},
        "line2": {"base_right": l2["base_right"], "s3_right": l2["new_right"], "gained": l2["gained"],
                  "lost": l2["lost"], "net": l2["net"], "gate_v2": entry, "sign_test_p_one_sided": l2["sign_test_p"],
                  "gained_keys": l2["gained_keys"], "lost_keys": l2["lost_keys"],
                  "board": bootstrap_difference(v7_b, v7_n, n_boot=n_boot, seed=seed)},
        "outcome": outcome,
        "recorded": {
            "b0_pool_oracle": {"bank": len(_pool_oracle(base, keys)), "s3": len(_pool_oracle(new, keys))},
            "time": {"bank_task_s": round(sum(sb.values()), 1), "s3_task_s": round(sum(sn.values()), 1),
                     "ratio_sum": None if ratio_sum is None else round(ratio_sum, 3),
                     "ratio_median": None if ratio_med is None else round(ratio_med, 3), "neutral": neutral},
            "cap_stops": {"bank": cap(base), "s3": cap(new)},
            "gained_tasks": sorted({k.rsplit("_", 1)[0] for k in l1["gained_keys"]}),
            "lost_tasks": sorted({k.rsplit("_", 1)[0] for k in l1["lost_keys"]}),
            "kernel_of_output": {key: kernel_of.get(key.rsplit("_", 1)[0]) for key in sorted(keys)},
        },
        "fingerprints": {"bank": base.fingerprint(), "s3": new.fingerprint()},
        "sources": {"bank": base.sources, "s3": new.sources},
    })
    return res


def summary_lines(res: dict) -> list[str]:
    out = ["S3 reading (experiments/arc2-bank/S3_RULE.md)"]
    for k, kv in res["kernels"].items():
        out.append(f"  {k}: {kv['status']}; run {len(kv['run'])}, not run {len(kv['not_run'])}; beams checked "
                   f"{kv['beams_checked']}; problems {kv.get('n_problems', len(kv['problems']))}")
    if "line1" not in res:
        out.append(f"  outcome: {res['outcome']}")
        return out
    l1, l2 = res["line1"], res["line2"]
    out += [f"  on {res['n_outputs']} outputs of {res['n_tasks_run']} tasks run (not run {len(res['not_run_tasks'])}, "
            f"report only {len(res['report_only_tasks'])})",
            "| track | line | bank | S3 | gained | lost | net | p (one-sided) | reading |",
            f"| ARC-2 | 1 B0 alone | {l1['base_right']} | {l1['s3_right']} | {l1['gained']} | {l1['lost']} | "
            f"{l1['net']:+d} | {l1['sign_test_p_one_sided']} | {l1['verdict']} (gain needs net >= {l1['net_needed']} "
            f"and p <= 0.05) |",
            f"| ARC-2 | 2 merged (v7_direct) | {l2['base_right']} | {l2['s3_right']} | {l2['gained']} | {l2['lost']} | "
            f"{l2['net']:+d} | {l2['sign_test_p_one_sided']} | gate v2 {'PASS' if l2['gate_v2'] else 'FAIL'} |",
            f"  outcome: {res['outcome']}"]
    r = res["recorded"]
    out.append(f"  B0-pool oracle: bank {r['b0_pool_oracle']['bank']}, S3 {r['b0_pool_oracle']['s3']}; time ratio "
               f"{r['time']['ratio_sum']} (median {r['time']['ratio_median']}); cap stops bank {len(r['cap_stops']['bank'])}, "
               f"S3 {len(r['cap_stops']['s3'])}")
    out.append(f"  board points: line 1 {l1['board']['mean_points']:+.2f} (sd {l1['board']['sd_points']}), "
               f"line 2 {l2['board']['mean_points']:+.2f} (sd {l2['board']['sd_points']})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--runs", default=str(RUNS), help="directory holding kaggle-A2-s3-k<i>-v1")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--floor-only", default="", metavar="K",
                    help="the queue's post-command: section 6 on kernel K only, no line (the reading waits for all)")
    args = ap.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    archives = {k: Path(args.runs) / ARCHIVE.format(k=k) for k in plan["kernels"]}
    if args.floor_only:
        k = args.floor_only
        rep = check_kernel(k, plan["kernels"][k], archives[k] / "pool", bank_environment())
        if archives[k].is_dir():
            (archives[k] / "s3_floor.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(f"S3 floor {k}: {rep['status']}; run {len(rep['run'])}, not run {len(rep['not_run'])}; beams checked "
              f"{rep['beams_checked']}; problems {rep.get('n_problems', len(rep['problems']))}")
        return 0 if rep["status"] == "valid" else 1
    res = read_s3(plan, archives)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(res)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
