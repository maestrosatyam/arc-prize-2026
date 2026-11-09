#!/usr/bin/env python3
"""S1 (experiments/arc2-bank/S1_RULE.md) section 4: B0 with 128 training sequences against the bank's B0 on the same
tasks. CPU only.

    python scripts/arc2_s1_read.py          # archives experiments/runs/kaggle-A2-s1-k<i>-v1 -> experiments/arc2-bank/s1_result.json
    python scripts/arc2_s1_read.py --floor-only k1

Per kernel of kaggle/arc2/bank/s1/S1_PLAN.json, the floor (section 7): pool files only for its listed tasks; run info
per task and in pool/_run.json (variant s1-<k>, 128 sequences, ttt_aug_n 16, floor 0.2, seed 0) and the solver's
torch / transformers / unsloth / peft / triton equal to the bank's; rank_check_ok, kgmon equal to the submitted pair
and B0R's candidate fields on every output. A failed item makes the kernel's tasks "report only" (read apart). A
listed task without a pool file or with end_time_reached is "not run" and leaves both arms.

Lines on the N outputs of the tasks run, against the bank's B0 on the same outputs (Kaggle's solutions copy as truth;
pass@2 of each arm's own attempts; exact one-sided sign test):
  1. B0 alone: gain at net >= ceil(5 N / 145) and p <= 0.05; no gain at net <= floor(N / 145); otherwise report.
  2. Entry line: TCP (section 5; eval/arc2_bank.py tcp, the bank's E13 as the reference) against v7_direct on the
     bank, gate v2 (net >= 3, lost <= 1).
Recorded: the B0-pool oracle in each arm, gained / lost outputs and tasks, the time ratio (sum and median), training
seconds per task (the kernel log's train_runtime), cap stops per arm, the unscaled merged line, TCP's delta, k,
samples kept and B0 cut, and the board-point bootstrap of each line.
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

from eval.arc2_bank import (BANK_PARTS, CHALLENGES, Q27_BANK, b0r_check, bootstrap_difference, compare,  # noqa: E402
                            download_gaps, grids_equal, load_bank, replay, tcp, tcp_e13)

SCHEMA = "arc2_s1_result.v1"
PLAN = ROOT / "kaggle" / "arc2" / "bank" / "s1" / "S1_PLAN.json"
RUNS = ROOT / "experiments" / "runs"
OUT = ROOT / "experiments" / "arc2-bank" / "s1_result.json"
ARCHIVE = "kaggle-A2-s1-{k}-v1"
ENV_KEYS = ("torch", "transformers", "unsloth", "peft", "triton")
TRAIN_RUNTIME = re.compile(r"training stats for puzzle (\w+): .*?'train_runtime': ([0-9.]+)")


def bank_environment(pools=BANK_PARTS) -> dict:
    run = json.loads((Path(pools[0]) / "_run.json").read_text(encoding="utf-8"))["run"]
    solver = run["environment"]["solver"]
    return {k: solver.get(k) for k in ENV_KEYS}


def check_kernel(k: str, spec: dict, pool: Path, bank_env: dict) -> dict:
    """Section 7 on one kernel's pool directory: {status: valid | report_only | missing, run, not_run, problems}."""
    listed = list(spec["task_ids"])
    if not pool.is_dir():
        return {"kernel": k, "status": "missing", "run": [], "not_run": listed, "problems": [f"no archive at {pool}"]}
    problems, run, not_run = [], [], []
    files = {p.stem: p for p in pool.glob("*.json") if not p.name.startswith("_")}
    extra = sorted(set(files) - set(listed))
    if extra:
        problems.append(f"pool files for unlisted tasks: {extra}")
    gaps = download_gaps(pool)
    if gaps:
        problems.append(f"download incomplete, re-fetch the archive: {gaps}")
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
        want = {"ttt_sequences": 128, "ttt_aug_n": 16, "dfs_min_prob": 0.2, "bank_seed": 0}
        got = {key: info.get(key) for key in want}
        if got != want:
            problems.append(f"{t}: run info {got} != {want}")
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
    meta_path = pool / "_run.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("run") or {}
        want = {"variant": f"s1-{k}", "ttt_sequences": 128, "dfs_min_prob": 0.2, "bank_seed": 0}
        got = {key: meta.get(key) for key in want}
        if got != want:
            problems.append(f"_run.json run {got} != {want}")
        if not (meta.get("s1") or {}).get("rule"):
            problems.append(f"_run.json s1 {meta.get('s1')}")
        solver = ((meta.get("environment") or {}).get("solver") or {})
        env = {key: solver.get(key) for key in ENV_KEYS}
        if env != bank_env:
            problems.append(f"environment {env} != the bank's {bank_env}")
    else:
        problems.append("no pool/_run.json")
    return {"kernel": k, "status": "report_only" if problems else "valid", "run": run, "not_run": not_run,
            "problems": problems[:50], "n_problems": len(problems), "download_gaps": gaps}


def train_runtimes(archive: Path) -> dict[str, float]:
    out = {}
    for log in archive.glob("*.log"):
        text = log.read_text(encoding="utf-8", errors="replace").replace("\\n", "\n")
        for m in TRAIN_RUNTIME.finditer(text):
            out[m.group(1)] = float(m.group(2))
    return out


def _pool_oracle(bank, keys: set[str]) -> int:
    return sum(1 for o in bank.outputs() if o.key in keys
               and any(bank.truth(o) is not None and grids_equal(c["grid"], bank.truth(o)) for c in o.candidates))


def read_s1(plan: dict, archives: dict[str, Path], *, bank_pools=BANK_PARTS, q27_dir=Q27_BANK,
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
    v7_b = replay(base, "v7_direct")
    ref = tcp_e13(base, None, tasks)
    tc = tcp(new, ref, None, tasks)
    v7_tcp = replay(tc["bank"], "v7_direct")
    v7_unscaled = replay(new, "v7_direct")
    l1 = compare(b0_b, b0_n, n_boot=0)
    l2 = compare(v7_b, v7_tcp, n_boot=0)
    lu = compare(v7_b, v7_unscaled, n_boot=0)
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
    if verdict1 == "gain" and entry:
        outcome = ("entry candidate: decode-many's entry template with 128 sequences and B0's phase end "
                   "moved by delta, and a ~12 h commit run for the Nov 2 slot, under its own rule")
    elif verdict1 == "gain":
        outcome = ("S1 is a training lever that does not pay for its time in this entry; S2 (steps, not samples) may be "
                   "proposed, learning only")
    elif verdict1 == "no gain":
        outcome = "S1 is not a lever on this evidence; recorded"
    else:
        outcome = "reported; no decision"
    sb = {t: float(base.tasks[t].elapsed_s or 0.0) for t in tasks}
    sn = {t: float(new.tasks[t].elapsed_s or 0.0) for t in tasks}
    trn = {}
    for k in valid:
        trn.update(train_runtimes(archives[k]))
    keys = set(b0_b.per_output)
    cap = lambda bank: sorted(t for t in tasks if bank.tasks[t].run.get("stop_reason") == "per_task_cap")  # noqa: E731
    res.update({
        "n_outputs": n, "tasks": tasks,
        "line1": {"base_right": l1["base_right"], "s1_right": l1["new_right"], "gained": l1["gained"], "lost": l1["lost"],
                  "net": l1["net"], "sign_test_p_one_sided": p, "net_needed": need,
                  "no_gain_at_or_below": math.floor(n / 145), "verdict": verdict1,
                  "gained_keys": l1["gained_keys"], "lost_keys": l1["lost_keys"],
                  "board": bootstrap_difference(b0_b, b0_n, n_boot=n_boot, seed=seed)},
        "line2": {"base_right": l2["base_right"], "s1_right": l2["new_right"], "gained": l2["gained"], "lost": l2["lost"],
                  "net": l2["net"], "gate_v2": entry, "sign_test_p_one_sided": l2["sign_test_p"],
                  "gained_keys": l2["gained_keys"], "lost_keys": l2["lost_keys"],
                  "tcp": {kk: v for kk, v in tc.items() if kk != "bank"}, "reference_e13": ref,
                  "board": bootstrap_difference(v7_b, v7_tcp, n_boot=n_boot, seed=seed)},
        "outcome": outcome,
        "recorded": {
            "b0_pool_oracle": {"bank": _pool_oracle(base, keys), "s1": _pool_oracle(new, keys)},
            "time": {"bank_task_s": round(sum(sb.values()), 1), "s1_task_s": round(sum(sn.values()), 1),
                     "ratio_sum": round(sum(sn.values()) / sum(sb.values()), 3),
                     "ratio_median": round(statistics.median(sn[t] / sb[t] for t in tasks if sb[t]), 3)},
            "train_runtime_s": {"n": len(trn), "mean": round(sum(trn.values()) / len(trn), 1) if trn else None,
                                "per_task": trn},
            "cap_stops": {"bank": cap(base), "s1": cap(new)},
            "unscaled_merged": {"base_right": lu["base_right"], "s1_right": lu["new_right"], "gained": lu["gained"],
                                "lost": lu["lost"], "net": lu["net"]},
            "gained_tasks": sorted({k.rsplit("_", 1)[0] for k in l1["gained_keys"]}),
            "lost_tasks": sorted({k.rsplit("_", 1)[0] for k in l1["lost_keys"]}),
            "kernel_of_output": {key: kernel_of.get(key.rsplit("_", 1)[0]) for key in sorted(keys)},
        },
        "fingerprints": {"bank": base.fingerprint(), "s1": new.fingerprint()},
        "sources": {"bank": base.sources, "s1": new.sources},
    })
    return res


def summary_lines(res: dict) -> list[str]:
    out = [f"S1 reading ({res.get('rule')})"]
    for k, x in res["kernels"].items():
        out.append(f"  {k}: {x['status']}; run {len(x['run'])}, not run {len(x['not_run'])}; problems {x.get('n_problems', 0)}")
    if "line1" not in res:
        out.append(f"  outcome: {res.get('outcome')}")
        return out
    l1, l2 = res["line1"], res["line2"]
    out.append(f"  on {res['n_outputs']} outputs of {res['n_tasks_run']} tasks run")
    out.append("| track | line | bank | S1 | gained | lost | net | p (one-sided) | reading |")
    out.append(f"| ARC-2 | 1 B0 alone | {l1['base_right']} | {l1['s1_right']} | {l1['gained']} | {l1['lost']} | {l1['net']:+d} | "
               f"{l1['sign_test_p_one_sided']} | {l1['verdict']} |")
    t = l2["tcp"]
    out.append(f"| ARC-2 | 2 TCP merged (delta {t['delta_s']:.0f} s, k {t['k']:.2f}) | {l2['base_right']} | {l2['s1_right']} | "
               f"{l2['gained']} | {l2['lost']} | {l2['net']:+d} | {l2['sign_test_p_one_sided']} | gate v2 {'PASS' if l2['gate_v2'] else 'FAIL'} |")
    r = res["recorded"]
    out.append(f"  outcome: {res['outcome']}")
    out.append(f"  oracle bank {r['b0_pool_oracle']['bank']}, S1 {r['b0_pool_oracle']['s1']}; time ratio {r['time']['ratio_sum']} "
               f"(median {r['time']['ratio_median']}); cap stops bank {len(r['cap_stops']['bank'])}, S1 {len(r['cap_stops']['s1'])}; "
               f"unscaled merged {r['unscaled_merged']['net']:+d}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--runs", default=str(RUNS), help="directory holding kaggle-A2-s1-k<i>-v1")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--floor-only", default="", metavar="K", help="write <archive>/s1_floor.json for one kernel; exit 1 unless valid")
    args = ap.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    archives = {k: Path(args.runs) / ARCHIVE.format(k=k) for k in plan["kernels"]}
    if args.floor_only:
        k = args.floor_only
        rep = check_kernel(k, plan["kernels"][k], archives[k] / "pool", bank_environment())
        if archives[k].is_dir():
            (archives[k] / "s1_floor.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps({key: rep[key] for key in ("kernel", "status", "n_problems") if key in rep}))
        return 0 if rep["status"] == "valid" else 1
    res = read_s1(plan, archives)
    gaps = {k: x["download_gaps"] for k, x in res["kernels"].items() if x.get("download_gaps")}
    if gaps:
        print(f"download incomplete, re-fetch these archives before reading (nothing written): {gaps}")
        return 2
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for line in summary_lines(res):
        print(line)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
