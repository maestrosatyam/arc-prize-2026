#!/usr/bin/env python3
"""Read experiment C2 (ARC-2: the sample-count ceiling on the output bank) by its frozen rule. CPU only.

experiments/arc2-bank/C2_RULE.md (frozen at 24ef327), section 3, on the 145 unsealed outputs with Kaggle's solutions
copy as truth. Two pools:

  bank pool = B0's pool (both parts) + the 27B bank's samples 0 and 1 on the band (30 tasks);
  C2 pool   = the bank pool + the C2 kernels' samples (2 and 3 on the band, 0 and 1 on the 71 tasks beyond),
              merged by task and sample index (eval.arc2_bank.load_bank with several 27B directories).

  line 1  headline: the oracle (any correct grid in B0's pool or the 27B samples) on the C2 pool against the bank
          pool; gate PASS when oracle at C2 >= 60 of 145
  line 2  which lever: (a) the band, four samples against two (oracle and the 27B-only union on the band outputs);
          (b) beyond the band, the oracle gain from the first two samples against B0's pool coverage; guidance
          flags band_gain_below_2 and beyond_gain_at_least_4 (recorded, not a gate)
  line 3  pick rate: v7_direct and q27_first on both pools, outputs right and the pick rate = outputs where the
          attempt the rule fills from the 27B is right / outputs where some 27B sample is right and B0's attempt_1
          is wrong; the 27B-only union (any sample right) and the q27_any bound on both pools

With the task-cluster bootstrap spread (tasks resampled with replacement) of the oracle difference C2 - bank and of
oracle at C2, and the per-output table (B0 pool right / bank oracle right / C2 oracle right), written to
experiments/arc2-bank/c2_result.json. Nothing under q27d_sealed/ is read: the loader refuses a sealed directory.

Usage: arc2_c2_read.py [--c2 DIR ...] [--out experiments/arc2-bank/c2_result.json] [--n-boot 2000] [--seed 0]
       (tests: [--pools DIR ...] [--q27 DIR] [--challenges FILE] [--solutions FILE])

A missing C2 directory is skipped: the result is marked partial and lists the tasks without C2 samples (the rule's
"report only" reading on the valid tasks).

--report-only TASK ... drops those tasks from lines 1 to 3 and the bootstrap (the validity floor's "report only"
tasks); the per-output table keeps every output with a report_only column, and --floor-scope TEXT records the
decision that set the scope (2026-10-29 14:18 IST: section 4, the affected tasks only).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (BANK_PARTS, CHALLENGES, Q27_BANK, RUNS, Bank, Result, SealedAccessError,  # noqa: E402
                            bootstrap_board, bootstrap_difference, load_bank, mean_sd, percentile, pool_hit_keys,
                            q27_hit_keys, replay)

SCHEMA = "arc2_c2_result.v1"
RULE_REF = "experiments/arc2-bank/C2_RULE.md (frozen 24ef327)"
C2_DIRS = (RUNS / "kaggle-A2-q27-c2a-v1" / "q27d", RUNS / "kaggle-A2-q27-c2b-v1" / "q27d")
DEFAULT_OUT = ROOT / "experiments" / "arc2-bank" / "c2_result.json"
GATE_MIN = 60              # section 3, line 1: oracle at C2 >= 60 of 145 (+5 over the bank's 55)
BAND_GAIN_MIN = 2          # line 2 guidance (a): the band's gain below +2
BEYOND_GAIN_MIN = 4        # line 2 guidance (b): the gain beyond the band at or above +4
LINE_RULES = ("oracle", "q27_any", "v7_direct", "q27_first", "b0")
PICK_SLOT = {"v7_direct": "c2", "q27_first": "c1"}    # the attempt each rule fills from the 27B samples
PER_OUTPUT_COLUMNS = ["task_id", "test_index", "b0_pool_right", "bank_oracle_right", "c2_oracle_right",
                      "n_q27_samples_c2", "in_band", "report_only"]
Q27_SCHEMA = "arc2_q27_direct.v1"


def tasks_with_records(d: Path) -> set[str]:
    """Task ids with a 27B sample record in a q27d/ directory (per-task records with sample records, or lines of
    samples.jsonl), grids or not: the tasks the kernel ran."""
    out: set[str] = set()
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if rec.get("schema") == Q27_SCHEMA and (rec.get("n_sample_records") or len(rec.get("samples") or [])):
            out.add(rec.get("task_id", p.stem))
    jl = d / "samples.jsonl"
    if jl.is_file():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if s.get("task_id"):
                out.add(s["task_id"])
    return out


def per_task_right(res: Result) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in res.per_output.values():
        out[v["task_id"]] = out.get(v["task_id"], 0) + int(bool(v["right"]))
    return out


def task_bootstrap(per_task: dict[str, float], n_boot: int, seed: int) -> dict:
    """Tasks resampled with replacement (as many as there are); the spread of the summed value, in outputs."""
    rng = random.Random(seed)
    tasks = sorted(per_task)
    vals = [per_task[t] for t in tasks]
    n = len(vals)
    if not n:
        return {"n_tasks": 0}
    observed = sum(vals)
    if n_boot <= 0:
        return {"n_tasks": n, "n_boot": 0, "observed": observed}
    sums = [sum(vals[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)]
    m, sd = mean_sd(sums)
    return {"n_tasks": n, "n_boot": n_boot, "observed": observed, "mean": round(m, 3), "sd": round(sd, 3),
            "ci95": [round(percentile(sums, 0.025), 3), round(percentile(sums, 0.975), 3)]}


def read_pool(bank: Bank) -> dict:
    """The line rules replayed on one pool, B0's pool coverage, the 27B-only union and the pick-rate denominator
    (outputs where some 27B sample is right and B0's attempt_1 is wrong)."""
    res = {r: replay(bank, r) for r in LINE_RULES}
    b0 = res["b0"]
    q_hits = q27_hit_keys(bank)
    denom = {k for k in q_hits if not b0.per_output[k]["c1"]}
    pick = {}
    for r, slot in PICK_SLOT.items():
        picked = sorted(k for k in denom if res[r].per_output[k][slot])
        pick[r] = {"n_right": res[r].n_right, "picked_attempt": "attempt_2" if slot == "c2" else "attempt_1",
                   "picked_right": len(picked), "denominator": len(denom),
                   "pick_rate": round(len(picked) / len(denom), 4) if denom else None, "picked_keys": picked}
    return {"results": res, "pool_hits": pool_hit_keys(bank), "q27_hits": q_hits, "denominator": denom,
            "n_right": {r: res[r].n_right for r in LINE_RULES}, "pick": pick}


def without_tasks(bank: Bank, drop: set[str]) -> Bank:
    """The bank without the given tasks (their B0 pool and 27B samples); task sizes are kept, so the band is the
    same."""
    return dataclasses.replace(bank, tasks={t: v for t, v in bank.tasks.items() if t not in drop},
                               q27={t: v for t, v in bank.q27.items() if t not in drop})


def _n(keys: set[str], within: set[str]) -> int:
    return len(keys & within)


def read_c2(pool_dirs=None, q27_bank: Path | str = Q27_BANK, c2_dirs=C2_DIRS, *, challenges=CHALLENGES,
            solutions=None, n_boot: int = 2000, seed: int = 0, gate_min: int = GATE_MIN,
            report_only=(), floor_scope: str = "") -> dict:
    """The lines of C2_RULE.md section 3 on the bank pool and the C2 pool (see the module docstring)."""
    pools = list(pool_dirs) if pool_dirs is not None else list(BANK_PARTS)
    c2_dirs = [Path(d) for d in c2_dirs]
    present = [d for d in c2_dirs if d.is_dir()]
    missing_dirs = [str(d) for d in c2_dirs if not d.is_dir()]
    bank = load_bank(pools, [q27_bank], challenges=challenges, solutions=solutions)
    c2 = load_bank(pools, [q27_bank, *present], challenges=challenges, solutions=solutions)
    if set(c2.tasks) != set(bank.tasks) or c2.n_outputs != bank.n_outputs:
        raise RuntimeError("the C2 pool and the bank pool differ in tasks or outputs")
    ro = sorted(set(report_only))
    unknown = [t for t in ro if t not in bank.tasks]
    if unknown:
        raise ValueError(f"report-only tasks not in the bank: {unknown}")
    full_bank, full_c2 = bank, c2
    if ro:
        bank, c2 = without_tasks(bank, set(ro)), without_tasks(c2, set(ro))
    band = bank.band()
    band_set = set(band)
    outs = list(bank.outputs())
    all_keys = {o.key for o in outs}
    band_keys = {o.key for o in outs if o.task_id in band_set}
    beyond_keys = all_keys - band_keys
    ran: set[str] = set()
    for d in present:
        ran |= tasks_with_records(d)
    missing_tasks = sorted(t for t in bank.task_ids if t not in ran)
    partial = bool(missing_dirs) or bool(missing_tasks)

    rb, rc = read_pool(bank), read_pool(c2)
    rbf, rcf = (read_pool(full_bank), read_pool(full_c2)) if ro else (rb, rc)
    ob, oc = rb["results"]["oracle"], rc["results"]["oracle"]
    ob_keys, oc_keys = ob.right_keys(), oc.right_keys()
    gain = oc.n_right - ob.n_right
    headline = {"n_outputs": bank.n_outputs, "oracle_bank": ob.n_right, "oracle_c2": oc.n_right, "gain": gain,
                "gate_min": gate_min, "gate": "PASS" if oc.n_right >= gate_min else "FAIL",
                "gate_rule": f"oracle at C2 >= {gate_min} of 145 (C2_RULE.md section 3, line 1)"
                             + (f"; read on the {bank.n_outputs} outputs of {len(bank.tasks)} tasks, {len(ro)} task(s) "
                                f"report only (floor scope)" if ro else ""),
                "partial": partial, "report_only_tasks": ro}
    band_gain = _n(oc_keys, band_keys) - _n(ob_keys, band_keys)
    beyond_gain = _n(oc_keys, beyond_keys) - _n(ob_keys, beyond_keys)
    lever = {
        "band": {"task_ids": band, "n_tasks": len(band), "n_outputs": len(band_keys), "samples_bank": "0 and 1",
                 "samples_c2": "0 to 3", "oracle_bank": _n(ob_keys, band_keys), "oracle_c2": _n(oc_keys, band_keys),
                 "gain": band_gain, "q27_union_bank": _n(rb["q27_hits"], band_keys),
                 "q27_union_c2": _n(rc["q27_hits"], band_keys),
                 "q27_union_gain": _n(rc["q27_hits"], band_keys) - _n(rb["q27_hits"], band_keys),
                 "b0_pool_right": _n(rb["pool_hits"], band_keys)},
        "beyond": {"n_tasks": len(bank.tasks) - len(band), "n_outputs": len(beyond_keys), "samples_c2": "0 and 1",
                   "b0_pool_right": _n(rb["pool_hits"], beyond_keys), "oracle_bank": _n(ob_keys, beyond_keys),
                   "oracle_c2": _n(oc_keys, beyond_keys), "gain": beyond_gain,
                   "q27_union_bank": _n(rb["q27_hits"], beyond_keys), "q27_union_c2": _n(rc["q27_hits"], beyond_keys)},
        "guidance": {"band_gain_below_2": band_gain < BAND_GAIN_MIN, "beyond_gain_at_least_4": beyond_gain >= BEYOND_GAIN_MIN,
                     "thresholds": {"band_gain_min": BAND_GAIN_MIN, "beyond_gain_min": BEYOND_GAIN_MIN},
                     "note": "recorded with the result, not a gate (C2_RULE.md section 3, line 2)"}}
    pick = {"definition": "pick rate = outputs where the attempt the rule fills from the 27B (v7_direct: attempt_2; "
                          "q27_first: attempt_1) is right / outputs where some 27B sample is right and B0's attempt_1 "
                          "is wrong; q27_union = outputs where some 27B sample is right (the 27B-only oracle); q27_any = "
                          "the registered bound (any correct 27B sample first, else B0's attempts)"}
    for name, r in (("bank", rb), ("c2", rc)):
        pick[name] = {"denominator": len(r["denominator"]), "denominator_keys": sorted(r["denominator"]),
                      "v7_direct": r["pick"]["v7_direct"], "q27_first": r["pick"]["q27_first"],
                      "q27_union": len(r["q27_hits"]), "q27_any": r["n_right"]["q27_any"], "b0": r["n_right"]["b0"]}
    pt_b, pt_c = per_task_right(ob), per_task_right(oc)
    diff = {t: pt_c[t] - pt_b.get(t, 0) for t in pt_c}
    boot = {"n_boot": n_boot, "seed": seed, "unit": "outputs; tasks resampled with replacement, as many as the bank's",
            "oracle_c2_minus_bank": task_bootstrap(diff, n_boot, seed), "oracle_c2": task_bootstrap(pt_c, n_boot, seed),
            "board_points_120_tasks": ({"difference": bootstrap_difference(ob, oc, 120, n_boot, seed),
                                        "oracle_c2": bootstrap_board(oc, 120, n_boot, seed)} if n_boot > 0
                                       else {"n_boot": 0})}
    obf_keys, ocf_keys = rbf["results"]["oracle"].right_keys(), rcf["results"]["oracle"].right_keys()
    rows = [[o.task_id, o.test_index, o.key in rbf["pool_hits"], o.key in obf_keys, o.key in ocf_keys,
             len(full_c2.q27.get(o.task_id, {})), o.task_id in band_set, o.task_id in ro] for o in full_bank.outputs()]
    scope = None
    if ro or floor_scope:
        ro_keys = {o.key for o in full_bank.outputs() if o.task_id in ro}
        scope = {"decision": floor_scope, "report_only_tasks": ro, "n_outputs_report_only": len(ro_keys),
                 "n_tasks_read": len(bank.tasks), "n_outputs_read": bank.n_outputs,
                 "report_only_right": {"b0_pool": _n(rbf["pool_hits"], ro_keys), "oracle_bank": _n(obf_keys, ro_keys),
                                       "oracle_c2": _n(ocf_keys, ro_keys)},
                 "all_tasks": {"n_outputs": full_bank.n_outputs, "oracle_bank": rbf["results"]["oracle"].n_right,
                               "oracle_c2": rcf["results"]["oracle"].n_right,
                               "fingerprint_bank": full_bank.fingerprint(), "fingerprint_c2": full_c2.fingerprint()}}
    return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule": RULE_REF,
            "pools": {"bank": {"dirs": bank.sources, "fingerprint": bank.fingerprint(), "n_tasks": len(bank.tasks),
                               "n_outputs": bank.n_outputs, "n_q27_tasks": len(bank.q27),
                               "n_q27_samples": sum(len(v) for v in bank.q27.values())},
                      "c2": {"dirs": c2.sources, "c2_dirs": [str(d) for d in present], "missing_dirs": missing_dirs,
                             "fingerprint": c2.fingerprint(), "n_tasks": len(c2.tasks), "n_outputs": c2.n_outputs,
                             "n_q27_tasks": len(c2.q27), "n_q27_samples": sum(len(v) for v in c2.q27.values()),
                             "tasks_with_records": len(ran)}},
            "rules": {"bank": rb["n_right"], "c2": rc["n_right"]},
            "headline": headline, "which_lever": lever, "pick_rate": pick, "bootstrap": boot,
            "per_output_columns": PER_OUTPUT_COLUMNS, "per_output": rows,
            "partial": partial, "missing_tasks": missing_tasks, "missing_dirs": missing_dirs, "floor_scope": scope}


def _rate(p: dict) -> str:
    r = p["pick_rate"]
    return f"{p['picked_right']}/{p['denominator']}" + (f" = {r:.3f}" if r is not None else "")


def summary_lines(res: dict) -> list[str]:
    pb, pc = res["pools"]["bank"], res["pools"]["c2"]
    h, lv, pk, bt = res["headline"], res["which_lever"], res["pick_rate"], res["bootstrap"]
    part = f"partial: {'true' if res['partial'] else 'false'}"
    if res["partial"]:
        part += f" ({len(res['missing_tasks'])} tasks without C2 samples"
        if res["missing_dirs"]:
            part += f"; missing dirs: {', '.join(res['missing_dirs'])}"
        part += ")"
    lines = [f"C2 read ({res['rule']}): {pb['n_tasks']} tasks, {pb['n_outputs']} outputs; bank pool {pb['fingerprint']} "
             f"({pb['n_q27_samples']} 27B samples on {pb['n_q27_tasks']} tasks); C2 pool {pc['fingerprint']} "
             f"({pc['n_q27_samples']} 27B samples on {pc['n_q27_tasks']} tasks, {len(pc['c2_dirs'])} C2 dir(s)); {part}",
             *([f"floor scope: {len(res['floor_scope']['report_only_tasks'])} task(s) report only "
                f"({' '.join(res['floor_scope']['report_only_tasks'])}; {res['floor_scope']['n_outputs_report_only']} "
                f"outputs, right: {res['floor_scope']['report_only_right']}); all tasks: oracle bank "
                f"{res['floor_scope']['all_tasks']['oracle_bank']}, C2 {res['floor_scope']['all_tasks']['oracle_c2']} of "
                f"{res['floor_scope']['all_tasks']['n_outputs']}; decision: {res['floor_scope']['decision']}"]
              if res.get("floor_scope") else []),
             f"line 1 headline: oracle bank {h['oracle_bank']}, C2 {h['oracle_c2']} ({h['gain']:+d}) of {h['n_outputs']}; "
             f"gate oracle_c2 >= {h['gate_min']}: {h['gate']}"]
    b, y, g = lv["band"], lv["beyond"], lv["guidance"]
    lines.append(f"line 2a band ({b['n_tasks']} tasks, {b['n_outputs']} outputs): oracle {b['oracle_bank']} -> "
                 f"{b['oracle_c2']} ({b['gain']:+d}); 27B union {b['q27_union_bank']} -> {b['q27_union_c2']} "
                 f"({b['q27_union_gain']:+d}); band_gain_below_2: {str(g['band_gain_below_2']).lower()}")
    lines.append(f"line 2b beyond ({y['n_tasks']} tasks, {y['n_outputs']} outputs): oracle {y['oracle_bank']} (B0 pool "
                 f"{y['b0_pool_right']}) -> {y['oracle_c2']} ({y['gain']:+d}); 27B union {y['q27_union_bank']} -> "
                 f"{y['q27_union_c2']}; beyond_gain_at_least_4: {str(g['beyond_gain_at_least_4']).lower()}")
    for r in ("v7_direct", "q27_first"):
        a, c = pk["bank"][r], pk["c2"][r]
        lines.append(f"line 3 {r} ({a['picked_attempt']}): bank {a['n_right']} right, pick {_rate(a)}; C2 {c['n_right']} "
                     f"right, pick {_rate(c)}")
    lines.append(f"line 3 27B-only union: bank {pk['bank']['q27_union']}, C2 {pk['c2']['q27_union']}; q27_any bound: bank "
                 f"{pk['bank']['q27_any']}, C2 {pk['c2']['q27_any']}; b0 {pk['bank']['b0']}")
    d, o = bt["oracle_c2_minus_bank"], bt["oracle_c2"]
    bd = bt["board_points_120_tasks"].get("difference") or {}
    lines.append(f"bootstrap ({bt['n_boot']} task resamples, seed {bt['seed']}): oracle C2 - bank {d.get('observed')} "
                 f"(sd {d.get('sd')}, 95% {d.get('ci95')}); oracle C2 {o.get('observed')} (sd {o.get('sd')}, 95% "
                 f"{o.get('ci95')}); board points at 120 tasks: difference {bd.get('mean_points')} sd {bd.get('sd_points')}")
    if res["missing_tasks"]:
        lines.append(f"tasks without C2 samples ({len(res['missing_tasks'])}): {' '.join(res['missing_tasks'])}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c2", nargs="*", default=None, metavar="DIR",
                    help="the C2 kernels' q27d directories (default: the c2a and c2b archives)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="write the result here ('' to skip)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pools", nargs="*", default=None, metavar="DIR", help="B0 pool directories (default: both parts)")
    ap.add_argument("--q27", default=str(Q27_BANK), help="the 27B bank's q27d (samples 0 and 1 on the band)")
    ap.add_argument("--challenges", default=str(CHALLENGES))
    ap.add_argument("--solutions", default="", help="extra solutions file (default: the Kaggle copies beside the pools)")
    ap.add_argument("--report-only", nargs="*", default=[], metavar="TASK",
                    help="tasks the validity floor makes report only (out of lines 1 to 3 and the bootstrap)")
    ap.add_argument("--floor-scope", default="", help="the decision that set the report-only scope (recorded)")
    args = ap.parse_args(argv)
    try:
        res = read_c2(args.pools, args.q27, args.c2 if args.c2 is not None else C2_DIRS,
                      challenges=args.challenges or None, solutions=args.solutions or None, n_boot=args.n_boot,
                      seed=args.seed, report_only=args.report_only, floor_scope=args.floor_scope)
    except SealedAccessError as exc:
        print(f"refused: {exc}")
        return 2
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    print("\n".join(summary_lines(res)))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
