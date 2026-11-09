#!/usr/bin/env python3
"""Decode-many reading (experiments/arc2-bank/DM_RULE.md sections 4 to 6). CPU only.

    python scripts/arc2_dm_read.py                  # archives experiments/runs/kaggle-A2-dm-k<i>-v1 -> experiments/arc2-bank/dm_result.json
    python scripts/arc2_dm_read.py --proof-only     # the proof kernel's path checks (section 7), no decision

Per kernel of kaggle/arc2/bank/dm/DM_PLAN.json, the floor (section 6, items 1 to 4): pool/_run.json's run info
(variant dm-<k>, 64 sequences, dm: floor 0.1, 32 views, 32 rescores, caps 3,600 / 1,620 s, the self-test) and the
solver environment equal to the bank's; every task log's settings and completeness (calls, flags, beams' completing
calls, 32 scores and 8 call times per grid); the widest setting replayed under the run's own caps reproduces the
kernel's export (candidate grids and the submitted pair). A failed item makes the kernel's tasks "report only". A
listed task without a log or pool file, or with end_time_reached, is "not run" and leaves every setting. The proof
kernel is a shard of the main run only when its code-cell sha256 equals the main kernels'; otherwise report only.

Settings (section 3): the replay's 12 pools (scripts/arc2_dm_replay.py) x ranking {kgmon, probmul3} = 24; s0 =
(16, 0.2, 8, kgmon). Each setting's lines against s0 replayed from the same run: B0 alone (pass@2 of its two
attempts) and TCP merged (eval/arc2_bank.py tcp: s0's E13 as the reference; v7_direct on the setting's pool with the
27B samples cut, against s0's v7_direct with every sample). Whole-run items 5 (s0's emulated seconds within 0.90 to
1.10 of the bank's) and 6 (s0 coverage short on at most 2 tasks); either failing makes every TCP line report only.
Selection (section 4): halves by sha256("dm-half:" + task id), even first byte = A; choose on A the highest TCP net
(ties: fewer lost, smaller delta, grid order; no winner below net +1); confirm on B (net >= +1, lost <= 1); qualify on
all outputs with gate v2 (net >= +3, lost <= 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.arc2_bank import (BANK_PARTS, CHALLENGES, PLACEHOLDER, Q27_BANK, Rule, _top2, b0r_check,  # noqa: E402
                            bootstrap_difference, download_gaps, grids_equal, load_bank, merge_v7_pairs, replay, sign_test_p,
                            tcp, tcp_e13)
import arc2_dm_replay as R  # noqa: E402

SCHEMA = "arc2_dm_result.v1"
PLAN = ROOT / "kaggle" / "arc2" / "bank" / "dm" / "DM_PLAN.json"
RUNS = ROOT / "experiments" / "runs"
ARCHIVE = "kaggle-A2-dm-{k}-v1"
REPLAY_OUT = ROOT / "experiments" / "arc2-bank" / "dm"
OUT = ROOT / "experiments" / "arc2-bank" / "dm_result.json"
ENV_KEYS = ("torch", "transformers", "unsloth", "peft", "triton")
RANKINGS = ("kgmon", "probmul3")
S0 = (16, 0.2, 8, "kgmon")
TIME_BAND = (0.90, 1.10)
S0_SHORT_MAX = 2
HALF_SALT = "dm-half:"


def setting_id(v: int, f: float, r: int, rank: str) -> str:
    return f"{R.setting_name(v, f, r)}_{rank}"


GRID = [(v, f, r, rk) for (v, f, r) in R.SETTINGS for rk in RANKINGS]   # grid order: views, floor, rescoring, ranking


def half_of(task_id: str) -> str:
    return "A" if hashlib.sha256((HALF_SALT + task_id).encode()).digest()[0] % 2 == 0 else "B"


# ----------------------------------------------------------------------------
# rules: B0 alone and merged, per ranking

def _probmul3_pairs(task, bank, rng):
    return [_top2(o.ranked("rank_probmul3")) if o.has_pool else (o.attempt_1, o.attempt_2) for o in task.outputs]


def _kgmon_pairs(task, bank, rng):
    return [_top2(o.ranked("rank_kgmon")) if o.has_pool else (o.attempt_1, o.attempt_2) for o in task.outputs]


RULES = {
    ("kgmon", "alone"): Rule("dm_kgmon", _kgmon_pairs, "b0", "kgmon top two"),
    ("probmul3", "alone"): Rule("dm_probmul3", _probmul3_pairs, "b0", "probmul3 top two"),
    ("kgmon", "merged"): Rule("dm_kgmon_v7", lambda t, b, g: merge_v7_pairs(t, b, _kgmon_pairs(t, b, g)), "merge",
                              "v7's merge on kgmon's two"),
    ("probmul3", "merged"): Rule("dm_probmul3_v7", lambda t, b, g: merge_v7_pairs(t, b, _probmul3_pairs(t, b, g)),
                                 "merge", "v7's merge on probmul3's two"),
}


def cmp_on(base, new, keys: set[str] | None = None) -> dict:
    shared = sorted(k for k in set(base.per_output) & set(new.per_output) if keys is None or k in keys)
    g = [k for k in shared if new.per_output[k]["right"] and not base.per_output[k]["right"]]
    lo = [k for k in shared if base.per_output[k]["right"] and not new.per_output[k]["right"]]
    p = sign_test_p(len(g), len(lo))
    return {"n": len(shared), "base_right": sum(1 for k in shared if base.per_output[k]["right"]),
            "new_right": sum(1 for k in shared if new.per_output[k]["right"]), "gained": len(g), "lost": len(lo),
            "net": len(g) - len(lo), "p": None if p is None else round(p, 5), "gained_keys": g, "lost_keys": lo}


# ----------------------------------------------------------------------------
# the floor

def bank_environment(pools=BANK_PARTS) -> dict:
    run = json.loads((Path(pools[0]) / "_run.json").read_text(encoding="utf-8"))["run"]
    solver = run["environment"]["solver"]
    return {k: solver.get(k) for k in ENV_KEYS}


def check_log(log: dict, n_test: int | None) -> list[str]:
    """Items 2 and 3 on one task log."""
    bad = []
    t = log.get("task_id")
    want = {"floor": 0.1, "task_cap_s": 3600, "dfs_cap_s": 1620, "rescore_n": 32}
    got = {k: log.get(k) for k in want}
    if got != want:
        bad.append(f"{t}: log settings {got}")
    views = log.get("views") or {}
    a, b = views.get("a") or [], views.get("b") or []
    n = n_test or log.get("n_test") or 0
    if len(a) != 16 * n or len(b) != 16 * n or set(a) & set(b):
        bad.append(f"{t}: views {len(a)} + {len(b)} for {n} outputs, overlap {len(set(a) & set(b))}")
    for bt in log.get("batches") or []:
        us, lo = bt.get("call_us") or [], bt.get("call_lo02") or []
        if len(us) != len(lo) or any(x > y for x, y in zip(us, us[1:])) or (us and us[-1] / 1e6 > bt["dfs_s"] + 1e-6):
            bad.append(f"{t} batch {bt.get('i')}: call log inconsistent ({len(us)} starts, {len(lo)} flags)")
        for bm in bt.get("beams") or []:
            c = bm.get("call")
            if not isinstance(c, int) or not -1 <= c < len(us):
                bad.append(f"{t} batch {bt.get('i')}: a beam without its completing call ({c})")
            elif bm["score"] < R.LO_SCORE and c >= 0 and not lo[c]:
                bad.append(f"{t} batch {bt.get('i')}: a beam below -log 0.2 completed in a call not marked so")
    for g in log.get("grids") or []:
        if len(g.get("score_aug") or []) != 32 or len(g.get("call_s") or []) != 8:
            bad.append(f"{t}: grid with {len(g.get('score_aug') or [])} scores, {len(g.get('call_s') or [])} call times")
    return bad


def replay_matches_export(log: dict, pool_doc: dict) -> list[str]:
    """Item 4: the widest setting under the run's own caps reproduces the kernel's export."""
    em = R.emulate_task(log, 32, 0.1, 32, task_cap_s=1e12, dfs_cap_s=1e12, run_dfs_cap_s=1e12)
    doc = R.pool_doc(log, em, 32, int(pool_doc.get("n_test") or len(pool_doc["outputs"])), "check")
    bad = []
    for mine, theirs in zip(doc["outputs"], pool_doc["outputs"]):
        a = sorted(json.dumps(c["grid"]) for c in mine["candidates"])
        b = sorted(json.dumps(c["grid"]) for c in theirs["candidates"])
        sub = theirs.get("submitted") or {}
        pair_t = [sub.get("attempt_1"), sub.get("attempt_2")] if sub else (theirs.get("selected") or [])[:2]
        pair_m = [mine["submitted"]["attempt_1"], mine["submitted"]["attempt_2"]] if mine["candidates"] else []
        if a != b:
            bad.append(f"{log['task_id']} output {mine['test_index']}: replay candidates {len(a)} != export {len(b)}")
        elif theirs["candidates"] and not all(grids_equal(x, y) for x, y in zip(pair_m, pair_t)):
            bad.append(f"{log['task_id']} output {mine['test_index']}: replay pair differs from the export's")
    return bad


def check_kernel(k: str, spec: dict, archive: Path, bank_env: dict) -> dict:
    listed = list(spec["task_ids"])
    pool, logs = archive / "pool", archive / "dm"
    if not pool.is_dir():
        return {"kernel": k, "status": "missing", "run": [], "not_run": listed, "problems": [f"no archive at {archive}"],
                "exact_checks": []}
    problems, run, not_run, checks = [], [], [], []
    files = {p.stem: p for p in pool.glob("*.json") if not p.name.startswith("_")}
    extra = sorted(set(files) - set(listed))
    if extra:
        problems.append(f"pool files for unlisted tasks: {extra}")
    gaps = download_gaps(pool, logs)
    if gaps:
        problems.append(f"download incomplete, re-fetch the archive: {gaps}")
    for t in listed:
        lp = logs / f"{t}.json.gz"
        if t not in files or not lp.is_file():
            not_run.append(t)
            continue
        doc = json.loads(files[t].read_text(encoding="utf-8"))
        log = R.load_log(lp)
        if (doc.get("run") or {}).get("end_time_reached") or log.get("end_time_reached"):
            not_run.append(t)
            continue
        run.append(t)
        problems += check_log(log, int(doc.get("n_test") or len(doc.get("outputs") or [])))
        for o in doc.get("outputs") or []:
            if o.get("rank_check_ok") is False:
                problems.append(f"{t} output {o.get('test_index')}: rank_check_ok false")
            for c in o.get("candidates") or []:
                if b0r_check(c, n_aug=32):
                    problems.append(f"{t} output {o.get('test_index')}: candidate fields {b0r_check(c, n_aug=32)}")
        problems += replay_matches_export(log, doc)
        for bt in log.get("batches") or []:
            if bt.get("exact_check"):
                checks.append(dict(bt["exact_check"], task=t, rank=log.get("rank")))
    meta_path = pool / "_run.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("run") or {}
        want = {"variant": f"dm-{k}", "ttt_sequences": 64, "bank_seed": 0}
        got = {key: meta.get(key) for key in want}
        if got != want:
            problems.append(f"_run.json run {got} != {want}")
        d = meta.get("dm") or {}
        dwant = {"floor": 0.1, "views": 32, "rescore_n": 32, "task_cap_s": 3600, "dfs_cap_s": 1620, "selftest": True}
        if {key: d.get(key) for key in dwant} != dwant:
            problems.append(f"_run.json dm {d}")
        solver = ((meta.get("environment") or {}).get("solver") or {})
        env = {key: solver.get(key) for key in ENV_KEYS}
        if env != bank_env:
            problems.append(f"environment {env} != the bank's {bank_env}")
    else:
        problems.append("no pool/_run.json")
    return {"kernel": k, "status": "report_only" if problems else "valid", "run": run, "not_run": not_run,
            "problems": problems[:50], "n_problems": len(problems), "download_gaps": gaps, "exact_checks": checks,
            "exact_checks_all_equal": all(c.get("beams_equal") and c.get("calls_floor02") == c.get("calls_lo02_in_floor01")
                                          for c in checks) if checks else None}


def code_hash(archive_or_kernel_dir: Path) -> str | None:
    p = archive_or_kernel_dir / "PROVENANCE.json"
    return json.loads(p.read_text(encoding="utf-8")).get("code_cells_sha256") if p.is_file() else None


# ----------------------------------------------------------------------------
# the reading

def read_dm(plan: dict, archives: dict[str, Path], replay_dir: Path, *, bank_pools=BANK_PARTS, q27_dir=Q27_BANK,
            challenges=CHALLENGES, solutions=None, kernel_dirs: dict[str, Path] | None = None, n_boot: int = 4000,
            proof_only: bool = False) -> dict:
    bank_env = bank_environment(bank_pools)
    proof = plan.get("proof")
    which = [proof] if proof_only else list(plan["kernels"])
    kernels = {k: check_kernel(k, plan["kernels"][k], archives[k], bank_env) for k in which}
    # the proof as a shard (section 7): its code cells equal the main kernels'
    shard = None
    if not proof_only and proof in kernels:
        hashes = {k: code_hash((kernel_dirs or {}).get(k, archives[k])) for k in kernels}
        mains = {h for k, h in hashes.items() if k != proof}
        shard = bool(hashes.get(proof)) and mains == {hashes.get(proof)}
        if not shard and kernels[proof]["status"] == "valid":
            kernels[proof]["status"] = "report_only"
            kernels[proof]["problems"].append("proof code cells differ from the main kernels': report only")
    valid = [k for k in kernels if kernels[k]["status"] == "valid"]
    tasks = sorted(t for k in valid for t in kernels[k]["run"])
    res = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule": plan.get("rule"),
           "proof_only": proof_only, "proof_is_shard": shard, "kernels": kernels, "n_tasks_run": len(tasks),
           "report_only_tasks": sorted(t for k in kernels if kernels[k]["status"] == "report_only" for t in kernels[k]["run"]),
           "not_run_tasks": sorted(t for k in kernels for t in kernels[k]["not_run"])}
    if not tasks:
        res["outcome"] = "no task run in a valid kernel: nothing to read"
        return res
    summary = R.replay([archives[k] for k in valid], replay_dir, tasks=tasks,
                       run_tasks={archives[k]: kernels[k]["run"] for k in valid})
    (replay_dir / "dm_replay.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    banks = {name: load_bank([replay_dir / "pools" / name], q27_dir, challenges=challenges, solutions=solutions,
                             tasks=tasks) for name in summary["settings"]}
    ref_bank = load_bank(list(bank_pools), q27_dir, challenges=challenges, solutions=solutions, tasks=tasks)
    s0_name = R.setting_name(*S0[:3])
    s0 = banks[s0_name]
    s0_secs = {t: summary["settings"][s0_name]["tasks"][t]["elapsed_s"] for t in tasks}
    s0_e13 = tcp_e13(s0, s0_secs, tasks)
    base_alone = replay(s0, RULES[("kgmon", "alone")])
    base_merged = replay(s0, RULES[("kgmon", "merged")])
    halves = {"A": {f"{o.task_id}_{o.test_index}" for o in s0.outputs() if half_of(o.task_id) == "A"},
              "B": {f"{o.task_id}_{o.test_index}" for o in s0.outputs() if half_of(o.task_id) == "B"}}
    # whole-run items 5 and 6
    bank_secs = {t: ref_bank.tasks[t].elapsed_s for t in tasks}
    ratio = sum(s0_secs.values()) / sum(bank_secs.values())
    s0_short = summary["settings"][s0_name]["coverage_short"]
    item5 = TIME_BAND[0] <= ratio <= TIME_BAND[1]
    item6 = len(s0_short) <= S0_SHORT_MAX
    tcp_valid = item5 and item6 and not proof_only
    lines = {}
    for v, f, r, rk in GRID:
        sid = setting_id(v, f, r, rk)
        b = banks[R.setting_name(v, f, r)]
        secs = {t: summary["settings"][R.setting_name(v, f, r)]["tasks"][t]["elapsed_s"] for t in tasks}
        tc = tcp(b, s0_e13, secs, tasks)
        alone = replay(b, RULES[(rk, "alone")])
        merged = replay(tc["bank"], RULES[(rk, "merged")])
        lines[sid] = {
            "setting": {"views": v, "floor": f, "rescore_n": r, "ranking": rk},
            "alone": {h: cmp_on(base_alone, alone, halves[h] if h != "all" else None) for h in ("A", "B", "all")},
            "tcp": {h: cmp_on(base_merged, merged, halves[h] if h != "all" else None) for h in ("A", "B", "all")},
            "delta_s": tc["delta_s"], "k": tc["k"], "samples_kept": tc["samples_kept"],
            "b0_cut_fraction": tc["b0_cut_fraction"], "b0_cut_tasks": tc["b0_cut_tasks"],
            "time_ratio_to_s0": round(sum(secs.values()) / sum(s0_secs.values()), 4),
            "oracle": sum(1 for o in b.outputs() if b.truth(o) is not None
                          and any(grids_equal(c["grid"], b.truth(o)) for c in o.candidates)),
            "cap_stops": sorted(t for t in tasks if b.tasks[t].run.get("stop_reason") == "per_task_cap"),
            "coverage_short": summary["settings"][R.setting_name(v, f, r)]["coverage_short"],
        }
    # selection (section 4)
    s0_id = setting_id(*S0)
    cands = [setting_id(*g) for g in GRID if setting_id(*g) != s0_id]
    order = {sid: i for i, sid in enumerate(cands)}
    best = sorted(cands, key=lambda sid: (-lines[sid]["tcp"]["A"]["net"], lines[sid]["tcp"]["A"]["lost"],
                                          lines[sid]["delta_s"], order[sid]))
    winner = best[0] if best and lines[best[0]]["tcp"]["A"]["net"] >= 1 else None
    sel = {"winner": winner, "ranked_on_A": [(sid, lines[sid]["tcp"]["A"]["net"], lines[sid]["tcp"]["A"]["lost"])
                                             for sid in best[:8]]}
    if proof_only:
        outcome = "proof: path checks only (DM_RULE.md section 7); its lines feed no decision"
    elif winner is None:
        outcome = "no winner on half A: the bank's decoding stays"
    else:
        lb, la = lines[winner]["tcp"]["B"], lines[winner]["tcp"]["all"]
        confirmed = lb["net"] >= 1 and lb["lost"] <= 1
        qualified = confirmed and la["net"] >= 3 and la["lost"] <= 1 and tcp_valid
        sel.update({"confirmed_on_B": confirmed, "gate_v2_all": la["net"] >= 3 and la["lost"] <= 1,
                    "qualified": qualified})
        if not tcp_valid:
            outcome = "TCP lines report only (validity item 5 or 6): nothing qualifies"
        elif qualified:
            outcome = (f"{winner} qualifies: next an entry draft (v7-vr2 with the setting, B0's phase end "
                       f"moved by its delta) and a commit run for the Nov 2 slot, under its own rule")
        elif confirmed:
            outcome = f"{winner} confirmed on B but not qualified (gate v2 on all): recorded; no entry"
        else:
            outcome = f"{winner} not confirmed on half B: the bank's decoding stays"
    rng_boot = {}
    if winner:
        w = lines[winner]["setting"]
        wb = banks[R.setting_name(w["views"], w["floor"], w["rescore_n"])]
        wsecs = {t: summary["settings"][R.setting_name(w["views"], w["floor"], w["rescore_n"])]["tasks"][t]["elapsed_s"]
                 for t in tasks}
        wt = tcp(wb, s0_e13, wsecs, tasks)
        rng_boot = {"alone": bootstrap_difference(base_alone, replay(wb, RULES[(w["ranking"], "alone")]), n_boot=n_boot),
                    "tcp": bootstrap_difference(base_merged, replay(wt["bank"], RULES[(w["ranking"], "merged")]),
                                                n_boot=n_boot),
                    "against_bank_v7_direct": cmp_on(replay(ref_bank, "v7_direct"),
                                                      replay(wt["bank"], RULES[(w["ranking"], "merged")]))}
    res.update({
        "tasks": tasks, "halves": {h: sorted({k.rsplit("_", 1)[0] for k in halves[h]}) for h in halves},
        "n_outputs": len(base_alone.per_output),
        "validity": {"item5_s0_time_ratio": round(ratio, 4), "item5_pass": item5, "item6_s0_coverage_short": s0_short,
                     "item6_pass": item6, "tcp_valid": tcp_valid},
        "s0": {"e13": s0_e13, "alone_against_bank_b0": cmp_on(replay(ref_bank, "b0"), base_alone),
               "merged_against_bank_v7_direct": cmp_on(replay(ref_bank, "v7_direct"), base_merged),
               "time_per_task_against_bank": {t: [round(s0_secs[t], 1), bank_secs[t]] for t in tasks}},
        "lines": lines, "selection": sel, "outcome": outcome, "winner_extra": rng_boot,
        "replay": {"dir": str(replay_dir), "settings_seconds": {n: x["seconds"] for n, x in summary["settings"].items()}},
    })
    return res


def summary_lines(res: dict) -> list[str]:
    out = [f"decode-many reading ({res.get('rule')}){' — proof only' if res.get('proof_only') else ''}"]
    for k, x in res["kernels"].items():
        out.append(f"  {k}: {x['status']}; run {len(x['run'])}, not run {len(x['not_run'])}; problems {x.get('n_problems', 0)}; "
                   f"exactness checks {len(x.get('exact_checks') or [])} all equal {x.get('exact_checks_all_equal')}")
    if "lines" not in res:
        out.append(f"  outcome: {res.get('outcome')}")
        return out
    v = res["validity"]
    out.append(f"  {res['n_tasks_run']} tasks, {res['n_outputs']} outputs; s0 time / bank {v['item5_s0_time_ratio']} "
               f"(item 5 {'PASS' if v['item5_pass'] else 'FAIL'}); s0 coverage short {len(v['item6_s0_coverage_short'])} "
               f"(item 6 {'PASS' if v['item6_pass'] else 'FAIL'})")
    s = res["s0"]
    out.append(f"  s0 against the bank: B0 {s['alone_against_bank_b0']['new_right']} vs {s['alone_against_bank_b0']['base_right']}, "
               f"merged {s['merged_against_bank_v7_direct']['new_right']} vs {s['merged_against_bank_v7_direct']['base_right']}")
    out.append("| track | setting | B0 alone net (A/B/all) | TCP net (A/B/all) | lost all | delta s | k |")
    for sid, x in res["lines"].items():
        a, t = x["alone"], x["tcp"]
        out.append(f"| ARC-2 | {sid} | {a['A']['net']:+d}/{a['B']['net']:+d}/{a['all']['net']:+d} | "
                   f"{t['A']['net']:+d}/{t['B']['net']:+d}/{t['all']['net']:+d} | {t['all']['lost']} | {x['delta_s']:.0f} | {x['k']:.2f} |")
    out.append(f"  selection: {res['selection']}")
    out.append(f"  outcome: {res['outcome']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--replay-out", default=str(REPLAY_OUT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--proof-only", action="store_true", help="the proof kernel's path checks; no decision")
    ap.add_argument("--floor-only", default="", metavar="K",
                    help="items 2 to 4 on one kernel's archive (<archive>/dm_floor.json); exit 1 unless valid")
    args = ap.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    archives = {k: Path(args.runs) / ARCHIVE.format(k=k) for k in plan["kernels"]}
    if args.floor_only:
        k = args.floor_only
        rep = check_kernel(k, plan["kernels"][k], archives[k], bank_environment())
        if archives[k].is_dir():
            (archives[k] / "dm_floor.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps({key: rep.get(key) for key in ("kernel", "status", "n_problems", "download_gaps")}))
        return 0 if rep["status"] == "valid" else 1
    kernel_dirs = {k: PLAN.parent / k for k in plan["kernels"]}
    out = Path(args.out) if not args.proof_only else Path(args.out).with_name("dm_proof_result.json")
    replay_dir = Path(args.replay_out) / ("proof" if args.proof_only else "main")
    res = read_dm(plan, archives, replay_dir, kernel_dirs=kernel_dirs, proof_only=args.proof_only)
    gaps = {k: x["download_gaps"] for k, x in res["kernels"].items() if x.get("download_gaps")}
    if gaps:
        print(f"download incomplete, re-fetch these archives before reading (nothing written): {gaps}")
        return 2
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for line in summary_lines(res):
        print(line)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
