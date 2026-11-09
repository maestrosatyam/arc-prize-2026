#!/usr/bin/env python3
"""TS reading: temperature sampling as a second B0 search (experiments/arc2-bank/TS_RULE.md sections 3 to 7). CPU only.

    python scripts/arc2_ts_read.py                  # archives experiments/runs/kaggle-A2-ts-k<i>-v1 -> experiments/arc2-bank/ts_result.json
    python scripts/arc2_ts_read.py --floor-only K   # items 2 to 4 and 6 on one kernel's archive (<archive>/ts_floor.json)

Per kernel of kaggle/arc2/bank/ts/TS_PLAN.json, the floor (section 7):
  item 2  pool/_run.json's run info (variant ts-<k>, 64 sequences, floor 0.2, seed 0; ts: temperature 1.0, 2 rounds,
          abandon 0.001, caps 540 / 1,200 s, the self-test) and the solver environment equal to the bank's; every task
          log's settings
  item 3  logs complete: every round has one row per view of its batch; every T grid has 8 scores, and 2 call times
          when T rescored it
  item 4  G+T2 and T2 rebuilt on CPU from G's pool and the T log reproduce pool_gt and pool_t (candidate grids and the
          two selected) on every task; G's pool passes B0R's floor checks
A failed item makes the kernel's tasks "report only". "Not run" (S1's definition): a listed task without G's pool file,
or with end_time_reached; it leaves every arm. A task whose T failed (an error, no T pool, or T stopped by the global
end time) leaves every T arm; item 6: more than 2 such tasks make every T line report only.

Arms, per task (section 3): G (the kernel's pool); G+Tn and Tn for n = 1, 2 rounds, rebuilt from the logs (n = 2 is
the run as exported). Seconds: G's elapsed; Tn's = T's logged seconds less the rounds beyond n and the rescoring of
grids only those rounds found; G+Tn's = G's plus Tn's. Pools are written in the bank's format to
experiments/arc2-bank/ts/pools/<arm>/ for eval/arc2_bank.py.

Lines on the outputs of the tasks run, in-run and paired against G (kgmon both): TCP merged (G's E13 as reference,
E13's clip 2,400 s for G+T) on halves A and B (DM_RULE.md's halves) and all; selection between G+T1 and G+T2 on A
(net >= +1; ties: fewer lost, smaller delta, T1), confirmation on B (net >= +1, lost <= 1), qualification on all
(gate v2: net >= +3, lost <= 1; item 5, G's seconds within 0.90 to 1.10 of the bank's, else TCP is report only). The
sampling line: G+T2 against G, B0 alone, exact McNemar (one-sided); gain net >= ceil(5N/145) and p <= 0.05, no gain
net <= floor(N/145). Recorded: G against the bank's B0, the other arms, probmul3, oracles, T's sample counts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.arc2_bank import (BANK_PARTS, CHALLENGES, Q27_BANK, b0r_check, binom_tail, bootstrap_difference,  # noqa: E402
                            download_gaps, grids_equal, load_bank, replay, tcp, tcp_e13)
import arc2_dm_replay as R  # noqa: E402
from arc2_dm_read import ENV_KEYS, RULES, bank_environment, cmp_on, half_of  # noqa: E402

SCHEMA = "arc2_ts_result.v1"
PLAN = ROOT / "kaggle" / "arc2" / "bank" / "ts" / "TS_PLAN.json"
RUNS = ROOT / "experiments" / "runs"
ARCHIVE = "kaggle-A2-ts-{k}-v1"
POOLS_OUT = ROOT / "experiments" / "arc2-bank" / "ts" / "pools"
OUT = ROOT / "experiments" / "arc2-bank" / "ts_result.json"
TIME_BAND = (0.90, 1.10)
T_FAIL_MAX = 2
GT_CAP_S = 2400.0
SETTINGS = (1, 2)
TS_WANT = {"temperature": 1.0, "rounds": 2, "abandon_p": 0.001, "batch_cap_s": 540, "task_cap_s": 1200, "selftest": True}
LOG_WANT = {"temperature": 1.0, "rounds": 2, "batch_cap_s": 540, "task_cap_s": 1200}


def load_log(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _gkey(grid) -> tuple:
    return tuple(tuple(int(x) for x in row) for row in grid)


# ----------------------------------------------------------------------------
# the arms, rebuilt from G's pool and the T log

def g_samples(g_doc: dict) -> dict[int, list[dict]]:
    """G's beams per output from its pool: [{key, view, beam_score, score_aug, solution, order, found_s}]."""
    out = {}
    for o in g_doc.get("outputs") or []:
        rows = []
        for c in o.get("candidates") or []:
            for b in c.get("beams") or []:
                rows.append({"key": b["key"], "view": b["key"].rsplit(".out", 1)[0], "beam_score": float(b["beam_score"]),
                             "score_aug": list(c["score_aug"]), "solution": c["grid"],
                             "order": b.get("order", 0), "found_s": b.get("elapsed_s", 0.0)})
        out[int(o["test_index"])] = sorted(rows, key=lambda r: r["order"])
    return out


def t_samples(log: dict, n_rounds: int) -> dict[str, list[dict]]:
    """T's unique samples per view from rounds below n_rounds, in the kernel's order (batches, then rounds, then rows):
    [{view, beam_score, score_aug, solution, grid, round, batch}]."""
    grids = log.get("grids") or []
    per_view: dict[str, list[dict]] = {}
    for b in log.get("batches") or []:
        for rd in b.get("rounds") or []:
            if int(rd["r"]) >= n_rounds:
                continue
            for row in rd.get("rows") or []:
                if row.get("status") != "ended" or row.get("grid") is None:
                    continue
                g = grids[int(row["grid"])]
                lst = per_view.setdefault(row["view"], [])
                if any(_gkey(x["solution"]) == _gkey(g["grid"]) for x in lst):
                    continue
                lst.append({"view": row["view"], "beam_score": float(row["score"]), "score_aug": list(g["score_aug"]),
                            "solution": g["grid"], "grid": int(row["grid"]), "round": int(rd["r"]), "batch": int(b["i"])})
    return per_view


def t_seconds(log: dict, n_rounds: int) -> dict:
    """Tn's seconds: T's logged seconds less the rounds at or beyond n_rounds and the rescoring of the grids T rescored
    that only those rounds found. "short": T's own caps stopped T2 where Tn would have gone on."""
    total = float(log.get("t_s") or 0.0)
    if n_rounds >= int(log.get("rounds") or 2):
        return {"t_s": round(total, 3), "short": []}
    kept = {int(row["grid"]) for b in log.get("batches") or [] for rd in b.get("rounds") or [] if int(rd["r"]) < n_rounds
            for row in rd.get("rows") or [] if row.get("status") == "ended" and row.get("grid") is not None}
    drop = sum(float(rd["s"]) for b in log.get("batches") or [] for rd in b.get("rounds") or [] if int(rd["r"]) >= n_rounds)
    for i, g in enumerate(log.get("grids") or []):
        if g.get("rescored_by_t") and i not in kept:
            drop += float(g.get("setup_s") or 0.0) + sum(float(x) for x in g.get("call_s") or [])
    # T's task cap stopped T2 before its last batches, where the cheaper Tn would have gone on
    short = ["T2 stopped by T's task cap before every batch"] if log.get("stop") == "per_task_cap" else []
    return {"t_s": round(max(total - drop, 0.0), 3), "short": short}


def arm_doc(task_id: str, n_test: int, g_doc: dict, log: dict | None, arm: str) -> tuple[dict, float]:
    """One arm's pool document (the bank's format; candidates ranked by R.rank_output, kgmon and probmul3) and its
    seconds. arm: "g", "gt1", "gt2", "t1", "t2"."""
    g_secs = float((g_doc.get("run") or {}).get("elapsed_s") or 0.0)
    train_s = float((log or {}).get("train_s") or 0.0)
    gs = g_samples(g_doc)
    if arm == "g":
        return g_doc, g_secs
    n = int(arm[-1])
    ts = t_samples(log, n)
    tsec = t_seconds(log, n)["t_s"]
    with_g = arm.startswith("gt")
    outputs = []
    for j in range(n_test):
        rows = [dict(x) for x in gs.get(j, [])] if with_g else []
        order = len(rows)
        g_by_view: dict[str, set] = {}
        for x in rows:
            g_by_view.setdefault(x["view"], set()).add(_gkey(x["solution"]))
        views = [v for v in (log.get("views") or []) if v.split(".")[0] == f"{task_id}_{j}"]
        for v in views:
            for i, s in enumerate(ts.get(v, [])):
                if with_g and _gkey(s["solution"]) in g_by_view.get(v, set()):
                    continue
                rows.append({"key": f"{v}.t{i}", "beam_score": s["beam_score"], "score_aug": s["score_aug"],
                             "solution": s["solution"], "order": order, "found_s": round((g_secs if with_g else train_s)
                                                                                          + tsec, 3)})
                order += 1
        cands, selected = R.rank_output(rows)
        outputs.append({"test_index": j, "n_files": None, "n_beams": len(rows), "n_unique": len(cands),
                        "rank_check_ok": True,
                        "margin_kgmon": (cands[0]["score_kgmon"] - cands[1]["score_kgmon"]) if len(cands) > 1 else None,
                        "candidates": cands, "selected": selected})
    secs = (g_secs if with_g else train_s) + tsec
    run = {"elapsed_s": round(secs, 3), "arm": arm, "g_elapsed_s": g_secs, "t_s": tsec, "train_s": train_s,
           "stop_reason": "exhausted", "end_time_reached": False}
    return {"schema": "arc2_pool_audit.v1", "task_id": task_id, "n_test": n_test, "selection": "kgmon (R.rank_output)",
            "run": run, "outputs": outputs}, secs


def same_export(mine: dict, theirs: dict) -> list[str]:
    bad = []
    for a, b in zip(mine["outputs"], theirs.get("outputs") or []):
        ga = sorted(json.dumps(c["grid"]) for c in a["candidates"])
        gb = sorted(json.dumps(c["grid"]) for c in b.get("candidates") or [])
        if ga != gb:
            bad.append(f"{mine['task_id']} output {a['test_index']}: rebuilt candidates {len(ga)} != export {len(gb)}")
        elif not all(grids_equal(x, y) for x, y in zip(a["selected"], b.get("selected") or [])):
            bad.append(f"{mine['task_id']} output {a['test_index']}: rebuilt pair differs from the export's")
    if len(mine["outputs"]) != len(theirs.get("outputs") or []):
        bad.append(f"{mine['task_id']}: {len(mine['outputs'])} outputs rebuilt, {len(theirs.get('outputs') or [])} exported")
    return bad


# ----------------------------------------------------------------------------
# the floor

def check_log(log: dict) -> list[str]:
    """Items 2 and 3 on one task log."""
    t = log.get("task_id")
    bad = []
    got = {k: log.get(k) for k in LOG_WANT}
    if got != LOG_WANT or abs(float(log.get("abandon_score") or 0) + math.log(0.001)) > 1e-9:
        bad.append(f"{t}: log settings {got}, abandon {log.get('abandon_score')}")
    want_seed = int.from_bytes(hashlib.sha256(("ts:" + str(t)).encode()).digest()[:8], "big") % (2**63)
    if log.get("seed") != want_seed:
        bad.append(f"{t}: sampling seed {log.get('seed')}")
    for b in log.get("batches") or []:
        nv = len(b.get("views") or [])
        for rd in b.get("rounds") or []:
            rows = rd.get("rows") or []
            if len(rows) != nv or sorted(r["row"] for r in rows) != list(range(nv)):
                bad.append(f"{t} batch {b.get('i')} round {rd.get('r')}: {len(rows)} rows for {nv} views")
            for r in rows:
                if r.get("status") == "ended" and r.get("grid") is None:
                    bad.append(f"{t} batch {b.get('i')}: an ended row without its grid")
    for i, g in enumerate(log.get("grids") or []):
        if len(g.get("score_aug") or []) != 8 or (g.get("rescored_by_t") and len(g.get("call_s") or []) != 2):
            bad.append(f"{t}: grid {i} with {len(g.get('score_aug') or [])} scores, {len(g.get('call_s') or [])} call times")
    return bad


def t_failed(log: dict | None, t_doc: dict | None, gt_doc: dict | None) -> str | None:
    if log is None:
        return "no T log"
    if log.get("error"):
        return f"T error: {str(log['error'])[:200]}"
    if t_doc is None or gt_doc is None:
        return "no T pool"
    if log.get("end_time_reached") or log.get("stop") == "global_end_time":
        return "T stopped by the global end time"
    return None


def check_kernel(k: str, spec: dict, archive: Path, bank_env: dict) -> dict:
    listed = list(spec["task_ids"])
    pool, pool_t, pool_gt, logs = archive / "pool", archive / "pool_t", archive / "pool_gt", archive / "ts"
    if not pool.is_dir():
        return {"kernel": k, "status": "missing", "run": [], "not_run": listed, "t_failed": {},
                "problems": [f"no archive at {archive}"], "n_problems": 1, "download_gaps": []}
    problems, run, not_run, tfail = [], [], [], {}
    files = {p.stem: p for p in pool.glob("*.json") if not p.name.startswith("_")}
    extra = sorted(set(files) - set(listed))
    if extra:
        problems.append(f"pool files for unlisted tasks: {extra}")
    gaps = set(download_gaps(pool))
    meta_path = pool / "_run.json"
    meta = (json.loads(meta_path.read_text(encoding="utf-8")).get("run") or {}) if meta_path.is_file() else {}
    ts_meta = meta.get("ts") or {}
    for name, d in (("logs", logs), ("pool_t", pool_t), ("pool_gt", pool_gt)):
        for f in ts_meta.get(name) or []:
            if not (d / f).is_file():
                gaps.add(f.split(".")[0])
    gaps = sorted(gaps)
    if gaps:
        problems.append(f"download incomplete, re-fetch the archive: {gaps}")
    for t in listed:
        if t not in files:
            not_run.append(t)
            continue
        doc = json.loads(files[t].read_text(encoding="utf-8"))
        if (doc.get("run") or {}).get("end_time_reached"):
            not_run.append(t)
            continue
        run.append(t)
        for o in doc.get("outputs") or []:
            if o.get("rank_check_ok") is False:
                problems.append(f"{t} output {o.get('test_index')}: rank_check_ok false")
            for c in o.get("candidates") or []:
                if b0r_check(c):
                    problems.append(f"{t} output {o.get('test_index')}: candidate fields {b0r_check(c)}")
        lp = logs / f"{t}.json.gz"
        log = load_log(lp) if lp.is_file() else None
        tp, gp = pool_t / f"{t}.json", pool_gt / f"{t}.json"
        t_doc = json.loads(tp.read_text(encoding="utf-8")) if tp.is_file() else None
        gt_doc = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() else None
        why = t_failed(log, t_doc, gt_doc)
        if why:
            tfail[t] = why
            continue
        problems += check_log(log)
        n_test = int(doc.get("n_test") or len(doc.get("outputs") or []))
        problems += same_export(arm_doc(t, n_test, doc, log, "gt2")[0], gt_doc)
        problems += same_export(arm_doc(t, n_test, doc, log, "t2")[0], t_doc)
    if meta:
        want = {"variant": f"ts-{k}", "ttt_sequences": 64, "bank_seed": 0, "dfs_min_prob": 0.2}
        got = {key: meta.get(key) for key in want}
        if got != want:
            problems.append(f"_run.json run {got} != {want}")
        if {key: ts_meta.get(key) for key in TS_WANT} != TS_WANT:
            problems.append(f"_run.json ts {ts_meta}")
        solver = ((meta.get("environment") or {}).get("solver") or {})
        env = {key: solver.get(key) for key in ENV_KEYS}
        if env != bank_env:
            problems.append(f"environment {env} != the bank's {bank_env}")
    else:
        problems.append("no pool/_run.json")
    return {"kernel": k, "status": "report_only" if problems else "valid", "run": run, "not_run": not_run,
            "t_failed": tfail, "problems": problems[:50], "n_problems": len(problems), "download_gaps": gaps}


# ----------------------------------------------------------------------------
# the reading

def mcnemar_two_sided(gained: int, lost: int) -> float | None:
    n = gained + lost
    return min(1.0, 2 * binom_tail(max(gained, lost), n)) if n else None


def write_arms(archives: dict[str, Path], runs: dict[str, list[str]], t_ok: set[str], out: Path) -> dict:
    """Every arm's pools under out/<arm>/ and per-task seconds."""
    if out.exists():
        shutil.rmtree(out)
    secs: dict[str, dict[str, float]] = {a: {} for a in ("g", "gt1", "gt2", "t1", "t2")}
    shorts: dict[str, list] = {}
    for k, tasks in runs.items():
        for t in tasks:
            g_doc = json.loads((archives[k] / "pool" / f"{t}.json").read_text(encoding="utf-8"))
            n_test = int(g_doc.get("n_test") or len(g_doc["outputs"]))
            log = load_log(archives[k] / "ts" / f"{t}.json.gz") if t in t_ok else None
            for arm in secs:
                if arm != "g" and t not in t_ok:
                    continue
                doc, s = arm_doc(t, n_test, g_doc, log, arm)
                d = out / arm
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{t}.json").write_text(json.dumps(doc), encoding="utf-8")
                secs[arm][t] = s
            if log is not None:
                sh = t_seconds(log, 1)["short"]
                if sh:
                    shorts[t] = sh
    return {"seconds": secs, "t1_short": shorts}


def read_ts(plan: dict, archives: dict[str, Path], pools_out: Path, *, bank_pools=BANK_PARTS, q27_dir=Q27_BANK,
            challenges=CHALLENGES, solutions=None, n_boot: int = 4000) -> dict:
    bank_env = bank_environment(bank_pools)
    kernels = {k: check_kernel(k, plan["kernels"][k], archives[k], bank_env) for k in plan["kernels"]}
    valid = [k for k in kernels if kernels[k]["status"] == "valid"]
    tasks = sorted(t for k in valid for t in kernels[k]["run"])
    t_fail = {t: why for k in valid for t, why in kernels[k]["t_failed"].items()}
    t_tasks = sorted(t for t in tasks if t not in t_fail)
    res = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule": plan.get("rule"),
           "kernels": kernels, "n_tasks_run": len(tasks), "n_tasks_t": len(t_tasks), "t_failed": t_fail,
           "report_only_tasks": sorted(t for k in kernels if kernels[k]["status"] == "report_only" for t in kernels[k]["run"]),
           "not_run_tasks": sorted(t for k in kernels for t in kernels[k]["not_run"])}
    if not t_tasks:
        res["outcome"] = "no task with T run in a valid kernel: nothing to read"
        return res
    arms = write_arms({k: archives[k] for k in valid}, {k: kernels[k]["run"] for k in valid}, set(t_tasks), pools_out)
    secs = arms["seconds"]
    banks = {a: load_bank([pools_out / a], q27_dir, challenges=challenges, solutions=solutions, tasks=t_tasks)
             for a in secs}
    ref_bank = load_bank(list(bank_pools), q27_dir, challenges=challenges, solutions=solutions, tasks=tasks)
    g_all = load_bank([pools_out / "g"], q27_dir, challenges=challenges, solutions=solutions, tasks=tasks)
    g = banks["g"]
    g_secs = {t: secs["g"][t] for t in t_tasks}
    g_e13 = tcp_e13(g, g_secs, t_tasks)
    base_alone = replay(g, RULES[("kgmon", "alone")])
    base_merged = replay(g, RULES[("kgmon", "merged")])
    halves = {"A": {f"{o.task_id}_{o.test_index}" for o in g.outputs() if half_of(o.task_id) == "A"},
              "B": {f"{o.task_id}_{o.test_index}" for o in g.outputs() if half_of(o.task_id) == "B"}}
    # item 5 (whole run): G's seconds against the bank's on the tasks run
    bank_secs = {t: ref_bank.tasks[t].elapsed_s for t in tasks}
    ratio = sum(secs["g"][t] for t in tasks) / sum(bank_secs.values())
    item5 = TIME_BAND[0] <= ratio <= TIME_BAND[1]
    item6 = len(t_fail) <= T_FAIL_MAX
    tcp_valid = item5 and item6
    lines = {}
    for n in SETTINGS:
        arm = f"gt{n}"
        b = banks[arm]
        s = {t: secs[arm][t] for t in t_tasks}
        tc = tcp(b, g_e13, s, t_tasks, cap_s=GT_CAP_S)
        alone = replay(b, RULES[("kgmon", "alone")])
        merged = replay(tc["bank"], RULES[("kgmon", "merged")])
        lines[f"G+T{n}"] = {
            "rounds": n,
            "alone": {h: cmp_on(base_alone, alone, halves[h] if h != "all" else None) for h in ("A", "B", "all")},
            "tcp": {h: cmp_on(base_merged, merged, halves[h] if h != "all" else None) for h in ("A", "B", "all")},
            "probmul3_alone": cmp_on(base_alone, replay(b, RULES[("probmul3", "alone")])),
            "delta_s": tc["delta_s"], "k": tc["k"], "samples_kept": tc["samples_kept"], "samples_all": tc["samples_all"],
            "b0_cut_fraction": tc["b0_cut_fraction"], "b0_cut_tasks": tc["b0_cut_tasks"], "e13": tc["e13"],
            "time_ratio_to_g": round(sum(s.values()) / sum(g_secs.values()), 4),
        }
    # selection (section 4)
    best = sorted(lines, key=lambda a: (-lines[a]["tcp"]["A"]["net"], lines[a]["tcp"]["A"]["lost"], lines[a]["delta_s"],
                                        lines[a]["rounds"]))
    winner = best[0] if lines[best[0]]["tcp"]["A"]["net"] >= 1 else None
    sel = {"winner": winner, "ranked_on_A": [(a, lines[a]["tcp"]["A"]["net"], lines[a]["tcp"]["A"]["lost"]) for a in best]}
    if winner is None:
        outcome = "no winner on half A: the bank's decoding stays"
    else:
        lb, la = lines[winner]["tcp"]["B"], lines[winner]["tcp"]["all"]
        confirmed = lb["net"] >= 1 and lb["lost"] <= 1
        qualified = confirmed and la["net"] >= 3 and la["lost"] <= 1 and tcp_valid
        sel.update({"confirmed_on_B": confirmed, "gate_v2_all": la["net"] >= 3 and la["lost"] <= 1, "qualified": qualified})
        if not tcp_valid:
            outcome = "TCP lines report only (validity item 5 or 6): nothing qualifies"
        elif qualified:
            outcome = (f"{winner} qualifies: next an entry draft (v7-vr2 with T after each task's search, B0's "
                       f"phase end moved by its delta) and a commit run for the Nov 2 slot, under its own rule")
        elif confirmed:
            outcome = f"{winner} confirmed on B but not qualified (gate v2 on all): recorded; no entry"
        else:
            outcome = f"{winner} not confirmed on half B: the bank's decoding stays"
    # the sampling line (B0 alone, G+T2 against G)
    sl = lines["G+T2"]["alone"]["all"]
    n_out = sl["n"]
    if not item6:
        sampling = "report only (validity item 6: T failed on more than 2 tasks)"
    elif sl["net"] >= math.ceil(5 * n_out / 145) and sl["p"] is not None and sl["p"] <= 0.05:
        sampling = "gain: temperature sampling adds right outputs to B0's DFS at about its own search time (paper)"
    elif sl["net"] <= math.floor(n_out / 145):
        sampling = ("no gain: temperature sampling is not a lever for B0 on this evidence; with decode-many, this closes "
                    "the sampling question")
    else:
        sampling = "reported (between the gain and no-gain lines)"
    # recorded beside
    gb = cmp_on(replay(ref_bank, "b0"), replay(g_all, RULES[("kgmon", "alone")]))
    agree_cands = agree_sel = n_cmp = 0
    for t in tasks:
        for o_g, o_b in zip(g_all.tasks[t].outputs, ref_bank.tasks[t].outputs):
            n_cmp += 1
            agree_cands += int(sorted(json.dumps(c["grid"]) for c in o_g.candidates)
                               == sorted(json.dumps(c["grid"]) for c in o_b.candidates))
            agree_sel += int(all(grids_equal(x, y) for x, y in zip((o_g.attempt_1, o_g.attempt_2), (o_b.attempt_1, o_b.attempt_2))))

    def oracle(b):
        return sum(1 for o in b.outputs() if b.truth(o) is not None and any(grids_equal(c["grid"], b.truth(o)) for c in o.candidates))

    t_only, vote_moved = [], []
    for o in banks["gt2"].outputs():
        truth = banks["gt2"].truth(o)
        if truth is None:
            continue
        og = next(x for x in g.tasks[o.task_id].outputs if x.test_index == o.test_index)
        in_g = [c for c in og.candidates if grids_equal(c["grid"], truth)]
        in_gt = [c for c in o.candidates if grids_equal(c["grid"], truth)]
        if in_gt and not in_g:
            t_only.append(f"{o.task_id}_{o.test_index}")
        elif in_gt and in_g and in_g[0]["rank_kgmon"] > 2 >= in_gt[0]["rank_kgmon"]:
            vote_moved.append(f"{o.task_id}_{o.test_index}")
    stats = {"rows": 0, "ended": 0, "abandoned": 0, "unended": 0, "cut": 0, "invalid": 0, "in_g_view": 0, "new_grids": 0}
    for k in valid:
        for t in kernels[k]["run"]:
            if t not in set(t_tasks):
                continue
            log = load_log(archives[k] / "ts" / f"{t}.json.gz")
            for b in log.get("batches") or []:
                for rd in b.get("rounds") or []:
                    for row in rd.get("rows") or []:
                        stats["rows"] += 1
                        stats[row["status"]] = stats.get(row["status"], 0) + 1
                        stats["in_g_view"] += int(bool(row.get("in_g_view")))
            stats["new_grids"] += sum(1 for x in log.get("grids") or [] if x.get("rescored_by_t"))
    res.update({
        "tasks": tasks, "t_tasks": t_tasks,
        "halves": {h: sorted({x.rsplit("_", 1)[0] for x in halves[h]}) for h in halves},
        "n_outputs": len(base_alone.per_output),
        "validity": {"item5_g_time_ratio": round(ratio, 4), "item5_pass": item5, "item6_t_failed": len(t_fail),
                     "item6_pass": item6, "tcp_valid": tcp_valid},
        "g": {"e13": g_e13, "against_bank_b0": dict(gb, p_two_sided=mcnemar_two_sided(gb["gained"], gb["lost"])),
              "candidates_equal": agree_cands, "selections_equal": agree_sel, "outputs_compared": n_cmp,
              "time_ratio_to_bank": round(ratio, 4),
              "time_per_task": {t: [round(secs["g"][t], 1), bank_secs[t]] for t in tasks}},
        "lines": lines, "selection": sel, "outcome": outcome,
        "sampling_line": {"line": sl, "decision": sampling},
        "recorded": {
            "t_alone": {f"T{n}": dict(cmp_on(base_alone, replay(banks[f"t{n}"], RULES[("kgmon", "alone")])),
                                      time_ratio_to_g_search=round(sum(secs[f"t{n}"].values()) /
                                                                   max(1e-9, sum(g_secs.values())), 4))
                        for n in SETTINGS},
            "oracle": {a: oracle(banks[a]) for a in ("g", "t2", "gt2", "gt1")},
            "right_grids_only_t_found": t_only, "right_grids_t_lifted_into_top2": vote_moved,
            "samples": stats, "t1_short": arms["t1_short"],
            "bootstrap_sampling_line": bootstrap_difference(base_alone, replay(banks["gt2"], RULES[("kgmon", "alone")]),
                                                            n_boot=n_boot),
        },
    })
    return res


def summary_lines(res: dict) -> list[str]:
    out = [f"TS reading ({res.get('rule')})"]
    for k, x in res["kernels"].items():
        out.append(f"  {k}: {x['status']}; run {len(x['run'])}, not run {len(x['not_run'])}, T failed {len(x['t_failed'])}; "
                   f"problems {x.get('n_problems', 0)}")
    if "lines" not in res:
        out.append(f"  outcome: {res.get('outcome')}")
        return out
    v, gl = res["validity"], res["g"]["against_bank_b0"]
    out.append(f"  {res['n_tasks_run']} tasks ({res['n_tasks_t']} with T), {res['n_outputs']} outputs; G time / bank "
               f"{v['item5_g_time_ratio']} (item 5 {'PASS' if v['item5_pass'] else 'FAIL'}); T failed {v['item6_t_failed']} "
               f"(item 6 {'PASS' if v['item6_pass'] else 'FAIL'})")
    out.append(f"  G against the bank's B0: {gl['new_right']} vs {gl['base_right']} (G only {gl['gained']}, bank only "
               f"{gl['lost']}, two-sided p {gl['p_two_sided']}); candidates equal {res['g']['candidates_equal']}, "
               f"selections equal {res['g']['selections_equal']} of {res['g']['outputs_compared']}")
    out.append("| track | arm | B0 alone net (A/B/all) | TCP net (A/B/all) | lost all | delta s | k | time / G |")
    for a, x in res["lines"].items():
        al, t = x["alone"], x["tcp"]
        out.append(f"| ARC-2 | {a} | {al['A']['net']:+d}/{al['B']['net']:+d}/{al['all']['net']:+d} | "
                   f"{t['A']['net']:+d}/{t['B']['net']:+d}/{t['all']['net']:+d} | {t['all']['lost']} | {x['delta_s']:.0f} | "
                   f"{x['k']:.2f} | {x['time_ratio_to_g']:.2f} |")
    out.append(f"  selection: {res['selection']}")
    out.append(f"  outcome: {res['outcome']}")
    out.append(f"  sampling line: {res['sampling_line']['line']['net']:+d} (p {res['sampling_line']['line']['p']}): "
               f"{res['sampling_line']['decision']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--pools-out", default=str(POOLS_OUT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--floor-only", default="", metavar="K",
                    help="items 2 to 4 and T failures on one kernel's archive (<archive>/ts_floor.json); exit 1 unless valid")
    args = ap.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    archives = {k: Path(args.runs) / ARCHIVE.format(k=k) for k in plan["kernels"]}
    if args.floor_only:
        k = args.floor_only
        rep = check_kernel(k, plan["kernels"][k], archives[k], bank_environment())
        if archives[k].is_dir():
            (archives[k] / "ts_floor.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps({key: rep.get(key) for key in ("kernel", "status", "n_problems", "download_gaps")}
                         | {"t_failed": len(rep.get("t_failed") or {})}))
        return 0 if rep["status"] == "valid" else 1
    res = read_ts(plan, archives, Path(args.pools_out))
    gaps = {k: x["download_gaps"] for k, x in res["kernels"].items() if x.get("download_gaps")}
    if gaps:
        print(f"download incomplete, re-fetch these archives before reading (nothing written): {gaps}")
        return 2
    Path(args.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for line in summary_lines(res):
        print(line)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
