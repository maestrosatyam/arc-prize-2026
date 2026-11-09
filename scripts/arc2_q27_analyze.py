#!/usr/bin/env python3
"""Analyse the q27 run (Qwen3.8-27B REPL agent on the 40 ARC-AGI-2 dev tasks; experiments/arc2-q27/DESIGN.md)
against B0's two audit runs on the same 65 test outputs, and simulate NVARC + 27B hybrids.

Inputs
  --q27 DIR      the run's q27/ directory: <task_id>.json per task, _run.json, optional _stages.json and
                 _kaggle_solutions_audit_tasks.json
  --b0 NAME=DIR  B0 audit pools (default v1 and v2: experiments/runs/kaggle-A2-pool-audit-v{1,2}/pool)
  ground truth   --solutions, else the first Kaggle solutions copy found in the q27 directory or the pool
                 directories (also looked up in parent checkouts, since the copies are not committed), else the
                 local ARC-AGI-2 copy (differs from Kaggle's file on 3 dev tasks; flagged)

Reports (denominator: the 65 outputs of Kaggle's file)
  27B           pass@2, pass@1, attempt_1 train-verified and how often it is right; per task stop reason,
                generated / prompt tokens, turns, wall seconds
  overlap       per output: 27B vs B0 v1 and v2; outputs 27B solves that B0 missed in both runs; oracle union
  hybrid        routing rules from B0's own signals (fewer than 2 candidates; kgmon margin below a threshold;
                B0's attempt 1 breaks the output-size rule the demonstrations fix) x merge policies (27B's
                train-verified answer replaces attempt 2 or takes attempt 1) x 27B per-task token caps x B0 budget
                (unchanged, or test-time training halved), scored on each B0 run, with the projected 240-task wall
                time: B0 mean task seconds x 240 / 4 ranks + B0 overhead + 27B start-up + routed share x 240 x 27B
                generated tokens per routed task / the run's measured generated tokens per second
  decision      stage 2 (DESIGN.md) if 27B solves >= 3 outputs B0 missed in both runs and a rule fits <= 11 h

Usage: python scripts/arc2_q27_analyze.py --q27 <run>/q27 [--json out.json] [--per-output]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from arc2_pool_audit_analyze import (HIDDEN_TASKS, HIDDEN_TASKS_SOURCE, RANKS, WALL_TARGET_H,  # noqa: E402
                                     _attempts, _run_times, find_solutions, load_pool)
from eval.arc2_score import grids_equal  # noqa: E402

SPLITS = ROOT / "splits" / "arc2_eval_split_v1.json"
DEFAULT_B0 = {"v1": "experiments/runs/kaggle-A2-pool-audit-v1/pool", "v2": "experiments/runs/kaggle-A2-pool-audit-v2/pool"}
KAGGLE_COPY = "_kaggle_solutions_audit_tasks.json"
KAGGLE_CHALLENGES_COPY = "_kaggle_challenges_q27_tasks.json"
NEW_OUTPUT_THRESHOLD = 3
DEFAULT_SWAP_S = 1200.0
# B0 time cut: halve the test-time training sequences (survey candidate #3: 64 instead of 128 sequences; LB 28.06 vs
# 27.22, single draws). Saving assumed = half of the test-time training share of task time; that share is about 190 s
# of 626 s (30 %) per task in the audit log (experiments/arc2-pool-audit/DESIGN.md, W1 cost and scaling); TTT time
# assumed linear in the number of sequences; accuracy effect assumed 0 (not measured).
TTT_SHARE = 0.30
B0_CUTS = {"none": 0.0, "ttt_half": TTT_SHARE / 2}
MERGES = ("a2_if_verified", "a1_if_verified", "a1_if_verified_else_a2", "a2_always")
FIXED_MARGINS = (0.5, 1.0, 2.0, 4.0)


# ----------------------------------------------------------------------------
# inputs

def dev_task_ids() -> list[str]:
    return sorted(json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]["dev"])


def load_q27(q27_dir: Path) -> tuple[dict[str, dict], dict, list]:
    docs = {}
    for p in sorted(q27_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        docs[d.get("task_id", p.stem)] = d
    run = json.loads((q27_dir / "_run.json").read_text(encoding="utf-8")) if (q27_dir / "_run.json").is_file() else {}
    stages = json.loads((q27_dir / "_stages.json").read_text(encoding="utf-8")) if (q27_dir / "_stages.json").is_file() else []
    return docs, run, stages


def find_ground_truth(explicit: str, dirs: list[Path]) -> tuple[dict, str, bool]:
    """(solutions, source, is_kaggle_copy)."""
    if explicit:
        sol, src = find_solutions(explicit)
        return sol, src, True
    cands = []
    for d in dirs:
        cands.append(d / KAGGLE_COPY)
        try:
            rel = d.resolve().relative_to(ROOT)
        except ValueError:
            continue
        cands += [base / rel / KAGGLE_COPY for base in ROOT.parents]
    for c in cands:
        if c.is_file():
            return json.loads(c.read_text(encoding="utf-8")), f"{c} (Kaggle evaluation solutions)", True
    sol, src = find_solutions(None)
    return sol, src + " (local ARC-AGI-2 copy: differs from Kaggle's file on abc82100, b6f77b65, faa9f03d)", False


def size_rule(task: dict | None) -> str | tuple | None:
    """'same' when every demonstration output has its input's size, a (rows, cols) tuple when all outputs share one
    size, else None."""
    if not task:
        return None
    shape = lambda g: (len(g), len(g[0]) if g else 0)  # noqa: E731
    tr = task.get("train", [])
    if tr and all(shape(p["output"]) == shape(p["input"]) for p in tr):
        return "same"
    shapes = {shape(p["output"]) for p in tr}
    return shapes.pop() if len(shapes) == 1 else None


def breaks_size_rule(grid, rule, test_input) -> bool:
    if rule is None or not isinstance(grid, list) or not grid:
        return False
    s = (len(grid), len(grid[0]) if isinstance(grid[0], list) else 0)
    want = (len(test_input), len(test_input[0])) if rule == "same" else rule
    return s != want


def b0_view(pool: Path, task_ids: list[str], sol: dict, challenges: dict | None) -> dict[str, Any]:
    docs, run, _ = load_pool(pool)
    times = _run_times(docs, run, task_ids, RANKS)
    outs = {}
    for t in task_ids:
        d = docs.get(t)
        by_j = {o["test_index"]: o for o in (d or {}).get("outputs", [])}
        rule = size_rule((challenges or {}).get(t))
        for j, s in enumerate(sol[t]):
            o = by_j.get(j)
            if o is None:
                outs[(t, j)] = {"a1": None, "a2": None, "n_unique": 0, "margin": None, "correct": False,
                                "correct1": False, "breaks_size": False}
                continue
            a1, a2, _ = _attempts(o)
            tests = ((challenges or {}).get(t) or {}).get("test", [])
            test_in = tests[j].get("input") if j < len(tests) else None
            outs[(t, j)] = {"a1": a1, "a2": a2, "n_unique": int(o.get("n_unique") or 0), "margin": o.get("margin_kgmon"),
                            "correct": grids_equal(a1, s) or grids_equal(a2, s), "correct1": grids_equal(a1, s),
                            "breaks_size": bool(test_in) and breaks_size_rule(a1, rule, test_in)}
    el = times["elapsed"]
    return {"outputs": outs, "elapsed": el, "overhead_s": times["overhead_s"], "overhead_source": times["overhead_source"],
            "mean_task_s": statistics.fmean(el.values()) if el else None, "n_timed": len(el)}


def q27_view(docs: dict[str, dict], task_ids: list[str], sol: dict) -> dict[str, Any]:
    outs, tasks = {}, {}
    for t in task_ids:
        d = docs.get(t)
        va = (d or {}).get("verified_at") or {}
        tasks[t] = {"present": d is not None, "stop": (d or {}).get("stop_reason", "missing"),
                    "verified": bool((d or {}).get("verified")),
                    "verified_tokens": va.get("task_tokens"),
                    "gen": int(((d or {}).get("tokens") or {}).get("generated", 0)),
                    "prompt": int(((d or {}).get("tokens") or {}).get("prompt", 0)),
                    "turns": int((d or {}).get("turns", 0)), "wall_s": float((d or {}).get("wall_s", 0.0)),
                    "order": (d or {}).get("order"), "n_test": len(sol[t])}
        by_j = {o["test_index"]: o for o in (d or {}).get("outputs", [])}
        for j, s in enumerate(sol[t]):
            o = by_j.get(j)
            if o is None:
                outs[(t, j)] = {"a1": None, "a2": None, "v1": False, "correct": False, "correct1": False}
                continue
            outs[(t, j)] = {"a1": o["attempt_1"], "a2": o["attempt_2"], "v1": bool(o["attempt_1_verified"]),
                            "correct": grids_equal(o["attempt_1"], s) or grids_equal(o["attempt_2"], s),
                            "correct1": grids_equal(o["attempt_1"], s)}
    return {"outputs": outs, "tasks": tasks}


# ----------------------------------------------------------------------------
# hybrid simulation

def route(rule: str, theta: float | None, task: str, b0: dict, keys_by_task: dict) -> bool:
    outs = [b0["outputs"][k] for k in keys_by_task[task]]
    lt2 = any(o["n_unique"] < 2 for o in outs)
    low = any(o["margin"] is None or o["margin"] < theta for o in outs) if theta is not None else False
    size = any(o["breaks_size"] for o in outs)
    return {"none": False, "all": True, "lt2": lt2, "margin": lt2 or low, "size": size,
            "lt2|size": lt2 or size, "margin|size": lt2 or low or size}[rule]


def _dedupe2(grids: list) -> list:
    out = []
    for g in grids:
        if g is not None and not any(grids_equal(g, h) for h in out):
            out.append(g)
    return out[:2]


def merge(policy: str, b: dict, q: dict | None, q_verified: bool) -> list:
    base = [b["a1"], b["a2"]]
    if q is None or q.get("a1") is None:
        return _dedupe2(base)
    if policy == "a2_if_verified":
        return _dedupe2([b["a1"], q["a1"], b["a2"]] if q_verified else base)
    if policy == "a1_if_verified":
        return _dedupe2([q["a1"], b["a1"], b["a2"]] if q_verified else base)
    if policy == "a1_if_verified_else_a2":
        return _dedupe2([q["a1"], b["a1"], b["a2"]] if q_verified else [b["a1"], q["a1"], b["a2"]])
    if policy == "a2_always":
        return _dedupe2([b["a1"], q["a1"], b["a2"]])
    raise ValueError(policy)


def q27_under_cap(task: dict, out: dict, cap: int | None) -> tuple[dict | None, bool, int]:
    """(27B answer usable under the cap or None, verified within the cap, generated tokens charged)."""
    if not task["present"]:
        return None, False, 0
    if cap is None or task["gen"] <= cap:
        return out, out["v1"], task["gen"]
    vt = task["verified_tokens"]
    if out["v1"] and vt is not None and vt <= cap:
        return out, True, cap
    return None, False, cap


def simulate(b0s: dict[str, dict], q: dict, sol: dict, task_ids: list[str], tok_per_s: float, swap_s: float,
             hidden_tasks: int, caps: list[int | None], thetas: list[float], wall_target_h: float) -> list[dict]:
    keys_by_task = {t: [(t, j) for j in range(len(sol[t]))] for t in task_ids}
    rules = [("none", None), ("all", None), ("lt2", None), ("size", None), ("lt2|size", None)]
    rules += [("margin", th) for th in thetas] + [("margin|size", th) for th in thetas]
    rows = []
    for rule, th in rules:
        for policy in (MERGES if rule != "none" else ("-",)):
            for cap in (caps if rule != "none" else [None]):
                for cut_name, cut in B0_CUTS.items():
                    row = {"rule": rule, "theta": th, "merge": policy, "cap": cap, "b0_cut": cut_name}
                    for name, b0 in b0s.items():
                        routed = [t for t in task_ids if route(rule, th, t, b0, keys_by_task)]
                        hits, charged = 0, 0
                        for t in task_ids:
                            on = t in routed
                            for k in keys_by_task[t]:
                                bq, qv, _ = (q27_under_cap(q["tasks"][t], q["outputs"][k], cap) if on else (None, False, 0))
                                final = merge(policy, b0["outputs"][k], bq, qv) if on else _dedupe2(
                                    [b0["outputs"][k]["a1"], b0["outputs"][k]["a2"]])
                                hits += any(grids_equal(g, sol[t][k[1]]) for g in final)
                            if on:
                                charged += q27_under_cap(q["tasks"][t], q["outputs"][keys_by_task[t][0]], cap)[2]
                        share = len(routed) / len(task_ids)
                        b0_h = ((b0["mean_task_s"] * (1 - cut)) * hidden_tasks / RANKS + b0["overhead_s"]) / 3600
                        q_tokens_per_task = charged / len(routed) if routed else 0.0
                        q_h = ((swap_s + share * hidden_tasks * q_tokens_per_task / tok_per_s) / 3600) if routed else 0.0
                        row[name] = {"pass2": hits, "routed": len(routed), "b0_h": round(b0_h, 2), "q27_h": round(q_h, 2),
                                     "total_h": round(b0_h + q_h, 2)}
                    row["mean_pass2"] = round(statistics.fmean(row[n]["pass2"] for n in b0s), 2)
                    row["max_total_h"] = max(row[n]["total_h"] for n in b0s)
                    row["fits"] = row["max_total_h"] <= wall_target_h
                    rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# report

def analyze(q27_dir: Path, b0_dirs: dict[str, Path], solutions: str = "", hidden_tasks: int = HIDDEN_TASKS,
            wall_target_h: float = WALL_TARGET_H, swap_s: float | None = None, tok_per_s: float | None = None,
            caps: list[int | None] | None = None, task_ids: list[str] | None = None,
            challenges: dict | None = None) -> dict[str, Any]:
    task_ids = task_ids or dev_task_ids()
    sol_all, sol_src, kaggle_gt = find_ground_truth(solutions, [q27_dir, *b0_dirs.values()])
    absent = [t for t in task_ids if t not in sol_all]
    if absent:
        raise SystemExit(f"ground truth lacks {absent}")
    sol = {t: sol_all[t] for t in task_ids}
    ch_src = "given"
    if challenges is None:
        challenges, ch_src = _challenges_for(task_ids, q27_dir)
    docs, run, stages = load_q27(q27_dir)
    q = q27_view(docs, task_ids, sol)
    b0s = {n: b0_view(d, task_ids, sol, challenges) for n, d in b0_dirs.items()}
    keys = [(t, j) for t in task_ids for j in range(len(sol[t]))]
    n_out = len(keys)

    # throughput and start-up of the run
    agent_s = (run.get("run") or {}).get("agent_wall_s") or run.get("agent_wall_s")
    gen = sum(v["gen"] for v in q["tasks"].values())
    measured_tps = (gen / agent_s) if agent_s else None
    tps = tok_per_s or measured_tps or 300.0
    tps_src = "--tok-per-s" if tok_per_s else ("measured: generated tokens / agent wall" if measured_tps else
                                               "assumed 300 tok/s (no agent wall time in _run.json)")
    start_s = (run.get("run") or {}).get("agent_start_s")
    swap = swap_s if swap_s is not None else (float(start_s) if start_s else DEFAULT_SWAP_S)
    swap_src = "--swap-overhead-s" if swap_s is not None else (
        "measured: run start to agent start (install, server, smoke request)" if start_s else
        f"default {DEFAULT_SWAP_S:.0f} s")

    # 27B alone
    qo = q["outputs"]
    v1_outs = [k for k in keys if qo[k]["v1"]]
    rep: dict[str, Any] = {
        "ground_truth": sol_src, "ground_truth_is_kaggle": kaggle_gt, "challenges_for_size_rule": ch_src,
        "n_tasks": len(task_ids), "n_outputs": n_out,
        "q27_dir": str(q27_dir), "q27_tasks_present": sum(v["present"] for v in q["tasks"].values()),
        "q27": {"pass2": sum(qo[k]["correct"] for k in keys), "pass1": sum(qo[k]["correct1"] for k in keys),
                "attempt1_verified": len(v1_outs), "attempt1_verified_correct": sum(qo[k]["correct1"] for k in v1_outs),
                "attempt1_unverified_correct": sum(qo[k]["correct1"] for k in keys if not qo[k]["v1"]),
                "tasks_verified": sum(v["verified"] for v in q["tasks"].values()),
                "stop_reasons": _count(v["stop"] for v in q["tasks"].values()),
                "generated_tokens": gen, "prompt_tokens": sum(v["prompt"] for v in q["tasks"].values()),
                "agent_wall_s": agent_s, "gen_tok_per_s": round(tps, 1), "gen_tok_per_s_source": tps_src,
                "startup_s": swap, "startup_source": swap_src,
                "server_option_set": ((run.get("run") or {}).get("server_option_set")),
                "hard_stop": (run.get("run") or {}).get("hard_stop")},
    }
    # overlap
    b0c = {n: {k: b["outputs"][k]["correct"] for k in keys} for n, b in b0s.items()}
    names = list(b0s)
    either = {k: any(b0c[n][k] for n in names) for k in keys}
    both_missed = {k: not either[k] for k in keys}
    new = sorted(f"{t}_{j}" for (t, j) in keys if qo[(t, j)]["correct"] and both_missed[(t, j)])
    rep["b0"] = {n: {"pass2": sum(b0c[n].values()), "pass1": sum(b["outputs"][k]["correct1"] for k in keys),
                     "mean_task_s": round(b["mean_task_s"], 1) if b["mean_task_s"] else None,
                     "overhead_s": b["overhead_s"], "overhead_source": b["overhead_source"],
                     "projected_h": round(((b["mean_task_s"] or 0) * hidden_tasks / RANKS + b["overhead_s"]) / 3600, 2),
                     "lt2_tasks": sum(1 for t in task_ids if any(b["outputs"][(t, j)]["n_unique"] < 2 for j in range(len(sol[t])))),
                     "size_rule_breaks": sum(b["outputs"][k]["breaks_size"] for k in keys)}
                 for n, b in b0s.items()}
    rep["overlap"] = {
        **{f"q27_and_{n}": sum(qo[k]["correct"] and b0c[n][k] for k in keys) for n in names},
        **{f"q27_only_vs_{n}": sum(qo[k]["correct"] and not b0c[n][k] for k in keys) for n in names},
        **{f"{n}_only": sum(b0c[n][k] and not qo[k]["correct"] for k in keys) for n in names},
        **{f"union_q27_{n}": sum(qo[k]["correct"] or b0c[n][k] for k in keys) for n in names},
        "union_all": sum(qo[k]["correct"] or either[k] for k in keys),
        "q27_new_vs_both_b0_runs": len(new), "q27_new_vs_both_b0_runs_ids": new,
        "q27_new_verified": sorted(f"{t}_{j}" for (t, j) in keys if qo[(t, j)]["correct1"] and qo[(t, j)]["v1"] and both_missed[(t, j)]),
    }
    rep["per_task"] = [{"task_id": t, **q["tasks"][t], **{f"b0_{n}_s": b0s[n]["elapsed"].get(t) for n in names}}
                       for t in sorted(task_ids, key=lambda t: (q["tasks"][t]["order"] is None, q["tasks"][t]["order"] or 0, t))]
    rep["per_output"] = [{"output": f"{t}_{j}", "q27": qo[(t, j)]["correct"], "q27_attempt1_verified": qo[(t, j)]["v1"],
                          **{f"b0_{n}": b0c[n][(t, j)] for n in names},
                          **{f"b0_{n}_n_unique": b0s[n]["outputs"][(t, j)]["n_unique"] for n in names},
                          **{f"b0_{n}_margin": b0s[n]["outputs"][(t, j)]["margin"] for n in names}}
                         for (t, j) in keys]

    # hybrid
    caps = caps if caps is not None else [None, 32768, 24576, 16384]
    margins = sorted({m for b in b0s.values() for m in (min((b["outputs"][(t, j)]["margin"] if b["outputs"][(t, j)]["margin"] is not None else float("-inf"))
                                                             for j in range(len(sol[t]))) for t in task_ids) if m != float("-inf")})
    qs = [margins[int(p * (len(margins) - 1))] for p in (0.1, 0.25, 0.5, 0.75, 0.9)] if margins else []
    thetas = sorted({round(x, 3) for x in (*FIXED_MARGINS, *qs)})
    rows = simulate(b0s, q, sol, task_ids, tps, swap, hidden_tasks, caps, thetas, wall_target_h)
    base = next(r for r in rows if r["rule"] == "none" and r["b0_cut"] == "none")
    fitting = [r for r in rows if r["fits"] and r["rule"] != "none"]
    best = max(fitting, key=lambda r: (r["mean_pass2"], -r["max_total_h"])) if fitting else None
    best_any = max((r for r in rows if r["rule"] != "none"), key=lambda r: (r["mean_pass2"], -r["max_total_h"]))
    rep["hybrid"] = {"thetas": thetas, "caps": caps, "b0_cuts": B0_CUTS, "merges": MERGES, "n_rows": len(rows),
                     "baseline": base, "best_fitting": best, "best_ignoring_time": best_any,
                     "top_fitting": sorted(fitting, key=lambda r: (-r["mean_pass2"], r["max_total_h"]))[:12],
                     "assumptions": {
                         "hidden_tasks": f"{hidden_tasks} ({HIDDEN_TASKS_SOURCE})" if hidden_tasks == HIDDEN_TASKS else str(hidden_tasks),
                         "b0_time": "mean task seconds of that B0 run x hidden tasks / 4 ranks + that run's overhead",
                         "b0_cut_ttt_half": f"task time x (1 - {TTT_SHARE / 2:.2f}): halving test-time training sequences "
                                            f"(survey candidate #3: 64 vs 128 sequences) saves half of the TTT share of task time, "
                                            f"about 190 of 626 s ({TTT_SHARE:.0%}) per task in the audit log "
                                            f"(experiments/arc2-pool-audit/DESIGN.md, W1 cost and scaling); TTT time assumed linear "
                                            f"in sequences; accuracy effect assumed 0 (survey: 28.06 vs 27.22, single LB draws)",
                         "q27_time": f"start-up {swap:.0f} s ({swap_src}) + routed share x hidden tasks x generated tokens per routed "
                                     f"task / {tps:.0f} tok/s ({tps_src})",
                         "q27_cap": "a routed task's 27B answer counts under a per-task cap only if the task finished within it, or "
                                    "its first verifying submit came within it; tokens charged min(task tokens, cap)",
                         "sequential": "B0 runs first on all tasks, then 27B on the routed ones (same GPUs)",
                     }}
    rep["uncertainty"] = (
        f"B0 alone scores {', '.join(f'{n} {rep['b0'][n]['pass2']}' for n in names)} of {n_out} on identical runs (run-to-run "
        "variation >= 4 outputs); 27B was run once and varies too (temperature 1.0). The best rule is chosen on the same "
        "65 outputs it is scored on, so its gain is optimistic; differences of a few outputs between rules are within noise. "
        "Dev tasks have 1.63 test outputs per task against about 1.08 in the hidden set, and public evaluation tasks may be "
        "in the 27B model's training data.")
    rep["decision"] = _decision(rep, best, wall_target_h)
    return rep


def _challenges_for(task_ids: list[str], q27_dir: Path | None = None) -> tuple[dict | None, str]:
    """Demonstration pairs for the size-rule signal: Kaggle's challenges as the run copied them, else the local
    ARC-AGI-2 copy (3 dev tasks differ from Kaggle's file; a missing test input disables the signal there)."""
    for p in ([q27_dir / KAGGLE_CHALLENGES_COPY] if q27_dir else []) + [ROOT / "eval" / "data" / "arc-agi_evaluation-challenges.json"]:
        if p.is_file():
            ch = json.loads(p.read_text(encoding="utf-8"))
            return {t: ch[t] for t in task_ids if t in ch}, str(p)
    return None, "none"


def _count(it) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _decision(rep: dict, best: dict | None, wall_target_h: float) -> dict[str, Any]:
    n_new = rep["overlap"]["q27_new_vs_both_b0_runs"]
    ok_new = n_new >= NEW_OUTPUT_THRESHOLD
    ok_time = best is not None
    text = (f"27B solves {n_new} outputs B0 missed in both audit runs (need >= {NEW_OUTPUT_THRESHOLD}); "
            + (f"best rule within {wall_target_h:g} h: {_rule_name(best)}, mean pass@2 {best['mean_pass2']} "
               f"(B0 alone {rep['hybrid']['baseline']['mean_pass2']}), projected {best['max_total_h']} h"
               if ok_time else f"no rule projects within {wall_target_h:g} h"))
    verdict = "build the hybrid submission notebook (stage 2)" if ok_new and ok_time else "do not build the hybrid"
    return {"stage2": ok_new and ok_time, "new_outputs_ok": ok_new, "time_ok": ok_time, "text": text, "verdict": verdict}


def _rule_name(r: dict) -> str:
    th = f"<{r['theta']:g}" if r.get("theta") is not None else ""
    cap = "no cap" if r["cap"] is None else f"cap {r['cap']}"
    return f"route {r['rule']}{th}, merge {r['merge']}, 27B {cap}, B0 cut {r['b0_cut']}"


def render(rep: dict, per_output: bool = False) -> str:
    q, o = rep["q27"], rep["overlap"]
    n = rep["n_outputs"]
    names = list(rep["b0"])
    L = [f"q27 run: {rep['q27_dir']}  (tasks with a result file: {rep['q27_tasks_present']}/{rep['n_tasks']})",
         f"ground truth: {rep['ground_truth']}",
         "",
         f"27B pass@2: {q['pass2']}/{n}   pass@1: {q['pass1']}/{n}   tasks verified: {q['tasks_verified']}/{rep['n_tasks']}",
         f"attempt_1 train-verified on {q['attempt1_verified']} outputs, correct {q['attempt1_verified_correct']}; "
         f"unverified attempt_1 correct {q['attempt1_unverified_correct']}",
         f"stop reasons: {q['stop_reasons']}; server option set {q['server_option_set']}; hard stop {q['hard_stop']}",
         f"tokens: generated {q['generated_tokens']}, prompt {q['prompt_tokens']}; agent wall {q['agent_wall_s']} s; "
         f"{q['gen_tok_per_s']} generated tok/s ({q['gen_tok_per_s_source']}); start-up {q['startup_s']:.0f} s ({q['startup_source']})",
         ""]
    for nm in names:
        b = rep["b0"][nm]
        L.append(f"B0 {nm}: pass@2 {b['pass2']}/{n}, pass@1 {b['pass1']}; mean task {b['mean_task_s']} s, overhead "
                 f"{b['overhead_s']} s -> {b['projected_h']} h for the hidden set; tasks with an output of < 2 candidates "
                 f"{b['lt2_tasks']}; attempt 1 breaks the size rule on {b['size_rule_breaks']} outputs")
    L.append("")
    L.append("overlap (outputs of %d): " % n + ", ".join(f"{k} {v}" for k, v in o.items() if not k.endswith("_ids")
                                                         and k != "q27_new_verified"))
    L.append(f"27B solves, B0 missed in both runs ({o['q27_new_vs_both_b0_runs']}): {o['q27_new_vs_both_b0_runs_ids']}; "
             f"of these with a train-verified attempt_1: {o['q27_new_verified']}")
    L.append("")
    L.append("per task (cheap-first order): task  stop  verified  gen/prompt tokens  turns  wall s  | B0 task s per run")
    for r in rep["per_task"]:
        b0s = "  ".join(f"{r.get(f'b0_{nm}_s')}" for nm in names)
        L.append(f"  {r['task_id']} {str(r['order']):>4} {r['stop']:13s} {str(r['verified']):5s} {r['gen']:7d}/{r['prompt']:8d} "
                 f"{r['turns']:4d} {r['wall_s']:8.1f} | {b0s}")
    h = rep["hybrid"]
    L.append("")
    L.append(f"hybrid simulation: {h['n_rows']} combinations; margins tried {h['thetas']}; 27B per-task caps {h['caps']}")
    for k, v in h["assumptions"].items():
        L.append(f"  assumption {k}: {v}")
    bl = h["baseline"]
    L.append(f"  B0 alone: " + ", ".join(f"{nm} {bl[nm]['pass2']} ({bl[nm]['total_h']} h)" for nm in names))
    L.append("  best fitting combinations (mean pass@2 over the B0 runs; per run pass@2 / routed tasks / projected h):")
    for r in h["top_fitting"]:
        L.append(f"    {_rule_name(r):75s} mean {r['mean_pass2']:5.1f}  " +
                 "  ".join(f"{nm} {r[nm]['pass2']}/{r[nm]['routed']}/{r[nm]['total_h']}h" for nm in names))
    if not h["top_fitting"]:
        L.append("    none fits")
    ba = h["best_ignoring_time"]
    L.append(f"  best ignoring time: {_rule_name(ba)} mean {ba['mean_pass2']}, max projected {ba['max_total_h']} h")
    L.append("")
    L.append("uncertainty: " + rep["uncertainty"])
    L.append("")
    d = rep["decision"]
    L.append(f"decision (stage 2): {d['text']}: {d['verdict']}")
    if per_output:
        L.append("")
        L.append("per output: output  27B  27B-verified  " + "  ".join(f"B0-{nm}" for nm in names))
        for r in rep["per_output"]:
            L.append(f"  {r['output']:12s} {int(r['q27'])}  {int(r['q27_attempt1_verified'])}  " +
                     "  ".join(f"{int(r[f'b0_{nm}'])} (n {r[f'b0_{nm}_n_unique']}, m {r[f'b0_{nm}_margin']})" for nm in names))
    return "\n".join(L)


def _parse_caps(text: str) -> list[int | None]:
    return [None if c.strip() in ("full", "none", "") else int(c) for c in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q27", required=True, help="the run's q27/ directory")
    ap.add_argument("--b0", action="append", default=[], help="NAME=POOL_DIR (repeatable; default v1 and v2 audit pools)")
    ap.add_argument("--solutions", default="")
    ap.add_argument("--hidden-tasks", type=int, default=HIDDEN_TASKS)
    ap.add_argument("--wall-target-h", type=float, default=WALL_TARGET_H)
    ap.add_argument("--swap-overhead-s", type=float, default=None, help="27B start-up in the hybrid (default: measured)")
    ap.add_argument("--tok-per-s", type=float, default=None, help="27B generated tokens per second (default: measured)")
    ap.add_argument("--caps", default="full,32768,24576,16384", help="27B per-task generated-token caps to simulate")
    ap.add_argument("--json", default="")
    ap.add_argument("--per-output", action="store_true")
    args = ap.parse_args(argv)
    b0 = dict(x.split("=", 1) for x in args.b0) if args.b0 else DEFAULT_B0
    b0_dirs = {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in b0.items()}
    rep = analyze(Path(args.q27), b0_dirs, args.solutions, args.hidden_tasks, args.wall_target_h, args.swap_overhead_s,
                  args.tok_per_s, _parse_caps(args.caps))
    print(render(rep, args.per_output))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
