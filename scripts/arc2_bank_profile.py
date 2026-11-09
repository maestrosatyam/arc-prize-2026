#!/usr/bin/env python3
"""Profile the ARC-AGI-2 output bank (eval/arc2_bank.py): E2 breakdown, E5 contamination probe, E13 throughput model.
CPU only, no GPU, no network.

E2   per rule: the score (outputs right / outputs) by task-length quartile (quartiles_by_size: q1 = the shortest
     quarter, the hybrid's band) and by outputs per task (1 against 2 or more); the whole-bank official score and
     task_score, the hidden-mix proxy (the hidden set has about 1.08 test outputs per task against 1.43 in the public
     set, so a task-weighted number is closer to the board's mix); the reweighted score (each output weighted by
     1 / outputs of its task) equals task_score and is reported as a check.
E5   per rule: the score on the bank tasks that overlap ARC-AGI-1 (ARC1_OVERLAP, 7 of the 120) against the rest, with
     counts; coverage (rule oracle) on the same split.
E13  per-task elapsed_s against task size (characters of the task JSON) and test outputs: ordinary least squares,
     elapsed = a + b size and elapsed = a + b size + c n_outputs; the stop reasons and DFS cuts; then a projection of
     the B0 phase on the hidden set: task sizes drawn with replacement from the public size distribution, test-output
     counts drawn so the mean is --hidden-outputs-per-task, each task's time from the fit clipped to the per-task cap,
     the ranks taking tasks greedily in route order (shortest first); the wall, its spread over resamples, the margin
     against the hybrid's phase end (9 h 18 m = 33480 s) and against B0's own end (12 h - 10 min = 42600 s), and the
     fraction of tasks each budget would cut.

Usage: python scripts/arc2_bank_profile.py [--rules b0 v7_direct ...] [--json experiments/arc2-bank/profile.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (ARC1_OVERLAP, Q27_BANK, Bank, Result, load_bank, mean_sd, percentile,  # noqa: E402
                            quartiles_by_size, replay)

SCHEMA = "arc2_bank_profile.v1"
DEFAULT_JSON = ROOT / "experiments" / "arc2-bank" / "profile.json"
DEFAULT_RULES = ("b0", "v7_direct", "q27_first", "q27_agree", "oracle")
PHASE_END_S = 33480          # the hybrid's B0 phase end: 9 h 18 m after notebook start
B0_END_S = 42600             # B0 alone: the 12 h kernel limit minus 10 min
PER_TASK_CAP_S = 1200
STARTUP_S = 340              # B0's start-up to the first finished task (the v7 commit run's first_task_done_s, 337 s)
HIDDEN_TASKS = 240
HIDDEN_OUTPUTS_PER_TASK = 1.08
PUBLIC_OUTPUTS_PER_TASK = 1.43


# ----------------------------------------------------------------------------
# E2 breakdown

def group_stats(result: Result, task_ids) -> dict:
    """Official score, task_score and the 1/outputs reweighted score of a rule on a set of tasks."""
    ids = set(task_ids)
    by_task: dict[str, list[bool]] = {}
    for v in result.per_output.values():
        if v["task_id"] in ids:
            by_task.setdefault(v["task_id"], []).append(bool(v["right"]))
    n_out = sum(len(h) for h in by_task.values())
    n_right = sum(sum(h) for h in by_task.values())
    n_tasks = len(by_task)
    task_score = (sum(sum(h) / len(h) for h in by_task.values()) / n_tasks) if n_tasks else None
    reweighted = (sum(r / len(h) for h in by_task.values() for r in h) / n_tasks) if n_tasks else None
    return {"n_tasks": n_tasks, "n_outputs": n_out, "n_right": n_right,
            "score": round(n_right / n_out, 4) if n_out else None,
            "task_score": round(task_score, 4) if task_score is not None else None,
            "reweighted": round(reweighted, 4) if reweighted is not None else None}


def groups_of(bank: Bank) -> dict[str, list[str]]:
    """The E2 groups: the whole bank, the four length quartiles, tasks with one test output, tasks with two or more."""
    g = {"all": bank.task_ids}
    g.update(quartiles_by_size(bank))
    g["outputs_1"] = [t for t in bank.task_ids if len(bank.tasks[t].outputs) == 1]
    g["outputs_2plus"] = [t for t in bank.task_ids if len(bank.tasks[t].outputs) >= 2]
    return g


def breakdown(bank: Bank, results: dict[str, Result]) -> dict:
    groups = groups_of(bank)
    out = {}
    for name, ids in groups.items():
        n_out = sum(len(bank.tasks[t].outputs) for t in ids)
        out[name] = {"n_tasks": len(ids), "n_outputs": n_out,
                     "outputs_per_task": round(n_out / len(ids), 3) if ids else None,
                     "rules": {r: group_stats(res, ids) for r, res in results.items()}}
    return {"note": "score = outputs right / outputs (official); task_score = mean over tasks of the fraction right, "
                    f"the hidden-mix proxy (hidden set about {HIDDEN_OUTPUTS_PER_TASK} outputs per task, public "
                    f"{PUBLIC_OUTPUTS_PER_TASK}); reweighted = each output weighted 1/outputs of its task = task_score",
            "quartiles": "by task JSON length over the 120 public tasks; q1 = the shortest quarter (the hybrid's band)",
            "groups": out}


# ----------------------------------------------------------------------------
# E5 contamination probe

def contamination(bank: Bank, results: dict[str, Result], coverage: Result) -> dict:
    overlap = [t for t in ARC1_OVERLAP if t in bank.tasks]
    rest = [t for t in bank.task_ids if t not in set(ARC1_OVERLAP)]
    splits = {}
    for name, ids in (("overlap", overlap), ("rest", rest)):
        splits[name] = {"n_tasks": len(ids), "n_outputs": sum(len(bank.tasks[t].outputs) for t in ids),
                        "rules": {r: group_stats(res, ids) for r, res in results.items()},
                        "coverage": group_stats(coverage, ids)}
    return {"note": "ARC-AGI-1 overlap (splits/holdout_policy.md); coverage = rule oracle (a correct grid anywhere in "
                    "the pool or the 27B samples)",
            "overlap_ids_all": list(ARC1_OVERLAP), "overlap_ids_in_bank": overlap,
            "n_overlap_in_bank": len(overlap), "n_overlap_all": len(ARC1_OVERLAP), "splits": splits}


# ----------------------------------------------------------------------------
# E13 throughput model

def fit_ols(xs: list[list[float]], y: list[float], names: list[str]) -> dict:
    """Ordinary least squares with an intercept: coefficients, residual SD (n - p degrees of freedom), R^2."""
    X = np.array([[1.0] + [float(v) for v in r] for r in xs], dtype=float)
    Y = np.array(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ssr = float(resid @ resid)
    sst = float(((Y - Y.mean()) ** 2).sum()) if len(Y) else 0.0
    dof = len(Y) - X.shape[1]
    return {"terms": ["intercept"] + list(names), "coef": [float(b) for b in beta], "n": int(len(Y)),
            "resid_sd": round(math.sqrt(ssr / dof), 3) if dof > 0 else None,
            "r2": round(1.0 - ssr / sst, 4) if sst > 0 else None}


def predict(model: dict, size: float, n_outputs: float) -> float:
    c = model["coef"]
    return c[0] + c[1] * size + (c[2] * n_outputs if len(c) > 2 else 0.0)


def run_rows(bank: Bank) -> list[dict]:
    rows = []
    for t in bank.task_ids:
        task = bank.tasks[t]
        if task.elapsed_s is None or task.size is None:
            continue
        r = task.run
        rows.append({"task_id": t, "size": task.size, "n_outputs": len(task.outputs), "elapsed_s": task.elapsed_s,
                     "stop_reason": task.stop_reason, "rank": r.get("rank"),
                     "batches_total": r.get("batches_total"), "batches_decoded": r.get("batches_decoded"),
                     "dfs_batches_cut_540": int(r.get("dfs_batches_cut_540") or 0),
                     "dfs_batches_cut_end_time": int(r.get("dfs_batches_cut_end_time") or 0),
                     "per_task_cap_s": r.get("per_task_cap_s"), "dfs_cap_s": r.get("dfs_cap_s")})
    return rows


def simulate_ranks(times: list[float], ranks: int, budget_work_s: float) -> dict:
    """Greedy: the next task in order goes to the rank that is free first. A task is cut when its rank's cumulative
    time exceeds the budget for task work; not started when the rank is already past it."""
    free = [0.0] * max(1, ranks)
    cut = not_started = 0
    for t in times:
        i = min(range(len(free)), key=lambda k: (free[k], k))
        start = free[i]
        free[i] = start + t
        if free[i] > budget_work_s:
            cut += 1
            if start >= budget_work_s:
                not_started += 1
    return {"makespan_s": max(free), "cut": cut, "not_started": not_started}


def draw_outputs(rng: random.Random, mean: float) -> int:
    """floor(mean) outputs, one more with probability mean - floor(mean) (1.08: one output w.p. 0.92, else two)."""
    base = int(math.floor(mean))
    return base + (1 if rng.random() < mean - base else 0)


def project(sizes: list[int], model: dict, *, n_tasks: int = HIDDEN_TASKS, outputs_per_task: float = HIDDEN_OUTPUTS_PER_TASK,
            ranks: int = 4, overhead_s: float = 600.0, startup_s: float = STARTUP_S, cap_s: float = PER_TASK_CAP_S,
            budgets: tuple[float, ...] = (PHASE_END_S, B0_END_S), n_resample: int = 200, seed: int = 0) -> dict:
    """The B0 phase on a hidden-sized set: sizes drawn with replacement from ``sizes``, outputs per task drawn around
    ``outputs_per_task``, each task's time predicted by ``model`` and clipped to [0, cap]; wall = sum / ranks +
    start-up + overhead; the cut fraction at each budget from the greedy rank simulation in route order."""
    rng = random.Random(seed)
    sizes = sorted(sizes)
    walls, makespans, sums, at_cap = [], [], [], []
    per_budget: dict[float, dict[str, list[float]]] = {b: {"cut": [], "not_started": []} for b in budgets}
    for _ in range(n_resample):
        drawn = [(sizes[rng.randrange(len(sizes))], draw_outputs(rng, outputs_per_task)) for _ in range(n_tasks)]
        drawn.sort(key=lambda p: p[0])                       # route order: shortest task JSON first
        raw = [predict(model, s, n) for s, n in drawn]
        times = [min(max(t, 0.0), cap_s) for t in raw]
        total = sum(times)
        sums.append(total)
        walls.append(total / ranks + startup_s + overhead_s)
        at_cap.append(sum(1 for t in raw if t >= cap_s))
        makespans.append(simulate_ranks(times, ranks, math.inf)["makespan_s"] + startup_s + overhead_s)
        for b in budgets:
            sim = simulate_ranks(times, ranks, b - startup_s - overhead_s)
            per_budget[b]["cut"].append(sim["cut"] / n_tasks if n_tasks else 0.0)
            per_budget[b]["not_started"].append(sim["not_started"] / n_tasks if n_tasks else 0.0)

    def spread(v: list[float]) -> dict:
        m, sd = mean_sd(v)
        return {"mean": round(m, 1), "sd": round(sd, 1), "p10": round(percentile(v, 0.10), 1),
                "p50": round(percentile(v, 0.50), 1), "p90": round(percentile(v, 0.90), 1), "max": round(max(v), 1)}

    wall = spread(walls)
    resid_wall_sd = (model["resid_sd"] * math.sqrt(n_tasks) / ranks) if model.get("resid_sd") else None
    out_b = {}
    for b in budgets:
        cut, ns = per_budget[b]["cut"], per_budget[b]["not_started"]
        out_b[str(int(b))] = {"budget_s": b, "margin_mean_s": round(b - wall["mean"], 1),
                              "margin_p90_s": round(b - wall["p90"], 1),
                              "fits_mean": bool(b >= wall["mean"]), "fits_p90": bool(b >= wall["p90"]),
                              "cut_fraction_mean": round(sum(cut) / len(cut), 4),
                              "cut_fraction_p90": round(percentile(cut, 0.90), 4),
                              "cut_tasks_mean": round(sum(cut) / len(cut) * n_tasks, 1),
                              "not_started_fraction_mean": round(sum(ns) / len(ns), 4)}
    return {"n_tasks": n_tasks, "outputs_per_task": outputs_per_task, "ranks": ranks, "startup_s": startup_s,
            "overhead_s": overhead_s, "cap_s": cap_s, "n_resample": n_resample, "seed": seed,
            "n_public_sizes": len(sizes), "model": {"terms": model["terms"], "coef": model["coef"]},
            "sum_task_s": spread(sums), "wall_s": wall, "wall_greedy_s": spread(makespans),
            "resid_wall_sd_s": round(resid_wall_sd, 1) if resid_wall_sd is not None else None,
            "tasks_at_cap_mean": round(sum(at_cap) / len(at_cap), 2), "budgets": out_b}


def reading(proj: dict, phase_end_s: float, b0_end_s: float) -> str:
    w = proj["wall_s"]
    p = proj["budgets"][str(int(phase_end_s))]
    q = proj["budgets"][str(int(b0_end_s))]
    short = 100.0 * max(0.0, w["mean"] - phase_end_s) / phase_end_s
    verdict = ("fits at p90 with no task cut: on this model the cut was not needed" if p["fits_p90"] and p["cut_fraction_mean"] == 0 else
               "fits on the mean but not at p90: a small margin, not a cut, was needed" if p["fits_mean"] else
               f"does not fit by {short:.0f}% of the phase: the phase end would cut about {p['cut_tasks_mean']:.0f} of "
               f"{proj['n_tasks']} tasks ({100 * p['cut_fraction_mean']:.1f}%, the longest in route order)")
    return (f"Does full B0 (64 sequences, all 16 views) fit in the hybrid's 9 h 18 m on {proj['n_tasks']} tasks? "
            f"Predicted wall {w['mean'] / 3600:.2f} h (p10-p90 {w['p10'] / 3600:.2f}-{w['p90'] / 3600:.2f} h) against "
            f"{phase_end_s / 3600:.2f} h: margin {p['margin_mean_s'] / 60:+.0f} min (p90 {p['margin_p90_s'] / 60:+.0f} min); "
            f"{verdict}. Against B0's own {b0_end_s / 3600:.2f} h: margin {q['margin_mean_s'] / 60:+.0f} min, cut "
            f"fraction {q['cut_fraction_mean']:.3f}. v6nb cut B0 to a quarter of its training and half its views to make "
            f"room and lost about 3 points; the shortfall above is the room the model says was needed.")


def throughput(bank: Bank, *, hidden_tasks: int, hidden_outputs_per_task: float, ranks: int, overhead_s: float,
               startup_s: float, phase_end_s: float, b0_end_s: float, n_resample: int, seed: int) -> dict:
    rows = run_rows(bank)
    el = [r["elapsed_s"] for r in rows]
    m, sd = mean_sd(el)
    stop = {}
    for r in rows:
        stop[str(r["stop_reason"])] = stop.get(str(r["stop_reason"]), 0) + 1
    cap_s = next((float(r["per_task_cap_s"]) for r in rows if r.get("per_task_cap_s")), float(PER_TASK_CAP_S))
    fit_size = fit_ols([[r["size"]] for r in rows], el, ["size"]) if len(rows) >= 3 else None
    fit_both = fit_ols([[r["size"], r["n_outputs"]] for r in rows], el, ["size", "n_outputs"]) if len(rows) >= 4 else None
    capped = [r for r in rows if r["stop_reason"] == "per_task_cap"]
    cap = {"cap_s": cap_s, "public_n": len(rows), "public_hit_cap": len(capped),
           "public_hit_cap_rate": round(len(capped) / len(rows), 4) if rows else None,
           "public_elapsed_ge_cap": sum(1 for r in rows if r["elapsed_s"] >= cap_s),
           "capped_elapsed_mean_s": round(sum(r["elapsed_s"] for r in capped) / len(capped), 1) if capped else None,
           "capped_elapsed_max_s": max((r["elapsed_s"] for r in capped), default=None),
           "capped_n_outputs": [r["n_outputs"] for r in capped],
           "predicted_hidden_by_rate": round(len(capped) / len(rows) * hidden_tasks, 1) if rows else None}
    sizes = list(bank.sizes_all.values()) or [r["size"] for r in rows]
    proj = proj_size = None
    if fit_both is not None:
        kw = dict(n_tasks=hidden_tasks, outputs_per_task=hidden_outputs_per_task, ranks=ranks, overhead_s=overhead_s,
                  startup_s=startup_s, cap_s=cap_s, budgets=(phase_end_s, b0_end_s), n_resample=n_resample, seed=seed)
        proj = project(sizes, fit_both, **kw)
        proj_size = project(sizes, fit_size, **kw)
        cap["predicted_hidden_by_model"] = proj["tasks_at_cap_mean"]
    return {"note": "per-task elapsed_s of the bank run against task size (characters of the task JSON) and test "
                    "outputs; OLS; the projection draws hidden-set sizes from the public size distribution",
            "rows": rows,
            "elapsed_s": {"n": len(el), "mean": round(m, 1), "sd": round(sd, 1), "p25": round(percentile(el, 0.25), 1),
                          "p50": round(percentile(el, 0.50), 1), "p75": round(percentile(el, 0.75), 1),
                          "max": max(el) if el else None, "sum": round(sum(el), 1)},
            "stop_reason": stop,
            "dfs_cuts": {"tasks_cut_540": sum(1 for r in rows if r["dfs_batches_cut_540"] > 0),
                         "tasks_cut_end_time": sum(1 for r in rows if r["dfs_batches_cut_end_time"] > 0),
                         "tasks_any_cut": sum(1 for r in rows if r["dfs_batches_cut_540"] > 0 or r["dfs_batches_cut_end_time"] > 0)},
            "fit": {"size": fit_size, "size_outputs": fit_both},
            "cap": cap,
            "projection": proj, "projection_size_only": proj_size,
            "reading": reading(proj, phase_end_s, b0_end_s) if proj else None,
            "limits": ["public task lengths stand in for the hidden ones (sizes drawn with replacement from the "
                       f"{len(sizes)} public tasks); the hidden lengths are unknown",
                       "timing is the bank run's on L4 x4 (4 ranks, one task per rank, 64 sequences, DFS cap 540 s, "
                       "pool export on); another machine or build shifts the coefficients",
                       "the hybrid's phase 2 (the 27B) is not modelled; --overhead-s stands for everything outside "
                       "B0's task loop after start-up",
                       "the fit is linear in size with a residual SD of a few hundred seconds; the projection's spread "
                       "is from resampling sizes only (resid_wall_sd_s is the residual's own contribution to the wall)",
                       "the per-task cap stops the loop between batches, so capped public tasks ran past 1200 s; "
                       "the model clips at the cap"]}


# ----------------------------------------------------------------------------
# summary

def _cell(s: dict) -> str:
    return f"{s['n_right']}/{s['n_outputs']} {s['score']:.3f}" if s["score"] is not None else "-"


def summary_lines(res: dict) -> list[str]:
    rules = list(res["rules"])
    b = res["bank"]
    lines = [f"bank {b['fingerprint']}: {b['n_tasks']} tasks, {b['n_outputs']} outputs, {b['n_tasks_all']} public tasks sized; "
             f"rules {' '.join(rules)}",
             "E2 breakdown (right/outputs score; task_score rows = hidden-mix proxy)",
             "| group | tasks | outputs | out/task | " + " | ".join(rules) + " |"]
    for name, g in res["breakdown"]["groups"].items():
        lines.append(f"| {name} | {g['n_tasks']} | {g['n_outputs']} | {g['outputs_per_task']} | "
                     + " | ".join(_cell(g["rules"][r]) for r in rules) + " |")
        lines.append(f"| {name} task_score | | | | "
                     + " | ".join(f"{g['rules'][r]['task_score']:.3f}" if g["rules"][r]["task_score"] is not None else "-" for r in rules) + " |")
    c = res["contamination"]
    lines.append(f"E5 contamination probe: ARC-AGI-1 overlap {c['n_overlap_in_bank']} of {c['n_overlap_all']} in the bank "
                 f"({' '.join(c['overlap_ids_in_bank'])})")
    lines.append("| split | tasks | outputs | " + " | ".join(rules) + " | coverage |")
    for name, s in c["splits"].items():
        lines.append(f"| {name} | {s['n_tasks']} | {s['n_outputs']} | " + " | ".join(_cell(s["rules"][r]) for r in rules)
                     + f" | {_cell(s['coverage'])} |")
    t = res["throughput"]
    e = t["elapsed_s"]
    lines.append(f"E13 throughput: elapsed_s n {e['n']} mean {e['mean']} p25 {e['p25']} p50 {e['p50']} p75 {e['p75']} max {e['max']}; "
                 f"stop_reason {t['stop_reason']}; DFS cuts (540 s / end time / any) {t['dfs_cuts']['tasks_cut_540']} / "
                 f"{t['dfs_cuts']['tasks_cut_end_time']} / {t['dfs_cuts']['tasks_any_cut']}")
    lines.append("| model | a (s) | b (s/char) | c (s/output) | resid sd (s) | R^2 |")
    for name, f in t["fit"].items():
        if f is None:
            continue
        co = f["coef"]
        c_cell = f"{co[2]:.1f}" if len(co) > 2 else "-"
        lines.append(f"| {name} | {co[0]:.1f} | {co[1]:.5f} | {c_cell} | {f['resid_sd']} | {f['r2']} |")
    cap = t["cap"]
    lines.append(f"per-task cap {cap['cap_s']:.0f} s: public {cap['public_hit_cap']}/{cap['public_n']} stopped by the cap "
                 f"({cap['public_elapsed_ge_cap']} ran >= cap; capped tasks mean {cap['capped_elapsed_mean_s']} s, max "
                 f"{cap['capped_elapsed_max_s']} s); predicted on {res['arguments']['hidden_tasks']} tasks: "
                 f"{cap['predicted_hidden_by_rate']} by rate, {cap.get('predicted_hidden_by_model')} by model")
    p = t["projection"]
    if p:
        w = p["wall_s"]
        lines.append(f"projection: {p['n_tasks']} tasks, {p['outputs_per_task']} outputs/task, {p['ranks']} ranks, start-up "
                     f"{p['startup_s']:.0f} s, overhead {p['overhead_s']:.0f} s, {p['n_resample']} resamples, seed {p['seed']}; "
                     f"model size_outputs; wall mean {w['mean']:.0f} s = {w['mean'] / 3600:.2f} h (p10-p90 {w['p10']:.0f}-{w['p90']:.0f} s), "
                     f"greedy makespan mean {p['wall_greedy_s']['mean']:.0f} s; residual-implied wall sd {p['resid_wall_sd_s']} s; "
                     f"size-only model wall mean {t['projection_size_only']['wall_s']['mean']:.0f} s")
        lines.append("| budget (s) | h | margin mean (min) | margin p90 (min) | cut fraction mean | cut fraction p90 | cut tasks mean | not started |")
        for k, v in p["budgets"].items():
            lines.append(f"| {k} | {v['budget_s'] / 3600:.2f} | {v['margin_mean_s'] / 60:+.0f} | {v['margin_p90_s'] / 60:+.0f} | "
                         f"{v['cut_fraction_mean']:.3f} | {v['cut_fraction_p90']:.3f} | {v['cut_tasks_mean']} | {v['not_started_fraction_mean']:.3f} |")
        lines.append("reading: " + t["reading"])
    lines.append("limits: " + "; ".join(t["limits"]))
    return lines


# ----------------------------------------------------------------------------
# CLI

def profile(bank: Bank, rules: list[str], *, hidden_tasks: int = HIDDEN_TASKS,
            hidden_outputs_per_task: float = HIDDEN_OUTPUTS_PER_TASK, ranks: int = 4, overhead_s: float = 600.0,
            startup_s: float = STARTUP_S, phase_end_s: float = PHASE_END_S, b0_end_s: float = B0_END_S,
            n_resample: int = 200, seed: int = 0, arguments: dict | None = None) -> dict:
    results = {r: replay(bank, r) for r in rules}
    coverage = results.get("oracle") or replay(bank, "oracle")
    return {"schema": SCHEMA, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bank": {"fingerprint": bank.fingerprint(), "n_tasks": len(bank.tasks), "n_outputs": bank.n_outputs,
                     "n_tasks_all": len(bank.sizes_all), "n_q27_tasks": len(bank.q27), "sources": bank.sources},
            "arguments": dict(arguments or {}, hidden_tasks=hidden_tasks, hidden_outputs_per_task=hidden_outputs_per_task,
                              ranks=ranks, overhead_s=overhead_s, startup_s=startup_s, phase_end_s=phase_end_s,
                              b0_end_s=b0_end_s, resamples=n_resample, seed=seed),
            "rules": {r: res.summary() for r, res in results.items()},
            "breakdown": breakdown(bank, results),
            "contamination": contamination(bank, results, coverage),
            "throughput": throughput(bank, hidden_tasks=hidden_tasks, hidden_outputs_per_task=hidden_outputs_per_task,
                                     ranks=ranks, overhead_s=overhead_s, startup_s=startup_s, phase_end_s=phase_end_s,
                                     b0_end_s=b0_end_s, n_resample=n_resample, seed=seed)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    ap.add_argument("--pools", nargs="+", default=None, help="pool directories (default: both bank parts' pool/)")
    ap.add_argument("--q27", default="", help="the 27B bank directory (default: the q27 bank)")
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--hidden-tasks", type=int, default=HIDDEN_TASKS)
    ap.add_argument("--hidden-outputs-per-task", type=float, default=HIDDEN_OUTPUTS_PER_TASK)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--overhead-s", type=float, default=600.0)
    ap.add_argument("--startup-s", type=float, default=STARTUP_S, help="B0's start-up to the first finished task")
    ap.add_argument("--phase-end-s", type=float, default=PHASE_END_S, help="the hybrid's B0 phase end (9 h 18 m)")
    ap.add_argument("--b0-end-s", type=float, default=B0_END_S, help="B0 alone: 12 h - 10 min")
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    bank = load_bank(args.pools, args.q27 or Q27_BANK)
    res = profile(bank, args.rules, hidden_tasks=args.hidden_tasks, hidden_outputs_per_task=args.hidden_outputs_per_task,
                  ranks=args.ranks, overhead_s=args.overhead_s, startup_s=args.startup_s, phase_end_s=args.phase_end_s,
                  b0_end_s=args.b0_end_s, n_resample=args.resamples, seed=args.seed,
                  arguments={"rules": args.rules, "pools": args.pools, "q27": args.q27 or str(Q27_BANK)})
    for line in summary_lines(res):
        print(line)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
