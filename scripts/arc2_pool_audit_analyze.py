#!/usr/bin/env python3
"""Analyse the ARC-AGI-2 pool audit output (kaggle/arc2/audit; experiments/arc2-pool-audit/DESIGN.md).

Reads the pool directory the audit kernel writes (<task_id>.json per task, _run.json) and the ground truth, and
reports per test output and in total, always with denominators:

  selected pass@2   official metric (eval/arc2_score.py) on the attempts the notebook submitted
                    (pool "submitted"; "selected" when the file was not finalized). Tasks without a pool file
                    score 0 and stay in the denominator.
  coverage          the correct grid is anywhere in the candidate pool (oracle with unlimited attempts)
  top-k by rank     the correct grid is among the first k candidates of the notebook's own ranking
                    (score_kgmon, the one it submits from; k = 2 is the oracle best-2 under that ranking and
                    equals selected pass@2 unless the export and the submission disagree), and of the decoder's
                    other built-in ranking score_full_probmul_3
  rank of correct   kgmon / probmul3 rank of the correct candidate when it is in the pool
  pool size         unique candidates and beams per test output
  coverage gap      coverage - selected pass@2, in outputs (decision 1: selector work only if >= 3 outputs)

Budget use (second question: is the fixed per-task budget spent where it helps):
  stop reason       per task: per_task_cap / global_end_time / dfs_cap / exhausted; rank; task elapsed seconds
  time to correct   seconds from task start (test-time training included) until the first beam with the correct
                    grid was appended, and the same as a fraction of the task's time (resolution: one DFS batch)
  top-2 settled     share of tasks (and outputs) whose final kgmon top-2, as a set, is already the top-2 when only
                    the beams appended by 25 / 50 / 75 % of the task's time are counted
  margin AUC        score_kgmon(rank 1) - score_kgmon(rank 2) as a predictor of pass@2 and of pass@1 correctness
                    (Mann-Whitney AUC, outputs with >= 2 candidates; n_pos / n_neg given)
  decision 2        adaptive budget experiment only if margin AUC (pass@2) >= 0.8 and the top-2 is fixed by 50 %
                    of the time in >= 50 % of tasks

Ground truth, first found: --solutions (a {task: [grid, ...]} file or an ARC-AGI-2 data/evaluation directory);
third_party/ARC-AGI-2/data/evaluation in this checkout or in a parent checkout; eval/data/arc-agi_evaluation-solutions.json
(built from the same ARC-AGI-2 commit, see eval/data/MANIFEST.json). When the run copied Kaggle's solutions
(pool/_kaggle_solutions_audit_tasks.json) they are compared with the local ones.

Comparison (--compare BASELINE_POOL; --pool is the new run, e.g. W1 against the audit), on the same outputs and
ground truth: pass@2, pass@1, coverage and coverage gap side by side; outputs gained and lost (pass@2, coverage);
per-task elapsed ratio new/baseline (median, max); stop reasons (tasks at the 1200 s cap, cut by the global end
time); pool sizes; a full-run projection, mean task seconds x hidden tasks (240, compliance/sources.json) / 4 ranks
+ the run's measured overhead, and per 100 tasks; the W1 decision (DESIGN.md, section W1).

Usage: python scripts/arc2_pool_audit_analyze.py --pool /path/to/output/pool [--json out.json] [--per-output]
       python scripts/arc2_pool_audit_analyze.py --pool /path/to/w1/pool --compare /path/to/audit/pool [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc2.selection.metrics import output_metrics  # noqa: E402
from eval.arc2_score import grids_equal, score_submission  # noqa: E402

SPLITS = ROOT / "splits" / "arc2_eval_split_v1.json"
LOCAL_SOLUTIONS = ROOT / "eval" / "data" / "arc-agi_evaluation-solutions.json"
GAP_THRESHOLD = 3
TOP_K = (1, 2, 3, 5, 10)
PLACEHOLDER = [[0]]


# ----------------------------------------------------------------------------
# inputs

def _from_task_dir(d: Path) -> dict[str, list]:
    out = {}
    for p in sorted(d.glob("*.json")):
        task = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = [t["output"] for t in task["test"]]
    return out


def find_solutions(explicit: str | None = None) -> tuple[dict[str, list], str]:
    """(solutions {task_id: [grid per test output]}, description of the source)."""
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return _from_task_dir(p), f"{p} (ARC-AGI-2 per-task files)"
        return json.loads(p.read_text(encoding="utf-8")), f"{p}"
    for base in (ROOT, *ROOT.parents):
        d = base / "third_party" / "ARC-AGI-2" / "data" / "evaluation"
        if d.is_dir() and any(d.glob("*.json")):
            return _from_task_dir(d), f"{d} (ARC-AGI-2 repository, per-task files)"
    if LOCAL_SOLUTIONS.is_file():
        return json.loads(LOCAL_SOLUTIONS.read_text(encoding="utf-8")), f"{LOCAL_SOLUTIONS} (built from ARC-AGI-2, see MANIFEST.json)"
    raise FileNotFoundError("no ground truth found; pass --solutions")


def dev_task_ids() -> list[str]:
    return sorted(json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]["dev"])


def load_pool(pool_dir: Path) -> tuple[dict[str, dict], dict | None, dict | None]:
    """(task docs by id, _run.json or None, Kaggle solutions copy or None)."""
    docs = {}
    for p in sorted(pool_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        docs[doc.get("task_id", p.stem)] = doc
    run = pool_dir / "_run.json"
    kag = pool_dir / "_kaggle_solutions_audit_tasks.json"
    return (docs,
            json.loads(run.read_text(encoding="utf-8")) if run.is_file() else None,
            json.loads(kag.read_text(encoding="utf-8")) if kag.is_file() else None)


# ----------------------------------------------------------------------------
# analysis

def _attempts(out: dict) -> tuple[Any, Any, str]:
    sub = out.get("submitted")
    if isinstance(sub, dict):
        return sub.get("attempt_1"), sub.get("attempt_2"), "submitted"
    sel = list(out.get("selected", [])) + [PLACEHOLDER, PLACEHOLDER]
    return sel[0], sel[1], "selected"


def _rank_of(cands: list[dict], label, key: str) -> int | None:
    for c in cands:
        if grids_equal(c.get("grid"), label):
            return int(c[key])
    return None


def _dist(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    q = statistics.quantiles(s, n=4, method="inclusive") if len(s) > 1 else [s[0]] * 3
    return {"n": len(s), "min": s[0], "p25": q[0], "median": statistics.median(s), "mean": round(statistics.fmean(s), 2),
            "p75": q[2], "max": s[-1]}


def _buckets(values: list[int], edges=((0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, None))) -> dict[str, int]:
    out = {}
    for lo, hi in edges:
        name = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        out[name] = sum(1 for v in values if v >= lo and (hi is None or v <= hi))
    return out


def _rank_buckets(ranks: list[int]) -> dict[str, int]:
    return _buckets(ranks, ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, None)))


# ----------------------------------------------------------------------------
# second question: is the fixed per-task budget spent where it helps

def auc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """P(score of a positive > score of a negative), ties count 1/2 (Mann-Whitney); None if a class is empty."""
    if not scores_pos or not scores_neg:
        return None
    wins = 0.0
    for p in scores_pos:
        for q in scores_neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(scores_pos) * len(scores_neg))


def _kgmon_top2(cands: list[dict], until_s: float | None) -> list[tuple] | None:
    """Top two grids (as tuples) under score_kgmon using only beams appended by until_s (None: all beams).

    Per grid: number of its beams - mean of their augmentation means (arc_decoder.getter_kgmon); ties keep the
    final export order. None when a beam has no discovery time.
    """
    scored = []
    for c in cands:
        beams = c.get("beams", [])
        if until_s is not None:
            if any(b.get("elapsed_s") is None for b in beams):
                return None
            beams = [b for b in beams if b["elapsed_s"] <= until_s]
        if not beams:
            continue
        score = len(beams) - statistics.fmean(b.get("aug_mean", 0.0) for b in beams)
        scored.append((-score, c["rank_kgmon"], tuple(map(tuple, c["grid"]))))
    scored.sort()
    return [g for _, _, g in scored[:2]]


def budget_analysis(rows: list[dict], docs: dict[str, dict], task_ids: list[str],
                    fractions=(0.25, 0.5, 0.75)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # per-task budget use
    tasks = [docs[t] for t in task_ids if t in docs]
    out["stop_reason"] = dict(Counter((d.get("run") or {}).get("stop_reason", "unknown") for d in tasks))
    out["rank"] = dict(Counter(str((d.get("run") or {}).get("rank")) for d in tasks))
    out["task_elapsed_s"] = _dist([round((d.get("run") or {}).get("elapsed_s")) for d in tasks
                                   if isinstance((d.get("run") or {}).get("elapsed_s"), (int, float))])

    # time to the first correct candidate
    t_abs, t_frac = [], []
    for r in rows:
        if r.get("first_correct_s") is not None:
            t_abs.append(r["first_correct_s"])
            if r.get("task_elapsed_s"):
                t_frac.append(r["first_correct_s"] / r["task_elapsed_s"])
    n_in_pool = sum(1 for r in rows if r["in_pool"])
    out["time_to_first_correct_s"] = {"of_outputs_with_correct_in_pool": n_in_pool, "with_time": len(t_abs),
                                      "dist": _dist([round(x, 1) for x in t_abs])}
    out["time_to_first_correct_fraction"] = _dist([round(x, 3) for x in t_frac])

    # top-2 already fixed at a fraction of the task's time
    settle: dict[str, Any] = {}
    timed = []
    for d in tasks:
        total = (d.get("run") or {}).get("elapsed_s")
        outs = d.get("outputs", [])
        if not isinstance(total, (int, float)) or total <= 0 or not any(o.get("candidates") for o in outs):
            continue
        if any(_kgmon_top2(o.get("candidates", []), total) is None for o in outs):
            continue
        timed.append((d, total))
    for f in fractions:
        n_task, n_out, n_out_all = 0, 0, 0
        for d, total in timed:
            fixed_all = True
            for o in d["outputs"]:
                cands = o.get("candidates", [])
                if not cands:
                    continue
                n_out_all += 1
                fixed = set(_kgmon_top2(cands, f * total)) == set(_kgmon_top2(cands, None))
                n_out += fixed
                fixed_all = fixed_all and fixed
            n_task += fixed_all
        settle[f"{int(f * 100)}%"] = {"tasks": n_task, "of_tasks": len(timed),
                                      "share_tasks": (n_task / len(timed)) if timed else None,
                                      "outputs": n_out, "of_outputs": n_out_all}
    out["top2_fixed_by_time_fraction"] = settle
    out["tasks_without_timing_or_candidates"] = len(tasks) - len(timed)

    # does the kgmon margin (rank 1 minus rank 2) predict correctness
    with_margin = [r for r in rows if r.get("margin_kgmon") is not None]
    single = [r for r in rows if r["has_pool"] and r["n_unique"] == 1]
    out["margin_auc"] = {
        "pass_at_2": {"auc": auc([r["margin_kgmon"] for r in with_margin if r["selected_correct"]],
                                 [r["margin_kgmon"] for r in with_margin if not r["selected_correct"]]),
                      "n_pos": sum(1 for r in with_margin if r["selected_correct"]),
                      "n_neg": sum(1 for r in with_margin if not r["selected_correct"])},
        "pass_at_1": {"auc": auc([r["margin_kgmon"] for r in with_margin if r["correct_1"]],
                                 [r["margin_kgmon"] for r in with_margin if not r["correct_1"]]),
                      "n_pos": sum(1 for r in with_margin if r["correct_1"]),
                      "n_neg": sum(1 for r in with_margin if not r["correct_1"])},
        "outputs_with_margin": len(with_margin),
        "outputs_single_candidate": {"n": len(single), "correct": sum(1 for r in single if r["selected_correct"])},
        "outputs_empty_pool": sum(1 for r in rows if r["n_unique"] == 0),
    }
    return out


def analyze(pool_dir: Path, solutions: dict[str, list], task_ids: list[str] | None = None,
            gap_threshold: int = GAP_THRESHOLD, auc_threshold: float = 0.8,
            settle_threshold: float = 0.5) -> dict[str, Any]:
    docs, run, kaggle_copy = load_pool(pool_dir)
    if task_ids is None:
        task_ids = list(run["task_ids"]) if run and run.get("task_ids") else dev_task_ids()
    absent_truth = [t for t in task_ids if t not in solutions]
    if absent_truth:
        raise KeyError(f"ground truth lacks {absent_truth}")
    truth = {t: solutions[t] for t in task_ids}

    submission, rows = {}, []
    for tid in task_ids:
        doc = docs.get(tid)
        outs = {o["test_index"]: o for o in doc["outputs"]} if doc else {}
        if doc:
            submission[tid] = []
        for i, label in enumerate(truth[tid]):
            o = outs.get(i)
            cands = o.get("candidates", []) if o else []
            a1, a2, source = _attempts(o) if o else (None, None, "none")
            if doc:
                submission[tid].append({"attempt_1": a1, "attempt_2": a2})
            m = output_metrics([a1, a2], [c["grid"] for c in cands], label)
            rk = _rank_of(cands, label, "rank_kgmon")
            rp = _rank_of(cands, label, "rank_probmul3")
            correct_c = next((c for c in cands if grids_equal(c.get("grid"), label)), None)
            times = [b["elapsed_s"] for b in (correct_c or {}).get("beams", []) if b.get("elapsed_s") is not None]
            margin = o.get("margin_kgmon") if o else None
            if margin is None and len(cands) > 1:
                margin = cands[0]["score_kgmon"] - cands[1]["score_kgmon"]
            rows.append({
                "first_correct_s": min(times) if times else None,
                "task_elapsed_s": (doc.get("run") or {}).get("elapsed_s") if doc else None,
                "margin_kgmon": margin,
                "task_id": tid, "test_index": i, "has_pool": o is not None, "attempt_source": source,
                "n_unique": len(cands), "n_beams": int(o.get("n_beams", 0)) if o else 0,
                "in_pool": m["oracle"], "rank_kgmon": rk, "rank_probmul3": rp,
                "submitted_matches_selected": o.get("submitted_matches_selected") if o else None,
                "truncated": bool(doc and doc.get("run", {}).get("batches_decoded", 0) < doc.get("run", {}).get("batches_total", 0)),
            })
    official = score_submission(submission, truth)
    by_key = {(r["task_id"], r["test_index"]): r for r in official["per_output"]}
    for r in rows:
        off = by_key[(r["task_id"], r["test_index"])]
        r["selected_correct"] = off["correct"]
        r["correct_1"], r["correct_2"] = off["correct_1"], off["correct_2"]

    n = len(rows)
    n_pool = sum(r["has_pool"] for r in rows)
    n_sel = official["n_solved"]
    n_cov = sum(r["in_pool"] for r in rows)

    def topk(key: str) -> dict[str, int]:
        return {f"top{k}": sum(1 for r in rows if r[key] is not None and r[key] <= k) for k in TOP_K}

    gap = n_cov - n_sel
    missed = [f"{r['task_id']}_{r['test_index']} (kgmon rank {r['rank_kgmon']}, probmul3 rank {r['rank_probmul3']}, "
              f"pool {r['n_unique']})" for r in rows if r["in_pool"] and not r["selected_correct"]]
    report = {
        "pool_dir": str(pool_dir),
        "n_tasks": len(task_ids),
        "n_tasks_with_pool": sum(1 for t in task_ids if t in docs),
        "tasks_without_pool": [t for t in task_ids if t not in docs],
        "tasks_truncated": sorted({r["task_id"] for r in rows if r["truncated"]}),
        "n_outputs": n,
        "n_outputs_with_pool": n_pool,
        "selected_pass_at_2": {"n": n_sel, "of": n, "rate": n_sel / n, "official_score": official["score"]},
        "selected_pass_at_1": {"n": sum(r["correct_1"] for r in rows), "of": n},
        "coverage": {"n": n_cov, "of": n, "rate": n_cov / n},
        "coverage_gap": {"n": gap, "of": n, "rate": gap / n},
        "top_k_kgmon": topk("rank_kgmon"),
        "top_k_probmul3": topk("rank_probmul3"),
        "rank_of_correct_kgmon": _rank_buckets([r["rank_kgmon"] for r in rows if r["rank_kgmon"] is not None]),
        "rank_of_correct_probmul3": _rank_buckets([r["rank_probmul3"] for r in rows if r["rank_probmul3"] is not None]),
        "pool_size_unique": {"dist": _dist([r["n_unique"] for r in rows if r["has_pool"]]),
                             "buckets": _buckets([r["n_unique"] for r in rows if r["has_pool"]])},
        "pool_size_beams": {"dist": _dist([r["n_beams"] for r in rows if r["has_pool"]])},
        "in_pool_not_selected": missed,
        "submitted_selected_mismatches": [f"{r['task_id']}_{r['test_index']}" for r in rows if r["submitted_matches_selected"] is False],
        "attempt_sources": dict(Counter(r["attempt_source"] for r in rows)),
        "decision": {
            "threshold_outputs": gap_threshold,
            "selector_work": gap >= gap_threshold,
            "text": (f"coverage gap {gap} of {n} outputs >= {gap_threshold}: selector work is justified"
                     if gap >= gap_threshold else
                     f"coverage gap {gap} of {n} outputs < {gap_threshold}: stop ARC-AGI-2 method work, keep B0"),
        },
        "run": run.get("run") if run else None,
        "per_output": rows,
    }
    budget = budget_analysis(rows, docs, task_ids)
    a = budget["margin_auc"]["pass_at_2"]["auc"]
    s = budget["top2_fixed_by_time_fraction"].get("50%", {}).get("share_tasks")
    go = a is not None and s is not None and a >= auc_threshold and s >= settle_threshold
    budget["decision"] = {
        "auc_threshold": auc_threshold, "settle_threshold_share_tasks_at_50pct": settle_threshold,
        "adaptive_budget_experiment": go,
        "text": (f"margin AUC (pass@2) {a if a is None else round(a, 3)} "
                 f"(n_pos {budget['margin_auc']['pass_at_2']['n_pos']}, n_neg {budget['margin_auc']['pass_at_2']['n_neg']}); "
                 f"top-2 fixed by 50% of the task's time in {s if s is None else round(s, 3)} of tasks "
                 f"({budget['top2_fixed_by_time_fraction'].get('50%', {}).get('tasks')}/"
                 f"{budget['top2_fixed_by_time_fraction'].get('50%', {}).get('of_tasks')}): "
                 + ("an adaptive budget experiment follows" if go else
                    f"no adaptive budget experiment (needs AUC >= {auc_threshold} and share >= {settle_threshold})")),
    }
    report["budget"] = budget
    if kaggle_copy is not None:
        same = [t for t in task_ids if t in kaggle_copy and kaggle_copy[t] == truth[t]]
        report["kaggle_solutions_check"] = {"tasks_compared": sum(1 for t in task_ids if t in kaggle_copy),
                                            "identical": len(same),
                                            "differ": [t for t in task_ids if t in kaggle_copy and kaggle_copy[t] != truth[t]]}
    return report


def render(rep: dict[str, Any], solutions_source: str, per_output: bool = False) -> str:
    n = rep["n_outputs"]
    pct = lambda k: f"{k} / {n} ({100 * k / n:.1f}%)" if n else "0 / 0"  # noqa: E731
    lines = [
        f"pool audit: {rep['pool_dir']}",
        f"ground truth: {solutions_source}",
    ]
    if "kaggle_solutions_check" in rep:
        k = rep["kaggle_solutions_check"]
        lines.append(f"Kaggle solutions copy: {k['identical']}/{k['tasks_compared']} tasks identical to the local ground truth"
                     + (f"; differ {k['differ']}" if k["differ"] else ""))
    else:
        lines.append("Kaggle solutions copy: none in the pool directory (local ground truth only)")
    lines += [
        f"tasks: {rep['n_tasks_with_pool']} / {rep['n_tasks']} with a pool file; without: {rep['tasks_without_pool'] or 'none'}; "
        f"truncated by a time cap: {rep['tasks_truncated'] or 'none'}",
        f"test outputs: {n} (with a pool entry: {rep['n_outputs_with_pool']})",
        "",
        f"selected pass@2 (official metric): {pct(rep['selected_pass_at_2']['n'])}",
        f"selected pass@1:                   {pct(rep['selected_pass_at_1']['n'])}",
        f"coverage (correct anywhere):       {pct(rep['coverage']['n'])}",
        f"coverage gap (coverage - pass@2):  {pct(rep['coverage_gap']['n'])}",
        "",
        "correct within the first k candidates (denominator " + str(n) + "):",
        "  k          " + "  ".join(f"{k:>5}" for k in TOP_K),
        "  kgmon      " + "  ".join(f"{rep['top_k_kgmon'][f'top{k}']:>5}" for k in TOP_K) + "   (submitted ranking; k=2 is oracle best-2 by own ranking)",
        "  probmul3   " + "  ".join(f"{rep['top_k_probmul3'][f'top{k}']:>5}" for k in TOP_K),
        f"rank of correct when in pool (kgmon):    {rep['rank_of_correct_kgmon']}",
        f"rank of correct when in pool (probmul3): {rep['rank_of_correct_probmul3']}",
        f"pool size, unique candidates per output: {rep['pool_size_unique']['dist']}",
        f"  buckets: {rep['pool_size_unique']['buckets']}",
        f"pool size, beams per output: {rep['pool_size_beams']['dist']}",
        f"in pool but not selected ({len(rep['in_pool_not_selected'])}): {rep['in_pool_not_selected'] or 'none'}",
        f"submitted != exported selection: {rep['submitted_selected_mismatches'] or 'none'}; attempts from {rep['attempt_sources']}",
        "",
        f"decision 1 (selection): {rep['decision']['text']}",
    ]
    b = rep.get("budget")
    if b:
        ttc = b["time_to_first_correct_s"]
        lines += [
            "",
            "budget use:",
            f"  stop reason per task: {b['stop_reason']}; tasks per rank: {b['rank']}",
            f"  task elapsed s: {b['task_elapsed_s']}",
            f"  time to first correct candidate, s ({ttc['with_time']} of {ttc['of_outputs_with_correct_in_pool']} "
            f"outputs with the correct grid in the pool): {ttc['dist']}",
            f"  same, as a fraction of the task's time: {b['time_to_first_correct_fraction']}",
            "  final top-2 already fixed at a fraction of the task's time: "
            + "; ".join(f"{k}: {v['tasks']}/{v['of_tasks']} tasks, {v['outputs']}/{v['of_outputs']} outputs"
                        for k, v in b["top2_fixed_by_time_fraction"].items())
            + f" (tasks without timing or candidates: {b['tasks_without_timing_or_candidates']})",
            "  margin (kgmon rank 1 - rank 2) vs correctness: "
            + "; ".join(f"{k} AUC {v['auc'] if v['auc'] is None else round(v['auc'], 3)} (n_pos {v['n_pos']}, n_neg {v['n_neg']})"
                        for k, v in b["margin_auc"].items() if isinstance(v, dict) and "auc" in v)
            + f"; outputs with a margin {b['margin_auc']['outputs_with_margin']}, single-candidate "
              f"{b['margin_auc']['outputs_single_candidate']}, empty pool {b['margin_auc']['outputs_empty_pool']}",
            f"decision 2 (budget): {b['decision']['text']}",
        ]
    if per_output:
        lines.append("")
        lines.append("task_id   out  pool  beams  in_pool  rank_kg  rank_pm  selected")
        for r in rep["per_output"]:
            lines.append(f"{r['task_id']}  {r['test_index']:>3}  {r['n_unique']:>4}  {r['n_beams']:>5}  {str(r['in_pool']):>7}  "
                         f"{str(r['rank_kgmon']):>7}  {str(r['rank_probmul3']):>7}  {r['selected_correct']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# comparison of two runs on the same outputs (W1 vs the audit; DESIGN.md section W1)

HIDDEN_TASKS = 240
HIDDEN_TASKS_SOURCE = ("compliance/sources.json (competition data page: the hidden rerun uses 240 tasks; "
                       "the placeholder test file has 240 tasks, 259 test inputs)")
RANKS = 4
WALL_TARGET_H = 11.0
GAIN_THRESHOLD = 2
DEFAULT_OVERHEAD_S = 600.0


def _dfs_min_prob(docs: dict[str, dict], run: dict | None) -> float | None:
    v = ((run or {}).get("run") or {}).get("dfs_min_prob")
    if v is None:
        vals = {(d.get("run") or {}).get("dfs_min_prob") for d in docs.values()} - {None}
        v = vals.pop() if len(vals) == 1 else None
    return v


def _run_times(docs: dict[str, dict], run: dict | None, task_ids: list[str], ranks: int) -> dict[str, Any]:
    el = {t: float(docs[t]["run"]["elapsed_s"]) for t in task_ids
          if t in docs and isinstance((docs[t].get("run") or {}).get("elapsed_s"), (int, float))}
    final_s = ((run or {}).get("run") or {}).get("final_cell_s_after_start")
    if isinstance(final_s, (int, float)) and el:
        overhead, source = max(0.0, final_s - sum(el.values()) / ranks), "measured: final cell time - task seconds / ranks"
    else:
        overhead, source = DEFAULT_OVERHEAD_S, f"default {DEFAULT_OVERHEAD_S:.0f} s (no final cell time in _run.json)"
    reasons = {t: (docs[t].get("run") or {}).get("stop_reason", "unknown") for t in task_ids if t in docs}
    return {"elapsed": el, "overhead_s": round(overhead, 1), "overhead_source": source, "stop_reason": reasons}


def _project(mean_task_s: float | None, n_tasks: int, ranks: int, overhead_s: float) -> float | None:
    if mean_task_s is None:
        return None
    return (mean_task_s * n_tasks / ranks + overhead_s) / 3600


def compare(base_rep: dict, new_rep: dict, base_pool: Path, new_pool: Path, hidden_tasks: int = HIDDEN_TASKS,
            hidden_source: str = HIDDEN_TASKS_SOURCE, ranks: int = RANKS, wall_target_h: float = WALL_TARGET_H,
            gain_threshold: int = GAIN_THRESHOLD) -> dict[str, Any]:
    """Side-by-side numbers of two analysed runs over the same test outputs, the full-run projection and the W1 decision."""
    key = lambda r: (r["task_id"], r["test_index"])  # noqa: E731
    b_rows = {key(r): r for r in base_rep["per_output"]}
    n_rows = {key(r): r for r in new_rep["per_output"]}
    if set(b_rows) != set(n_rows):
        raise ValueError(f"the runs are not over the same outputs: only baseline {sorted(set(b_rows) - set(n_rows))[:5]}, "
                         f"only new {sorted(set(n_rows) - set(b_rows))[:5]}")
    ids = lambda rows, k: sorted(f"{t}_{i}" for (t, i) in rows if k(t, i))  # noqa: E731
    gained = lambda f: ids(n_rows, lambda t, i: n_rows[(t, i)][f] and not b_rows[(t, i)][f])  # noqa: E731
    lost = lambda f: ids(n_rows, lambda t, i: b_rows[(t, i)][f] and not n_rows[(t, i)][f])  # noqa: E731
    task_ids = sorted({t for t, _ in n_rows})
    b_docs, b_run, _ = load_pool(base_pool)
    n_docs, n_run, _ = load_pool(new_pool)
    bt, nt = _run_times(b_docs, b_run, task_ids, ranks), _run_times(n_docs, n_run, task_ids, ranks)

    def side(f) -> dict[str, Any]:
        b, n = f(base_rep), f(new_rep)
        return {"baseline": b, "new": n, "delta": n - b}

    both = [t for t in task_ids if t in bt["elapsed"] and t in nt["elapsed"] and bt["elapsed"][t] > 0]
    ratios = {t: nt["elapsed"][t] / bt["elapsed"][t] for t in both}
    mean = lambda v: statistics.fmean(v) if v else None  # noqa: E731
    b_mean, n_mean = mean(list(bt["elapsed"].values())), mean(list(nt["elapsed"].values()))
    reasons = lambda t: dict(Counter(t["stop_reason"].values()))  # noqa: E731
    n_cut = sum(1 for r in nt["stop_reason"].values() if r == "global_end_time")
    lower_bound = bool(n_cut or new_rep["tasks_without_pool"])
    proj_b = _project(b_mean, hidden_tasks, ranks, bt["overhead_s"])
    proj_n = _project(n_mean, hidden_tasks, ranks, nt["overhead_s"])
    fits = proj_n is not None and proj_n <= wall_target_h

    g2 = new_rep["selected_pass_at_2"]["n"] - base_rep["selected_pass_at_2"]["n"]
    gc = new_rep["coverage"]["n"] - base_rep["coverage"]["n"]
    n = new_rep["n_outputs"]
    if g2 >= gain_threshold and fits:
        rule, text = 1, (f"pass@2 +{g2} of {n} >= {gain_threshold} and projected full run {proj_n:.2f} h <= {wall_target_h:g} h: "
                         "promote W1 toward a submission candidate"
                         + (" (projection is a lower bound: tasks cut or missing; check before promoting)" if lower_bound else ""))
    elif g2 >= gain_threshold:
        rule, text = 2, (f"pass@2 +{g2} of {n} >= {gain_threshold} but projected full run "
                         f"{'n/a' if proj_n is None else f'{proj_n:.2f} h'} > {wall_target_h:g} h: not promoted as is; "
                         "next a time-bounded variant (widen only low-margin outputs, D11)")
    elif gc >= gain_threshold:
        rule, text = 3, (f"pass@2 {g2:+d} of {n} < +{gain_threshold}, coverage +{gc} >= {gain_threshold}: "
                         "next step is selection on the wider pool")
    else:
        rule, text = 4, (f"pass@2 {g2:+d}, coverage {gc:+d} of {n} (both < +{gain_threshold}): stop ARC-AGI-2 method work, "
                         "B0 stays the entry")
    return {
        "baseline": {"pool_dir": str(base_pool), "dfs_min_prob": _dfs_min_prob(b_docs, b_run)},
        "new": {"pool_dir": str(new_pool), "dfs_min_prob": _dfs_min_prob(n_docs, n_run)},
        "n_tasks": len(task_ids),
        "n_outputs": n,
        "outputs_with_pool": side(lambda r: r["n_outputs_with_pool"]),
        "tasks_without_pool": {"baseline": base_rep["tasks_without_pool"], "new": new_rep["tasks_without_pool"]},
        "selected_pass_at_2": side(lambda r: r["selected_pass_at_2"]["n"]),
        "selected_pass_at_1": side(lambda r: r["selected_pass_at_1"]["n"]),
        "coverage": side(lambda r: r["coverage"]["n"]),
        "coverage_gap": side(lambda r: r["coverage_gap"]["n"]),
        "pass_at_2_gained": gained("selected_correct"), "pass_at_2_lost": lost("selected_correct"),
        "coverage_gained": gained("in_pool"), "coverage_lost": lost("in_pool"),
        "task_elapsed_s": {"baseline": _dist([round(v) for v in bt["elapsed"].values()]),
                           "new": _dist([round(v) for v in nt["elapsed"].values()])},
        "elapsed_ratio": {"n_tasks": len(both), "dist": _dist([round(v, 3) for v in ratios.values()]),
                          "max_task": max(ratios, key=ratios.get) if ratios else None,
                          "ratio_of_means": round(n_mean / b_mean, 3) if b_mean and n_mean else None},
        "stop_reason": {"baseline": reasons(bt), "new": reasons(nt)},
        "tasks_at_per_task_cap": {"baseline": sum(1 for r in bt["stop_reason"].values() if r == "per_task_cap"),
                                  "new": sum(1 for r in nt["stop_reason"].values() if r == "per_task_cap")},
        "tasks_cut_by_global_end_time": {"baseline": sum(1 for r in bt["stop_reason"].values() if r == "global_end_time"),
                                         "new": n_cut},
        "pool_size_unique": {"baseline": base_rep["pool_size_unique"]["dist"], "new": new_rep["pool_size_unique"]["dist"]},
        "pool_size_beams": {"baseline": base_rep["pool_size_beams"]["dist"], "new": new_rep["pool_size_beams"]["dist"]},
        "projection": {
            "formula": "mean task seconds x hidden tasks / ranks + overhead (model load, rank start, idle tail, final cell)",
            "hidden_tasks": hidden_tasks, "hidden_tasks_source": hidden_source, "ranks": ranks,
            "mean_task_s": {"baseline": None if b_mean is None else round(b_mean, 1),
                            "new": None if n_mean is None else round(n_mean, 1)},
            "overhead_s": {"baseline": bt["overhead_s"], "new": nt["overhead_s"],
                           "source": {"baseline": bt["overhead_source"], "new": nt["overhead_source"]}},
            "full_run_h": {"baseline": None if proj_b is None else round(proj_b, 2),
                           "new": None if proj_n is None else round(proj_n, 2)},
            "per_100_tasks_h": {"baseline": None if b_mean is None else round(b_mean * 100 / ranks / 3600, 2),
                                "new": None if n_mean is None else round(n_mean * 100 / ranks / 3600, 2)},
            "target_h": wall_target_h, "new_fits_target": fits, "new_is_lower_bound": lower_bound,
        },
        "decision": {"gain_threshold": gain_threshold, "wall_target_h": wall_target_h, "pass_at_2_gain": g2,
                     "coverage_gain": gc, "rule": rule, "text": text},
    }


def render_compare(c: dict[str, Any], solutions_source: str) -> str:
    def floor(v):
        return f"{v}" if v is not None else "not recorded (B0 default 0.2)"

    def row(name, k):
        s = c[k]
        return f"  {name:<26}{s['baseline']:>9}{s['new']:>9}{s['delta']:>+9}"

    def dshort(d):
        return "n 0" if not d.get("n") else f"median {d['median']}, mean {d['mean']}, p75 {d['p75']}, max {d['max']} (n {d['n']})"

    e, p, n = c["elapsed_ratio"], c["projection"], c["n_outputs"]
    lst = lambda v: f"{len(v)} {v}" if v else "0"  # noqa: E731
    h = lambda v: "n/a" if v is None else f"{v:.2f} h"  # noqa: E731
    lines = [
        f"compare: new {c['new']['pool_dir']} vs baseline {c['baseline']['pool_dir']}",
        f"ground truth: {solutions_source}",
        f"DFS probability floor: baseline {floor(c['baseline']['dfs_min_prob'])}; new {floor(c['new']['dfs_min_prob'])}",
        f"same outputs: {n} test outputs over {c['n_tasks']} tasks; tasks without a pool file: baseline "
        f"{c['tasks_without_pool']['baseline'] or 'none'}, new {c['tasks_without_pool']['new'] or 'none'}",
        "",
        f"  {'(outputs of ' + str(n) + ')':<26}{'baseline':>9}{'new':>9}{'delta':>9}",
        row("selected pass@2", "selected_pass_at_2"),
        row("selected pass@1", "selected_pass_at_1"),
        row("coverage", "coverage"),
        row("coverage gap", "coverage_gap"),
        row("outputs with a pool entry", "outputs_with_pool"),
        f"pass@2 gained: {lst(c['pass_at_2_gained'])}; lost: {lst(c['pass_at_2_lost'])}",
        f"coverage gained: {lst(c['coverage_gained'])}; lost: {lst(c['coverage_lost'])}",
        "",
        f"task elapsed s: baseline {dshort(c['task_elapsed_s']['baseline'])}; new {dshort(c['task_elapsed_s']['new'])}",
        f"per-task elapsed ratio new/baseline ({e['n_tasks']} tasks in both): median {e['dist'].get('median')}, "
        f"max {e['dist'].get('max')} ({e['max_task']}), min {e['dist'].get('min')}; ratio of means {e['ratio_of_means']}",
        f"stop reason: baseline {c['stop_reason']['baseline']}; new {c['stop_reason']['new']}",
        f"tasks at the 1200 s per-task cap: baseline {c['tasks_at_per_task_cap']['baseline']}, new "
        f"{c['tasks_at_per_task_cap']['new']}; cut by the global end time: baseline "
        f"{c['tasks_cut_by_global_end_time']['baseline']}, new {c['tasks_cut_by_global_end_time']['new']}",
        f"pool size, unique candidates per output: baseline {dshort(c['pool_size_unique']['baseline'])}; "
        f"new {dshort(c['pool_size_unique']['new'])}",
        f"pool size, beams per output: baseline {dshort(c['pool_size_beams']['baseline'])}; "
        f"new {dshort(c['pool_size_beams']['new'])}",
        "",
        f"full-run projection: {p['formula']}; {p['hidden_tasks']} hidden tasks ({p['hidden_tasks_source']}), "
        f"{p['ranks']} ranks",
        f"  mean task s: baseline {p['mean_task_s']['baseline']}, new {p['mean_task_s']['new']}; overhead s: baseline "
        f"{p['overhead_s']['baseline']} ({p['overhead_s']['source']['baseline']}), new {p['overhead_s']['new']} "
        f"({p['overhead_s']['source']['new']})",
        f"  full run: baseline {h(p['full_run_h']['baseline'])}, new {h(p['full_run_h']['new'])} (target {p['target_h']:g} h: "
        f"{'fits' if p['new_fits_target'] else 'does not fit'}"
        f"{'; lower bound, tasks cut or missing' if p['new_is_lower_bound'] else ''}); per 100 tasks without overhead: "
        f"baseline {h(p['per_100_tasks_h']['baseline'])}, new {h(p['per_100_tasks_h']['new'])}",
        "",
        f"decision W1 (rule {c['decision']['rule']}): {c['decision']['text']}",
    ]
    return "\n".join(lines)


def _solutions_for(pool_dirs: list[Path], explicit: str) -> tuple[dict[str, list], str]:
    """Explicit file, else the first Kaggle solutions copy among the pool directories, else the local copy."""
    if not explicit:
        for d in pool_dirs:
            kaggle_copy = d / "_kaggle_solutions_audit_tasks.json"
            if kaggle_copy.is_file():
                # Kaggle's evaluation set differs from the ARC-AGI-2 GitHub copy on some tasks (extra or changed
                # test inputs), and the run decoded Kaggle's version, so Kaggle's solutions are the ground truth.
                return json.loads(kaggle_copy.read_text(encoding="utf-8")), f"{kaggle_copy} (Kaggle evaluation solutions)"
    return find_solutions(explicit or None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool directory from the audit kernel output")
    ap.add_argument("--solutions", default="", help="ground truth file or ARC-AGI-2 data/evaluation directory")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: _run.json task_ids, else the dev split)")
    ap.add_argument("--gap-threshold", type=int, default=GAP_THRESHOLD)
    ap.add_argument("--auc-threshold", type=float, default=0.8)
    ap.add_argument("--settle-threshold", type=float, default=0.5,
                    help="share of tasks whose final top-2 is fixed by 50%% of their time")
    ap.add_argument("--json", default="", help="write the full report (with per-output rows) here")
    ap.add_argument("--per-output", action="store_true")
    ap.add_argument("--compare", default="", metavar="BASELINE_POOL",
                    help="baseline pool directory (e.g. the audit's); --pool is the new run; prints both side by side")
    ap.add_argument("--hidden-tasks", type=int, default=HIDDEN_TASKS,
                    help=f"task count of the full (hidden) run for the time projection (default {HIDDEN_TASKS})")
    ap.add_argument("--ranks", type=int, default=RANKS)
    ap.add_argument("--wall-target-h", type=float, default=WALL_TARGET_H)
    ap.add_argument("--gain-threshold", type=int, default=GAIN_THRESHOLD)
    args = ap.parse_args(argv)
    tasks = args.tasks.split(",") if args.tasks else None
    if args.compare:
        new_pool, base_pool = Path(args.pool), Path(args.compare)
        solutions, source = _solutions_for([new_pool, base_pool], args.solutions)
        if tasks is None:
            _, run, _ = load_pool(new_pool)
            tasks = list(run["task_ids"]) if run and run.get("task_ids") else dev_task_ids()
        reps = {name: analyze(d, solutions, tasks, args.gap_threshold, args.auc_threshold, args.settle_threshold)
                for name, d in (("baseline", base_pool), ("new", new_pool))}
        hidden_source = HIDDEN_TASKS_SOURCE if args.hidden_tasks == HIDDEN_TASKS else "--hidden-tasks"
        cmp = compare(reps["baseline"], reps["new"], base_pool, new_pool, args.hidden_tasks, hidden_source, args.ranks,
                      args.wall_target_h, args.gain_threshold)
        copies = [load_pool(d)[2] for d in (base_pool, new_pool)]
        if all(c is not None for c in copies):
            cmp["kaggle_solutions_copies_identical"] = all(copies[0].get(t) == copies[1].get(t) for t in tasks)
        print(render_compare(cmp, source)
              + ("" if cmp.get("kaggle_solutions_copies_identical", True)
                 else "\nWARNING: the two runs' Kaggle solutions copies differ"))
        if args.json:
            Path(args.json).write_text(json.dumps({"ground_truth": source, "compare": cmp, **reps}, indent=1) + "\n",
                                       encoding="utf-8")
        return 0
    solutions, source = _solutions_for([Path(args.pool)], args.solutions)
    rep = analyze(Path(args.pool), solutions, tasks, args.gap_threshold,
                  args.auc_threshold, args.settle_threshold)
    rep["ground_truth"] = source
    print(render(rep, source, args.per_output))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
