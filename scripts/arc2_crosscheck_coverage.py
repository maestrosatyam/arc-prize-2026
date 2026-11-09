"""Cross-check of the ARC-AGI-2 account against an independent public release of NVARC-style candidate pools.

CPU only, read-only. Inputs are two public Kaggle datasets, downloaded beforehand and read as JSON only:
- the candidate pools (``--pools``: public_eval_*.json, the full public evaluation, 120 tasks / 172 outputs each);
- the analysis code release (``--code``: README and paper; nothing in it is run, only hashed for provenance);
and, optionally, the public notebook's source (``--notebook``, hashed only), whose scorer is re-implemented here:
per test output, group the candidates by distinct grid (count = rows with that grid, nlls = their aug_nlls and
score_aug concatenated), score count + a * (-mean nll) with a in {0, 0.02, 0.1, 0.5}, covered = the right grid in
the pool, selected = in the top two; per configuration the a with the most selected outputs.

Sections of the result:
1. instrument: that scorer, as coded, on the 120 public tasks (competition answers: the bank's Kaggle copies for the
   101 bank tasks, eval/data's file for the other 19, whose output counts equal the pools'; aggregates only), on
   eval/data's file as it is (167 outputs), and the pools' own selector (count field, unscored grids last) against
   the pool release's published counts;
2. account: on the bank's 101 unsealed tasks (145 outputs; eval/arc2_bank.py refuses the sealed 19), per pool:
   submitted and right (top two at the best a on these outputs, and at a = 0), in the pool but not submitted,
   never generated; beside B0 and v7's direct merge (scripts/arc2_paper_account.py) and B0's pool under the same
   scorer;
3. agreement with B0 per pool: 2x2 of pool coverage and of top-two right, the outputs where coverage disagrees,
   exact McNemar p;
4. attribution: efficiency per configuration and the empty-pool regression, on 172 and on the 145.

Usage: envs/kaggle-sim/bin/python -I scripts/arc2_crosscheck_coverage.py --pools <pools dir> --code <code dir>
       [--notebook <notebook .py>] [--out experiments/arc2-bank/crosscheck_coverage/result.json]
Deterministic: no clock, no randomness; numpy for the means and the least squares, as the notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.arc2_bank import LOCAL_SOLUTIONS, RULES, _q27_grids, grids_equal, load_bank, replay  # noqa: E402

# the notebook's LABEL, in its sorted-glob order
POOLS = {
    "public_eval_gap_notebook_failed_in_aimo.json": "full-clock notebook",
    "public_eval_h7_per_input_deadlines.json": "per-input deadlines",
    "public_eval_linked_16view_v3.json": "16 views",
    "public_eval_naug16_v8.json": "fewer views",
    "public_eval_submitted_config_matched.json": "submitted config",
}
SUBMISSION = "public_eval_gap_notebook_submission.json"   # the full-clock notebook's own submission file
ALPHAS = [0.0, 0.02, 0.1, 0.5]
# the pool release's own counts (its paper, Sections 4.1-4.3, Tables 4 and 5): 172 outputs, its selector
# (mean NLL minus count, unscored grids last), task level = every output of the task
PUBLISHED = {
    "submitted config": {"outputs_in_pool": 42, "outputs_hit": 35, "empty": 53, "tasks_covered": 27, "tasks_solved": 24,
                         "arc_score": 22.2},
    "full-clock notebook": {"outputs_in_pool": 53, "empty": 14, "tasks_covered": 32,
                            "submission": {"outputs_hit": 45, "tasks_solved": 29, "arc_score": 27.9}},
    "16 views": {"empty": 65, "tasks_covered": 27, "tasks_solved": 24, "arc_score": 22.5},
    "per-input deadlines": {"empty": 34, "tasks_covered": 30, "tasks_solved": 29, "arc_score": 26.0,
                            "unscored_rows": 93},
    "fewer views": {"tasks_covered": 24, "tasks_solved": 19},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(g) -> tuple:
    return tuple(tuple(r) for r in g)


def load_pool(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["pools"] if isinstance(d, dict) and "pools" in d else d


# ----------------------------------------------------------------------------- the scorer

def group(cands: list, count_field: bool) -> dict:
    """The notebook's group(): one entry per distinct grid. As coded, count = rows with the grid; with count_field
    the rows' own ``count`` (default 1) is summed instead."""
    G: dict = {}
    for c in cands:
        if "grid" not in c:
            continue
        e = G.setdefault(norm(c["grid"]), {"count": 0, "nlls": []})
        e["count"] += int(c.get("count", 1)) if count_field else 1
        e["nlls"].extend(c.get("aug_nlls", []))
        e["nlls"].extend(c.get("score_aug", []))
    return G


def score(e: dict, a: float, unscored_last: bool) -> float:
    """As coded: count + a * (-mean nll), a grid without nlls scoring as nll 0. unscored_last: such a grid ranks
    after every scored one (the pools' own selector)."""
    if not e["nlls"]:
        return -math.inf if unscored_last else e["count"] + a * (-0.0)
    return e["count"] + a * (-float(np.mean(e["nlls"])))


def top2(G: dict, a: float, unscored_last: bool) -> list:
    return sorted(G, key=lambda g: score(G[g], a, unscored_last), reverse=True)[:2]


def assess(cands: list, ref, variant: str) -> dict:
    """One output: empty (no rows), covered, and selected per a."""
    cf = variant == "count_field"
    if not cands:
        return {"empty": True, "covered": False, "sel": {a: False for a in ALPHAS + [1.0]}, "n_grids": 0}
    G = group(cands, cf)
    r = norm(ref)
    cov = r in G
    sel = {a: (cov and r in top2(G, a, cf)) for a in ALPHAS + [1.0]}
    return {"empty": False, "covered": cov, "sel": sel, "n_grids": len(G)}


def evaluate(pools: dict, sol: dict, variant: str) -> dict:
    """Per output (task order of ``sol``) plus the notebook's aggregates per a."""
    per: dict = {}
    for pid, refs in sol.items():
        p = pools.get(pid)
        if p is None:
            continue
        for i, ref in enumerate(refs):
            per[f"{pid}_{i}"] = assess(p[i] if i < len(p) else [], ref, variant) | {"task": pid}
    return per


def aggregate(per: dict, a: float) -> dict:
    tot = len(per)
    cov = sum(v["covered"] for v in per.values())
    sel = sum(v["sel"][a] for v in per.values())
    empty = sum(v["empty"] for v in per.values())
    tasks: dict = {}
    for v in per.values():
        t = tasks.setdefault(v["task"], [True, True, 0, 0])
        t[0] &= v["covered"]
        t[1] &= v["sel"][a]
        t[2] += 1
        t[3] += v["sel"][a]
    pct = lambda x: 100.0 * x / max(tot, 1)  # noqa: E731
    return {"a": a, "outputs": tot, "covered": cov, "selected": sel, "empty": empty,
            "coverage_pct": round(pct(cov), 2), "selected_pct": round(pct(sel), 2),
            "efficiency_pct": round(pct(sel) / pct(cov) * 100, 2) if cov else 0.0,
            "tasks": len(tasks), "tasks_covered": sum(t[0] for t in tasks.values()),
            "tasks_solved": sum(t[1] for t in tasks.values()),
            "arc_score": round(100.0 * sum(t[3] / t[2] for t in tasks.values()) / max(len(tasks), 1), 2)}


def best_a(per: dict) -> float:
    """max() over ALPHAS by selected, the first maximum kept (the notebook's choice)."""
    return max(ALPHAS, key=lambda a: aggregate(per, a)["selected"])


def regression(rows: list[dict]) -> dict:
    """The notebook's 'quantitative claim': coverage on empty pools by least squares, efficiency spread."""
    e = np.array([r["empty"] for r in rows], float)
    c = np.array([r["coverage_pct"] for r in rows], float)
    eff = np.array([r["efficiency_pct"] for r in rows], float)
    A = np.vstack([e, np.ones_like(e)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, c, rcond=None)
    pred = A @ [slope, icpt]
    r2 = 1 - ((c - pred) ** 2).sum() / ((c - c.mean()) ** 2).sum()
    return {"n": len(rows), "intercept": round(float(icpt), 3), "slope_per_empty": round(float(slope), 4),
            "pp_per_10_empty": round(float(slope) * 10, 2), "r2": round(float(r2), 3),
            "efficiency_mean": round(float(eff.mean()), 1), "efficiency_min": round(float(eff.min()), 1),
            "efficiency_max": round(float(eff.max()), 1)}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", required=True, help="markrowset/arc-agi-ttt-candidate-pools, unzipped")
    ap.add_argument("--code", required=True, help="markrowset/arc-ttt-coverage-code, unzipped (hashed only)")
    ap.add_argument("--notebook", help="the public notebook's source (hashed only)")
    ap.add_argument("--out", default=str(ROOT / "experiments/arc2-bank/crosscheck_coverage/result.json"))
    args = ap.parse_args()
    P, C = Path(args.pools), Path(args.code)

    # --- our bank: 101 unsealed tasks, 145 outputs, competition answers (Kaggle copies beside the pools)
    bank = load_bank()
    assert len(bank.task_ids) == 101 and bank.n_outputs == 145
    keys = [o.key for o in bank.outputs()]
    truth = {o.key: bank.truth(o) for o in bank.outputs()}
    assert all(truth[k] is not None for k in keys)
    b0_right = replay(bank, RULES["b0"]).right_keys()
    mg_right = replay(bank, RULES["v7_direct"]).right_keys()
    b0_pool, mg_pool, b0_empty = {}, {}, {}
    b0_rows: dict = {}
    for t in bank.task_ids:
        task = bank.tasks[t]
        for j, o in enumerate(task.outputs):
            tr = truth[o.key]
            b0_pool[o.key] = any(grids_equal(c["grid"], tr) for c in o.candidates)
            mg_pool[o.key] = b0_pool[o.key] or any(grids_equal(g, tr) for _, g in _q27_grids(bank, task, j))
            b0_empty[o.key] = not o.candidates
            # B0's pool in the release's row format: one row per distinct grid, count = views that produced it
            b0_rows[o.key] = [{"grid": c["grid"], "count": int(c.get("n_beams") or 1),
                               "aug_nlls": list(c.get("score_aug") or [])} for c in o.candidates]
    acc = lambda right, pool: {"right": sum(k in right for k in keys),  # noqa: E731
                               "pool_not_submitted": sum(pool[k] and k not in right for k in keys),
                               "never": sum(not pool[k] for k in keys)}
    ours = {"b0": acc(b0_right, b0_pool) | {"empty": sum(b0_empty.values())},
            "v7_direct": acc(mg_right, mg_pool)}
    assert (ours["b0"]["right"], ours["b0"]["pool_not_submitted"], ours["b0"]["never"]) == (40, 6, 99)
    assert (ours["v7_direct"]["right"], ours["v7_direct"]["pool_not_submitted"], ours["v7_direct"]["never"]) == (45, 10, 90)
    assert all(b0_pool[k] for k in b0_right)

    # --- answers for the instrument check: the competition's 172 = bank copies (101) + eval/data (other 19)
    local = json.loads(LOCAL_SOLUTIONS.read_text(encoding="utf-8"))
    pools = {name: load_pool(P / name) for name in POOLS}
    ref_pool = pools["public_eval_submitted_config_matched.json"]
    assert set(ref_pool) == set(local)
    sol172 = {}
    for t in local:   # eval/data's task order
        if t in bank.tasks:
            sol172[t] = bank.solutions[t]
        else:
            assert len(local[t]) == len(ref_pool[t]), "output counts of the other tasks differ from the pools"
            sol172[t] = local[t]
    n172 = sum(len(v) for v in sol172.values())
    n_local = sum(len(v) for v in local.values())
    misaligned = sorted(t for t in bank.task_ids if local[t] != bank.solutions[t])
    sol145 = {t: bank.solutions[t] for t in bank.task_ids}
    for name, p in pools.items():
        assert len(p) == 120 and all(len(p[t]) == len(sol172[t]) for t in sol172), name

    # --- 1. instrument check
    instrument: dict = {"answers_172": f"bank Kaggle copies for the 101 bank tasks + {LOCAL_SOLUTIONS.name} for the "
                                       f"other 19 (output counts equal the pools'); aggregates only",
                        "outputs_172": n172, "outputs_local": n_local,
                        "local_file_tasks_with_fewer_outputs": len(misaligned),
                        "table1_as_coded_172": [], "table1_as_coded_local_file": [],
                        "count_field_selector_172": [], "per_alpha_selected_as_coded_172": {}}
    rows172, rows_local = [], []
    for name, label in POOLS.items():
        per = evaluate(pools[name], sol172, "as_coded")
        a = best_a(per)
        r = aggregate(per, a) | {"name": label, "file": name}
        rows172.append(r)
        instrument["per_alpha_selected_as_coded_172"][label] = {str(x): aggregate(per, x)["selected"] for x in ALPHAS}
        perl = evaluate(pools[name], local, "as_coded")
        rows_local.append(aggregate(perl, best_a(perl)) | {"name": label})
        perc = evaluate(pools[name], sol172, "count_field")
        own = aggregate(perc, 1.0)
        instrument["count_field_selector_172"].append(
            {"name": label, "a": 1.0, "covered": own["covered"], "selected": own["selected"], "empty": own["empty"],
             "efficiency_pct": own["efficiency_pct"], "tasks_covered": own["tasks_covered"],
             "tasks_solved": own["tasks_solved"], "arc_score": own["arc_score"],
             "best_a_selected": {str(x): aggregate(perc, x)["selected"] for x in ALPHAS},
             "unscored_rows": sum(1 for t in pools[name].values() for cs in t for c in cs
                                  if not (c.get("aug_nlls") or c.get("score_aug"))),
             "published": PUBLISHED.get(label, {})})
    instrument["table1_as_coded_172"] = rows172
    instrument["table1_as_coded_local_file"] = rows_local
    instrument["claim_as_coded_172"] = regression(rows172)
    p_ = np.mean([r["coverage_pct"] for r in rows172]) / 100
    instrument["claim_as_coded_172"]["binomial_se_pp_n172"] = round(float(100 * (p_ * (1 - p_) / 172) ** 0.5), 1)
    sub = json.loads((P / SUBMISSION).read_text(encoding="utf-8"))
    hits, per_task = 0, []
    for t, refs in sol172.items():
        h = [any(sub[t][i][k] == ref for k in ("attempt_1", "attempt_2")) for i, ref in enumerate(refs)]
        hits += sum(h)
        per_task.append(sum(h) / len(h))
    instrument["full_clock_submission_172"] = {"outputs_hit": hits, "tasks_solved": sum(x == 1 for x in per_task),
                                               "arc_score": round(100 * sum(per_task) / len(per_task), 2)}

    # --- 2. account on the bank's 145 outputs, 3. agreement with B0
    account: dict = {"ours": ours, "pools": [], "b0_pool_release_scorer": {}}
    agreement: dict = {}
    union_cov = {k: False for k in keys}
    for name, label in POOLS.items():
        per = evaluate(pools[name], sol145, "as_coded")
        assert list(per) == keys
        a = best_a(per)
        perc = evaluate(pools[name], sol145, "count_field")
        ac = best_a(perc)
        cov = {k: per[k]["covered"] for k in keys}
        for k in keys:
            union_cov[k] |= cov[k]
        right = {k for k in keys if per[k]["sel"][a]}
        right0 = {k for k in keys if per[k]["sel"][0.0]}
        rightc = {k for k in keys if perc[k]["sel"][ac]}
        n_cov = sum(cov.values())
        account["pools"].append({
            "name": label, "file": name, "best_a": a,
            "best_a_same_as_172": a == next(r["a"] for r in rows172 if r["file"] == name), "covered": n_cov,
            "right": len(right), "pool_not_submitted": n_cov - len(right), "never": 145 - n_cov,
            "right_a0": len(right0), "pool_not_submitted_a0": n_cov - len(right0),
            "count_field": {"best_a": ac, "right": len(rightc), "right_a1": sum(perc[k]["sel"][1.0] for k in keys)},
            "empty": sum(per[k]["empty"] for k in keys),
            "never_nonempty": sum((not cov[k]) and not per[k]["empty"] for k in keys),
            "efficiency_pct": round(100 * len(right) / n_cov, 1) if n_cov else 0.0})
        both = sum(cov[k] and b0_pool[k] for k in keys)
        only_t = sorted(k for k in keys if cov[k] and not b0_pool[k])
        only_o = sorted(k for k in keys if b0_pool[k] and not cov[k])
        s_only_t = sorted(k for k in right if k not in b0_right)
        s_only_o = sorted(k for k in b0_right if k not in right)
        agreement[label] = {
            "coverage": {"both": both, "only_theirs": len(only_t), "only_ours": len(only_o),
                         "neither": 145 - both - len(only_t) - len(only_o), "mcnemar_p": mcnemar(len(only_t), len(only_o))},
            "top2_right": {"a": a, "both": len(right & b0_right), "only_theirs": len(s_only_t),
                           "only_ours": len(s_only_o),
                           "neither": 145 - len(right | b0_right), "mcnemar_p": mcnemar(len(s_only_t), len(s_only_o))},
            "disagreements_with_losing_pool_empty": {
                "only_theirs_b0_empty": sum(b0_empty[k] for k in only_t),
                "only_ours_theirs_empty": sum(per[k]["empty"] for k in only_o)},
            "coverage_disagreements": {
                "only_theirs": [{"output": k, "b0_pool_empty": b0_empty[k], "merge_pool_has_it": mg_pool[k]}
                                for k in only_t],
                "only_ours": [{"output": k, "their_pool_empty": per[k]["empty"], "b0_right": k in b0_right}
                              for k in only_o]},
            "top2_disagreements": {"only_theirs": s_only_t, "only_ours": s_only_o}}
    # the full-clock notebook's own submission on the 145
    gap = load_pool(P / "public_eval_gap_notebook_failed_in_aimo.json")
    sub_right = {k for k in keys
                 if any(sub[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])][x] == truth[k] for x in ("attempt_1", "attempt_2"))}
    gap_cov = {k: any(norm(c["grid"]) == norm(truth[k]) for c in gap[k.rsplit("_", 1)[0]][int(k.rsplit("_", 1)[1])])
               for k in keys}
    n_gc = sum(gap_cov.values())
    account["full_clock_submission"] = {"right": len(sub_right), "right_outside_pool": sum(not gap_cov[k] for k in sub_right),
                                        "pool_not_submitted": sum(gap_cov[k] and k not in sub_right for k in keys),
                                        "never": 145 - n_gc}
    agreement["full-clock notebook"]["submission_vs_b0"] = {
        "both": len(sub_right & b0_right), "only_theirs": len(sub_right - b0_right),
        "only_ours": len(b0_right - sub_right), "neither": 145 - len(sub_right | b0_right),
        "mcnemar_p": mcnemar(len(sub_right - b0_right), len(b0_right - sub_right))}
    account["union_of_five_pools"] = {"covered": sum(union_cov.values()),
                                      "covered_with_b0": sum(union_cov[k] or b0_pool[k] for k in keys),
                                      "covered_with_merge": sum(union_cov[k] or mg_pool[k] for k in keys),
                                      "b0_only": sorted(k for k in keys if b0_pool[k] and not union_cov[k]),
                                      "union_only_vs_merge": sorted(k for k in keys if union_cov[k] and not mg_pool[k])}
    # B0's pool under the release's scorer (count = views that produced the grid, 8 augmentation scores)
    perb = {k: assess(b0_rows[k], truth[k], "count_field") for k in keys}
    assert all(perb[k]["covered"] == b0_pool[k] for k in keys)
    selb = {a: sum(perb[k]["sel"][a] for k in keys) for a in ALPHAS + [1.0]}
    ab = max(ALPHAS, key=lambda a: selb[a])
    account["b0_pool_release_scorer"] = {"best_a": ab, "right": selb[ab], "right_a0": selb[0.0], "right_a1": selb[1.0],
                                         "selected_per_a": {str(a): selb[a] for a in ALPHAS + [1.0]},
                                         "covered": sum(b0_pool.values()), "empty": sum(b0_empty.values())}

    # --- 4. attribution on the 145: the regression with B0 as a sixth point
    rows145 = [{"empty": r["empty"], "coverage_pct": 100 * r["covered"] / 145,
                "efficiency_pct": 100 * r["right"] / r["covered"] if r["covered"] else 0.0} for r in account["pools"]]
    b0row = {"empty": ours["b0"]["empty"], "coverage_pct": 100 * (40 + 6) / 145, "efficiency_pct": 100 * 40 / 46}
    fit5 = regression(rows145)
    pred_b0 = fit5["intercept"] + fit5["slope_per_empty"] * b0row["empty"]
    attribution = {"as_coded_172": instrument["claim_as_coded_172"],
                   "count_field_172": regression([{"empty": r["empty"], "coverage_pct": 100 * r["covered"] / n172,
                                                    "efficiency_pct": r["efficiency_pct"]}
                                                   for r in instrument["count_field_selector_172"]]),
                   "as_coded_145": fit5,
                   "b0_on_their_line_145": {"b0_empty": b0row["empty"], "b0_coverage_pct": round(b0row["coverage_pct"], 2),
                                            "predicted_coverage_pct": round(pred_b0, 2),
                                            "b0_efficiency_pct": round(b0row["efficiency_pct"], 1),
                                            "merge_efficiency_pct": round(100 * 45 / 55, 1)},
                   "with_b0_145": regression(rows145 + [b0row]),
                   "b0_never_generated": {"total": ours["b0"]["never"], "empty_pool": ours["b0"]["empty"],
                                          "nonempty_without_right": ours["b0"]["never"] - ours["b0"]["empty"]}}

    out = {"schema": "arc2_crosscheck_coverage.v1",
           "bank": {"fingerprint": bank.fingerprint(), "n_tasks": 101, "n_outputs": 145},
           "inputs": {"pools_readme_sha256": sha(P / "README.md"), "code_readme_sha256": sha(C / "README.md"),
                      "pool_files_sha256": {n: sha(P / n) for n in list(POOLS) + [SUBMISSION]},
                      "notebook_sha256": sha(Path(args.notebook)) if args.notebook else None,
                      "local_solutions_sha256": sha(LOCAL_SOLUTIONS)},
           "alphas": ALPHAS, "instrument": instrument, "account_145": account, "agreement_145": agreement,
           "attribution": attribution}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print("Table 1 as coded, 172 outputs:")
    for r in rows172:
        print(f"  {r['name']:20s} a={r['a']:<4} cov {r['coverage_pct']:5.1f}% sel {r['selected_pct']:5.1f}% "
              f"eff {r['efficiency_pct']:5.1f}% empty {r['empty']:3d} tasks {r['tasks_covered']}/{r['tasks']}")
    print("  claim:", instrument["claim_as_coded_172"])
    print("local file as is:", [(r["name"], r["outputs"], r["covered"], r["selected"], r["empty"]) for r in rows_local])
    print("count-field selector (a=1) vs published:")
    for r in instrument["count_field_selector_172"]:
        print(f"  {r['name']:20s} cov {r['covered']} sel {r['selected']} empty {r['empty']} tasks cov {r['tasks_covered']} "
              f"solved {r['tasks_solved']} score {r['arc_score']} | published {r['published']}")
    print("  full-clock submission:", instrument["full_clock_submission_172"])
    print("Account on 145: ours", ours)
    for r in account["pools"]:
        print(f"  {r['name']:20s} a*={r['best_a']:<4} right {r['right']:2d} pool {r['pool_not_submitted']:2d} never {r['never']:3d}"
              f" | a=0 right {r['right_a0']:2d} | count field {r['count_field']} | empty {r['empty']}")
    print("  full-clock submission:", account["full_clock_submission"])
    print("  B0 pool, release scorer:", account["b0_pool_release_scorer"])
    print("  union:", {k: v for k, v in account["union_of_five_pools"].items() if not isinstance(v, list)})
    for label, g in agreement.items():
        print(f"  agree {label:20s} cov {g['coverage']} top2 {g['top2_right']} "
              f"losing pool empty {g['disagreements_with_losing_pool_empty']}")
    print("attribution:", json.dumps({k: v for k, v in attribution.items()}, indent=None))


if __name__ == "__main__":
    main()
