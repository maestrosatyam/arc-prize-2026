"""Rescue replay analysis: the pre-set rule of experiments/arc2-q27/RESCUE_REPLAY_RULE.md (frozen
2026-10-27 23:51 IST, as written in cbcbf70) on the run archive. Prints the rule line; writes replay_decision.json.

    python scripts/arc2_rescue_replay_analyze.py experiments/runs/kaggle-A2-rescue-replay-v1

A unit is (cut sample, test index, draw). Its answer in an arm is what that one rescued sample would put first under
the entry's ranking: arc_harness.selection.rank(task, [record], j, "vr")[0] (a verified clean program's output first,
then the written grid, then an unverified program's output). Gain: block right, plain not; loss: plain right, block
not. Helps: gains - losses >= 3, one-sided sign test p <= 0.10 on the discordant units, and block's verified-clean
wrong answers <= plain's + 1. Does not help: gains - losses <= 0. Otherwise report. Validity: >= 15 cut samples with a
saved tail; median completion tokens >= 5 per arm; empty or zero-token replies <= 5% per arm; no sealed task read.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "arc2" / "hybrid_vr"))
sys.path.insert(0, str(ROOT))

from arc_harness.grids import hardcoded_risk  # noqa: E402
from arc_harness.selection import rank  # noqa: E402

VRBANK = ROOT / "experiments" / "runs" / "kaggle-A2-VRbank-v1" / "hybrid" / "p2"


def p_one_sided(gains: int, losses: int) -> float:
    n = gains + losses
    return sum(comb(n, i) for i in range(gains, n + 1)) / 2 ** n if n else 1.0


def top(task, rec, j):
    r = rank(task, [rec], j, "vr")
    return r[0].grid if r else None


def either_right(rec, j, truth):
    g = rec.get("grids") or []
    po = (rec.get("verify") or {}).get("program_outputs") or []
    return (j < len(g) and g[j] == truth) or (j < len(po) and po[j] == truth)


def analyze(run: Path) -> dict:
    p2 = run / "hybrid" / "p2"
    tasks = json.loads((p2 / "_kaggle_challenges_replay_tasks.json").read_text())
    sols = json.loads((p2 / "_kaggle_solutions_replay_tasks.json").read_text())
    from eval import arc2_bank as B
    sealed = set()
    for d in B.BANK_SEALED:
        m = Path(d) / B.SEALED_MANIFEST
        if m.is_file():
            sealed |= set(json.loads(m.read_text())["task_ids"])
    assert not set(tasks) & sealed
    recs = [json.loads(l) for l in (p2 / "replay.jsonl").read_text().splitlines() if l.strip()]
    jobs = json.loads((p2 / "replay_jobs.json").read_text())
    by = {(r["arm"], r["unit"], r["draw"]): r for r in recs}
    cuts = sorted({j["unit"] for j in jobs if j["arm"] in ("plain", "block")})
    arms = {}
    for arm in ("plain", "block", "R1", "R2"):
        rr = [r for r in recs if r["arm"] == arm]
        toks = [int(r.get("gen_tokens") or 0) for r in rr]
        right = either = vclean = vclean_right = ran = 0
        for r in rr:
            t = tasks[r["task_id"]]
            v = r.get("verify") or {}
            ran += bool(v.get("ran"))
            clean = bool(v.get("passed")) and not hardcoded_risk(r.get("program") or "")
            for j in range(len(t["test"])):
                truth = sols[r["task_id"]][j]
                right += top(t, r, j) == truth
                either += either_right(r, j, truth)
                if clean:
                    vclean += 1
                    po = v.get("program_outputs") or []
                    vclean_right += j < len(po) and po[j] == truth
        arms[arm] = {"jobs": sum(1 for j in jobs if j["arm"] == arm), "records": len(rr),
                     "empty_or_zero": sum(1 for x in toks if not x), "median_tokens": statistics.median(toks) if toks else 0,
                     "tokens": sum(toks), "right": right, "right_in_grid_or_program": either, "programs_ran": ran,
                     "verified_clean_outputs": vclean, "verified_clean_right": vclean_right,
                     "verified_clean_wrong": vclean - vclean_right,
                     "examples": sum(r.get("n_examples", 0) for r in rr), "with_rule_line": sum(r.get("n_rule_lines", 0) for r in rr)}
    gains, losses, units, by_stage, by_rep = 0, 0, 0, {}, {}
    for (arm, unit, draw), r in by.items():
        if arm != "plain" or ("block", unit, draw) not in by:
            continue
        b = by[("block", unit, draw)]
        t = tasks[r["task_id"]]
        for j in range(len(t["test"])):
            truth = sols[r["task_id"]][j]
            pr, br = top(t, r, j) == truth, top(t, b, j) == truth
            units += 1
            g, l_ = int(br and not pr), int(pr and not br)
            gains += g
            losses += l_
            for key, d in ((r.get("stage"), by_stage), (r.get("rep"), by_rep)):
                e = d.setdefault(str(key), {"gains": 0, "losses": 0})
                e["gains"] += g
                e["losses"] += l_
    p = p_one_sided(gains, losses)
    valid = (len(cuts) >= 15 and all(arms[a]["median_tokens"] >= 5 for a in ("plain", "block"))
             and all(arms[a]["empty_or_zero"] <= 0.05 * max(1, arms[a]["records"]) for a in ("plain", "block")))
    extra_vwrong = arms["block"]["verified_clean_wrong"] - arms["plain"]["verified_clean_wrong"]
    net = gains - losses
    if not valid:
        decision = "no decision (validity failed): report only"
    elif net >= 3 and p <= 0.10 and extra_vwrong <= 1:
        decision = "block helps: evidence for the block in v8's rescue request (joins v8 only if the pool is published in time)"
    elif net <= 0:
        decision = "block does not help: v8's rescue request stays v7-vr2's"
    else:
        decision = "report only: v8's rescue request stays v7-vr2's"
    # the VR-bank's own recorded rescue of the same cut samples (one sample, as recorded)
    recorded = {"units": 0, "right": 0}
    for rep in ("A", "B"):
        f = VRBANK / f"rep_{rep}" / "vr" / "samples.jsonl"
        if not f.is_file():
            continue
        ss = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for s in ss:
            if s["stage"].startswith("rescue") and f"{rep}:{s.get('parent')}" in cuts:
                t = tasks.get(s["task_id"])
                if t is None:
                    continue
                for j in range(len(t["test"])):
                    recorded["units"] += 1
                    recorded["right"] += top(t, s, j) == sols[s["task_id"]][j]
    return {"rule": "experiments/arc2-q27/RESCUE_REPLAY_RULE.md (frozen 2026-10-27 23:51 IST)", "decision": decision,
            "valid": valid, "cut_samples": len(cuts), "paired_units": units, "gains": gains, "losses": losses,
            "net": net, "sign_test_p_one_sided": round(p, 4), "extra_verified_clean_wrong": extra_vwrong,
            "thresholds": {"net": 3, "p": 0.10, "extra_verified_wrong": 1}, "arms": arms,
            "gains_losses_by_stage": by_stage, "gains_losses_by_replicate": by_rep,
            "vrbank_recorded_rescue_of_the_same_cuts": recorded}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    res = analyze(a.run)
    (a.run / "replay_decision.json").write_text(json.dumps(res, indent=1) + "\n")
    ar = res["arms"]
    print(f"rescue replay rule: {res['decision']} | {res['cut_samples']} cut samples, {res['paired_units']} paired units: "
          f"gains {res['gains']}, losses {res['losses']}, net {res['net']}, p {res['sign_test_p_one_sided']}, extra "
          f"verified-wrong {res['extra_verified_clean_wrong']}")
    for k in ("plain", "block", "R1", "R2"):
        print(f"  {k}: {ar[k]}")
    print(f"  VR-bank's recorded rescue of the same cuts: {res['vrbank_recorded_rescue_of_the_same_cuts']}")
    print(f"wrote {a.run / 'replay_decision.json'}")


if __name__ == "__main__":
    main()
