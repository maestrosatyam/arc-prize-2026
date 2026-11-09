#!/usr/bin/env python3
"""DC1 analysis (experiments/arc2-decider/DC1_RULE.md), after the kernel's archive arrives. CPU only.

  1. Validity (pre-set): the CPU twin passed (its log in the archive says "preflight: PASS"); the run's record names no
     problem (the sealed split's problems are recorded apart) and is not a twin's; the unsealed file is complete; every
     unsealed output with two or more B0 candidates outside the over_16384 tier has both orders answered.
  2. Gate: scripts/arc2_bank_replay.py for the four dc1 rules (dc_fit, dc_long, dc_keep1, dc_keep1_long) against b0 on
     the 145 unsealed outputs (the ledger lines are written unless --no-ledger); both gate versions; adoption under v2
     inside dc1.
  3. The sealed check: the exact command, printed, not run (--sealed --final-pick, once).
  4. Recorded, no decision: outputs per tier (fit_4096, long_16384, over_16384); the decider's rank of the right grid
     where B0's pool holds it but B0 missed it, and on B0's right outputs; its top choice against B0's top; the
     calibration line (the decider's mean top probability where its top is right against where it is wrong, on the
     outputs with the right grid in the pool, and the two orders' agreement rate on the top); seconds per request.
  5. One decision line.

Usage: python scripts/arc2_dc1_analyze.py [--archive experiments/runs/kaggle-A2-DC1-v1] [--no-ledger] [--json out]
Prints "pending" while the archive has no probabilities.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import arc2_bank as B  # noqa: E402

ARCHIVE = ROOT / "experiments" / "runs" / "kaggle-A2-DC1-v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rel(p: Path) -> Path:
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


def twin_passed(archive: Path) -> bool:
    logs = list((archive / "preflight").glob("*.log")) + list((archive / "preflight").glob("*.txt"))
    return any("preflight: PASS" in p.read_text(errors="replace") for p in logs)


def validity(archive: Path, doc: dict, run: dict, bank) -> dict:
    problems = list(run.get("problems") or [])
    if run.get("twin") or doc.get("twin"):
        problems.append("the archive holds a twin's file")
    over = {(t, o["test_index"]) for t, outs in (doc.get("tasks") or {}).items() for o in outs
            if o.get("tier") == "over_16384"}
    want = sum(1 for o in bank.outputs() if len(o.ranked("rank_kgmon")) >= 2 and (o.task_id, o.test_index) not in over)
    got = sum(1 for outs in (doc.get("tasks") or {}).values() for o in outs
              if len([r for r in o.get("orders") or [] if r.get("by_candidate")]) == 2)
    if not doc.get("complete"):
        problems.append("the unsealed file is not marked complete")
    if got != want:
        problems.append(f"{got} unsealed outputs answered in both orders, {want} have two or more B0 candidates")
    if not twin_passed(archive):
        problems.append("no 'preflight: PASS' in the archived twin log")
    return {"ok": not problems, "problems": problems, "answered": got, "eligible": want, "over_16384": len(over)}


def gate(probs_path: Path, no_ledger: bool, n_boot: int) -> dict:
    R = _load("arc2_bank_replay", ROOT / "scripts" / "arc2_bank_replay.py")
    args = argparse.Namespace(rule=list(B.DC1_RULES), base="b0", pools=None, q27="", tasks="", sealed=False,
                              final_pick=False, n_boot=n_boot, seed=0, rules_file=str(R.RULES_FILE), ledger=str(R.LEDGER),
                              no_ledger=no_ledger, null_calibration=str(R.NULL_CALIBRATION), json="", glimmer="",
                              glimmer_sealed="", decider=str(probs_path), decider_sealed="")
    rep = R.run(args)
    return {r["rule"]: {k: r[k] for k in ("n_right", "n_outputs", "gained", "lost", "net", "gate_v2", "gate_v1",
                                          "sign_test_p", "gained_keys", "lost_keys", "registration")}
            for r in rep["rules"]} | {"_base": rep["base"]}


def recorded(doc: dict, bank) -> dict:
    per = {(t, o["test_index"]): o for t, outs in (doc.get("tasks") or {}).items() for o in outs}
    head, b0_right, agree_b0, agree_orders, n, secs, fit = [], [], 0, 0, 0, [], 0
    tiers, cal = {}, {"right": [], "wrong": []}
    for o in bank.outputs():
        rec = per.get((o.task_id, o.test_index))
        cands = o.ranked("rank_kgmon")
        if rec is None or len(cands) < 2:
            continue
        tiers[rec.get("tier")] = tiers.get(rec.get("tier"), 0) + 1
        if rec.get("tier") == "over_16384":
            continue
        n += 1
        fit += bool(rec.get("fit_4096"))
        probs = {B.grid_key(c["grid"]): c.get("prob_mean") for c in rec.get("candidates") or []}
        pv = [probs.get(B.grid_key(c["grid"])) for c in cands]
        ranked = sorted(range(len(cands)), key=lambda i: (pv[i] is None, -(pv[i] or 0.0), i))
        agree_b0 += ranked[0] == 0
        bys = [r["by_candidate"] for r in rec.get("orders") or [] if r.get("by_candidate")]
        tops = [max(range(len(b)), key=lambda i: (b[i] if b[i] is not None else -1.0, -i)) for b in bys]
        agree_orders += len(tops) == 2 and tops[0] == tops[1]
        secs += [r["seconds"] for r in rec.get("orders") or [] if r.get("seconds") is not None]
        truth = bank.truth(o)
        if truth is None:
            continue
        tk = B.grid_key(truth)
        pos = next((i for i, c in enumerate(cands) if B.grid_key(c["grid"]) == tk), None)
        if pos is None:
            continue
        if pv[ranked[0]] is not None:
            cal["right" if ranked[0] == pos else "wrong"].append((pv[ranked[0]], len(cands)))
        b0_hit = tk in {B.grid_key(o.attempt_1), B.grid_key(o.attempt_2)}
        row = {"output": o.key, "b0_rank": pos + 1, "decider_rank": ranked.index(pos) + 1, "of": len(cands),
               "fit_4096": bool(rec.get("fit_4096"))}
        (b0_right if b0_hit else head).append(row)
    q = sorted(secs)
    mean = lambda v: round(sum(v) / len(v), 4) if v else None   # noqa: E731
    # fewer options give both a higher top probability and a better chance of being right: each group's mean option
    # count is recorded beside its mean top probability
    calibration = {g: {"n": len(cal[k]), "mean_top_probability": mean([p for p, _ in cal[k]]),
                       "mean_options": mean([m for _, m in cal[k]])}
                   for g, k in (("top_right", "right"), ("top_wrong", "wrong"))}
    calibration.update({
                        "orders_agree_on_top_rate": round(agree_orders / n, 4) if n else None})
    return {"outputs": n, "tiers": tiers, "fit_4096": fit, "calibration": calibration,
            "decider_top_equals_b0_top": agree_b0,
            "orders_agree_on_top": agree_orders, "headroom": head, "b0_right": b0_right,
            "b0_right_kept_in_top2": sum(r["decider_rank"] <= 2 for r in b0_right),
            "headroom_lifted_to_top2": sum(r["decider_rank"] <= 2 for r in head),
            "seconds_per_request": {"median": statistics.median(q) if q else None,
                                    "p90": q[int(0.9 * (len(q) - 1))] if q else None, "n": len(q)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    arch = Path(a.archive)
    pp, sp = arch / "dc1" / "dc1_probs_unsealed.json", arch / "dc1" / "sealed" / "dc1_probs_sealed.json"
    if not pp.is_file():
        print(f"DC1: pending (no {pp})")
        return 0
    doc = json.loads(pp.read_text())
    rp = arch / "dc1" / "dc1_run.json"
    run = json.loads(rp.read_text()) if rp.is_file() else {"problems": ["no dc1_run.json"]}
    bank = B.load_bank(decider=pp)
    v = validity(arch, doc, run, bank)
    print(f"DC1 validity: {'PASS' if v['ok'] else 'FAIL'}: {v['answered']} of {v['eligible']} eligible outputs answered "
          f"in both orders ({v['over_16384']} over 16,384 recorded, not asked); problems {v['problems'][:8]}; sealed "
          f"split problems (recorded apart) {(run.get('sealed_problems') or [])[:5]}")
    print(f"model {run.get('model')}; environment {run.get('environment')}")
    g = gate(pp, a.no_ledger or not v["ok"], a.n_boot)
    rec = recorded(doc, bank)
    print(f"calibration (recorded): {rec['calibration']}")
    print(f"recorded: tiers {rec['tiers']}; {rec['outputs']} outputs ranked, {rec['fit_4096']} fit 4,096 in both orders; decider top = B0 "
          f"top on {rec['decider_top_equals_b0_top']}; the two orders agree on the top on {rec['orders_agree_on_top']}; "
          f"seconds per request {rec['seconds_per_request']}")
    print(f"  headroom (right in B0's pool, missed by B0): {rec['headroom']}; lifted into the top two: "
          f"{rec['headroom_lifted_to_top2']} of {len(rec['headroom'])}")
    print(f"  B0's right outputs kept in the decider's top two: {rec['b0_right_kept_in_top2']} of {len(rec['b0_right'])}")
    cmd = (f"envs/kaggle-sim/bin/python scripts/arc2_bank_replay.py --sealed --final-pick --rule <rule> "
           f"--decider {_rel(pp)} --decider-sealed {_rel(sp)}")
    passing = [r for r in B.DC1_RULES if g[r]["gate_v2"]]
    if not v["ok"]:
        decision = "INVALID: the pre-set validity checks failed; no gate reading counts"
    elif passing:
        decision = (f"gate v2 PASSED by {passing} (pending the choice of gate); next: the sealed sign check "
                    f"once: {cmd}")
    else:
        decision = ("DROP: neither dc1 rule passes gate v2 ("
                    + "; ".join(f"{r} net {g[r]['net']:+d}, lost {g[r]['lost']}" for r in B.DC1_RULES) + ")")
    print(f"DC1 decision: {decision}")
    if a.json:
        Path(a.json).write_text(json.dumps({"validity": v, "gate": g, "recorded": rec, "decision": decision},
                                           indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
