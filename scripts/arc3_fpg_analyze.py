#!/usr/bin/env python3
"""FPG analysis (run a3-fpg-public): the frozen rule reports/research/ARC3_FPG_FP8_PUBLIC_RULE_2026-10-30.md (e672dbe;
00:34 IST Oct 30) on the run's benchmark.json, against P0's public commit run
(experiments/runs/kaggle-P0-commit-v1: 100 game-passes, mean 48.96). OIF's analyzer (scripts/arc3_oif_analyze.py)
supplies the loaders and the side lines; there is no message here.

Validity (any failing: no decision, report only):
  - the CPU twin passed ("preflight: PASS" in the archived twin log; it carries the settings line and A3-L1's check 5b);
  - the server loaded with no error (serve.log reaches the ready line with no error before it);
  - at least 96 of 100 game-passes end with a scorecard.
Reading: D = FPG's mean game score - P0's 48.96, game-paired permutation on per-game means (pass labels shuffled within
each game), two-sided, 20,000 draws, seed 0 (WM1's test):
  FP8 loses game skill    D <= -5 and p <= 0.05
  no measurable loss      D >= -2
  report                  otherwise
Recorded beside, no decision: the first model action time (load); levels per pass and wins; per-game deltas; moves per
clear; tokens per action and per game-pass; wall-budget endings.

  envs/arc3/bin/python -I scripts/arc3_fpg_analyze.py [--archive experiments/runs/kaggle-A3-FPG-public-v1]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = ROOT / "experiments" / "runs" / "kaggle-A3-FPG-public-v1"
LOSS_D, LOSS_P, NO_LOSS_D = -5.0, 0.05, -2.0
RULE = ("FPG, frozen (reports/research/ARC3_FPG_FP8_PUBLIC_RULE_2026-10-30.md at e672dbe; "
        "00:34 IST Oct 30): validity (twin passed; server loads with no error; >= 96 "
        "of 100 game-passes scored); D = FPG mean game score - P0's 48.96, game-paired permutation, two-sided, 20,000 "
        "draws: D <= -5 and p <= 0.05 FP8 loses game skill; D >= -2 no measurable loss; otherwise report")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


A = _load("arc3_oif_analyze", HERE / "arc3_oif_analyze.py")
P0_RUN, P0_MEAN, GAMES = A.P0_RUN, A.P0_MEAN, A.GAMES


def band(d: float, p: float) -> str:
    if d <= LOSS_D and p <= LOSS_P:
        return "FP8 loses game skill"
    if d >= NO_LOSS_D:
        return "no measurable loss"
    return "report"


def analyze(archive: Path, p0_path: Path = P0_RUN / "benchmark.json") -> dict:
    out: dict = {"rule": RULE, "archive": str(archive)}
    twin = A.twin_passed(archive)
    bench = archive / "benchmark.json"
    if not bench.is_file():
        out.update({"decision": "pending: no benchmark.json yet", "twin": twin})
        return out
    fpg, p0 = A.load_runs(bench), A.load_runs(p0_path)
    passes = int(json.loads(bench.read_text(encoding="utf-8")).get("n_passes") or 0)
    scored = [r for r in fpg if r["state"] in A.SCORECARD_STATES and isinstance(r["score"], (int, float))]
    expected = GAMES * passes
    need = math.ceil(A.MIN_SCORECARDS_SHARE * expected)
    serve = A.server_load(archive)
    checks = {"twin_passed": twin is True, "server_loaded_no_error": serve["ok"],
              "scorecards_96": expected > 0 and len(scored) >= need}
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "passes": passes, "game_passes": len(fpg),
                       "scored": len(scored), "scored_needed": need, "server": serve,
                       "left_out_no_score": [f"{r['game']}_p{r['pass']}" for r in fpg if r not in scored]}
    d, p = A.WM.permutation([{"game": r["game"], "score": r["score"]} for r in p0],
                            [{"game": r["game"], "score": r["score"]} for r in scored])
    b = band(d, p)
    out["reading"] = {"FPG_mean": round(st.mean(r["score"] for r in scored), 2) if scored else None,
                      "P0_mean": round(st.mean(r["score"] for r in p0), 2), "P0_rule_mean": P0_MEAN,
                      "D": round(d, 2), "p": round(p, 4), "band": b, "game_passes": len(scored),
                      "P0_game_passes": len(p0)}
    out["decision"] = b if out["validity"]["valid"] else "no decision (validity failed): report only"
    out["fpg"], out["p0"] = A.side(scored), A.side(p0)
    games = sorted({r["game"] for r in p0} | {r["game"] for r in scored})
    out["per_game"] = {}
    for g in games:
        f = [r["score"] for r in scored if r["game"] == g]
        q = [r["score"] for r in p0 if r["game"] == g]
        out["per_game"][g] = {"FPG": round(st.mean(f), 2) if f else None, "P0": round(st.mean(q), 2) if q else None,
                              "delta": round(st.mean(f) - st.mean(q), 2) if f and q else None}
    return out


def format_report(rep: dict) -> str:
    if "validity" not in rep:
        return f"FPG (a3-fpg-public): {rep['decision']}; twin {rep.get('twin')}\nDecision: {rep['decision']}\n"
    v, rd, f, b = rep["validity"], rep["reading"], rep["fpg"], rep["p0"]
    L = ["FPG (a3-fpg-public): the FP8 entry's model (A3-L1's FP8-PLE build) on the 25 full public games, on P0's "
         f"commit run; {v['passes']} passes, {v['game_passes']} game-passes",
         f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
         f"  scored {v['scored']} (need {v['scored_needed']}); server ready line {v['server'].get('ready_line')}, errors "
         f"before it {v['server'].get('errors_before_ready')}; left out of D (no score) {v['left_out_no_score']}",
         "",
         "| Track | Run | game-passes | mean score | levels per pass | won | moves per clear (median; mean) | tokens per "
         "action | tokens per game-pass (mean; median) | wall-budget endings | first model action s (min; median) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("P0 (Intel, public commit)", b), ("FPG (FP8-PLE, public save run)", f)):
        m, fa = s["moves_per_clear"], s["first_model_action_s"]
        L.append(f"| ARC-3 | {name} | {s['game_passes']} | {s['mean_score']} | {s['levels_per_pass']} | {s['won']} | "
                 f"{m['median']}; {m['mean']} | {s['tokens_per_action']} | {s['tokens_per_game_pass']['mean']}; "
                 f"{s['tokens_per_game_pass']['median']} | {s['wall_budget_endings']} | {fa['min']}; {fa['median']} |")
    L += ["",
          f"D = {rd['D']:+.2f} (FPG {rd['FPG_mean']} - P0 {rd['P0_mean']}; rule's P0 {rd['P0_rule_mean']}), game-paired "
          f"permutation, two-sided, 20,000 draws: p {rd['p']:.4f}; band: {rd['band']}",
          "",
          "| Track | game | FPG | P0 | delta |", "|---|---|---|---|---|"]
    L += [f"| ARC-3 | {g} | {x['FPG']} | {x['P0']} | {x['delta']} |" for g, x in rep["per_game"].items()]
    L += ["", f"Rule: {rep['rule']}", f"Decision: {rep['decision']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--json", type=Path, default=None, help="default: <archive>/fpg_decision.json")
    a = ap.parse_args(argv)
    rep = analyze(a.archive)
    print(format_report(rep), end="")
    if a.archive.is_dir() and "validity" in rep:
        (a.json or a.archive / "fpg_decision.json").write_text(json.dumps(rep, indent=1, default=str) + "\n",
                                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
