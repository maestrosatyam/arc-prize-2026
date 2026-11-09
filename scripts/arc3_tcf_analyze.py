#!/usr/bin/env python3
"""TCF analysis (ARC-3, run a3-tcf-public): ThinkingCap-ka2's model on the 25 public games, two passes, against P0's
public commit run (25 games x 4 passes, mean 48.96), under the frozen rule (reports/research/
ARC3_PRE_RESET_RULES_2026-10-29.md, section TCF, 731383e; stamped in a13ca99).

Validity:
  - the twin passed (a log under the archive's preflight/ reads "preflight: PASS");
  - the view assembled to layout tcv-2663e102-on-intel-861536dd (the kernel log's "r1u view:" line);
  - the server loaded with no error: serve.log reaches "The server is fired up and ready to roll!" with no traceback or
    error line before it (the sitecustomize/wrapt notice aside);
  - at least 48 of 50 game-passes end with a scorecard (state won, gave_up or cancelled, with a final score).
Reading: D = TCF's mean game score minus P0's, with WM1's game-paired permutation test (experiments/arc3-wm/
wm_report.py: pass labels shuffled within each game, 20,000 draws, seed 0, two-sided).
  - weaker on full games: D <= -5 and p <= 0.05;
  - not weaker: D >= -2;
  - otherwise: report.
Revision 2 (17:13 IST Oct 29, before any TCF data): the "weaker" line stands
only if TCF's median generated tokens per game-pass (benchmark.json's per-action generated_tokens, summed per game-pass)
is at least 80% of P0's; below that it reads "report (time-bound)". "Not weaker" and "report" are unchanged. Recorded
beside, no decision: P0 cut at 14,509 s of each game's wall time and rescored (a lower bound for P0), and P0 cut per game
at TCF's median generated tokens for that game and rescored (compute-matched). --baseline-only prints the baseline-side
values (P0's median tokens per game-pass and the wall-cut P0 mean) without reading a TCF archive.
Exactly one decision line; with validity failed it reads "no decision (validity failed): report only".
Recorded beside: levels cleared per pass, moves per clear, tokens per action, first model action time, per-game deltas,
and the keep-alive's RESETs; and (17:12 IST) generated tokens per game-pass beside P0's, overall and
per game, with the wall time per game-pass, since P0's formula gives each TCF game half P0's time (14,509 s at
concurrency 50 against 29,018 s at 100).

  envs/arc3/bin/python -I scripts/arc3_tcf_analyze.py --run experiments/runs/kaggle-A3-TCF-public-v1 \
      --baseline experiments/runs/kaggle-P0-commit-v1
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

LAYOUT = "tcv-2663e102-on-intel-861536dd"
P0_MEAN = 48.96
N_EXPECTED, MIN_SCORECARDS = 50, 48
SCORECARD_STATES = {"won", "gave_up", "cancelled"}
READY = "The server is fired up and ready to roll!"
SERVE_ERROR = re.compile(r"Traceback|Error|error|CUDA out of memory|Killed|exited")
SERVE_NOISE = ("sitecustomize", "wrapt")
TCF_GAME_S = 532 * 60 * 50 // 110          # each TCF game's runtime (P0's formula at concurrency 50): 14,509 s
COMPUTE_GUARD = 0.8                        # revision 2: "weaker" needs TCF's median tokens per game-pass >= 80% of P0's
RULE = ("TCF, frozen (16:3x IST Oct 29; ARC3_PRE_RESET_RULES_2026-10-29.md at 731383e, "
        "stamped in a13ca99; revision 2, 17:13 IST Oct 29): validity (twin passed; layout "
        "tcv-2663e102-on-intel-861536dd; server loads with no error; >= 48/50 game-passes end with a scorecard); D = TCF "
        "mean game score - P0's 48.96, game-paired permutation on per-game means as WM1: weaker on full games at D <= -5 "
        "and p <= 0.05, standing only if TCF's median generated tokens per game-pass >= 80% of P0's (else report "
        "(time-bound)); not weaker at D >= -2; otherwise report")


def _wm_report():
    """WM1's loader and permutation test (experiments/arc3-wm/wm_report.py), from this checkout or the working dir."""
    for root in (Path(__file__).resolve().parents[1], Path.cwd()):
        p = root / "experiments" / "arc3-wm" / "wm_report.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("wm_report", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["wm_report"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("experiments/arc3-wm/wm_report.py")


def band(d: float, p: float) -> str:
    if d <= -5 and p <= 0.05:
        return "weaker on full games"
    if d >= -2:
        return "not weaker"
    return "report"


def guarded(reading: str, tcf_median_tokens: float, p0_median_tokens: float) -> str:
    """Revision 2: the "weaker" line stands only with TCF's median tokens per game-pass at least 80% of P0's."""
    if reading == "weaker on full games" and tcf_median_tokens < COMPUTE_GUARD * p0_median_tokens:
        return "report (time-bound)"
    return reading


def with_walls(wm, run_dir: Path, tag: str) -> list[dict]:
    """WM1's rows, each with the game wall time (history wallclock_seconds, from game start) at each cleared level."""
    rows = wm.load_rows(run_dir, tag)
    runs = json.loads((run_dir / "benchmark.json").read_text())["game_runs"]
    assert len(rows) == len(runs)
    for row, r in zip(rows, runs):
        walls = [float(x.get("wallclock_seconds") or 0.0) for x in r["history"]]
        acc, row["clear_wall"] = 0, []
        for a in row["apl"][:row["cleared"]]:
            acc += a
            row["clear_wall"].append(walls[min(len(walls), acc) - 1] if walls else 0.0)
    return rows


def wall_cut_score(wm, row: dict, cut_s: float) -> float:
    k = 0
    while k < len(row["clear_wall"]) and row["clear_wall"][k] <= cut_s:
        k += 1
    return wm.AK.game_score(row["apl"], row["base"], k)


def token_cut_score(wm, row: dict, budget: float) -> float:
    k = 0
    while k < len(row["clear_tok"]) and row["clear_tok"][k] <= budget:
        k += 1
    return wm.AK.game_score(row["apl"], row["base"], k)


def baseline_side(wm, baseline: Path) -> dict:
    """The values fixed before any TCF data: P0's median tokens per game-pass, the guard's floor, the wall-cut P0 mean."""
    B = with_walls(wm, baseline, "P0")
    med = st.median(r["tokens"] for r in B)
    return {"p0_game_passes": len(B), "p0_mean": round(st.mean(r["score"] for r in B), 2),
            "p0_median_tokens_per_game_pass": med, "guard_floor_tokens": round(COMPUTE_GUARD * med, 1),
            "p0_wall_cut_s": TCF_GAME_S, "p0_wall_cut_mean": round(st.mean(wall_cut_score(wm, r, TCF_GAME_S) for r in B), 2),
            "p0_rows_rescore_check": all(abs(wall_cut_score(wm, r, float("inf")) - r["score"]) < 1e-6 for r in B)}


def kernel_log_text(run: Path) -> str:
    """The kernel log the queue saves (<slug>.log: a JSON list of stream records), as one text."""
    logs = sorted(p for p in run.glob("*.log") if p.name != "post_command.log")
    out = []
    for p in logs:
        raw = p.read_text(errors="replace")
        try:
            out.append("".join(r.get("data", "") for r in json.loads(raw)))
        except (json.JSONDecodeError, AttributeError):
            out.append(raw)
    return "\n".join(out)


def view_layout(text: str) -> str | None:
    m = re.search(r"r1u view: \{.*?'layout': '([^']+)'", text)
    return m.group(1) if m else None


def keepalive(text: str) -> dict:
    first = None
    m = re.search(r"keep-alive: watchdog stopped; (\d+) RESET\(s\); first model action (None|\d+) s after start", text)
    if m:
        first = None if m.group(2) == "None" else int(m.group(2))
    resets = []
    m2 = re.search(r"keep-alive: \d+ RESET\(s\) (\[.*?\]); first model action", text)
    if m2:
        try:
            resets = [tuple(x) for x in ast.literal_eval(m2.group(1))]
        except (ValueError, SyntaxError):
            resets = []
    return {"found": bool(m), "first_model_action_s": first, "resets": len(resets) if m2 else (int(m.group(1)) if m else None),
            "reset_games": dict(Counter(g for _, g in resets))}


def server_load(run: Path) -> dict:
    p = run / "serve.log"
    if not p.is_file():
        return {"ok": False, "detail": "serve.log missing"}
    lines = p.read_text(errors="replace").splitlines()
    ready = next((i for i, x in enumerate(lines) if READY in x), None)
    upto = lines if ready is None else lines[:ready]
    bad = [x[:200] for x in upto if SERVE_ERROR.search(x) and not any(n in x for n in SERVE_NOISE)]
    after = sum(1 for x in lines[ready:] if "Traceback" in x) if ready is not None else None
    return {"ok": ready is not None and not bad, "ready_line": ready, "errors_before_ready": bad[:5],
            "tracebacks_after_ready": after}


def twin_passed(run: Path) -> bool:
    logs = list((run / "preflight").glob("*.log")) if (run / "preflight").is_dir() else []
    return any("preflight: PASS" in p.read_text(errors="replace") for p in logs)


def scorecards(bench: dict) -> dict:
    runs = bench.get("game_runs", [])
    ok = [r for r in runs if r.get("state") in SCORECARD_STATES and isinstance(r.get("final_score"), (int, float))]
    return {"game_passes": len(runs), "with_scorecard": len(ok), "states": dict(Counter(r.get("state") for r in runs))}


def moves_per_clear(rows: list[dict]) -> dict:
    per = [a for r in rows for a in r["apl"][:r["cleared"]]]
    return {"levels": len(per), "median": st.median(per) if per else None,
            "mean": round(st.mean(per), 1) if per else None}


def side(rows: list[dict]) -> dict:
    passes = sorted({r["pass"] for r in rows})
    actions, tokens = sum(r["actions"] for r in rows), sum(r["tokens"] for r in rows)
    return {"game_passes": len(rows), "mean_score": round(st.mean(r["score"] for r in rows), 2) if rows else None,
            "levels_per_pass": {p: round(st.mean(r["cleared"] for r in rows if r["pass"] == p), 2) for p in passes},
            "levels_mean": round(st.mean(r["cleared"] for r in rows), 2) if rows else None,
            "won": sum(r["won"] for r in rows), "moves_per_clear": moves_per_clear(rows),
            "tokens_per_action": round(tokens / actions, 1) if actions else None, "actions": actions, "tokens": tokens,
            "tokens_per_game_pass": {"mean": round(tokens / len(rows)) if rows else None,
                                     "median": round(st.median(r["tokens"] for r in rows)) if rows else None},
            "wall_s_per_game_pass_median": round(st.median(r["wall_s"] or 0 for r in rows)) if rows else None}


def analyze(run: Path, baseline: Path) -> dict:
    wm = _wm_report()
    bench_path = run / "benchmark.json"
    if not bench_path.is_file():
        return {"decision": "pending", "reason": f"{bench_path} missing", "rule": RULE}
    bench = json.loads(bench_path.read_text())
    R_all, B = wm.load_rows(run, "TCF"), with_walls(wm, baseline, "P0")
    R = [r for r in R_all if isinstance(r["score"], (int, float))]   # a game-pass without a score is left out, counted
    d, p = wm.permutation(B, R)
    text = kernel_log_text(run)
    layout, ka, serve, sc = view_layout(text), keepalive(text), server_load(run), scorecards(bench)
    checks = {"twin_passed": twin_passed(run), "same_layout": layout == LAYOUT, "server_loaded_no_error": serve["ok"],
              "scorecards_48_of_50": sc["with_scorecard"] >= MIN_SCORECARDS and sc["game_passes"] <= N_EXPECTED}
    valid = all(checks.values())
    tcf_med = st.median(r["tokens"] for r in R) if R else 0
    p0_med = st.median(r["tokens"] for r in B)
    reading = guarded(band(d, p), tcf_med, p0_med)
    base_side = baseline_side(wm, baseline)
    tcf_game_med = {g: st.median(r["tokens"] for r in R if r["game"] == g) for g in {r["game"] for r in R}}
    matched = [token_cut_score(wm, r, tcf_game_med[r["game"]]) for r in B if r["game"] in tcf_game_med]
    games = sorted({r["game"] for r in R} | {r["game"] for r in B})
    per_game = []
    for g in games:
        r_, b_ = [x["score"] for x in R if x["game"] == g], [x["score"] for x in B if x["game"] == g]
        rt, bt = [x["tokens"] for x in R if x["game"] == g], [x["tokens"] for x in B if x["game"] == g]
        per_game.append({"game": g, "tcf": round(st.mean(r_), 2) if r_ else None, "p0": round(st.mean(b_), 2) if b_ else None,
                         "delta": round(st.mean(r_) - st.mean(b_), 2) if r_ and b_ else None, "tcf_passes": len(r_),
                         "tcf_tokens": round(st.mean(rt)) if rt else None, "p0_tokens": round(st.mean(bt)) if bt else None})
    return {
        "rule": RULE, "run": str(run), "baseline": str(baseline),
        "validity": {"valid": valid, "checks": checks, "layout": layout, "server": serve, "scorecards": sc,
                     "left_out_no_score": [f"{r['game']}_p{r['pass']}" for r in R_all if r not in R]},
        "d": round(d, 2), "p": round(p, 4), "p0_mean_rows": round(st.mean(r["score"] for r in B), 2),
        "p0_mean_rule": P0_MEAN, "d_vs_rule_mean": round(st.mean(r["score"] for r in R) - P0_MEAN, 2) if R else None,
        "reading": reading, "band_unguarded": band(d, p),
        "compute": {"tcf_median_tokens_per_game_pass": tcf_med, "p0_median_tokens_per_game_pass": p0_med,
                    "ratio": round(tcf_med / p0_med, 3) if p0_med else None, "guard": COMPUTE_GUARD,
                    "guard_met": tcf_med >= COMPUTE_GUARD * p0_med},
        "recorded_baselines": {"p0_wall_cut": {"cut_s": TCF_GAME_S, "mean": base_side["p0_wall_cut_mean"],
                                               "d_against": round(st.mean(r["score"] for r in R) - base_side["p0_wall_cut_mean"], 2)
                                               if R else None},
                               "p0_compute_matched": {"budget": "TCF's median generated tokens for each game",
                                                      "game_passes": len(matched),
                                                      "mean": round(st.mean(matched), 2) if matched else None,
                                                      "d_against": round(st.mean(r["score"] for r in R) - st.mean(matched), 2)
                                                      if matched and R else None}},
        "baseline_side": base_side,
        "decision": reading if valid else "no decision (validity failed): report only",
        "tcf": side(R), "p0": side(B), "first_model_action_s": ka["first_model_action_s"], "keepalive": ka,
        "per_game": sorted(per_game, key=lambda x: (x["delta"] is None, x["delta"] if x["delta"] is not None else 0)),
        "wall": {"start": bench.get("start_time"), "end": bench.get("end_time")},
    }


def format_report(rep: dict) -> str:
    if rep["decision"] == "pending":
        return f"TCF: {rep['reason']}\nRule: {rep['rule']}\nDecision: pending\n"
    v, t, b = rep["validity"], rep["tcf"], rep["p0"]
    L = ["TCF (a3-tcf-public): ThinkingCap-ka2's model on the 25 public games, 2 passes, against P0's public commit run",
         "",
         f"validity: {'valid' if v['valid'] else 'INVALID'} {v['checks']}",
         f"  layout {v['layout']}; server ready line {v['server']['ready_line']}, errors before it "
         f"{v['server']['errors_before_ready']}, tracebacks after it {v['server']['tracebacks_after_ready']}",
         f"  scorecards {v['scorecards']['with_scorecard']}/{v['scorecards']['game_passes']} "
         f"(need {MIN_SCORECARDS} of {N_EXPECTED}); states {v['scorecards']['states']}; left out of D (no score) "
         f"{v['left_out_no_score']}",
         "",
         "| Track | Run | game-passes | mean score | levels per pass | levels mean | won | moves per clear (median; mean) "
         "| tokens per action | generated tokens per game-pass (mean; median) | wall s per game-pass (median) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("TCF (ThinkingCap-ka2's model)", t), ("P0 (public commit run)", b)):
        m = s["moves_per_clear"]
        L.append(f"| ARC-3 | {name} | {s['game_passes']} | {s['mean_score']} | {s['levels_per_pass']} | {s['levels_mean']} "
                 f"| {s['won']} | {m['median']}; {m['mean']} | {s['tokens_per_action']} | "
                 f"{s['tokens_per_game_pass']['mean']}; {s['tokens_per_game_pass']['median']} | "
                 f"{s['wall_s_per_game_pass_median']} |")
    ka = rep["keepalive"]
    L += ["",
          f"D = {rep['d']:+.2f} (TCF {t['mean_score']} - P0 {rep['p0_mean_rows']}; against the rule's 48.96: "
          f"{rep['d_vs_rule_mean']:+.2f}), game-paired permutation p {rep['p']:.4f} (20,000 draws)",
          f"compute guard (revision 2): TCF median generated tokens per game-pass {rep['compute']['tcf_median_tokens_per_game_pass']} "
          f"against P0's {rep['compute']['p0_median_tokens_per_game_pass']} (ratio {rep['compute']['ratio']}; "
          f"'weaker' stands at >= {COMPUTE_GUARD}): {'met' if rep['compute']['guard_met'] else 'not met'}",
          f"recorded, no decision: P0 cut at {TCF_GAME_S} s of game wall time and rescored "
          f"{rep['recorded_baselines']['p0_wall_cut']['mean']} (D against it "
          f"{rep['recorded_baselines']['p0_wall_cut']['d_against']}); P0 cut per game at TCF's median tokens and rescored "
          f"{rep['recorded_baselines']['p0_compute_matched']['mean']} over "
          f"{rep['recorded_baselines']['p0_compute_matched']['game_passes']} game-passes (D against it "
          f"{rep['recorded_baselines']['p0_compute_matched']['d_against']})",
          f"first model action {rep['first_model_action_s']} s after start; keep-alive RESETs {ka['resets']} "
          f"{ka['reset_games']}",
          f"wall {rep['wall']['start']} -> {rep['wall']['end']}",
          "time per game (P0's save-run formula scales it with concurrency): TCF 14,509 s at concurrency 50, P0 29,018 s "
          "at concurrency 100; generated tokens per game-pass above show whether the halving bound",
          "",
          "per-game deltas (TCF - P0, mean over passes; generated tokens per game-pass TCF vs P0):"]
    L += [f"  {x['game']}: {x['tcf']} vs {x['p0']} ({x['delta']:+.2f}; {x['tcf_passes']} passes; tokens per game-pass "
          f"{x['tcf_tokens']} vs {x['p0_tokens']})"
          if x["delta"] is not None else f"  {x['game']}: TCF {x['tcf']} P0 {x['p0']}" for x in rep["per_game"]]
    L += ["", f"Rule: {rep['rule']}",
          f"Reading: {rep['reading']} (D {rep['d']:+.2f}, p {rep['p']:.4f}; band before the compute guard: "
          f"{rep['band_unguarded']})",
          f"Decision: {rep['decision']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None, help="default: <run>/tcf_decision.json")
    ap.add_argument("--baseline-only", action="store_true", help="print the baseline-side values; read no TCF archive")
    a = ap.parse_args(argv)
    if a.baseline_only:
        bs = baseline_side(_wm_report(), a.baseline)
        print(f"TCF baseline side (before any TCF data): P0 {bs['p0_game_passes']} game-passes, mean {bs['p0_mean']}; "
              f"median generated tokens per game-pass {bs['p0_median_tokens_per_game_pass']} (the guard's 80% floor "
              f"{bs['guard_floor_tokens']}); P0 cut at {bs['p0_wall_cut_s']} s of game wall time and rescored: mean "
              f"{bs['p0_wall_cut_mean']}; full-time rescoring reproduces every P0 score: {bs['p0_rows_rescore_check']}")
        return
    if a.run is None:
        ap.error("--run is required unless --baseline-only")
    rep = analyze(a.run, a.baseline)
    print(format_report(rep), end="")
    out = a.json or a.run / "tcf_decision.json"
    if a.run.is_dir():
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
