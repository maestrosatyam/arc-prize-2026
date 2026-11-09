#!/usr/bin/env python3
"""GS1 analysis (experiments/arc2-glimmer-scorer/GS1_RULE.md with its Revision 1), after the scoring kernel's archive
arrives. CPU only.

Revision 1 (2026-10-28 00:43 IST): only A2-S1's low-margin outputs are scored (B0's top-two margin below its
split's median), in one view (identity); the family is gs_low alone; the time gate uses B0's projected p90 end from the
bank's throughput model (E13) instead of 42,600 s.

  1. Validity (pre-set): the CPU twin passed (its log in the archive says "preflight: PASS"); the scorer's tokenizer
     checks passed (pinned channel ids, sp90's prompt sizes); every distinct candidate of the unsealed bank's
     low-margin outputs has a finite score in the view; the view's inverse restores every grid (the kernel's record per
     request, and again here); score ties at most 5% of the scored candidates (a candidate whose score equals another
     candidate's of the same output).
  2. Gate: scripts/arc2_bank_replay.py for gs_low against b0 on the 145 unsealed outputs (the ledger line is written
     unless --no-ledger); gate v2 (net >= 3 and lost <= 1). Gained, lost and net.
  3. The sealed check: the exact command, printed, not run (--sealed --final-pick, once; agree in sign = net > 0 on
     the sealed tasks).
  4. Time gate: the scorer's projected time for this scope on the 240-task hidden run = its measured start-up to a
     serving server + the hidden run's low-margin B0 candidates (bank rate per output x 240 x 1.08 outputs) x the
     measured seconds per request (one view). B0's projected p90 end = E13's p90 wall for B0 on the hidden run. It
     passes when B0's p90 end + the scorer's time + 1,800 s <= 43,200 s.
  5. Recorded, no decision: gained and lost outputs by task; Glimmer's rank of the right candidate where B0 missed it
     (scored outputs); seconds and tokens per candidate.
  6. One decision line.

Usage: python scripts/arc2_gs1_analyze.py [--archive experiments/runs/kaggle-A2-GS1-v1] [--no-ledger] [--json out.json]
Prints "pending" while the archive has no scores.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (BANK_PARTS, GS1_RULES, GS1_VIEWS, Q27_BANK, _q27_grids, glimmer_score,  # noqa: E402
                            grid_key, load_bank, load_glimmer_scores, low_margin_keys, replay)

DEFAULT_ARCHIVE = ROOT / "experiments" / "runs" / "kaggle-A2-GS1-v1"
LIMIT_S = 43200
SPARE_S = 1800
HIDDEN_TASKS = 240
HIDDEN_OUTPUTS_PER_TASK = 1.08
TIE_MAX = 0.05
RULE = "gs_low"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def views_mod():
    return _load("gs1_views", ROOT / "kaggle" / "arc2" / "gs1" / "gs1_views.py")


# ----------------------------------------------------------------------------
# validity

def scored_candidates(bank) -> dict:
    """{output key: [grid ...]} distinct candidates (B0's pool, then the 27B's grids) of the low-margin outputs, as the
    kernel's dataset has them."""
    low = low_margin_keys(bank)
    out = {}
    for t in bank.task_ids:
        task = bank.tasks[t]
        for j, o in enumerate(task.outputs):
            if o.key not in low:
                continue
            seen = []
            for g in [c["grid"] for c in o.candidates] + [g for _, g in _q27_grids(bank, task, j)]:
                if g not in seen:
                    seen.append(g)
            out[o.key] = seen
    return out


def validity(archive: Path, bank, scores: dict, run: dict) -> dict:
    V = views_mod()
    twin_logs = sorted((archive / "preflight").glob("*.log")) if (archive / "preflight").is_dir() else []
    twin_ok = any("preflight: PASS" in p.read_text(errors="replace") for p in twin_logs)
    chk = ((run.get("checks") or {}).get("server") or {})
    server_ok = bool(chk.get("ok")) and chk.get("header_ok") is True and not chk.get("sp90_mismatch")
    cands = scored_candidates(bank)
    missing, incomplete, rt_bad, n, ties = [], [], [], 0, 0
    for key, grids in cands.items():
        tid, j = key.rsplit("_", 1)
        per = (scores.get(tid) or {}).get(int(j)) or {}
        vals = []
        for g in grids:
            n += 1
            rec = per.get(grid_key(g))
            if rec is None:
                missing.append(key)
                continue
            s = glimmer_score(rec)
            if s is None:
                incomplete.append(key)
            else:
                vals.append(s)
            for v in GS1_VIEWS:
                if not ((rec.get("views") or {}).get(v) or {}).get("roundtrip_ok") or not V.roundtrip_ok(v, g):
                    rt_bad.append((key, v))
        ties += sum(1 for i, a in enumerate(vals) if any(abs(a - b) < 1e-9 for k, b in enumerate(vals) if k != i))
    outside = sorted(f"{t}_{j}" for t, outs in scores.items() for j in outs if f"{t}_{j}" not in cands)
    tie_frac = ties / n if n else 0.0
    ok = twin_ok and server_ok and not missing and not incomplete and not rt_bad and tie_frac <= TIE_MAX and n > 0
    return {"ok": ok, "twin_passed": twin_ok, "twin_logs": [str(p) for p in twin_logs], "server_checks": server_ok,
            "low_margin_outputs": len(cands), "candidates": n, "missing": len(missing),
            "missing_outputs": sorted(set(missing))[:10], "incomplete": len(incomplete),
            "incomplete_outputs": sorted(set(incomplete))[:10], "roundtrip_failures": len(rt_bad), "ties": ties,
            "tie_fraction": round(tie_frac, 4), "scored_outside_scope": outside[:10]}


# ----------------------------------------------------------------------------
# gate

def gate(scores_path: Path, no_ledger: bool, n_boot: int) -> dict:
    R = _load("arc2_bank_replay", ROOT / "scripts" / "arc2_bank_replay.py")
    args = argparse.Namespace(rule=list(GS1_RULES), base="b0", pools=None, q27="", tasks="", sealed=False,
                              final_pick=False, n_boot=n_boot, seed=0, rules_file=str(R.RULES_FILE), ledger=str(R.LEDGER),
                              no_ledger=no_ledger, null_calibration=str(R.NULL_CALIBRATION), json="",
                              glimmer=str(scores_path), glimmer_sealed="")
    rep = R.run(args)
    return {r["rule"]: {k: r[k] for k in ("n_right", "n_outputs", "gained", "lost", "net", "gate_v2", "gate_v1",
                                          "sign_test_p", "gained_keys", "lost_keys", "registration")}
            for r in rep["rules"]} | {"_base": rep["base"]}


def _rel(p: Path) -> Path:
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


def sealed_command(scores_path: Path, sealed_path: Path) -> str:
    return (f"envs/kaggle-sim/bin/python scripts/arc2_bank_replay.py --sealed --final-pick --rule {RULE} "
            f"--glimmer {_rel(scores_path)} --glimmer-sealed {_rel(sealed_path)}")


# ----------------------------------------------------------------------------
# time gate (revision 1)

def startup_s(archive: Path) -> float | None:
    """Seconds from the notebook start to the first serving server (weights, install, server start), from _stages.json."""
    p = archive / "gs1" / "_stages.json"
    if not p.is_file():
        return None
    rows = json.loads(p.read_text())
    ok = [r["s"] for r in rows if r.get("stage") == "server" and r.get("status") == "ok"]
    return float(ok[0]) if ok else None


def time_gate(bank, run: dict, archive: Path) -> dict:
    PR = _load("arc2_bank_profile", ROOT / "scripts" / "arc2_bank_profile.py")
    tp = run.get("throughput") or {}
    s_req = tp.get("s_per_request")
    st = startup_s(archive)
    if not s_req or st is None:
        return {"ok": False, "reason": f"not measured: seconds per request {s_req}, start-up {st}"}
    outs = list(bank.outputs())
    low = low_margin_keys(bank)
    b0_low = sum(len({grid_key(c["grid"]) for c in o.candidates}) for o in outs if o.key in low)
    all_low = sum(len(v) for v in scored_candidates(bank).values())
    hidden_outputs = HIDDEN_TASKS * HIDDEN_OUTPUTS_PER_TASK
    per_out = b0_low / len(outs)
    n_hidden = hidden_outputs * per_out
    k = len(GS1_VIEWS)
    scorer_s = st + n_hidden * k * s_req
    th = PR.throughput(bank, hidden_tasks=HIDDEN_TASKS, hidden_outputs_per_task=HIDDEN_OUTPUTS_PER_TASK, ranks=4,
                       overhead_s=600.0, startup_s=PR.STARTUP_S, phase_end_s=PR.PHASE_END_S, b0_end_s=PR.B0_END_S,
                       n_resample=200, seed=0)
    b0_p90 = float(th["projection"]["wall_s"]["p90"])
    end = b0_p90 + scorer_s
    return {"ok": end + SPARE_S <= LIMIT_S, "s_per_request": s_req, "views": list(GS1_VIEWS), "startup_s": st,
            "bank_outputs": len(outs), "bank_low_margin_outputs": len(low), "bank_low_margin_b0_candidates": b0_low,
            "bank_low_margin_candidates_with_27b": all_low, "b0_low_candidates_per_output": round(per_out, 4),
            "hidden_outputs": hidden_outputs, "hidden_candidates": round(n_hidden, 1), "scorer_s": round(scorer_s, 1),
            "b0_p90_end_s": round(b0_p90, 1), "b0_mean_end_s": th["projection"]["wall_s"]["mean"],
            "end_s": round(end, 1), "limit_s": LIMIT_S, "spare_s": SPARE_S,
            "margin_s": round(LIMIT_S - SPARE_S - end, 1),
            "scorer_budget_s": round(LIMIT_S - SPARE_S - b0_p90, 1),
            "max_s_per_request": round((LIMIT_S - SPARE_S - b0_p90 - st) / (n_hidden * k), 3) if n_hidden else None}


# ----------------------------------------------------------------------------
# recorded

def recorded(bank, scores: dict, g: dict, run: dict) -> dict:
    b0 = replay(bank, "b0")
    low = low_margin_keys(bank)
    ranks = []
    for o in bank.outputs():
        if o.key not in low or b0.per_output.get(o.key, {}).get("right"):
            continue
        truth = bank.truth(o)
        per = (scores.get(o.task_id) or {}).get(o.test_index) or {}
        if truth is None or grid_key(truth) not in per:
            continue
        scored = sorted(((glimmer_score(r), k) for k, r in per.items() if glimmer_score(r) is not None), key=lambda x: -x[0])
        rk = next((i + 1 for i, (_, k) in enumerate(scored) if k == grid_key(truth)), None)
        rec = per[grid_key(truth)]
        ranks.append({"output": o.key, "glimmer_rank": rk, "of": len(scored), "b0_rank": rec.get("b0_rank"),
                      "sources": rec.get("sources")})
    by_task = {}
    for rule in GS1_RULES:
        for kind in ("gained_keys", "lost_keys"):
            for k in g[rule][kind]:
                by_task.setdefault(k.rsplit("_", 1)[0], {}).setdefault(rule, []).append(("+" if kind == "gained_keys" else "-") + k)
    tp = run.get("throughput") or {}
    return {"right_candidate_ranks_where_b0_missed": ranks, "by_task": by_task,
            "seconds_per_candidate": tp.get("s_per_candidate"), "tokens_per_candidate": tp.get("tokens_per_request"),
            "scoring_wall_s": tp.get("scoring_wall_s"), "prompt_tok_per_s": tp.get("tok_per_s"),
            "requests_ok": tp.get("requests_ok")}


# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    archive = Path(args.archive)
    sp = archive / "gs1" / "gs1_scores_unsealed.json"
    ss = archive / "gs1" / "sealed" / "gs1_scores_sealed.json"
    if not sp.is_file():
        print(f"GS1: pending (no scores at {sp})")
        return 0
    run = json.loads((archive / "gs1" / "_gs1.json").read_text()) if (archive / "gs1" / "_gs1.json").is_file() else {}
    scores = load_glimmer_scores(sp)
    bank = load_bank(list(BANK_PARTS), Q27_BANK, glimmer=scores)
    val = validity(archive, bank, scores, run)
    print(f"validity: twin passed {val['twin_passed']}, scorer's tokenizer checks {val['server_checks']}, low-margin outputs "
          f"{val['low_margin_outputs']}, candidates {val['candidates']}: missing {val['missing']} {val['missing_outputs']}, "
          f"incomplete {val['incomplete']} {val['incomplete_outputs']}, view round-trip failures {val['roundtrip_failures']}, "
          f"ties {val['ties']} ({100 * val['tie_fraction']:.1f}%, at most {100 * TIE_MAX:.0f}%); scored outside the scope "
          f"{val['scored_outside_scope']} -> {'ok' if val['ok'] else 'NOT MET'}")
    g = gate(sp, args.no_ledger, args.n_boot)
    r = g[RULE]
    print(f"gate v2 {RULE}: gained {r['gained']}, lost {r['lost']}, net {r['net']:+d} ({r['n_right']}/{r['n_outputs']} right) "
          f"-> {'PASS' if r['gate_v2'] else 'no'}")
    cmd = sealed_command(sp, ss)
    sl = archive / "gs1" / "sealed" / "scores_sealed.jsonl"
    want = (((run.get("checks") or {}).get("inputs") or {}).get("scored") or {}).get("sealed", {}).get("candidates", 0)
    got = sum(1 for line in sl.read_text().splitlines() if '"mean"' in line and '"error"' not in line) if sl.is_file() else 0
    print(f"sealed coverage (request counts only; no sealed score is read here): {got} of {want * len(GS1_VIEWS)} requests "
          f"scored, file {'present' if ss.is_file() else 'MISSING'}")
    if r["gate_v2"]:
        print(f"sealed check (once; agree in sign = net > 0 on the 19 sealed tasks, from its 'sealed tasks' line): {cmd}")
    else:
        print(f"sealed check: not applicable (gate v2 not passed); it would be: {cmd}")
    tg = time_gate(bank, run, archive)
    if "scorer_s" in tg:
        print(f"time gate (revision 1): scorer = start-up {tg['startup_s']:.0f} s + {tg['hidden_candidates']:.0f} hidden-run "
              f"candidates ({tg['hidden_outputs']:.0f} outputs x {tg['b0_low_candidates_per_output']} low-margin B0 "
              f"candidates per output; bank {tg['bank_low_margin_b0_candidates']} on {tg['bank_low_margin_outputs']} of "
              f"{tg['bank_outputs']} outputs) x {len(tg['views'])} view x {tg['s_per_request']} s = {tg['scorer_s']:.0f} s; "
              f"B0's E13 p90 end {tg['b0_p90_end_s']:.0f} s (mean {tg['b0_mean_end_s']:.0f} s); end {tg['end_s']:.0f} s + "
              f"{tg['spare_s']} s spare against {tg['limit_s']} s: margin {tg['margin_s']:+.0f} s; the scorer's budget "
              f"{tg['scorer_budget_s']:.0f} s, at most {tg['max_s_per_request']} s per request -> "
              f"{'FITS' if tg['ok'] else 'does not fit'}")
    else:
        print(f"time gate: {tg.get('reason')}")
    rec = recorded(bank, scores, g, run)
    print(f"recorded: gained/lost by task {rec['by_task']}")
    print(f"recorded: Glimmer's rank of the right candidate where B0 missed it (low-margin outputs): "
          f"{[(x['output'], x['glimmer_rank'], x['of'], x['b0_rank']) for x in rec['right_candidate_ranks_where_b0_missed']]}")
    print(f"recorded: {rec['seconds_per_candidate']} s and {rec['tokens_per_candidate']} prompt tokens per candidate (one "
          f"view); {rec['requests_ok']} requests in {rec['scoring_wall_s']} s at {rec['prompt_tok_per_s']} prompt tok/s")
    if not val["ok"]:
        decision = "GS1: INVALID (validity not met): report only"
    elif not r["gate_v2"]:
        decision = f"GS1: gate v2 not passed (gs_low net {r['net']:+d}, lost {r['lost']}): declined, report only"
    else:
        tg_txt = (f"time gate fits (margin {tg['margin_s']:+.0f} s)" if tg.get("ok") else
                  f"time gate does not fit ({tg.get('margin_s', tg.get('reason'))}): report only")
        decision = (f"GS1: gate v2 PASS for gs_low (net {r['net']:+d}, lost {r['lost']}); {tg_txt}; next the sealed check "
                    f"(command above), once")
    print(decision)
    if args.json:
        Path(args.json).write_text(json.dumps({"validity": val, "gate": g, "sealed_command": cmd, "time_gate": tg,
                                               "recorded": rec, "decision": decision}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
