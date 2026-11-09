#!/usr/bin/env python3
"""TCI60 analysis (run a3-tci60): the frozen rule reports/research/ARC3_TCI60_RULE_2026-10-30.md (7475f7d;
Oct 30) on the run's r1_results.jsonl: arm A16 (R1g's A7L, the told insight) on the
ThinkingCap checkpoint at 320 actions / 3,600 s per case-arm, on the 35 replayed stuck states, paired with R1v (the same
arm at 320 / 1,800 s, 24 of 35).

R1v's analyzer (scripts/arc3_r1v_analyze.py at e720df2, sha256-checked; ops/arc3_r1v_analyze.py is the same bytes)
supplies the loaders, the usage figures and the paired tables against its runs; R1v itself is added as a comparison.

Validity (any failing: no decision, report only):
  - the CPU twin passed ("preflight: PASS" in its log: <archive>/preflight/, or the latest <archive less -vN>-twin-vM
    when the twin ran as its own step before a --no-twin push);
  - 35 of 35 case-arms ended, none missing, skipped or cut by the hard stop, with no oracle errors;
  - the insight shown in 35 of 35 (oracle_kind insight).
Case-arms missing or cut by the hard stop are reported as incomplete, as R1t's analyzer does: a count they could move
across a threshold reads "incomplete".
Reading: TCI60's clears of 35, paired with R1v's on the same states; exact McNemar p (two-sided) beside.
  time adds on ThinkingCap   >= 28
  no added effect            <= 25
  report                     26 or 27
Recorded beside, no decision: pairs with R1t (Intel, the insight at 320 / 60 min) and R1u (ThinkingCap without it), and
R1v's other comparisons; clears within 160 actions / 1,800 s against R1v's 21, and within R1v's 320 / 1,800 s;
mechanics-only clears of 22; tokens per request and per action, actions per case-minute; ends; the case-clustered test
of reports/research/insight_net_2026-10-24.py (sign flips over cases, case bootstrap) against R1v and R1u.

  envs/arc3/bin/python -I scripts/arc3_tci60_analyze.py [--archive experiments/runs/kaggle-A3-TCI60-v1] [--json FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "experiments" / "runs"
ARCHIVE = RUNS / "kaggle-A3-TCI60-v1"
R1V_COMMIT, R1V_ANALYZER = "e720df2", "scripts/arc3_r1v_analyze.py"
R1V_ANALYZER_SHA = "38d445ff31b291b15ad536ca1c1771383f27743720436e4aebf615a7dbb42753"
INSIGHT_NET = ROOT / "reports" / "research" / "insight_net_2026-10-24.py"
ARM = "A16"
N_CASES = 35
BUDGET = (320, 3600.0)          # TCI60's per case-arm
R1V_BUDGET = (320, 1800.0)
TIME_ADDS, NO_EFFECT = 28, 25
R1V_LABEL = "A16 (R1v)"
R1V_GROUP = ("told insight (ThinkingCap), 320 actions / 30 min", ((R1V_LABEL, "kaggle-R1v-tcinsight-v1", "A16"),))
RULE = ("TCI60, frozen (reports/research/ARC3_TCI60_RULE_2026-10-30.md at 7475f7d; 09:3x "
        "IST Oct 30): validity (twin passed; 35 of 35 case-arms ended, no oracle errors; insight shown 35 of 35); TCI60's "
        f"clears of 35 paired with R1v's 24: >= {TIME_ADDS} time adds on ThinkingCap; <= {NO_EFFECT} no added effect; "
        "otherwise report; exact McNemar p (two-sided) against R1v beside")


def r1v_analyzer() -> types.ModuleType:
    """R1v's analyzer from e720df2, run as if it sat in scripts/ (it finds the repository from its own path)."""
    src = subprocess.run(["git", "show", f"{R1V_COMMIT}:{R1V_ANALYZER}"], cwd=ROOT, capture_output=True,
                         check=True).stdout
    assert hashlib.sha256(src).hexdigest() == R1V_ANALYZER_SHA, f"{R1V_ANALYZER} at {R1V_COMMIT} is not R1v's"
    mod = types.ModuleType("arc3_r1v_analyze")
    mod.__file__ = str(ROOT / R1V_ANALYZER)
    sys.modules["arc3_r1v_analyze"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    mod.BUDGET = BUDGET         # cleared_over_budget: TCI60's budget
    return mod


V = r1v_analyzer()
RA = V.RA


def _insight_net() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("insight_net", INSIGHT_NET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def twin_logs(run: Path) -> list[Path]:
    logs = sorted((run / "preflight").glob("*.log")) if (run / "preflight").is_dir() else []
    m = re.fullmatch(r"(.+)-v\d+", run.name)
    if logs or not m:
        return logs
    twins = [p for p in run.parent.glob(f"{m.group(1)}-twin-v*") if re.fullmatch(r".+-twin-v\d+", p.name)]
    twins.sort(key=lambda p: int(p.name.rsplit("-v", 1)[1]))
    return sorted(twins[-1].glob("*.log")) if twins else []


def twin_passed(run: Path) -> bool | None:
    logs = twin_logs(run)
    return any("preflight: PASS" in p.read_text(errors="replace") for p in logs) if logs else None


def band(n: int) -> str:
    if n >= TIME_ADDS:
        return "time adds on ThinkingCap"
    if n <= NO_EFFECT:
        return "no added effect"
    return "report"


def reading(n: int, open_n: int) -> tuple[str, str]:
    """The band on clears of 35; `open_n` case-arms missing, skipped or cut by the hard stop could each still have
    cleared, so a count they could move across a threshold is incomplete (R1t's analyzer)."""
    hi = n + open_n
    if band(n) == band(hi):
        return band(n), f"{ARM} {n}/35" + (f" ({open_n} open case-arm(s) cannot change the band)" if open_n else "")
    return "incomplete", (f"{ARM} {n}/35 with {open_n} case-arm(s) missing or cut by the hard stop: {n}-{hi} spans a "
                          f"threshold ({NO_EFFECT}, {TIME_ADDS})")


def clustered(rows: dict, other: dict, cases: list[str]) -> dict | None:
    """insight_net's case-clustered test on one run each: per case cleared (TCI60) - cleared (other)."""
    if not other:
        return None
    net = _insight_net()
    diff = {c: int(V.cleared(rows.get(c))) - int(V.cleared(other.get(c))) for c in cases}
    obs, p, lo, hi = net.test(diff)
    return {"net_clears": obs, "p": round(p, 4), "ci95": [lo, hi], "cases": len(cases)}


def analyze(archive: Path, runs_root: Path = RUNS, labels: dict | None = None) -> dict:
    labels = RA.load_labels(RA.LABELS_CSV) if labels is None else labels
    groups = [(g, [(lab, runs_root / run / "r1_results.jsonl", arm) for lab, run, arm in specs])
              for g, specs in tuple(V.GROUPS) + (R1V_GROUP,)]
    results = archive / "r1_results.jsonl"
    rep = V.analyze(results, groups, labels)
    rep["rule"], rep["archive"], rep["twin_passed"] = RULE, str(archive), twin_passed(archive)
    if rep["arm"] is None:
        rep["decision"], rep["decision_reason"] = "pending", f"no TCI60 results ({results})"
        return rep
    rec = rep["arm"]
    rows = V.load_arm(results, ARM)
    r1v = V.load_arm(runs_root / "kaggle-R1v-tcinsight-v1" / "r1_results.jsonl", "A16")
    r1u = V.load_arm(runs_root / "kaggle-R1u-thinkingcap-v2" / "r1_results.jsonl", "A15")
    acases = sorted(set(rows) | set(r1v))
    skipped = [c for c in acases if c in rows and not RA.ran(rows[c])]
    open_cases = set(rec["missing"]) | set(skipped) | set(rec["hard_stop_cut"])
    n_open = len(open_cases) + max(0, N_CASES - len(acases))
    w1800 = [c for c in acases if V.cleared_within(rows.get(c), R1V_BUDGET)]
    rec.update({
        "skipped": skipped, "open_case_arms": n_open,
        "within_320_1800": len(w1800), "after_1800_cases": [c for c in acases if V.cleared(rows.get(c)) and c not in w1800],
        "vs_r1v": rec["pairs"].get(R1V_GROUP[0], {}).get(R1V_LABEL, {}).get("all"),
        "r1v_all": sum(V.cleared(r1v.get(c)) for c in acases) if r1v else None,
        "r1v_within_160_1800": sum(1 for c in acases if V.cleared_within(r1v.get(c), V.R30)) if r1v else None,
        "clustered": {"vs A16 (R1v)": clustered(rows, r1v, acases), "vs A15 (R1u)": clustered(rows, r1u, acases)},
        "r1v_missed_ends": {c: (r1v.get(c) or {}).get("end_reason") for c in acases if r1v and not V.cleared(r1v.get(c))},
    })
    i = rec["insight"]
    checks = {"twin_passed": rep["twin_passed"] is True,
              "35_case_arms_ended": rec["n"] == N_CASES and rec["ran"] == N_CASES and n_open == 0,
              "no_oracle_errors": i["errors"] == 0,
              "insight_shown_35": i["shown"] == N_CASES and i["kinds"] == ["insight"]}
    rep["validity"] = {"valid": all(checks.values()), "checks": checks}
    b, why = reading(rec["all"], n_open)
    rep["reading"] = {"band": b, "reason": why, "clears": rec["all"], "r1v": rec["r1v_all"],
                      "p_vs_r1v": (rec["vs_r1v"] or {}).get("p")}
    rep["decision"] = b if rep["validity"]["valid"] else "no decision (validity failed): report only"
    rep["decision_reason"] = why
    return rep


def _p(p) -> str:
    return "n/a" if p is None else f"{p:.3f}"


def _na(x) -> str:
    return "n/a" if x is None else str(x)


def format_report(rep: dict) -> str:
    L = [f"TCI60 (a3-tci60): arm {ARM} (R1g's A7L, told insight) on ThinkingCap at 320 actions / 3600 s, paired with "
         "R1v (the same at 320 / 1800 s); R1v's comparisons beside", f"labels on the cases: {rep['labels']}", "",
         "| Track | run | clears of 35 | mechanics-only of 22 | within 160 / 1800 | tokens per request | tokens per "
         "action | actions per case-minute (median) | actions per case-arm (median) |",
         "|---|---|---|---|---|---|---|---|---|"]

    def row(name, r):
        u = r["usage"]
        return (f"| ARC-3 | {name} | {r['all']} | {r['partial']} | {r['within_160_1800']} | "
                f"{_na(u['tokens_per_request'])} | {_na(u['tokens_per_action'])} | {_na(u['apm_median'])} | "
                f"{_na(u['actions_median'])} |")

    for g, runs in rep["groups"].items():
        for lab, r in runs.items():
            L.append(row(f"{lab}: {g}", r))
    rec = rep["arm"]
    tail = ["", f"Rule: {rep['rule']}", f"Decision: {rep['decision']} ({rep['decision_reason']})"]
    if rec is None:
        return "\n".join(L + [f"| ARC-3 | {ARM} (TCI60) | pending | | | | | | |"] + tail) + "\n"
    L.append(row(f"{ARM} (TCI60): told insight (ThinkingCap), 320 actions / 60 min", rec))
    v = rep["validity"]
    L += ["", f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']} (twin log: {rep['twin_passed']})",
          f"case-arms in the archive {rec['n']}, ran {rec['ran']}; missing {rec['missing']}; skipped {rec['skipped']}; "
          f"cut by the hard stop {rec['hard_stop_cut']}; open {rec['open_case_arms']}; ends {rec['ends']}",
          f"insight shown in {rec['insight']['shown']} case-arms ({rec['insight']['shown_partial']} mechanics-only); "
          f"kinds {rec['insight']['kinds']}; errors {rec['insight']['errors']}; ran without it {rec['insight']['not_shown']}",
          f"clears within 160 actions / 1800 s {rec['within_160_1800']} (R1v {_na(rec['r1v_within_160_1800'])}); within "
          f"R1v's 320 / 1800 s {rec['within_320_1800']}; after 1800 s {len(rec['after_1800_cases'])} {rec['after_1800_cases']}",
          f"R1v's misses and how they ended there: {rec['r1v_missed_ends']}"]
    if rec["cleared_over_budget"]:
        L.append(f"  cleared past 320 actions / 3600 s (budget check): {rec['cleared_over_budget']}")
    u = rec["usage"]
    L += [f"tokens per request {_na(u['tokens_per_request'])} ({_na(u['requests'])} requests over {u['request_counted']} "
          f"case-arms); tokens per action {_na(u['tokens_per_action'])} ({u['generated_tokens']:,} / {u['actions']:,}); "
          f"actions per case-minute median {_na(u['apm_median'])} over {u['apm_n']}", "",
          "| Track | scope | TCI60 vs | TCI60 only | other only | both | neither | exact McNemar p (two-sided) |",
          "|---|---|---|---|---|---|---|---|"]
    for scope in ("all", "partial"):
        for g, runs in rec["pairs"].items():
            for lab, pr in runs.items():
                q = pr[scope]
                L.append(f"| ARC-3 | {scope} | {lab} | {q['arm_only']} | {q['other_only']} | {q['both']} | "
                         f"{q['neither']} | {_p(q['p'])} |")
    L += ["", "case-clustered test (insight_net_2026-10-24.py: sign flips over cases, 100,000; case bootstrap 95%):"]
    for lab, c in rec["clustered"].items():
        L.append(f"  {lab}: " + ("n/a" if c is None else f"net clears {c['net_clears']:+d} over {c['cases']} cases, "
                                                         f"p {c['p']}, 95% interval {c['ci95']}"))
    q = rec["vs_r1v"] or {}
    L += ["", f"Reading: TCI60 {rec['all']}/35 against R1v {_na(rec['r1v_all'])}/35 (TCI60 only {q.get('arm_only')}, R1v "
          f"only {q.get('other_only')}; exact McNemar p {_p(q.get('p'))}); band: {rep['reading']['band']}"]
    return "\n".join(L + tail) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    ap.add_argument("--json", type=Path, default=None, help="default: <archive>/tci60_decision.json")
    a = ap.parse_args(argv)
    rep = analyze(a.archive, a.runs_root)
    print(format_report(rep), end="")
    if a.archive.is_dir() and rep.get("arm") is not None:
        (a.json or a.archive / "tci60_decision.json").write_text(json.dumps(rep, indent=1, default=str) + "\n",
                                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
