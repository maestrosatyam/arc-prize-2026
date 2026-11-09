#!/usr/bin/env python3
"""OIF analysis (run a3-oif-public): the frozen rule reports/research/ARC3_OIF_ORACLE_FULLGAME_RULE_2026-10-29.md
(acba2fb; 22:22 IST Oct 29) on the run's benchmark.json, oif_levels.jsonl and
oif_state_*.json, against P0's public commit run (experiments/runs/kaggle-P0-commit-v1: 100 game-passes, mean 48.96).

Validity (any failing: no decision, report only):
  - the CPU twin passed ("preflight: PASS" in the archived twin log: <archive>/preflight/, or the latest
    <archive less -vN>-twin-vM when the twin ran as its own step before a --no-twin push);
  - the server loaded with no error (serve.log reaches the ready line with no error before it; TCF's check);
  - at least 96 of 100 game-passes end with a scorecard;
  - every level entered logs whether a message was shown and its text hash: each level a scored game-pass cleared, and
    the level it ended on when it took an action there, has an entry record and a state record;
  - runtime (change 3): at least 95% of the listed levels entered show their message exactly once (state
    record shown == 1, entry record armed with C1's sha256, one entry per game-pass and level); the levels shown twice
    or not at all are listed.
Reading: D = OIF's mean game score - P0's 48.96, game-paired permutation on per-game means (pass labels shuffled within
each game), two-sided, 20,000 draws, seed 0 (WM1's test, experiments/arc3-wm/wm_report.py):
  insight raises full-game score   D >= +5 and p <= 0.05
  no measurable gain               D <= +1
  report                           otherwise
Fallback, only with 2 passes (change c): one-sided against P0's full 48.96; OIF above 48.96 reads "a perfect
insight is worth more than doubling the time", otherwise report.
Recorded beside, no decision: levels per pass and wins; per-game deltas; the share of levels entered that had a text;
the score on levels with a text against levels without (P0 on the same levels beside); moves per clear; tokens per
action and per game-pass; wall-budget endings; the first model action time; whether the message is in the request at
the level's last model call (change 4); any unlisted level entered whose per-level log shows a message
(22:2x IST Oct 29).

  envs/arc3/bin/python -I scripts/arc3_oif_analyze.py [--archive experiments/runs/kaggle-A3-OIF-public-v1]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "experiments" / "runs"
ARCHIVE = RUNS / "kaggle-A3-OIF-public-v1"
P0_RUN = RUNS / "kaggle-P0-commit-v1"
LEVELS = ROOT / "experiments" / "arc3-c1" / "levels.jsonl"
LEVELS_SHA = "1c8a0dff01f4a5ec07068f2ca5640ee60e34fa672734095911b498c51a21d6ab"   # 7d7abd6
P0_MEAN = 48.96
GAMES = 25
MIN_SCORECARDS_SHARE = 0.96
SHOWN_ONCE_SHARE = 0.95
RAISE_D, RAISE_P, NO_GAIN_D = 5.0, 0.05, 1.0
GAME_RUNTIME_S = 532 * 60 * 100 // 110      # P0's per-game runtime at 4 passes: 29,018 s
SCORE_CAP = 1.15
READY = "The server is fired up and ready to roll!"
SERVE_ERROR = re.compile(r"Traceback|Error|error|CUDA out of memory|Killed|exited")
SERVE_NOISE = ("sitecustomize", "wrapt")
SCORECARD_STATES = {"won", "gave_up", "cancelled"}
RULE = ("OIF, frozen (reports/research/ARC3_OIF_ORACLE_FULLGAME_RULE_2026-10-29.md at acba2fb; "
        "22:22 IST Oct 29): validity (twin passed; server loads with no "
        "error; >= 96 of 100 game-passes scored; every level entered logs shown and text hash; >= 95% of listed levels "
        "entered shown exactly once); D = OIF mean game score - P0's 48.96, game-paired permutation, two-sided, 20,000 "
        "draws: D >= +5 and p <= 0.05 insight raises full-game score; D <= +1 no measurable gain; otherwise report")
_PASS = re.compile(r"_p(\d+)\.html$")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


WM = _load("wm_report", ROOT / "experiments" / "arc3-wm" / "wm_report.py")


def c1_sha() -> dict:
    """(game_id, level) -> sha256 of C1's insight message (the list at 7d7abd6)."""
    assert hashlib.sha256(LEVELS.read_bytes()).hexdigest() == LEVELS_SHA, "C1's level list changed"
    out = {}
    for line in LEVELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["game_id"], int(r["level"]))] = hashlib.sha256(r["insight"]["message"].encode("utf-8")).hexdigest()
    return out


def load_runs(path: Path) -> list[dict]:
    """One row per game-pass: game, pass (from solver_analysis_html), score, levels, actions, tokens, wall."""
    b = json.loads(path.read_text(encoding="utf-8"))
    rows, seen = [], collections.Counter()
    for r in b["game_runs"]:
        g = r["game_id"].split("-")[0]
        m = _PASS.search(str(r.get("solver_analysis_html") or ""))
        p = int(m.group(1)) if m else seen[g]
        seen[g] += 1
        h = r.get("history") or []
        rows.append({"game": g, "game_id": r["game_id"], "pass": p, "score": r.get("final_score"),
                     "state": r.get("state"), "cleared": int(r.get("levels_completed") or 0),
                     "levels": int(r["number_of_levels"]), "actions": len(h),
                     "tokens": sum(int(x.get("generated_tokens") or 0) for x in h),
                     "first_action_s": float(h[0].get("wallclock_seconds") or 0) if h else None,
                     "wall_s": r.get("final_wallclock_seconds"), "apl": list(r.get("actions_per_level") or []),
                     "base": list(r.get("base_actions_per_level") or []), "won": r.get("state") == "won"})
    return rows


def levels_reached(r: dict) -> list[int]:
    """The levels a game-pass had an opener on: every cleared level, and the level it ended on when it took an action
    there."""
    out = list(range(1, r["cleared"] + 1))
    played = r["actions"] - sum(r["apl"][:r["cleared"]])
    if not r["won"] and r["cleared"] < r["levels"] and played > 0:
        out.append(r["cleared"] + 1)
    return out


def level_score(r: dict, lv: int) -> float:
    """The competition's level score: min(1.15, (baseline / actions)^2) for a cleared level, else 0."""
    if lv > r["cleared"] or lv > len(r["apl"]) or lv > len(r["base"]):
        return 0.0
    a = int(r["apl"][lv - 1])
    return min(SCORE_CAP, (int(r["base"][lv - 1]) / a) ** 2) if a > 0 else SCORE_CAP


def one_sided(a: list[dict], b: list[dict], draws: int = 20_000, seed: int = 0) -> tuple[float, float]:
    """WM1's game-paired permutation, one-sided (b above a): the fallback's reading."""
    by = collections.defaultdict(lambda: ([], []))
    for r in a:
        by[r["game"]][0].append(r["score"])
    for r in b:
        by[r["game"]][1].append(r["score"])
    games = sorted(g for g in by if by[g][0] and by[g][1])

    def stat(groups):
        return st.mean([x for _, bb in groups for x in bb]) - st.mean([x for aa, _ in groups for x in aa])

    obs = stat([by[g] for g in games])
    rng, hits = random.Random(seed), 0
    for _ in range(draws):
        groups = []
        for g in games:
            pool = by[g][0] + by[g][1]
            rng.shuffle(pool)
            groups.append((pool[:len(by[g][0])], pool[len(by[g][0]):]))
        hits += stat(groups) >= obs - 1e-12
    return obs, (hits + 1) / (draws + 1)


def server_load(run: Path) -> dict:
    p = run / "serve.log"
    if not p.is_file():
        return {"ok": False, "detail": "serve.log missing"}
    lines = p.read_text(errors="replace").splitlines()
    ready = next((i for i, x in enumerate(lines) if READY in x), None)
    upto = lines if ready is None else lines[:ready]
    bad = [x[:200] for x in upto if SERVE_ERROR.search(x) and not any(n in x for n in SERVE_NOISE)]
    return {"ok": ready is not None and not bad, "ready_line": ready, "errors_before_ready": bad[:5]}


def twin_logs(run: Path) -> list[Path]:
    """The twin's logs: <run>/preflight/ when the queue ran the twin with the GPU job; otherwise the latest sibling twin
    archive (<run less -vN>-twin-vM), the twin run as its own step before a --no-twin GPU push."""
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


def load_logs(run: Path) -> tuple[dict, dict, list]:
    """Entry records by (game, pass, level) (lists), state records by (game, pass, level), and log errors."""
    entries, errors = collections.defaultdict(list), []
    p = run / "oif_levels.jsonl"
    for line in (p.read_text(encoding="utf-8").splitlines() if p.is_file() else []):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") and "level" not in r:
            errors.append(r["error"])
            continue
        entries[(str(r["game_id"]).split("-")[0], r.get("pass_index"), int(r["level"]))].append(r)
    states = {}
    for sp in sorted(run.glob("oif_state_*.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        for lv, rec in (d.get("levels") or {}).items():
            states[(str(d["game_id"]).split("-")[0], d.get("pass_index"), int(lv))] = {**rec, "game_id": d["game_id"]}
    return entries, states, errors


def side(rows: list[dict]) -> dict:
    passes = sorted({r["pass"] for r in rows})
    actions, tokens = sum(r["actions"] for r in rows), sum(r["tokens"] for r in rows)
    per = [a for r in rows for a in r["apl"][:r["cleared"]]]
    first = [r["first_action_s"] for r in rows if r["first_action_s"] is not None]
    return {"game_passes": len(rows), "mean_score": round(st.mean(r["score"] for r in rows), 2) if rows else None,
            "levels_per_pass": {p: round(st.mean(r["cleared"] for r in rows if r["pass"] == p), 2) for p in passes},
            "levels_mean": round(st.mean(r["cleared"] for r in rows), 2) if rows else None,
            "won": sum(r["won"] for r in rows),
            "moves_per_clear": {"levels": len(per), "median": st.median(per) if per else None,
                                "mean": round(st.mean(per), 1) if per else None},
            "tokens_per_action": round(tokens / actions, 1) if actions else None,
            "tokens_per_game_pass": {"mean": round(tokens / len(rows)) if rows else None,
                                     "median": round(st.median(r["tokens"] for r in rows)) if rows else None},
            "wall_budget_endings": sum(1 for r in rows if not r["won"] and float(r["wall_s"] or 0) >= GAME_RUNTIME_S - 60),
            "first_model_action_s": {"min": round(min(first)) if first else None,
                                     "median": round(st.median(first)) if first else None}}


def analyze(archive: Path, p0_path: Path = P0_RUN / "benchmark.json") -> dict:
    out: dict = {"rule": RULE, "archive": str(archive)}
    twin = twin_passed(archive)
    bench = archive / "benchmark.json"
    if not bench.is_file():
        out.update({"decision": "pending: no benchmark.json yet", "twin": twin})
        return out
    sha = c1_sha()
    oif, p0 = load_runs(bench), load_runs(p0_path)
    passes = int(json.loads(bench.read_text(encoding="utf-8")).get("n_passes") or 0)
    entries, states, log_errors = load_logs(archive)
    scored = [r for r in oif if r["state"] in SCORECARD_STATES and isinstance(r["score"], (int, float))]
    expected = GAMES * passes
    need = math.ceil(MIN_SCORECARDS_SHARE * expected)

    missing = []
    for r in scored:
        for lv in levels_reached(r):
            k = (r["game"], r["pass"], lv)
            if not entries.get(k) or k not in states:
                missing.append(f"{r['game']}_p{r['pass']}_L{lv}")
    listed_entered, once, twice, never, wrong_hash = [], [], [], [], []
    for k, recs in sorted(entries.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else -1, kv[0][2])):
        gid = recs[0].get("game_id")
        want = sha.get((gid, k[2]))
        if want is None:
            continue
        tag = f"{k[0]}_p{k[1]}_L{k[2]}"
        listed_entered.append(tag)
        shown = int((states.get(k) or {}).get("shown") or 0)
        if any(x.get("message_sha256") != want for x in recs):
            wrong_hash.append(tag)
        if len(recs) > 1 or shown >= 2:
            twice.append(tag)
        elif shown == 0:
            never.append(tag)
        elif shown == 1 and recs[0].get("armed") and recs[0].get("message_sha256") == want:
            once.append(tag)
    share_once = len(once) / len(listed_entered) if listed_entered else 0.0
    serve = server_load(archive)
    checks = {"twin_passed": twin is True, "server_loaded_no_error": serve["ok"],
              "scorecards_96": expected > 0 and len(scored) >= need,
              "levels_logged": not missing, "shown_once_95": bool(listed_entered) and share_once >= SHOWN_ONCE_SHARE}
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "passes": passes, "game_passes": len(oif),
                       "scored": len(scored), "scored_needed": need, "server": serve,
                       "levels_without_record": missing[:50], "levels_without_record_count": len(missing),
                       "listed_entered": len(listed_entered), "shown_once": len(once),
                       "shown_once_share": round(share_once, 4), "shown_twice": twice, "shown_never": never,
                       "wrong_hash": wrong_hash, "log_errors": log_errors[:10]}

    a = [{"game": r["game"], "score": r["score"]} for r in p0]
    b = [{"game": r["game"], "score": r["score"]} for r in scored]
    d, p = WM.permutation(a, b)
    oif_mean = st.mean(r["score"] for r in scored) if scored else None
    if d >= RAISE_D and p <= RAISE_P:
        band = "insight raises full-game score"
    elif d <= NO_GAIN_D:
        band = "no measurable gain"
    else:
        band = "report"
    out["reading"] = {"OIF_mean": round(oif_mean, 2) if oif_mean is not None else None,
                      "P0_mean": round(st.mean(r["score"] for r in p0), 2), "P0_rule_mean": P0_MEAN,
                      "D": round(d, 2), "p": round(p, 4), "band": band, "game_passes": len(scored),
                      "P0_game_passes": len(p0)}
    if passes == 2:
        d1, p1 = one_sided(a, b)
        out["fallback"] = {"passes": 2, "D": round(d1, 2), "p_one_sided": round(p1, 4),
                           "reading": "a perfect insight is worth more than doubling the time"
                           if oif_mean is not None and oif_mean > P0_MEAN else "report"}
        band = out["fallback"]["reading"]
    out["decision"] = band if out["validity"]["valid"] else "no decision (validity failed): report only"

    # recorded beside
    out["oif"], out["p0"] = side(scored), side(p0)
    games = sorted({r["game"] for r in p0} | {r["game"] for r in scored})
    out["per_game"] = {}
    for g in games:
        o = [r["score"] for r in scored if r["game"] == g]
        q = [r["score"] for r in p0 if r["game"] == g]
        out["per_game"][g] = {"OIF": round(st.mean(o), 2) if o else None, "P0": round(st.mean(q), 2) if q else None,
                              "delta": round(st.mean(o) - st.mean(q), 2) if o and q else None,
                              "listed_levels": sum(1 for (gid, _lv) in sha if gid.split("-")[0] == g)}
    entered = [(k, recs) for k, recs in entries.items()]
    with_text = sum(1 for _k, recs in entered if recs[0].get("listed"))
    out["text_share"] = {"levels_entered": len(entered), "with_text": with_text,
                         "share": round(with_text / len(entered), 3) if entered else None}
    gid_of = {r["game"]: r["game_id"] for r in p0 + scored}
    lv_scores = {"OIF": {"with_text": [], "without": []}, "P0": {"with_text": [], "without": []}}
    for name, rows in (("OIF", scored), ("P0", p0)):
        for r in rows:
            for lv in levels_reached(r):
                key = "with_text" if (gid_of[r["game"]], lv) in sha else "without"
                lv_scores[name][key].append(level_score(r, lv))
    out["level_scores"] = {n: {k: {"levels": len(v), "mean": round(st.mean(v), 3) if v else None}
                               for k, v in x.items()} for n, x in lv_scores.items()}
    # recorded only (22:2x IST Oct 29): an unlisted level entered whose per-level log shows a message
    unl = []
    for k, recs in sorted(entries.items(), key=lambda kv: str(kv[0])):
        if sha.get((recs[0].get("game_id"), k[2])) is not None:
            continue
        s = states.get(k) or {}
        if any(x.get("listed") or x.get("armed") or x.get("message_sha256") for x in recs) \
                or s.get("listed") or s.get("armed") or int(s.get("shown") or 0) > 0:
            unl.append(f"{k[0]}_p{k[1]}_L{k[2]}")
    out["unlisted_with_message"] = unl
    shown_states = [(k, s) for k, s in states.items() if s.get("listed")]
    out["message_in_prompt_last_call"] = {
        "listed_levels": len(shown_states),
        "in_prompt": sum(1 for _k, s in shown_states if s.get("in_prompt_last_call") is True),
        "dropped": [f"{k[0]}_p{k[1]}_L{k[2]}" for k, s in shown_states if s.get("in_prompt_last_call") is False],
        "first_call_with_message": dict(collections.Counter(s.get("first_call_with_message") for _k, s in shown_states))}
    return out


def _p(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def format_report(rep: dict) -> str:
    if "validity" not in rep:
        return f"OIF (a3-oif-public): {rep['decision']}; twin {rep.get('twin')}\nDecision: {rep['decision']}\n"
    v, rd, o, b = rep["validity"], rep["reading"], rep["oif"], rep["p0"]
    L = ["OIF (a3-oif-public): C1's oracle insight once at each listed level of the 25 full public games, on P0's Intel "
         f"commit run; {v['passes']} passes, {v['game_passes']} game-passes",
         f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
         f"  scored {v['scored']} (need {v['scored_needed']}); server ready line {v['server'].get('ready_line')}, errors "
         f"before it {v['server'].get('errors_before_ready')}; levels without a record {v['levels_without_record_count']} "
         f"{v['levels_without_record'][:10]}",
         f"  runtime: listed levels entered {v['listed_entered']}, shown exactly once {v['shown_once']} "
         f"({v['shown_once_share']:.1%}; need 95%); shown twice {v['shown_twice'][:20]}; not shown {v['shown_never'][:20]}; "
         f"hash not C1's {v['wrong_hash'][:10]}; log errors {v['log_errors'][:3]}",
         "",
         "| Track | Run | game-passes | mean score | levels per pass | won | moves per clear (median; mean) | tokens per "
         "action | tokens per game-pass (mean; median) | wall-budget endings | first model action s (min; median) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("P0 (Intel, public commit)", b), ("OIF (P0 + C1's insight per level)", o)):
        m, f = s["moves_per_clear"], s["first_model_action_s"]
        L.append(f"| ARC-3 | {name} | {s['game_passes']} | {s['mean_score']} | {s['levels_per_pass']} | {s['won']} | "
                 f"{m['median']}; {m['mean']} | {s['tokens_per_action']} | {s['tokens_per_game_pass']['mean']}; "
                 f"{s['tokens_per_game_pass']['median']} | {s['wall_budget_endings']} | {f['min']}; {f['median']} |")
    ls = rep["level_scores"]
    mp = rep["message_in_prompt_last_call"]
    L += ["",
          f"D = {rd['D']:+.2f} (OIF {rd['OIF_mean']} - P0 {rd['P0_mean']}; rule's P0 {rd['P0_rule_mean']}), game-paired "
          f"permutation, two-sided, 20,000 draws: p {_p(rd['p'])}; band: {rd['band']}"]
    if "fallback" in rep:
        fb = rep["fallback"]
        L.append(f"fallback (2 passes): one-sided p {_p(fb['p_one_sided'])} against P0's full 48.96: {fb['reading']}")
    L += ["",
          "recorded beside, no decision:",
          f"  levels entered with a text: {rep['text_share']['with_text']} of {rep['text_share']['levels_entered']} "
          f"({rep['text_share']['share']})",
          f"  level score (cleared: min(1.15, (baseline/actions)^2); else 0), levels reached: OIF with a text "
          f"{ls['OIF']['with_text']['mean']} over {ls['OIF']['with_text']['levels']}, without "
          f"{ls['OIF']['without']['mean']} over {ls['OIF']['without']['levels']}; P0 on the same levels: with a text "
          f"{ls['P0']['with_text']['mean']} over {ls['P0']['with_text']['levels']}, without {ls['P0']['without']['mean']} "
          f"over {ls['P0']['without']['levels']}",
          f"  message in the request at the level's last model call: {mp['in_prompt']} of {mp['listed_levels']} listed "
          f"levels entered; dropped on {len(mp['dropped'])} {mp['dropped'][:10]}"
          + ("  -> Limits: the harness dropped the message on some levels, so the result is below the perfect-insight "
             "ceiling" if mp["dropped"] else ""),
          f"  first model call carrying the message (count of levels): {mp['first_call_with_message']}",
          f"  unlisted levels entered whose log shows a message: {len(rep['unlisted_with_message'])} "
          f"{rep['unlisted_with_message'][:10]}",
          "",
          "| Track | game | OIF | P0 | delta | listed levels |", "|---|---|---|---|---|---|"]
    L += [f"| ARC-3 | {g} | {x['OIF']} | {x['P0']} | {x['delta']} | {x['listed_levels']} |"
          for g, x in rep["per_game"].items()]
    L += ["", f"Rule: {rep['rule']}", f"Decision: {rep['decision']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--json", type=Path, default=None, help="default: <archive>/oif_decision.json")
    a = ap.parse_args(argv)
    rep = analyze(a.archive)
    print(format_report(rep), end="")
    if a.archive.is_dir() and "validity" in rep:
        (a.json or a.archive / "oif_decision.json").write_text(json.dumps(rep, indent=1, default=str) + "\n",
                                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
