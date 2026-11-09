#!/usr/bin/env python3
"""HL analysis (runs a3-hlb-public, a3-hla-public): the frozen rule reports/research/ARC3_HINT_LIBRARY_RULE_2026-10-30.md
(a561b9f; 19:27 IST Oct 30) on one arm's
benchmark.json and HL logs, against FPG's public run (experiments/runs/kaggle-A3-FPG-public-v1: 100 game-passes, mean
44.21). OIF's analyzer (scripts/arc3_oif_analyze.py) supplies the loaders and the side lines.

Validity (any failing: no decision, report only):
  - the CPU twin passed (its own archive <archive less -vN>-twin-vM, or <archive>/preflight/);
  - the server loaded with no error;
  - at least 96 of 100 game-passes end with a scorecard;
  - every level entered logs its lookups (a level record and at least one lookup with trigger, picked ids and picker);
  - on at least 95% of levels entered, the picked hints are in the system prompt at the level's last model call;
  - HLA: at least 90% of its calls return a valid id list.
Reading: D = the arm's mean game score - FPG's 44.21, game-paired permutation on per-game means (pass labels shuffled
within each game), two-sided, 20,000 draws, seed 0 (WM1's test); each arm read at p <= 0.025:
  entry candidate       D >= +5 and p <= 0.025
  no measurable gain    D <= +1
  report                otherwise
Recorded beside, no decision: the other arm, paired by game, when its archive is there; wins and levels per pass;
per-game deltas; lookups per level and the stuck-trigger share; the most-picked hints and their source; HLA's fallback
count and its call time share; tokens per action; wall-budget endings.

  envs/arc3/bin/python -I scripts/arc3_hl_analyze.py --archive experiments/runs/kaggle-A3-HLB-public-v1
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "experiments" / "runs"
FPG_RUN = RUNS / "kaggle-A3-FPG-public-v1"
FPG_MEAN = 44.21
POOL = ROOT / "experiments" / "arc3-hl" / "pool.json"
ENTRY_D, ENTRY_P, NO_GAIN_D = 5.0, 0.025, 1.0
IN_PROMPT_SHARE, HLA_VALID_SHARE = 0.95, 0.90
ARCHIVES = {"hlb": "kaggle-A3-HLB-public-v1", "hla": "kaggle-A3-HLA-public-v1", "hls": "kaggle-A3-HLS-public-v1",
            "hlc": "kaggle-A3-HLC-public-v1"}
RULE = ("HL, frozen (reports/research/ARC3_HINT_LIBRARY_RULE_2026-10-30.md at a561b9f; "
        "19:27 IST Oct 30): validity (twin passed; server loads with no error; "
        ">= 96 of 100 game-passes scored; every level entered logs its lookups; picked hints in the system prompt at "
        "the level's last call on >= 95% of levels entered; HLA: >= 90% of its calls valid); D = arm mean game score - "
        "FPG's 44.21, game-paired permutation, two-sided, 20,000 draws, each arm at p <= 0.025: D >= +5 and p <= 0.025 "
        "entry candidate; D <= +1 no measurable gain; otherwise report")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


A = _load("arc3_oif_analyze", HERE / "arc3_oif_analyze.py")
GAMES = A.GAMES


def band(d: float, p: float) -> str:
    if d >= ENTRY_D and p <= ENTRY_P:
        return "entry candidate"
    if d <= NO_GAIN_D:
        return "no measurable gain"
    return "report"


def arm_of(archive: Path) -> str | None:
    for arm, name in ARCHIVES.items():
        if archive.name.startswith(name.rsplit("-v", 1)[0]):
            return arm
    return None


def load_logs(run: Path) -> tuple[dict, dict, dict, list]:
    """Level records and lookups by (game, pass, level), state records by (game, pass, level), and arm errors."""
    levels, lookups, errors = collections.defaultdict(list), collections.defaultdict(list), []
    p = run / "hl_lookups.jsonl"
    for line in (p.read_text(encoding="utf-8").splitlines() if p.is_file() else []):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "error":
            errors.append(r.get("error"))
            continue
        k = (str(r["game_id"]).split("-")[0], r.get("pass_index"), int(r["level"]))
        (levels if r.get("kind") == "level" else lookups)[k].append(r)
    states = {}
    for sp in sorted(run.glob("hl_state_*.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        for lv, rec in (d.get("levels") or {}).items():
            states[(str(d["game_id"]).split("-")[0], d.get("pass_index"), int(lv))] = rec
    return levels, lookups, states, errors


def _permutation(a: list[dict], b: list[dict]) -> tuple[float, float]:
    return A.WM.permutation([{"game": r["game"], "score": r["score"]} for r in a],
                            [{"game": r["game"], "score": r["score"]} for r in b])


def _scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["state"] in A.SCORECARD_STATES and isinstance(r["score"], (int, float))]


def analyze(archive: Path, fpg_path: Path = FPG_RUN / "benchmark.json") -> dict:
    if arm_of(archive) in ("hls", "hlc"):
        return analyze_cs(archive, fpg_path)
    arm = arm_of(archive)
    out: dict = {"rule": RULE, "archive": str(archive), "arm": arm}
    twin = A.twin_passed(archive)
    bench = archive / "benchmark.json"
    if not bench.is_file():
        out.update({"decision": "pending: no benchmark.json yet", "twin": twin})
        return out
    rows, fpg = A.load_runs(bench), A.load_runs(fpg_path)
    passes = int(json.loads(bench.read_text(encoding="utf-8")).get("n_passes") or 0)
    scored = _scored(rows)
    expected = GAMES * passes
    need = math.ceil(A.MIN_SCORECARDS_SHARE * expected)
    serve = A.server_load(archive)
    levels, lookups, states, errors = load_logs(archive)
    entered = [(r["game"], r["pass"], lv) for r in rows for lv in A.levels_reached(r)]
    missing = [k for k in entered if not levels.get(k) or not lookups.get(k)
               or not all(x.get("trigger") and x.get("picker") and "ids" in x for x in lookups[k])]
    in_prompt = [k for k in entered if (states.get(k) or {}).get("in_prompt_last_call") is True]
    share_in_prompt = len(in_prompt) / len(entered) if entered else 0.0
    all_looks = [x for v in lookups.values() for x in v]
    hla_looks = [x for x in all_looks if "hla" in x]
    hla_valid = sum(1 for x in hla_looks if (x.get("hla") or {}).get("valid"))
    checks = {"twin_passed": twin is True, "server_loaded_no_error": serve["ok"],
              "scorecards_96": expected > 0 and len(scored) >= need, "lookups_logged": bool(entered) and not missing,
              "in_prompt_95": share_in_prompt >= IN_PROMPT_SHARE}
    if arm == "hla":
        checks["hla_valid_90"] = bool(hla_looks) and hla_valid / len(hla_looks) >= HLA_VALID_SHARE
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "passes": passes, "game_passes": len(rows),
                       "scored": len(scored), "scored_needed": need, "server": serve,
                       "left_out_no_score": [f"{r['game']}_p{r['pass']}" for r in rows if r not in scored],
                       "levels_entered": len(entered), "levels_without_lookups": [f"{g}_p{p}_L{lv}" for g, p, lv in missing][:20],
                       "n_levels_without_lookups": len(missing),
                       "in_prompt_last_call": len(in_prompt), "in_prompt_share": round(share_in_prompt, 4),
                       "not_in_prompt": [f"{g}_p{p}_L{lv}" for g, p, lv in entered if (g, p, lv) not in set(in_prompt)][:20],
                       "arm_errors": errors[:10], "n_arm_errors": len(errors),
                       "hla_calls": len(hla_looks), "hla_valid": hla_valid,
                       "hla_valid_share": round(hla_valid / len(hla_looks), 4) if hla_looks else None}
    d, p = _permutation(fpg, scored)
    b = band(d, p)
    out["reading"] = {"arm_mean": round(st.mean(r["score"] for r in scored), 2) if scored else None,
                      "FPG_mean": round(st.mean(r["score"] for r in fpg), 2), "FPG_rule_mean": FPG_MEAN,
                      "D": round(d, 2), "p": round(p, 4), "band": b, "game_passes": len(scored),
                      "FPG_game_passes": len(fpg)}
    out["decision"] = b if out["validity"]["valid"] else "no decision (validity failed): report only"
    out["arm_side"], out["fpg_side"] = A.side(scored), A.side(fpg)
    games = sorted({r["game"] for r in fpg} | {r["game"] for r in scored})
    out["per_game"] = {}
    for g in games:
        f = [r["score"] for r in scored if r["game"] == g]
        q = [r["score"] for r in fpg if r["game"] == g]
        out["per_game"][g] = {"arm": round(st.mean(f), 2) if f else None, "FPG": round(st.mean(q), 2) if q else None,
                              "delta": round(st.mean(f) - st.mean(q), 2) if f and q else None}
    # recorded beside
    per_level = collections.Counter(len(v) for v in lookups.values())
    triggers = collections.Counter(x.get("trigger") for x in all_looks)
    pickers = collections.Counter(x.get("picker") for x in all_looks)
    pool = {e["id"]: e for e in json.loads(POOL.read_text(encoding="utf-8"))["entries"]} if POOL.is_file() else {}
    picked = collections.Counter(i for x in all_looks for i in x.get("ids") or [])
    hla_s = sum(float((x.get("hla") or {}).get("seconds") or 0) for x in hla_looks)
    wall = sum(float(r["wall_s"] or 0) for r in rows)
    out["beside"] = {
        "lookups": len(all_looks), "lookups_per_level": dict(sorted(per_level.items())),
        "lookups_per_level_mean": round(len(all_looks) / len(lookups), 2) if lookups else None,
        "triggers": dict(triggers), "stuck_share": round((triggers["stuck_actions"] + triggers["stuck_time"])
                                                         / len(all_looks), 4) if all_looks else None,
        "pickers": dict(pickers),
        "most_picked": [{"id": i, "n": n, "source": (pool.get(i) or {}).get("source"),
                         "text": (pool.get(i) or {}).get("text", "")[:120]} for i, n in picked.most_common(10)],
        "picked_by_source": dict(collections.Counter((pool.get(i) or {}).get("source") for i in picked.elements())),
        "hla_fallbacks": pickers.get("hla_fallback", 0), "hla_seconds": round(hla_s, 1),
        "hla_time_share": round(hla_s / wall, 4) if wall and hla_looks else None,
    }
    other = next((a for a in ARCHIVES if a != arm), None) if arm else None
    other_bench = RUNS / ARCHIVES[other] / "benchmark.json" if other else None
    if other_bench is not None and other_bench.is_file():
        o = _scored(A.load_runs(other_bench))
        dd, pp = _permutation(o, scored)
        out["beside"]["vs_other_arm"] = {"other": other, "other_mean": round(st.mean(r["score"] for r in o), 2),
                                         "D": round(dd, 2), "p": round(pp, 4)}
    return out


def _p(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


# ------------------------------------------------------------------------------------------------- HLS and HLC
# Rule reports/research/ARC3_HL_CACHE_SAFE_RULE_2026-11-01.md (f2c4ab2f): the cache gate, then the score gate, then the
# pick; the gain claim as in the HL rule.

RULE_CS = ("HLS and HLC, frozen (reports/research/ARC3_HL_CACHE_SAFE_RULE_2026-11-01.md at f2c4ab2f; "
           "00:22 IST Nov 1): validity (twin passed; server loads with no error; >= 96 of 100 "
           "game-passes scored); D = arm mean game score - FPG's 44.21, game-paired permutation, two-sided, 20,000 draws. "
           "1. cache gate: prefix-cache hit >= 92% and >= 19,100 actions, else not eligible (92% without the actions: "
           "report); 2. score gate: D >= -2 eligible, D <= -5 dropped, between report (a judgement call); 3. pick: the "
           "eligible arm with the larger D, HLS if within 2.0; 4. gain claim only at D >= +5 and p <= 0.025")
CACHE_GATE, ACTIONS_GATE = 0.92, 19_100
KEEP_D, DROP_D, PICK_MARGIN = -2.0, -5.0, 2.0
GAIN_D, GAIN_P = 5.0, 0.025
HLB_RUN = RUNS / "kaggle-A3-HLB-public-v1"
P0_RUN = A.P0_RUN
PREFILL = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")


def cache_hit(run: Path) -> dict:
    """Prefix-cache hit from serve.log: cached tokens over cached plus new, over the prefill lines."""
    new = cached = lines = 0
    p = run / "serve.log"
    for line in (p.open(encoding="utf-8", errors="replace") if p.is_file() else []):
        m = PREFILL.search(line)
        if m:
            lines += 1
            new += int(m.group(2))
            cached += int(m.group(3))
    return {"prefill_lines": lines, "new_tokens": new, "cached_tokens": cached,
            "hit": round(cached / (cached + new), 4) if cached + new else None}


def system_runs(run: Path) -> dict:
    """The system prompt's sha256 runs per game-pass, from the arm's state files (levels in order)."""
    per = {}
    for sp in sorted(run.glob("hl_state_*.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        runs: list[str] = []
        for lv in sorted(d.get("levels") or {}, key=int):
            for h, _n in d["levels"][lv].get("system_sha") or []:
                if not runs or runs[-1] != h:
                    runs.append(h)
        per[f"{str(d['game_id']).split('-')[0]}_p{d.get('pass_index')}"] = {"runs": len(runs), "distinct": len(set(runs))}
    return {"game_passes": len(per), "one_prompt": sum(1 for v in per.values() if v["distinct"] == 1),
            "max_distinct": max((v["distinct"] for v in per.values()), default=0),
            "more_than_one": sorted(k for k, v in per.items() if v["distinct"] > 1)[:20]}


def score_at(r: dict, n: int) -> float:
    """Game score from the levels cleared within the game-pass's first n actions, in the benchmark's form:
    100 x min(s / w, c / w), s the level-index weighted level scores (each capped at 1.15), c the cleared levels'
    weights, w all levels' weights. Equals benchmark.json final_score at n = inf. Recorded beside only."""
    total, done = 0, []
    for lv in range(1, r["cleared"] + 1):
        total += int(r["apl"][lv - 1])
        if total > n:
            break
        done.append(lv)
    w = sum(range(1, r["levels"] + 1))
    s, c = sum(lv * A.level_score(r, lv) for lv in done), sum(done)
    return 100 * min(s / w, c / w) if w else 0.0


def early_clears(rows: list[dict], k: int = 25) -> dict:
    entered = sum(len(A.levels_reached(r)) for r in rows)
    fast = sum(1 for r in rows for lv in range(1, r["cleared"] + 1) if int(r["apl"][lv - 1]) <= k)
    return {"levels_entered": entered, "cleared_within": fast, "share": round(fast / entered, 4) if entered else None}


def cache_gate(hit: float | None, actions: int) -> str:
    if hit is not None and hit >= CACHE_GATE and actions >= ACTIONS_GATE:
        return "pass"
    if hit is not None and hit >= CACHE_GATE:
        return "report: hit reached, actions short (the throughput account is wrong for this arm)"
    return "not eligible"


def score_gate(d: float) -> str:
    if d >= KEEP_D:
        return "eligible"
    if d <= DROP_D:
        return "dropped"
    return "report (a judgement call)"


def analyze_cs(archive: Path, fpg_path: Path = FPG_RUN / "benchmark.json", other: bool = True) -> dict:
    arm = arm_of(archive)
    out: dict = {"rule": RULE_CS, "archive": str(archive), "arm": arm}
    twin = A.twin_passed(archive)
    bench = archive / "benchmark.json"
    if not bench.is_file():
        out.update({"decision": "pending: no benchmark.json yet", "twin": twin})
        return out
    rows, fpg = A.load_runs(bench), A.load_runs(fpg_path)
    passes = int(json.loads(bench.read_text(encoding="utf-8")).get("n_passes") or 0)
    scored = _scored(rows)
    need = math.ceil(A.MIN_SCORECARDS_SHARE * GAMES * passes)
    serve = A.server_load(archive)
    checks = {"twin_passed": twin is True, "server_loaded_no_error": serve["ok"],
              "scorecards_96": passes > 0 and len(scored) >= need}
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "passes": passes, "game_passes": len(rows),
                       "scored": len(scored), "scored_needed": need, "server": serve,
                       "left_out_no_score": [f"{r['game']}_p{r['pass']}" for r in rows if r not in scored]}
    cache, actions = cache_hit(archive), sum(r["actions"] for r in rows)
    ref = {"FPG": {**cache_hit(FPG_RUN), "actions": sum(r["actions"] for r in fpg)}}
    if (HLB_RUN / "benchmark.json").is_file():
        ref["HLB"] = {**cache_hit(HLB_RUN), "actions": sum(r["actions"] for r in A.load_runs(HLB_RUN / "benchmark.json"))}
    d, p = _permutation(fpg, scored)
    g1 = cache_gate(cache["hit"], actions)
    g2 = score_gate(d)
    eligible = out["validity"]["valid"] and g1 == "pass" and g2 == "eligible"
    out["recorded"] = {"cache": cache, "actions": actions, "system_prompt_runs": system_runs(archive), "reference": ref}
    out["reading"] = {"arm_mean": round(st.mean(r["score"] for r in scored), 2) if scored else None,
                      "FPG_mean": round(st.mean(r["score"] for r in fpg), 2), "FPG_rule_mean": FPG_MEAN,
                      "D": round(d, 2), "p": round(p, 4), "cache_gate": g1, "score_gate": g2, "eligible": eligible,
                      "gain_claim": d >= GAIN_D and p <= GAIN_P}
    if not out["validity"]["valid"]:
        out["decision"] = "no decision (validity failed): report only"
    elif g1 != "pass":
        out["decision"] = f"cache gate: {g1}"
    else:
        out["decision"] = f"score gate: {g2}" + (" (gain claim met)" if out["reading"]["gain_claim"] else "")
    # recorded beside
    hlb = A.load_runs(HLB_RUN / "benchmark.json") if (HLB_RUN / "benchmark.json").is_file() else []
    p0 = A.load_runs(P0_RUN / "benchmark.json") if (P0_RUN / "benchmark.json").is_file() else []
    at = {}
    for n in (100, 150):
        arm_at = [{"game": r["game"], "score": score_at(r, n)} for r in scored]
        at[n] = {"arm": round(st.mean(x["score"] for x in arm_at), 2) if arm_at else None}
        for name, ref_rows in (("FPG", fpg), ("HLB", hlb)):
            if ref_rows:
                ra = [{"game": r["game"], "score": score_at(r, n)} for r in _scored(ref_rows)]
                dd, pp = A.WM.permutation(ra, arm_at)
                at[n][name] = {"mean": round(st.mean(x["score"] for x in ra), 2), "D": round(dd, 2), "p": round(pp, 4)}
    levels, lookups, states, errors = load_logs(archive)
    looks = [x for v in lookups.values() for x in v]
    hlc = {}
    if arm == "hlc":
        entered = [(r["game"], r["pass"], lv) for r in rows for lv in A.levels_reached(r)]
        trimmed = [k for k in entered if (states.get(k) or {}).get("in_prompt_last_call") is False]
        hlc = {"lookups_per_level": round(len(looks) / len(lookups), 2) if lookups else None,
               "levels_entered": len(entered), "section_trimmed_from_view": len(trimmed),
               "trimmed_share": round(len(trimmed) / len(entered), 4) if entered else None,
               "section_in_system_calls": sum(int(v.get("section_in_system") or 0) for v in states.values())}
    out["beside"] = {"score_at_actions": at,
                     "early_clears": {"arm": early_clears(scored), "FPG": early_clears(fpg),
                                      **({"P0": early_clears(p0)} if p0 else {})},
                     "lookups": len(looks), "pickers": dict(collections.Counter(x.get("picker") for x in looks)),
                     "arm_errors": len(errors), "hlc": hlc}
    out["arm_side"], out["fpg_side"] = A.side(scored), A.side(fpg)
    out["per_game"] = {}
    for g in sorted({r["game"] for r in fpg} | {r["game"] for r in scored}):
        f = [r["score"] for r in scored if r["game"] == g]
        q = [r["score"] for r in fpg if r["game"] == g]
        out["per_game"][g] = {"arm": round(st.mean(f), 2) if f else None, "FPG": round(st.mean(q), 2) if q else None,
                              "delta": round(st.mean(f) - st.mean(q), 2) if f and q else None}
    if other:
        o = "hlc" if arm == "hls" else "hls"
        orun = RUNS / ARCHIVES[o]
        if (orun / "benchmark.json").is_file():
            orep = analyze_cs(orun, fpg_path, other=False)
            out["pick"] = pick({arm: out, o: orep})
    return out


def pick(reps: dict) -> dict:
    """The eligible arm with the larger D; HLS if the two are within 2.0 points."""
    el = {a: r["reading"]["D"] for a, r in reps.items() if (r.get("reading") or {}).get("eligible")}
    if not el:
        return {"pick": None, "eligible": [], "why": "no arm eligible"}
    if len(el) == 2 and abs(el["hls"] - el["hlc"]) <= PICK_MARGIN:
        return {"pick": "hls", "eligible": sorted(el), "why": f"within {PICK_MARGIN} points: HLS"}
    best = max(el, key=el.get)
    return {"pick": best, "eligible": sorted(el), "why": "the eligible arm with the larger D"}


def format_report_cs(rep: dict) -> str:
    name = (rep.get("arm") or "?").upper()
    if "validity" not in rep:
        return f"{name} (HL cache-safe): {rep['decision']}; twin {rep.get('twin')}\nDecision: {rep['decision']}\n"
    v, rd, rc, bs = rep["validity"], rep["reading"], rep["recorded"], rep["beside"]
    s, f = rep["arm_side"], rep["fpg_side"]
    ref = rc["reference"]
    L = [f"{name} (arm {rep['arm']}, the cache-safe hint rule) on FPG's public run; {v['passes']} passes, "
         f"{v['game_passes']} game-passes",
         f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
         f"  scored {v['scored']} (need {v['scored_needed']}); server ready line {v['server'].get('ready_line')}, errors "
         f"before it {v['server'].get('errors_before_ready')}; left out of D {v['left_out_no_score']}",
         "",
         "| Track | Run | prefix-cache hit | prefill lines | actions | mean score |",
         "|---|---|---|---|---|---|"]
    for label, c, act, mean in [("FPG", ref["FPG"], ref["FPG"]["actions"], f["mean_score"])] + (
            [("HLB", ref["HLB"], ref["HLB"]["actions"], None)] if "HLB" in ref else []) + [
            (name, rc["cache"], rc["actions"], s["mean_score"])]:
        L.append(f"| ARC-3 | {label} | {c['hit']:.1%} | {c['prefill_lines']} | {act} | {mean if mean is not None else '41.27 (HL read)'} |")
    sr = rc["system_prompt_runs"]
    L += ["",
          f"system prompt per game-pass (state files): {sr['one_prompt']} of {sr['game_passes']} with one sha256; max "
          f"{sr['max_distinct']}; more than one: {sr['more_than_one']}",
          f"1. cache gate (hit >= 92% and actions >= 19,100): {rd['cache_gate']}",
          f"2. D = {rd['D']:+.2f} ({name} {rd['arm_mean']} - FPG {rd['FPG_mean']}; rule's FPG {rd['FPG_rule_mean']}), "
          f"game-paired permutation, two-sided, 20,000 draws: p {_p(rd['p'])}; score gate: {rd['score_gate']}",
          f"   eligible: {rd['eligible']}; gain claim (D >= +5 and p <= 0.025): {rd['gain_claim']}"]
    if "pick" in rep:
        L.append(f"3. pick: {rep['pick']['pick']} ({rep['pick']['why']}; eligible {rep['pick']['eligible']})")
    else:
        L.append("3. pick: pending the other arm's archive")
    L += ["", "recorded beside, no decision:"]
    for n, x in bs["score_at_actions"].items():
        L.append(f"  score at {n} actions per game-pass: {name} {x['arm']}; " + "; ".join(
            f"{k} {y['mean']} (D {y['D']:+.2f}, p {_p(y['p'])})" for k, y in x.items() if k != "arm"))
    ec = bs["early_clears"]
    L.append("  levels cleared within 25 actions of entering: " + "; ".join(
        f"{k} {x['cleared_within']} of {x['levels_entered']} ({x['share']})" for k, x in ec.items()))
    L.append(f"  lookups {bs['lookups']}, pickers {bs['pickers']}, arm errors {bs['arm_errors']}")
    if bs["hlc"]:
        h = bs["hlc"]
        L.append(f"  HLC: lookups per level {h['lookups_per_level']}; section trimmed from view at the level's last call "
                 f"on {h['section_trimmed_from_view']} of {h['levels_entered']} levels ({h['trimmed_share']}); calls "
                 f"with the section in the system prompt {h['section_in_system_calls']}")
    L += ["",
          "| Track | Run | game-passes | mean score | levels per pass | won | tokens per action | wall-budget endings |",
          "|---|---|---|---|---|---|---|---|"]
    for label, x in (("FPG", f), (name, s)):
        L.append(f"| ARC-3 | {label} | {x['game_passes']} | {x['mean_score']} | {x['levels_per_pass']} | {x['won']} | "
                 f"{x['tokens_per_action']} | {x['wall_budget_endings']} |")
    L += ["", "| Track | game | " + name + " | FPG | delta |", "|---|---|---|---|---|"]
    L += [f"| ARC-3 | {g} | {x['arm']} | {x['FPG']} | {x['delta']} |" for g, x in rep["per_game"].items()]
    L += ["", f"Rule: {rep['rule']}", f"Decision: {rep['decision']}"]
    return "\n".join(L) + "\n"


def format_report(rep: dict) -> str:
    if rep.get("arm") in ("hls", "hlc"):
        return format_report_cs(rep)
    name = (rep.get("arm") or "?").upper()
    if "validity" not in rep:
        return f"{name} (HL): {rep['decision']}; twin {rep.get('twin')}\nDecision: {rep['decision']}\n"
    v, rd, s, f, bs = rep["validity"], rep["reading"], rep["arm_side"], rep["fpg_side"], rep["beside"]
    L = [f"{name} (HL, arm {rep['arm']}): the hint library in the system prompt on FPG's public run; {v['passes']} "
         f"passes, {v['game_passes']} game-passes",
         f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
         f"  scored {v['scored']} (need {v['scored_needed']}); server ready line {v['server'].get('ready_line')}, errors "
         f"before it {v['server'].get('errors_before_ready')}; left out of D (no score) {v['left_out_no_score']}",
         f"  levels entered {v['levels_entered']}: without lookups {v['n_levels_without_lookups']} "
         f"{v['levels_without_lookups'][:10]}; hints in the system prompt at the last call {v['in_prompt_last_call']} "
         f"({v['in_prompt_share']:.1%}; need 95%) not {v['not_in_prompt'][:10]}; arm errors {v['n_arm_errors']}"
         + (f"; HLA calls {v['hla_calls']}, valid {v['hla_valid']} ({v['hla_valid_share']}; need 0.90)"
            if rep["arm"] == "hla" else ""),
         "",
         "| Track | Run | game-passes | mean score | levels per pass | won | moves per clear (median; mean) | tokens per "
         "action | tokens per game-pass (mean; median) | wall-budget endings |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for label, x in (("FPG (FP8-PLE, public save run)", f), (f"{name} (FPG + the hint library)", s)):
        m = x["moves_per_clear"]
        L.append(f"| ARC-3 | {label} | {x['game_passes']} | {x['mean_score']} | {x['levels_per_pass']} | {x['won']} | "
                 f"{m['median']}; {m['mean']} | {x['tokens_per_action']} | {x['tokens_per_game_pass']['mean']}; "
                 f"{x['tokens_per_game_pass']['median']} | {x['wall_budget_endings']} |")
    L += ["",
          f"D = {rd['D']:+.2f} ({name} {rd['arm_mean']} - FPG {rd['FPG_mean']}; rule's FPG {rd['FPG_rule_mean']}), "
          f"game-paired permutation, two-sided, 20,000 draws: p {_p(rd['p'])} (read at 0.025); band: {rd['band']}",
          "",
          "recorded beside, no decision:",
          f"  lookups {bs['lookups']}: per level {bs['lookups_per_level']} (mean {bs['lookups_per_level_mean']}); "
          f"triggers {bs['triggers']}; stuck share {bs['stuck_share']}; pickers {bs['pickers']}",
          f"  picked by source {bs['picked_by_source']}",
          "  most picked: " + "; ".join(f"{x['id']} ({x['source']}) x{x['n']}" for x in bs["most_picked"])]
    if rep["arm"] == "hla":
        L.append(f"  HLA fallbacks {bs['hla_fallbacks']}; HLA call time {bs['hla_seconds']} s, share of game-pass wall "
                 f"time {bs['hla_time_share']}")
    if "vs_other_arm" in bs:
        o = bs["vs_other_arm"]
        L.append(f"  against {o['other'].upper()} ({o['other_mean']}), paired by game: D {o['D']:+.2f}, p {_p(o['p'])}")
    L += ["", "| Track | game | " + name + " | FPG | delta |", "|---|---|---|---|---|"]
    L += [f"| ARC-3 | {g} | {x['arm']} | {x['FPG']} | {x['delta']} |" for g, x in rep["per_game"].items()]
    L += ["", f"Rule: {rep['rule']}", f"Decision: {rep['decision']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None, help="default: <archive>/hl_decision.json")
    a = ap.parse_args(argv)
    rep = analyze(a.archive)
    print(format_report(rep), end="")
    if a.archive.is_dir() and "validity" in rep:
        (a.json or a.archive / "hl_decision.json").write_text(json.dumps(rep, indent=1, default=str) + "\n",
                                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
