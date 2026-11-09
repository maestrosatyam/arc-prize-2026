#!/usr/bin/env python3
"""C1S analysis (run a3-c1s-selfinsight): the frozen rule of reports/research/ARC3_C1S_SELF_INSIGHT_RULE_2026-10-29.md
(frozen at 99ea2c9) on the three kernels' r1_results.jsonl and c1s_texts.jsonl, arm A7SF (C1's fresh start with the
insight the served model wrote at the level start), against C1's archived arms A0F and A7F on the same levels (C1's
level_map.tsv) and C1P's A7cF where its level map is in. Probe costs: experiments/arc3-c1s/probe_cost.jsonl (1c9e404).

Validity (any failing: no decision, report only):
  - the CPU twin passed for every kernel ("preflight: PASS" in the archived twin log);
  - at least 95% of the 157 level starts replay to their hash on the GPU run (in the generation step or the run);
    failures are excluded here and from C1's arms;
  - at least 95% of the levels get a non-empty generated text; levels without one are excluded and counted;
  - at least 95% of the case-arms that ran end by a clear, the action budget or the wall budget;
  - every notebook cell other than the arm's definition, the introduction, the embedded bundle and the generation step
    is byte-identical to C1's kernel for the same split (kaggle/arc3/c1s/build_c1s_notebook.assert_against_c1 on the
    committed notebooks; the hard-stop line is listed with them); each level ran in its C1 kernel.
Reading (probe-priced): a level counts as cleared by A7SF only when its clear actions plus its probe cost fit within 160.
S = those clears over the N levels compared; M = S - A0F with C1's A0F on the same levels, exact two-sided McNemar on
the discordant pairs (exactly one line):
  self-insight helps   M >= 10 and p <= 0.01
  no help              M <= 3
  report               otherwise
Entry draft: "helps" and A7SF's probe-priced measured value (level-index-weighted points per game, mean over games,
clear actions plus probe cost) above A0F's.
Recorded beside, no decision: the free-probe reading (S, M and p with probes not counted); S against A7F and against
C1P's P on the shared levels; LI1's format lines per level (no "Key insight:", under 8 words, characters); key-element
match (a content word shared with the level's oracle insight text, LI1's content_words; and LI1's key_elements.json on
the levels it covers); speed per arm (wall-budget endings, tokens per action, actions per case-minute, clears within 80
actions / 900 s); clears by band and game; measured value per game, probe-priced and free, beside A0F's and A7F's; probe
cost per level and generation time per level; thinking off (levels where any reasoning text came back); inputs as
committed; the message path (rows that showed the generated message once).

  envs/arc3/bin/python -I scripts/arc3_c1s_analyze.py [--archive experiments/runs/kaggle-A3-C1S-selfinsight-v1]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "experiments" / "runs"
ARCHIVE = RUNS / "kaggle-A3-C1S-selfinsight-v1"
C1_ARCHIVE = RUNS / "kaggle-A3-C1-levelstarts-v1"
C1P_ARCHIVE = RUNS / "kaggle-A3-C1P-placebo-v1"
PROBES = ROOT / "experiments" / "arc3-c1s" / "probe_cost.jsonl"
PROBES_JSONL_SHA256 = "eaa924ab47e21966cb642cfc9e3dc5b8d63cb7d57832cf7fe3d3988d3879b85d"
C1_LEVELS = ROOT / "experiments" / "arc3-c1" / "levels.jsonl"
KEY_ELEMENTS = ROOT / "experiments" / "arc3-team-play" / "key_elements.json"
KERNELS = ("k1", "k2", "k3")
ARM = "A7SF"
BUDGET = 160
HELPS_M, HELPS_P, NO_HELP_M = 10, 0.01, 3
TEXT_SHARE = 0.95
RULE = ("C1S, frozen (2026-10-29 14:25 IST; ARC3_C1S_SELF_INSIGHT_RULE_2026-10-29.md at 99ea2c9): "
        "validity (twin passed; >= 95% of level starts replay to their hash, failures excluded here and from C1's arms; "
        ">= 95% of levels get a non-empty generated text; >= 95% of case-arms end by clear / action budget / wall budget; "
        "cells byte-identical to C1's except the arm, the introduction, the bundle and the generation step); "
        "probe-priced: a clear counts only if clear actions + probe cost <= 160; M = S - A0F (C1's A0F), exact McNemar: "
        "M >= 10 and p <= 0.01 self-insight helps; M <= 3 no help; otherwise report; entry draft: helps and A7SF's "
        "probe-priced measured value above A0F's")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


C1A = _load("arc3_c1_analyze", HERE / "arc3_c1_analyze.py")
LIA = _load("arc3_li1_analyze", HERE / "arc3_li1_analyze.py")
mcnemar_p, cleared, cleared_within, level_score, usage = C1A.mcnemar_p, C1A.cleared, C1A.cleared_within, \
    C1A.level_score, C1A.usage
content_words = LIA.content_words


def load_probes(path: Path = PROBES) -> dict[str, dict]:
    data = path.read_bytes()
    if path == PROBES:
        assert hashlib.sha256(data).hexdigest() == PROBES_JSONL_SHA256, "probe_cost.jsonl differs from the committed list"
    return {r["case_id"]: r for r in map(json.loads, filter(str.strip, data.decode("utf-8").splitlines()))}


def load_level_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split("\t")
    return {row["case_id"]: row for row in (dict(zip(cols, l.split("\t"))) for l in lines[1:] if l.strip())}


def load_run(archive: Path) -> dict:
    """Per kernel: the twin verdict, the A7SF rows, the generated texts, the generation summary."""
    out = {"rows": {}, "texts": {}, "twins": {}, "generation": {}, "dupes": collections.defaultdict(list), "other": []}
    for k in KERNELS:
        d = archive / k
        logs = sorted((d / "preflight").glob("*.log")) if (d / "preflight").is_dir() else []
        out["twins"][k] = any("preflight: PASS" in p.read_text(encoding="utf-8", errors="replace") for p in logs) \
            if logs else None
        f = d / "r1_results.jsonl"
        if f.is_file():
            for r in map(json.loads, filter(str.strip, f.read_text(encoding="utf-8").splitlines())):
                if r.get("arm") != ARM:
                    out["other"].append(f"{r.get('case_id')} {r.get('arm')}")
                    continue
                cid = str(r.get("case_id"))
                if cid in out["rows"]:
                    out["dupes"][k].append(cid)
                out["rows"][cid] = dict(r, _kernel=k)
        t = d / "c1s_texts.jsonl"
        if t.is_file():
            for r in map(json.loads, filter(str.strip, t.read_text(encoding="utf-8").splitlines())):
                out["texts"][str(r["case_id"])] = dict(r, _kernel=k)
        g = d / "c1s_generation.json"
        if g.is_file():
            out["generation"][k] = json.loads(g.read_text(encoding="utf-8"))
    out["dupes"] = dict(out["dupes"])
    return out


def c1_rows(c1_archive: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for k in KERNELS:
        f = c1_archive / k / "r1_results.jsonl"
        if f.is_file():
            for r in map(json.loads, filter(str.strip, f.read_text(encoding="utf-8").splitlines())):
                out[(str(r["case_id"]), str(r["arm"]))] = r
    return out


def cells_identical(root: Path = ROOT) -> dict:
    """build_c1s_notebook.assert_against_c1 and assert_twin on the committed C1S notebooks."""
    out = {}
    try:
        S = _load("build_c1s_notebook", root / "kaggle" / "arc3" / "c1s" / "build_c1s_notebook.py")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    for k in KERNELS:
        d = root / "kaggle" / "arc3" / "c1s" / k
        try:
            nb = json.loads((d / f"{S.slug(k)}.ipynb").read_text(encoding="utf-8"))
            pre = json.loads((d / "preflight" / f"{S.slug(k)}-preflight.ipynb").read_text(encoding="utf-8"))
            S.assert_against_c1(nb, k)
            S.assert_twin(pre, nb, k)
            out[k] = True
        except Exception as exc:  # noqa: BLE001
            out[k] = f"{type(exc).__name__}: {exc}"[:200]
    return {"ok": all(v is True for v in out.values()), "kernels": out,
            "differences": "cell 0 (introduction); cell 17: header, arms line, bundle, arm registration, generation step"}


def priced_clear(r: dict | None, cost: int) -> bool:
    """A7SF's clear counted only when its clear actions plus the level's probe cost fit within the budget."""
    return cleared(r) and r.get("clear_actions") is not None and int(r["clear_actions"]) + int(cost) <= BUDGET


def priced_score(r: dict | None, cost: int, baseline: int) -> float:
    if not priced_clear(r, cost):
        return 0.0
    return min(C1A.SCORE_CAP, (baseline / (int(r["clear_actions"]) + int(cost))) ** 2)


def analyze(archive: Path, levels: list[dict], probes: dict[str, dict], level_map: dict[str, dict],
            c1p_map: dict[str, dict] | None = None, c1_archive: Path = C1_ARCHIVE, static: dict | None = None,
            key_elements: Path = KEY_ELEMENTS) -> dict:
    run_data = load_run(archive)
    rows, texts, twins = run_data["rows"], run_data["texts"], run_data["twins"]
    lv = {l["case_id"]: l for l in levels}
    out: dict = {"rule": RULE, "archive": str(archive), "levels_listed": len(levels),
                 "kernels_present": sorted({r["_kernel"] for r in rows.values()}), "twins": twins,
                 "duplicate_rows": run_data["dupes"], "other_arm_rows": run_data["other"],
                 "extra_rows": sorted(set(rows) - set(lv)), "generation": run_data["generation"]}
    missing_kernels = [k for k in KERNELS if not (archive / k / "r1_results.jsonl").is_file()]
    if missing_kernels:
        out["missing_kernels"] = missing_kernels
    static = static if static is not None else {"cells": cells_identical()}
    out["static"] = static

    # validity
    no_text, gen_replay_bad, replay_bad, kernel_bad, missing_text_row, played_without_text = [], [], [], [], [], []
    for cid, l in lv.items():
        t = texts.get(cid)
        if t is None:
            missing_text_row.append(cid)
        elif not t.get("message"):
            no_text.append(cid)
            if str(t.get("error") or "").startswith("ValueError: the level start does not replay"):
                gen_replay_bad.append(cid)
        r = rows.get(cid)
        if r is not None:
            if r.get("replay_ok") is not True or r.get("end_reason") == "replay_mismatch":
                replay_bad.append(cid)
            if r["_kernel"] != l["kernel"]:
                kernel_bad.append(cid)
            if not (t or {}).get("message"):
                played_without_text.append(cid)
    with_text = [c for c in lv if (texts.get(c) or {}).get("message")]
    not_played = sorted(c for c in with_text if c not in rows)
    ran = [c for c in with_text if c in rows and c not in replay_bad]
    ended = sum(1 for c in ran if rows[c].get("end_reason") in C1A.RAN_TO_END)
    ended_of = len(ran) + len(not_played)          # a case-arm that never ran did not end by a clear or a budget
    n_list = len(levels)
    replay_fail = sorted(set(replay_bad) | set(gen_replay_bad))
    checks = {
        "twin_passed": all(twins.get(k) is True for k in KERNELS),
        "replay_95": n_list - len(replay_fail) - len(missing_text_row) >= math.ceil(C1A.REPLAY_SHARE * n_list),
        "texts_95": len(with_text) >= math.ceil(TEXT_SHARE * n_list),
        "ended_95": bool(ran) and ended >= math.ceil(C1A.ENDED_SHARE * ended_of),
        "cells_identical_to_c1": bool(static["cells"].get("ok")) and not kernel_bad and not run_data["dupes"]
        and not run_data["other"] and not out["extra_rows"],
    }
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "replay_failures": replay_fail,
                       "replay_needed": math.ceil(C1A.REPLAY_SHARE * n_list), "no_text": sorted(no_text),
                       "no_text_row": sorted(missing_text_row), "texts": len(with_text),
                       "texts_needed": math.ceil(TEXT_SHARE * n_list), "with_text_not_played": not_played,
                       "played_without_text": sorted(played_without_text), "ended": ended,
                       "ended_needed": math.ceil(C1A.ENDED_SHARE * ended_of), "kernel_mismatch": sorted(kernel_bad)}
    comp = [c for c in ran if c in level_map]
    out["validity"]["not_in_c1_level_map"] = sorted(c for c in ran if c not in level_map)

    cost = {c: int(probes[c]["probe_cost"]) for c in lv}

    def c1c(c: str, arm: str) -> bool:
        return level_map[c][f"{arm}_cleared"] == "1"

    def clears(c: str, arm: str) -> bool:
        if arm == "S":
            return priced_clear(rows[c], cost[c])
        if arm == "S_free":
            return cleared(rows[c])
        if arm == "P":
            return (c1p_map or {})[c]["A7cF_cleared"] == "1"
        return c1c(c, arm)

    def paired(cs: list[str], x: str, y: str) -> dict:
        b = sum(clears(c, x) and not clears(c, y) for c in cs)
        d = sum(clears(c, y) and not clears(c, x) for c in cs)
        sx, sy = sum(clears(c, x) for c in cs), sum(clears(c, y) for c in cs)
        return {"N": len(cs), x: sx, y: sy, "diff": sx - sy, f"{x}_only": b, f"{y}_only": d, "p": mcnemar_p(b, d)}

    # reading: probe-priced
    rd = paired(comp, "S", "A0F")
    m, p = rd["diff"], rd["p"]
    if m >= HELPS_M and p is not None and p <= HELPS_P:
        band = "self-insight helps"
    elif m <= NO_HELP_M:
        band = "no help"
    else:
        band = "report"
    out["reading"] = {"N": rd["N"], "S": rd["S"], "A0F": rd["A0F"], "M": m, "S_only": rd["S_only"],
                      "A0F_only": rd["A0F_only"], "p": p, "band": band,
                      "lost_to_probe_cost": sum(clears(c, "S_free") and not clears(c, "S") for c in comp)}

    # measured value per game: probe-priced, free, A0F, A7F
    games = collections.defaultdict(list)
    for c in comp:
        games[lv[c]["game"]].append(c)
    value = {}
    for game, cs in sorted(games.items()):
        n_levels = lv[cs[0]]["n_levels"]
        weight = n_levels * (n_levels + 1) / 2
        pts = {"S": sum(lv[c]["level"] * priced_score(rows[c], cost[c], lv[c]["baseline_actions"]) for c in cs),
               "S_free": sum(lv[c]["level"] * level_score(rows[c], lv[c]["baseline_actions"]) for c in cs),
               "A0F": sum(lv[c]["level"] * float(level_map[c]["A0F_score"]) for c in cs),
               "A7F": sum(lv[c]["level"] * float(level_map[c]["A7F_score"]) for c in cs)}
        value[game] = {"levels_in_run": len(cs), "n_levels": n_levels,
                       **{f"{a}_points": round(100 * v / weight, 2) for a, v in pts.items()}}
    mean = {a: round(statistics.mean(v[f"{a}_points"] for v in value.values()), 2) if value else None
            for a in ("S", "S_free", "A0F", "A7F")}
    out["measured_value"] = {"per_game": value, "games": len(value), "mean_points_over_games": mean}
    entry = band == "self-insight helps" and mean["S"] is not None and mean["S"] > mean["A0F"]
    if missing_kernels:
        out["decision"] = f"pending: kernels {missing_kernels} not in yet"
        out["entry_draft"] = "pending"
    elif not out["validity"]["valid"]:
        out["decision"] = "no decision (validity failed): report only"
        out["entry_draft"] = "no (no decision)"
    else:
        out["decision"] = band
        out["entry_draft"] = (f"{'yes' if entry else 'no'} (helps: {band == 'self-insight helps'}; probe-priced measured "
                              f"value {mean['S']} vs A0F {mean['A0F']})")

    # recorded beside
    fr = paired(comp, "S_free", "A0F")
    out["free_probe"] = {"S": fr["S_free"], "A0F": fr["A0F"], "M": fr["diff"], "S_only": fr["S_free_only"],
                         "A0F_only": fr["A0F_only"], "p": fr["p"]}
    out["oracle_gap"] = {"probe_priced": paired(comp, "A7F", "S"), "free": paired(comp, "A7F", "S_free")}
    if c1p_map:
        shared = [c for c in comp if c in c1p_map]
        out["vs_c1p"] = {"probe_priced": paired(shared, "S", "P"), "free": paired(shared, "S_free", "P")}
    else:
        out["vs_c1p"] = None

    tx = [texts[c] for c in with_text]
    out["format"] = {"levels": len(tx), "no_key_insight_label": sum(bool(t.get("fallback")) for t in tx),
                     "under_8_words": sum(int(t.get("words") or 0) < 8 for t in tx),
                     "chars_median": statistics.median(len(t["text"]) for t in tx) if tx else None,
                     "chars_min": min((len(t["text"]) for t in tx), default=None),
                     "chars_max": max((len(t["text"]) for t in tx), default=None),
                     "at_cap": sum(len(t["text"]) > 0.95 * 400 for t in tx)}
    km = [c for c in with_text if content_words(lv[c]["insight"]["text"]) & content_words(texts[c]["text"])]
    ke = json.loads(key_elements.read_text(encoding="utf-8")) if key_elements.is_file() else {"levels": []}
    kw = {f"{l['game']}_L{int(l['level'])}": content_words(" ".join(l["key_objects"]) + " " + l["key_use"])
          for l in ke["levels"]}
    ke_cover = [c for c in with_text if c in kw]
    out["key_element_match"] = {
        "oracle_text": {"match": len(km), "levels": len(with_text),
                        "S_on_match": sum(clears(c, "S") for c in km if c in comp),
                        "S_on_no_match": sum(clears(c, "S") for c in comp if c not in km)},
        "key_elements_json": {"match": sum(bool(kw[c] & content_words(texts[c]["text"])) for c in ke_cover),
                              "levels": len(ke_cover)}}

    crow = c1_rows(c1_archive)

    def row_of(c: str, arm: str) -> dict | None:
        return rows[c] if arm == ARM else crow.get((c, arm))

    speed = {}
    for a in ("A0F", ARM, "A7F"):
        rs = [row_of(c, a) for c in comp]
        if not rs or any(r is None for r in rs):
            speed[a] = None
            continue
        u = usage(rs)
        speed[a] = {"wall_budget_endings": sum(r.get("end_reason") == "wall_budget" for r in rs),
                    "tokens_per_action": u["tokens_per_action"],
                    "actions_per_case_minute": u["actions_per_case_minute_median"],
                    "within_80_900": sum(cleared_within(r) for r in rs)}
    out["speed"] = speed

    arms4 = ("A0F", "S", "S_free", "A7F")

    def per(key):
        res = collections.defaultdict(lambda: {"levels": 0, **{a: 0 for a in arms4}})
        for c in comp:
            g = key(c)
            res[g]["levels"] += 1
            for a in arms4:
                res[g][a] += int(clears(c, a))
        return dict(res)

    out["by_band"] = {k: v for k, v in sorted(per(lambda c: lv[c]["band"]).items(), key=lambda kv: C1A.BANDS.index(kv[0]))}
    out["by_game"] = dict(sorted(per(lambda c: lv[c]["game"]).items()))

    costs = [cost[c] for c in comp]
    gen = [float(texts[c]["gen_s"]) for c in lv if c in texts and texts[c].get("gen_s") is not None]
    toks = [int(texts[c]["new_tokens"]) for c in lv if c in texts and texts[c].get("new_tokens") is not None]
    out["probe_cost"] = {"median": statistics.median(costs) if costs else None, "min": min(costs, default=None),
                         "max": max(costs, default=None)}
    out["generation_time"] = {"levels": len(gen), "median_s": round(statistics.median(gen), 2) if gen else None,
                              "max_s": max(gen, default=None), "total_s": round(sum(gen), 1),
                              "new_tokens_median": statistics.median(toks) if toks else None,
                              "at_token_cap": sum(t >= 256 for t in toks)}
    out["thinking_off"] = {"reasoning_returned": sorted(c for c in lv if (texts.get(c) or {}).get("reasoning")),
                           "finish_length": sum((texts.get(c) or {}).get("finish_reason") == "length" for c in lv)}
    out["inputs_as_committed"] = {"equal": sum((texts.get(c) or {}).get("input_matches_committed") is True for c in lv),
                                  "not_equal": sorted(c for c in lv if c in texts
                                                      and texts[c].get("input_matches_committed") is not True)}
    shown = [c for c in ran if int(rows[c].get("oracle_shown") or 0) == 1 and int(rows[c].get("oracle_errors") or 0) == 0
             and rows[c].get("oracle_kind") == "insight" and int(rows[c].get("oracle_chars") or 0) == len(texts[c]["message"])
             and int(rows[c].get("fresh_notice_shown") or 0) == 1]
    out["message_path"] = {"shown_once_as_generated": len(shown), "rows": len(ran),
                           "other": sorted(set(ran) - set(shown))}

    level_rows = []
    for c in sorted(comp, key=lambda c: (lv[c]["game"], lv[c]["level"])):
        r, t = rows[c], texts[c]
        level_rows.append({
            "case_id": c, "game": lv[c]["game"], "level": lv[c]["level"], "band": lv[c]["band"], "kernel": lv[c]["kernel"],
            "probe_cost": cost[c], "A0F_cleared": int(c1c(c, "A0F")), "A7F_cleared": int(c1c(c, "A7F")),
            "A7SF_cleared_free": int(cleared(r)), "A7SF_cleared_priced": int(priced_clear(r, cost[c])),
            "A7SF_clear_actions": r.get("clear_actions") if cleared(r) else None,
            "A7SF_score_priced": round(priced_score(r, cost[c], lv[c]["baseline_actions"]), 3),
            "A7SF_score_free": round(level_score(r, lv[c]["baseline_actions"]), 3), "A7SF_end": r.get("end_reason"),
            "text_chars": len(t["text"]), "text_words": t.get("words"), "no_key_insight_label": int(bool(t.get("fallback"))),
            "under_8_words": int(int(t.get("words") or 0) < 8), "oracle_word_match": int(c in km),
            "gen_s": t.get("gen_s"), "new_tokens": t.get("new_tokens")})
    out["level_map"] = level_rows
    return out


def _p(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def format_report(rep: dict) -> str:
    v, rd = rep["validity"], rep["reading"]
    lines = [f"C1S (a3-c1s-selfinsight): A7SF (C1's fresh start with the insight the served model wrote at the level "
             f"start) against C1's A0F and A7F on the same levels, 160 actions / 1800 s; {rep['levels_listed']} levels "
             f"listed; kernels present {rep['kernels_present']}; twins {rep['twins']}"]
    if rep.get("missing_kernels"):
        lines.append(f"kernels without results yet: {rep['missing_kernels']}")
    lines += [f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
              f"  replay failures {len(v['replay_failures'])} (need <= {rep['levels_listed'] - v['replay_needed']}) "
              f"{v['replay_failures']}; texts {v['texts']} (need {v['texts_needed']}), no text {v['no_text']}, no text "
              f"row {len(v['no_text_row'])}; ended {v['ended']} (need {v['ended_needed']}); kernel mismatches "
              f"{v['kernel_mismatch']}; with a text but not played {v['with_text_not_played']}",
              f"  cells vs C1 {rep['static']['cells']}",
              ""]
    sp = rep["speed"]
    lines.append(f"{'track':<7}{'arm':<6}{'clears':>9}{'wall-budget':>13}{'within 80/900':>15}{'tokens/action':>15}"
                 f"{'actions/case-min':>18}")
    n = rd["N"]
    counts = {"A0F": rd["A0F"], ARM: rep["free_probe"]["S"], "A7F": rep["oracle_gap"]["free"]["A7F"]}
    for a in ("A0F", ARM, "A7F"):
        s = sp.get(a) or {}
        lines.append(f"{'ARC-3':<7}{a:<6}{counts[a]:>5}/{n:<3}{str(s.get('wall_budget_endings')):>13}"
                     f"{str(s.get('within_80_900')):>15}{str(s.get('tokens_per_action')):>15}"
                     f"{str(s.get('actions_per_case_minute')):>18}")
    lines.append(f"  (A7SF clears here are free-probe; probe-priced S = {rd['S']}, {rd['lost_to_probe_cost']} lost to the "
                 "probe cost)")
    fp, og, vp = rep["free_probe"], rep["oracle_gap"], rep["vs_c1p"]
    lines += ["",
              f"free-probe reading: S {fp['S']} - A0F {fp['A0F']} = {fp['M']} (A7SF only {fp['S_only']}, A0F only "
              f"{fp['A0F_only']}, p {_p(fp['p'])})",
              f"oracle gap: A7F {og['probe_priced']['A7F']} - S {og['probe_priced']['S']} = {og['probe_priced']['diff']} "
              f"probe-priced (p {_p(og['probe_priced']['p'])}); A7F - S free = {og['free']['diff']} (p {_p(og['free']['p'])})"]
    if vp:
        lines.append(f"against C1P's P on {vp['probe_priced']['N']} shared levels: S {vp['probe_priced']['S']} - P "
                     f"{vp['probe_priced']['P']} = {vp['probe_priced']['diff']} probe-priced (p {_p(vp['probe_priced']['p'])}); "
                     f"free {vp['free']['diff']} (p {_p(vp['free']['p'])})")
    else:
        lines.append("against C1P's P: C1P's level map not in yet")
    fm, ke = rep["format"], rep["key_element_match"]
    lines += [f"format (LI1's lines): {fm['levels']} texts; no 'Key insight:' {fm['no_key_insight_label']}; under 8 words "
              f"{fm['under_8_words']}; characters median {fm['chars_median']}, range {fm['chars_min']}-{fm['chars_max']}, "
              f"near the 400 cap {fm['at_cap']}",
              f"key-element match: a content word shared with the oracle insight text {ke['oracle_text']['match']}/"
              f"{ke['oracle_text']['levels']} (S on matching {ke['oracle_text']['S_on_match']}, on the rest "
              f"{ke['oracle_text']['S_on_no_match']}); LI1's key_elements.json {ke['key_elements_json']['match']}/"
              f"{ke['key_elements_json']['levels']}",
              f"probe cost per level: median {rep['probe_cost']['median']}, range {rep['probe_cost']['min']}-"
              f"{rep['probe_cost']['max']}",
              f"generation: {rep['generation_time']['levels']} levels, median {rep['generation_time']['median_s']} s, max "
              f"{rep['generation_time']['max_s']} s, {rep['generation_time']['total_s']} s in all; new tokens median "
              f"{rep['generation_time']['new_tokens_median']}, at the 256 cap {rep['generation_time']['at_token_cap']}",
              f"thinking off: reasoning text returned on {len(rep['thinking_off']['reasoning_returned'])} levels "
              f"{rep['thinking_off']['reasoning_returned']}; finish 'length' {rep['thinking_off']['finish_length']}",
              f"inputs as committed: {rep['inputs_as_committed']['equal']}; not {rep['inputs_as_committed']['not_equal']}",
              f"message path: generated message shown once on {rep['message_path']['shown_once_as_generated']}/"
              f"{rep['message_path']['rows']} rows; other {rep['message_path']['other']}",
              "", "by shortest-solution band (levels, A0F, S probe-priced, S free, A7F):"]
    lines += [f"  ARC-3 {k:<6} {x['levels']:>3} {x['A0F']:>4} {x['S']:>4} {x['S_free']:>4} {x['A7F']:>4}"
              for k, x in rep["by_band"].items()]
    lines.append("by game (levels, A0F, S probe-priced, S free, A7F):")
    lines += [f"  ARC-3 {k:<6} {x['levels']:>3} {x['A0F']:>4} {x['S']:>4} {x['S_free']:>4} {x['A7F']:>4}"
              for k, x in rep["by_game"].items()]
    mv = rep["measured_value"]
    lines += ["", f"measured value: level-index-weighted points, mean over {mv['games']} games {mv['mean_points_over_games']}"]
    lines += [f"  ARC-3 {k:<6} A0F {x['A0F_points']:>6} S {x['S_points']:>6} S free {x['S_free_points']:>6} A7F "
              f"{x['A7F_points']:>6} ({x['levels_in_run']} of {x['n_levels']} levels)" for k, x in mv["per_game"].items()]
    lines += ["", f"Rule: {rep['rule']}",
              f"Reading (probe-priced): M = {rd['M']} (S {rd['S']} - A0F {rd['A0F']} of N = {rd['N']}); A7SF only "
              f"{rd['S_only']}, A0F only {rd['A0F_only']}; p {_p(rd['p'])}; band {rd['band']}",
              f"Decision: {rep['decision']}",
              f"Entry draft: {rep['entry_draft']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    a = ap.parse_args(argv)
    rep = analyze(a.archive, C1A.load_levels(C1_LEVELS), load_probes(), load_level_map(C1_ARCHIVE / "level_map.tsv"),
                  load_level_map(C1P_ARCHIVE / "level_map.tsv"))
    print(format_report(rep))
    if a.archive.is_dir():
        (a.archive / "c1s_decision.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "level_map"},
                                                                indent=1, default=str) + "\n", encoding="utf-8")
        cols = list(rep["level_map"][0]) if rep["level_map"] else []
        (a.archive / "level_map.tsv").write_text("\t".join(cols) + "\n" + "".join(
            "\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n" for r in rep["level_map"]), encoding="utf-8")
        print(f"wrote {a.archive / 'c1s_decision.json'} and level_map.tsv")


if __name__ == "__main__":
    main()
