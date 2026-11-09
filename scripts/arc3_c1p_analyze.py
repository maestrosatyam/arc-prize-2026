#!/usr/bin/env python3
"""C1P analysis (run a3-c1p-placebo): the frozen rule of reports/research/ARC3_C1P_PLACEBO_RULE_2026-10-29.md (frozen at
f4c1658; revision 1 at 1b8666f) on the three kernels' r1_results.jsonl, arm A7cF (C1's fresh start with R1f's
length-matched control text), against C1's archived arms A0F and A7F on the same levels (C1's level_map.tsv).
Levels: experiments/arc3-c1p/texts.jsonl (146: C1's 157 less the 11 with no control; sha256 checked).

Validity (any failing: no decision, report only):
  - the CPU twin passed for every kernel ("preflight: PASS" in the archived twin log);
  - at least 95% of level starts replay to their hash on the GPU run (replay_ok); failures are excluded here and from
    C1's arms;
  - at least 95% of case-arms, over the levels that replayed, end by a clear, the action budget or the wall budget;
  - the placebo texts equal the committed list: the texts reproduce from the pipeline with the committed hash and the
    35 R1 cases match R1f's A7c byte for byte (scripts/arc3_c1p_texts.py); the run's levels are the list's; every row
    shows the level's control once (kind control, its claim ids, the committed length, no oracle error) after the
    fresh-start line (the twin checked the bytes by sha256);
  - every notebook cell other than the arm's definition, its messages and the level list is byte-identical to C1's
    kernel for the same level list (kaggle/arc3/c1p/build_c1p_notebook.assert_against_c1 on the committed notebooks);
    each level ran in its C1 kernel.
Reading, P = A7cF clears over the N levels that replayed, C = A7F - P with C1's A7F on the same levels, exact two-sided
McNemar on the discordant pairs (exactly one line):
  content supported       C >= 10 and p <= 0.01
  content not supported   C <= 3
  report                  otherwise
Recorded beside, no decision: P - A0F with its McNemar p and the share of C1's gain recovered, (P - A0F) / 26; per arm
(A0F, A7cF, A7F) wall-budget endings, tokens per action, actions per case-minute and clears within 80 actions / 900 s
(A0F's and A7F's from C1's r1_results.jsonl when present); clears by shortest-solution band and by game; the
level-index-weighted score per game (C1's measured value) for the three arms; on the levels shared with the R1 stuck
states, A7cF against T1's A7cL case by case; the fallback count, C on the clean levels and the shared words per fallback
level.

  envs/arc3/bin/python -I scripts/arc3_c1p_analyze.py [--archive experiments/runs/kaggle-A3-C1P-placebo-v1]
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
ARCHIVE = RUNS / "kaggle-A3-C1P-placebo-v1"
C1_ARCHIVE = RUNS / "kaggle-A3-C1-levelstarts-v1"
T1_RUN = "kaggle-A3-T1-placebo30-v1"
TEXTS = ROOT / "experiments" / "arc3-c1p" / "texts.jsonl"
TEXTS_JSONL_SHA256 = "f64041e751db038e0b0a183f02c0a74081fed305075f6f583234ee53d0a3a6b0"
TEXTS_SHA256 = "bdceef19c7d1c121e67be34875ddba617123b02ba3b5c647be7fc2033b68eeb2"
C1_LEVELS = ROOT / "experiments" / "arc3-c1" / "levels.jsonl"
KERNELS = ("k1", "k2", "k3")
ARM = "A7cF"
ARMS3 = ("A0F", ARM, "A7F")
C1_GAIN = 26
SUPPORTED_C, SUPPORTED_P, NOT_SUPPORTED_C = 10, 0.01, 3
RULE = ("C1P, frozen (2026-10-29 12:56 IST; ARC3_C1P_PLACEBO_RULE_2026-10-29.md at f4c1658, "
        "revision 1 at 1b8666f): validity (twin passed; >= 95% of level starts replay to their hash, failures excluded "
        "here and from C1's arms; >= 95% of case-arms end by clear / action budget / wall budget; placebo texts equal the "
        "committed hash and the 35 R1 cases match R1f's A7c; cells byte-identical to C1's except the arm, its messages and "
        "the level list); C = A7F - P over N levels (C1's A7F), exact McNemar: C >= 10 and p <= 0.01 content supported; "
        "C <= 3 content not supported; otherwise report")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


C1A = _load("arc3_c1_analyze", HERE / "arc3_c1_analyze.py")
mcnemar_p, cleared, cleared_within, level_score, usage = C1A.mcnemar_p, C1A.cleared, C1A.cleared_within, \
    C1A.level_score, C1A.usage


def load_texts(path: Path = TEXTS) -> list[dict]:
    data = path.read_bytes()
    if path == TEXTS:
        assert hashlib.sha256(data).hexdigest() == TEXTS_JSONL_SHA256, "texts.jsonl differs from the committed texts"
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def texts_hash(texts: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[r["case_id"], r["control"]["message"]] for r in texts],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def load_level_map(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split("\t")
    return {row["case_id"]: row for row in (dict(zip(cols, l.split("\t"))) for l in lines[1:] if l.strip())}


def load_rows(archive: Path) -> tuple[dict[str, dict], dict[str, bool | None], dict[str, list], list[str]]:
    """case_id -> the A7cF row with its kernel; per kernel the twin verdict; duplicate rows; rows of other arms."""
    rows, twins, dupes, other = {}, {}, collections.defaultdict(list), []
    for k in KERNELS:
        d = archive / k
        logs = sorted((d / "preflight").glob("*.log")) if (d / "preflight").is_dir() else []
        twins[k] = any("preflight: PASS" in p.read_text(encoding="utf-8", errors="replace") for p in logs) if logs else None
        f = d / "r1_results.jsonl"
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("arm") != ARM:
                other.append(f"{r.get('case_id')} {r.get('arm')}")
                continue
            cid = str(r.get("case_id"))
            if cid in rows:
                dupes[k].append(cid)
            rows[cid] = dict(r, _kernel=k)
    return rows, twins, dict(dupes), other


def c1_rows(c1_archive: Path) -> dict[tuple[str, str], dict]:
    """C1's own rows, (case_id, arm) -> row, when its r1_results.jsonl files are present (for the speed lines)."""
    out = {}
    for k in KERNELS:
        f = c1_archive / k / "r1_results.jsonl"
        if f.is_file():
            for line in filter(str.strip, f.read_text(encoding="utf-8").splitlines()):
                r = json.loads(line)
                out[(str(r["case_id"]), str(r["arm"]))] = r
    return out


def texts_reproduce() -> dict:
    """The texts pipeline rerun (scripts/arc3_c1p_texts.py): the committed hash and the 35-case R1f reproduction."""
    try:
        X = _load("arc3_c1p_texts", HERE / "arc3_c1p_texts.py")
        res = X.build()
    except (SystemExit, AssertionError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)[:300]}
    rep = res["repro"]
    ok = X.texts_sha256(res["kept"]) == TEXTS_SHA256 and len(rep) == 35 \
        and all(r["message_bytes_equal"] and r["entry_equal"] for r in rep)
    return {"ok": ok, "texts_sha256": X.texts_sha256(res["kept"]),
            "r1f_message_bytes_equal": sum(r["message_bytes_equal"] for r in rep), "r1f_cases": len(rep)}


def cells_identical(root: Path = ROOT) -> dict:
    """build_c1p_notebook.assert_against_c1 and assert_twin on the committed C1P notebooks."""
    out = {}
    try:
        P = _load("build_c1p_notebook", root / "kaggle" / "arc3" / "c1p" / "build_c1p_notebook.py")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    for k in KERNELS:
        d = root / "kaggle" / "arc3" / "c1p" / k
        try:
            nb = json.loads((d / f"{P.slug(k)}.ipynb").read_text(encoding="utf-8"))
            pre = json.loads((d / "preflight" / f"{P.slug(k)}-preflight.ipynb").read_text(encoding="utf-8"))
            P.assert_against_c1(nb, k)
            P.assert_twin(pre, nb, k)
            out[k] = True
        except Exception as exc:  # noqa: BLE001
            out[k] = f"{type(exc).__name__}: {exc}"[:200]
    return {"ok": all(v is True for v in out.values()), "kernels": out}


def analyze(archive: Path, texts: list[dict], level_map: dict[str, dict], c1_levels: list[dict],
            runs_root: Path = RUNS, c1_archive: Path = C1_ARCHIVE, static: dict | None = None) -> dict:
    rows, twins, dupes, other = load_rows(archive)
    by_id = {t["case_id"]: t for t in texts}
    lv = {l["case_id"]: l for l in c1_levels}
    out: dict = {"rule": RULE, "archive": str(archive), "levels_listed": len(texts),
                 "kernels_present": sorted({r["_kernel"] for r in rows.values()}), "twins": twins,
                 "duplicate_rows": dupes, "other_arm_rows": other, "extra_rows": sorted(set(rows) - set(by_id))}
    missing_kernels = [k for k in KERNELS if not (archive / k / "r1_results.jsonl").is_file()]
    if missing_kernels:
        out["missing_kernels"] = missing_kernels
    static = static if static is not None else {"texts": texts_reproduce(), "cells": cells_identical()}
    out["static"] = static

    # validity
    replay_bad, kernel_bad, missing, text_bad = [], [], [], {}
    for cid, t in by_id.items():
        r = rows.get(cid)
        if r is None:
            missing.append(cid)
            continue
        if r.get("replay_ok") is not True or r.get("end_reason") == "replay_mismatch":
            replay_bad.append(cid)
        if r["_kernel"] != t["kernel"]:
            kernel_bad.append(cid)
    run = [c for c in by_id if c not in replay_bad and c not in missing]
    ended = sum(1 for c in run if rows[c].get("end_reason") in C1A.RAN_TO_END)
    for c in run:
        r, ctl = rows[c], by_id[c]["control"]
        got = (r.get("oracle_kind"), int(r.get("oracle_shown") or 0), int(r.get("oracle_errors") or 0),
               list(r.get("oracle_claim_ids") or []), int(r.get("oracle_chars") or 0), int(r.get("fresh_notice_shown") or 0))
        want = ("control", 1, 0, list(ctl["claim_ids"]), len(ctl["message"]), 1)
        if got != want:
            text_bad[c] = {"got": got, "want": want}
    not_in_map = sorted(c for c in run if c not in level_map)
    n_list = len(texts)
    checks = {
        "twin_passed": all(twins.get(k) is True for k in KERNELS),
        "replay_95": (n_list - len(replay_bad) - len(missing)) >= math.ceil(C1A.REPLAY_SHARE * n_list),
        "ended_95": bool(run) and ended >= math.ceil(C1A.ENDED_SHARE * len(run)),
        "texts_equal_committed": not text_bad and not out["extra_rows"] and not missing and not other
        and texts_hash(texts) == TEXTS_SHA256 and bool(static["texts"].get("ok")) and not not_in_map,
        "cells_identical_to_c1": bool(static["cells"].get("ok")) and not kernel_bad and not dupes,
    }
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "replay_failures": sorted(replay_bad),
                       "missing": sorted(missing), "ended": ended, "ended_needed": math.ceil(C1A.ENDED_SHARE * len(run)),
                       "replay_needed": math.ceil(C1A.REPLAY_SHARE * n_list), "text_mismatch": text_bad,
                       "kernel_mismatch": sorted(kernel_bad), "not_in_c1_level_map": not_in_map}
    run = [c for c in run if c in level_map]

    def c1c(c: str, arm: str) -> bool:
        return level_map[c][f"{arm}_cleared"] == "1"

    def clears(c: str, arm: str) -> bool:
        return cleared(rows[c]) if arm == ARM else c1c(c, arm)

    def paired(cs: list[str], x: str, y: str) -> dict:
        b = sum(clears(c, x) and not clears(c, y) for c in cs)
        d = sum(clears(c, y) and not clears(c, x) for c in cs)
        return {"N": len(cs), x: sum(clears(c, x) for c in cs), y: sum(clears(c, y) for c in cs),
                "diff": sum(clears(c, x) for c in cs) - sum(clears(c, y) for c in cs),
                f"{x}_only": b, f"{y}_only": d, "p": mcnemar_p(b, d)}

    # reading: C = A7F - P
    cr = paired(run, "A7F", ARM)
    c_val, p = cr["diff"], cr["p"]
    if c_val >= SUPPORTED_C and p is not None and p <= SUPPORTED_P:
        band = "content supported"
    elif c_val <= NOT_SUPPORTED_C:
        band = "content not supported"
    else:
        band = "report"
    out["reading"] = {"N": cr["N"], "P": cr[ARM], "A7F": cr["A7F"], "A0F": sum(c1c(c, "A0F") for c in run), "C": c_val,
                      "A7F_only": cr["A7F_only"], "A7cF_only": cr[f"{ARM}_only"], "p": p, "band": band}
    complete = not missing_kernels and not missing
    if not complete:
        out["decision"] = f"pending: kernels {missing_kernels or 'all present'}, {len(missing)} level(s) without a row"
    else:
        out["decision"] = band if out["validity"]["valid"] else "no decision (validity failed): report only"

    # recorded beside
    me = paired(run, ARM, "A0F")
    out["message_effect"] = {"P_minus_A0F": me["diff"], "A7cF_only": me[f"{ARM}_only"], "A0F_only": me["A0F_only"],
                             "p": me["p"], "share_of_c1_gain": round(me["diff"] / C1_GAIN, 3)}
    out["c1_on_these_levels"] = paired(run, "A7F", "A0F")
    crow = c1_rows(c1_archive)

    def row_of(c: str, arm: str) -> dict | None:
        return rows[c] if arm == ARM else crow.get((c, arm))

    speed = {}
    for a in ARMS3:
        rs = [row_of(c, a) for c in run]
        if any(r is None for r in rs):
            speed[a] = None
            continue
        u = usage(rs)
        speed[a] = {"wall_budget_endings": sum(r.get("end_reason") == "wall_budget" for r in rs),
                    "tokens_per_action": u["tokens_per_action"],
                    "actions_per_case_minute": u["actions_per_case_minute_median"],
                    "within_80_900": sum(cleared_within(r) for r in rs)}
    out["speed"] = speed

    def per(key):
        res = collections.defaultdict(lambda: {"levels": 0, **{a: 0 for a in ARMS3}})
        for c in run:
            g = key(c)
            res[g]["levels"] += 1
            for a in ARMS3:
                res[g][a] += int(clears(c, a))
        return dict(res)

    out["by_band"] = {k: v for k, v in sorted(per(lambda c: lv[c]["band"]).items(), key=lambda kv: C1A.BANDS.index(kv[0]))}
    out["by_game"] = dict(sorted(per(lambda c: by_id[c]["game"]).items()))

    games = collections.defaultdict(list)
    for c in run:
        games[by_id[c]["game"]].append(c)
    value = {}
    for game, cs in sorted(games.items()):
        n_levels = lv[cs[0]]["n_levels"]
        weight = n_levels * (n_levels + 1) / 2
        pts = {a: 100 * sum(lv[c]["level"] * (level_score(rows[c], lv[c]["baseline_actions"]) if a == ARM
                                              else float(level_map[c][f"{a}_score"])) for c in cs) / weight for a in ARMS3}
        value[game] = {"levels_in_run": len(cs), "n_levels": n_levels, **{f"{a}_points": round(pts[a], 2) for a in ARMS3}}
    out["measured_value"] = {"per_game": value, "games": len(value),
                             "mean_points_over_games": {a: round(statistics.mean(v[f"{a}_points"] for v in value.values()), 2)
                                                        if value else None for a in ARMS3}}

    # the levels shared with the R1 stuck states: A7cF against T1's A7cL, case by case
    t1 = runs_root / T1_RUN / "r1_results.jsonl"
    if t1.is_file():
        pairs = []
        for r in map(json.loads, filter(str.strip, t1.read_text(encoding="utf-8").splitlines())):
            lid = f"{str(r['case_id']).split('_')[0]}_L{int(r['target_level'])}"
            if r.get("arm") == "A7cL" and lid in run:
                pairs.append((r["case_id"], lid, cleared(r), cleared(rows[lid])))
        out["t1_overlap"] = {"levels": len({p[1] for p in pairs}), "cases": len(pairs),
                             "A7cL_cleared": sum(p[2] for p in pairs), "A7cF_cleared": sum(p[3] for p in pairs),
                             "agree": sum(p[2] == p[3] for p in pairs),
                             "A7cF_only": sum(p[3] and not p[2] for p in pairs),
                             "A7cL_only": sum(p[2] and not p[3] for p in pairs)}
    else:
        out["t1_overlap"] = None

    fb = [c for c in run if by_id[c]["fallback"]]
    clean = [c for c in run if not by_id[c]["fallback"]]
    out["fallback"] = {"levels": sum(t["fallback"] for t in texts), "in_run": len(fb),
                       "shared_words": {c: by_id[c]["control"]["shared_words"] for c in fb},
                       "P_on_fallback": sum(clears(c, ARM) for c in fb),
                       "A7F_on_fallback": sum(clears(c, "A7F") for c in fb),
                       "A0F_on_fallback": sum(clears(c, "A0F") for c in fb)}
    cc = paired(clean, "A7F", ARM)
    out["clean"] = {"N": cc["N"], "P": cc[ARM], "A7F": cc["A7F"], "A0F": sum(c1c(c, "A0F") for c in clean), "C": cc["diff"],
                    "A7F_only": cc["A7F_only"], "A7cF_only": cc[f"{ARM}_only"], "p": cc["p"]}

    level_rows = []
    for c in sorted(run, key=lambda c: (by_id[c]["game"], by_id[c]["level"])):
        r = rows[c]
        level_rows.append({"case_id": c, "game": by_id[c]["game"], "level": by_id[c]["level"], "band": lv[c]["band"],
                           "fallback": int(by_id[c]["fallback"]), "A0F_cleared": int(c1c(c, "A0F")),
                           "A7cF_cleared": int(cleared(r)), "A7F_cleared": int(c1c(c, "A7F")),
                           "A7cF_clear_actions": r.get("clear_actions") if cleared(r) else None,
                           "A7cF_score": round(level_score(r, lv[c]["baseline_actions"]), 3),
                           "A7cF_end": r.get("end_reason")})
    out["level_map"] = level_rows
    return out


def _p(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def format_report(rep: dict) -> str:
    v, rd = rep["validity"], rep["reading"]
    lines = [f"C1P (a3-c1p-placebo): A7cF (C1's fresh start with R1f's control text) against C1's A0F and A7F on the same "
             f"levels, 160 actions / 1800 s; {rep['levels_listed']} levels listed; kernels present {rep['kernels_present']}; "
             f"twins {rep['twins']}"]
    if rep.get("missing_kernels"):
        lines.append(f"kernels without results yet: {rep['missing_kernels']}")
    st = rep["static"]
    lines += [f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
              f"  replay failures {len(v['replay_failures'])} (need <= {rep['levels_listed'] - v['replay_needed']}) "
              f"{v['replay_failures']}; missing {len(v['missing'])}; ended {v['ended']} (need {v['ended_needed']}); "
              f"text mismatches {len(v['text_mismatch'])}; kernel mismatches {v['kernel_mismatch']}",
              f"  texts rerun {st['texts']}; cells vs C1 {st['cells']}",
              ""]
    sp = rep["speed"]
    lines.append(f"{'track':<7}{'arm':<6}{'clears':>9}{'wall-budget':>13}{'within 80/900':>15}{'tokens/action':>15}"
                 f"{'actions/case-min':>18}")
    counts = {"A0F": rd["A0F"], "A7cF": rd["P"], "A7F": rd["A7F"]}
    for a in ARMS3:
        s = sp.get(a) or {}
        lines.append(f"{'ARC-3':<7}{a:<6}{counts[a]:>5}/{rd['N']:<3}{str(s.get('wall_budget_endings')):>13}"
                     f"{str(s.get('within_80_900')):>15}{str(s.get('tokens_per_action')):>15}"
                     f"{str(s.get('actions_per_case_minute')):>18}")
    me, c1 = rep["message_effect"], rep["c1_on_these_levels"]
    lines += ["",
              f"message effect P - A0F = {me['P_minus_A0F']} (A7cF only {me['A7cF_only']}, A0F only {me['A0F_only']}, "
              f"p {_p(me['p'])}); share of C1's gain recovered (P - A0F) / {C1_GAIN} = {me['share_of_c1_gain']}",
              f"C1 on these levels: A7F {c1['A7F']} - A0F {c1['A0F']} = {c1['diff']} (A7F only {c1['A7F_only']}, A0F only "
              f"{c1['A0F_only']}, p {_p(c1['p'])})",
              "", "by shortest-solution band (levels, A0F, A7cF, A7F):"]
    lines += [f"  ARC-3 {k:<6} {x['levels']:>3} {x['A0F']:>4} {x['A7cF']:>4} {x['A7F']:>4}" for k, x in rep["by_band"].items()]
    lines.append("by game (levels, A0F, A7cF, A7F):")
    lines += [f"  ARC-3 {k:<6} {x['levels']:>3} {x['A0F']:>4} {x['A7cF']:>4} {x['A7F']:>4}" for k, x in rep["by_game"].items()]
    mv = rep["measured_value"]
    lines += ["", f"measured value (recorded, no decision): level-index-weighted points, mean over {mv['games']} games "
                  f"{mv['mean_points_over_games']}"]
    lines += [f"  ARC-3 {k:<6} A0F {x['A0F_points']:>6} A7cF {x['A7cF_points']:>6} A7F {x['A7F_points']:>6} "
              f"({x['levels_in_run']} of {x['n_levels']} levels)" for k, x in mv["per_game"].items()]
    t1 = rep["t1_overlap"]
    lines.append("" if t1 else "T1 overlap: T1's r1_results.jsonl not present")
    if t1:
        lines.append(f"R1 stuck-state levels: {t1['levels']} levels, {t1['cases']} cases; T1 A7cL {t1['A7cL_cleared']}, "
                     f"A7cF {t1['A7cF_cleared']}; agree {t1['agree']}/{t1['cases']}; A7cF only {t1['A7cF_only']}, A7cL only "
                     f"{t1['A7cL_only']}")
    fb, cl = rep["fallback"], rep["clean"]
    lines += [f"fallback (fewest shared words): {fb['levels']} levels, {fb['in_run']} in the run; on them A0F "
              f"{fb['A0F_on_fallback']}, A7cF {fb['P_on_fallback']}, A7F {fb['A7F_on_fallback']}",
              f"  shared words: {fb['shared_words']}",
              f"clean levels: C = A7F {cl['A7F']} - P {cl['P']} = {cl['C']} of N = {cl['N']} (A7F only {cl['A7F_only']}, "
              f"A7cF only {cl['A7cF_only']}, p {_p(cl['p'])}); A0F {cl['A0F']}",
              "", f"Rule: {rep['rule']}",
              f"Reading: C = {rd['C']} ({rd['A7F']} - {rd['P']} of N = {rd['N']}); A7F only {rd['A7F_only']}, A7cF only "
              f"{rd['A7cF_only']}; p {_p(rd['p'])}; band {rd['band']}",
              f"Decision: {rep['decision']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    a = ap.parse_args(argv)
    rep = analyze(a.archive, load_texts(), load_level_map(C1_ARCHIVE / "level_map.tsv"), C1A.load_levels(C1_LEVELS),
                  a.runs_root)
    print(format_report(rep))
    if a.archive.is_dir():
        (a.archive / "c1p_decision.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "level_map"},
                                                                indent=1, default=str) + "\n", encoding="utf-8")
        cols = list(rep["level_map"][0]) if rep["level_map"] else []
        (a.archive / "level_map.tsv").write_text("\t".join(cols) + "\n" + "".join(
            "\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n" for r in rep["level_map"]), encoding="utf-8")
        print(f"wrote {a.archive / 'c1p_decision.json'} and level_map.tsv")


if __name__ == "__main__":
    main()
