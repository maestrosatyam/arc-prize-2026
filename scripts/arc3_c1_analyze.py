#!/usr/bin/env python3
"""C1 analysis (run a3-c1-levelstarts): the frozen rule of reports/research/ARC3_C1_LEVEL_STARTS_RULE_2026-10-29.md
(2026-10-29 00:12 IST; written at d72c696, stamped in 2b1525d) on the three
kernels' r1_results.jsonl, arms A0F (unchanged) and A7F (told insight), each level played from a fresh level start,
160 actions / 30 min per case-arm. Levels: experiments/arc3-c1/levels.jsonl (sha256 checked).

Validity (any failing: no decision, report only):
  - the CPU twin passed for every kernel ("preflight: PASS" in the archived twin log);
  - at least 95% of level starts replay to their hash on the GPU run (both arms' rows replay_ok); failures are
    excluded from both arms and counted;
  - at least 95% of case-arms per arm, over the levels that replayed, end by a clear, the action budget or the wall budget;
  - the insight list and texts equal the committed list: the run's levels are the list's; A7F shows the level's insight
    once (kind insight, claim k1, length of the committed message, no oracle error) and A0F none (the twin checked the
    bytes by sha256 against the committed hash);
  - both arms of each level ran in the same kernel (the list's kernel).
Reading, G = A7F clears - A0F clears over the N levels in the run, exact two-sided McNemar on the discordant pairs
(exactly one line):
  generalises   G >= 10 and p <= 0.01
  confined      G <= 3
  report        otherwise
Recorded beside, no decision: clears per arm by shortest-solution band and by game; clears within 80 actions / 900 s;
tokens per action and actions per case-minute; the measured value (per level the competition score min(1.15,
(baseline / actions)^2) for a clear, 0 otherwise; per game the level-index-weighted gain in points, its mean over games,
against the projection p x 10 public-game points with p = G / N); the level map against people (level_map.tsv); the
overlap with the 35 stuck states and agreement with R1g's arms on them.

  envs/arc3/bin/python -I scripts/arc3_c1_analyze.py [--archive experiments/runs/kaggle-A3-C1-levelstarts-v1]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "experiments" / "runs"
ARCHIVE = RUNS / "kaggle-A3-C1-levelstarts-v1"
LEVELS = ROOT / "experiments" / "arc3-c1" / "levels.jsonl"
LEVELS_SHA256 = "1c8a0dff01f4a5ec07068f2ca5640ee60e34fa672734095911b498c51a21d6ab"
KERNELS = ("k1", "k2", "k3")
ARMS = ("A0F", "A7F")
R1G_RUNS = (("R1g v1", "kaggle-R1g-v1"), ("R1g v2", "kaggle-R1g-v2"))
RAN_TO_END = ("cleared", "action_budget", "wall_budget")
REPLAY_SHARE = ENDED_SHARE = 0.95
GENERALISES_G, GENERALISES_P, CONFINED_G = 10, 0.01, 3
FAST = (80, 900.0)
BANDS = ("1-10", "11-20", "21-40", "41-80", "81+")
SCORE_CAP = 1.15
RULE = ("C1, frozen (2026-10-29 00:12 IST; ARC3_C1_LEVEL_STARTS_RULE_2026-10-29.md at d72c696): validity "
        "(twin passed; >= 95% of level starts replay to their hash, failures excluded from both arms; >= 95% of case-arms "
        "per arm end by clear / action budget / wall budget; insight list and texts equal the committed list; both arms "
        "of each level in the same kernel); G = A7F - A0F clears over N levels, exact McNemar: G >= 10 and p <= 0.01 "
        "generalises; G <= 3 confined; otherwise report")


def mcnemar_p(b: int, c: int) -> float | None:
    """Exact two-sided McNemar: binomial test of b against b + c discordant pairs at 1/2."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_levels(path: Path = LEVELS) -> list[dict]:
    data = path.read_bytes()
    if path == LEVELS:
        assert hashlib.sha256(data).hexdigest() == LEVELS_SHA256, "levels.jsonl differs from the committed list"
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def load_rows(archive: Path) -> tuple[dict[tuple[str, str], dict], dict[str, bool | None], dict[str, list]]:
    """(case_id, arm) -> row with its kernel; per kernel the twin verdict (None when no twin log); duplicate rows."""
    rows, twins, dupes = {}, {}, collections.defaultdict(list)
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
            key = (str(r.get("case_id")), str(r.get("arm")))
            if key in rows:
                dupes[k].append(key)
            rows[key] = dict(r, _kernel=k)
    return rows, twins, dict(dupes)


def cleared(r: dict | None) -> bool:
    return bool(r) and bool(r.get("cleared"))


def cleared_within(r: dict | None, limit=FAST) -> bool:
    if not cleared(r) or r.get("clear_actions") is None or r.get("clear_wall_s") is None:
        return False
    return int(r["clear_actions"]) <= limit[0] and float(r["clear_wall_s"]) <= limit[1]


def level_score(r: dict | None, baseline: int) -> float:
    """The competition's level score: min(1.15, (baseline / actions)^2) for a clear, else 0."""
    if not cleared(r):
        return 0.0
    actions = int(r.get("clear_actions") or r.get("actions_used") or 0)
    return min(SCORE_CAP, (baseline / actions) ** 2) if actions > 0 else SCORE_CAP


def usage(rs: list[dict]) -> dict:
    acts = sum(int(r.get("actions_used") or 0) for r in rs)
    toks = sum(int(r.get("generated_tokens") or 0) for r in rs)
    apm = [int(r.get("actions_used") or 0) / (float(r["wall_s"]) / 60) for r in rs if float(r.get("wall_s") or 0) > 0]
    return {"tokens_per_action": round(toks / acts) if acts else None,
            "actions_per_case_minute_median": round(statistics.median(apm), 2) if apm else None}


def analyze(archive: Path, levels: list[dict], runs_root: Path = RUNS) -> dict:
    rows, twins, dupes = load_rows(archive)
    by_id = {l["case_id"]: l for l in levels}
    out: dict = {"rule": RULE, "archive": str(archive), "levels_listed": len(levels),
                 "kernels_present": sorted({r["_kernel"] for r in rows.values()}), "twins": twins,
                 "duplicate_rows": dupes, "extra_rows": sorted({c for c, _ in rows} - set(by_id))}
    missing_kernels = [k for k in KERNELS if not (archive / k / "r1_results.jsonl").is_file()]
    if missing_kernels:
        out["missing_kernels"] = missing_kernels

    # validity
    replay_bad, kernel_bad, missing = [], [], []
    for cid, lv in by_id.items():
        rr = [rows.get((cid, a)) for a in ARMS]
        if any(r is None for r in rr):
            missing.append(cid)
            continue
        if any(r.get("replay_ok") is not True or r.get("end_reason") == "replay_mismatch" for r in rr):
            replay_bad.append(cid)
        if any(r["_kernel"] != lv["kernel"] for r in rr):
            kernel_bad.append(cid)
    run = [c for c in by_id if c not in replay_bad and c not in missing]
    ended = {a: sum(1 for c in run if rows[(c, a)].get("end_reason") in RAN_TO_END) for a in ARMS}
    text_bad = {}
    for c in run:
        lv, r0, r7 = by_id[c], rows[(c, "A0F")], rows[(c, "A7F")]
        got7 = (r7.get("oracle_kind"), int(r7.get("oracle_shown") or 0), int(r7.get("oracle_errors") or 0),
                list(r7.get("oracle_claim_ids") or []), int(r7.get("oracle_chars") or 0))
        want7 = ("insight", 1, 0, [lv["k1"]], len(lv["insight"]["message"]))
        got0 = (r0.get("oracle_kind"), int(r0.get("oracle_shown") or 0))
        if got7 != want7 or got0 != (None, 0) or int(r0.get("fresh_notice_shown") or 0) != 1 \
                or int(r7.get("fresh_notice_shown") or 0) != 1:
            text_bad[c] = {"A7F": got7, "want": want7, "A0F": got0,
                           "fresh": (r0.get("fresh_notice_shown"), r7.get("fresh_notice_shown"))}
    n_list = len(levels)
    checks = {
        "twin_passed": all(twins.get(k) is True for k in KERNELS),
        "replay_95": (n_list - len(replay_bad) - len(missing)) >= math.ceil(REPLAY_SHARE * n_list),
        "ended_95_per_arm": bool(run) and all(ended[a] >= math.ceil(ENDED_SHARE * len(run)) for a in ARMS),
        "texts_equal_committed": not text_bad and not out["extra_rows"] and not missing,
        "same_kernel": not kernel_bad and not dupes,
    }
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "replay_failures": sorted(replay_bad),
                       "missing": sorted(missing), "ended": ended, "ended_needed": math.ceil(ENDED_SHARE * len(run)),
                       "replay_needed": math.ceil(REPLAY_SHARE * n_list), "text_mismatch": text_bad,
                       "kernel_mismatch": sorted(kernel_bad)}

    # reading
    n = len(run)
    c0 = sum(cleared(rows[(c, "A0F")]) for c in run)
    c7 = sum(cleared(rows[(c, "A7F")]) for c in run)
    b = sum(cleared(rows[(c, "A7F")]) and not cleared(rows[(c, "A0F")]) for c in run)
    d = sum(cleared(rows[(c, "A0F")]) and not cleared(rows[(c, "A7F")]) for c in run)
    g, p = c7 - c0, mcnemar_p(b, d)
    if g >= GENERALISES_G and p is not None and p <= GENERALISES_P:
        reading = "generalises"
    elif g <= CONFINED_G:
        reading = "confined"
    else:
        reading = "report"
    complete = not missing_kernels and not missing
    out["reading"] = {"N": n, "A0F": c0, "A7F": c7, "G": g, "A7F_only": b, "A0F_only": d, "p": p, "band": reading}
    if not complete:
        out["decision"] = f"pending: kernels {missing_kernels or 'all present'}, {len(missing)} level(s) without both rows"
    else:
        out["decision"] = reading if out["validity"]["valid"] else "no decision (validity failed): report only"

    # recorded beside
    def per(group_key):
        res = collections.defaultdict(lambda: {"levels": 0, "A0F": 0, "A7F": 0})
        for c in run:
            k = group_key(by_id[c])
            res[k]["levels"] += 1
            for a in ARMS:
                res[k][a] += int(cleared(rows[(c, a)]))
        return dict(res)

    out["by_band"] = {k: v for k, v in sorted(per(lambda l: l["band"]).items(), key=lambda kv: BANDS.index(kv[0]))}
    out["by_game"] = dict(sorted(per(lambda l: l["game"]).items()))
    out["within_80_900"] = {a: sum(cleared_within(rows[(c, a)]) for c in run) for a in ARMS}
    out["usage"] = {a: usage([rows[(c, a)] for c in run]) for a in ARMS}
    out["end_reasons"] = {a: dict(collections.Counter(str(rows[(c, a)].get("end_reason")) for c in run)) for a in ARMS}

    games = collections.defaultdict(list)
    for c in run:
        games[by_id[c]["game"]].append(c)
    value = {}
    for game, cs in sorted(games.items()):
        n_levels = by_id[cs[0]]["n_levels"]
        weight = n_levels * (n_levels + 1) / 2
        pts = {a: 100 * sum(by_id[c]["level"] * level_score(rows[(c, a)], by_id[c]["baseline_actions"]) for c in cs) / weight
               for a in ARMS}
        value[game] = {"levels_in_run": len(cs), "n_levels": n_levels, "A0F_points": round(pts["A0F"], 2),
                       "A7F_points": round(pts["A7F"], 2), "gain_points": round(pts["A7F"] - pts["A0F"], 2)}
    gains = [v["gain_points"] for v in value.values()]
    pr = g / n if n else None
    out["measured_value"] = {"per_game": value, "mean_gain_points_over_games": round(statistics.mean(gains), 2) if gains else None,
                             "games": len(gains), "p_converted": round(pr, 3) if pr is not None else None,
                             "projection_p_x_10_points": round(10 * pr, 2) if pr is not None else None}

    level_map = []
    for c in sorted(run, key=lambda c: (by_id[c]["game"], by_id[c]["level"])):
        lv = by_id[c]
        row = {"case_id": c, "game": lv["game"], "level": lv["level"], "band": lv["band"],
               "baseline_actions": lv["baseline_actions"], "solution_actions": lv["solution_actions"]}
        for a in ARMS:
            r = rows[(c, a)]
            row[f"{a}_cleared"] = int(cleared(r))
            row[f"{a}_clear_actions"] = r.get("clear_actions") if cleared(r) else None
            row[f"{a}_score"] = round(level_score(r, lv["baseline_actions"]), 3)
        level_map.append(row)
    out["level_map"] = level_map

    # overlap with the 35 stuck states: R1g's arms on the same levels
    overlap = {}
    for label, run_name in R1G_RUNS:
        f = runs_root / run_name / "r1_results.jsonl"
        if not f.is_file():
            continue
        agree = {"A7L~A7F": [0, 0], "A0Lb~A0F": [0, 0]}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            parts = str(r["case_id"]).split("_")
            lid = f"{parts[0]}_L{int(r['target_level'])}"
            if lid not in run:
                continue
            pair = {"A7L": ("A7L~A7F", "A7F"), "A0Lb": ("A0Lb~A0F", "A0F")}.get(r["arm"])
            if pair:
                agree[pair[0]][1] += 1
                agree[pair[0]][0] += int(cleared(r) == cleared(rows[(lid, pair[1])]))
        overlap[label] = {k: f"{v[0]}/{v[1]} agree" for k, v in agree.items()}
    out["overlap_r1"] = {"levels_shared": sorted({c for c in run if c in _r1_levels(runs_root)}), "agreement": overlap}
    return out


def _r1_levels(runs_root: Path) -> set[str]:
    f = runs_root / R1G_RUNS[0][1] / "r1_results.jsonl"
    if not f.is_file():
        return set()
    return {f"{str(r['case_id']).split('_')[0]}_L{int(r['target_level'])}"
            for r in map(json.loads, filter(str.strip, f.read_text(encoding="utf-8").splitlines()))}


def format_report(rep: dict) -> str:
    v, rd = rep["validity"], rep["reading"]
    lines = [f"C1 (a3-c1-levelstarts): A0F unchanged vs A7F told insight from fresh level starts, 160 actions / 1800 s; "
             f"{rep['levels_listed']} levels listed; kernels present {rep['kernels_present']}; twins {rep['twins']}"]
    if rep.get("missing_kernels"):
        lines.append(f"kernels without results yet: {rep['missing_kernels']}")
    lines += [f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}",
              f"  replay failures {len(v['replay_failures'])} (need <= {rep['levels_listed'] - v['replay_needed']}) "
              f"{v['replay_failures']}; missing {len(v['missing'])}; ended {v['ended']} (need {v['ended_needed']} each); "
              f"text mismatches {len(v['text_mismatch'])}; kernel mismatches {v['kernel_mismatch']}",
              "",
              f"{'track':<7}{'arm':<6}{'clears':>8}{'within 80/900':>15}{'tokens/action':>15}{'actions/case-min':>18}"]
    for a in ARMS:
        u = rep["usage"][a]
        lines.append(f"{'ARC-3':<7}{a:<6}{rd[a]:>5}/{rd['N']:<3}{rep['within_80_900'][a]:>14}{str(u['tokens_per_action']):>15}"
                     f"{str(u['actions_per_case_minute_median']):>18}")
    lines += [f"end reasons: {rep['end_reasons']}", "", "by shortest-solution band (levels, A0F, A7F):"]
    lines += [f"  ARC-3 {k:<6} {x['levels']:>3} {x['A0F']:>4} {x['A7F']:>4}" for k, x in rep["by_band"].items()]
    lines.append("by game (levels, A0F, A7F):")
    lines += [f"  ARC-3 {k:<6} {x['levels']:>3} {x['A0F']:>4} {x['A7F']:>4}" for k, x in rep["by_game"].items()]
    mv = rep["measured_value"]
    lines += ["", f"measured value (recorded, no decision): mean level-index-weighted gain {mv['mean_gain_points_over_games']} "
                  f"points over {mv['games']} games; projection p x 10 = {mv['projection_p_x_10_points']} (p = G / N = "
                  f"{mv['p_converted']})"]
    lines += [f"  ARC-3 {k:<6} A0F {x['A0F_points']:>6} A7F {x['A7F_points']:>6} gain {x['gain_points']:>6} "
              f"({x['levels_in_run']} of {x['n_levels']} levels)" for k, x in mv["per_game"].items()]
    ov = rep["overlap_r1"]
    pv = "n/a" if rd["p"] is None else f"{rd['p']:.4f}"
    lines += [f"overlap with the 35 stuck states: {len(ov['levels_shared'])} levels; agreement {ov['agreement']}",
              "", f"Rule: {rep['rule']}",
              f"Reading: G = {rd['G']} ({rd['A7F']} - {rd['A0F']} of N = {rd['N']}); A7F only {rd['A7F_only']}, A0F only "
              f"{rd['A0F_only']}; p {pv}; band {rd['band']}",
              f"Decision: {rep['decision']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    a = ap.parse_args(argv)
    rep = analyze(a.archive, load_levels(), a.runs_root)
    print(format_report(rep))
    if a.archive.is_dir():
        (a.archive / "c1_decision.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "level_map"},
                                                               indent=1, default=str) + "\n", encoding="utf-8")
        cols = list(rep["level_map"][0]) if rep["level_map"] else []
        (a.archive / "level_map.tsv").write_text("\t".join(cols) + "\n" + "".join(
            "\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n" for r in rep["level_map"]), encoding="utf-8")
        print(f"wrote {a.archive / 'c1_decision.json'} and level_map.tsv")


if __name__ == "__main__":
    main()
