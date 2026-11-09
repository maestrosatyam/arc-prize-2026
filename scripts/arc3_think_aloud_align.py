"""ARC-AGI-3 think-aloud alignment (rule: experiments/arc3-think-aloud/RULE.md; descriptive, CPU only).

Steps (rule sections in brackets):
  identity  [1]  level counts per game against C1's n_levels; one human trajectory per game replayed offline on the
                 competition files (needs the environment with arc_agi; run that step with --env-dir).
  measures  [3-5] per trajectory x level recognition of K(L) (the adjudicated key map) and N(L); level summaries for
                 humans and agents; comparisons with the team onsets, the R1/T1 arms on the 25 stuck levels and the
                 C1, C1P and C1S cells; two-stage bootstrap and level permutations.

The dataset is read by path with `python -I`; nothing is written into it. Outputs hold derived numbers only. Agent
configurations are written as G1-G10 with their harness type.

Usage:
  python -I scripts/arc3_think_aloud_align.py identity --data <dir> --env-dir ~/data/arc3/environment_files
  python -I scripts/arc3_think_aloud_align.py measures --data <dir> --team <team tables dir>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/arc3-think-aloud"
C1_LEVELS = ROOT / "experiments/arc3-c1/levels.jsonl"
CASES = ROOT / "experiments/arc3-recovery/cases"
RUNS = ROOT / "experiments/runs"
RULE_LABELS = ROOT / "experiments/arc3-failure-annotation/labels_rule_A.csv"
SEED = 20261029
DRAWS = 10_000

# R1/T1 arms on the 35 cases (rule section 4)
ARMS = {
    "insight30": [("kaggle-R1g-v1", "A7L"), ("kaggle-R1g-v2", "A7L")],
    "unchanged30": [("kaggle-R1d-v1", "A0L"), ("kaggle-R1g-v1", "A0Lb"), ("kaggle-R1g-v2", "A0Lb"),
                    ("kaggle-R1j-v1", "A0Lc")],
    "r1v_insight30": [("kaggle-R1v-tcinsight-v1", "A16")],
    "t1_placebo30": [("kaggle-A3-T1-placebo30-v1", "A7cL")],
    "r1f_insight15": [("kaggle-R1f-v1", "A7")],
    "r1f_placebo15": [("kaggle-R1f-v1", "A7c")],
    "r1f_unchanged15": [("kaggle-R1f-v1", "A0d")],
}
C1_MAPS = {
    "c1": (RUNS / "kaggle-A3-C1-levelstarts-v1/level_map.tsv", {"A0F": "A0F_cleared", "A7F": "A7F_cleared"}),
    "c1p": (RUNS / "kaggle-A3-C1P-placebo-v1/level_map.tsv", {"A7cF": "A7cF_cleared"}),
    "c1s": (RUNS / "kaggle-A3-C1S-selfinsight-v1/level_map.tsv", {"A7SF": "A7SF_cleared_priced"}),
}


# ---------------------------------------------------------------- loading

def jload(p: Path):
    return json.loads(p.read_text())


def level_counts(levels: list[dict], total: int | None) -> list[int] | None:
    """Per-level action counts, read as per-level or cumulative (rule section 1); None if neither fits."""
    n = [int(x["actions"]) for x in sorted(levels, key=lambda x: int(x["level"]))]
    if not n:
        return None
    if total is None or sum(n) == total:
        return n
    if all(b >= a for a, b in zip(n, n[1:])) and n[-1] == total:
        return [n[0]] + [b - a for a, b in zip(n, n[1:])]
    return None


def trajectories(data: Path) -> list[dict]:
    """One record per trajectory: kind, game, entity, per-level counts and clears, annotation mechanisms."""
    ix = jload(data / "index.json")
    configs = sorted({str(Path(r["path"]).relative_to(Path("games") / r["game"] / "llm"))
                      for r in ix["records"] if r["kind"] == "agent"})
    anon = {c: (f"G{i + 1}", c.split("/")[0]) for i, c in enumerate(configs)}
    out = []
    for r in ix["records"]:
        d = data / r["path"]
        md = jload(d / "metadata.json")
        if r["kind"] == "human":
            ac = jload(d / md["actions"])
            levels, total = ac["levels"], len(ac["actions"])
            entity, harness = r["name"], "human"
            last_attempted = max((int(x["level"]) for x in levels), default=0)
        else:
            ac = jload(d / md["actions"])["paper"]
            levels, total = ac["levels"], int(ac["actions"])
            cfg = str(Path(r["path"]).relative_to(Path("games") / r["game"] / "llm"))
            entity, harness = anon[cfg]
            last_attempted = int(ac.get("last_attempted_level") or max((int(x["level"]) for x in levels), default=0))
        counts = level_counts(levels, total)
        an = jload(d / md["annotations"])
        out.append(dict(id=r["id"], kind=r["kind"], game=r["game"], entity=entity, harness=harness, path=d,
                        meta=md, levels=sorted(levels, key=lambda x: int(x["level"])), counts=counts,
                        total=total, last_attempted=last_attempted, state=r.get("state"),
                        mechanisms={m["mechanism_id"]: m for m in an["mechanisms"]}))
    return out


# ---------------------------------------------------------------- identity (rule section 1)

def cmd_identity(a) -> int:
    data = Path(a.data).expanduser()
    c1 = [json.loads(l) for l in C1_LEVELS.open()]
    n_levels = {r["game"]: int(r["n_levels"]) for r in c1}
    game_ids = {r["game"]: r["game_id"] for r in c1}
    ix = jload(data / "index.json")
    tr = trajectories(data)
    res = {}
    for g in sorted(n_levels):
        mine = [t for t in tr if t["game"] == g]
        ds_levels = max((int(x["level"]) for t in mine for x in t["levels"]), default=0)
        bad_counts = sorted(t["id"] if t["kind"] == "human" else t["entity"] for t in mine if t["counts"] is None)
        res[g] = dict(dataset_levels=ds_levels, c1_n_levels=n_levels[g], I1=ds_levels == n_levels[g],
                      trajectories_bad_counts=bad_counts)
    revision1 = None
    if a.env_dir:
        import logging
        import arc_agi
        from arcengine import GameAction
        logging.disable(logging.WARNING)
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=a.env_dir)
        def replay(r):
            d = data / r["path"]
            ac = jload(d / jload(d / "metadata.json")["actions"])
            counts = level_counts(ac["levels"], len(ac["actions"]))
            try:
                env = arcade.make(game_ids[r["game"]])
                f = env.reset()
                done_at, prev = [], int(f.levels_completed)
                for i, act in enumerate(ac["actions"], 1):
                    aid = int(act[0])
                    ga = GameAction.RESET if aid == 0 else GameAction.from_id(aid)
                    f = env.step(ga, data={"x": int(act[1]), "y": int(act[2])} if aid == 6 else None)
                    lc = int(f.levels_completed)
                    if lc > prev:
                        done_at.extend([i] * (lc - prev))
                        prev = lc
                engine_counts = [b - a_ for a_, b in zip([0] + done_at, done_at)]
                state = getattr(f.state, "name", str(f.state))
                return dict(trajectory=r["id"], state=state, engine_counts=engine_counts, dataset_counts=counts,
                            ok=state == "WIN" and counts is not None and engine_counts == counts)
            except Exception as e:  # a format problem, not an outcome: the game stays count-matched
                return dict(trajectory=r["id"], ok=None, error=f"{type(e).__name__}: {e}"[:200])

        firsts = {}
        for r in ix["records"]:
            if r["kind"] == "human":
                firsts.setdefault(r["game"], r)
        for g, r in firsts.items():
            if g in res:
                res[g]["I2"] = replay(r)
        if a.all_humans:  # proposed revision 1: replay every human trajectory
            per = {}
            for r in ix["records"]:
                if r["kind"] == "human" and r["game"] in res:
                    first = r["id"] == firsts[r["game"]]["id"]
                    per[r["id"]] = res[r["game"]]["I2"]["ok"] if first else replay(r)["ok"]
            by_game = defaultdict(lambda: dict(humans=0, replay=0))
            for k, v in per.items():
                g_ = k.split("__")[1]
                by_game[g_]["humans"] += 1
                by_game[g_]["replay"] += bool(v)
            revision1 = dict(rule="RULE.md revision 1 (23:52 IST Oct 29): I2' per trajectory",
                             matched=sorted(g_ for g_, v in by_game.items() if v["replay"] > 0 and res[g_]["I1"]),
                             per_game=dict(sorted(by_game.items())),
                             trajectories_ok=sorted(k for k, v in per.items() if v),
                             trajectories_failed=sorted(k for k, v in per.items() if v is False),
                             trajectories_unreplayable=sorted(k for k, v in per.items() if v is None))
    for g, v in res.items():
        i2 = (v.get("I2") or {}).get("ok")
        v["status"] = ("excluded" if not v["I1"] or i2 is False else "matched" if i2 else "count-matched")
    OUT.mkdir(parents=True, exist_ok=True)
    summ = dict(rule="experiments/arc3-think-aloud/RULE.md section 1", games=res,
                revision1=revision1,
                matched=sorted(g for g, v in res.items() if v["status"] == "matched"),
                count_matched=sorted(g for g, v in res.items() if v["status"] == "count-matched"),
                excluded=sorted(g for g, v in res.items() if v["status"] == "excluded"))
    (OUT / "identity.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("matched", "count_matched", "excluded")}))
    return 0


# ---------------------------------------------------------------- recognition (rule section 3)

def recog(m: dict, variant: str) -> dict:
    """One mechanism instance: eligible, recognised, moment (level, level_action, absolute_action), trigger."""
    rr = m.get("recognition_review") or {}
    st = rr.get("status")
    elig = (m.get("eligibility") or {}).get("status") == "eligible" and st != "not_yet_observable"
    out = dict(eligible=elig, recognised=False, level=None, level_action=None, absolute_action=None, trigger=None)
    av = m.get("availability") or {}
    if av.get("status") == "triggered":
        out["trigger"] = (av.get("objective_first_exposure") or {}).get("absolute_action")
    if not elig or st != "recognized":
        return out
    at = rr.get("recognized_at") or {}
    if variant == "s1":
        if rr.get("primary_evidence_type") != "explicit_reasoning":
            er = (rr.get("evidence_timings") or {}).get("explicit_reasoning")
            if not er or not er.get("anchor"):
                return out
            at = er["anchor"]
        else:
            er = (rr.get("evidence_timings") or {}).get("explicit_reasoning")
            if er and er.get("anchor"):
                at = er["anchor"]
    if variant == "s2" and rr.get("confidence") not in ("high", "medium"):
        return out
    if at.get("level") is None:
        return out
    out.update(recognised=True, level=int(at["level"]), level_action=at.get("level_action"),
               absolute_action=at.get("absolute_action"))
    return out


def unit(t: dict, L: int, S: list[str], variant: str) -> dict | None:
    """Trajectory x level for the mechanism set S (rule sections 3-4); None if not reached or not eligible."""
    if t["last_attempted"] < L:
        return None
    rs = [recog(t["mechanisms"][m], variant) for m in S if m in t["mechanisms"]]
    rs = [r for r in rs if r["eligible"]]
    if not rs:
        return None
    lv = {int(x["level"]): x for x in t["levels"]}
    cleared = bool(lv.get(L, {}).get("completed"))
    n_L = t["counts"][L - 1] if t["counts"] and L <= len(t["counts"]) else None
    rec = [r for r in rs if r["recognised"]]
    first = min(rec, key=lambda r: (r["level"], r["level_action"] if r["level_action"] is not None else 10**9),
                default=None)
    u = dict(id=t["id"], kind=t["kind"], entity=t["entity"], harness=t["harness"], cleared=cleared,
             actions_to_clear=n_L if cleared else None, recognised=first is not None, m2=None, cls="never",
             lag=None, seconds=None)
    if first is None:
        return u
    if first["level"] < L:
        u["cls"] = "before_level"
    elif first["level"] == L:
        u["m2"] = first["level_action"]
        if cleared and n_L is not None and first["level_action"] is not None and first["level_action"] > n_L:
            u["cls"] = "after_clear"
        else:
            u["cls"] = "before_clear" if cleared else "at_level_uncleared"
    else:
        u["cls"] = "after_clear"
    if first["trigger"] is not None and first["absolute_action"] is not None:
        u["lag"] = int(first["absolute_action"]) - int(first["trigger"])
    if t["kind"] == "human" and u["cls"] == "before_clear" and first["absolute_action"] is not None:
        start_action = sum(t["counts"][: L - 1]) + 1
        u["seconds"] = seconds_between(t, start_action, int(first["absolute_action"]))
    return u


_TL: dict = {}


def seconds_between(t: dict, a0: int, a1: int) -> float | None:
    """Seconds between the first timeline points at two effective action counts, within one segment (M4)."""
    key = t["id"]
    if key not in _TL:
        segs = []
        for s in t["meta"].get("segments", []):
            p = t["path"] / s["timeline"]
            tl = json.loads(gzip.open(p).read()) if p.suffix == ".gz" else jload(p)
            first = {}
            for pt in tl["points"]:
                c = pt.get("effective_action_count")
                if c is not None and c not in first and pt.get("kind") in ("action", "animation", "start"):
                    first[c] = pt["time_ms"]
            segs.append(first)
        _TL[key] = segs
    for first in _TL[key]:
        if a0 in first and a1 in first:
            return (first[a1] - first[a0]) / 1000.0
    return None


# ---------------------------------------------------------------- our side

def our_levels(team_dir: Path | None) -> dict:
    lv = {}
    for r in map(json.loads, C1_LEVELS.open()):
        lv[(r["game"], int(r["level"]))] = dict(game=r["game"], level=int(r["level"]), stuck=False, cases=[])
    for d in sorted(CASES.iterdir()):
        m = jload(d / "meta.json")
        k = (m["game_id"].split("-")[0], int(m["target_level"]))
        lv[k]["stuck"] = True
        lv[k]["cases"].append(m["case_id"])
    labels = {r["case_id"]: r["rule_discovered"] for r in csv.DictReader(RULE_LABELS.open())}
    per_case = defaultdict(dict)
    for name, srcs in ARMS.items():
        for run, arm in srcs:
            for r in map(json.loads, (RUNS / run / "r1_results.jsonl").open()):
                if r["arm"] == arm:
                    per_case[r["case_id"]].setdefault(name, []).append(1.0 if r["cleared"] else 0.0)
    for v in lv.values():
        if not v["cases"]:
            continue
        labs = {labels.get(c) for c in v["cases"]}
        v["rule_label"] = labs.pop() if len(labs) == 1 else "mixed"
        for name in ARMS:
            vals = [statistics.mean(per_case[c][name]) for c in v["cases"] if per_case[c].get(name)]
            v[name] = statistics.mean(vals) if vals else None
        if v.get("insight30") is not None and v.get("unchanged30") is not None:
            v["insight_gain30"] = v["insight30"] - v["unchanged30"]
        if v.get("t1_placebo30") is not None and v.get("unchanged30") is not None:
            v["placebo_gain30"] = v["t1_placebo30"] - v["unchanged30"]
    for tag, (path, cols) in C1_MAPS.items():
        for r in csv.DictReader(path.open(), delimiter="\t"):
            k = (r["game"], int(r["level"]))
            for arm, col in cols.items():
                lv[k][f"{tag}_{arm}"] = int(r[col])
    if team_dir:
        for r in csv.DictReader((team_dir / "arc3_levels.csv").open()):
            k = (r["game"], int(r["level"]))
            if k in lv:
                s = r.get("insight_level_step")
                lv[k].update(team_status=r["status"], team_how_found=r["how_found"],
                             team_onset=float(s) if s not in (None, "") else None)
    return lv


def load_key_map() -> dict:
    sha_file, path = OUT / "key_map.sha256", OUT / "key_map.csv"
    want = sha_file.read_text().split()[0]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if want != got:
        raise SystemExit(f"key_map.csv sha256 {got} differs from the committed {want}: refusing to run (rule 2)")
    km = defaultdict(lambda: {"key": [], "part": []})
    for r in csv.DictReader(path.open()):
        if r["label"] in ("key", "part"):
            km[(r["game"], int(r["level"]))][r["label"]].append(r["mechanism_id"])
    return km


# ---------------------------------------------------------------- statistics

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def share(us, f):
    us = list(us)
    return sum(1 for u in us if f(u)) / len(us) if us else None


STATS = {
    "M1": lambda us: share(us, lambda u: u["recognised"]),
    "M2": lambda us: med(u["m2"] for u in us),
    "before_first_clear": lambda us: share(us, lambda u: u["cls"] in ("before_level", "before_clear")),
    "M4": lambda us: med(u["seconds"] for u in us),
    "M5": lambda us: med(u["lag"] for u in us),
    "M6": lambda us: med(u["actions_to_clear"] for u in us),
    "clear": lambda us: share(us, lambda u: u["cleared"]),
}


def level_stat(units_by_level: dict, levels: list, stat: str) -> dict:
    out = {}
    for k in levels:
        us = units_by_level.get(k) or []
        out[k] = STATS[stat](us) if us else None
    return out


def resampled(units_by_level: dict, k, stat: str, rng):
    """Second stage of the bootstrap: trajectories within one drawn level, drawn afresh for each occurrence."""
    us = units_by_level.get(k) or []
    return STATS[stat]([rng.choice(us) for _ in us]) if us else None


def spearman(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 4:
        return None

    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def ci(vals):
    v = sorted(x for x in vals if x is not None)
    if len(v) < 100:
        return None
    return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]


def contrast(units_by_level, A, B, stat, draws=DRAWS, seed=SEED) -> dict:
    """Median over levels of the per-level stat, group A minus group B; two-stage bootstrap and permutation."""
    base = level_stat(units_by_level, A + B, stat)
    a = [base[k] for k in A if base[k] is not None]
    b = [base[k] for k in B if base[k] is not None]
    if not a or not b:
        return dict(stat=stat, nA=len(a), nB=len(b), diff=None)
    obs = statistics.median(a) - statistics.median(b)
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        xa = [resampled(units_by_level, rng.choice(A), stat, rng) for _ in A]
        xb = [resampled(units_by_level, rng.choice(B), stat, rng) for _ in B]
        xa = [x for x in xa if x is not None]
        xb = [x for x in xb if x is not None]
        if xa and xb:
            boots.append(statistics.median(xa) - statistics.median(xb))
    pool = [(k, base[k]) for k in A + B if base[k] is not None]
    hits = 0
    for _ in range(draws):
        rng.shuffle(pool)
        pa = [v for _, v in pool[: len(a)]]
        pb = [v for _, v in pool[len(a):]]
        if abs(statistics.median(pa) - statistics.median(pb)) >= abs(obs) - 1e-12:
            hits += 1
    c = ci(boots)
    return dict(stat=stat, nA=len(a), nB=len(b), medA=statistics.median(a), medB=statistics.median(b),
                diff=round(obs, 4), ci95=c, perm_p=round((hits + 1) / (draws + 1), 4),
                excludes_0=bool(c and (c[0] > 0 or c[1] < 0)))


def corr(units_by_level, levels, ours: dict, stat, draws=DRAWS, seed=SEED) -> dict:
    """Spearman across levels of our per-level value against the human per-level stat; level bootstrap."""
    base = level_stat(units_by_level, levels, stat)
    keep = [k for k in levels if base[k] is not None and ours.get(k) is not None]
    if len(keep) < 4:
        return dict(stat=stat, n=len(keep), rho=None)
    rho = spearman([ours[k] for k in keep], [base[k] for k in keep])
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        rk = [rng.choice(keep) for _ in keep]
        boots.append(spearman([ours[k] for k in rk], [resampled(units_by_level, k, stat, rng) for k in rk]))
    c = ci(boots)
    return dict(stat=stat, n=len(keep), rho=None if rho is None else round(rho, 4), ci95=c,
                excludes_0=bool(c and (c[0] > 0 or c[1] < 0)))


# ---------------------------------------------------------------- measures (rule sections 4-5)

def build_units(tr, levels, km, variant, which="K"):
    hum, agt = defaultdict(list), defaultdict(list)
    mech_intro = {}
    for t in tr:
        for mid, m in t["mechanisms"].items():
            mech_intro[(t["game"], mid)] = [int(x) for x in (m.get("mechanism") or {}).get("introduced_levels", [])]
    for (g, L) in levels:
        if which == "K":
            e = km.get((g, L), {"key": [], "part": []})
            S = e["key"] + (e["part"] if variant == "s3" else []) or e["part"]
        else:
            S = sorted({mid for (gg, mid), ls in mech_intro.items() if gg == g and L in ls})
        if not S:
            continue
        for t in tr:
            if t["game"] != g:
                continue
            u = unit(t, L, S, variant)
            if u is not None:
                (hum if t["kind"] == "human" else agt)[(g, L)].append(u)
    return hum, agt


def cmd_measures(a) -> int:
    data = Path(a.data).expanduser()
    ident = jload(OUT / "identity.json")
    excluded = set(ident["excluded"])
    matched_games = set(ident["matched"])
    suffix, drop = "_frozen", set()
    if a.reading in ("revision1", "replay_only"):  # revision 1 (RULE.md, 23:52 IST Oct 29)
        r1 = ident["revision1"]
        matched_games = set(r1["matched"])
        excluded = set(ident["games"]) - matched_games
        suffix = "" if a.reading == "revision1" else "_replay_only"
        if a.reading == "replay_only":
            drop = set(r1["trajectories_failed"]) | set(r1["trajectories_unreplayable"])
    km = load_key_map()
    tr = [t for t in trajectories(data) if t["game"] not in excluded and t["counts"] is not None
          and t["id"] not in drop]
    ours = our_levels(Path(a.team).expanduser() if a.team else None)
    levels = sorted(k for k in ours if k[0] not in excluded)
    mapped = [k for k in levels if km.get(k, {}).get("key") or km.get(k, {}).get("part")]
    stuck = [k for k in mapped if ours[k]["stuck"]]
    rest = [k for k in mapped if not ours[k]["stuck"]]
    result = dict(rule="experiments/arc3-think-aloud/RULE.md (frozen acc8d87, revision 1)", reading=a.reading, levels=len(levels), mapped=len(mapped),
                  part_mapped=sum(1 for k in mapped if not km[k]["key"]), unmapped=len(levels) - len(mapped),
                  excluded_games=sorted(excluded), dropped_trajectories=sorted(drop), stuck_mapped=len(stuck),
                  readings={})
    agreement = jload(OUT / "coding_agreement.json")
    low = bool(agreement["low_agreement"])

    def primaries(hum, lv_set):
        conv = [k for k in lv_set if ours[k].get("c1_A0F") == 0 and ours[k].get("c1_A7F") == 1]
        both = [k for k in lv_set if ours[k].get("c1_A0F") == 1 and ours[k].get("c1_A7F") == 1]
        st = [k for k in lv_set if ours[k]["stuck"]]
        rs = [k for k in lv_set if not ours[k]["stuck"]]
        out = dict(P1=contrast(hum, st, rs, "M2"), P2=contrast(hum, conv, both, "M2"),
                   P3=corr(hum, lv_set, {k: ours[k].get("team_onset") for k in lv_set}, "M2"))
        for name, v in out.items():
            ex = v.get("excludes_0")
            v["reading"] = (("goes with" if name == "P3" else "differs") if ex else "no difference shown")
        return out

    humK, _ = build_units(tr, mapped, km, "primary")
    humN, _ = build_units(tr, levels, km, "primary", which="N")
    mK = [k for k in mapped if k[0] in matched_games]
    mN = [k for k in levels if k[0] in matched_games]
    result["primaries"] = dict(
        agreement=dict(kappa=agreement["kappa"], low_agreement=low),
        lead="N(L)" if low else "K(L)",
        K_matched=primaries(humK, mK), N_matched=primaries(humN, mN),
        K_matched_and_count_matched=primaries(humK, mapped),
        count_matched_games=sorted(ident["count_matched"]))
    rows = {}
    for variant in ("primary", "s1", "s2", "s3"):
        hum, agt = build_units(tr, mapped, km, variant)
        R = {}
        R["overview"] = {
            coh: {st: dict(levels=sum(1 for v in level_stat(u, mapped, st).values() if v is not None),
                           median_over_levels=med(level_stat(u, mapped, st).values()))
                  for st in ("M1", "M2", "before_first_clear", "M4", "M5", "M6", "clear")}
            for coh, u in (("human", hum), ("agents", agt))}
        R["stuck_vs_rest"] = {st: contrast(hum, stuck, rest, st) for st in ("M1", "M2", "before_first_clear")}
        # humans against agents, per level paired difference
        hs, ag = level_stat(hum, mapped, "M1"), level_stat(agt, mapped, "M1")
        R["human_minus_agents_M1"] = med([hs[k] - ag[k] for k in mapped if hs[k] is not None and ag[k] is not None])
        hs2, ag2 = level_stat(hum, mapped, "M2"), level_stat(agt, mapped, "M2")
        R["human_minus_agents_M2"] = med([hs2[k] - ag2[k] for k in mapped if hs2[k] is not None and ag2[k] is not None])
        # team onsets
        onset = {k: ours[k].get("team_onset") for k in mapped}
        R["team_onset_vs_M2"] = corr(hum, mapped, onset, "M2")
        dt = [k for k in mapped if ours[k].get("team_how_found") == "deliberate test"]
        bd = [k for k in mapped if ours[k].get("team_how_found") == "studying the board before acting"]
        R["how_found_deliberate_vs_board"] = {st: contrast(hum, dt, bd, st) for st in ("M2", "M1")}
        # stuck levels: told-insight and placebo gains
        for g in ("insight_gain30", "placebo_gain30", "r1v_insight30", "r1f_insight15", "r1f_placebo15"):
            vals = {k: ours[k].get(g) for k in stuck}
            R[f"stuck_{g}"] = {st: corr(hum, stuck, vals, st) for st in ("M2", "M1")}
        strata = defaultdict(list)
        for k in stuck:
            strata[ours[k].get("rule_label")].append(k)
        R["stuck_by_rule_label"] = {lab: {st: med(level_stat(hum, ks, st).values()) for st in ("M1", "M2")}
                                    | {"levels": len(ks)} for lab, ks in strata.items()}
        # C1 cells
        cell = {}
        for k in mapped:
            o = ours[k]
            if "c1_A0F" in o:
                cell[k] = {(1, 1): "both", (0, 1): "converted", (1, 0): "lost", (0, 0): "neither"}[
                    (o["c1_A0F"], o["c1_A7F"])]
        by = defaultdict(list)
        for k, c in cell.items():
            by[c].append(k)
        R["c1_cells"] = {c: {st: med(level_stat(hum, ks, st).values()) for st in ("M1", "M2", "before_first_clear")}
                         | {"levels": len(ks)} for c, ks in by.items()}
        R["c1_converted_vs_both"] = {st: contrast(hum, by["converted"], by["both"], st)
                                     for st in ("M1", "M2", "before_first_clear")}
        R["c1_neither_vs_both"] = {st: contrast(hum, by["neither"], by["both"], st)
                                   for st in ("M1", "M2", "before_first_clear")}
        a0f0 = [k for k in mapped if ours[k].get("c1_A0F") == 0 and "c1p_A7cF" in ours[k]]
        pc = [k for k in a0f0 if ours[k]["c1p_A7cF"] == 1]
        pn = [k for k in a0f0 if ours[k]["c1p_A7cF"] == 0]
        R["c1p_placebo_converted_vs_not"] = {st: contrast(hum, pc, pn, st) for st in ("M1", "M2")}
        a0f1 = [k for k in mapped if ours[k].get("c1_A0F") == 1 and "c1s_A7SF" in ours[k]]
        sl = [k for k in a0f1 if ours[k]["c1s_A7SF"] == 0]
        sk = [k for k in a0f1 if ours[k]["c1s_A7SF"] == 1]
        R["c1s_self_lost_vs_kept"] = {st: contrast(hum, sl, sk, st) for st in ("M1", "M2")}
        if variant == "primary":
            hN, aN = build_units(tr, levels, km, variant, which="N")
            R["N_overview"] = {coh: {st: med(level_stat(u, levels, st).values()) for st in ("M1", "M2")}
                               for coh, u in (("human", hN), ("agents", aN))}
            per_cfg = defaultdict(lambda: defaultdict(list))
            for k, us in agt.items():
                for u in us:
                    per_cfg[(u["entity"], u["harness"])][k].append(u)
            R["agents_by_configuration"] = {
                f"{e} ({h})": {st: med(level_stat(u, mapped, st).values()) for st in ("M1", "M2", "clear")}
                for (e, h), u in sorted(per_cfg.items())}
            for k in levels:
                o = ours[k]
                h, ag_ = hum.get(k, []), agt.get(k, [])
                e = km.get(k, {"key": [], "part": []})
                rows[k] = dict(track="ARC-3", game=k[0], level=k[1], stuck=int(o["stuck"]), cases=len(o["cases"]),
                               map=("key" if e["key"] else "part" if e["part"] else "unmapped"),
                               K=";".join(e["key"] or e["part"]),
                               human_n=len(h), **{f"human_{s}": STATS[s](h) if h else None
                                                  for s in ("M1", "M2", "before_first_clear", "M4", "M5", "M6")},
                               agent_n=len(ag_), **{f"agent_{s}": STATS[s](ag_) if ag_ else None
                                                    for s in ("clear", "M1", "M2", "before_first_clear")},
                               team_status=o.get("team_status"), team_onset=o.get("team_onset"),
                               team_how_found=o.get("team_how_found"), rule_label=o.get("rule_label"),
                               **{n: o.get(n) for n in list(ARMS) + ["insight_gain30", "placebo_gain30"]},
                               c1_A0F=o.get("c1_A0F"), c1_A7F=o.get("c1_A7F"), c1p_A7cF=o.get("c1p_A7cF"),
                               c1s_A7SF=o.get("c1s_A7SF"))
        result["readings"][variant] = R
    def walk(o):
        if isinstance(o, dict):
            if "ci95" in o and o["ci95"] is not None:
                yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    ivs = list(walk(result))
    result["intervals"] = dict(computed=len(ivs), exclude_0=sum(1 for v in ivs if v.get("excludes_0")),
                               expected_by_chance_at_5pct=round(0.05 * len(ivs), 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"RESULT{suffix}.json").write_text(json.dumps(result, indent=1, default=str))
    with (OUT / f"levels{suffix}.csv").open("w", newline="") as fh:
        cols = list(next(iter(rows.values())).keys())
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: (round(v, 4) if isinstance(v, float) else v) for c, v in rows[k].items()})
    print(json.dumps({k: result[k] for k in ("levels", "mapped", "part_mapped", "unmapped", "stuck_mapped",
                                             "intervals")}))
    return 0


def cmd_agree(a) -> int:
    """Cohen's kappa over (level, mechanism) labels of two coders; levels with the same key set (rule section 2)."""
    sheet = jload(Path(a.sheet).expanduser())
    lab = {}
    for name, path in (("A", a.coder_a), ("B", a.coder_b)):
        d = {}
        for r in csv.DictReader(Path(path).expanduser().open()):
            d[(r["game"], int(r["level"]), r["mechanism_id"])] = r["label"].strip()
        lab[name] = d
    pairs, same_key, levels = [], 0, 0
    for g in sheet["games"]:
        for lv in g["levels"]:
            levels += 1
            ka, kb = set(), set()
            for m in g["mechanisms"]:
                k = (g["game"], lv["level"], m["mechanism_id"])
                la, lb = lab["A"].get(k, "none"), lab["B"].get(k, "none")
                pairs.append((la, lb))
                if la == "key":
                    ka.add(m["mechanism_id"])
                if lb == "key":
                    kb.add(m["mechanism_id"])
            same_key += ka == kb
    cats = ("key", "part", "none")
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pe = sum((sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else None
    table = {f"{x}/{y}": sum(1 for p in pairs if p == (x, y)) for x in cats for y in cats}
    out = dict(rule="RULE.md section 2 (floor 0.40)", pairs=n, levels=levels, observed_agreement=round(po, 4),
               kappa=None if kappa is None else round(kappa, 4), low_agreement=kappa is None or kappa < 0.40,
               levels_same_key_set=same_key, confusion_A_by_B=table)
    (OUT / "coding_agreement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("identity")
    p.add_argument("--data", required=True)
    p.add_argument("--env-dir")
    p.add_argument("--all-humans", action="store_true", help="also replay every human trajectory (revision 1)")
    p = sub.add_parser("measures")
    p.add_argument("--data", required=True)
    p.add_argument("--team")
    p.add_argument("--reading", choices=("revision1", "replay_only", "frozen"), default="revision1",
                   help="revision1 = primary; replay_only = sensitivity; frozen = the rule before revision 1")
    p = sub.add_parser("agree")
    p.add_argument("--sheet", required=True)
    p.add_argument("--coder-a", required=True)
    p.add_argument("--coder-b", required=True)
    a = ap.parse_args()
    return dict(identity=cmd_identity, measures=cmd_measures, agree=cmd_agree)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
