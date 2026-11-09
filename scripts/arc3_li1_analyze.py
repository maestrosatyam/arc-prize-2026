#!/usr/bin/env python3
"""LI1 analysis (run a3-li1-agent): the frozen rule of experiments/arc3-li1/LI1_RULE.md (frozen at 1e2e318,
01:20 IST Oct 29) on the agent run's r1_results.jsonl, arms A7G (the fold's
tuned 27B's insight) and A7B (the untuned 27B's), the 35 stuck states, 160 actions / 30 min.

Validity (any failing: no decision, report only):
  - the CPU twins passed (training kernel and agent kernel; the archived twin logs);
  - the fold separation holds (folds.json: no case's game in its fold model's training games; the training record);
  - all 35 cases have both texts, committed by hash (texts.jsonl's sha256 equals the agent kernel's PROVENANCE), and the
    rows show each case-arm's message shown once with the committed length;
  - at least 33 of 35 case-arms per arm end by a clear, the action budget or the wall budget;
  - the replayed prefixes match.
Reading, A7G against A7B, exact two-sided McNemar on the discordant pairs (exactly one line):
  learned insight transfers   A7G - A7B >= 4, p <= 0.05 and A7G >= 18
  no transfer                 A7G - A7B <= 1
  report                      otherwise
Recorded beside, no decision: clears within 80 actions / 900 s; clears per fold; A7G against R1g's A7L v1 and v2 and
both arms against the four unchanged samples (paired); the partial (22) and yes (11) groups; tokens per action;
training loss per fold; text lengths; key-element match (content-word overlap with key_elements.json); format (outputs
with no "Key insight:", texts under 8 words).

  envs/arc3/bin/python -I scripts/arc3_li1_analyze.py [--results experiments/runs/kaggle-A3-LI1-agent-v1/r1_results.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "experiments" / "runs"
LI1 = ROOT / "experiments" / "arc3-li1"
AGENT = "kaggle-A3-LI1-agent-v1"
TRAIN = "kaggle-A3-LI1-train-v1"
ARMS = ("A7G", "A7B")
KIND = {"A7G": "learned", "A7B": "untuned"}
RAN_TO_END = ("cleared", "action_budget", "wall_budget")
VALID_ENDED, N = 33, 35
FAST = (80, 900.0)
COMPARE = (("told R1g v1 A7L", "kaggle-R1g-v1", "A7L"), ("told R1g v2 A7L", "kaggle-R1g-v2", "A7L"),
           ("unchanged R1d A0L", "kaggle-R1d-v1", "A0L"), ("unchanged R1g v1 A0Lb", "kaggle-R1g-v1", "A0Lb"),
           ("unchanged R1g v2 A0Lb", "kaggle-R1g-v2", "A0Lb"), ("unchanged R1j A0Lc", "kaggle-R1j-v1", "A0Lc"))
LABELS = ROOT / "experiments" / "arc3-failure-annotation" / "labels_rule_A.csv"
KEY_ELEMENTS = ROOT / "experiments" / "arc3-team-play" / "key_elements.json"
RULE = ("LI1, frozen (experiments/arc3-li1/LI1_RULE.md at 1e2e318): validity (twins passed; fold separation; both texts "
        "for all 35, committed by hash, shown once with the committed length; >= 33/35 per arm ended by clear / action "
        "budget / wall budget; prefixes match); A7G vs A7B, exact McNemar: A7G - A7B >= 4, p <= 0.05 and A7G >= 18 "
        "learned insight transfers; A7G - A7B <= 1 no transfer; otherwise report")

# R1f's control-selection word lists (kaggle/arc3/r1/build_r1_notebook.py, copied verbatim): the key-element match
_STOPWORDS = frozenset(
    "about after also another because before being both does each either every from have into more most must "
    "only onto other over same some such than that their them then there these they this those through under "
    "until upon when where which while will with without your".split())
_GENERIC_WORDS = frozenset(
    "action actions move moves moved moving player piece pieces cell cells colour colours color level levels left "
    "right bottom step steps board grid click clicks clicked game once last first shown shows remaining used part "
    "fraction".split())


def content_words(text: str) -> set[str]:
    """Lowercase alphabetic tokens of four or more letters, not stopwords or generic game words; a trailing s removed."""
    out = set()
    for w in re.findall(r"[a-z]+", str(text).lower()):
        if len(w) >= 4 and w not in _STOPWORDS and w not in _GENERIC_WORDS:
            out.add(w[:-1] if w.endswith("s") else w)
    return out


def mcnemar_p(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_arm(path: Path, arm: str) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["case_id"]: r for r in rows if r.get("arm") == arm}


def cleared(r) -> bool:
    return bool(r) and bool(r.get("cleared"))


def within(r, limit=FAST) -> bool:
    return cleared(r) and r.get("clear_actions") is not None and r.get("clear_wall_s") is not None \
        and int(r["clear_actions"]) <= limit[0] and float(r["clear_wall_s"]) <= limit[1]


def pair(a: dict, b: dict, cases: list[str]) -> dict:
    x = sum(cleared(a.get(c)) and not cleared(b.get(c)) for c in cases)
    y = sum(cleared(b.get(c)) and not cleared(a.get(c)) for c in cases)
    return {"arm_only": x, "other_only": y, "both": sum(cleared(a.get(c)) and cleared(b.get(c)) for c in cases),
            "p": mcnemar_p(x, y)}


def twin_passed(archive: Path) -> bool | None:
    logs = sorted((archive / "preflight").glob("*.log")) if (archive / "preflight").is_dir() else []
    if not logs:
        return None
    return any("preflight: PASS" in p.read_text(encoding="utf-8", errors="replace") for p in logs)


def analyze(results: Path, texts_path: Path = LI1 / "texts.jsonl", runs_root: Path = RUNS) -> dict:
    out: dict = {"rule": RULE, "results": str(results)}
    texts = [json.loads(l) for l in texts_path.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if texts_path.is_file() else []
    by_text = {(t["case_id"], t["arm"]): t for t in texts}
    texts_sha = hashlib.sha256(texts_path.read_bytes()).hexdigest() if texts_path.is_file() else None
    folds = json.loads((LI1 / "folds.json").read_text())
    fold_of = {c: s["fold"] for s in folds["separation"] for c in s["stuck_cases"]}
    sep_ok = all(not s["violations"] for s in folds["separation"]) and all(
        set(s["held_out_games"]).isdisjoint(s["training_games"]) for s in folds["separation"])
    agent_archive = results.parent
    prov_path = ROOT / "kaggle" / "arc3" / "li1" / "agent" / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
    rows = {a: load_arm(results, a) for a in ARMS}
    cases = sorted(set(fold_of) or set(rows["A7G"]) | set(rows["A7B"]))
    if not rows["A7G"] and not rows["A7B"]:
        out["decision"] = f"pending: no agent results ({results})"
        return out

    shown_bad = []
    for c in cases:
        for a in ARMS:
            r, t = rows[a].get(c), by_text.get((c, a))
            if r is None or t is None or r.get("oracle_kind") != KIND[a] or int(r.get("oracle_shown") or 0) != 1 \
                    or int(r.get("oracle_errors") or 0) != 0 or int(r.get("oracle_chars") or 0) != len(t["message"]):
                shown_bad.append((c, a))
    ended = {a: sum(1 for c in cases if (rows[a].get(c) or {}).get("end_reason") in RAN_TO_END) for a in ARMS}
    replay_bad = sorted({c for a in ARMS for c in cases if (rows[a].get(c) or {}).get("replay_ok") is not True})
    train_archive = runs_root / TRAIN
    checks = {
        "twins_passed": twin_passed(train_archive) is True and twin_passed(agent_archive) is True,
        "fold_separation": sep_ok and all(by_text.get((c, "A7G"), {}).get("fold") == fold_of.get(c) for c in cases),
        "texts_complete_committed_shown": len(cases) == N and all((c, a) in by_text for c in cases for a in ARMS)
        and texts_sha is not None and texts_sha == (prov.get("texts") or {}).get("sha256") and not shown_bad,
        "ended_33_per_arm": all(ended[a] >= VALID_ENDED for a in ARMS),
        "prefixes_match": not replay_bad,
    }
    out["validity"] = {"valid": all(checks.values()), "checks": checks, "ended": ended, "shown_mismatch": shown_bad,
                       "replay_mismatch": replay_bad, "texts_sha256": texts_sha}

    g = sum(cleared(rows["A7G"].get(c)) for c in cases)
    b = sum(cleared(rows["A7B"].get(c)) for c in cases)
    pr = pair(rows["A7G"], rows["A7B"], cases)
    d = g - b
    if d >= 4 and pr["p"] is not None and pr["p"] <= 0.05 and g >= 18:
        band = "learned insight transfers"
    elif d <= 1:
        band = "no transfer"
    else:
        band = "report"
    out["reading"] = {"A7G": g, "A7B": b, "difference": d, **pr, "band": band}
    out["decision"] = band if out["validity"]["valid"] else "no decision (validity failed): report only"

    labels = {r["case_id"]: r["rule_discovered"] for r in csv.DictReader(open(LABELS, newline=""))} if LABELS.is_file() else {}
    groups = {gname: [c for c in cases if labels.get(c) == gname] for gname in ("partial", "yes")}
    out["groups"] = {gname: {a: sum(cleared(rows[a].get(c)) for c in cs) for a in ARMS} | {"n": len(cs)}
                     for gname, cs in groups.items()}
    out["within_80_900"] = {a: sum(within(rows[a].get(c)) for c in cases) for a in ARMS}
    out["per_fold"] = {k: {"cases": sum(1 for c in cases if fold_of.get(c) == k),
                           **{a: sum(cleared(rows[a].get(c)) for c in cases if fold_of.get(c) == k) for a in ARMS}}
                       for k in range(5)}
    acts = {a: sum(int((rows[a].get(c) or {}).get("actions_used") or 0) for c in cases) for a in ARMS}
    toks = {a: sum(int((rows[a].get(c) or {}).get("generated_tokens") or 0) for c in cases) for a in ARMS}
    out["tokens_per_action"] = {a: round(toks[a] / acts[a]) if acts[a] else None for a in ARMS}
    comp = {}
    for label, run, arm in COMPARE:
        other = load_arm(runs_root / run / "r1_results.jsonl", arm)
        if other:
            comp[label] = {"clears": sum(cleared(other.get(c)) for c in cases),
                           **{a: pair(rows[a], other, cases) for a in ARMS}}
    out["comparisons"] = comp

    ke = json.loads(KEY_ELEMENTS.read_text()) if KEY_ELEMENTS.is_file() else {"levels": []}
    kw = {(l["game"], int(l["level"])): content_words(" ".join(l["key_objects"]) + " " + l["key_use"]) for l in ke["levels"]}
    match = {a: 0 for a in ARMS}
    for c in cases:
        game, level = c.split("_")[0], int(re.search(r"_L(\d+)_", c).group(1))
        K = kw.get((game, level), set())
        for a in ARMS:
            t = by_text.get((c, a))
            if t and K & content_words(t["text"]):
                match[a] += 1
    out["key_element_match"] = match
    out["format"] = {a: {"no_key_insight_label": sum(1 for c in cases if (by_text.get((c, a)) or {}).get("fallback")),
                         "under_8_words": sum(1 for c in cases if int((by_text.get((c, a)) or {}).get("words") or 0) < 8),
                         "chars_median": sorted(len((by_text.get((c, a)) or {}).get("text", "")) for c in cases)[len(cases) // 2]}
                     for a in ARMS}
    trs = sorted(train_archive.rglob("li1_train.json")) if train_archive.is_dir() else []
    tr = trs[0] if len(trs) == 1 else train_archive / "li1_train.json"
    if tr.is_file():
        t = json.loads(tr.read_text())
        out["training"] = {k: {"steps": f["steps"], "loss_first": round(f["losses"][0], 3),
                               "loss_last": round(f["losses"][-1], 3), "s_per_step": f["s_per_step"]}
                           for k, f in sorted((t.get("folds") or {}).items())}
    return out


def format_report(rep: dict) -> str:
    if "reading" not in rep:
        return f"LI1: {rep['decision']}"
    v, r = rep["validity"], rep["reading"]
    pv = "n/a" if r["p"] is None else f"{r['p']:.4f}"
    lines = ["LI1 (a3-li1-agent): A7G (tuned 27B's insight, held-out fold) vs A7B (untuned 27B's), 35 stuck states, "
             "160 actions / 1800 s",
             f"validity: {'valid' if v['valid'] else 'NOT valid'} {v['checks']}; ended {v['ended']} (need 33 each); "
             f"shown mismatches {v['shown_mismatch']}; replay mismatches {v['replay_mismatch']}",
             "",
             f"| track | arm | clears /35 | partial /{rep['groups']['partial']['n']} | yes /{rep['groups']['yes']['n']} "
             "| within 80 / 900 | tokens per action | key-element match /35 | no 'Key insight:' | under 8 words |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        lines.append(f"| ARC-3 | {a} | {r[a]} | {rep['groups']['partial'][a]} | {rep['groups']['yes'][a]} | "
                     f"{rep['within_80_900'][a]} | {rep['tokens_per_action'][a]} | {rep['key_element_match'][a]} | "
                     f"{rep['format'][a]['no_key_insight_label']} | {rep['format'][a]['under_8_words']} |")
    lines += ["", "per fold (cases, A7G, A7B): " + "; ".join(f"{k}: {x['cases']}, {x['A7G']}, {x['A7B']}"
                                                           for k, x in rep["per_fold"].items()),
              "text lengths (median characters): " + "; ".join(f"{a} {rep['format'][a]['chars_median']}" for a in ARMS)]
    for label, c in rep["comparisons"].items():
        lines.append(f"vs {label} ({c['clears']}/35): " + "; ".join(
            f"{a} only {c[a]['arm_only']}, other only {c[a]['other_only']}, p "
            f"{'n/a' if c[a]['p'] is None else format(c[a]['p'], '.3f')}" for a in ARMS))
    if rep.get("training"):
        lines.append("training: " + "; ".join(f"fold {k} {x['steps']} steps, loss {x['loss_first']} -> {x['loss_last']}, "
                                              f"{x['s_per_step']} s/step" for k, x in rep["training"].items()))
    lines += ["", f"Rule: {rep['rule']}",
              f"Reading: A7G {r['A7G']} - A7B {r['A7B']} = {r['difference']}; A7G only {r['arm_only']}, A7B only "
              f"{r['other_only']}; p {pv}; band {r['band']}",
              f"Decision: {rep['decision']}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RUNS / AGENT / "r1_results.jsonl")
    ap.add_argument("--texts", type=Path, default=LI1 / "texts.jsonl")
    ap.add_argument("--runs-root", type=Path, default=RUNS)
    a = ap.parse_args(argv)
    rep = analyze(a.results, a.texts, a.runs_root)
    print(format_report(rep))
    if a.results.parent.is_dir() and "reading" in rep:
        (a.results.parent / "li1_decision.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
