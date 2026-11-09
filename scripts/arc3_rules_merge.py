"""Merge per-game rule fragments and rule-discovery label drafts; check quotes; cross-tabulate.

  merge   --frag-dir D --out experiments/arc3-rules/rules.json
  labels  --frag-dir D --out experiments/arc3-failure-annotation/labels_rule_A.csv
          (concatenates drafts in cases.csv order; checks every failed case has one row, values are in the
          allowed sets, and each evidence_quote occurs verbatim in its dossier, whitespace-normalised)
  check   --labels experiments/arc3-failure-annotation/labels_rule_A.csv   (same checks, no write)
  crosstab --labels ... --other experiments/arc3-failure-annotation/labels_A.csv [--other ...labels_B.csv]
  status  --frag-dir D   (writes experiments/arc3-rules/STATUS.md)
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re

ANN = "experiments/arc3-failure-annotation"
COLS = ["case_id", "game_id", "level", "rule_discovered", "evidence_quote", "evidence_turn", "key_insight_stated",
        "first_correct_turn", "executed_after_discovery", "notes", "labeller"]
RD = ["yes", "partial", "no", "unclear"]


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def load_frags(d: str) -> dict:
    out = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        try:
            out[os.path.basename(f)[:-5]] = json.load(open(f))
        except json.JSONDecodeError as e:
            print(f"skip {f}: {e}")
    return out


def failed_cases():
    return [r for r in csv.DictReader(open(f"{ANN}/cases.csv")) if r["outcome"] == "F"]


def check(rows) -> list[str]:
    errs = []
    want = {r["case_id"] for r in failed_cases()}
    have = [r["case_id"] for r in rows]
    for c in sorted(want - set(have)):
        errs.append(f"missing row {c}")
    for c in sorted(set(have) - want):
        errs.append(f"unexpected row {c}")
    for c, n in collections.Counter(have).items():
        if n > 1:
            errs.append(f"duplicate {c}")
    for r in rows:
        c = r["case_id"]
        if r["rule_discovered"] not in RD:
            errs.append(f"{c}: rule_discovered={r['rule_discovered']!r}")
        if r["key_insight_stated"] not in ("yes", "no"):
            errs.append(f"{c}: key_insight_stated={r['key_insight_stated']!r}")
        if r["executed_after_discovery"] not in ("yes", "no", "n.a."):
            errs.append(f"{c}: executed_after_discovery={r['executed_after_discovery']!r}")
        if (r["key_insight_stated"] == "yes") != bool(r["first_correct_turn"].strip()):
            errs.append(f"{c}: key_insight_stated vs first_correct_turn inconsistent")
        if r["rule_discovered"] == "yes" and r["key_insight_stated"] != "yes":
            errs.append(f"{c}: discovered=yes but key_insight_stated=no")
        p = f"{ANN}/dossiers/{c}.md"
        q = r["evidence_quote"].strip().strip('"').strip("'")
        if q and os.path.exists(p):
            text = norm(open(p).read())
            parts = [x for x in re.split(r"\s*(?:\.\.\.|…)\s*", q) if x.strip()]
            if not all(norm(x) in text for x in parts):
                errs.append(f"{c}: quote not found verbatim: {q[:80]!r}")
    return errs


def cmd_merge(a):
    out = load_frags(a.frag_dir)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"{len(out)} games -> {a.out}")
    for g, d in out.items():
        print(f"{g}: levels {d.get('levels_written')} checks {d.get('engine_checks')}/{d.get('engine_checks_passed')} "
              f"status {d.get('status')}")


ALL_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0", "r11l",
             "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]


def cmd_status(a):
    """Write experiments/arc3-rules/STATUS.md from the fragments and the per-game .md files."""
    failed = collections.defaultdict(set)
    for r in failed_cases():
        failed[r["game_id"].split("-")[0]].add(int(r["level"]))
    frags = load_frags(a.frag_dir)
    rows, tot = [], collections.Counter()
    for g in ALL_GAMES:
        d = frags.get(g)
        md = os.path.exists(f"{a.rules_dir}/{g}.md")
        fl = sorted(failed.get(g, []))
        if not d:
            rows.append(f"| {g} | {'yes' if fl else 'no'} | {','.join(map(str, fl)) or '-'} | - | - | - | - | "
                        f"{'md only' if md else 'not started'} |")
            continue
        n = d.get("n_levels", "?")
        lw = sorted(int(x) for x in d.get("levels", {}).keys() if str(x).isdigit())
        cl = d.get("claims", [])
        eng = [c for c in cl if c.get("status") == "engine"]
        fl_done = all(x in lw for x in fl)
        tot["games"] += 1
        tot["claims"] += len(cl)
        tot["engine"] += len(eng)
        tot["levels"] += len(lw)
        tot["n_levels"] += n if isinstance(n, int) else 0
        tot["failed_done"] += int(bool(fl) and fl_done)
        rows.append(f"| {g} | {'yes' if fl else 'no'} | {','.join(map(str, fl)) or '-'} | "
                    f"{'yes' if fl_done else 'no'} | {len(lw)}/{n} | {len(cl)} | {len(eng)} | {d.get('status', '')} |")
    in_cases = sum(1 for g in ALL_GAMES if failed.get(g))
    out = [
        "# arc3-rules status",
        "",
        f"Updated with each commit. Games with a fragment: {tot['games']}/25; games in cases.csv with all failed "
        f"levels written: {tot['failed_done']}/{in_cases}; levels written {tot['levels']}/{tot['n_levels']} (over "
        f"games with a fragment); claims {tot['claims']}, engine-checked {tot['engine']}, source-only "
        f"{tot['claims'] - tot['engine']}.",
        "",
        "Claim counts come from the `claims` list in rules.json; `status` is the writer's own note. Settle a "
        "disputed claim with `python scripts/arc3_rules_probe.py claim <id>` (see README.md).",
        "",
        "| game | in cases.csv | failed levels | failed levels written | levels written | claims | engine-checked | status |",
        "|---|---|---|---|---|---|---|---|",
    ] + rows
    if a.labels and os.path.exists(a.labels):
        n = sum(1 for _ in csv.DictReader(open(a.labels)))
        out += ["", f"labels_rule_A.csv: {n}/35 failed cases labelled."]
    open(f"{a.rules_dir}/STATUS.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out[:3]))


def read_rows(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    return rows


def cmd_labels(a):
    rows = read_rows(sorted(glob.glob(f"{a.frag_dir}/*.csv")))
    order = {r["case_id"]: i for i, r in enumerate(failed_cases())}
    rows.sort(key=lambda r: order.get(r["case_id"], 999))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "").strip() for k in COLS})
    errs = check(list(csv.DictReader(open(a.out))))
    print(f"{len(rows)} rows -> {a.out}; {len(errs)} problems")
    for e in errs:
        print(" ", e)


def cmd_check(a):
    errs = check(list(csv.DictReader(open(a.labels))))
    print(f"{len(errs)} problems")
    for e in errs:
        print(" ", e)


def cmd_crosstab(a):
    lab = {r["case_id"]: r for r in csv.DictReader(open(a.labels))}
    strat = {r["case_id"]: r["stratum"] for r in failed_cases()}
    print("rule_discovered counts (n=%d):" % len(lab))
    for s in ("A", "B", None):
        sel = [r for c, r in lab.items() if s is None or strat[c] == s]
        cnt = collections.Counter(r["rule_discovered"] for r in sel)
        print(f"  stratum {s or 'all'} (n={len(sel)}): " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in RD))
    ki = collections.Counter(r["key_insight_stated"] for r in lab.values())
    ex = collections.Counter(r["executed_after_discovery"] for r in lab.values())
    print(f"  key_insight_stated: {dict(ki)}; executed_after_discovery: {dict(ex)}")
    for other in a.other:
        o = {r["case_id"]: r for r in csv.DictReader(open(other))}
        cats = sorted({o[c]["failure_category"] for c in lab if c in o})
        print(f"\n| failure_category ({os.path.basename(other)}) | " + " | ".join(RD) + " | total |")
        print("|---|" + "---|" * (len(RD) + 1))
        for cat in cats:
            cs = [c for c in lab if c in o and o[c]["failure_category"] == cat]
            cnt = collections.Counter(lab[c]["rule_discovered"] for c in cs)
            print(f"| {cat} | " + " | ".join(str(cnt.get(k, 0)) for k in RD) + f" | {len(cs)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("merge"); m.add_argument("--frag-dir", required=True); m.add_argument("--out", required=True)
    l = sp.add_parser("labels"); l.add_argument("--frag-dir", required=True); l.add_argument("--out", required=True)
    c = sp.add_parser("check"); c.add_argument("--labels", required=True)
    x = sp.add_parser("crosstab"); x.add_argument("--labels", required=True); x.add_argument("--other", action="append", default=[])
    s = sp.add_parser("status"); s.add_argument("--frag-dir", required=True)
    s.add_argument("--rules-dir", default="experiments/arc3-rules")
    s.add_argument("--labels", default=f"{ANN}/labels_rule_A.csv")
    a = ap.parse_args()
    {"status": cmd_status, "merge": cmd_merge, "labels": cmd_labels, "check": cmd_check, "crosstab": cmd_crosstab}[a.cmd](a)


if __name__ == "__main__":
    main()
