#!/usr/bin/env python3
"""R1x rule-card compliance from the run's transcripts (no GPU).

For each case-arm transcript (transcripts/<case>_A20.txt from the Kaggle output), counts the model turns and the
RULE CARD / PREDICTION lines the model wrote itself (sections other than [USER PROMPT] and [SYSTEM PROMPT]), and the
model turns that followed a failure. A failure is a reported effect of GAME_OVER, a reset, UNDO, or an action that
changed nothing. Writes one row per case with the last card and the case's key insight for coding by hand.

usage: python scripts/arc3_rulecard_compliance.py <transcripts dir> <r1_results.jsonl> <cases dir> <out.csv>
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^\[([A-Z][A-Z ]+?)(?:\]| \d|$)")
NOT_MODEL = {"USER PROMPT", "SYSTEM PROMPT"}
CARD = re.compile(r"RULE CARD\b[^:\n]*:\s*(.+)")
PRED = re.compile(r"PREDICTION\b[^:\n]*:\s*(.+)")
FAIL = re.compile(r"GAME_OVER|game over|\bRESET\b|\bUNDO\b|no effect|nothing changed|no change", re.I)


def sections(lines):
    name, buf = None, []
    for ln in lines:
        m = HEADER.match(ln)
        if m:
            if name is not None:
                yield name, buf
            name, buf = m.group(1).strip(), []
        else:
            buf.append(ln)
    if name is not None:
        yield name, buf


def scan(path: Path) -> dict:
    turns = cards = preds = turns_with_card = turns_with_pred = 0
    after_fail = after_fail_card = 0
    pending_fail = False
    last_card = ""
    for name, buf in sections(path.read_text(errors="replace").splitlines()):
        text = "\n".join(buf)
        if name == "USER PROMPT" or name.startswith("TOOL RESULT"):
            if FAIL.search(text):
                pending_fail = True
            continue
        if name in NOT_MODEL:
            continue
        if name == "THINKING":
            turns += 1
            c, p = CARD.findall(text), PRED.findall(text)
            cards += len(c)
            preds += len(p)
            turns_with_card += bool(c)
            turns_with_pred += bool(p)
            if pending_fail:
                after_fail += 1
                after_fail_card += bool(c)
                pending_fail = False
            if c:
                last_card = c[-1].strip()
        else:
            c = CARD.findall(text)
            if c:
                cards += len(c)
                last_card = c[-1].strip()
            preds += len(PRED.findall(text))
    return dict(turns=turns, cards=cards, preds=preds, turns_with_card=turns_with_card,
                turns_with_pred=turns_with_pred, after_fail=after_fail, after_fail_card=after_fail_card,
                last_card=last_card[:600])


def main(tdir: str, results: str, cases: str, out: str) -> None:
    res = {}
    for ln in Path(results).read_text().splitlines():
        r = json.loads(ln)
        if r.get("arm") == "A20":
            res[r["case_id"]] = r
    rows = []
    for case_id in sorted(res):
        t = Path(tdir) / f"{case_id}_A20.txt"
        row = {"case_id": case_id, "cleared": res[case_id].get("cleared"), **scan(t)}
        oracle = json.loads((Path(cases) / case_id / "oracle.json").read_text())
        row["insight"] = oracle["insight"]["text"]
        rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("turns", "cards", "preds", "turns_with_card", "turns_with_pred",
                                                 "after_fail", "after_fail_card")}
    print(f"cases {n}; with any card {sum(r['cards'] > 0 for r in rows)}; with any prediction {sum(r['preds'] > 0 for r in rows)}")
    print(f"model turns {tot['turns']}; with a card {tot['turns_with_card']} ({tot['turns_with_card'] / tot['turns']:.0%}); "
          f"with a prediction {tot['turns_with_pred']} ({tot['turns_with_pred'] / tot['turns']:.0%})")
    print(f"turns right after a failure {tot['after_fail']}; with a card {tot['after_fail_card']}")
    for label, sel in (("cleared", [r for r in rows if r["cleared"]]), ("not cleared", [r for r in rows if not r["cleared"]])):
        t = sum(r["turns"] for r in sel) or 1
        print(f"  {label}: {len(sel)} cases, card turns {sum(r['turns_with_card'] for r in sel) / t:.0%}, "
              f"prediction turns {sum(r['turns_with_pred'] for r in sel) / t:.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
