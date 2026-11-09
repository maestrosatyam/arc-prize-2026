"""Coding sheet for the ARC-AGI-3 think-aloud alignment (rule: experiments/arc3-think-aloud/RULE.md, section 2).

Builds, per game, the dataset's mechanism definitions (id, name, rule, introduced levels, scope) beside C1's
insight text for each of its levels. It reads `mechanisms.json` files and `experiments/arc3-c1/levels.jsonl` only:
no trajectory, annotation or outcome file is opened. The sheet holds the dataset's definition texts, so it is
written outside the repository; only the adjudicated map (mechanism ids and labels) is committed.

Usage: python -I scripts/arc3_think_aloud_sheet.py --data <think-aloud dir> --out <sheet.json outside the repo>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C1_LEVELS = ROOT / "experiments/arc3-c1/levels.jsonl"
CASES = ROOT / "experiments/arc3-recovery/cases"


def stuck_levels() -> set[tuple[str, int]]:
    out = set()
    for d in CASES.iterdir():
        m = json.loads((d / "meta.json").read_text())
        out.add((m["game_id"].split("-")[0], int(m["target_level"])))
    return out


def build(data: Path) -> dict:
    levels = [json.loads(l) for l in C1_LEVELS.open()]
    stuck = stuck_levels()
    games = {}
    for r in levels:
        g = r["game"]
        if g not in games:
            mech = json.loads((data / "games" / g / "mechanisms.json").read_text())
            games[g] = dict(game=g, n_levels=r["n_levels"], levels=[], mechanisms=[
                dict(mechanism_id=m["mechanism_id"], name=m.get("name", ""), rule=m.get("rule", ""),
                     introduced_levels=m.get("introduced_levels", []), scope=m.get("detection_scope"))
                for m in mech])
        games[g]["levels"].append(dict(level=int(r["level"]), stuck=(g, int(r["level"])) in stuck,
                                       insight=r["insight"]["text"]))
    for g in games.values():
        g["levels"].sort(key=lambda x: x["level"])
    return dict(levels=len(levels), stuck_levels=len(stuck), games=[games[k] for k in sorted(games)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    if ROOT in out.parents:
        raise SystemExit("the sheet holds the dataset's definition texts: write it outside the repository")
    sheet = build(Path(a.data).expanduser())
    blob = json.dumps(sheet, ensure_ascii=False, indent=1)
    out.write_text(blob)
    n_mech = sum(len(g["mechanisms"]) for g in sheet["games"])
    print(json.dumps(dict(games=len(sheet["games"]), levels=sheet["levels"], stuck_levels=sheet["stuck_levels"],
                          mechanisms=n_mech, sha256=hashlib.sha256(blob.encode()).hexdigest())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
