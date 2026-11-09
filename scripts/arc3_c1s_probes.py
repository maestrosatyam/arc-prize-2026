#!/usr/bin/env python3
"""C1S probe-cost list (run a3-c1s-selfinsight, arm A7SF): per C1 level start, LI1's generation input and the actions its
one-step probes would cost in a real run.

Rule: reports/research/ARC3_C1S_SELF_INSIGHT_RULE_2026-10-29.md (frozen at 99ea2c9; 2026-10-29
14:25 IST), section "Probe cost (fixed now)". Committed with its hash before any push.

Per level of C1's list (experiments/arc3-c1/levels.jsonl, 7d7abd6; 157 levels):
  - the level start replayed from RESET on the competition engine with LI1's code (scripts/arc3_li1_records.py:
    replay, then the board hash and level checked against the C1 row);
  - LI1's observe and prompt_of, unchanged: the available actions, each applied once from a fresh copy of the start,
    MOUSE at the centres of the 8 largest non-background components;
  - probe cost = one action per probe, plus one RESET after each probe that changed the frame or the level state, i.e.
    every effect line other than "<probe>: no change" (a line with no changed cell but a level completed or a game over
    carries a flag and counts as changed); RESETs count as actions.
The prompt itself holds the board as text and is not written; its sha256 is, so the kernel can check that it builds the
same input. Where LI1 built a record for the same level start (156 of the 157), the prompt must equal that record's.

Outputs (experiments/arc3-c1s/): probe_cost.jsonl (one row per level in C1's list order: case_id, game, level, kernel,
n_levels, probes, changed, probe_cost, prompt_sha256, prompt_chars, li1_record), check.json (counts, the probe-cost
spread, hashes).

Usage: envs/arc3/bin/python -I scripts/arc3_c1s_probes.py --env-dir ~/data/arc3/environment_files
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("arc3_li1_records", ROOT / "scripts" / "arc3_li1_records.py")
LI1 = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("arc3_li1_records", LI1)
_spec.loader.exec_module(LI1)

OUT = ROOT / "experiments" / "arc3-c1s"
C1_LEVELS = ROOT / "experiments" / "arc3-c1" / "levels.jsonl"
C1_LEVELS_SHA256 = "1c8a0dff01f4a5ec07068f2ca5640ee60e34fa672734095911b498c51a21d6ab"
LI1_RECORDS = ROOT / "experiments" / "arc3-li1" / "records.jsonl"
LI1_MANIFEST = ROOT / "experiments" / "arc3-li1" / "MANIFEST.json"
KERNELS = ("k1", "k2", "k3")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_cost(effects: list[str]) -> tuple[int, int]:
    """(probe cost, probes that changed the frame or the level state): one action per probe, plus a RESET after each
    changed one."""
    changed = sum(0 if line.endswith(": no change") else 1 for line in effects)
    return len(effects) + changed, changed


def list_sha256(rows: list[dict]) -> str:
    """The list as the run uses it: [case_id, probe_cost, prompt_sha256] in C1's list order."""
    return hashlib.sha256(json.dumps([[r["case_id"], r["probe_cost"], r["prompt_sha256"]] for r in rows])
                          .encode("utf-8")).hexdigest()


def build(env_dir: str) -> dict:
    assert sha256_file(C1_LEVELS) == C1_LEVELS_SHA256, "C1's levels.jsonl differs from the committed list"
    levels = [json.loads(l) for l in C1_LEVELS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(levels) == 157
    li1 = {}
    if LI1_RECORDS.is_file():
        man = json.loads(LI1_MANIFEST.read_text(encoding="utf-8"))
        assert sha256_file(LI1_RECORDS) == man["files"]["records.jsonl"], "LI1 records.jsonl differs from its manifest"
        li1 = {r["id"]: r for r in map(json.loads, LI1_RECORDS.read_text(encoding="utf-8").splitlines())}

    os.environ.setdefault("ONLY_RESET_LEVELS", "true")
    import logging
    logging.disable(logging.WARNING)
    import arc_agi
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))

    rows, problems = [], []
    for r in levels:
        env, raw = LI1.replay(arcade, r["game_id"], r["prefix_actions"])
        grid = [[int(v) for v in row] for row in raw.frame[-1]]
        board = hashlib.sha256(json.dumps(grid).encode()).hexdigest()[:16]
        if board != r["board_hash_at_cut"] or LI1.C1L.fnv1a(grid) != r["frame_fnv"] \
                or int(raw.levels_completed) != r["level"] - 1 or int(raw.win_levels) != r["n_levels"]:
            problems.append(f"{r['case_id']}: the level start does not replay to its hash")
            continue
        obs = LI1.observe(env, raw)
        prompt = LI1.prompt_of(r["level"], int(raw.win_levels), obs)
        cost, changed = probe_cost(obs["effects"])
        rec = li1.get(f"{r['game']}_L{r['level']}")
        same = None if rec is None else rec["prompt"] == prompt
        if same is False:
            problems.append(f"{r['case_id']}: prompt differs from LI1's record for the same level start")
        rows.append({"case_id": r["case_id"], "game": r["game"], "level": r["level"], "kernel": r["kernel"],
                     "n_levels": int(raw.win_levels), "probes": len(obs["effects"]), "changed": changed,
                     "probe_cost": cost, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                     "prompt_chars": len(prompt), "li1_record": "equal" if same else "none"})
    if problems:
        raise SystemExit("probe-cost list not written:\n  " + "\n  ".join(problems))
    return {"rows": rows, "li1_records_present": bool(li1)}


def check(res: dict) -> dict:
    rows = res["rows"]
    costs = [r["probe_cost"] for r in rows]
    by_kernel = collections.Counter(r["kernel"] for r in rows)
    by_game = collections.defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r["probe_cost"])
    return {
        "schema": "arc3_c1s_probe_cost.v1", "rule": "reports/research/ARC3_C1S_SELF_INSIGHT_RULE_2026-10-29.md (99ea2c9)",
        "levels": len(rows), "kernels": dict(sorted(by_kernel.items())),
        "probe_cost": {"median": statistics.median(costs), "min": min(costs), "max": max(costs), "sum": sum(costs),
                       "zero": sum(c == 0 for c in costs), "over_80": sum(c > 80 for c in costs)},
        "probes": {"median": statistics.median(r["probes"] for r in rows), "max": max(r["probes"] for r in rows)},
        "changed_share": round(sum(r["changed"] for r in rows) / max(1, sum(r["probes"] for r in rows)), 3),
        "by_game": {g: {"levels": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)}
                    for g, v in sorted(by_game.items())},
        "li1_record_equal": sum(r["li1_record"] == "equal" for r in rows),
        "li1_record_none": sorted(r["case_id"] for r in rows if r["li1_record"] == "none"),
        "inputs": {"experiments/arc3-c1/levels.jsonl": C1_LEVELS_SHA256,
                   "scripts/arc3_li1_records.py": sha256_file(ROOT / "scripts" / "arc3_li1_records.py")},
        "list_sha256": list_sha256(rows),
    }


def write(res: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_cost.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in res["rows"]),
                                          encoding="utf-8")
    chk = check(res)
    chk["probe_cost_jsonl_sha256"] = sha256_file(out / "probe_cost.jsonl")
    (out / "check.json").write_text(json.dumps(chk, indent=1) + "\n", encoding="utf-8")
    return chk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    chk = write(build(str(Path(a.env_dir).expanduser())), Path(a.out))
    print(json.dumps({k: v for k, v in chk.items() if k != "by_game"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
