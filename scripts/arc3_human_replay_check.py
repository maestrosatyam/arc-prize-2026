"""Replay human ARC-AGI-3 recordings through the competition's game files and check them frame by frame.

Source: Hugging Face dataset magic-sword/arc_agi_3_public_demo_human_testing (parquet shards; one row per
human session; `trajectory` is a JSON list of {timestamp, data: FrameDataRaw with action_input}).
For every session on the exact game version used by the competition, replays each recorded action_input
with arc_agi (OFFLINE mode, competition environment_files) and compares the resulting frame, levels
completed and state with the recording. Prints a per-game table; --json writes per-session rows.

Usage: python scripts/arc3_human_replay_check.py --parquet-dir <dir> --env-dir <environment_files> [--won-only]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import sys

import pyarrow.parquet as pq
import arc_agi
from arcengine import GameAction


def replay(arcade, game_id: str, events: list[dict]) -> dict:
    env = arcade.make(game_id)
    if env is None:
        return {"ok": False, "error": "make failed"}
    first = env.reset()
    steps = mism = 0
    first_mismatch = None
    last = first
    for i, ev in enumerate(events[1:], start=1):
        d = ev["data"]
        ai = d.get("action_input") or {}
        raw = str(ai.get("id"))
        action = GameAction.from_id(int(raw)) if raw.isdigit() else GameAction.from_name(raw)
        data = {k: v for k, v in (ai.get("data") or {}).items() if k in ("x", "y")}
        last = env.step(action, data=data or None)
        steps += 1
        if last is None:
            return {"ok": False, "error": f"step {i} returned None", "steps": steps}
        frame = [f.tolist() if hasattr(f, "tolist") else f for f in last.frame]
        same = (frame == d["frame"] and int(last.levels_completed) == int(d["levels_completed"])
                and str(getattr(last.state, "name", last.state)) == d["state"])
        if not same:
            mism += 1
            if first_mismatch is None:
                first_mismatch = i
    rec_last = events[-1]["data"]
    return {"ok": mism == 0, "steps": steps, "mismatches": mism, "first_mismatch": first_mismatch,
            "levels_replayed": int(last.levels_completed), "levels_recorded": int(rec_last["levels_completed"]),
            "state_replayed": str(getattr(last.state, "name", last.state)), "state_recorded": rec_last["state"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--versions-from", default="experiments/runs/kaggle-P0-commit-v1/benchmark.json",
                    help="benchmark.json whose game_ids define the competition versions")
    ap.add_argument("--won-only", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    b = json.load(open(args.versions_from))
    versions = {r["game_id"] for r in (b.get("game_runs") or b)}
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=args.env_dir)
    rows = []
    for f in sorted(glob.glob(f"{args.parquet_dir}/*.parquet")):
        for r in pq.read_table(f, columns=["env", "game_id", "guid", "won", "trajectory"]).to_pylist():
            if r["game_id"] not in versions or (args.won_only and not r["won"]):
                continue
            events = json.loads(r["trajectory"])
            res = replay(arcade, r["game_id"], events)
            res.update(env=r["env"], game_id=r["game_id"], guid=r["guid"], won=r["won"])
            rows.append(res)
            print(f"{r['env']} {r['guid'][:8]} won={r['won']} ok={res['ok']} steps={res.get('steps')} "
                  f"mismatch={res.get('mismatches')} levels {res.get('levels_replayed')}/{res.get('levels_recorded')}",
                  file=sys.stderr)
    g = collections.defaultdict(lambda: [0, 0, 0])
    for x in rows:
        t = g[x["env"]]; t[0] += 1; t[1] += int(x["ok"]); t[2] += int(x["ok"] and x["won"])
    print("game  sessions  exact_replays  exact_winning_replays")
    for k in sorted(g):
        print(f"{k}  {g[k][0]:3}  {g[k][1]:3}  {g[k][2]:3}")
    print(f"total sessions {len(rows)}, exact {sum(x['ok'] for x in rows)}, winning exact "
          f"{sum(x['ok'] and x['won'] for x in rows)}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
