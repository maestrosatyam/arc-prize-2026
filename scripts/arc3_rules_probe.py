"""Probe ARC-AGI-3 game rules on the competition engine (arc_agi OFFLINE) and settle disputed rule claims.

Used to engine-check rule statements written from game source (experiments/arc3-rules/). A disagreement
between a checker and the rules file is resolved by running the claim's recorded check, or any other action
sequence, on the engine.

Subcommands:
  extract  --parquet-dir D --out F.json
           Per game, pick the shortest human winning session that replays frame-exactly on the engine
           (experiments/arc3-human-data/replay_check_won.json), else the shortest whose win reproduces;
           replay it and store its actions split by the engine's levels_completed.
  human    GAME [--level N]
           Print the stored human action sequence per level (or for level N).
  run      GAME --level N [--actions A] [--via-human] [--show] [--show-every]
           Reach level N (set_level(N-1) on a fresh game, or replay the stored human prefix of levels
           1..N-1 with --via-human, which keeps cross-level state and the true levels_completed), apply the
           actions, print per-action changed-cell summaries (count, bbox, colour transitions a->b:n),
           levels_completed and state.
  claim    ID|PREFIX|all [--rules experiments/arc3-rules/rules.json] [--show] [--show-every]
           Run the engine check recorded for a claim id in rules.json (e.g. tr87.L6.k1; a prefix such as
           tr87.L6 runs all of that level's claims) and judge it against the recorded expectation
           (level_up, no_level_up, game_over, no_game_over; otherwise print the trace for inspection).

Action tokens (comma separated): 1-5 and 7 keys; 6:x:y click at column x, row y; R (or 0) reset;
H = the stored human sequence for the level; KxN = token K repeated N times (e.g. 3x10, 6:12:40x5).
Env: ARC3_ENV_DIR (environment_files dir), ARC3_HUMAN_JSON (output of extract), ARC3_RULES (rules path).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

ENV_DIR = os.environ.get("ARC3_ENV_DIR", "")
HUMAN_JSON = os.environ.get("ARC3_HUMAN_JSON", "")
VERSIONS = {
    "ar25": "0c556536", "bp35": "0a0ad940", "cn04": "2fe56bfb", "g50t": "5849a774", "ka59": "38d34dbb",
    "lf52": "271a04aa", "ls20": "9607627b", "m0r0": "492f87ba", "r11l": "495a7899", "re86": "8af5384d",
    "s5i5": "18d95033", "sc25": "635fd71a", "sk48": "d8078629", "sp80": "589a99af", "su15": "1944f8ab",
    "tn36": "ef4dde99", "tr87": "cd924810", "tu93": "0768757b", "wa30": "ee6fef47",
    "cd82": "fb555c5d", "dc22": "fdcac232", "ft09": "0d8bbf25", "lp85": "305b61c3", "sb26": "7fbdac44",
    "vc33": "5430563c",
}


def arcade():
    import arc_agi
    logging.disable(logging.WARNING)
    if not ENV_DIR:
        sys.exit("set ARC3_ENV_DIR to the competition environment_files directory")
    return arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_DIR)


def parse_action(tok: str):
    from arcengine import GameAction
    tok = tok.strip()
    if tok.upper() in ("R", "0"):
        return GameAction.RESET, None
    parts = tok.split(":")
    a = GameAction.from_id(int(parts[0]))
    data = {"x": int(parts[1]), "y": int(parts[2])} if len(parts) == 3 else None
    return a, data


def fmt(frame) -> str:
    return "\n".join("".join("0123456789abcdef"[v] if 0 <= v < 16 else "?" for v in row) for row in frame)


def summarize(prev, cur) -> str:
    if prev is None or cur is None:
        return ""
    if prev.shape != cur.shape:
        return f"shape {prev.shape}->{cur.shape}"
    d = np.argwhere(prev != cur)
    if len(d) == 0:
        return "no change"
    (y0, x0), (y1, x1) = d.min(0), d.max(0)
    pairs = {}
    for y, x in d:
        k = (int(prev[y, x]), int(cur[y, x]))
        pairs[k] = pairs.get(k, 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:6]
    return f"{len(d)} cells bbox x{x0}-{x1} y{y0}-{y1} " + " ".join(f"{a}->{b}:{n}" for (a, b), n in top)


def load_human(game: str):
    if not HUMAN_JSON or not os.path.exists(HUMAN_JSON):
        sys.exit("set ARC3_HUMAN_JSON to the output of `extract`")
    return json.load(open(HUMAN_JSON))[game]


def cmd_extract(args):
    import glob
    import pyarrow.parquet as pq
    ok = json.load(open(args.replay_json))
    best = {}
    for r in ok:
        if not r["won"] or r["state_replayed"] != "WIN":
            continue
        key = (0 if r["ok"] else 1, r["steps"])
        if r["env"] not in best or key < best[r["env"]][1]:
            best[r["env"]] = (r["guid"], key)
    want = {v[0]: k for k, v in best.items()}
    arc = arcade()
    out = {}
    for f in sorted(glob.glob(f"{args.parquet_dir}/*.parquet")):
        for r in pq.read_table(f, columns=["env", "game_id", "guid", "trajectory"]).to_pylist():
            if r["guid"] not in want:
                continue
            env = arc.make(r["game_id"])
            env.reset()
            levels, cur, lc = {}, [], 0
            for e in json.loads(r["trajectory"])[1:]:
                ai = e["data"].get("action_input") or {}
                raw = str(ai.get("id"))
                if not raw.isdigit():
                    from arcengine import GameAction
                    raw = str(GameAction.from_name(raw).value)
                xy = ai.get("data") or {}
                tok = raw if not ("x" in xy and "y" in xy) else f"{raw}:{xy['x']}:{xy['y']}"
                a, data = parse_action(tok)
                fr = env.step(a, data=data)
                cur.append(tok)
                if fr is not None and int(fr.levels_completed) > lc:
                    levels[str(lc + 1)] = cur
                    cur, lc = [], int(fr.levels_completed)
            if cur:
                levels[str(lc + 1) + "_tail"] = cur
            out[r["env"]] = {"guid": r["guid"], "game_id": r["game_id"], "frame_exact": best[r["env"]][1][0] == 0,
                             "levels_won_on_replay": lc, "levels": levels}
    json.dump(out, open(args.out, "w"))
    print(f"{len(out)} games -> {args.out}")
    for k in sorted(out):
        print(k, "frame_exact" if out[k]["frame_exact"] else "win-reproducing", out[k]["levels_won_on_replay"])


def cmd_human(args):
    h = load_human(args.game)
    print("session", h["guid"], "frame_exact" if h.get("frame_exact", True) else "win-reproducing")
    for k, v in h["levels"].items():
        if args.level and k != str(args.level):
            continue
        print(f"L{k} ({len(v)}): {','.join(v)}")


def reach(game: str, level: int, via_human: bool, show: bool = False):
    """Return (env, frame, board) positioned at the start of `level`."""
    env = arcade().make(f"{game}-{VERSIONS[game]}")
    f = env.reset()
    prev = np.array(f.frame[-1])
    if level > 1:
        if via_human:
            h = load_human(game)
            for L in range(1, level):
                for tok in h["levels"][str(L)]:
                    a, data = parse_action(tok)
                    f = env.step(a, data=data)
            prev = np.array(f.frame[-1])
            print(f"start L{level} via human prefix ({h['guid'][:8]}): levels_completed={f.levels_completed} "
                  f"state={f.state.name} level_index={env._game.level_index}")
            if int(f.levels_completed) != level - 1:
                print("WARNING: human prefix did not reach this level")
        else:
            env._game.set_level(level - 1)
            g = env._game
            prev = np.array(g.camera.render(g.current_level.get_sprites()))
            print(f"start L{level} via set_level({level - 1}) (levels_completed reads 0; cross-level state not carried)")
    else:
        print(f"start L1: state={f.state.name} available={f.available_actions}")
    if show:
        print(fmt(prev))
    return env, f, prev


def expand(game: str, level: int, actions: str) -> list[str]:
    toks = []
    for t in (actions.split(",") if actions else []):
        t = t.strip()
        if not t:
            continue
        if t.upper() == "H":
            toks += load_human(game)["levels"][str(level)]
        elif "x" in t and t.rsplit("x", 1)[1].isdigit():  # 3x10, 6:12:40x5
            k, n = t.rsplit("x", 1)
            toks += [k] * int(n)
        else:
            toks.append(t)
    return toks


def play(env, f, prev, toks, show_every=False):
    """Apply tokens, print per-step diffs; return (levels_completed before, [(lc, state)] per step, board)."""
    trace = []
    lc0 = int(getattr(f, "levels_completed", 0) or 0)
    for i, tok in enumerate(toks, 1):
        a, data = parse_action(tok)
        f = env.step(a, data=data)
        if f is None or not f.frame:
            st = getattr(getattr(f, "state", None), "name", None)
            print(f"{i:3} {tok:>10}  (no frame) state={st}")
            trace.append((trace[-1][0] if trace else lc0, st))
            continue
        cur = np.array(f.frame[-1])
        print(f"{i:3} {tok:>10}  lc={f.levels_completed} {f.state.name:13} frames={len(f.frame)} "
              f"{summarize(prev, cur)}")
        if show_every:
            print(fmt(cur))
        prev = cur
        trace.append((int(f.levels_completed), f.state.name))
    return lc0, trace, prev


def cmd_run(args):
    env, f, prev = reach(args.game, args.level, args.via_human, args.show)
    toks = expand(args.game, args.level, args.actions)
    _, _, prev = play(env, f, prev, toks, args.show_every)
    if args.show and toks:
        print(fmt(prev))


def load_claims(path: str) -> dict:
    """Claims from rules.json ({game: {claims: [...]}}) or a directory of per-game fragments."""
    games = {}
    if os.path.isdir(path):
        import glob
        for fp in glob.glob(f"{path}/*.json"):
            games[os.path.basename(fp)[:-5]] = json.load(open(fp))
    else:
        games = json.load(open(path))
    return {c["id"]: c for g in games.values() for c in g.get("claims", [])}


def judge(expect: str, lc0: int, trace) -> str:
    if not expect or not trace:
        return "inspect"
    lcs = [t[0] for t in trace]
    states = [t[1] for t in trace]
    up = [i for i, v in enumerate(lcs, 1) if v > lc0] or [i for i, s in enumerate(states, 1) if s == "WIN"]
    if expect == "level_up":
        if up and up[0] == len(trace):
            return "PASS"
        return f"FAIL (level up earlier, at step {up[0]})" if up else "FAIL (no level up)"
    if expect == "no_level_up":
        return "PASS" if not up else f"FAIL (level up at step {up[0]})"
    if expect == "game_over":
        return "PASS" if states[-1] == "GAME_OVER" else f"FAIL (final state {states[-1]})"
    if expect == "no_game_over":
        return "PASS" if "GAME_OVER" not in states else f"FAIL (GAME_OVER at step {states.index('GAME_OVER') + 1})"
    return "inspect"


def cmd_claim(args):
    claims = load_claims(args.rules)
    ids = sorted(c for c in claims if args.claim == "all" or c == args.claim or c.startswith(args.claim + "."))
    if not ids:
        sys.exit(f"no claim {args.claim!r} in {args.rules}")
    res = []
    for cid in ids:
        c = claims[cid]
        chk = c.get("check")
        print(f"\n== {cid} [{c.get('status')}] {c.get('text', '')}\n   source {c.get('lines', '')}")
        if not chk:
            print("   no engine check recorded (source-only)")
            res.append((cid, "source-only"))
            continue
        game = cid.split(".")[0]
        level = int(chk.get("level", 1))
        env, f, prev = reach(game, level, chk.get("start", "set_level") == "human", args.show)
        lc0, trace, _ = play(env, f, prev, expand(game, level, chk.get("actions", "")), args.show_every)
        v = judge(chk.get("expect", ""), lc0, trace)
        if chk.get("note"):
            print("   look for:", chk["note"])
        print(f"   expect {chk.get('expect') or '-'}: {v}")
        res.append((cid, v))
    if len(res) > 1:
        print("\nsummary:")
        for cid, v in res:
            print(f"  {cid}: {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--parquet-dir", required=True)
    e.add_argument("--replay-json", default="experiments/arc3-human-data/replay_check_won.json")
    e.add_argument("--out", required=True)
    h = sp.add_parser("human")
    h.add_argument("game")
    h.add_argument("--level", type=int, default=0)
    r = sp.add_parser("run")
    r.add_argument("game")
    r.add_argument("--level", type=int, default=1)
    r.add_argument("--actions", default="")
    r.add_argument("--via-human", action="store_true")
    r.add_argument("--show", action="store_true")
    r.add_argument("--show-every", action="store_true")
    c = sp.add_parser("claim")
    c.add_argument("claim", help="claim id (e.g. sk48.L3.k1), a prefix (sk48.L3), or 'all'")
    c.add_argument("--rules", default=os.environ.get("ARC3_RULES", "experiments/arc3-rules/rules.json"),
                   help="rules.json or a directory of per-game fragments")
    c.add_argument("--show", action="store_true")
    c.add_argument("--show-every", action="store_true")
    a = ap.parse_args()
    {"extract": cmd_extract, "human": cmd_human, "run": cmd_run, "claim": cmd_claim}[a.cmd](a)


if __name__ == "__main__":
    main()
