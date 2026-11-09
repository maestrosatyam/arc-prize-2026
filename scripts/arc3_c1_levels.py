"""C1 level list (run a3-c1-levelstarts): every verified level start whose key insight is engine-verified, with its
insight message, R1-format prefix and start-frame hashes, split into three kernels by game.

Rule (frozen 2026-10-29 00:12 IST, written at d72c696, stamped in 2b1525d):
reports/research/ARC3_C1_LEVEL_STARTS_RULE_2026-10-29.md. A level is in the run only if its key-insight claim
(<game>.L<n>.k1, the claim R1f's oracle maps to insight.claim_ids) has status "engine" in rules.json; the level-level
engine_checked flag is not used. Texts come from R1f's oracle pipeline (kaggle/arc3/c1/c1_oracle.py); none is edited.

Checks (any failure stops the script, nothing is written):
  - the pinned inputs: level_starts.jsonl (4d1a1cd), rules.json (75bc5e3), the R1f case bundle, the release's
    arc3_solutions.jsonl (the shortest-solution bands);
  - the pipeline reproduces the 35 R1 cases' oracle.json insight entries (message byte for byte, and every field)
    from the R1f bundle R1f and R1g ran on;
  - every one of the 183 starts replays on the competition engine (arc_agi OFFLINE, ONLY_RESET_LEVELS as the harness
    sets it) to its stated level and FNV frame hash; the R1 runner's board hash (sha256 of the frame, 16 hex) is
    taken from the same frame;
  - every kept insight message is within r1_recovery's 1,500-character cap; a level whose text the pipeline empties
    (every clause an action sequence or count) is excluded with that reason.

Outputs (experiments/arc3-c1/): levels.jsonl (one row per kept level, the run's list), excluded.jsonl (the levels
left out and why), check.json (the 35-case reproduction and the replay), LEVELS.md (counts, kernels, hashes).

Usage: envs/arc3/bin/python -I scripts/arc3_c1_levels.py --env-dir ~/data/arc3/environment_files
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import logging
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle" / "arc3" / "c1"))
import c1_oracle  # noqa: E402

OUT = ROOT / "experiments" / "arc3-c1"
LEVEL_STARTS = ROOT / "experiments" / "arc3-team-play" / "level_starts" / "level_starts.jsonl"
LEVEL_STARTS_SHA256 = "6f1a43443789f036e806fa344cfe556de724876a6792a8a31d59a7aab805281a"
RULES = ROOT / "experiments" / "arc3-rules" / "rules.json"
RULES_SHA256 = "0bce2aaf6408285b337fcd35b5ffe2fbe1b887e36379d103d77b23dcaff89784"
R1F_BUNDLE = ROOT / "kaggle" / "arc3" / "r1" / "r1f" / "dataset" / "r1_cases.tar.gz"
R1F_BUNDLE_SHA256 = "e50cdc86c06807f137bcf5b52013532cd0c8872180fc7baae5fa8da24383b303"
RELEASE = Path("~/data/arc3-play-release").expanduser()
SOLUTIONS = "data/play/derived/arc3_solutions.jsonl"

# three kernels split by game (both arms of every level in one kernel)
KERNELS = ("k1", "k2", "k3")
BANDS = ((1, 10, "1-10"), (11, 20, "11-20"), (21, 40, "21-40"), (41, 80, "41-80"), (81, 10 ** 9, "81+"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv1a(grid) -> str:
    """The release's frame hash: FNV-1a 32-bit over height, width, then every cell (scripts/arc3_team_play_levelstarts.py)."""
    h = 0x811C9DC5
    for b in [len(grid), len(grid[0])] + [int(v) for row in grid for v in row]:
        h = ((h ^ (b & 0xFF)) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def band(n: int) -> str:
    return next(name for lo, hi, name in BANDS if lo <= int(n) <= hi)


def case_id(game: str, level: int) -> str:
    return f"{game}_L{int(level)}"


def r1_prefix(prefix_actions: list[dict]) -> list[dict]:
    """The level start's actions in the R1 bundle's format. The opening RESET is the game start (r1_runner.offline_game
    starts the game; R1's prefixes begin at action 1); later RESETs are actions as recorded. ACTION6 carries x, y."""
    assert prefix_actions and prefix_actions[0]["name"] == "RESET", "a level start must begin with RESET"
    out = []
    for i, a in enumerate(prefix_actions[1:], start=1):
        data = {"x": int(a["x"]), "y": int(a["y"])} if a["name"] == "ACTION6" else {}
        out.append({"action_num": i, "name": a["name"], "data": data})
    return out


def r1f_insights() -> dict[str, dict]:
    """case id -> the insight entry of its oracle.json in the R1f bundle."""
    out = {}
    with tarfile.open(R1F_BUNDLE, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith("/oracle.json"):
                o = json.loads(tar.extractfile(m).read())
                out[m.name.split("/", 1)[0]] = o
    return out


def replay(arcade, game_id: str, prefix: list[dict]) -> dict:
    """The start frame on the engine: the game started (RESET), then the R1-format prefix."""
    from arcengine import GameAction
    env = arcade.make(game_id)
    raw = env.step(GameAction.RESET, data={}, reasoning=None)
    for a in prefix:
        raw = env.step(getattr(GameAction, a["name"]), data=dict(a.get("data") or {}), reasoning=None)
    grid = [[int(v) for v in row] for row in raw.frame[-1]]
    return {"board_hash": hashlib.sha256(json.dumps(grid).encode()).hexdigest()[:16], "fnv": fnv1a(grid),
            "level": int(raw.levels_completed) + 1, "n_levels": getattr(env, "level_count", None)}


def assign_kernels(levels: list[dict]) -> dict[str, str]:
    """game -> kernel: games by level count (most first, then name), each to the kernel with the fewest levels so
    far (then the lowest index)."""
    per_game = collections.Counter(r["game"] for r in levels)
    load = {k: 0 for k in KERNELS}
    out = {}
    for game, n in sorted(per_game.items(), key=lambda kv: (-kv[1], kv[0])):
        k = min(KERNELS, key=lambda k: (load[k], KERNELS.index(k)))
        out[game] = k
        load[k] += n
    return out


def build(env_dir: str) -> dict:
    for path, want in ((LEVEL_STARTS, LEVEL_STARTS_SHA256), (RULES, RULES_SHA256), (R1F_BUNDLE, R1F_BUNDLE_SHA256)):
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"pinned input changed: {path.relative_to(ROOT)} sha256 {got}, expected {want}")
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    starts = [json.loads(line) for line in LEVEL_STARTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    solutions_path = RELEASE / SOLUTIONS
    solutions = {(r["game"].split("-")[0], int(r["level"])): int(r["n_actions"])
                 for r in map(json.loads, solutions_path.read_text(encoding="utf-8").splitlines())}
    if len(starts) != 183 or len(solutions) != 183:
        raise SystemExit(f"expected 183 level starts and 183 solutions, got {len(starts)} and {len(solutions)}")

    # 1. the pipeline against the 35 R1 cases (the R1f bundle's oracle.json, as R1f and R1g ran)
    r1f = r1f_insights()
    repro = []
    for cid, o in sorted(r1f.items()):
        mine = c1_oracle.insight_entry(rules, o["game"], int(o["level"]))
        theirs = o["insight"]
        repro.append({"case_id": cid, "game": o["game"], "level": int(o["level"]),
                      "message_bytes_equal": mine["message"].encode("utf-8") == theirs["message"].encode("utf-8"),
                      "entry_equal": mine == theirs,
                      "message_sha256": hashlib.sha256(mine["message"].encode("utf-8")).hexdigest()})
    bad = [r["case_id"] for r in repro if not (r["message_bytes_equal"] and r["entry_equal"])]
    if len(repro) != 35 or bad:
        raise SystemExit(f"R1f reproduction failed: {len(repro)} cases, mismatched {bad}")

    # 2. every start on the engine, and the insight entry of every level
    os.environ.setdefault("ONLY_RESET_LEVELS", "true")
    logging.disable(logging.WARNING)
    import arc_agi
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(env_dir))
    kept, excluded, replays = [], [], []
    for s in starts:
        game, level = s["game"].split("-")[0], int(s["level"])
        prefix = r1_prefix(s["prefix_actions"])
        frame = replay(arcade, s["game"], prefix)
        ok = frame["fnv"] == s["hash_expected"] and frame["level"] == level
        replays.append({"case_id": case_id(game, level), "ok": ok, "fnv": frame["fnv"],
                        "fnv_expected": s["hash_expected"], "level": frame["level"]})
        if not ok:
            raise SystemExit(f"{s['game']} level {level}: engine replay gives level {frame['level']} hash "
                             f"{frame['fnv']}, level_starts has {s['hash_expected']}")
        entry = c1_oracle.insight_entry(rules, game, level)
        status = entry["claims_status"][f"{game}.L{level}.k1"]
        row = {"case_id": case_id(game, level), "game": game, "game_id": s["game"], "level": level,
               "n_levels": len(rules[game]["levels"]), "source_case": s["case"],
               "baseline_actions": int(s["baseline_actions"]),
               "solution_actions": solutions[(game, level)], "band": band(solutions[(game, level)]),
               "k1": f"{game}.L{level}.k1", "k1_status": status}
        if status != "engine":
            excluded.append({**row, "why": f"key-insight claim status {status!r}, not 'engine'"})
            continue
        if not entry["text"].strip():
            # rule 1 of the pipeline removed every clause (each held an action sequence or count): no text to show
            excluded.append({**row, "why": "R1f's pipeline leaves no insight text (every clause holds an action "
                                           "sequence or count)", "key_insight": rules[game]["levels"][str(level)]["key_insight"]})
            continue
        if len(entry["message"]) > c1_oracle.ORACLE_MAX_MESSAGE_CHARS:
            raise SystemExit(f"{row['case_id']}: insight message over {c1_oracle.ORACLE_MAX_MESSAGE_CHARS} chars")
        kept.append({**row, "cut": len(prefix), "board_hash_at_cut": frame["board_hash"], "frame_fnv": frame["fnv"],
                     "prefix_actions": prefix, "insight": entry})

    kernel_of = assign_kernels(kept)
    for r in kept:
        r["kernel"] = kernel_of[r["game"]]
    return {"kept": kept, "excluded": excluded, "repro": repro, "replays": replays,
            "solutions_sha256": sha256_file(solutions_path)}


def levels_bytes(kept: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in kept).encode("utf-8")


def texts_sha256(kept: list[dict]) -> str:
    """The insight texts as the run uses them: [case_id, message] in list order."""
    return hashlib.sha256(json.dumps([[r["case_id"], r["insight"]["message"]] for r in kept],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def write(res: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    kept, excluded = res["kept"], res["excluded"]
    lb = levels_bytes(kept)
    (out / "levels.jsonl").write_bytes(lb)
    (out / "excluded.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in excluded), encoding="utf-8")
    summary = {
        "levels": len(kept), "excluded": len(excluded), "starts": len(kept) + len(excluded),
        "levels_jsonl_sha256": hashlib.sha256(lb).hexdigest(), "texts_sha256": texts_sha256(kept),
        "inputs": {"level_starts.jsonl": LEVEL_STARTS_SHA256, "rules.json": RULES_SHA256,
                   "r1f r1_cases.tar.gz": R1F_BUNDLE_SHA256, "arc3_solutions.jsonl": res["solutions_sha256"],
                   "c1_oracle.py": sha256_file(ROOT / "kaggle" / "arc3" / "c1" / "c1_oracle.py")},
        "r1f_reproduction": {"cases": len(res["repro"]), "message_bytes_equal": sum(r["message_bytes_equal"] for r in res["repro"]),
                             "entry_equal": sum(r["entry_equal"] for r in res["repro"]), "cases_detail": res["repro"]},
        "engine_replay": {"starts": len(res["replays"]), "ok": sum(r["ok"] for r in res["replays"])},
        "kernels": {k: {"levels": sum(1 for r in kept if r["kernel"] == k), "case_arms": 2 * sum(1 for r in kept if r["kernel"] == k),
                        "games": sorted({r["game"] for r in kept if r["kernel"] == k}),
                        "prefix_actions": sum(r["cut"] for r in kept if r["kernel"] == k)} for k in KERNELS},
        "bands_kept": dict(sorted(collections.Counter(r["band"] for r in kept).items())),
        "bands_all": dict(sorted(collections.Counter(r["band"] for r in kept + excluded).items())),
        "r1_cases_in_run": sorted({f"{r['game']}_L{r['level']}" for r in res["repro"]}
                                  & {r["case_id"] for r in kept}),
    }
    (out / "check.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    res = build(str(Path(a.env_dir).expanduser()))
    s = write(res, Path(a.out))
    print(json.dumps({k: v for k, v in s.items() if k not in ("r1f_reproduction",)}, indent=1))
    print(f"r1f reproduction: {s['r1f_reproduction']['message_bytes_equal']}/35 messages byte-equal, "
          f"{s['r1f_reproduction']['entry_equal']}/35 entries equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
