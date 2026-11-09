#!/usr/bin/env python3
"""Prepare the inputs of the R1 paired recovery experiment from the P0 logs.

Subcommands
  cases         cut points, replay prefixes, conversation restores and sandbox
                preludes for the failed cases -> experiments/arc3-recovery/cases/
  replay-check  replay every prefix offline and compare board and level with the
                recorded events -> experiments/arc3-recovery/replay_check.json
  bundle        pack the admitted cases into one gzip+base64 text file for the
                notebook -> kaggle/arc3/r1/r1_cases.tar.gz.b64

Inputs
  experiments/arc3-failure-annotation/cases.csv        the sampled cases (outcome F)
  <raw>/<game>_p<k>_requests.jsonl                     P0 model requests
  <raw>/artifacts/<game>_p<k>_events.jsonl             P0 per-action boards
  <env-dir>/<game4>/<hash>/{metadata.json,<game4>.py}  competition environment files

Cut rule (DESIGN.md): A_start = actions executed before the failed level began,
n = actions spent on the failed level, cut_nominal = A_start + floor(n / 2)
(A_start when n < 4). The restore record is the last opener request
(request_index_within_turn == 1) whose action count is <= cut_nominal; the
replay prefix ends at that record's action count, so the restored conversation
and the restored board agree.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_CSV = ROOT / "experiments" / "arc3-failure-annotation" / "cases.csv"
OUT_DIR = ROOT / "experiments" / "arc3-recovery"
CASES_DIR = OUT_DIR / "cases"
BUNDLE_PATH = ROOT / "kaggle" / "arc3" / "r1" / "r1_cases.tar.gz.b64"
DEFAULT_HARNESS_SRC = ROOT / "third_party" / "duck-harness"

IMAGE_PLACEHOLDER = "[frame image omitted in restore]"
MIN_SPENT_FOR_HALF = 4
MOUSE_RE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")


# ----------------------------------------------------------------------------
# inputs

def read_failed_cases() -> list[dict]:
    with open(CASES_CSV) as f:
        return [r for r in csv.DictReader(f) if r["outcome"] == "F"]


def load_events(path: Path) -> list[dict]:
    evs = [json.loads(line) for line in open(path)]
    for e in evs:
        for k in ("action_num", "analysis_step", "level"):
            if isinstance(e.get(k), str):
                e[k] = int(float(e[k]))
    return evs


def iter_requests(path: Path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def action_from_event(e: dict) -> dict:
    name = str(e.get("action_name") or "")
    data: dict = {}
    if name == "ACTION6":
        m = MOUSE_RE.search(str(e.get("action_display") or ""))
        if not m:
            raise ValueError(f"cannot parse mouse coordinates from {e.get('action_display')!r}")
        data = {"x": int(m.group(2)), "y": int(m.group(1))}
    return {"action_num": int(e["action_num"]), "name": name, "data": data,
            "display": e.get("action_display"), "level_after": int(e["level"]),
            "level_completed": bool(e.get("level_completed")), "game_over": bool(e.get("game_over"))}


# ----------------------------------------------------------------------------
# cut point

def cut_point(actions: list[dict], level: int) -> dict:
    """A_start, n_spent and cut_nominal for the failed level from the action events."""
    before = [a for a in actions if a["level_before"] < level]
    on_level = [a for a in actions if a["level_before"] == level]
    a_start = len(before)
    n_spent = len(on_level)
    if n_spent < MIN_SPENT_FOR_HALF:
        cut = a_start
    else:
        cut = a_start + n_spent // 2
    return {"a_start": a_start, "n_spent": n_spent, "cut_nominal": cut}


def annotate_level_before(actions: list[dict]) -> None:
    prev = 1
    for a in actions:
        a["level_before"] = prev
        prev = a["level_after"]


# ----------------------------------------------------------------------------
# restore

def strip_images(messages: list[dict]) -> tuple[list[dict], int]:
    out = []
    n = 0
    for m in messages:
        m = json.loads(json.dumps(m))
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    n += 1
                    parts.append({"type": "text", "text": IMAGE_PLACEHOLDER})
                else:
                    parts.append(p)
            m["content"] = parts
        out.append(m)
    return out, n


def tool_call_codes(messages: list[dict]) -> list[str]:
    codes = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") != "python":
                continue
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except (json.JSONDecodeError, TypeError):
                continue
            code = args.get("code")
            if isinstance(code, str) and code.strip():
                codes.append(code)
    return codes


def build_prelude(codes: list[str]) -> tuple[str, list[str]]:
    """Imports and the latest top-level definition of every function/class/lambda name."""
    imports: list[str] = []
    seen_imports: set[str] = set()
    defs: dict[str, str] = {}
    order: list[str] = []
    for code in codes:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                src = ast.get_source_segment(code, node) or ast.unparse(node)
                if src not in seen_imports:
                    seen_imports.add(src)
                    imports.append(src)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                src = ast.get_source_segment(code, node) or ast.unparse(node)
                if node.name not in defs:
                    order.append(node.name)
                defs[node.name] = src
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda) \
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                src = ast.get_source_segment(code, node) or ast.unparse(node)
                if name not in defs:
                    order.append(name)
                defs[name] = src
    parts = []
    if imports:
        # the sandbox refuses some modules (time, builtins, ...); one refused
        # import must not abort the definitions that follow it
        parts.append("\n".join(
            "try:\n" + "\n".join("    " + line for line in src.split("\n")) + "\nexcept Exception:\n    pass"
            for src in imports
        ))
    for name in order:
        parts.append(defs[name])
    text = "\n\n".join(parts) + ("\n" if parts else "")
    return text, order


def find_restore(records: list[dict], cut_nominal: int) -> dict | None:
    """Last opener request whose action count (record action - 1) is <= cut_nominal."""
    best = None
    for r in records:
        if r.get("event") != "request":
            continue
        if int(r.get("request_index_within_turn") or 0) != 1:
            continue
        count = int(r.get("action") or 1) - 1
        if count <= cut_nominal:
            best = r
    return best


def cmd_cases(args) -> None:
    raw = Path(args.raw)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for case in read_failed_cases():
        cid, game_id, p, level = case["case_id"], case["game_id"], int(case["pass"]), int(case["level"])
        ev_path = raw / "artifacts" / f"{game_id}_p{p}_events.jsonl"
        rq_path = raw / f"{game_id}_p{p}_requests.jsonl"
        if not ev_path.exists() or not rq_path.exists():
            print(f"skip {cid}: missing logs", file=sys.stderr)
            continue
        evs = load_events(ev_path)
        actions = [action_from_event(e) for e in evs if e.get("type") == "action"]
        annotate_level_before(actions)
        cp = cut_point(actions, level)

        # the restore record and the prelude need the request stream up to it
        restore = None
        restore_pos = -1
        per_record_codes: list[list[str]] = []
        for r in iter_requests(rq_path):
            if r.get("event") != "request":
                continue
            count = int(r.get("action") or 1) - 1
            if count > cp["cut_nominal"]:
                break
            per_record_codes.append(tool_call_codes(r.get("messages") or []))
            if int(r.get("request_index_within_turn") or 0) == 1:
                restore = r
                restore_pos = len(per_record_codes) - 1
        if restore is None:
            print(f"skip {cid}: no opener request at or before action {cp['cut_nominal']}", file=sys.stderr)
            continue
        # prior python calls only: the opener record's messages hold every call
        # made before the restored turn; later records belong to the dropped turn
        codes: list[str] = []
        seen_codes: set[str] = set()
        for code in (c for lst in per_record_codes[:restore_pos + 1] for c in lst):
            if code not in seen_codes:
                seen_codes.add(code)
                codes.append(code)
        cut = int(restore["action"]) - 1
        msgs = restore["messages"]
        assert msgs and msgs[0].get("role") == "system", cid
        assert msgs[-1].get("role") == "user", cid
        history, n_images = strip_images(msgs[1:-1])
        # only the codes the restored history could have run: everything before
        # the restore opener. The loop above already stopped at that opener's
        # action count; codes from the opener record itself are prior calls.
        prelude, names = build_prelude(codes)
        prefix = [a for a in actions if a["action_num"] <= cut]
        assert [a["action_num"] for a in prefix] == list(range(1, cut + 1)), cid
        at_cut = next((e for e in evs if e.get("type") == "action" and int(e["action_num"]) == cut), None)
        if at_cut is None:
            at_cut = next(e for e in evs if e.get("type") == "initial")
        board = at_cut.get("board")
        board_hash = hashlib.sha256(json.dumps(board).encode()).hexdigest()[:16]
        level_at_cut = int(at_cut.get("level"))

        d = CASES_DIR / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "prefix_actions.json").write_text(json.dumps(
            [{"action_num": a["action_num"], "name": a["name"], "data": a["data"]} for a in prefix]) + "\n")
        (d / "restore.json").write_text(json.dumps(history) + "\n")
        (d / "prelude.py").write_text(prelude)
        meta = {
            "case_id": cid, "game_id": game_id, "pass": p, "target_level": level,
            "a_start": cp["a_start"], "n_spent": cp["n_spent"], "cut_nominal": cp["cut_nominal"],
            "cut": cut, "prefix_len": len(prefix), "actions_on_level_in_prefix": max(0, cut - cp["a_start"]),
            "restore_analysis_step": restore.get("analysis_step"),
            "restore_history_messages": len(history), "restore_images_omitted": n_images,
            "restore_bytes": len(json.dumps(history)),
            "prelude_names": names, "prelude_bytes": len(prelude),
            "level_at_cut": level_at_cut, "board_hash_at_cut": board_hash,
            "pass_total_actions": len(actions),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        index.append(meta)
        print(f"{cid}: A_start={cp['a_start']} n={cp['n_spent']} cut_nominal={cp['cut_nominal']} cut={cut} "
              f"level_at_cut={level_at_cut} history={len(history)} msgs ({meta['restore_bytes']} B) "
              f"prelude={len(names)} defs")
    (OUT_DIR / "cases_index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {len(index)} cases under {CASES_DIR}")


# ----------------------------------------------------------------------------
# replay check

def _import_harness(src: Path):
    for sub in ("tufa-arc-agi-framework/src", "ARC3-Inference"):
        p = str(src / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import arc_agi  # noqa: F401
    import arcengine  # noqa: F401
    import taaf.game_api  # noqa: F401
    return arc_agi, arcengine, taaf.game_api


def make_offline_game(game_api, arc_agi, env_dir: str, game_id: str):
    spec = game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    return game_api.GameAPI(env_name=game_id, arcade_spec=spec)


def grid_of(state) -> list[list[int]]:
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return [[int(c) for c in row] for row in rows]


def level_of(game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def replay_prefix(game, arcengine, prefix: list[dict], on_step=None):
    for a in prefix:
        action = arcengine.ActionInput(id=getattr(arcengine.GameAction, a["name"]), data=dict(a["data"]))
        game.execute_action(action, generated_tokens=0, uncached_input_tokens=0)
        if on_step is not None:
            on_step(a, game)


def cmd_replay_check(args) -> None:
    arc_agi, arcengine, game_api = _import_harness(Path(args.harness_src))
    raw = Path(args.raw)
    os.environ.setdefault("RECORDINGS_DIR", str(Path(args.recordings_dir)))
    session = __import__("taaf.game", fromlist=["RunSession"]).RunSession(record_intermediate_states=False)
    results = []
    for d in sorted(CASES_DIR.iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        prefix = json.loads((d / "prefix_actions.json").read_text())
        evs = load_events(raw / "artifacts" / f"{meta['game_id']}_p{meta['pass']}_events.jsonl")
        boards = {int(e["action_num"]): (e.get("board"), int(e["level"])) for e in evs if e.get("type") == "action"}
        init = next(e for e in evs if e.get("type") == "initial")
        boards[0] = (init.get("board"), int(init.get("level")))
        game = make_offline_game(game_api, arc_agi, args.env_dir, meta["game_id"])
        t0 = time.monotonic()
        game.start_game(session)
        rec = {"case_id": meta["case_id"], "game_id": meta["game_id"], "cut": meta["cut"],
               "match": False, "first_divergent_action": None, "initial_match": grid_of(game.current_state) == boards[0][0],
               "error": None}
        if not rec["initial_match"]:
            rec["first_divergent_action"] = 0
        divergent = [None]

        def check(a, g):
            if divergent[0] is not None:
                return
            want = boards.get(a["action_num"])
            if want is None:
                return
            if grid_of(g.current_state) != want[0] or level_of(g) != want[1]:
                divergent[0] = a["action_num"]

        try:
            replay_prefix(game, arcengine, prefix, on_step=check)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        if rec["first_divergent_action"] is None:
            rec["first_divergent_action"] = divergent[0]
        want_board, want_level = boards[meta["cut"]]
        final_ok = rec["error"] is None and grid_of(game.current_state) == want_board and level_of(game) == want_level
        rec["match"] = bool(final_ok and rec["first_divergent_action"] is None)
        rec["level_at_cut"] = level_of(game)
        rec["replay_seconds"] = round(time.monotonic() - t0, 2)
        try:
            game.finish_game()
        except Exception:  # noqa: BLE001
            pass
        results.append(rec)
        print(f"{rec['case_id']}: match={rec['match']} first_divergent={rec['first_divergent_action']} "
              f"err={rec['error']} ({rec['replay_seconds']} s)")
    n_ok = sum(r["match"] for r in results)
    out = {
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arc_agi_version": getattr(arc_agi, "__version__", None) or _dist_version("arc-agi"),
        "arcengine_version": _dist_version("arcengine"),
        "env_dir": args.env_dir,
        "n_cases": len(results), "n_match": n_ok,
        "excluded": [r["case_id"] for r in results if not r["match"]],
        "cases": results,
    }
    (OUT_DIR / "replay_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"replay check: {n_ok}/{len(results)} match; excluded {out['excluded']}")


def _dist_version(name: str) -> str | None:
    try:
        import importlib.metadata as m
        return m.version(name)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# bundle

def cmd_bundle(args) -> None:
    check = json.loads((OUT_DIR / "replay_check.json").read_text())
    admitted = [c["case_id"] for c in check["cases"] if c["match"]]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for cid in admitted:
            d = CASES_DIR / cid
            for name in ("meta.json", "prefix_actions.json", "restore.json", "prelude.py"):
                tar.add(d / name, arcname=f"{cid}/{name}")
    gz = gzip.compress(buf.getvalue(), compresslevel=9)
    b64 = base64.b64encode(gz).decode("ascii")
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.write_text("\n".join(b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n")
    sha = hashlib.sha256(gz).hexdigest()
    manifest = {"cases": admitted, "n_cases": len(admitted), "tar_gz_bytes": len(gz), "b64_bytes": len(b64),
                "tar_gz_sha256": sha}
    (OUT_DIR / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"bundle: {len(admitted)} cases, {len(gz)} B gz, {len(b64)} B b64, sha256 {sha}")
    if len(b64) > 15 * 1024 * 1024:
        print("bundle exceeds 15 MB", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--raw", required=True)
    c.set_defaults(fn=cmd_cases)
    r = sub.add_parser("replay-check")
    r.add_argument("--raw", required=True)
    r.add_argument("--env-dir", required=True)
    r.add_argument("--harness-src", default=str(DEFAULT_HARNESS_SRC))
    r.add_argument("--recordings-dir", default="/tmp/arc3_recovery_recordings")
    r.set_defaults(fn=cmd_replay_check)
    b = sub.add_parser("bundle")
    b.set_defaults(fn=cmd_bundle)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
