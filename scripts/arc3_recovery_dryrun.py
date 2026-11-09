#!/usr/bin/env python3
"""CPU dry run of the R1 plumbing: replay -> restore -> play with a scripted analyzer.

No model is involved. The real ToolAgent runs with its chat call replaced by a
script that cycles through an inspection call, a 10-action batch and a snippet
that overruns the tool timeout, so the batch cap (A1), the timeout
cancellation (A2) and the budget stop are all exercised through the real
sandbox, step_env and session code.

  python scripts/arc3_recovery_dryrun.py --harness-src /tmp/p18/src \
      --env-dir /tmp/p18/compx/environment_files --out /tmp/p18/dryrun

--harness-src is a tree with Franzen's patch and the R1 patch applied
(kaggle/arc3/r1/build_r1_notebook.py --patched-tree writes one). Exit code 0
only if every check passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "experiments" / "arc3-recovery" / "cases"

# P0's analyzer configuration (notebook cell 4), minus serving and scheduling.
P0_ENV = {
    "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:9/v1",
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": "dryrun",
    "INFERENCE_ANALYZER_MODEL": "dryrun",
    "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    # P0 uses 30; the dry run shortens every arm's tool timeout so the overrun
    # snippet times out in seconds (A2's own value is set via --tool-timeout)
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "6",
    "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str((116 + 12) * 1024),
    "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024),
    "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024),
    "ARC3_HISTORY_ASSISTANT_TURNS": "150",
    "ARC3_HISTORY_TURN_DRAIN": "30",
    "ARC3_HISTORY_DRAIN_COALESCE": "1",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    "ARC3_MIDDLE_TRUNCATION": "1",
    "ARC3_ACTION_ECHO": "1",
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_ACTION_INFO": "1",
    "ARC3_DEDUPE_MULTICALL_LINE": "1",
    "ARC3_FRAME_DIFF_HINT": "1",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "10",
    "ARC3_DIFF_IMAGE": "1",
    "ARC3_GAMEOVER_DIFF_IMAGE": "1",
    "ARC3_ANIMATION": "1",
    "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_BATCH_NOOP_BLOCK": "1",
    "ARC3_STALE_STATE_BLOCK": "1",
    "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1",
    "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
    "EXPOSE_UNDO": "on",
    "ONLY_RESET_LEVELS": "true",
    "ARC3_MAX_ACTIVE_STREAMS": "0",
    "ARC3_WARMUP_ACTION_GAMES": "0",
    "MPLBACKEND": "Agg",
}

INSPECT = "print(current_frame.ascii[:120])\nprint(valid_actions)\n"
# a batch of valid moves, alternating so a wall does not no-op the whole batch
PICK = (
    "_moves = [a for a in valid_actions if a in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE')]\n"
    "if _moves:\n"
    "    _batch = [_moves[i % len(_moves)] for i in range(10)]\n"
    "else:\n"
    "    _batch = [{'action': 'MOUSE', 'row': 20 + 3 * i, 'col': 20 + 3 * i} for i in range(10)]\n"
)
# one 10-action batch per call; two such calls in a row with no read-back
# between them, so the second one is what the A1 cap truncates (a second
# action() inside the same snippet would be refused by P0's stale-state guard
# whenever the first batch ended on a no-op)
BATCH = PICK + (
    "r = action(_batch)\n"
    "print(r.get('executed_count'), r.get('stop_reason'))\n"
)
# about 9 s of CPU on this machine when idle: past the dry-run tool timeout
OVERRUN = PICK + (
    "r = action(_batch[:1])\n"
    "s = 0\n"
    "for i in range(160000000):\n"
    "    s += i\n"
    "action(_batch[:2])\n"
)
SCRIPT = [INSPECT, BATCH, BATCH, OVERRUN, INSPECT, BATCH]


def _import_harness(src: Path) -> None:
    for sub in ("tufa-arc-agi-framework/src", "ARC3-Inference"):
        p = str(src / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness-src", required=True)
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--cases-dir", default=str(CASES_DIR))
    ap.add_argument("--out", default="/tmp/arc3_recovery_dryrun")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: first two)")
    ap.add_argument("--action-budget", type=int, default=24)
    ap.add_argument("--wall-budget", type=float, default=120.0)
    ap.add_argument("--tool-timeout", type=int, default=6, help="A2 tool timeout for the dry run")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    for k, v in P0_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("RECORDINGS_DIR", str(Path(args.out) / "recordings"))
    _import_harness(Path(args.harness_src))

    from inference.agent import r1_recovery as r1
    from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult
    from inference.framework import r1_runner
    from inference.framework.solver import HarnessSolver

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cases = r1_runner.load_cases_dir(Path(args.cases_dir))
    ids = [c for c in args.cases.split(",") if c] or sorted(cases)[:2]
    cases = {cid: cases[cid] for cid in ids}

    class ScriptedAgent(ToolAgent):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._script_pos = 0
            self._script_calls = 0

        def _chat_completion(self, messages, *, tools, request_timeout_seconds=None, max_tokens=None, enable_thinking=None):
            code = SCRIPT[self._script_pos % len(SCRIPT)]
            self._script_pos += 1
            self._script_calls += 1
            call = {"id": f"call_{self._script_calls}", "type": "function",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})}}
            return _ChatCompletionResult(
                message={"role": "assistant", "content": "", "reasoning_content": "scripted", "tool_calls": [call]},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 1000, "completion_tokens": 40, "total_tokens": 1040},
                served_by="dryrun",
            )

    solver = HarnessSolver(label="r1-dryrun", model="dryrun", analyzer_timeout=30.0, save_request_logs=True,
                           concurrency=args.concurrency)
    solver.job_dir = out
    solver.setup()
    arm_flags = {k: dict(v) for k, v in r1.ARM_FLAGS.items()}
    arm_flags["A2"]["ARC3_R1_TOOL_TIMEOUT"] = str(args.tool_timeout)
    arm_flags["A2"]["ARC3_R1_TIMEOUT_CANCEL"] = "1"

    def factory(game, index):
        return ScriptedAgent(model="dryrun", timeout=30.0, save_request_logs=True, dispatch_index=index)

    t0 = time.monotonic()
    results = r1_runner.run_pilot(
        solver=solver, cases=cases, job_dir=out, env_dir=args.env_dir, results_path=out / "r1_results.jsonl",
        analyzer_factory=factory, concurrency=args.concurrency, action_budget=args.action_budget,
        wall_budget_s=args.wall_budget, hard_stop_monotonic=time.monotonic() + 20 * 60, arm_flags=arm_flags,
    )
    solver.teardown()
    print(r1_runner.summary_table(results))
    print(f"dry run wall {time.monotonic() - t0:.0f} s")

    # checks
    failures = []
    by = {(r["case_id"], r["arm"]): r for r in results}
    if len(results) != len(cases) * 3:
        failures.append(f"expected {len(cases) * 3} results, got {len(results)}")
    for (cid, arm), r in by.items():
        if r.get("error"):
            failures.append(f"{cid} {arm}: error {str(r['error'])[:300]}")
        if not r.get("replay_ok"):
            failures.append(f"{cid} {arm}: replay not ok")
        if r.get("end_reason") not in ("action_budget", "cleared", "wall_budget"):
            failures.append(f"{cid} {arm}: end_reason {r.get('end_reason')}")
        if arm in ("A0", "A2") and int(r.get("capped_batches") or 0):
            failures.append(f"{cid} {arm}: batch cap fired outside A1")
        if arm == "A1" and int(r.get("notice_shown") or 0) != 1:
            failures.append(f"{cid} {arm}: notice shown {r.get('notice_shown')} times")
        if arm == "A1" and int(r.get("capped_batches") or 0) < 1:
            failures.append(f"{cid} {arm}: no capped batch")
        if arm != "A1" and int(r.get("notice_shown") or 0):
            failures.append(f"{cid} {arm}: notice shown outside A1")
        if int(r.get("timeouts") or 0) < 1:
            failures.append(f"{cid} {arm}: no tool timeout observed")
        if arm == "A2" and int(r.get("timeouts") or 0) and int(r.get("kept_functions") or 0) == 0:
            failures.append(f"{cid} {arm}: no prelude functions seeded")
        if int(r.get("restore_history_messages") or 0) < 1:
            failures.append(f"{cid} {arm}: history not restored")
        stem = f"{cid}_{arm}"
        for rel in (f"{stem}_requests.jsonl", f"artifacts/{stem}_events.jsonl", f"transcripts/{stem}.txt"):
            if not (out / rel).exists():
                failures.append(f"{cid} {arm}: missing {rel}")
        req_path = out / f"{stem}_requests.jsonl"
        if req_path.exists():
            first = json.loads(req_path.read_text().splitlines()[0])
            msgs = first["messages"]
            if len(msgs) < int(r.get("restore_history_messages") or 0) + 2:
                failures.append(f"{cid} {arm}: first request has {len(msgs)} messages, restore had {r.get('restore_history_messages')}")
            text = json.dumps(msgs[-1])
            if arm == "A1" and r1.PROBE_NOTICE[:40] not in text:
                failures.append(f"{cid} {arm}: probe notice absent from first opener")
            if arm != "A1" and r1.PROBE_NOTICE[:40] in text:
                failures.append(f"{cid} {arm}: probe notice present")
        ev_path = out / "artifacts" / f"{stem}_events.jsonl"
        if ev_path.exists():
            evs = [json.loads(l) for l in ev_path.read_text().splitlines() if l.strip()]
            acts = [e for e in evs if e.get("type") == "action"]
            if not acts or acts[0].get("action_num") != 1 or evs[0].get("type") != "initial":
                failures.append(f"{cid} {arm}: events sidecar does not start with the replayed prefix")
            if acts and acts[-1].get("action_num") != r["cut"] + r["actions_used"]:
                failures.append(f"{cid} {arm}: events end at {acts[-1].get('action_num')}, expected {r['cut'] + r['actions_used']}")
    a2 = [r for r in results if r["arm"] == "A2"]
    if a2 and not any(int(r.get("cancelled_actions") or 0) for r in a2):
        print("note: no action was cancelled at the A2 deadline in this run (timing-dependent)")
    if failures:
        print("DRY RUN FAILED")
        for f in failures:
            print(" -", f)
        return 1
    print("DRY RUN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
