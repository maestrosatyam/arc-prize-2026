#!/usr/bin/env python3
"""Analyse the 27B output bank (kaggle/arc2/bank/q27: Qwen3.8-27B direct answers, two samples per task on the 30
shortest ARC-AGI-2 evaluation tasks) and compare it with the earlier direct runs of the same model on the tasks they
share (reports/EVAL_REVIEW_2026-10-26.md, proposal P2b).

Inputs
  --q27d DIR          the bank's q27d/ directory (arc2_q27_direct.v1 records: <task_id>.json, _run.json)
  --compare NAME=DIR  an earlier direct run's q27d/ directory (repeatable; default: the Qwen 27B direct runs under
                      experiments/runs that exist on this checkout)
  --solutions FILE    ground truth (default: _kaggle_solutions_audit_tasks.json in the bank directory)
  --json OUT          write the full result

Reports (denominator: the test outputs of Kaggle's evaluation file for the bank tasks)
  samples   per sample: outputs right, finish (stop / length), status (answered / rescued / no grid), tokens
  tasks     per task: outputs right in sample 0, sample 1, their union, and in the submitted attempts (attempt_1 /
            attempt_2 after de-duplication); the within-task split (both samples right / one / neither)
  compare   on the outputs shared with each earlier run: that run's samples right against the bank's two; all Qwen
            27B samples pooled per output (a first per-output success-rate estimate) and the oracle union

Usage: python scripts/arc2_q27_bank_analyze.py --q27d experiments/runs/kaggle-A2-q27-bank-v1/q27d [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_score import grids_equal  # noqa: E402

KAGGLE_COPY = "_kaggle_solutions_audit_tasks.json"
# Earlier direct runs of the same model (Qwen3.8-27B-FP8, vLLM): one or two samples on 20 or 40 tasks each.
DEFAULT_COMPARE = {
    "Q27C": "experiments/runs/kaggle-A2-Q27C-v1/q27d",
    "Q27D": "experiments/runs/kaggle-A2-Q27D-v1/q27d",
    "Q27F": "experiments/runs/kaggle-A2-Q27F-direct-v1/q27d",
    "Q27G": "experiments/runs/kaggle-A2-Q27G-cap90-v1/q27d",
    "Q27H": "experiments/runs/kaggle-A2-Q27H-nocache-v1/q27d",
    "Q27N": "experiments/runs/kaggle-A2-Q27N-nomtp-v1/q27d",
    "Q27T": "experiments/runs/kaggle-A2-Q27T-quality-v1/q27d",
}


def load_records(q27d: Path) -> tuple[dict[str, dict], dict]:
    docs: dict[str, dict] = {}
    for p in sorted(q27d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("schema") == "arc2_q27_direct.v1":
            docs[d.get("task_id", p.stem)] = d
    run = json.loads((q27d / "_run.json").read_text(encoding="utf-8")) if (q27d / "_run.json").is_file() else {}
    return docs, run


def sample_right(sample: dict, solutions: list) -> list[bool]:
    grids = sample.get("grids") or []
    return [bool(i < len(grids) and grids[i] is not None and grids_equal(grids[i], sol)) for i, sol in enumerate(solutions)]


def analyse_bank(docs: dict[str, dict], solutions: dict[str, list]) -> dict[str, Any]:
    tasks = []
    samples = []
    n_outputs = 0
    right_by_sample: dict[int, int] = {}
    union = 0
    attempts_right = 0
    attempt_1_right = 0
    split = {"both": 0, "one": 0, "neither": 0}
    for tid in sorted(docs, key=lambda t: docs[t].get("order", 0)):
        d = docs[tid]
        sol = solutions.get(tid)
        if sol is None:
            continue
        n_outputs += len(sol)
        per_sample = {}
        for s in sorted(d.get("samples", []), key=lambda x: x.get("sample", 0)):
            r = sample_right(s, sol)
            per_sample[s.get("sample", 0)] = r
            right_by_sample[s.get("sample", 0)] = right_by_sample.get(s.get("sample", 0), 0) + sum(r)
            samples.append({"task_id": tid, "sample": s.get("sample"), "status": s.get("status"), "finish": s.get("finish"),
                            "gen_tokens": s.get("gen_tokens"), "prompt_tokens": s.get("prompt_tokens"),
                            "n_requests": s.get("n_requests"), "wall_s": s.get("wall_s"), "right": r})
        task_union = [any(per_sample[k][i] for k in per_sample) for i in range(len(sol))]
        union += sum(task_union)
        for i in range(len(sol)):
            n_right = sum(1 for k in per_sample if per_sample[k][i])
            split["both" if n_right >= 2 else "one" if n_right == 1 else "neither"] += 1
        a1 = a2 = 0
        for o in d.get("outputs", []):
            i = o.get("test_index", 0)
            if i >= len(sol):
                continue
            r1 = grids_equal(o.get("attempt_1"), sol[i])
            r2 = grids_equal(o.get("attempt_2"), sol[i])
            a1 += r1
            a2 += r1 or r2
        attempt_1_right += a1
        attempts_right += a2
        tasks.append({"task_id": tid, "order": d.get("order"), "n_test": len(sol), "stop_reason": d.get("stop_reason"),
                      "sample_right": {str(k): sum(v) for k, v in per_sample.items()}, "union_right": sum(task_union),
                      "attempt_1_right": a1, "pass_at_2_right": a2, "tokens": (d.get("tokens") or {}).get("generated"),
                      "wall_s": d.get("wall_s"), "finish": [s.get("finish") for s in d.get("samples", [])],
                      "status": [s.get("status") for s in d.get("samples", [])]})
    gen = [s["gen_tokens"] or 0 for s in samples]
    return {
        "n_tasks": len(tasks), "n_outputs": n_outputs, "n_samples": len(samples),
        "right_by_sample": {str(k): v for k, v in sorted(right_by_sample.items())},
        "union_right": union, "attempt_1_right": attempt_1_right, "pass_at_2_right": attempts_right,
        "within_task_split": split,
        "finish": {k: sum(1 for s in samples if s["finish"] == k) for k in ("stop", "length")},
        "status": {k: sum(1 for s in samples if s["status"] == k) for k in ("answered", "rescued", "no grid")},
        "gen_tokens_total": sum(gen), "gen_tokens_mean_per_sample": (sum(gen) / len(gen)) if gen else None,
        "tasks": tasks, "samples": samples,
    }


def compare_runs(bank_docs: dict[str, dict], solutions: dict[str, list], compare: dict[str, Path]) -> dict[str, Any]:
    pooled: dict[tuple[str, int], list[tuple[str, int, bool]]] = {}
    for tid, d in bank_docs.items():
        sol = solutions.get(tid)
        if sol is None:
            continue
        for s in d.get("samples", []):
            for i, r in enumerate(sample_right(s, sol)):
                pooled.setdefault((tid, i), []).append(("bank", s.get("sample", 0), r))
    runs = {}
    for name, q27d in compare.items():
        if not q27d.is_dir():
            runs[name] = {"present": False}
            continue
        docs, run = load_records(q27d)
        shared = sorted(t for t in docs if t in bank_docs and t in solutions)
        n_out = sum(len(solutions[t]) for t in shared)
        run_right = 0
        bank_s0 = bank_s1 = 0
        agree = {"run_right_bank_both": 0, "run_right_bank_one": 0, "run_right_bank_neither": 0,
                 "run_wrong_bank_both": 0, "run_wrong_bank_one": 0, "run_wrong_bank_neither": 0}
        n_run_samples = 0
        for t in shared:
            sol = solutions[t]
            bank_rs = [sample_right(s, sol) for s in sorted(bank_docs[t].get("samples", []), key=lambda x: x.get("sample", 0))]
            for s in docs[t].get("samples", []):
                n_run_samples += 1
                r = sample_right(s, sol)
                for i in range(len(sol)):
                    pooled.setdefault((t, i), []).append((name, s.get("sample", 0), r[i]))
                    run_right += r[i]
                    k = sum(1 for br in bank_rs if br[i])
                    agree[("run_right_" if r[i] else "run_wrong_") + ("bank_both" if k >= 2 else "bank_one" if k == 1 else "bank_neither")] += 1
            if bank_rs:
                bank_s0 += sum(bank_rs[0])
                bank_s1 += sum(bank_rs[1]) if len(bank_rs) > 1 else 0
        runs[name] = {"present": True, "dir": str(q27d), "n_shared_tasks": len(shared), "n_shared_outputs": n_out,
                      "n_run_samples": n_run_samples, "run_samples_right": run_right,
                      "bank_sample_0_right": bank_s0, "bank_sample_1_right": bank_s1, "agreement": agree,
                      "run_n_samples_per_task": run.get("n_samples_per_task")}
    per_output = []
    for (tid, i), rs in sorted(pooled.items()):
        k = sum(1 for r in rs if r[2])
        per_output.append({"task_id": tid, "test_index": i, "n_samples": len(rs), "n_right": k, "p_right": k / len(rs),
                           "bank_right": sum(1 for r in rs if r[0] == "bank" and r[2]),
                           "runs_right": sorted({r[0] for r in rs if r[2]})})
    never = sum(1 for o in per_output if o["n_right"] == 0)
    always = sum(1 for o in per_output if o["n_right"] == o["n_samples"])
    mixed = len(per_output) - never - always
    oracle = sum(1 for o in per_output if o["n_right"] > 0)
    return {"runs": runs, "per_output": per_output,
            "pooled": {"n_outputs": len(per_output), "never_right": never, "always_right": always, "mixed": mixed,
                       "oracle_union_right": oracle,
                       "n_samples_total": sum(o["n_samples"] for o in per_output)}}


def summary_lines(res: dict[str, Any]) -> list[str]:
    b = res["bank"]
    lines = [f"27B bank: {b['n_tasks']} tasks, {b['n_outputs']} outputs, {b['n_samples']} samples",
             f"  outputs right: sample 0 {b['right_by_sample'].get('0')}, sample 1 {b['right_by_sample'].get('1')}, "
             f"union {b['union_right']}; submitted attempt_1 {b['attempt_1_right']}, pass@2 {b['pass_at_2_right']} / {b['n_outputs']}",
             f"  within task: both samples right {b['within_task_split']['both']}, one {b['within_task_split']['one']}, "
             f"neither {b['within_task_split']['neither']}",
             f"  finish stop {b['finish']['stop']} / length {b['finish']['length']}; status {b['status']}; "
             f"generated tokens {b['gen_tokens_total']} (mean {b['gen_tokens_mean_per_sample']:.0f} per sample)",
             "  task      n  s0 s1 un a1 p2  tokens  finish",]
    for t in b["tasks"]:
        lines.append(f"  {t['task_id']}  {t['n_test']}  {t['sample_right'].get('0', 0):2d} {t['sample_right'].get('1', 0):2d} "
                     f"{t['union_right']:2d} {t['attempt_1_right']:2d} {t['pass_at_2_right']:2d}  {t['tokens'] or 0:6d}  "
                     f"{'/'.join(str(f) for f in t['finish'])}")
    c = res["compare"]
    lines.append("earlier runs on the shared outputs (run samples right vs the bank's sample 0 / sample 1):")
    for name, r in c["runs"].items():
        if not r.get("present"):
            lines.append(f"  {name}: archive missing")
            continue
        a = r["agreement"]
        lines.append(f"  {name}: {r['n_shared_tasks']} tasks, {r['n_shared_outputs']} outputs; run {r['run_samples_right']} "
                     f"of {r['n_run_samples'] and r['n_shared_outputs'] * r['n_run_samples'] // max(r['n_shared_tasks'], 1)} "
                     f"sample-outputs right vs bank {r['bank_sample_0_right']} / {r['bank_sample_1_right']}; "
                     f"run right & bank both/one/neither {a['run_right_bank_both']}/{a['run_right_bank_one']}/{a['run_right_bank_neither']}, "
                     f"run wrong & bank both/one/neither {a['run_wrong_bank_both']}/{a['run_wrong_bank_one']}/{a['run_wrong_bank_neither']}")
    p = c["pooled"]
    lines.append(f"pooled Qwen 27B samples on the bank outputs: {p['n_samples_total']} sample-outputs over {p['n_outputs']} outputs; "
                 f"never right {p['never_right']}, always right {p['always_right']}, mixed {p['mixed']}; oracle union {p['oracle_union_right']}")
    mixed = [o for o in c["per_output"] if 0 < o["n_right"] < o["n_samples"]]
    for o in sorted(mixed, key=lambda o: -o["p_right"]):
        lines.append(f"  {o['task_id']}[{o['test_index']}] right {o['n_right']}/{o['n_samples']} (bank {o['bank_right']}/2)")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q27d", required=True)
    ap.add_argument("--compare", action="append", default=[], help="NAME=Q27D_DIR (repeatable)")
    ap.add_argument("--solutions", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    q27d = Path(args.q27d)
    docs, run = load_records(q27d)
    sol_path = Path(args.solutions) if args.solutions else q27d / KAGGLE_COPY
    if not sol_path.is_file():
        print(f"solutions file missing: {sol_path}", file=sys.stderr)
        return 2
    solutions = json.loads(sol_path.read_text(encoding="utf-8"))
    compare = {k: ROOT / v for k, v in DEFAULT_COMPARE.items()}
    if args.compare:
        compare = {}
        for item in args.compare:
            name, _, d = item.partition("=")
            compare[name] = Path(d)
    res = {"schema": "arc2_q27_bank_analysis.v1", "q27d": str(q27d), "solutions": str(sol_path),
           "run": {k: run.get(k) for k in ("finalized_utc", "n_tasks", "n_outputs", "n_samples_per_task", "tasks_complete",
                                           "tasks_answered", "generated_tokens", "agent_wall_s", "gen_tok_per_s")},
           "bank": analyse_bank(docs, solutions), "compare": compare_runs(docs, solutions, compare)}
    for line in summary_lines(res):
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
