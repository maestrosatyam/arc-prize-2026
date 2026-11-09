"""CLI.

  python -m arc_harness.run --solver vr --budget 60000 --tasks dev --backend oracle --out runs/x
  python -m arc_harness.run --solver vr --budget 33000 --kaggle /kaggle/input/... --backend vllm --deadline-s 25000

--tasks: dev | holdout | eval (all 120) | train:N (first N training tasks by size) | a json list of ids | a path
--backend: vllm (needs --url) | oracle (CPU simulator, keyed on solutions) | fixture (a jsonl of canned replies)
Writes out/samples.jsonl, out/submission.json, out/run.json, and when solutions exist out/score.json and
out/loss.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .budget import Budget, Caps
from .data import holdout_split, load_dir, load_kaggle, task_size
from .llm import OracleLLM, VLLMChat
from .loss import attribute, format_report
from .score import score_submission
from .solvers import Config, Runner

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def select_tasks(spec: str, data_root: str, kaggle_dir: str = None) -> dict:
    if kaggle_dir:
        ch = os.path.join(kaggle_dir, "arc-agi_test_challenges.json")
        sol = os.path.join(kaggle_dir, "arc-agi_test_solutions.json")
        if spec == "kaggle_test":
            return load_kaggle(ch, sol)
        if spec in ("kaggle_eval",):
            return load_kaggle(os.path.join(kaggle_dir, "arc-agi_evaluation_challenges.json"),
                               os.path.join(kaggle_dir, "arc-agi_evaluation_solutions.json"))
    ev = load_dir(os.path.join(data_root, "ARC-AGI-2/data/evaluation"))
    if spec == "eval":
        return ev
    if spec in ("dev", "holdout"):
        dev, hold = holdout_split(ev.keys())
        ids = dev if spec == "dev" else hold
        return {t: ev[t] for t in ids}
    if spec.startswith("train:"):
        tr = load_dir(os.path.join(data_root, "ARC-AGI-2/data/training"))
        n = int(spec.split(":")[1])
        ids = sorted(tr, key=lambda t: task_size(tr[t]))[:n]
        return {t: tr[t] for t in ids}
    if spec.startswith("dev:"):
        n = int(spec.split(":")[1])
        dev, _ = holdout_split(ev.keys())
        ids = sorted(dev, key=lambda t: task_size(ev[t]))[:n]
        return {t: ev[t] for t in ids}
    if os.path.exists(spec):
        with open(spec) as f:
            ids = json.load(f)
        pool = dict(ev)
        pool.update(load_dir(os.path.join(data_root, "ARC-AGI-2/data/training")))
        return {t: pool[t] for t in ids}
    raise SystemExit(f"unknown task spec {spec}")


def make_llm(args, tasks):
    if args.backend == "vllm":
        return VLLMChat(args.url, args.model, timeout_s=args.timeout_s, system=args.system or None,
                        extra=json.loads(args.extra_body) if args.extra_body else None,
                        template_kwargs=json.loads(args.template_kwargs) if args.template_kwargs else None)
    if args.backend == "oracle":
        return OracleLLM(tasks, seed=args.seed)
    raise SystemExit(args.backend)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="vr", choices=["direct", "poetiq", "vr"])
    ap.add_argument("--budget", type=int, default=60000, help="mean generated tokens per task")
    ap.add_argument("--tasks", default="dev")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--kaggle", default=None, help="Kaggle competition input dir")
    ap.add_argument("--backend", default="oracle", choices=["vllm", "oracle"])
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--system", default="", help="system prompt (e.g. an effort instruction)")
    ap.add_argument("--extra-body", default="", help='extra JSON fields for the request, e.g. {"reasoning_effort": "low"}')
    ap.add_argument("--template-kwargs", default="", help="extra chat_template_kwargs as JSON")
    ap.add_argument("--think-hint", action="store_true", help="tell the model its thinking budget in the prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=None, help="wall-clock seconds for the whole run")
    ap.add_argument("--reserve-s", type=float, default=900)
    ap.add_argument("--in-flight", type=int, default=20)
    ap.add_argument("--tokens-per-s", type=float, default=334)
    ap.add_argument("--think1", type=int, default=None)
    ap.add_argument("--think2", type=int, default=None)
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--shuffle-demos", action="store_true")
    ap.add_argument("--no-augment-s2", action="store_true", help="second samples on the plain view")
    ap.add_argument("--style", default="diagram", choices=["diagram", "plain"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--selection", default="")
    ap.add_argument("--merge", default="R0", choices=["R0", "verified_first", "harness_first", "verified_then_margin"])
    ap.add_argument("--external", default=None, help="json {task_id: [[attempt_1, attempt_2] per test]}")
    ap.add_argument("--no-validator", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    tasks = select_tasks(args.tasks, args.data_root, args.kaggle)
    llm = make_llm(args, tasks)
    overrides = {k: v for k, v in (("think1", args.think1), ("think2", args.think2)) if v}
    budget = Budget(args.budget, len(tasks), deadline=(time.time() + args.deadline_s) if args.deadline_s else None,
                    reserve_s=args.reserve_s, tokens_per_s=args.tokens_per_s, max_in_flight=args.in_flight,
                    caps=Caps.from_mean(args.budget, overrides))
    cfg = Config(solver=args.solver, temperature=args.temperature, style=args.style, shuffle_demos=args.shuffle_demos,
                 max_resamples=args.resamples, repair=not args.no_repair, augment_s2=not args.no_augment_s2, selection=args.selection, merge=args.merge,
                 use_validator=not args.no_validator, seed=args.seed, log=not args.quiet, think_hint=args.think_hint)
    external = None
    if args.external:
        with open(args.external) as f:
            external = json.load(f)
    print(f"[run] solver={args.solver} budget={args.budget} tasks={len(tasks)} outputs="
          f"{sum(len(t['test']) for t in tasks.values())} backend={llm.name} caps={budget.caps}", flush=True)
    runner = Runner(tasks, llm, budget, cfg, args.out, external)
    summary = runner.run()
    print("[run] budget:", json.dumps(summary["budget"]), flush=True)
    with open(runner.sub_path) as f:
        sub = json.load(f)
    if any("output" in te for t in tasks.values() for te in t["test"]):
        sc = score_submission(sub, tasks)
        rep = attribute(tasks, runner.samples, sub)
        with open(os.path.join(args.out, "score.json"), "w") as f:
            json.dump({k: v for k, v in sc.items() if k != "rows"} | {"rows": sc["rows"]}, f, indent=1)
        with open(os.path.join(args.out, "loss.json"), "w") as f:
            json.dump(rep, f, indent=1)
        gen = summary["budget"]["spent"]
        print(f"[score] {sc['right']}/{sc['n_outputs']} = {sc['score']:.4f}; attempt_1 right {sc['attempt1_right']}; "
              f"tasks fully right {sc['tasks_fully_right']}/{sc['n_tasks']}; "
              f"mean tokens/task {gen/len(tasks):.0f}; right per M tokens {1e6*sc['right']/max(1,gen):.1f}")
        print(format_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
