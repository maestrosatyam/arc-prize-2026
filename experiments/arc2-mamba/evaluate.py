"""Evaluation of one trained arm: in-context (no TTT) and with test-time training, on

* hpair: held-out training tasks, last train pair as the query (labels are train pairs; cell metrics available),
* htest: held-out training tasks, real test inputs (predictions only; scored by scripts/arc2_eval_local.py),
* synth: the nine synthetic tasks (test outputs inline).

Results append to runs/<arm>/eval_<mode>.jsonl, one line per (set, task, test index); a restart skips lines
already written.

    OMP_NUM_THREADS=2 nice -n 19 ../../envs/mamba/bin/python evaluate.py --arm mamba --mode icl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arcssm.data import load_split, load_synthetic, load_training, max_side  # noqa: E402
from arcssm.infer import predict, score, ttt_predict  # noqa: E402
from arcssm.models import build  # noqa: E402
from arcssm.util import LoadGuard, append_jsonl, limit_threads, rss_mb  # noqa: E402


def size_bucket(task) -> str:
    s = max_side(task)
    return "<=10" if s <= 10 else ("11-15" if s <= 15 else "16-30")


def load_model(path, device="cpu"):
    blob = torch.load(path, weights_only=False, map_location="cpu")
    m = build(blob["cfg"])
    m.load_state_dict(blob["model"])
    m.to(device)
    m.eval()
    return m


def items(split, tasks, synth, sets):
    """(set, task_id, train_pairs, test_inputs, truths | None, size bucket)."""
    out = []
    if "synth" in sets:
        for k, t in synth.items():
            out.append(("synth", k, t["train"], t["test"], t["test_out"], size_bucket(t)))
    for tid in split["heldout"]:
        t = tasks[tid]
        if "hpair" in sets and len(t["train"]) >= 2:
            sub = {"train": t["train"][:-1], "test": [t["train"][-1][0]]}
            out.append(("hpair", tid, t["train"][:-1], [t["train"][-1][0]], [t["train"][-1][1]], size_bucket(sub)))
        if "htest" in sets:
            out.append(("htest", tid, t["train"], t["test"], None, size_bucket(t)))
    return out


def grid_list(g):
    return None if g is None else [[int(v) for v in r] for r in g]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", choices=["icl", "ttt"], required=True)
    ap.add_argument("--sets", default="synth,hpair,htest")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--ttt-lr", type=float, default=None, help="default: runs/<arm>/ttt_lr.json, else 1e-3")
    ap.add_argument("--ttt-bs", type=int, default=2)
    ap.add_argument("--ttt-grid", default="0,4,8,16")
    ap.add_argument("--ttt-folds", type=int, default=2)
    ap.add_argument("--max-side", type=int, default=30, help="skip tasks whose largest grid exceeds this")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-dir", default=None, help="default runs/<arm>")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N pending items")
    ap.add_argument("--deadline-s", type=float, default=None, help="stop starting new items after this many seconds")
    ap.add_argument("--no-load-guard", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="TTT: recompute blocks in backward (GPU memory)")
    ap.add_argument("--size-first", action="store_true", help="evaluate <=10, then 11-15, then 16-30 items")
    args = ap.parse_args(argv)

    limit_threads(args.threads)
    t_start = time.monotonic()
    run = Path(args.run_dir) if args.run_dir else HERE / "runs" / args.arm
    model = load_model(args.model or run / "model.pt", args.device)
    model.grad_ckpt = args.grad_ckpt
    out_path = Path(args.out or run / f"eval_{args.mode}.jsonl")
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if "set" in r:
                done.add((r["set"], r["task_id"]))
    split = load_split()
    tasks = load_training()
    synth = load_synthetic()
    guard = LoadGuard(log=lambda m: append_jsonl(out_path.with_suffix(".log"), {"msg": m}))
    grid = tuple(int(x) for x in args.ttt_grid.split(","))
    if args.ttt_lr is None:
        lr_file = run / "ttt_lr.json"
        args.ttt_lr = json.loads(lr_file.read_text())["lr"] if lr_file.exists() else 1e-3
    todo = [it for it in items(split, tasks, synth, args.sets.split(",")) if (it[0], it[1]) not in done]
    if args.size_first:
        rank = {"<=10": 0, "11-15": 1, "16-30": 2}
        todo = sorted(todo, key=lambda it: rank.get(it[5], 3))  # stable within a bucket
    if args.limit is not None:
        todo = todo[: args.limit]
    for set_name, tid, train_pairs, test_inputs, truths, bucket in todo:
        if args.deadline_s is not None and time.monotonic() - t_start > args.deadline_s:
            append_jsonl(out_path.with_suffix(".log"), {"msg": f"deadline {args.deadline_s}s reached; stopped before {set_name}:{tid}"})
            break
        if not args.no_load_guard:
            guard.check()
        rec = {"set": set_name, "task_id": tid, "bucket": bucket, "n_train": len(train_pairs),
               "n_test": len(test_inputs)}
        big = max(max(max(g.shape) for p in train_pairs for g in p), max(max(g.shape) for g in test_inputs))
        t0 = time.process_time()
        w0 = time.monotonic()
        if big > args.max_side:
            preds, info = [None] * len(test_inputs), {"skipped": f"max side {big} > {args.max_side}"}
        elif args.mode == "icl":
            preds = [predict(model, train_pairs, x, f"{tid}:{k}", args.n_aug) for k, x in enumerate(test_inputs)]
            info = {}
        else:
            seed = int.from_bytes(tid.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
            torch.manual_seed(seed)
            preds, info = ttt_predict(model, train_pairs, test_inputs, tid, np.random.default_rng(seed), args.n_aug,
                                      grid, args.ttt_folds, args.ttt_lr, args.ttt_bs)
        rec["cpu_s"] = round(time.process_time() - t0, 2)
        rec["wall_s"] = round(time.monotonic() - w0, 2)
        rec["info"] = info
        rec["outputs"] = []
        for k, p in enumerate(preds):
            o = {"test_index": k, "attempted": p is not None,
                 "top": [grid_list(g) for g in (p["top"] if p else [])],
                 "votes": p["votes"] if p else [], "n_prompts": p["n_prompts"] if p else 0}
            if truths is not None:
                o.update(score(p["top"] if p else [], truths[k]))
            rec["outputs"].append(o)
        rec["rss_mb"] = round(rss_mb())
        append_jsonl(out_path, rec)


if __name__ == "__main__":
    main()
