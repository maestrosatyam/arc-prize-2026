"""Pick the TTT learning rate of one arm on meta-val tasks (never held-out): one leave-one-pair-out fold per task,
mean query NLL after the largest step count of the grid; writes runs/<arm>/ttt_lr.json.

    OMP_NUM_THREADS=2 nice -n 19 ../../envs/mamba/bin/python tune_ttt.py --arm mamba
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arcssm.data import load_split, load_training, max_side  # noqa: E402
from arcssm.infer import finetune, query_nll  # noqa: E402
from arcssm.util import atomic_write_json, limit_threads  # noqa: E402
from evaluate import load_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lrs", default="3e-4,1e-3,3e-3")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-dir", default=None, help="default runs/<arm>")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args(argv)
    limit_threads(args.threads)
    run = Path(args.run_dir) if args.run_dir else HERE / "runs" / args.arm
    if (run / "ttt_lr.json").exists():
        return
    model = load_model(run / "model.pt", args.device)
    model.grad_ckpt = args.grad_ckpt
    split, tasks = load_split(), load_training()
    ids = [t for t in split["meta_val"] if max_side(tasks[t]) <= args.max_side and len(tasks[t]["train"]) >= 3][: args.n_tasks]
    t0 = time.process_time()
    table = {}
    for lr in [float(x) for x in args.lrs.split(",")]:
        vals = []
        for tid in ids:
            pairs = tasks[tid]["train"]
            rest, (q_in, q_out) = pairs[:-1], pairs[-1]
            seed = int.from_bytes(tid.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
            torch.manual_seed(seed)
            base = query_nll(model, rest, q_in, q_out)
            m = finetune(model, rest, args.steps, np.random.default_rng(seed), lr, args.bs)
            vals.append((base, query_nll(m, rest, q_in, q_out)))
        table[lr] = {"nll_before": float(np.mean([a for a, _ in vals])), "nll_after": float(np.mean([b for _, b in vals]))}
    best = min(table, key=lambda k: table[k]["nll_after"])
    atomic_write_json(run / "ttt_lr.json", {"lr": best, "table": {str(k): v for k, v in table.items()}, "tasks": ids,
                                            "steps": args.steps, "bs": args.bs, "cpu_s": time.process_time() - t0})


if __name__ == "__main__":
    main()
