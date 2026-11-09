"""Teacher-forced NLL and token accuracy of the query output on hpair (held-out tasks, last train pair as query)
and on generated validation tasks, per arm; a finer comparison than exact match. Writes runs/<arm>/tf_eval.json.

    OMP_NUM_THREADS=2 nice -n 19 ../../envs/mamba/bin/python tf_eval.py --arm mamba
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arcssm.data import load_split, load_training  # noqa: E402
from arcssm.encoding import collate, encode, encoded_len  # noqa: E402
from arcssm.infer import amp, device_of, to_dev  # noqa: E402
from arcssm.util import atomic_write_json, limit_threads  # noqa: E402
from evaluate import load_model, size_bucket  # noqa: E402
from train import val_sets  # noqa: E402


@torch.no_grad()
def query_stats(model, e):
    dev = device_of(model)
    tok, row, col, role, tgt, _ = to_dev(collate([e]), dev)
    with amp(dev):
        logits = model(tok[:, :-1], row[:, :-1], col[:, :-1], role[:, :-1]).float()
    sl = slice(e.prefix_len - 1, len(e) - 1)
    lp = F.cross_entropy(logits[0, sl], tgt[0, sl], reduction="sum").item()
    ok = (logits[0, sl].argmax(-1) == tgt[0, sl])
    return lp, int(ok.numel()), int(ok.sum()), int(ok.all())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-dir", default=None, help="default runs/<arm>")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    limit_threads(args.threads)
    run = Path(args.run_dir) if args.run_dir else HERE / "runs" / args.arm
    model = load_model(run / "model.pt", args.device)
    split, tasks = load_split(), load_training()
    rows = []
    for tid in split["heldout"]:
        t = tasks[tid]
        if len(t["train"]) < 2:
            continue
        ctx, (q_in, q_out) = list(t["train"][:-1]), t["train"][-1]
        while ctx and encoded_len(ctx, q_in, q_out.shape) > args.max_len:
            ctx.pop()
        if not ctx:
            rows.append({"task_id": tid, "fits": False})
            continue
        lp, n, ok, ex = query_stats(model, encode(ctx, q_in, q_out))
        rows.append({"task_id": tid, "fits": True, "bucket": size_bucket({"train": t["train"][:-1], "test": [q_in]}),
                     "nll_sum": lp, "tokens": n, "tok_ok": ok, "tf_exact": ex})
    gen = val_sets(split, tasks, args.max_len)["gen"]
    g = [query_stats(model, e) for e in gen]

    def summ(rs):
        n = sum(r["tokens"] for r in rs)
        return {"tasks": len(rs), "tokens": n, "nll_per_token": sum(r["nll_sum"] for r in rs) / max(n, 1),
                "tok_acc": sum(r["tok_ok"] for r in rs) / max(n, 1), "tf_exact": sum(r["tf_exact"] for r in rs)}

    fit = [r for r in rows if r["fits"]]
    out = {"hpair": summ(fit), "hpair_not_fitting": sum(not r["fits"] for r in rows),
           "by_bucket": {b: summ([r for r in fit if r["bucket"] == b]) for b in ("<=10", "11-15", "16-30")},
           "gen": {"tasks": len(g), "nll_per_token": sum(x[0] for x in g) / max(sum(x[1] for x in g), 1),
                   "tok_acc": sum(x[2] for x in g) / max(sum(x[1] for x in g), 1), "tf_exact": sum(x[3] for x in g)},
           "rows": rows}
    atomic_write_json(run / "tf_eval.json", out)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
