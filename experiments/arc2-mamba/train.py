"""Meta-training of one arm under a fixed budget, resumable from runs/<arm>/ckpt.pt.

    OMP_NUM_THREADS=2 nice -n 19 ../../envs/mamba/bin/python train.py --arm mamba --cpu-budget 7200
    python train.py --arm hybrid_g --device cuda --budget-kind wall --cpu-budget 10800 --workers 4 --no-load-guard

CPU runs: budget and learning-rate schedule in process CPU seconds (both threads counted), so arms get equal CPU
time whatever the machine load. GPU runs (--budget-kind wall): the same in wall seconds on identical GPUs, bfloat16
autocast, task generation in DataLoader worker processes. Checkpoints and logs are written atomically; a killed
CPU run resumes where its last checkpoint left off (the data stream restarts from the saved RNG state).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arcssm.configs import ARMS  # noqa: E402
from arcssm.data import Mixture, load_split, load_training, max_side  # noqa: E402
from arcssm.encoding import collate, encode  # noqa: E402
from arcssm.generators import FAMILIES, make_task  # noqa: E402
from arcssm.infer import amp, device_of, to_dev  # noqa: E402
from arcssm.models import build, n_params  # noqa: E402
from arcssm.util import LoadGuard, append_jsonl, atomic_torch_save, atomic_write_json, limit_threads, rss_mb  # noqa: E402


def loss_from_arrays(model, arrays):
    dev = device_of(model)
    tok, row, col, role, tgt, mask = to_dev(arrays, dev)
    with amp(dev):
        logits = model(tok[:, :-1], row[:, :-1], col[:, :-1], role[:, :-1])
    lp = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
    m = mask.float()
    return (lp * m).sum() / m.sum().clamp(min=1), int(mask.sum())


def masked_loss(model, batch_encs):
    return loss_from_arrays(model, collate(batch_encs))


def val_sets(split, tasks, max_len=2048, n_gen=160, seed=12345, max_side_real=15):
    rng = np.random.default_rng(seed)
    gen = []
    fams = list(FAMILIES)
    for i in range(n_gen):
        fam = fams[i % len(fams)]
        pairs = make_task(rng, fam, 4, int(rng.integers(4, 11)))
        if pairs is None:
            continue
        gen.append(encode(pairs[:-1], pairs[-1][0], pairs[-1][1]))
    real = []
    for tid in split["meta_val"]:
        t = tasks[tid]
        if max_side(t) > max_side_real or len(t["train"]) < 2:
            continue
        e = encode(t["train"][:-1], t["train"][-1][0], t["train"][-1][1])
        if len(e) <= max_len:
            real.append(e)
    return {"gen": gen, "meta_val": real}


@torch.no_grad()
def evaluate_tf(model, encs, bs=8):
    """Teacher-forced loss and accuracy on the query output only (tokens after prefix_len)."""
    model.eval()
    dev = device_of(model)
    tot_lp, tot_n, tot_ok, exact = 0.0, 0, 0, 0
    encs = sorted(encs, key=len)
    for i in range(0, len(encs), bs):
        chunk = encs[i:i + bs]
        tok, row, col, role, tgt, mask = to_dev(collate(chunk), dev)
        with amp(dev):
            logits = model(tok[:, :-1], row[:, :-1], col[:, :-1], role[:, :-1])
        logits = logits.float()
        pred = logits.argmax(-1)
        for j, e in enumerate(chunk):
            sl = slice(e.prefix_len - 1, len(e) - 1)
            lp = F.cross_entropy(logits[j, sl], tgt[j, sl], reduction="sum").item()
            ok = (pred[j, sl] == tgt[j, sl])
            tot_lp += lp
            tot_n += ok.numel()
            tot_ok += int(ok.sum())
            exact += int(ok.all())
    model.train()
    return {"loss": tot_lp / max(tot_n, 1), "tok_acc": tot_ok / max(tot_n, 1), "tf_exact": exact / max(len(encs), 1),
            "n": len(encs)}


def lr_at(frac, base, warm=0.02, floor=0.1):
    if frac < warm:
        return base * frac / warm
    p = min(1.0, (frac - warm) / (1 - warm))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


class _Stream(torch.utils.data.IterableDataset):
    """Training batches generated in DataLoader workers (GPU runs); each worker has its own seed."""

    def __init__(self, real, seed, p_gen, max_len, max_side_gen, max_side_real, tokens_per_batch):
        super().__init__()
        self.args = (real, seed, p_gen, max_len, max_side_gen, max_side_real, tokens_per_batch)

    def __iter__(self):
        real, seed, p_gen, max_len, msg, msr, tpb = self.args
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        mix = Mixture(real, seed=seed * 1000 + 17 * wid + 1, p_gen=p_gen, max_len=max_len, max_side_gen=msg,
                      max_side_real=msr)
        for items in mix.batches(tpb):
            arrays = collate([e for _, e in items])
            yield [s for s, _ in items], sum(len(e) for _, e in items), arrays


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--cpu-budget", type=float, required=True, help="seconds of the budget kind")
    ap.add_argument("--budget-kind", choices=["cpu", "wall", "tokens"], default="cpu")
    ap.add_argument("--max-wall", type=float, default=None, help="stop after this many wall seconds whatever the budget")
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute blocks in backward (GPU memory)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tokens-per-batch", type=int, default=6144)
    ap.add_argument("--p-gen", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-side-gen", type=int, default=12)
    ap.add_argument("--max-side-real", type=int, default=15)
    ap.add_argument("--ckpt-every", type=float, default=600.0, help="budget seconds; <= 0 disables resume checkpoints")
    ap.add_argument("--val-every", type=float, default=900.0, help="budget seconds")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for task generation (0: in process)")
    ap.add_argument("--no-load-guard", action="store_true")
    args = ap.parse_args(argv)

    limit_threads(args.threads)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out or HERE / "runs" / args.arm)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log = lambda msg: append_jsonl(log_path, {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "msg": msg})

    split = load_split()
    tasks = load_training()
    real = {t: tasks[t] for t in split["meta_train"]}
    cfg = ARMS[args.arm]
    torch.manual_seed(args.seed)
    model = build(cfg).to(dev)
    model.grad_ckpt = args.grad_ckpt
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    mix = Mixture(real, seed=args.seed, p_gen=args.p_gen, max_len=args.max_len, max_side_gen=args.max_side_gen,
                  max_side_real=args.max_side_real)
    state = {"step": 0, "cpu_used": 0.0, "tokens": 0, "loss_tokens": 0, "seqs": 0, "paused_s": 0.0, "wall_used": 0.0,
             "src_counts": {}}
    ck = out / "ckpt.pt"
    if ck.exists():
        blob = torch.load(ck, weights_only=False, map_location=dev)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        state = blob["state"]
        mix.rng.bit_generator.state = blob["mix_rng"]
        torch.set_rng_state(blob["torch_rng"])
        log(f"resumed at step {state['step']} cpu {state['cpu_used']:.0f}")
    else:
        log(f"start arm={args.arm} params={n_params(model)} cfg={json.dumps(cfg.to_dict())} args={vars(args)}")
    meta = {"arm": args.arm, "params": n_params(model), "cfg": cfg.to_dict(), "args": vars(args),
            "split_sha256": split["sha256"], "torch": torch.__version__, "threads": torch.get_num_threads(),
            "device": str(dev), "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    atomic_write_json(out / "meta.json", meta)

    vs = val_sets(split, tasks, args.max_len, max_side_real=args.max_side_real)
    used_key = {"cpu": "cpu_used", "wall": "wall_used", "tokens": "tokens"}[args.budget_kind]

    def save():
        if args.ckpt_every <= 0:
            return
        atomic_torch_save(ck, {"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                               "mix_rng": mix.rng.bit_generator.state, "torch_rng": torch.get_rng_state()})

    def validate():
        r = {k: evaluate_tf(model, v) for k, v in vs.items()}
        append_jsonl(out / "val_log.jsonl", {"step": state["step"], "cpu_used": state["cpu_used"],
                                             "wall_used": state["wall_used"], "tokens": state["tokens"], **r})
        return r

    guard = None if args.no_load_guard else LoadGuard(log=log)
    model.train()
    last_ck = last_val = state[used_key]
    if state["step"] == 0:
        validate()
    c0, w0 = time.process_time(), time.monotonic()
    base_cpu, base_wall = state["cpu_used"], state["wall_used"]
    run_loss, run_n, log_w0, log_tok0 = 0.0, 0, time.monotonic(), state["tokens"]
    if args.workers > 0:
        stream = _Stream(real, args.seed, args.p_gen, args.max_len, args.max_side_gen, args.max_side_real,
                         args.tokens_per_batch)
        batches = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                                   pin_memory=dev.type == "cuda", prefetch_factor=4,
                                                   persistent_workers=True))
    else:
        batches = ((([s for s, _ in items]), sum(len(e) for _, e in items), collate([e for _, e in items]))
                   for items in mix.batches(args.tokens_per_batch))
    while state[used_key] < args.cpu_budget:
        if args.max_wall is not None and state["wall_used"] >= args.max_wall:
            state["stopped_by_max_wall"] = True
            log(f"max wall {args.max_wall}s reached at {state[used_key]} of {args.cpu_budget} {args.budget_kind}")
            break
        if guard is not None:
            state["paused_s"] += guard.check(on_pause=save)
        srcs, n_tok, arrays = next(batches)
        for s in srcs:
            state["src_counts"][s] = state["src_counts"].get(s, 0) + 1
        frac = state[used_key] / args.cpu_budget
        for g in opt.param_groups:
            g["lr"] = lr_at(frac, args.lr)
        loss, n_loss = loss_from_arrays(model, arrays)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state["step"] += 1
        state["seqs"] += len(srcs)
        state["tokens"] += n_tok
        state["loss_tokens"] += n_loss
        state["cpu_used"] = base_cpu + time.process_time() - c0
        state["wall_used"] = base_wall + time.monotonic() - w0
        run_loss += loss.item() * n_loss
        run_n += n_loss
        if state["step"] % 50 == 0:
            now = time.monotonic()
            append_jsonl(log_path, {"step": state["step"], "cpu_used": round(state["cpu_used"], 1),
                                    "wall_used": round(state["wall_used"], 1), "tokens": state["tokens"],
                                    "loss": run_loss / max(run_n, 1), "lr": opt.param_groups[0]["lr"],
                                    "tok_per_s": round((state["tokens"] - log_tok0) / max(now - log_w0, 1e-9)),
                                    "rss_mb": round(rss_mb()), "load5": round(os.getloadavg()[1], 2),
                                    **({"gpu_mem_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20)}
                                       if dev.type == "cuda" else {})})
            run_loss, run_n, log_w0, log_tok0 = 0.0, 0, now, state["tokens"]
        if state[used_key] - last_val >= args.val_every:
            validate()
            last_val = state[used_key]
        if state[used_key] - last_ck >= args.ckpt_every > 0:
            save()
            last_ck = state[used_key]
    save()
    final = validate()
    atomic_torch_save(out / "model.pt", {"cfg": cfg.to_dict(), "model": {k: v.cpu() for k, v in model.state_dict().items()},
                                         "state": state})
    atomic_write_json(out / "train_summary.json", {**meta, "state": state, "final_val": final})
    log("done")


if __name__ == "__main__":
    main()
