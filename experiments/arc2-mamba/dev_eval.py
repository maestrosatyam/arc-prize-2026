"""Predictions for the 40 dev tasks (label-free challenges embedded as dev_challenges.json) from trained models found
under an input root (Kaggle: the attached output of the run that trained them). In context and with TTT, smallest
tasks first, stopping new tasks at a deadline. Writes <out>/<arm>/dev_<mode>.jsonl and dev_<mode>_pred.json
({task_id: [[attempt 1, attempt 2] per test input]}); scoring happens offline through scripts/arc2_eval_local.py
(ReplaySolver, --split dev), aggregate only.

    python dev_eval.py --models hybrid_g_s0,hybrid_g_s1 --input-root /kaggle/input --out runs/x/dev --deadline-s 9000
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
from arcssm.infer import predict, ttt_predict  # noqa: E402
from arcssm.util import append_jsonl, atomic_write_json, limit_threads  # noqa: E402
from evaluate import grid_list, load_model  # noqa: E402


def find_run(root: Path, arm: str) -> Path:
    hits = sorted(p.parent for p in root.rglob("model.pt") if p.parent.name == arm)
    if not hits:
        raise FileNotFoundError(f"no {arm}/model.pt under {root}")
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of run directory names, e.g. hybrid_g_s0")
    ap.add_argument("--input-root", default="/kaggle/input", help="searched recursively for <arm>/model.pt")
    ap.add_argument("--challenges", default=str(HERE / "dev_challenges.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, required=True, help="whole job, from start")
    ap.add_argument("--modes", default="icl,ttt")
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--ttt-grid", default="0,8,16")
    ap.add_argument("--ttt-bs", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    limit_threads(args.threads)
    t0 = time.monotonic()
    chal = json.loads(Path(args.challenges).read_text())
    tasks = {tid: {"train": [(np.asarray(p["input"], dtype=np.int8), np.asarray(p["output"], dtype=np.int8)) for p in t["train"]],
                   "test": [np.asarray(x["input"], dtype=np.int8) for x in t["test"]]} for tid, t in chal.items()}
    side = lambda t: max(max(max(g.shape) for p in t["train"] for g in p), max(max(g.shape) for g in t["test"]))
    order = sorted(tasks, key=lambda k: (side(tasks[k]), k))
    out = Path(args.out)
    models = args.models.split(",")
    summary = {"models": {}, "n_tasks": len(tasks), "n_outputs": sum(len(t["test"]) for t in tasks.values())}
    for mi, arm in enumerate(models):
        run = find_run(Path(args.input_root), arm)
        model = load_model(run / "model.pt", args.device)
        model.grad_ckpt = True
        lr_file = run / "ttt_lr.json"
        ttt_lr = json.loads(lr_file.read_text())["lr"] if lr_file.exists() else 1e-3
        summary["models"][arm] = {"source": str(run), "ttt_lr": ttt_lr}
        for mode in args.modes.split(","):
            # each (model, mode) gets an equal share of the time left at its start
            stages_left = (len(models) - mi) * len(args.modes.split(",")) - args.modes.split(",").index(mode)
            stage_deadline = time.monotonic() + (args.deadline_s - (time.monotonic() - t0)) / max(stages_left, 1)
            rec_path = out / arm / f"dev_{mode}.jsonl"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(l)["task_id"] for l in rec_path.read_text().splitlines()} if rec_path.exists() else set()
            pred = {}
            n_done = 0
            for tid in order:
                if tid in done:
                    continue
                if time.monotonic() > stage_deadline:
                    break
                t = tasks[tid]
                w0 = time.monotonic()
                if mode == "icl":
                    preds = [predict(model, t["train"], x, f"{tid}:{k}", args.n_aug) for k, x in enumerate(t["test"])]
                    info = {}
                else:
                    seed = int.from_bytes(tid.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
                    torch.manual_seed(seed)
                    preds, info = ttt_predict(model, t["train"], t["test"], tid, np.random.default_rng(seed), args.n_aug,
                                              tuple(int(x) for x in args.ttt_grid.split(",")), 2, ttt_lr, args.ttt_bs)
                tops = [[grid_list(g) for g in (p["top"] if p else [])] for p in preds]
                append_jsonl(rec_path, {"task_id": tid, "max_side": side(t), "wall_s": round(time.monotonic() - w0, 2),
                                        "info": {k: v for k, v in info.items() if k != "loo_nll"}, "top": tops})
                n_done += 1
            for line in rec_path.read_text().splitlines():
                r = json.loads(line)
                pred[r["task_id"]] = r["top"]
            atomic_write_json(out / arm / f"dev_{mode}_pred.json", pred)
            summary["models"][arm][mode] = {"tasks_done": len(pred), "seconds": round(time.monotonic() - t0, 1)}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary["pass"] = all(m.get(md, {}).get("tasks_done", 0) > 0 for m in summary["models"].values()
                          for md in args.modes.split(","))
    atomic_write_json(out / "dev_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
