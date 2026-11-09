"""Option 4: a pretrained Mamba language model (state-spaces/mamba-130m-hf, revision 1e76775f, local copy in
envs/mamba/models) given the train pairs as a few-shot text prompt. Greedy decoding; attempt 1 from the task as
given, attempt 2 from the transposed task (inverted back). Same sets and record format as evaluate.py.

    OMP_NUM_THREADS=2 nice -n 19 ../../envs/mamba/bin/python pretrained_probe.py --sets synth,hpair,htest --max-side 10
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from arcssm.data import load_split, load_synthetic, load_training  # noqa: E402
from arcssm.encoding import Aug  # noqa: E402
from arcssm.infer import score, vote  # noqa: E402
from arcssm.util import LoadGuard, append_jsonl, limit_threads, rss_mb  # noqa: E402
from evaluate import grid_list, items  # noqa: E402

MODEL_DIR = ROOT / "envs" / "mamba" / "models" / "mamba-130m-hf"
REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
MAX_PROMPT = 4096


def g2t(g):
    return "\n".join(" ".join(str(int(v)) for v in r) for r in g)


def prompt(pairs, q):
    parts = []
    for k, (i, o) in enumerate(pairs, 1):
        parts.append(f"Example {k}\nInput:\n{g2t(i)}\nOutput:\n{g2t(o)}\n")
    parts.append(f"Test\nInput:\n{g2t(q)}\nOutput:\n")
    return "\n".join(parts)


def parse(text):
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if rows:
                break
            continue
        toks = line.split()
        if not all(t.isdigit() and len(t) == 1 for t in toks):
            break
        rows.append([int(t) for t in toks])
    if not rows or len({len(r) for r in rows}) != 1 or len(rows) > 30 or len(rows[0]) > 30:
        return None
    return np.asarray(rows, dtype=np.int8)


@torch.no_grad()
def answer(model, tok, pairs, q, aug):
    ps = [(aug.fwd(i), aug.fwd(o)) for i, o in pairs]
    qq = aug.fwd(q)
    while True:
        text = prompt(ps, qq)
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] <= MAX_PROMPT or len(ps) == 1:
            break
        ps = ps[:-1]
    if all(i.shape == o.shape for i, o in pairs):
        h, w = qq.shape
    else:
        h, w = max(o.shape[0] for _, o in ps), max(o.shape[1] for _, o in ps)
    max_new = int(2.2 * h * (w + 1)) + 8
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    g = parse(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    return (aug.inv(g) if g is not None else None), int(ids.shape[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="synth,hpair,htest")
    ap.add_argument("--max-side", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "runs" / "pretrained130m" / "eval_fewshot.jsonl"))
    args = ap.parse_args(argv)
    limit_threads(2)
    from transformers import AutoTokenizer, MambaForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = MambaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["set"], json.loads(l)["task_id"]) for l in out_path.read_text().splitlines()}
    guard = LoadGuard(log=lambda m: append_jsonl(out_path.with_suffix(".log"), {"msg": m}))
    for set_name, tid, train_pairs, test_inputs, truths, bucket in items(load_split(), load_training(),
                                                                          load_synthetic(), args.sets.split(",")):
        if (set_name, tid) in done:
            continue
        big = max(max(max(g.shape) for p in train_pairs for g in p), max(max(g.shape) for g in test_inputs))
        if big > args.max_side:
            continue
        guard.check()
        t0 = time.process_time()
        rec = {"set": set_name, "task_id": tid, "bucket": bucket, "n_train": len(train_pairs),
               "n_test": len(test_inputs), "model": f"state-spaces/mamba-130m-hf@{REVISION}", "outputs": []}
        for k, x in enumerate(test_inputs):
            res = [answer(model, tok, train_pairs, x, a) for a in (Aug(0), Aug(6))]
            ranked = vote([(g, 0.0) for g, _ in res])
            top = [g for g, _, _ in ranked[:2]]
            o = {"test_index": k, "attempted": True, "top": [grid_list(g) for g in top],
                 "prompt_tokens": [n for _, n in res], "parsed": [g is not None for g, _ in res]}
            if truths is not None:
                o.update(score(top, truths[k]))
            rec["outputs"].append(o)
        rec["cpu_s"] = round(time.process_time() - t0, 2)
        rec["rss_mb"] = round(rss_mb())
        append_jsonl(out_path, rec)


if __name__ == "__main__":
    main()
