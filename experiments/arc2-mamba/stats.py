"""Paired comparisons of each arm against the Transformer on the same held-out queries; writes stats.json.

* exact: hpair and htest outputs from tasks with all grids <= 15x15 (the set every mode attempts); counts solved
  by one run only, two-sided sign test on those discordant outputs;
* cell accuracy on hpair <= 15x15 (pooled cells), and changed-cell accuracy on its same-shape queries: difference
  with a task-level bootstrap 95% interval (5000 resamples, seed 0);
* teacher-forced NLL per token on hpair (tf_eval.json, no TTT): difference with the same bootstrap.

    ../../envs/mamba/bin/python stats.py
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arcssm.data import load_split, load_training  # noqa: E402
from arcssm.util import atomic_write_json  # noqa: E402

ARMS = ("mamba", "gru", "hybrid")


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def load(arm, mode):
    p = HERE / "runs" / arm / f"eval_{mode}.jsonl"
    return {(r["set"], r["task_id"]): r for r in map(json.loads, p.read_text().splitlines())}


def per_query(recs, tasks):
    """hpair <= 15: task -> (cells_ok, cells, changed_ok, changed) for attempt 1."""
    out = {}
    for (s, tid), r in recs.items():
        if s != "hpair" or r["bucket"] not in ("<=10", "11-15"):
            continue
        o = r["outputs"][0]
        q_in, q_out = tasks[tid]["train"][-1]
        ch_ok = ch = 0
        if q_in.shape == q_out.shape:
            diff = q_in != q_out
            ch = int(diff.sum())
            if o["top"] and np.asarray(o["top"][0]).shape == q_out.shape:
                ch_ok = int(((np.asarray(o["top"][0], dtype=np.int8) == q_out) & diff).sum())
        out[tid] = (o.get("cells_ok", 0), o.get("cells", 0), ch_ok, ch)
    return out


def solved_outputs(recs, host):
    out = set()
    for (s, tid), r in recs.items():
        if r["bucket"] not in ("<=10", "11-15"):
            continue
        if s == "hpair" and r["outputs"][0].get("exact2"):
            out.add(("hpair", tid, 0))
        if s == "htest":
            for i, o in enumerate(host[tid]["outputs"]):
                if o["correct_1"] or o["correct_2"]:
                    out.add(("htest", tid, i))
    return out


def boot(diff_num_a, den_a, diff_num_b, den_b, rng, n=5000):
    """Pooled-ratio difference a - b with task resampling."""
    num_a, num_b = np.asarray(diff_num_a, float), np.asarray(diff_num_b, float)
    da, db = np.asarray(den_a, float), np.asarray(den_b, float)
    point = num_a.sum() / da.sum() - num_b.sum() / db.sum()
    idx = rng.integers(0, len(num_a), (n, len(num_a)))
    d = num_a[idx].sum(1) / np.maximum(da[idx].sum(1), 1) - num_b[idx].sum(1) / np.maximum(db[idx].sum(1), 1)
    return {"diff": round(float(point), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4),
                                                     round(float(np.percentile(d, 97.5)), 4)]}


def main():
    tasks = load_training()
    load_split()
    rng = np.random.default_rng(0)
    out = {}
    for mode in ("icl", "ttt"):
        ref = load("transformer", mode)
        ref_host = json.loads((HERE / "runs" / "transformer" / f"host_htest_{mode}" / "per_task.json").read_text())
        ref_q = per_query(ref, tasks)
        ref_s = solved_outputs(ref, ref_host)
        for arm in ARMS:
            recs = load(arm, mode)
            host = json.loads((HERE / "runs" / arm / f"host_htest_{mode}" / "per_task.json").read_text())
            q = per_query(recs, tasks)
            s = solved_outputs(recs, host)
            ids = sorted(set(q) & set(ref_q))
            a_only, b_only = len(s - ref_s), len(ref_s - s)
            same_shape = [t for t in ids if q[t][3] > 0]
            out[f"{arm}_vs_transformer:{mode}"] = {
                "exact_outputs": {"arm": len(s), "transformer": len(ref_s), "arm_only": a_only,
                                  "transformer_only": b_only, "sign_p": round(sign_p(a_only, b_only), 4)},
                "cell_acc": boot([q[t][0] for t in ids], [q[t][1] for t in ids],
                                 [ref_q[t][0] for t in ids], [ref_q[t][1] for t in ids], rng),
                "changed_cell_acc": boot([q[t][2] for t in same_shape], [q[t][3] for t in same_shape],
                                         [ref_q[t][2] for t in same_shape], [ref_q[t][3] for t in same_shape], rng),
                "n_queries": len(ids), "n_same_shape": len(same_shape)}
    ref_tf = {r["task_id"]: r for r in json.loads((HERE / "runs" / "transformer" / "tf_eval.json").read_text())["rows"]
              if r["fits"]}
    for arm in ARMS:
        tf = {r["task_id"]: r for r in json.loads((HERE / "runs" / arm / "tf_eval.json").read_text())["rows"] if r["fits"]}
        ids = sorted(set(tf) & set(ref_tf))
        out[f"{arm}_vs_transformer:tf_nll"] = boot([tf[t]["nll_sum"] for t in ids], [tf[t]["tokens"] for t in ids],
                                                   [ref_tf[t]["nll_sum"] for t in ids], [ref_tf[t]["tokens"] for t in ids],
                                                   rng) | {"n_tasks": len(ids)}
    atomic_write_json(HERE / "stats.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
