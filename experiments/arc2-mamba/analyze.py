"""Aggregate evaluation records into results.json and the tables in RESULTS.md.

hpair and synth are scored from their own records (labels are train pairs or synthetic). htest predictions are
written to runs/<arm>/htest_<mode>_pred.json and scored only by the host evaluator
(scripts/arc2_eval_local.py --split train, ReplaySolver); its per_task.json is then restricted to held-out ids.

    ../../envs/mamba/bin/python analyze.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from arcssm.data import load_split  # noqa: E402
from arcssm.util import atomic_write_json  # noqa: E402

RUNS = [(arm, mode, HERE / "runs" / arm / f"eval_{mode}.jsonl")
        for arm in ("mamba", "transformer", "gru", "hybrid") for mode in ("icl", "ttt")]
RUNS.append(("pretrained130m", "fewshot", HERE / "runs" / "pretrained130m" / "eval_fewshot.jsonl"))
TAGS = json.loads((ROOT / "experiments" / "arc2-grammar" / "tasks.json").read_text())["tags"]


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def agg(recs, pred=lambda r: True):
    keys = ("exact1", "exact2", "shape1", "shape2", "cells_ok", "cells")
    a = {"tasks": 0, "outputs": 0, "attempted": 0, **{k: 0 for k in keys}, "cpu_s": 0.0}
    for r in recs:
        if not pred(r):
            continue
        a["tasks"] += 1
        a["cpu_s"] += r.get("cpu_s", 0.0)
        for o in r["outputs"]:
            a["outputs"] += 1
            a["attempted"] += int(o["attempted"])
            for k in keys:
                a[k] += o.get(k, 0)
    a["cpu_s"] = round(a["cpu_s"], 1)
    return a


def changed_cells(recs, tasks, synth):
    """Same-shape queries only: attempt-1 accuracy on cells where the truth differs from the query input
    (copying scores 0 there) and on cells where it does not."""
    import numpy as np

    a = {"queries": 0, "changed": 0, "changed_ok": 0, "kept": 0, "kept_ok": 0}
    for r in recs:
        if r["set"] == "hpair":
            q_in, q_out = tasks[r["task_id"]]["train"][-1]
        elif r["set"] == "synth":
            q_in, q_out = synth[r["task_id"]]["test"][0], synth[r["task_id"]]["test_out"][0]
        else:
            continue
        if q_in.shape != q_out.shape:
            continue
        o = r["outputs"][0]
        a["queries"] += 1
        diff = q_in != q_out
        a["changed"] += int(diff.sum())
        a["kept"] += int((~diff).sum())
        if o["top"] and np.asarray(o["top"][0]).shape == q_out.shape:
            p = np.asarray(o["top"][0], dtype=np.int8)
            a["changed_ok"] += int(((p == q_out) & diff).sum())
            a["kept_ok"] += int(((p == q_out) & ~diff).sum())
    return a


def group(tid):
    t = TAGS.get(tid)
    return None if t is None else ("inherited" if t["arc1_inherited"] else "new")


def write_preds(recs, path):
    pred = {}
    for r in recs:
        if r["set"] != "htest":
            continue
        pred[r["task_id"]] = [o["top"] for o in sorted(r["outputs"], key=lambda o: o["test_index"])]
    atomic_write_json(path, pred)
    return len(pred)


def host_score(pred_path, out_dir, heldout, recs):
    if not (out_dir / "per_task.json").exists():
        env = dict(os.environ, PYTHONPATH=str(HERE), ARC2_MAMBA_REPLAY=str(pred_path), OMP_NUM_THREADS="2")
        subprocess.run([str(ROOT / "envs" / "mamba" / "bin" / "python"), str(ROOT / "scripts" / "arc2_eval_local.py"),
                        "--solver", "arcssm.solver:ReplaySolver", "--split", "train", "--per-task-seconds", "5",
                        "--out", str(out_dir)], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
    per_task = json.loads((out_dir / "per_task.json").read_text())
    bucket = {r["task_id"]: r["bucket"] for r in recs if r["set"] == "htest"}
    attempted = {r["task_id"]: [o["attempted"] for o in r["outputs"]] for r in recs if r["set"] == "htest"}

    def tally(ids):
        a = {"tasks": 0, "outputs": 0, "attempted": 0, "exact1": 0, "exact2": 0, "shape2": 0}
        for tid in ids:
            a["tasks"] += 1
            for i, o in enumerate(per_task[tid]["outputs"]):
                a["outputs"] += 1
                a["attempted"] += int(attempted.get(tid, [False] * 9)[i] if tid in attempted else False)
                a["exact1"] += int(o["correct_1"])
                a["exact2"] += int(o["correct_1"] or o["correct_2"])
                a["shape2"] += int(o["shape_ok"])
        return a

    res = {"all": tally(heldout)}
    for b in ("<=10", "11-15", "16-30"):
        res[b] = tally([t for t in heldout if bucket.get(t) == b])
    for g in ("inherited", "new"):
        res[g] = tally([t for t in heldout if group(t) == g])
    res["solved_not_from_model"] = sorted(t for t in heldout for o in per_task[t]["outputs"]
                                          if (o["correct_1"] or o["correct_2"]) and not o["oracle_pool_hit"])
    res["solved_ids"] = sorted(t for t in heldout for o in per_task[t]["outputs"] if o["correct_1"] or o["correct_2"])
    return res


def copy_baseline(split):
    """Attempt 1 = the query input itself (no model), on hpair and synth, with evaluate.py's scorer."""
    import numpy as np

    from arcssm.data import load_synthetic, load_training
    from arcssm.infer import score
    from evaluate import items, size_bucket  # noqa: F401

    recs = []
    for set_name, tid, train_pairs, test_inputs, truths, bucket in items(split, load_training(), load_synthetic(),
                                                                          ["synth", "hpair"]):
        outs = []
        for k, x in enumerate(test_inputs):
            o = {"test_index": k, "attempted": True}
            o.update(score([np.asarray(x)], truths[k]))
            outs.append(o)
        recs.append({"set": set_name, "task_id": tid, "bucket": bucket, "cpu_s": 0.0, "outputs": outs})
    r = {"synth": agg(recs, lambda x: x["set"] == "synth"), "hpair": agg(recs, lambda x: x["set"] == "hpair")}
    for b in ("<=10", "11-15", "16-30"):
        r[f"hpair{b}"] = agg(recs, lambda x, b=b: x["set"] == "hpair" and x["bucket"] == b)
    for g in ("inherited", "new"):
        r[f"hpair_{g}"] = agg(recs, lambda x, g=g: x["set"] == "hpair" and group(x["task_id"]) == g)
    r["synth_per_task"] = {x["task_id"]: x["outputs"][0]["exact2"] for x in recs if x["set"] == "synth"}
    le15 = lambda x: x["set"] == "hpair" and x["bucket"] in ("<=10", "11-15")
    r["hpair_le15"] = agg(recs, le15)
    r["hpair_changed_le15"] = {"queries": 0, "changed": 0, "changed_ok": 0, "kept": 0, "kept_ok": 0}
    r["hpair_solved_ids"] = sorted(x["task_id"] for x in recs if x["set"] == "hpair" and x["outputs"][0]["exact2"])
    return r


def main():
    split = load_split()
    heldout = split["heldout"]
    out = {"split_sha256": split["sha256"], "runs": {"copy_input:baseline": copy_baseline(split)}, "training": {}}
    for arm in ("mamba", "transformer", "gru", "hybrid"):
        s = HERE / "runs" / arm / "train_summary.json"
        if s.exists():
            d = json.loads(s.read_text())
            curve = load(HERE / "runs" / arm / "val_log.jsonl")
            out["training"][arm] = {"params": d["params"], "state": d["state"], "final_val": d["final_val"],
                                    "curve": [{"cpu_used": round(c["cpu_used"]), "tokens": c["tokens"],
                                               "gen_tf_exact": c["gen"]["tf_exact"], "gen_tok_acc": c["gen"]["tok_acc"],
                                               "val_tf_exact": c["meta_val"]["tf_exact"],
                                               "val_tok_acc": c["meta_val"]["tok_acc"]} for c in curve]}
            ttt_lr = HERE / "runs" / arm / "ttt_lr.json"
            if ttt_lr.exists():
                out["training"][arm]["ttt_lr"] = json.loads(ttt_lr.read_text())
    from arcssm.data import load_synthetic, load_training

    tasks, synth = load_training(), load_synthetic()
    for arm, mode, path in RUNS:
        recs = load(path)
        if not recs:
            continue
        r = {"synth": agg(recs, lambda x: x["set"] == "synth"),
             "synth_per_task": {x["task_id"]: x["outputs"][0].get("exact2", 0) for x in recs if x["set"] == "synth"},
             "hpair": agg(recs, lambda x: x["set"] == "hpair")}
        for b in ("<=10", "11-15", "16-30"):
            r[f"hpair{b}"] = agg(recs, lambda x, b=b: x["set"] == "hpair" and x["bucket"] == b)
        for g in ("inherited", "new"):
            r[f"hpair_{g}"] = agg(recs, lambda x, g=g: x["set"] == "hpair" and group(x["task_id"]) == g)
        r["hpair_solved_ids"] = sorted(x["task_id"] for x in recs if x["set"] == "hpair"
                                       and any(o.get("exact2") for o in x["outputs"]))
        r["htest_cpu_s"] = agg(recs, lambda x: x["set"] == "htest")["cpu_s"]
        r["hpair_changed"] = changed_cells([x for x in recs if x["set"] == "hpair"], tasks, synth)
        le15 = lambda x: x["set"] == "hpair" and x["bucket"] in ("<=10", "11-15")
        r["hpair_le15"] = agg(recs, le15)
        r["hpair_changed_le15"] = changed_cells([x for x in recs if le15(x)], tasks, synth)
        r["synth_changed"] = changed_cells([x for x in recs if x["set"] == "synth"], tasks, synth)
        if any(x["set"] == "htest" for x in recs):
            pred_path = HERE / "runs" / arm / f"htest_{mode}_pred.json"
            write_preds(recs, pred_path)
            r["htest_host"] = host_score(pred_path, HERE / "runs" / arm / f"host_htest_{mode}", heldout, recs)
        if mode == "ttt":
            steps = [x["info"].get("steps") for x in recs if x["info"].get("steps") is not None]
            r["ttt_steps_hist"] = {str(s): steps.count(s) for s in sorted(set(steps))}
        out["runs"][f"{arm}:{mode}"] = r
    atomic_write_json(HERE / "results.json", out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("synth", "hpair", "htest_host")}
                      for k, v in out["runs"].items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
