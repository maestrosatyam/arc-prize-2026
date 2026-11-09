"""Aggregate a Kaggle GPU run (one directory per arm, <arm>_s<seed>) into <out>.json and print markdown tables.

Same metrics and scorers as analyze.py / stats.py. Evaluation stages stop at a deadline, so arms are compared on
the items every arm completed (the intersection); per-arm totals are reported with their own denominators too.
htest predictions are scored only through scripts/arc2_eval_local.py (ReplaySolver), restricted to held-out ids.

    ../../envs/mamba/bin/python analyze_gpu.py --root runs/kaggle-small-v1/bundle/experiments/arc2-mamba/runs/small \
        --out results_gpu_small
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import agg, changed_cells, group, host_score, write_preds  # noqa: E402
from arcssm.data import load_split, load_synthetic, load_training  # noqa: E402
from arcssm.util import atomic_write_json  # noqa: E402
from stats import boot, per_query, sign_p, solved_outputs  # noqa: E402

REF = "transformer_g_s0"


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def frac(a, b):
    return f"{a}/{b}" if b else "-"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="basename for <out>.json and host-evaluator dirs")
    ap.add_argument("--ref", default="transformer_g_s0", help="arm the paired comparisons are made against")
    ap.add_argument("--host-from", default=None, help="reuse host-evaluator results of an earlier --out")
    args = ap.parse_args(argv)
    global REF
    REF = args.ref
    root = Path(args.root)
    split = load_split()
    heldout = split["heldout"]
    tasks, synth = load_training(), load_synthetic()
    arms = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "train_summary.json").exists())
    res = {"root": str(root), "arms": {}, "paired": {}}
    recs = {a: {m: load_jsonl(root / a / f"eval_{m}.jsonl") for m in ("icl", "ttt")} for a in arms}

    # items completed by every arm, per mode
    common = {}
    for m in ("icl", "ttt"):
        keys = [{(r["set"], r["task_id"]) for r in recs[a][m]} for a in arms]
        common[m] = set.intersection(*keys) if keys else set()
    res["common_items"] = {m: len(v) for m, v in common.items()}

    for a in arms:
        s = json.loads((root / a / "train_summary.json").read_text())
        curve = load_jsonl(root / a / "val_log.jsonl")
        tf = json.loads((root / a / "tf_eval.json").read_text()) if (root / a / "tf_eval.json").exists() else {}
        ttt_lr = json.loads((root / a / "ttt_lr.json").read_text()) if (root / a / "ttt_lr.json").exists() else {}
        out = {"params": s["params"], "state": s["state"], "final_val": s["final_val"], "device": s.get("device_name"),
               "ttt_lr": ttt_lr.get("lr"),
               "curve": [{"tokens": c["tokens"], "wall": round(c.get("wall_used", 0)), "gen_loss": c["gen"]["loss"],
                          "gen_tf_exact": round(c["gen"]["tf_exact"] * c["gen"]["n"]), "val_loss": c["meta_val"]["loss"],
                          "val_tok_acc": c["meta_val"]["tok_acc"],
                          "val_tf_exact": round(c["meta_val"]["tf_exact"] * c["meta_val"]["n"]),
                          "n_gen": c["gen"]["n"], "n_val": c["meta_val"]["n"]} for c in curve],
               "tf_eval": {k: tf.get(k) for k in ("hpair", "by_bucket", "gen", "hpair_not_fitting")}}
        for m in ("icl", "ttt"):
            rr = recs[a][m]
            rc = [r for r in rr if (r["set"], r["task_id"]) in common[m]]
            mo = {"items": len(rr), "common_items": len(rc)}
            for tag, sub in (("all", rr), ("common", rc)):
                mo[tag] = {"synth": agg(sub, lambda x: x["set"] == "synth"),
                           "hpair": agg(sub, lambda x: x["set"] == "hpair"),
                           "hpair_changed": changed_cells([x for x in sub if x["set"] == "hpair"], tasks, synth),
                           "hpair_new": agg(sub, lambda x: x["set"] == "hpair" and group(x["task_id"]) == "new"),
                           "synth_solved": sorted(x["task_id"] for x in sub if x["set"] == "synth"
                                                  and any(o.get("exact2") for o in x["outputs"])),
                           "hpair_solved": sorted(x["task_id"] for x in sub if x["set"] == "hpair"
                                                  and any(o.get("exact2") for o in x["outputs"]))}
            le15 = [x for x in rc if x["set"] == "hpair" and x["bucket"] in ("<=10", "11-15")]
            mo["hpair_le15"] = agg(le15)
            mo["hpair_le15_changed"] = changed_cells(le15, tasks, synth)
            if any(r["set"] == "htest" for r in rr):
                pred = HERE / "runs" / f"{args.host_from or args.out}_host" / a / f"htest_{m}_pred.json"
                pred.parent.mkdir(parents=True, exist_ok=True)
                write_preds(rr, pred)
                ids = sorted({t for s_, t in common[m] if s_ == "htest"})
                mo["htest_host_common"] = host_score(pred, pred.parent / f"host_htest_{m}", ids, rr)
            if m == "ttt":
                steps = [r["info"].get("steps") for r in rr if isinstance(r.get("info"), dict) and r["info"].get("steps") is not None]
                mo["steps_hist"] = {str(k): steps.count(k) for k in sorted(set(steps))}
            mo["wall_s"] = round(sum(r.get("wall_s", 0) for r in rr), 1)
            out[m] = mo
        res["arms"][a] = out

    # paired comparisons against the Transformer on common items
    rng = np.random.default_rng(0)
    if REF in arms:
        for m in ("icl", "ttt"):
            ref = {(r["set"], r["task_id"]): r for r in recs[REF][m] if (r["set"], r["task_id"]) in common[m]}
            hp = lambda a_: HERE / "runs" / f"{args.host_from or args.out}_host" / a_ / f"host_htest_{m}" / "per_task.json"
            ref_host = json.loads(hp(REF).read_text()) if hp(REF).exists() else {}
            ref_q, ref_s = per_query(ref, tasks), solved_outputs(ref, ref_host)
            for a in arms:
                if a == REF:
                    continue
                rr = {(r["set"], r["task_id"]): r for r in recs[a][m] if (r["set"], r["task_id"]) in common[m]}
                host = json.loads(hp(a).read_text()) if hp(a).exists() else {}
                q, s_ = per_query(rr, tasks), solved_outputs(rr, host)
                ids = sorted(set(q) & set(ref_q))
                same = [t for t in ids if q[t][3] > 0]
                res["paired"][f"{a}_vs_{REF}:{m}"] = {
                    "exact_outputs": {"arm": len(s_), "ref": len(ref_s), "arm_only": len(s_ - ref_s),
                                      "ref_only": len(ref_s - s_), "sign_p": round(sign_p(len(s_ - ref_s), len(ref_s - s_)), 4)},
                    "cell_acc": boot([q[t][0] for t in ids], [q[t][1] for t in ids], [ref_q[t][0] for t in ids],
                                     [ref_q[t][1] for t in ids], rng),
                    "changed_cell_acc": boot([q[t][2] for t in same], [q[t][3] for t in same], [ref_q[t][2] for t in same],
                                             [ref_q[t][3] for t in same], rng),
                    "n_queries": len(ids), "n_same_shape": len(same)}
        ref_tf = {r["task_id"]: r for r in json.loads((root / REF / "tf_eval.json").read_text())["rows"] if r["fits"]}
        for a in arms:
            if a == REF or not (root / a / "tf_eval.json").exists():
                continue
            tf = {r["task_id"]: r for r in json.loads((root / a / "tf_eval.json").read_text())["rows"] if r["fits"]}
            ids = sorted(set(tf) & set(ref_tf))
            res["paired"][f"{a}_vs_{REF}:tf_nll"] = boot([tf[t]["nll_sum"] for t in ids], [tf[t]["tokens"] for t in ids],
                                                         [ref_tf[t]["nll_sum"] for t in ids], [ref_tf[t]["tokens"] for t in ids],
                                                         rng) | {"n_tasks": len(ids)}
    atomic_write_json(HERE / f"{args.out}.json", res)
    report(res)


def report(res):
    print(f"common items: {res['common_items']}\n")
    print("| Arm | Params | Tokens | Steps | Train wall s | stopped by wall cap | gen tf-exact | meta-val tf-exact | meta-val tok-acc | TTT lr |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a, x in res["arms"].items():
        st, v = x["state"], x["final_val"]
        print(f"| {a} | {x['params']:,} | {st['tokens']:,} | {st['step']} | {st['wall_used']:.0f} | "
              f"{bool(st.get('stopped_by_max_wall'))} | {frac(round(v['gen']['tf_exact'] * v['gen']['n']), v['gen']['n'])} | "
              f"{frac(round(v['meta_val']['tf_exact'] * v['meta_val']['n']), v['meta_val']['n'])} | {v['meta_val']['tok_acc']:.3f} | {x['ttt_lr']} |")
    print("\nTeacher-forced NLL per token on hpair (no TTT):\n")
    print("| Arm | all | <=10 | 11-15 | 16-30 | tok acc | tf-exact |")
    print("|---|---|---|---|---|---|---|")
    for a, x in res["arms"].items():
        t = x["tf_eval"]
        if not t.get("hpair"):
            continue
        b = t["by_bucket"]
        print(f"| {a} | {t['hpair']['nll_per_token']:.3f} | " + " | ".join(f"{b[k]['nll_per_token']:.3f}" for k in ("<=10", "11-15", "16-30"))
              + f" | {t['hpair']['tok_acc']:.3f} | {t['hpair']['tf_exact']}/{t['hpair']['tasks']} |")
    for m in ("icl", "ttt"):
        print(f"\n{m}: common items (all arms)\n")
        print("| Arm | items done | hpair exact@2 | hpair cell acc | changed cells | htest exact@2 (host) | synth exact@2 | new hpair exact@2 | wall s |")
        print("|---|---|---|---|---|---|---|---|---|")
        for a, x in res["arms"].items():
            mo = x[m]
            c = mo["common"]
            h = mo.get("htest_host_common", {}).get("all", {})
            print(f"| {a} | {mo['items']} | {frac(c['hpair']['exact2'], c['hpair']['outputs'])} | {pct(c['hpair']['cells_ok'], c['hpair']['cells'])} | "
                  f"{frac(c['hpair_changed']['changed_ok'], c['hpair_changed']['changed'])} ({pct(c['hpair_changed']['changed_ok'], c['hpair_changed']['changed'])}) | "
                  f"{frac(h.get('exact2', 0), h.get('outputs', 0))} | {frac(c['synth']['exact2'], c['synth']['outputs'])} | "
                  f"{frac(c['hpair_new']['exact2'], c['hpair_new']['outputs'])} | {mo['wall_s']:.0f} |")
    print("\nhpair, tasks with all grids <= 15x15 (121 queries):\n")
    print("| Run | exact@2 | exact@1 | shape@1 | cell acc | changed cells correct | unchanged cells correct |")
    print("|---|---|---|---|---|---|---|")
    for m in ("icl", "ttt"):
        for a, x in res["arms"].items():
            h, c = x[m]["hpair_le15"], x[m]["hpair_le15_changed"]
            print(f"| {a}, {m} | {frac(h['exact2'], h['outputs'])} | {frac(h['exact1'], h['outputs'])} | {frac(h['shape1'], h['outputs'])} | "
                  f"{pct(h['cells_ok'], h['cells'])} | {frac(c['changed_ok'], c['changed'])} ({pct(c['changed_ok'], c['changed'])}) | "
                  f"{frac(c['kept_ok'], c['kept'])} ({pct(c['kept_ok'], c['kept'])}) |")
    print("\nhtest (host evaluator), exact@2 by size and inheritance:\n")
    print("| Run | all | <=10 | 11-15 | 16-30 | new | inherited | solved ids |")
    print("|---|---|---|---|---|---|---|---|")
    for m in ("icl", "ttt"):
        for a, x in res["arms"].items():
            h = x[m].get("htest_host_common")
            if not h:
                continue
            print(f"| {a}, {m} | " + " | ".join(frac(h[k]["exact2"], h[k]["outputs"]) for k in ("all", "<=10", "11-15", "16-30", "new", "inherited"))
                  + f" | {', '.join(sorted(set(h['solved_ids'])))} |")
    print("\nPaired against the Transformer (common items):\n")
    for k, v in res["paired"].items():
        print(f"- {k}: {json.dumps(v)}")


if __name__ == "__main__":
    main()
