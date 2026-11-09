"""Replay-based ablation: what each component adds, from a run's saved samples (no model calls).

  python -m arc_harness.ablate runs/x [--external pool.json] [--tasks dev]

Stage ablations keep only the samples of some stages and re-run selection:
  s1            first samples (plus their rescues): "answer + program, verification as the selector"
  s1+repair     adds the repair round
  s1+repair+s2  adds the second samples (the full run)
Selection ablations re-rank the full sample set:
  grid_only     votes over written grids, no programs, no validator
  vr-no-validator
  vr            the harness policy
Merge ablations, when an external pool is given: R0 | verified_first | harness_first.
Also reports tokens by stage, the verified-but-wrong rate, and right outputs per million tokens for each arm.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

from .data import load_dir
from .grids import hardcoded_risk
from .run import ROOT, select_tasks
from .score import score_submission
from .selection import two_attempts
from .solvers import load_samples

STAGE_SETS = {
    "s1": {"s1", "rescue"},
    "s1+repair": {"s1", "rescue", "repair", "rescue_repair"},
    "s1+repair+s2": {"s1", "rescue", "repair", "rescue_repair", "s2", "rescue_s2"},
}


def replay(tasks: Dict[str, dict], samples: Dict[str, List[dict]], policy: str, stages=None,
           external=None, merge="R0", use_validator=True) -> dict:
    sub = {}
    tokens = 0
    for tid, task in tasks.items():
        ss = samples.get(tid, [])
        if stages is not None:
            keep = set(s["sample_id"] for s in ss if s["stage"] in stages)
            # a rescue's token cost belongs with its cut parent; keep parents of kept rescues for accounting
            ss_keep = [s for s in ss if s["sample_id"] in keep]
            parents = {s["parent"] for s in ss_keep if s.get("parent")}
            tokens += sum(s["gen_tokens"] for s in ss if s["sample_id"] in keep or s["sample_id"] in parents)
        else:
            ss_keep = ss
            tokens += sum(s["gen_tokens"] for s in ss)
        ext = (external or {}).get(tid)
        sub[tid] = two_attempts(task, ss_keep, policy, ext, merge, use_validator)
    sc = score_submission(sub, tasks)
    return {"right": sc["right"], "n": sc["n_outputs"], "score": sc["score"], "tokens": tokens,
            "per_M": 1e6 * sc["right"] / max(1, tokens), "attempt1_right": sc["attempt1_right"]}


def verified_stats(tasks, samples) -> dict:
    """How precise the verification bit is: verified programs right / verified programs."""
    n_ver = n_ver_right = n_prog = n_prog_right = n_hard = 0
    for tid, task in tasks.items():
        for s in samples.get(tid, []):
            v = s.get("verify") or {}
            if not v.get("ran"):
                continue
            for j, te in enumerate(task["test"]):
                if "output" not in te:
                    continue
                po = (v.get("program_outputs") or [None] * (j + 1))[j] if j < len(v.get("program_outputs") or []) else None
                if po is None:
                    continue
                n_prog += 1
                n_prog_right += po == te["output"]
                if v.get("passed"):
                    n_ver += 1
                    n_ver_right += po == te["output"]
                    n_hard += hardcoded_risk(s.get("program") or "")
    return {"program_outputs": n_prog, "program_outputs_right": n_prog_right, "verified": n_ver,
            "verified_right": n_ver_right, "verified_with_literals": n_hard,
            "verified_precision": n_ver_right / n_ver if n_ver else None,
            "verified_but_wrong_rate": (n_ver - n_ver_right) / n_ver if n_ver else None}


def tokens_by_stage(samples) -> dict:
    c = Counter()
    k = Counter()
    for ss in samples.values():
        for s in ss:
            c[s["stage"]] += s["gen_tokens"]
            k[s["stage"]] += 1
    return {st: {"calls": k[st], "tokens": c[st]} for st in c}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tasks", default="dev")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--external", default=None)
    args = ap.parse_args(argv)
    tasks = select_tasks(args.tasks, args.data_root)
    samples = load_samples(os.path.join(args.run_dir, "samples.jsonl"))
    tasks = {t: tasks[t] for t in tasks if t in samples} or tasks
    external = None
    if args.external:
        with open(args.external) as f:
            external = json.load(f)
    rows = []
    for name, stages in STAGE_SETS.items():
        r = replay(tasks, samples, "vr", stages)
        rows.append(("stages", name) + (r["right"], r["n"], r["tokens"], r["per_M"]))
    for name, pol, val in (("grid_only", "grid_only", False), ("vr-no-validator", "vr", False), ("vr", "vr", True),
                           ("poetiq-rank", "poetiq", False)):
        r = replay(tasks, samples, pol, None, use_validator=val)
        rows.append(("selection", name) + (r["right"], r["n"], r["tokens"], r["per_M"]))
    if external:
        for m in ("R0", "verified_first", "harness_first", "verified_then_margin"):
            r = replay(tasks, samples, "vr", None, external, m)
            rows.append(("merge", m) + (r["right"], r["n"], r["tokens"], r["per_M"]))
        def _att(e):
            return e["attempts"] if isinstance(e, dict) else e
        ext_only = {tid: [{"attempt_1": e[0], "attempt_2": e[1] if len(e) > 1 else [[0]]}
                          for e in _att(external.get(tid, []))] for tid in tasks}
        sc = score_submission(ext_only, tasks)
        rows.append(("merge", "external_only", sc["right"], sc["n_outputs"], 0, 0.0))
    out = {"rows": [dict(zip(("kind", "arm", "right", "n", "tokens", "right_per_M"), r)) for r in rows],
           "verified": verified_stats(tasks, samples), "tokens_by_stage": tokens_by_stage(samples)}
    with open(os.path.join(args.run_dir, "ablation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'kind':10s} {'arm':18s} {'right':>6s} {'n':>4s} {'tokens':>10s} {'right/M':>8s}")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:18s} {r[2]:6d} {r[3]:4d} {r[4]:10d} {r[5]:8.1f}")
    print("verification:", json.dumps(out["verified"]))
    print("tokens by stage:", json.dumps(out["tokens_by_stage"]))
    return 0


if __name__ == "__main__":
    main()
