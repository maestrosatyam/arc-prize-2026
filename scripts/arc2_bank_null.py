#!/usr/bin/env python3
"""Null-rule calibration of the adoption gate (E12) on the ARC-AGI-2 output bank (eval/arc2_bank.py).

Each null family is replayed over seeds 1..N and every replay is compared with its base: b0 for all families except
null_q27swap, whose base is v7_direct (it only reshuffles the 27B sample order of v7's merge). The fraction of seeds
that pass a gate is the gate's false-positive rate under a rule that carries no information.

  level-preserving   null_jitter / null_jitter1 / null_jitter2 (kgmon scores jittered, sigma 0.5 / 1 / 2) and
                     null_q27swap: close calls move at random, clear ones stay. Their net distribution is the
                     gate's null; the net a real rule must reach for <= 5 % (<= 1 %) false positives is the smallest
                     integer T with P(net >= T) <= 0.05 (0.01) over the seeds (ceil of the 95th / 99th percentile
                     up to ties).
  destructive        null_pair (two random candidates) and null_rerank (random permutation of the kgmon ranking):
                     they discard B0's ranking and lose about half the outputs, so their net is far negative and
                     never passes; their `gained` distribution tells how many outputs chance alone recovers.

Per seed: outputs right, gained, lost, net, the one-sided sign test p on the discordants, gate_v1 (net >= 4 and
p < 0.05) and gate_v2 (net >= 3 and lost <= 1). The task-level cluster bootstrap inside compare() does not enter
either gate; with --n-boot 0 (default) compare() is called with CLUSTER_BOOT_MIN resamples, because
cluster_bootstrap divides by n_boot, and the bootstrap block is left out of the per-seed rows.

Usage: python scripts/arc2_bank_null.py [--seeds 200] [--families null_jitter ...] [--n-boot 0]
                                        [--pools DIR ...] [--q27 DIR] [--json experiments/arc2-bank/null_calibration.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (CHALLENGES, GATE_V1, GATE_V2, NULL_FAMILIES, Q27_BANK, Bank, Result,  # noqa: E402
                            compare, load_bank, mean_sd, percentile, replay)

SCHEMA = "arc2_bank_null_calibration.v1"
DEFAULT_JSON = ROOT / "experiments" / "arc2-bank" / "null_calibration.json"
DEFAULT_SEEDS = 200
CLUSTER_BOOT_MIN = 50            # compare() with n_boot 0 divides by zero in cluster_bootstrap; smallest n used instead
BASES = {"null_q27swap": "v7_direct"}       # every other family is compared with b0
LEVEL_PRESERVING = ("null_jitter", "null_jitter1", "null_jitter2", "null_q27swap")
DESTRUCTIVE = ("null_pair", "null_rerank")
FP_LEVELS = ((5, 0.05), (1, 0.01))


# ----------------------------------------------------------------------------
# statistics

def _r(v: float | None, nd: int = 4) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, nd)


def dist(values: list[float]) -> dict:
    """mean, sd, p50, p95, p99, min, max of a per-seed quantity."""
    if not values:
        return {"n": 0}
    m, sd = mean_sd(values)
    return {"n": len(values), "mean": _r(m), "sd": _r(sd), "p50": _r(percentile(values, 0.5)),
            "p95": _r(percentile(values, 0.95)), "p99": _r(percentile(values, 0.99)),
            "min": min(values), "max": max(values)}


def net_needed(nets: list[int], alpha: float) -> int | None:
    """Smallest integer T with P(net >= T) <= alpha over the seeds (the gate threshold for that false-positive rate)."""
    if not nets:
        return None
    t = min(nets)
    while sum(1 for v in nets if v >= t) / len(nets) > alpha:
        t += 1
    return t


def fp_rate(rows: list[dict], gate: str) -> float | None:
    return _r(sum(1 for r in rows if r[gate]) / len(rows)) if rows else None


# ----------------------------------------------------------------------------
# calibration

def seed_row(base: Result, new: Result, seed: int, n_boot: int) -> dict:
    cmp = compare(base, new, n_boot=max(n_boot, CLUSTER_BOOT_MIN))
    row = {"seed": seed, "n_right": new.n_right, "n_right_1": new.n_right_1, "gained": cmp["gained"],
           "lost": cmp["lost"], "net": cmp["net"], "sign_test_p": cmp["sign_test_p"],
           "gate_v1": cmp["gate_v1"]["pass"], "gate_v2": cmp["gate_v2"]["pass"]}
    if n_boot > 0:
        boot = cmp["task_level"]["bootstrap"]
        row["task_mean_delta"] = cmp["task_level"]["mean_delta"]
        row["task_delta_ci95"] = boot.get("ci95")
    return row


def summarize(rows: list[dict]) -> dict:
    nets = [r["net"] for r in rows]
    return {"n_seeds": len(rows),
            "right": dist([r["n_right"] for r in rows]),
            "net": dist(nets), "gained": dist([r["gained"] for r in rows]), "lost": dist([r["lost"] for r in rows]),
            "sign_test_p": dist([r["sign_test_p"] for r in rows if r["sign_test_p"] is not None]),
            "fp_gate_v1": fp_rate(rows, "gate_v1"), "fp_gate_v2": fp_rate(rows, "gate_v2"),
            **{f"net_needed_fp{pct}": net_needed(nets, a) for pct, a in FP_LEVELS}}


def calibrate_family(bank: Bank, family: str, bases: dict[str, Result], n_seeds: int, n_boot: int) -> dict:
    base_name = BASES.get(family, "b0")
    base = bases[base_name]
    rows = [seed_row(base, replay(bank, f"{family}:{s}"), s, n_boot) for s in range(1, n_seeds + 1)]
    block = {"family": family, "base": base_name, "base_right": base.n_right, "n_outputs": base.n_outputs,
             "kind": "level-preserving" if family in LEVEL_PRESERVING else "destructive",
             "description": NULL_FAMILIES[family].__doc__.strip().splitlines()[0]}
    block.update(summarize(rows))
    block["score"] = {k: _r(block["right"][k] / base.n_outputs) for k in ("mean", "sd", "min", "max")} if rows else {}
    block["seeds"] = rows
    return block


def calibrate(bank: Bank, families: list[str], n_seeds: int, n_boot: int) -> dict:
    bases = {name: replay(bank, name) for name in sorted(set(BASES.get(f, "b0") for f in families))}
    fams = {f: calibrate_family(bank, f, bases, n_seeds, n_boot) for f in families}
    pooled_rows = [r for f in families if f in LEVEL_PRESERVING for r in fams[f]["seeds"]]
    pooled = {"families": [f for f in families if f in LEVEL_PRESERVING]}
    pooled.update(summarize(pooled_rows))
    return {"bases": {n: r.summary() for n, r in bases.items()},
            "compare_n_boot": max(n_boot, CLUSTER_BOOT_MIN), "n_boot_requested": n_boot,
            "note_n_boot": (f"compare() called with {CLUSTER_BOOT_MIN} task resamples (cluster_bootstrap divides by "
                            f"n_boot); the bootstrap does not enter the gates and is not kept per seed"
                            if n_boot <= 0 else "task-level bootstrap kept per seed (task_mean_delta, task_delta_ci95)"),
            "gates": {"gate_v1": f"net >= {GATE_V1['net_min']} and sign test p < {GATE_V1['p_max']}",
                      "gate_v2": f"net >= {GATE_V2['net_min']} and lost <= {GATE_V2['lost_max']} (proposed revision)"},
            "net_needed_definition": "smallest integer T with P(net >= T) <= alpha over the seeds",
            "families": fams, "pooled_level_preserving": pooled,
            "destructive": [f for f in families if f in DESTRUCTIVE]}


# ----------------------------------------------------------------------------
# text

def _ms(d: dict, k1: str = "mean", k2: str = "sd") -> str:
    return f"{d.get(k1, float('nan')):.2f} ({d.get(k2, float('nan')):.2f})"


def summary_lines(res: dict) -> list[str]:
    b = res["bank"]
    lines = [f"null-rule calibration (E12): {b['n_tasks']} tasks, {b['n_outputs']} outputs, {b['n_q27_tasks']} with 27B "
             f"samples; bank {b['fingerprint']}; {res['args']['seeds']} seeds per family",
             "bases: " + ", ".join(f"{n} {r['n_right']}/{r['n_outputs']}" for n, r in res['bases'].items()),
             "", "family | kind | base | seeds | right mean (sd) [min, max] | net mean (sd) | net p50 / p95 / p99 / max | "
             "gained mean / p95 / max | lost mean / p95 / max"]
    for f, blk in res["families"].items():
        r, n, g, lo = blk["right"], blk["net"], blk["gained"], blk["lost"]
        lines.append(f"{f} | {blk['kind']} | {blk['base']} ({blk['base_right']}) | {blk['n_seeds']} | {_ms(r)} "
                     f"[{r['min']}, {r['max']}] | {_ms(n)} | {n['p50']:.1f} / {n['p95']:.1f} / {n['p99']:.1f} / {n['max']} | "
                     f"{g['mean']:.2f} / {g['p95']:.1f} / {g['max']} | {lo['mean']:.2f} / {lo['p95']:.1f} / {lo['max']}")
    lines += ["", f"gates: v1 = {res['gates']['gate_v1']}; v2 = {res['gates']['gate_v2']}",
              "family | FP gate_v1 | FP gate_v2 | net needed for <= 5 % FP | <= 1 % FP"]
    for f, blk in res["families"].items():
        lines.append(f"{f} | {blk['fp_gate_v1']:.3f} | {blk['fp_gate_v2']:.3f} | {blk['net_needed_fp5']} | {blk['net_needed_fp1']}")
    p = res["pooled_level_preserving"]
    if p.get("n_seeds"):
        n = p["net"]
        lines.append(f"pooled level-preserving ({', '.join(p['families'])}; {p['n_seeds']} seeds) | {p['fp_gate_v1']:.3f} | "
                     f"{p['fp_gate_v2']:.3f} | {p['net_needed_fp5']} | {p['net_needed_fp1']}  (net p95 {n['p95']:.1f}, "
                     f"p99 {n['p99']:.1f}, max {n['max']})")
    des = [f for f in res["destructive"] if f in res["families"]]
    if des:
        lines.append("destructive nulls (B0's ranking discarded; chance alone recovers): " + "; ".join(
            f"{f} gained mean {res['families'][f]['gained']['mean']:.2f}, p95 {res['families'][f]['gained']['p95']:.1f}, "
            f"max {res['families'][f]['gained']['max']} of {res['families'][f]['base_right']} base-right outputs"
            for f in des))
    lines.append(res["note_n_boot"])
    return lines


# ----------------------------------------------------------------------------
# cli

def add_bank_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pools", nargs="+", default=None, metavar="DIR", help="pool directories (default: both bank parts)")
    ap.add_argument("--q27", default=str(Q27_BANK), help="the 27B bank directory")
    ap.add_argument("--challenges", default=str(CHALLENGES), help="challenges file (task sizes)")
    ap.add_argument("--solutions", default="", help="extra solutions file (default: the Kaggle copies beside the pools)")


def bank_from_args(args: argparse.Namespace) -> Bank:
    return load_bank(args.pools, args.q27, challenges=args.challenges or None, solutions=args.solutions or None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds 1..N per family")
    ap.add_argument("--families", nargs="+", default=list(NULL_FAMILIES), choices=list(NULL_FAMILIES))
    ap.add_argument("--n-boot", type=int, default=0,
                    help=f"task resamples of compare()'s cluster bootstrap per seed (0: {CLUSTER_BOOT_MIN} used, block dropped)")
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="write the result here ('' to skip)")
    add_bank_args(ap)
    args = ap.parse_args(argv)
    t0 = time.time()
    bank = bank_from_args(args)
    res = {"schema": SCHEMA, "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "args": {"seeds": args.seeds, "families": args.families, "n_boot": args.n_boot, "pools": bank.sources,
                    "q27": args.q27, "json": args.json},
           "bank": {"fingerprint": bank.fingerprint(), "n_tasks": len(bank.tasks), "n_outputs": bank.n_outputs,
                    "n_q27_tasks": len(bank.q27), "sources": bank.sources, "solutions": bank.solutions_source}}
    res.update(calibrate(bank, args.families, args.seeds, args.n_boot))
    res["elapsed_s"] = round(time.time() - t0, 1)
    print("\n".join(summary_lines(res)))
    print(f"{len(args.families) * args.seeds} replays in {res['elapsed_s']} s")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
