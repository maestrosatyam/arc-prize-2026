#!/usr/bin/env python3
"""Error bars on board readings (E1) from the ARC-AGI-2 output bank (eval/arc2_bank.py).

Every rule is replayed on the bank and its tasks are resampled with replacement to a board of n tasks (120: the
public board is about half of the 240 hidden tasks; 240: the hidden set); the score is output-weighted as on the
board, in points (100 x score). For every rule other than the base the paired difference new - base is resampled
the same way (bootstrap_difference), and

  MDD95 = 1.96 x sd of the resampled difference, in points: the board difference a single paired reading must show
          to lie outside the 95 % interval of task-resampling noise alone (no run noise, no 27B draw noise).

The bank's rules are read on 101 tasks (145 outputs); the resampled boards reuse those tasks, so the spread is the
task-sampling spread of this bank's hit pattern, not of the hidden tasks.

Usage: python scripts/arc2_bank_bootstrap.py [--rules b0 v7_direct q27_first q27_agree] [--n-tasks 120 240]
                                             [--n-boot 4000] [--pools DIR ...] [--q27 DIR]
                                             [--json experiments/arc2-bank/bootstrap.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (CHALLENGES, Q27_BANK, Bank, Result, bootstrap_board, bootstrap_difference,  # noqa: E402
                            load_bank, replay)

SCHEMA = "arc2_bank_bootstrap.v1"
DEFAULT_JSON = ROOT / "experiments" / "arc2-bank" / "bootstrap.json"
DEFAULT_RULES = ("b0", "v7_direct", "q27_first", "q27_agree")
DEFAULT_N_TASKS = (120, 240)
DEFAULT_N_BOOT = 4000
Z95 = 1.96
MDD_DEFINITION = ("MDD95 = 1.96 x sd_points of the task-resampled paired difference new - base: the board difference "
                  "(points) a single paired reading must exceed to be outside the 95 % interval of task-resampling noise")


def bootstrap_rules(bank: Bank, rules: list[str], base: str, n_tasks: list[int], n_boot: int, seed: int) -> dict:
    results: dict[str, Result] = {r: replay(bank, r) for r in rules}
    if base not in results:
        results[base] = replay(bank, base)
    boards = {r: [bootstrap_board(results[r], n, n_boot, seed) for n in n_tasks] for r in rules}
    diffs = {}
    for r in rules:
        if r == base:
            continue
        rows = []
        for n in n_tasks:
            d = bootstrap_difference(results[base], results[r], n, n_boot, seed)
            d["mdd95_points"] = round(Z95 * d["sd_points"], 2)
            rows.append(d)
        diffs[r] = rows
    return {"base": base, "n_tasks": n_tasks, "n_boot": n_boot, "seed": seed, "mdd_definition": MDD_DEFINITION,
            "rules": {r: results[r].summary() for r in results},
            "boards": boards, "differences": diffs}


def summary_lines(res: dict) -> list[str]:
    b = res["bank"]
    lines = [f"board bootstrap (E1): {b['n_tasks']} tasks, {b['n_outputs']} outputs; bank {b['fingerprint']}; "
             f"{res['n_boot']} task resamples, seed {res['seed']}",
             "", "rule | right / outputs (attempt_1) | bank points | board n | mean | sd | ci95 | mean outputs"]
    for r, rows in res["boards"].items():
        s = res["rules"][r]
        for row in rows:
            lines.append(f"{r} | {s['n_right']} / {s['n_outputs']} ({s['n_right_1']}) | {100 * s['score']:.2f} | {row['n_tasks_resampled']} | "
                         f"{row['mean_points']:.2f} | {row['sd_points']:.2f} | [{row['ci95_points'][0]:.2f}, "
                         f"{row['ci95_points'][1]:.2f}] | {row['mean_outputs']}")
    if res["differences"]:
        lines += ["", f"paired difference against {res['base']} (points)",
                  "rule | board n | mean | sd | ci95 | P(diff <= 0) | MDD95"]
        for r, rows in res["differences"].items():
            for d in rows:
                lines.append(f"{r} | {d['n_tasks_resampled']} | {d['mean_points']:+.2f} | {d['sd_points']:.2f} | "
                             f"[{d['ci95_points'][0]:+.2f}, {d['ci95_points'][1]:+.2f}] | {d['p_le_0']:.4f} | {d['mdd95_points']:.2f}")
        lines.append(res["mdd_definition"])
    return lines


def add_bank_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pools", nargs="+", default=None, metavar="DIR", help="pool directories (default: both bank parts)")
    ap.add_argument("--q27", default=str(Q27_BANK), help="the 27B bank directory")
    ap.add_argument("--challenges", default=str(CHALLENGES), help="challenges file (task sizes)")
    ap.add_argument("--solutions", default="", help="extra solutions file (default: the Kaggle copies beside the pools)")


def bank_from_args(args: argparse.Namespace) -> Bank:
    return load_bank(args.pools, args.q27, challenges=args.challenges or None, solutions=args.solutions or None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    ap.add_argument("--base", default="b0", help="the rule every other rule is paired with")
    ap.add_argument("--n-tasks", nargs="+", type=int, default=list(DEFAULT_N_TASKS), help="board sizes to resample to")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=str(DEFAULT_JSON), help="write the result here ('' to skip)")
    add_bank_args(ap)
    args = ap.parse_args(argv)
    t0 = time.time()
    bank = bank_from_args(args)
    res = {"schema": SCHEMA, "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "args": {"rules": args.rules, "base": args.base, "n_tasks": args.n_tasks, "n_boot": args.n_boot,
                    "seed": args.seed, "pools": bank.sources, "q27": args.q27, "json": args.json},
           "bank": {"fingerprint": bank.fingerprint(), "n_tasks": len(bank.tasks), "n_outputs": bank.n_outputs,
                    "n_q27_tasks": len(bank.q27), "sources": bank.sources, "solutions": bank.solutions_source}}
    res.update(bootstrap_rules(bank, args.rules, args.base, args.n_tasks, args.n_boot, args.seed))
    res["elapsed_s"] = round(time.time() - t0, 1)
    print("\n".join(summary_lines(res)))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
