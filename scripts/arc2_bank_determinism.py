#!/usr/bin/env python3
"""Compare the two candidate pools of the B0 determinism check (kaggle/arc2/bank, variant det) offline.

The kernel writes determinism.json itself; this script recomputes the comparison from the downloaded pool
directories with kaggle/arc2/bank/arc_bank_compare.py, so the reading can be reproduced from the archive.

Reading (kaggle/arc2/bank/det/PROVENANCE.json):
  identical         fixed seeds reproduce B0 bit for bit: one board reading per build is a method effect on fixed tasks
  selection-stable  the selected pairs reproduce though scores drift
  differs           seeding does not remove the run-to-run variation

    python scripts/arc2_bank_determinism.py --pool-a <archive>/pool_a --pool-b <archive>/pool_b [--tasks a,b,...] [--json OUT]

Without --tasks the task ids are the union of the pool files' names.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "kaggle" / "arc2" / "bank" / "arc_bank_compare.py"


def _load():
    spec = importlib.util.spec_from_file_location("arc_bank_compare", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arc_bank_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-a", required=True)
    ap.add_argument("--pool-b", required=True)
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: every pool file in either directory)")
    ap.add_argument("--json", default="", help="write the full report here")
    args = ap.parse_args(argv)
    cmp = _load()
    if args.tasks:
        ids = args.tasks.split(",")
    else:
        names = set()
        for d in (args.pool_a, args.pool_b):
            if os.path.isdir(d):
                names |= {n[:-5] for n in os.listdir(d) if n.endswith(".json") and not n.startswith("_")}
        ids = sorted(names)
    report = cmp.compare_pools(args.pool_a, args.pool_b, ids)
    print("\n".join(cmp.summary_lines(report)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
