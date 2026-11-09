#!/usr/bin/env python3
"""AMAX (experiments/arc2-bank/AMAX_RULE.md) section 4: the bank reading of B0R's amax for the amax learning entry.
CPU only.

    python scripts/arc2_amax_read.py          # -> experiments/arc2-bank/amax_result.json

The B0R reading (scripts/arc2_b0r_read.py read_b0r) on the 145 unsealed outputs, Kaggle's solutions copy as truth,
with this rule's gate instead of B0R's section 8:

- G1: b0r_amax against b0 (B0 as submitted, kgmon, 40);
- G2: b0r_amax_v7 against v7_direct (45);
- gate: net >= 0 in both. Pass = build the entry and its commit run; fail = stop, and the result is
  reported;
- no-change stop (added 23:17 IST): amax's attempt 2 equal to kgmon's on every output (0 differ) stops before
  the build even when the gate passes.

Recorded beside, no decision: gate v2 on G1 and G2, G3 (the 80 held-out outputs), the sign test, the task and board
bootstraps, the right grid's rank, and how many outputs amax's attempt 2 changes against kgmon's and zbeam's. Refused
unless the rule's status line reads FROZEN and the rule and code fingerprints are the ones the rule names. The
ledger lines come from scripts/arc2_bank_replay.py (--rule b0r_amax; --rule b0r_amax_v7 --base v7_direct).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.arc2_bank import Bank, _pair, b0r_code_fingerprint, b0r_pairs, get_rule, grid_key, load_bank  # noqa: E402
from arc2_b0r_dev import dev_ids  # noqa: E402
from arc2_b0r_read import read_b0r, summary_lines  # noqa: E402

RULE_DOC = ROOT / "experiments" / "arc2-bank" / "AMAX_RULE.md"
OUT = ROOT / "experiments" / "arc2-bank" / "amax_result.json"
RULE = "b0r_amax"
RULE_FINGERPRINT = "1e21aa4a7eaf41d4"     # b0r_dev.json's, the dev read of 8228b8c
CODE_FINGERPRINT = "81f5ea0f02b137e9"     # AMAX_RULE.md section 2: b0r_check's n_aug form, the same at 8
PASS = "build: the amax entry and one commit run (23:12 IST Nov 1); the submission is a separate decision"
STOP = "stop: no build; the result is reported"
NO_CHANGE = ("stop (no-change, section 4): amax's attempt 2 equals kgmon's on every output; no build, the result is "
             "reported")


def decide(g1_net: int, g2_net: int, n_differ: int) -> tuple[bool, str]:
    if not (g1_net >= 0 and g2_net >= 0):
        return False, STOP
    if n_differ == 0:
        return False, NO_CHANGE
    return True, PASS


def attempt2_changes(bank: Bank) -> dict:
    """Outputs with 3+ candidates, and where amax's attempt 2 differs from kgmon's (B0's) and from zbeam's."""
    n3 = vs_kgmon = vs_zbeam = 0
    for t in bank.task_ids:
        task = bank.tasks[t]
        amax, zbeam = b0r_pairs(task, "amax"), b0r_pairs(task, "zbeam")
        for o, a, z in zip(task.outputs, amax, zbeam):
            n3 += len(o.candidates) >= 3
            vs_kgmon += grid_key(a[1]) != grid_key(_pair(o.attempt_1, o.attempt_2)[1])
            vs_zbeam += grid_key(a[1]) != grid_key(z[1])
    return {"outputs_3plus": n3, "attempt2_differs_from_kgmon": vs_kgmon, "attempt2_differs_from_zbeam": vs_zbeam}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rule-doc", default=str(RULE_DOC))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    status = next((ln for ln in Path(args.rule_doc).read_text(encoding="utf-8").splitlines()
                   if ln.startswith("Status:")), "")
    if not status.startswith("Status: FROZEN"):
        raise SystemExit(f"refused: {args.rule_doc} is not frozen ({status or 'no status line'})")
    fp, cfp = get_rule(RULE).fingerprint(), b0r_code_fingerprint("amax")
    if (fp, cfp) != (RULE_FINGERPRINT, CODE_FINGERPRINT):
        raise SystemExit(f"refused: fingerprints {fp} / {cfp}, the rule names {RULE_FINGERPRINT} / {CODE_FINGERPRINT}")
    bank = load_bank()
    res = read_b0r(bank, RULE, dev_ids(), n_boot=args.n_boot, seed=args.seed)
    res["schema"] = "arc2_amax_result.v1"
    res["rule_doc"] = str(Path(args.rule_doc).relative_to(ROOT)) if Path(args.rule_doc).is_absolute() else args.rule_doc
    res["rule_status"] = status
    res["gate"] = "AMAX_RULE.md section 4: net >= 0 in G1 and in G2; 0 outputs where attempt 2 differs = stop"
    res["attempt2_changes"] = attempt2_changes(bank)
    res["gate_pass"], res["outcome"] = decide(res["G1"]["net"], res["G2"]["net"],
                                              res["attempt2_changes"]["attempt2_differs_from_kgmon"])
    res["beside"] = "gate v2, G3 and the bootstraps are recorded only; they decide nothing here"
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(res)).replace("B0R bank reading", "AMAX bank reading"))
    ch = res["attempt2_changes"]
    print(f"  gate (net >= 0 in G1 and G2, 1+ attempt 2 changed): {'PASS' if res['gate_pass'] else 'FAIL'}")
    print(f"  attempt 2: {ch['outputs_3plus']} outputs with 3+ candidates; differs from kgmon on "
          f"{ch['attempt2_differs_from_kgmon']}, from zbeam on {ch['attempt2_differs_from_zbeam']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
