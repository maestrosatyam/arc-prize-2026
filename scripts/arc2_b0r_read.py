#!/usr/bin/env python3
"""B0R (experiments/arc2-bank/B0R_RULE.md) section 5: the bank reading of the rule chosen on dev. CPU only.

    python scripts/arc2_b0r_read.py --rule b0r_amax          # -> experiments/arc2-bank/b0r_result.json

On the 145 unsealed outputs (both bank parts' pool/ and the 27B bank; pool_sealed/ refused), Kaggle's solutions copy
as truth:

- G1 (gate): the rule against b0 (B0 as submitted, 40), gate v2: net >= 3 and lost <= 1;
- G2 (gate): <rule>_v7 against v7_direct (45), gate v2;
- G3 (gate): on the bank's tasks outside the 40 dev tasks (61 tasks, 80 outputs), net >= 0 in G1's and G2's
  comparison.

Outcome (section 8): G2 and G3 pass = entry candidate; G1 and G3 pass with G2 failed = Stage 1's replay grid only;
otherwise kgmon stays. A gate pass on the 145 with G3 failed is reported as "gain on the dev tasks only".

Recorded beside: floor items 1 and 2 on the bank, the right grid's rank under kgmon and the rule on the covered
outputs, the sign test and task bootstrap (compare), the board-point bootstrap (bootstrap_difference, 120 tasks),
and the expected false positives (null_calibration.json). The rule must be b0r_dev.json's chosen rule and both its
rows must be in RULES.md. The full-bank ledger lines come from scripts/arc2_bank_replay.py (--rule <rule>; --rule
<rule>_v7 --base v7_direct).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.arc2_bank import (B0R_RULES, Bank, b0r_code_fingerprint, bootstrap_difference, compare, get_rule,  # noqa: E402
                            load_bank, replay)
from arc2_b0r_dev import OUT as DEV_OUT, RANK_BUCKETS, cmp, dev_ids, floor, rank_of_truth  # noqa: E402
from arc2_bank_replay import NULL_CALIBRATION, RULES_FILE, false_positive_rates, registered_rules  # noqa: E402

SCHEMA = "arc2_b0r_result.v1"
OUT = ROOT / "experiments" / "arc2-bank" / "b0r_result.json"


def read_b0r(bank: Bank, rule: str, dev: list[str], *, n_boot: int = 2000, seed: int = 0,
             null_calibration: Path = NULL_CALIBRATION) -> dict:
    if rule not in B0R_RULES:
        raise ValueError(f"{rule} is not a B0R rule ({', '.join(B0R_RULES)})")
    held = sorted(t for t in bank.task_ids if t not in set(dev))
    b0, v7 = replay(bank, "b0"), replay(bank, "v7_direct")
    new, new_v7 = replay(bank, get_rule(rule)), replay(bank, get_rule(rule + "_v7"))
    g1 = compare(b0, new, n_boot=n_boot, seed=seed)
    g2 = compare(v7, new_v7, n_boot=n_boot, seed=seed)
    g3 = {"n_tasks": len(held), "g1": cmp(b0.per_output, new.per_output, set(held)),
          "g2": cmp(v7.per_output, new_v7.per_output, set(held))}
    g3["pass"] = g3["g1"]["net"] >= 0 and g3["g2"]["net"] >= 0
    p1, p2 = g1["gate_v2"]["pass"], g2["gate_v2"]["pass"]
    if p2 and g3["pass"]:
        outcome = "entry candidate: joins Stage 1's replay grid; an entry build goes to Stage 2 only by a separate decision"
    elif p1 and g3["pass"]:
        outcome = "Stage 1's replay grid only; no entry on its own"
    elif (p1 or p2) and not g3["pass"]:
        outcome = "gain on the dev tasks only: decides nothing; kgmon stays"
    else:
        outcome = "kgmon stays; Stage 1's grid carries kgmon and probmul3"
    rk = rank_of_truth(bank, (rule,))
    return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule": rule,
            "rule_fingerprint": new.rule_fingerprint, "rule_v7_fingerprint": new_v7.rule_fingerprint,
            "code_fingerprint": b0r_code_fingerprint(rule.removeprefix("b0r_")),
            "bank_fingerprint": bank.fingerprint(), "sources": bank.sources, "solutions_source": bank.solutions_source,
            "n_tasks": len(bank.task_ids), "n_outputs": bank.n_outputs, "held_out_tasks": held,
            "floor": floor(bank),
            "G1": {"base": "b0", "base_right": g1["base_right"], "right": g1["new_right"], "gained": g1["gained"],
                   "lost": g1["lost"], "net": g1["net"], "gate_v2": p1, "gate_v1": g1["gate_v1"]["pass"],
                   "sign_test_p": g1["sign_test_p"], "gained_keys": g1["gained_keys"], "lost_keys": g1["lost_keys"],
                   "task_level": g1["task_level"], "board": bootstrap_difference(b0, new, seed=seed)},
            "G2": {"base": "v7_direct", "base_right": g2["base_right"], "right": g2["new_right"],
                   "gained": g2["gained"], "lost": g2["lost"], "net": g2["net"], "gate_v2": p2,
                   "gate_v1": g2["gate_v1"]["pass"], "sign_test_p": g2["sign_test_p"],
                   "gained_keys": g2["gained_keys"], "lost_keys": g2["lost_keys"], "task_level": g2["task_level"],
                   "board": bootstrap_difference(v7, new_v7, seed=seed)},
            "G3": g3, "outcome": outcome,
            "rank_of_truth": {"covered": rk["covered"],
                              **{k: {b: rk[k].get(b, 0) for b in RANK_BUCKETS} for k in ("kgmon", rule)}},
            "false_positive_rates": false_positive_rates(null_calibration)}


def summary_lines(res: dict) -> list[str]:
    r = res["rule"]
    g1, g2, g3 = res["G1"], res["G2"], res["G3"]
    fl = res["floor"]
    out = [f"B0R bank reading: {r} on {res['n_outputs']} outputs of {res['n_tasks']} tasks (bank {res['bank_fingerprint']})",
           f"  floor: item 1 failing outputs {fl['outputs_failing']} of {fl['n_outputs']}; item 2 mismatches "
           f"{len(fl['kgmon_vs_submitted_mismatch'])}",
           "| track | line | base | rule | gained | lost | net | gate |",
           f"| ARC-2 | G1 B0 alone | b0 {g1['base_right']} | {g1['right']} | {g1['gained']} | {g1['lost']} | "
           f"{g1['net']:+d} | v2 {'PASS' if g1['gate_v2'] else 'FAIL'} |",
           f"| ARC-2 | G2 merged | v7_direct {g2['base_right']} | {g2['right']} | {g2['gained']} | {g2['lost']} | "
           f"{g2['net']:+d} | v2 {'PASS' if g2['gate_v2'] else 'FAIL'} |",
           f"| ARC-2 | G3 held out, B0 alone | b0 {g3['g1']['base_right']} of {g3['g1']['n_outputs']} | "
           f"{g3['g1']['right']} | {g3['g1']['gained']} | {g3['g1']['lost']} | {g3['g1']['net']:+d} | |",
           f"| ARC-2 | G3 held out, merged | v7_direct {g3['g2']['base_right']} of {g3['g2']['n_outputs']} | "
           f"{g3['g2']['right']} | {g3['g2']['gained']} | {g3['g2']['lost']} | {g3['g2']['net']:+d} | "
           f"net >= 0 both {'PASS' if g3['pass'] else 'FAIL'} |",
           f"  outcome: {res['outcome']}",
           f"  board points (120-task bootstrap): G1 {g1['board']['mean_points']:+.2f} (sd {g1['board']['sd_points']}), "
           f"G2 {g2['board']['mean_points']:+.2f} (sd {g2['board']['sd_points']})",
           f"  sign test p: G1 {g1['sign_test_p']}, G2 {g2['sign_test_p']}"]
    rk = res["rank_of_truth"]
    out.append(f"  rank of the right grid on the {rk['covered']} covered outputs (1/2/3/4/5+): kgmon "
               + " ".join(str(rk["kgmon"][b]) for b in RANK_BUCKETS) + f"; {r} "
               + " ".join(str(rk[r][b]) for b in RANK_BUCKETS))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rule", required=True, choices=B0R_RULES)
    ap.add_argument("--dev", default=str(DEV_OUT), help="the dev choice (b0r_dev.json)")
    ap.add_argument("--rules-file", default=str(RULES_FILE))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    dev = json.loads(Path(args.dev).read_text(encoding="utf-8"))
    if dev.get("chosen") != args.rule:
        raise SystemExit(f"refused: {args.rule} is not the dev choice ({dev.get('outcome')})")
    reg = registered_rules(Path(args.rules_file))
    missing = [n for n in (args.rule, args.rule + "_v7") if n not in reg]
    if missing:
        raise SystemExit(f"refused: no dated RULES.md row for {', '.join(missing)}")
    res = read_b0r(load_bank(), args.rule, dev_ids(), n_boot=args.n_boot, seed=args.seed)
    res["dev_choice"] = {"file": str(args.dev), "utc": dev.get("utc"), "chosen": dev.get("chosen")}
    res["registration"] = {n: reg[n] for n in (args.rule, args.rule + "_v7")}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(res)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
