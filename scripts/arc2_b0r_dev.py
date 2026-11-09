#!/usr/bin/env python3
"""B0R (experiments/arc2-bank/B0R_RULE.md) section 4: the choice on the dev pools. CPU only; no bank number.

    python scripts/arc2_b0r_dev.py                    # the 6 dev pools -> experiments/arc2-bank/b0r_dev.json

On each of the 6 dev pools (B0 on the 40 dev tasks of splits/arc2_eval_split_v1.json; 65 outputs each in Kaggle's
file, 390 in all; truth is the Kaggle solutions copy beside each pool):

- floor item 1: every candidate's field meanings (eval.arc2_bank.b0r_check). A failing candidate's output keeps
  kgmon in every rule (counted); a pool with more than 5% of its outputs failing leaves the choice;
- floor item 2: kgmon recomputed from the pool equals the submitted pair on every output, else the study stops;
- each of the three rules (b0r_zbeam, b0r_amax, b0r_borda) against kgmon's top two (b0_kgmon): right, gained, lost.

A rule qualifies at pooled net >= +3 with net >= 0 on at least 5 of the 6 pools (all but one when a pool left the
choice). The chosen rule is the qualifier with the highest pooled net, then fewer lost, then the order of
B0R_RULES; with no qualifier the study ends ("kgmon stays") and the bank is not read.

Recorded beside, deciding nothing: the per-pool table (H1 v1/v2 marked as the bank's setting), the split-half check
(the choice re-run on each half of the dev tasks; halves = the ids sorted by sha256(id + "b0r-half"), first and last
20), the right grid's rank under kgmon and each rule pooled on dev, and the distinct outputs gained and lost.
Dev numbers never enter the bank's replay ledger (experiments/arc2-bank/replay_ledger.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (B0R_RULES, CHALLENGES, RUNS, Bank, b0r_check, b0r_code_fingerprint,  # noqa: E402
                            b0r_ranking, get_rule, grids_equal, load_bank, replay, rule_b0, rule_b0_kgmon)

SCHEMA = "arc2_b0r_dev.v1"
BANK_DIR = ROOT / "experiments" / "arc2-bank"
OUT = BANK_DIR / "b0r_dev.json"
SPLIT = ROOT / "splits" / "arc2_eval_split_v1.json"
RULE_DOC = "experiments/arc2-bank/B0R_RULE.md"
RULE_FROZEN = "166b24a (frozen 2026-10-29 15:41 IST; status line bb4d8a5)"
# (run, setting, the bank's setting): the pools LEVERS' floor replays used (W1 left out: floor 0.05)
DEV_POOLS = (("kaggle-A2-pool-audit-v1", "128 sequences", False), ("kaggle-A2-pool-audit-v2", "128 sequences", False),
             ("kaggle-A2-H1-v1", "64 sequences", True), ("kaggle-A2-H1-v2", "64 sequences", True),
             ("kaggle-A2-H2-v1", "32 sequences", False), ("kaggle-A2-adapt-dev-v1", "64 sequences, adapted base", False))
QUALIFY_NET = 3
FLOOR_FRAC = 0.05
HALF_SALT = "b0r-half"
RANK_BUCKETS = ("1", "2", "3", "4", "5+")


def dev_ids(path: Path = SPLIT) -> list[str]:
    return sorted(json.loads(path.read_text(encoding="utf-8"))["splits"]["dev"])


def halves(ids: list[str], salt: str = HALF_SALT) -> tuple[list[str], list[str]]:
    """The ids sorted by sha256(id + salt): first half and last half."""
    s = sorted(ids, key=lambda t: hashlib.sha256((t + salt).encode()).hexdigest())
    n = len(s) // 2
    return s[:n], s[n:]


def floor(bank: Bank) -> dict:
    """Floor items 1 and 2 on one pool."""
    n_c, fields, fail_keys = 0, Counter(), []
    for o in bank.outputs():
        bad = False
        for c in o.candidates:
            n_c += 1
            f = b0r_check(c)
            fields.update(f)
            bad = bad or bool(f)
        if bad:
            fail_keys.append(o.key)
    mismatch = []
    for t in bank.task_ids:
        task = bank.tasks[t]
        for o, p, q in zip(task.outputs, rule_b0(task, bank, None), rule_b0_kgmon(task, bank, None)):
            if not (grids_equal(p[0], q[0]) and grids_equal(p[1], q[1])):
                mismatch.append(o.key)
    n = bank.n_outputs
    return {"n_outputs": n, "n_candidates": n_c, "candidate_failures": dict(sorted(fields.items())),
            "outputs_failing": len(fail_keys), "outputs_failing_keys": fail_keys,
            "outputs_failing_frac": round(len(fail_keys) / n, 4) if n else 0.0,
            "item1_pool_ok": (len(fail_keys) / n if n else 0.0) <= FLOOR_FRAC,
            "kgmon_vs_submitted_mismatch": mismatch, "item2_ok": not mismatch}


def cmp(base: dict, new: dict, tasks: set[str] | None = None) -> dict:
    """gained / lost / net of ``new`` against ``base`` (per_output dicts) on the shared outputs of ``tasks``."""
    keys = sorted(k for k in base if k in new and (tasks is None or base[k]["task_id"] in tasks))
    gained = [k for k in keys if new[k]["right"] and not base[k]["right"]]
    lost = [k for k in keys if base[k]["right"] and not new[k]["right"]]
    return {"n_outputs": len(keys), "base_right": sum(base[k]["right"] for k in keys),
            "right": sum(new[k]["right"] for k in keys), "gained": len(gained), "lost": len(lost),
            "net": len(gained) - len(lost), "gained_keys": gained, "lost_keys": lost}


def choose(table: dict[str, dict[str, dict]], pools: list[str], rules=B0R_RULES) -> dict:
    """Section 4's choice on ``table[pool][rule]`` over ``pools``."""
    need = max(len(pools) - 1, 1)
    pooled, qual = {}, {}
    for r in rules:
        rows = [table[p][r] for p in pools]
        pooled[r] = {k: sum(x[k] for x in rows) for k in ("n_outputs", "base_right", "right", "gained", "lost", "net")}
        pooled[r]["pools_net_ge_0"] = sum(1 for x in rows if x["net"] >= 0)
        qual[r] = pooled[r]["net"] >= QUALIFY_NET and pooled[r]["pools_net_ge_0"] >= need
    order = sorted((r for r in rules if qual[r]), key=lambda r: (-pooled[r]["net"], pooled[r]["lost"], rules.index(r)))
    return {"pools": pools, "pools_needed_net_ge_0": need, "pooled": pooled, "qualifies": qual, "order": order,
            "chosen": order[0] if order else None}


def rank_of_truth(bank: Bank, rules=B0R_RULES) -> dict:
    """{kgmon: Counter, <rule>: Counter} of the right grid's rank (RANK_BUCKETS) on the outputs whose pool holds it;
    a rule whose ranking does not apply (floor item 1) counts kgmon's rank."""
    out = {"kgmon": Counter(), **{r: Counter() for r in rules}}
    covered = 0
    for o in bank.outputs():
        truth = bank.truth(o)
        kg = o.ranked("rank_kgmon")
        pos = next((i for i, c in enumerate(kg) if truth is not None and grids_equal(c["grid"], truth)), None)
        if pos is None:
            continue
        covered += 1
        out["kgmon"][_bucket(pos + 1)] += 1
        for r in rules:
            order = b0r_ranking(o, r.removeprefix("b0r_")) or kg
            i = next(i for i, c in enumerate(order) if grids_equal(c["grid"], truth))
            out[r][_bucket(i + 1)] += 1
    return {"covered": covered, **{k: dict(v) for k, v in out.items()}}


def _bucket(rank: int) -> str:
    return str(rank) if rank <= 4 else "5+"


def run(pools=DEV_POOLS, *, runs: Path = RUNS, split: Path = SPLIT, challenges=CHALLENGES) -> dict:
    ids = dev_ids(split)
    info, table, ranks, distinct, po = [], {}, {}, {}, {}
    stopped = None
    for name, setting, bank_setting in pools:
        d = runs / name / "pool"
        bank = load_bank([d], None, challenges=challenges)
        if sorted(bank.task_ids) != ids:
            raise SystemExit(f"{d}: task ids differ from the 40 dev tasks of {split.name}")
        fl = floor(bank)
        info.append({"pool": name, "dir": str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d),
                     "setting": setting, "bank_setting": bank_setting, "n_tasks": len(bank.task_ids),
                     "n_outputs": bank.n_outputs, "solutions_source": bank.solutions_source, "floor": fl,
                     "in_choice": fl["item1_pool_ok"]})
        if not fl["item2_ok"] and stopped is None:
            stopped = f"floor item 2: kgmon differs from the submitted pair on {name}"
        po[name] = {"b0_kgmon": replay(bank, "b0_kgmon").per_output}
        table[name] = {}
        for r in B0R_RULES:
            po[name][r] = replay(bank, get_rule(r)).per_output
            table[name][r] = cmp(po[name]["b0_kgmon"], po[name][r])
        ranks[name] = rank_of_truth(bank)
    in_choice = [p["pool"] for p in info if p["in_choice"]]
    full = choose(table, in_choice)
    a, b = halves(ids)
    half_tables = {h: {p: {r: cmp(po[p]["b0_kgmon"], po[p][r], set(t)) for r in B0R_RULES} for p in table}
                   for h, t in (("A", a), ("B", b))}
    split_half = {"salt": HALF_SALT, "A": a, "B": b,
                  "choice_A": choose(half_tables["A"], in_choice), "choice_B": choose(half_tables["B"], in_choice)}
    ch = full["chosen"]
    split_half["chosen_qualifies_in_both"] = (None if ch is None else
                                              bool(split_half["choice_A"]["qualifies"][ch]
                                                   and split_half["choice_B"]["qualifies"][ch]))
    for r in B0R_RULES:
        g = Counter(k for p in in_choice for k in table[p][r]["gained_keys"])
        lo = Counter(k for p in in_choice for k in table[p][r]["lost_keys"])
        distinct[r] = {"outputs_gained": len(g), "outputs_lost": len(lo), "gained_pools": dict(sorted(g.items())),
                       "lost_pools": dict(sorted(lo.items()))}
    pooled_ranks = {"covered": sum(ranks[p]["covered"] for p in in_choice)}
    for k in ("kgmon",) + B0R_RULES:
        c = Counter()
        for p in in_choice:
            c.update(ranks[p][k])
        pooled_ranks[k] = {bkt: c.get(bkt, 0) for bkt in RANK_BUCKETS}
    if stopped:
        full = dict(full, chosen=None, order=[])
    return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rule_doc": RULE_DOC,
            "rule_frozen": RULE_FROZEN,
            "rules": {r: {"fingerprint": get_rule(r).fingerprint(), "code_fingerprint":
                          b0r_code_fingerprint(r.removeprefix("b0r_")), "description": get_rule(r).description}
                      for r in B0R_RULES},
            "qualify_rule": f"pooled net >= +{QUALIFY_NET} and net >= 0 on at least {full['pools_needed_net_ge_0']} "
                            f"of {len(in_choice)} pools; order by pooled net, then fewer lost, then rule order",
            "pools": info, "table": table, "choice": full, "chosen": full["chosen"], "stopped": stopped,
            "outcome": (f"stopped: {stopped}" if stopped else
                        f"chosen {full['chosen']} for the bank" if full["chosen"] else
                        "no qualifier: kgmon stays; the bank is not read"),
            "split_half": split_half, "rank_of_truth": {"pooled": pooled_ranks, "per_pool": ranks},
            "distinct_outputs": distinct}


def summary_lines(res: dict) -> list[str]:
    out = [f"B0R dev choice ({res['rule_doc']}, frozen {res['rule_frozen']})"]
    for p in res["pools"]:
        fl = p["floor"]
        out.append(f"  {p['pool']:<26} {p['setting']}{' [bank setting]' if p['bank_setting'] else ''}: "
                   f"{p['n_outputs']} outputs; floor 1 failing outputs {fl['outputs_failing']} "
                   f"({'in' if p['in_choice'] else 'OUT of'} the choice); floor 2 mismatches "
                   f"{len(fl['kgmon_vs_submitted_mismatch'])}")
    out.append("| track | pool | kgmon right | " + " | ".join(f"{r} right / +g -l net" for r in B0R_RULES) + " |")
    for p in res["pools"]:
        rows = res["table"][p["pool"]]
        base = rows[B0R_RULES[0]]["base_right"]
        out.append(f"| ARC-2 | {p['pool']}{' (bank setting)' if p['bank_setting'] else ''} | {base} | " + " | ".join(
            f"{rows[r]['right']} / +{rows[r]['gained']} -{rows[r]['lost']} {rows[r]['net']:+d}" for r in B0R_RULES) + " |")
    ch = res["choice"]
    pooled = ch["pooled"]
    out.append(f"| ARC-2 | pooled ({len(ch['pools'])} pools) | {pooled[B0R_RULES[0]]['base_right']} | " + " | ".join(
        f"{pooled[r]['right']} / +{pooled[r]['gained']} -{pooled[r]['lost']} {pooled[r]['net']:+d}" for r in B0R_RULES)
        + " |")
    out.append("  pools with net >= 0: " + ", ".join(f"{r} {pooled[r]['pools_net_ge_0']}" for r in B0R_RULES)
               + f" (need {ch['pools_needed_net_ge_0']})")
    out.append("  qualifies: " + ", ".join(f"{r} {'yes' if ch['qualifies'][r] else 'no'}" for r in B0R_RULES))
    out.append(f"  outcome: {res['outcome']}")
    sh = res["split_half"]
    out.append(f"  split half: A chooses {sh['choice_A']['chosen']}, B chooses {sh['choice_B']['chosen']}; "
               f"chosen rule qualifies in both: {sh['chosen_qualifies_in_both']}")
    rk = res["rank_of_truth"]["pooled"]
    out.append(f"  rank of the right grid, pooled on dev ({rk['covered']} covered outputs; ranks 1/2/3/4/5+):")
    for k in ("kgmon",) + B0R_RULES:
        out.append(f"    {k:<10} " + " ".join(str(rk[k][b]) for b in RANK_BUCKETS))
    for r in B0R_RULES:
        d = res["distinct_outputs"][r]
        out.append(f"  {r}: distinct outputs gained {d['outputs_gained']}, lost {d['outputs_lost']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    res = run()
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(res)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
