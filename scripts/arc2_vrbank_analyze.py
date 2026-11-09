"""VR-bank analysis: the pre-set rule of experiments/arc2-q27/VR_BANK_RULE.md (frozen 2026-10-27
20:15 IST, as written in 8c553a6) on the run archive. Prints the rule lines; writes vrbank_decision.json.

    python scripts/arc2_vrbank_analyze.py experiments/runs/kaggle-A2-VRbank-v1

Reads, under the archive: hybrid/p2/rep_<A|B>/vr/samples.jsonl (every stage's record), hybrid/submission_b0.json (the
bank's B0 attempts as the kernel wrote them; checked against eval/arc2_bank.py's rule b0), and Kaggle's challenges and
solutions for the 30 band tasks (hybrid/p2/_kaggle_*_vrbank_tasks.json; else the bank's copies). The merge is the
entry's: arc_harness.selection.two_attempts(task, samples, "vr", b0, "verified_first", True), from
kaggle/arc2/hybrid_vr/arc_harness (the v7-vr2 files the kernel embedded; checked by sha256 against the archive's
written copy when present).

Definitions (the rule's):
  first sample      a replicate's s1 sample of the task and its rescue (stage "rescue")
  R                 repair and rescue_repair samples; S2 = s2 and rescue_s2 samples
  right             a sample has an output right when its written grid or its program's output equals the truth
  net right (R)     an output (task, test index, replicate) that a sample in R has right and neither the replicate's
                    first sample nor B0's two attempts have right; summed over A and B (rule 1); likewise for S2
  final(S)          the entry's two attempts from samples S over B0; right(S) = outputs right in either attempt
  verified clean    a sample whose program passed every demonstration and is not a lookup table (hardcoded_risk);
                    precision over its test outputs (rule 2)
Validity: both replicates ran s1 on >= 27 of 30 tasks; median s1 completion tokens >= 1,000; empty or zero-token
replies <= 5% of records; no sealed task among the tasks read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / "kaggle" / "arc2" / "hybrid_vr"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(ROOT))

from arc_harness.grids import hardcoded_risk  # noqa: E402
from arc_harness.selection import two_attempts  # noqa: E402

REPLICATES = ("A", "B")
FIRST = ("s1", "rescue")
R_STAGES = ("repair", "rescue_repair")
S2_STAGES = ("s2", "rescue_s2")
RESCUED = ("rescue", "rescue_repair", "rescue_s2")
PLACEHOLDER = [[0]]


def read_samples(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue                       # a line cut by the kill
        out.setdefault(s["task_id"], []).append(s)
    return out


def sample_right(s: dict, j: int, truth) -> bool:
    g = s.get("grids") or []
    po = (s.get("verify") or {}).get("program_outputs") or []
    return (j < len(g) and g[j] is not None and g[j] == truth) or (j < len(po) and po[j] is not None and po[j] == truth)


def b0_pool(sub: dict, tasks: dict) -> dict:
    """hybrid_vr.b0_pool: B0's attempts per test without placeholders."""
    pool = {}
    for t, task in tasks.items():
        ent = sub.get(t) or []
        pool[t] = [[g for g in ((ent[j] or {}).get("attempt_1"), (ent[j] or {}).get("attempt_2")) if g is not None and g != PLACEHOLDER]
                   if j < len(ent) else [] for j in range(len(task["test"]))]
    return pool


def right_set(tasks, samples, pool, sols, rep) -> set:
    out = set()
    for t, task in tasks.items():
        att = two_attempts(task, samples.get(t, []), "vr", pool.get(t), "verified_first", True)
        for j, e in enumerate(att):
            if sols[t][j] in (e.get("attempt_1"), e.get("attempt_2")):
                out.add((t, j, rep))
    return out


def net_right(tasks, samples, pool, sols, rep, stages) -> tuple[set, int]:
    """Outputs a sample of `stages` has right while neither the first sample nor B0's pool does; tokens of `stages`."""
    out, tok = set(), 0
    for t, task in tasks.items():
        ss = samples.get(t, [])
        tok += sum(int(s.get("gen_tokens") or 0) for s in ss if s["stage"] in stages)
        for j in range(len(task["test"])):
            truth = sols[t][j]
            if not any(sample_right(s, j, truth) for s in ss if s["stage"] in stages):
                continue
            if any(sample_right(s, j, truth) for s in ss if s["stage"] in FIRST):
                continue
            if truth in (pool.get(t) or [[]] * (j + 1))[j]:
                continue
            out.add((t, j, rep))
    return out, tok


def analyze(run: Path) -> dict:
    p2 = run / "hybrid" / "p2"
    ch_path = p2 / "_kaggle_challenges_vrbank_tasks.json"
    so_path = p2 / "_kaggle_solutions_vrbank_tasks.json"
    if not ch_path.is_file():
        ch_path = ROOT / "experiments/runs/kaggle-A2-q27-bank-v1/q27d/_kaggle_challenges_q27_tasks.json"
    if not so_path.is_file():
        so_path = ROOT / "experiments/runs/kaggle-A2-q27-bank-v1/q27d/_kaggle_solutions_audit_tasks.json"
    tasks = json.loads(ch_path.read_text(encoding="utf-8"))
    sols_all = json.loads(so_path.read_text(encoding="utf-8"))
    from eval import arc2_bank as B
    sealed = set()
    for d in B.BANK_SEALED:
        m = Path(d) / B.SEALED_MANIFEST
        if m.is_file():
            sealed |= set(json.loads(m.read_text(encoding="utf-8"))["task_ids"])
    assert len(tasks) == 30 and not set(tasks) & sealed, "the band tasks, none sealed"
    sols = {t: sols_all[t] for t in tasks}
    n_out = sum(len(tasks[t]["test"]) for t in tasks)
    assert all(len(sols[t]) == len(tasks[t]["test"]) for t in tasks)
    # B0: the kernel's file, checked against the bank's rule b0
    bank = B.load_bank(tasks=sorted(tasks), q27_dir=B.Q27_BANK)
    b0_file = run / "hybrid" / "submission_b0.json"
    b0_sub = json.loads(b0_file.read_text(encoding="utf-8")) if b0_file.is_file() else {}
    b0_bank = {t: [{"attempt_1": a1, "attempt_2": a2} for a1, a2 in B.rule_b0(bank.tasks[t], bank, None)] for t in tasks}
    b0_matches = all(b0_sub.get(t) == b0_bank[t] for t in tasks) if b0_sub else None
    pool = b0_pool(b0_sub or b0_bank, tasks)
    b0_right = {(t, j) for t in tasks for j in range(len(tasks[t]["test"])) if sols[t][j] in pool[t][j]}
    v7 = B.replay(bank, "v7_direct")
    v7_right = B.subset_score(v7, tasks)["n_right"]

    reps, valid_bits = {}, {}
    net_r, net_s2, tok_r, tok_s2, repair_rounds = set(), set(), 0, 0, 0
    eff = {"R": {"gained": 0, "lost": 0}, "S2": {"gained": 0, "lost": 0}}
    vc_total, vc_right, vc_stage = 0, 0, {}
    resc_total, resc_right, cut_tails = 0, 0, 0
    final_right = {}
    n_records = n_empty = 0
    for rep in REPLICATES:
        samples = read_samples(p2 / f"rep_{rep}" / "vr" / "samples.jsonl")
        allS = [s for v in samples.values() for s in v]
        n_records += len(allS)
        n_empty += sum(1 for s in allS if not int(s.get("gen_tokens") or 0))
        s1 = [s for s in allS if s["stage"] == "s1"]
        s1_tasks = {s["task_id"] for s in s1}
        s1_tok = [int(s.get("gen_tokens") or 0) for s in s1]
        valid_bits[rep] = {"s1_tasks": len(s1_tasks), "s1_median_tokens": statistics.median(s1_tok) if s1_tok else 0}
        nr, tr = net_right(tasks, samples, pool, sols, rep, R_STAGES)
        ns, ts = net_right(tasks, samples, pool, sols, rep, S2_STAGES)
        net_r |= nr
        net_s2 |= ns
        tok_r += tr
        tok_s2 += ts
        repair_rounds += sum(1 for s in allS if s["stage"] == "repair")
        full = right_set(tasks, samples, pool, sols, rep)
        final_right[rep] = full
        for key, stages in (("R", R_STAGES), ("S2", S2_STAGES)):
            without = right_set(tasks, {t: [s for s in v if s["stage"] not in stages] for t, v in samples.items()}, pool, sols, rep)
            eff[key]["gained"] += len(full - without)
            eff[key]["lost"] += len(without - full)
        for s in allS:
            t = s["task_id"]
            v = s.get("verify") or {}
            if v.get("passed") and not hardcoded_risk(s.get("program") or ""):
                po = v.get("program_outputs") or []
                for j in range(len(tasks[t]["test"])):
                    ok = j < len(po) and po[j] == sols[t][j]
                    vc_total += 1
                    vc_right += ok
                    st = vc_stage.setdefault(s["stage"], [0, 0])
                    st[0] += 1
                    st[1] += ok
            if s["stage"] in RESCUED:
                for j in range(len(tasks[t]["test"])):
                    resc_total += 1
                    resc_right += sample_right(s, j, sols[t][j])
            if s.get("finish") == "cut" and s.get("reasoning_tail"):
                cut_tails += 1
        stages = {}
        for s in allS:
            d = stages.setdefault(s["stage"], {"n": 0, "tokens": 0, "finish": {}})
            d["n"] += 1
            d["tokens"] += int(s.get("gen_tokens") or 0)
            d["finish"][s["finish"]] = d["finish"].get(s["finish"], 0) + 1
        reps[rep] = {"samples": len(allS), "tasks": len(samples), "stages": stages, "pass2_final": len(full),
                     "net_right_R": len(nr), "net_right_S2": len(ns), "tokens": sum(int(s.get("gen_tokens") or 0) for s in allS)}

    valid = (all(v["s1_tasks"] >= 27 and v["s1_median_tokens"] >= 1000 for v in valid_bits.values())
             and n_records > 0 and n_empty <= 0.05 * n_records)
    n1 = len(net_r)
    if not valid:
        rule1 = "no decision (validity failed): report only"
    elif repair_rounds < 20:
        rule1 = f"too few repair rounds ({repair_rounds} < 20): report only; v8 keeps v7-vr2's setting"
    elif n1 >= 4:
        rule1 = "keep: the repair round stays in v8 at its current caps"
    elif n1 <= 1:
        rule1 = "drop: the repair round leaves v8; its tokens go to second samples on the shortest quarter"
    else:
        rule1 = "report only (2-3): v8 keeps v7-vr2's setting"
    prec = vc_right / vc_total if vc_total else None
    if not valid:
        rule2 = "no decision (validity failed): report only"
    elif vc_total < 10:
        rule2 = f"too few verified-clean outputs ({vc_total} < 10): report only; v8 keeps verified_first"
    elif prec >= 0.90:
        rule2 = "keep verified_first in v8"
    elif prec < 0.75:
        rule2 = "v8 merges with R0"
    else:
        rule2 = "report only (75-90%): v8 keeps verified_first"
    rule3 = ("the paired rescue replay is built (>= 15 cut samples with a saved tail)" if cut_tails >= 15 else
             f"{cut_tails} cut samples with a tail (< 15): the replay runs on what there is, report only")
    agree = len({(t, j) for (t, j, r) in final_right["A"]} ^ {(t, j) for (t, j, r) in final_right["B"]})
    per_m = lambda n, tok: round(n / tok * 1e6, 2) if tok else None  # noqa: E731
    return {
        "rule": "experiments/arc2-q27/VR_BANK_RULE.md (frozen 2026-10-27 20:15 IST, as written in 8c553a6)",
        "valid": valid, "validity": {"replicates": valid_bits, "records": n_records, "empty_or_zero_token": n_empty,
                                     "sealed_read": False},
        "rule1_repair": {"decision": rule1, "net_right_outputs": n1, "repair_rounds": repair_rounds,
                         "outputs": sorted(f"{t}[{j}]{r}" for t, j, r in net_r), "tokens_R": tok_r,
                         "tokens_per_net_output": round(tok_r / n1) if n1 else None, "per_million": per_m(n1, tok_r)},
        "rule2_verified_first": {"decision": rule2, "verified_clean_outputs": vc_total, "right": vc_right,
                                 "precision": round(prec, 3) if prec is not None else None,
                                 "per_stage": {k: {"outputs": v[0], "right": v[1]} for k, v in vc_stage.items()}},
        "rule3_rescue_replay": {"decision": rule3, "cut_with_tail": cut_tails},
        "report_only": {
            "repair_effect_on_answers": eff["R"], "second_sample_effect_on_answers": eff["S2"],
            "second_sample_net_right": len(net_s2), "tokens_S2": tok_s2, "S2_per_million": per_m(len(net_s2), tok_s2),
            "rescue_precision": {"outputs": resc_total, "right": resc_right,
                                 "share": round(resc_right / resc_total, 3) if resc_total else None},
            "pass2": {"outputs": n_out, "b0_alone": len(b0_right), "v7_direct_bank_rule": v7_right,
                      **{f"final_{r}": reps[r]["pass2_final"] for r in REPLICATES}},
            "replicate_disagreement_outputs": agree, "b0_file_matches_bank": b0_matches,
        },
        "replicates": reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    res = analyze(a.run)
    (a.run / "vrbank_decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    r1, r2, r3, ro = res["rule1_repair"], res["rule2_verified_first"], res["rule3_rescue_replay"], res["report_only"]
    print(f"validity: {'ok' if res['valid'] else 'FAILED'} {res['validity']}")
    print(f"rule 1 (repair round): {r1['decision']} | net right {r1['net_right_outputs']} over A and B, "
          f"{r1['repair_rounds']} repair rounds, {r1['tokens_R']} tokens ({r1['per_million']} per million)")
    print(f"rule 2 (verified-first): {r2['decision']} | precision {r2['precision']} on {r2['verified_clean_outputs']} outputs")
    print(f"rule 3 (rescue replay): {r3['decision']}")
    print(f"report: pass@2 {ro['pass2']}; repair effect {ro['repair_effect_on_answers']}; second samples net right "
          f"{ro['second_sample_net_right']} ({ro['S2_per_million']} per million), effect {ro['second_sample_effect_on_answers']}; "
          f"rescue precision {ro['rescue_precision']}; A/B disagree on {ro['replicate_disagreement_outputs']} outputs; "
          f"B0 file = bank b0: {ro['b0_file_matches_bank']}")
    print(f"wrote {a.run / 'vrbank_decision.json'}")


if __name__ == "__main__":
    main()
