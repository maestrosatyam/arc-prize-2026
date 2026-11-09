#!/usr/bin/env python3
"""Paired analysis of the R1 / R1b / R1c / R1d / R1e recovery runs: every arm against A0 per case.

  python scripts/arc3_recovery_analyze.py r1_results.jsonl [r1b_results.jsonl r1c_results.jsonl ...]
      [--json out.json] [--permutations 20000] [--seed 0] [--labels labels_rule_A.csv]

Several results files are merged by (case, arm); a later file replaces an
earlier record of the same case-arm (reported under ``duplicates``).

Per comparison (arm vs A0): clear rates, discordant pairs (b = arm cleared and
A0 did not, c = A0 cleared and arm did not), exact McNemar p (two-sided binomial
on the discordant pairs), median actions to clear among cleared case-arms, and
the paired median of (arm - A0) actions among cases both arms cleared. A0b, the
second sample of the unchanged harness, is compared the same way: its
discordance with A0 is the reproducibility floor against which A1, A2 and A3
are read. Case-arms that did not run (skipped_hard_stop, replay_mismatch,
error) drop the case from the pair they belong to.

Against both baseline samples: every arm other than A0 and A0b is also paired
with A0b, with "A0 or A0b" (the case counts as cleared by the baseline when
either sample cleared it) and with "A0 and A0b" (both samples cleared it); the
composite pairs use the cases on which A0, A0b and the arm all ran. For a
composite control, actions to clear is the fewer of the clearing samples'
actions ("or") or the larger ("and").

A4 (dose-matched trim of the restored history): the cases whose restored conversation the trim
shortened, with the median messages and estimated tokens before and after, and
A4, A0 and A0b clear counts on those cases alone.

Across all arms: per-arm clear rates, the union of clears (cases cleared by at
least one arm, by every arm) with the union expected from independent draws at
the observed per-arm rates, and a permutation test of "arms do not matter":
arm labels are shuffled within each case and the between-arm spread of clear
rates (sum over arms of n_a (p_a - p)^2) is compared with the observed spread.

A0L (R1d: unchanged harness, 160 actions / 30 min per case-arm) is left out of
the sections above, which compare arms at the R1 budget, and gets a time
dose-response section instead. Its clears within the R1 budget ("A0L at 15
min") are the cleared case-arms with clear_actions <= 80 and clear_wall_s <= 900
(actions after the cut and seconds after the restore at the clearing action;
actions_used and wall_s stand in for rows without them); "A0L at 30 min" is
every A0L clear (the run itself stops at 160 actions / 1800 s). A0L at 15 min
is paired with A0 and A0b (it should sit within the reproducibility floor), A0L
at 30 min with A0, A0b, "A0 or A0b" and A0L at 15 min (exact McNemar), and the
levels cleared only after the R1 budget are listed. Decision rule
(experiments/arc3-recovery/DESIGN.md, R1d): gain = clears at 30 min - clears at
15 min; >= 5 time-limited, <= 2 competence-limited, otherwise inconclusive.

A6 (R1e: repertoire sweep) gets a section of its own when present. The
baseline samples are A0, A0b and A0c (A0c: a third sample of the unchanged
harness in the same run as A6). A6 is paired (exact McNemar) with each sample,
with "any baseline" (cleared by at least one sample) and "every baseline"
(cleared by all), on the cases where A6 and every present sample ran; the same
pairs are repeated within each rule-discovery group of
experiments/arc3-failure-annotation/labels_rule_A.csv (22 partial, 11 yes, 2
no). Fisher exact tests (two-sided, unpaired): A6's clear rate on partial vs
yes cases, the same for A0c, and the share of cases A6 cleared and A0c did not
(A6's gain over the same-run control) on partial vs yes. Decision rule
(DESIGN.md, R1e): build a full 25-game P0+A6 notebook if A6 clears at least
the best baseline sample + 4 on all cases, or at least the best sample + 3 on
the partial cases with no loss on the yes cases (A6 on the yes cases at least
the median of the samples there); otherwise stop.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = ROOT / "experiments" / "arc3-failure-annotation" / "labels_rule_A.csv"

NOT_RUN = {"skipped_hard_stop", "replay_mismatch", "error", None}
CONTROL = "A0"
ARM_ORDER = ("A0", "A0b", "A0c", "A1", "A2", "A3", "A4", "A5", "A6", "A0L")
BASELINES = ("A0", "A0b")
EITHER = "A0 or A0b"
BOTH = "A0 and A0b"
LONG_ARM = "A0L"                       # R1d: unchanged harness, doubled budget
LONG_ARMS = (LONG_ARM,)                # not compared at the R1 budget
R1_BUDGET = (80, 900.0)                # actions, wall seconds per case-arm (R1, R1b, R1c)
R1D_BUDGET = (160, 1800.0)
LONG_AT_15 = "A0L at 15 min"
LONG_AT_30 = "A0L at 30 min"
TIME_LIMITED_GAIN = 5                  # >= this many more clears at 30 than at 15 min: time-limited
COMPETENCE_LIMITED_GAIN = 2            # <= this many: competence-limited
SWEEP_ARM = "A6"                       # R1e: repertoire sweep
SWEEP_BASELINES = ("A0", "A0b", "A0c")  # unchanged-harness samples (A0c in the same run as A6)
SAME_RUN_CONTROL = "A0c"
ANY_BASELINE = "any baseline"
EVERY_BASELINE = "every baseline"
SWEEP_GAIN_ALL = 4                     # A6 >= best sample + this on all cases: build the P0+A6 notebook
SWEEP_GAIN_PARTIAL = 3                 # or >= best sample + this on the partial cases, no loss on yes


def load_results(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def merge_results(groups: list[list[dict]]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Merge results by (case, arm); a later record replaces an earlier one."""
    merged: dict[tuple[str, str], dict] = {}
    duplicates: list[tuple[str, str]] = []
    for group in groups:
        for r in group:
            key = (str(r.get("case_id")), str(r.get("arm")))
            if key in merged:
                duplicates.append(key)
            merged[key] = r
    return list(merged.values()), duplicates


def ran(r: dict) -> bool:
    return r.get("end_reason") not in NOT_RUN and not r.get("error")


def arm_sort_key(arm: str) -> tuple[int, str]:
    return (ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER), arm)


def mcnemar_exact_p(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value on the discordant pairs (binomial, p=0.5)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def by_case(results: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in results:
        out.setdefault(str(r.get("case_id")), {})[str(r.get("arm"))] = r
    return out


def paired(results: list[dict], arm: str, control: str = CONTROL) -> dict:
    pairs = []
    for cid, arms in sorted(by_case(results).items()):
        a, c = arms.get(arm), arms.get(control)
        if a is None or c is None:
            continue
        if not ran(a) or not ran(c):
            continue
        pairs.append((cid, bool(a.get("cleared")), bool(c.get("cleared")), a, c))
    n = len(pairs)
    arm_clear = sum(1 for _, x, _, _, _ in pairs if x)
    ctl_clear = sum(1 for _, _, y, _, _ in pairs if y)
    b = sum(1 for _, x, y, _, _ in pairs if x and not y)
    c_ = sum(1 for _, x, y, _, _ in pairs if y and not x)
    both = [(a, c) for _, x, y, a, c in pairs if x and y]
    out = {
        "arm": arm, "control": control, "n_pairs": n,
        "arm_cleared": arm_clear, "control_cleared": ctl_clear,
        "arm_clear_rate": round(arm_clear / n, 3) if n else None,
        "control_clear_rate": round(ctl_clear / n, 3) if n else None,
        "difference_points": round(100 * (arm_clear - ctl_clear) / n, 1) if n else None,
        "discordant_arm_only": b, "discordant_control_only": c_,
        "discordant_total": b + c_,
        "discordant_rate": round((b + c_) / n, 3) if n else None,
        "mcnemar_exact_p": mcnemar_exact_p(b, c_),
        "median_actions_to_clear_arm": median([int(a.get("actions_used") or 0) for _, x, _, a, _ in pairs if x]),
        "median_actions_to_clear_control": median([int(c.get("actions_used") or 0) for _, _, y, _, c in pairs if y]),
        "n_both_cleared": len(both),
        "paired_median_action_difference": median([int(a.get("actions_used") or 0) - int(c.get("actions_used") or 0) for a, c in both]),
        "paired_median_token_difference": median([int(a.get("generated_tokens") or 0) - int(c.get("generated_tokens") or 0) for a, c in both]),
        "cases_arm_only": [cid for cid, x, y, _, _ in pairs if x and not y],
        "cases_control_only": [cid for cid, x, y, _, _ in pairs if y and not x],
    }
    return out


def arm_totals(results: list[dict], arm: str) -> dict:
    rs = [r for r in results if r.get("arm") == arm and ran(r)]
    keys = ("probes", "readbacks", "batches", "capped_batches", "timeouts", "cancelled_actions", "fresh_notice_shown",
            "repertoire_shown", "repertoire_errors")
    cleared = sum(1 for r in rs if r.get("cleared"))
    return {
        "arm": arm, "n_run": len(rs), "cleared": cleared,
        "clear_rate": round(cleared / len(rs), 3) if rs else None,
        **{k: sum(int(r.get(k) or 0) for r in rs) for k in keys},
        "end_reasons": {reason: sum(1 for r in rs if r.get("end_reason") == reason)
                        for reason in sorted({str(r.get("end_reason")) for r in rs})},
        "median_wall_s": median([float(r.get("wall_s") or 0) for r in rs]),
        "median_replay_s": median([float(r.get("replay_s") or 0) for r in rs]),
        "median_generated_tokens": median([int(r.get("generated_tokens") or 0) for r in rs]),
        "median_restore_history_messages": median([int(r["restore_history_messages"]) for r in rs
                                                   if r.get("restore_history_messages") is not None]),
        "median_kept_functions": median([int(r["kept_functions"]) for r in rs if r.get("kept_functions") is not None]),
    }


def composite_baseline(results: list[dict], mode: str) -> list[dict]:
    """One derived control record per case on which A0 and A0b both ran.

    mode "or": cleared when either sample cleared (actions: the fewer among the
    clearing samples); mode "and": cleared when both did (actions: the larger).
    """
    label = EITHER if mode == "or" else BOTH
    out = []
    for cid, arms in sorted(by_case(results).items()):
        a, b = arms.get(BASELINES[0]), arms.get(BASELINES[1])
        if a is None or b is None or not ran(a) or not ran(b):
            continue
        xa, xb = bool(a.get("cleared")), bool(b.get("cleared"))
        cleared = (xa or xb) if mode == "or" else (xa and xb)
        acts = [int(r.get("actions_used") or 0) for r, x in ((a, xa), (b, xb)) if x]
        toks = [int(r.get("generated_tokens") or 0) for r, x in ((a, xa), (b, xb)) if x]
        pick = min if mode == "or" else max
        out.append({"case_id": cid, "arm": label, "cleared": cleared, "end_reason": "derived",
                    "actions_used": pick(acts) if acts and cleared else 0,
                    "generated_tokens": pick(toks) if toks and cleared else 0})
    return out


def baseline_comparisons(results: list[dict], arms: list[str]) -> list[dict]:
    """Every arm other than the baselines against A0b, A0-or-A0b and A0-and-A0b."""
    if not all(any(r.get("arm") == b for r in results) for b in BASELINES):
        return []
    either, both = composite_baseline(results, "or"), composite_baseline(results, "and")
    out = []
    for arm in arms:
        if arm in BASELINES:
            continue
        out.append(paired(results, arm, control=BASELINES[1]))
        for derived, label in ((either, EITHER), (both, BOTH)):
            ok_cases = {r["case_id"] for r in derived}
            rows = [r for r in results if r.get("arm") == arm and str(r.get("case_id")) in ok_cases] + derived
            out.append(paired(rows, arm, control=label))
    return out


def trim_summary(results: list[dict], arm: str = "A4") -> dict | None:
    """A4: the cases the trim shortened, sizes before and after, clears there for A4, A0 and A0b."""
    rs = [r for r in results if r.get("arm") == arm and ran(r) and r.get("restore_history_before_trim") is not None]
    if not rs:
        return None
    shortened = [r for r in rs if int(r["restore_history_before_trim"]) > int(r.get("restore_history_messages") or 0)]
    cases = {str(r.get("case_id")) for r in shortened}
    per_case = by_case(results)

    def clears(a: str) -> tuple[int, int]:
        xs = [per_case[c][a] for c in cases if a in per_case.get(c, {}) and ran(per_case[c][a])]
        return sum(1 for r in xs if r.get("cleared")), len(xs)

    return {
        "arm": arm, "n_run": len(rs), "n_shortened": len(shortened), "n_unchanged": len(rs) - len(shortened),
        "keep_fraction": sorted({r.get("trim_keep_fraction") for r in rs}, key=str),
        "median_messages_before": median([int(r["restore_history_before_trim"]) for r in rs]),
        "median_messages_after": median([int(r.get("restore_history_messages") or 0) for r in rs]),
        "median_tokens_before": median([int(r.get("restore_tokens_before_trim") or 0) for r in rs]),
        "median_tokens_after": median([int(r.get("restore_tokens_after_trim") or 0) for r in rs]),
        "cleared_on_shortened": {a: list(clears(a)) for a in (arm, *BASELINES)},
        "cases_shortened": sorted(cases),
    }


def clear_matrix(results: list[dict], arms: list[str]) -> dict[str, dict[str, bool]]:
    """case -> {arm: cleared} over the case-arms that ran."""
    out: dict[str, dict[str, bool]] = {}
    for cid, per_arm in by_case(results).items():
        row = {arm: bool(r.get("cleared")) for arm, r in per_arm.items() if arm in arms and ran(r)}
        if row:
            out[cid] = row
    return out


def union_of_clears(matrix: dict[str, dict[str, bool]], arms: list[str]) -> dict:
    cases = sorted(matrix)
    any_cleared = [cid for cid in cases if any(matrix[cid].values())]
    complete = [cid for cid in cases if all(a in matrix[cid] for a in arms)]
    all_cleared = [cid for cid in complete if all(matrix[cid][a] for a in arms)]
    rates = {}
    for a in arms:
        xs = [matrix[cid][a] for cid in cases if a in matrix[cid]]
        rates[a] = sum(xs) / len(xs) if xs else 0.0
    miss = 1.0
    for a in arms:
        miss *= 1.0 - rates[a]
    n_by_count: dict[int, int] = {}
    for cid in complete:
        k = sum(1 for a in arms if matrix[cid][a])
        n_by_count[k] = n_by_count.get(k, 0) + 1
    return {
        "arms": list(arms), "n_cases": len(cases), "n_complete_cases": len(complete),
        "cleared_by_at_least_one": len(any_cleared), "cleared_by_all": len(all_cleared),
        "expected_at_least_one_if_independent": round(len(cases) * (1.0 - miss), 2),
        "cases_cleared_by_at_least_one": any_cleared, "cases_cleared_by_all": all_cleared,
        "complete_cases_by_number_of_arms_clearing": {str(k): n_by_count[k] for k in sorted(n_by_count)},
    }


def spread_statistic(matrix: dict[str, dict[str, bool]], arms: list[str]) -> float:
    """Between-arm spread of clear rates: sum over arms of n_a (p_a - p)^2."""
    totals = {a: [0, 0] for a in arms}
    for row in matrix.values():
        for a, x in row.items():
            if a in totals:
                totals[a][0] += int(x)
                totals[a][1] += 1
    n_all = sum(n for _, n in totals.values())
    if n_all == 0:
        return 0.0
    p_all = sum(c for c, _ in totals.values()) / n_all
    return sum(n * ((c / n) - p_all) ** 2 for c, n in totals.values() if n)


def permutation_test(matrix: dict[str, dict[str, bool]], arms: list[str], draws: int = 20000, seed: int = 0) -> dict:
    """Shuffle arm labels within each case; p = share of draws with spread >= observed."""
    observed = spread_statistic(matrix, arms)
    rng = random.Random(seed)
    rows = [(list(row.keys()), list(row.values())) for row in matrix.values()]
    at_least = 0
    for _ in range(draws):
        shuffled: dict[str, dict[str, bool]] = {}
        for i, (keys, values) in enumerate(rows):
            vals = list(values)
            rng.shuffle(vals)
            shuffled[str(i)] = dict(zip(keys, vals))
        if spread_statistic(shuffled, arms) >= observed - 1e-12:
            at_least += 1
    return {
        "arms": list(arms), "n_cases": len(matrix), "draws": draws, "seed": seed,
        "statistic": "sum over arms of n_a (p_a - p)^2, arm labels permuted within case",
        "observed_spread": round(observed, 4),
        "p_value": round((at_least + 1) / (draws + 1), 4),
        "draws_at_or_above_observed": at_least,
    }


def clear_point(r: dict) -> tuple[int, float, bool] | None:
    """(actions after the cut, seconds after the restore, from clear fields) at the clear, or None.

    clear_actions / clear_wall_s are taken at the clearing action; rows without
    them fall back to actions_used / wall_s, taken when the case-arm ended."""
    if not r.get("cleared"):
        return None
    exact = r.get("clear_actions") is not None and r.get("clear_wall_s") is not None
    actions = r.get("clear_actions") if exact else r.get("actions_used")
    wall = r.get("clear_wall_s") if exact else r.get("wall_s")
    return int(actions or 0), float(wall or 0.0), exact


def cleared_within(r: dict, max_actions: int, max_wall_s: float) -> bool:
    point = clear_point(r)
    return point is not None and point[0] <= max_actions and point[1] <= max_wall_s


def long_arm_at(results: list[dict], label: str, budget: tuple[int, float] | None, arm: str = LONG_ARM) -> list[dict]:
    """Derived rows for the long arm: cleared within the budget (None: every clear of the run)."""
    out = []
    for r in results:
        if r.get("arm") != arm or not ran(r):
            continue
        x = bool(r.get("cleared")) if budget is None else cleared_within(r, *budget)
        point = clear_point(r)
        out.append({"case_id": r.get("case_id"), "arm": label, "cleared": x, "end_reason": "derived",
                    "actions_used": point[0] if x and point else 0, "generated_tokens": 0})
    return out


def time_verdict(gain: int) -> str:
    if gain >= TIME_LIMITED_GAIN:
        return "time-limited: next work is time allocation in the 9-hour run"
    if gain <= COMPETENCE_LIMITED_GAIN:
        return "competence-limited: harness work stops"
    return "inconclusive"


def time_dose_response(results: list[dict], arm: str = LONG_ARM) -> dict | None:
    """A0L clears within the R1 budget and within its own, against A0, A0b and each other."""
    rs = [r for r in results if r.get("arm") == arm and ran(r)]
    if not rs:
        return None
    at15 = long_arm_at(results, LONG_AT_15, R1_BUDGET, arm)
    at30 = long_arm_at(results, LONG_AT_30, None, arm)
    present = [b for b in BASELINES if any(r.get("arm") == b for r in results)]
    base = [r for r in results if r.get("arm") in present]
    vs15 = [paired(at15 + base, LONG_AT_15, control=b) for b in present]
    vs30 = [paired(at30 + base, LONG_AT_30, control=b) for b in present]
    if len(present) == len(BASELINES):
        either = composite_baseline(results, "or")
        ok_cases = {r["case_id"] for r in either}
        vs30.append(paired([r for r in at30 if str(r["case_id"]) in ok_cases] + either, LONG_AT_30, control=EITHER))
    internal = paired(at30 + at15, LONG_AT_30, control=LONG_AT_15)
    late = []
    for r in sorted(rs, key=lambda r: str(r.get("case_id"))):
        if r.get("cleared") and not cleared_within(r, *R1_BUDGET):
            actions, wall, exact = clear_point(r)
            over = [what for what, x in (("actions", actions > R1_BUDGET[0]), ("wall", wall > R1_BUDGET[1])) if x]
            late.append({"case_id": r.get("case_id"), "clear_actions": actions, "clear_wall_s": wall,
                         "over_r1_budget": " and ".join(over), "from_clear_fields": exact})
    n15 = sum(1 for r in at15 if r["cleared"])
    n30 = sum(1 for r in at30 if r["cleared"])
    cleared = [r for r in rs if r.get("cleared")]
    return {
        "arm": arm, "n_run": len(rs), "r1_budget": list(R1_BUDGET), "long_budget": list(R1D_BUDGET),
        "cleared_at_15": n15, "cleared_at_30": n30, "gain": n30 - n15,
        "cleared_without_clear_fields": sum(1 for r in cleared if not clear_point(r)[2]),
        "hard_stop_ends": sum(1 for r in rs if r.get("end_reason") == "hard_stop"),
        "end_reasons": {reason: sum(1 for r in rs if r.get("end_reason") == reason)
                        for reason in sorted({str(r.get("end_reason")) for r in rs})},
        "median_clear_wall_s": median([clear_point(r)[1] for r in cleared]),
        "median_clear_actions": median([clear_point(r)[0] for r in cleared]),
        "at_15_vs_baselines": vs15, "at_30_vs_baselines": vs30, "at_30_vs_at_15": internal,
        "cleared_after_15": late,
        "decision_rule": {"time_limited_if_gain_at_least": TIME_LIMITED_GAIN,
                          "competence_limited_if_gain_at_most": COMPETENCE_LIMITED_GAIN},
        "verdict": time_verdict(n30 - n15),
    }


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]] (sum of tables no more likely than observed)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    lo, hi = max(0, c1 - r2), min(c1, r1)

    def prob(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def load_labels(path: Path | None = LABELS_CSV) -> dict[str, str]:
    """case_id -> rule_discovered (yes / partial / no / unclear) from labels_rule_A.csv; {} if absent."""
    if path is None or not Path(path).is_file():
        return {}
    with open(path, newline="") as f:
        return {row["case_id"]: row["rule_discovered"].strip() for row in csv.DictReader(f)}


def baseline_sets(results: list[dict], baselines: tuple[str, ...], arm: str) -> tuple[list[str], list[dict]]:
    """Cases where the arm and every present baseline ran, and derived any / every baseline rows on them."""
    per_case = by_case(results)
    cases = sorted(cid for cid, arms in per_case.items()
                   if arm in arms and ran(arms[arm]) and all(b in arms and ran(arms[b]) for b in baselines))
    derived = []
    for cid in cases:
        xs = [bool(per_case[cid][b].get("cleared")) for b in baselines]
        for label, x in ((ANY_BASELINE, any(xs)), (EVERY_BASELINE, all(xs))):
            derived.append({"case_id": cid, "arm": label, "cleared": x, "end_reason": "derived",
                            "actions_used": 0, "generated_tokens": 0})
    return cases, derived


def sweep_group(results: list[dict], cases: list[str], derived: list[dict], baselines: tuple[str, ...],
                arm: str = SWEEP_ARM) -> dict:
    keep = set(cases)
    rows = [r for r in results if str(r.get("case_id")) in keep and r.get("arm") in (arm, *baselines)]
    rows += [r for r in derived if r["case_id"] in keep]
    per_case = by_case(rows)
    clears = {a: sum(1 for cid in cases if per_case[cid][a].get("cleared")) for a in (arm, *baselines)}
    sample_clears = [clears[b] for b in baselines]
    return {
        "n_cases": len(cases), "cleared": clears,
        "best_sample": max(sample_clears) if sample_clears else None,
        "median_sample": median(sample_clears) if sample_clears else None,
        "any_baseline": sum(1 for cid in cases if per_case[cid][ANY_BASELINE]["cleared"]),
        "every_baseline": sum(1 for cid in cases if per_case[cid][EVERY_BASELINE]["cleared"]),
        "pairs": [paired(rows, arm, control=b) for b in (*baselines, ANY_BASELINE, EVERY_BASELINE)],
    }


def sweep_verdict(all_cases: dict, partial: dict | None, yes: dict | None) -> tuple[bool, str]:
    reasons = []
    ok_all = all_cases["best_sample"] is not None and all_cases["cleared"][SWEEP_ARM] >= all_cases["best_sample"] + SWEEP_GAIN_ALL
    reasons.append(f"all cases: A6 {all_cases['cleared'][SWEEP_ARM]} vs best sample {all_cases['best_sample']} "
                   f"(needs +{SWEEP_GAIN_ALL}): {'met' if ok_all else 'not met'}")
    ok_partial = False
    if partial and yes and partial["n_cases"] and yes["n_cases"]:
        gain_ok = partial["cleared"][SWEEP_ARM] >= partial["best_sample"] + SWEEP_GAIN_PARTIAL
        no_loss = yes["cleared"][SWEEP_ARM] >= yes["median_sample"]
        ok_partial = gain_ok and no_loss
        reasons.append(f"partial cases: A6 {partial['cleared'][SWEEP_ARM]} vs best sample {partial['best_sample']} "
                       f"(needs +{SWEEP_GAIN_PARTIAL}): {'met' if gain_ok else 'not met'}; yes cases: A6 "
                       f"{yes['cleared'][SWEEP_ARM]} vs median sample {yes['median_sample']}: "
                       f"{'no loss' if no_loss else 'loss'}")
    else:
        reasons.append("partial / yes groups: labels or cases missing")
    build = ok_all or ok_partial
    head = ("build a full 25-game P0+A6 notebook for a leaderboard reading" if build
            else "no promotion: A6 does not beat the baseline samples by the margin")
    return build, head + " (" + "; ".join(reasons) + ")"


def sweep_section(results: list[dict], labels: dict[str, str] | None = None, arm: str = SWEEP_ARM) -> dict | None:
    """R1e: A6 against the baseline samples, overall and by rule-discovery group, and the decision."""
    if not any(r.get("arm") == arm and ran(r) for r in results):
        return None
    baselines = tuple(b for b in SWEEP_BASELINES if any(r.get("arm") == b and ran(r) for r in results))
    cases, derived = baseline_sets(results, baselines, arm)
    labels = labels or {}
    groups = {}
    for g in ("partial", "yes", "no"):
        sub = [c for c in cases if labels.get(c) == g]
        if sub:
            groups[g] = sweep_group(results, sub, derived, baselines, arm)
    overall = sweep_group(results, cases, derived, baselines, arm)
    per_case = by_case(results)
    fisher = []
    if "partial" in groups and "yes" in groups:
        part = [c for c in cases if labels.get(c) == "partial"]
        yes = [c for c in cases if labels.get(c) == "yes"]

        def count(cs, pred):
            return sum(1 for c in cs if pred(per_case[c]))

        tests = [(f"{arm} cleared", lambda a: bool(a[arm].get("cleared")))]
        if SAME_RUN_CONTROL in baselines:
            tests.append((f"{SAME_RUN_CONTROL} cleared", lambda a: bool(a[SAME_RUN_CONTROL].get("cleared"))))
            tests.append((f"{arm} cleared and {SAME_RUN_CONTROL} did not",
                          lambda a: bool(a[arm].get("cleared")) and not a[SAME_RUN_CONTROL].get("cleared")))
        for name, pred in tests:
            x_p, x_y = count(part, pred), count(yes, pred)
            fisher.append({"outcome": name, "partial": [x_p, len(part)], "yes": [x_y, len(yes)],
                           "fisher_exact_p": fisher_exact_p(x_p, len(part) - x_p, x_y, len(yes) - x_y)})
    rs = [r for r in results if r.get("arm") == arm and ran(r)]
    build, verdict = sweep_verdict(overall, groups.get("partial"), groups.get("yes"))
    return {
        "arm": arm, "baselines": list(baselines), "n_cases": len(cases), "labels_loaded": bool(labels),
        "sweeps_shown_total": sum(int(r.get("repertoire_shown") or 0) for r in rs),
        "median_sweeps_shown": median([int(r.get("repertoire_shown") or 0) for r in rs]),
        "case_arms_without_sweep": sum(1 for r in rs if not int(r.get("repertoire_shown") or 0)),
        "repertoire_errors": sum(int(r.get("repertoire_errors") or 0) for r in rs),
        "max_message_chars": max([int(r.get("repertoire_chars_max") or 0) for r in rs], default=0),
        "overall": overall, "groups": groups, "fisher": fisher,
        "cases_a6_only": [c for c in cases if per_case[c][arm].get("cleared")
                          and not any(per_case[c][b].get("cleared") for b in baselines)],
        "cases_baseline_only": [c for c in cases if not per_case[c][arm].get("cleared")
                                and any(per_case[c][b].get("cleared") for b in baselines)],
        "decision_rule": {"gain_all_cases": SWEEP_GAIN_ALL, "gain_partial_cases": SWEEP_GAIN_PARTIAL,
                          "no_loss_on_yes": "A6 clears on the yes cases >= the median baseline sample there"},
        "build_full_notebook": build, "verdict": verdict,
    }


def report(results: list[dict], permutations: int = 20000, seed: int = 0, duplicates: list | None = None,
           labels: dict[str, str] | None = None) -> dict:
    arms = sorted({str(r.get("arm")) for r in results}, key=arm_sort_key)
    budget_arms = [a for a in arms if a not in LONG_ARMS]
    not_run = [(r.get("case_id"), r.get("arm"), r.get("end_reason")) for r in results if not ran(r)]
    matrix = clear_matrix(results, budget_arms)
    rep = {
        "n_results": len(results),
        "n_cases": len({r.get("case_id") for r in results}),
        "arms": arms,
        "duplicates": [list(d) for d in (duplicates or [])],
        "not_run": not_run,
        "totals": [arm_totals(results, a) for a in arms],
        "comparisons": [paired(results, a) for a in budget_arms if a != CONTROL],
        "baseline_comparisons": baseline_comparisons(results, budget_arms),
        "trim": trim_summary(results),
        "union": union_of_clears(matrix, budget_arms) if budget_arms else None,
        "arms_do_not_matter": permutation_test(matrix, budget_arms, draws=permutations, seed=seed) if len(budget_arms) > 1 else None,
        "time": time_dose_response(results),
        "sweep": sweep_section(results, labels),
    }
    return rep


def _pair_line(c: dict) -> str:
    p = c["mcnemar_exact_p"]
    return (f"{c['arm']} vs {c['control']}: pairs {c['n_pairs']}; cleared {c['arm_cleared']} vs {c['control_cleared']} "
            f"({c['difference_points']} points); discordant {c['discordant_arm_only']}/{c['discordant_control_only']}; "
            f"McNemar exact p {'n/a' if p is None else f'{p:.3f}'}")


def format_report(rep: dict) -> str:
    lines = [f"results: {rep['n_results']} case-arms over {rep['n_cases']} cases; arms {', '.join(rep['arms'])}; "
             f"not run: {len(rep['not_run'])}" + (f"; duplicates replaced: {len(rep['duplicates'])}" if rep.get("duplicates") else "")]
    for t in rep["totals"]:
        lines.append(f"{t['arm']}: run {t['n_run']}, cleared {t['cleared']} ({t['clear_rate']}), probes {t['probes']}, "
                     f"capped batches {t['capped_batches']}, timeouts {t['timeouts']}, cancelled actions {t['cancelled_actions']}, "
                     f"median wall {t['median_wall_s']} s, median replay {t['median_replay_s']} s, "
                     f"median tokens {t['median_generated_tokens']}, median restored messages "
                     f"{t['median_restore_history_messages']}, median kept functions {t['median_kept_functions']}, "
                     f"ends {t['end_reasons']}")
    for c in rep["comparisons"]:
        p = c["mcnemar_exact_p"]
        tag = " (reproducibility floor)" if c["arm"] in ("A0b", "A0c") else ""
        lines.append(
            f"{c['arm']} vs {c['control']}{tag}: pairs {c['n_pairs']}; cleared {c['arm_cleared']} vs {c['control_cleared']} "
            f"({c['arm_clear_rate']} vs {c['control_clear_rate']}, {c['difference_points']} points); "
            f"discordant {c['discordant_arm_only']}/{c['discordant_control_only']} ({c['discordant_total']} of {c['n_pairs']}, "
            f"{c['discordant_rate']}); McNemar exact p "
            f"{'n/a' if p is None else f'{p:.3f}'}; median actions to clear {c['median_actions_to_clear_arm']} vs "
            f"{c['median_actions_to_clear_control']}; both cleared {c['n_both_cleared']}, paired median action difference "
            f"{c['paired_median_action_difference']}"
        )
    for c in rep.get("baseline_comparisons") or []:
        p = c["mcnemar_exact_p"]
        lines.append(
            f"{c['arm']} vs {c['control']}: pairs {c['n_pairs']}; cleared {c['arm_cleared']} vs {c['control_cleared']} "
            f"({c['difference_points']} points); discordant {c['discordant_arm_only']}/{c['discordant_control_only']}; "
            f"McNemar exact p {'n/a' if p is None else f'{p:.3f}'}"
        )
    tr = rep.get("trim")
    if tr:
        cl = tr["cleared_on_shortened"]
        lines.append(
            f"{tr['arm']} trim (keep fraction {tr['keep_fraction']}): shortened {tr['n_shortened']}/{tr['n_run']} restored "
            f"conversations, unchanged {tr['n_unchanged']}; median messages {tr['median_messages_before']} -> "
            f"{tr['median_messages_after']}, est tokens {tr['median_tokens_before']} -> {tr['median_tokens_after']}; on the "
            f"shortened cases cleared " + ", ".join(f"{a} {c}/{n}" for a, (c, n) in cl.items())
        )
    u = rep.get("union")
    if u:
        lines.append(
            f"union over {', '.join(u['arms'])}: {u['cleared_by_at_least_one']}/{u['n_cases']} cases cleared by at least one arm "
            f"(independent draws at the observed rates: {u['expected_at_least_one_if_independent']}); "
            f"{u['cleared_by_all']}/{u['n_complete_cases']} complete cases cleared by every arm; "
            f"complete cases by number of arms clearing {u['complete_cases_by_number_of_arms_clearing']}"
        )
    t = rep.get("arms_do_not_matter")
    if t:
        lines.append(
            f"arms do not matter (labels permuted within case, {t['draws']} draws): observed spread {t['observed_spread']}, "
            f"p {t['p_value']}"
        )
    d = rep.get("time")
    if d:
        lines.append(
            f"time dose-response ({d['arm']}, budget {d['long_budget'][0]} actions / {d['long_budget'][1]:.0f} s; R1 budget "
            f"{d['r1_budget'][0]} / {d['r1_budget'][1]:.0f} s): run {d['n_run']}; cleared within the R1 budget "
            f"{d['cleared_at_15']}, within its own {d['cleared_at_30']}, gain {d['gain']}; median clear at "
            f"{d['median_clear_actions']} actions, {d['median_clear_wall_s']} s; clears without clear fields "
            f"{d['cleared_without_clear_fields']}; hard-stop ends {d['hard_stop_ends']}; ends {d['end_reasons']}"
        )
        for c in d["at_15_vs_baselines"] + d["at_30_vs_baselines"] + [d["at_30_vs_at_15"]]:
            lines.append(_pair_line(c))
        late = ", ".join(f"{x['case_id']} ({x['clear_actions']} actions, {x['clear_wall_s']} s; over {x['over_r1_budget']})"
                         for x in d["cleared_after_15"])
        lines.append(f"cleared only after the R1 budget: {len(d['cleared_after_15'])}" + (f": {late}" if late else ""))
        lines.append(f"R1d decision (gain >= {d['decision_rule']['time_limited_if_gain_at_least']} time-limited, "
                     f"<= {d['decision_rule']['competence_limited_if_gain_at_most']} competence-limited): {d['verdict']}")
    s = rep.get("sweep")
    if s:
        o = s["overall"]
        lines.append(
            f"repertoire sweep ({s['arm']}; baselines {', '.join(s['baselines'])}; {s['n_cases']} cases where all ran): "
            f"sweeps shown {s['sweeps_shown_total']} (median {s['median_sweeps_shown']} per case-arm, "
            f"{s['case_arms_without_sweep']} case-arms without one), errors {s['repertoire_errors']}, max message "
            f"{s['max_message_chars']} chars; cleared " + ", ".join(f"{a} {n}" for a, n in o["cleared"].items())
            + f"; any baseline {o['any_baseline']}, every baseline {o['every_baseline']}"
        )
        for c in o["pairs"]:
            lines.append("  all: " + _pair_line(c))
        if not s["labels_loaded"]:
            lines.append("  rule-discovery labels not found; groups skipped")
        for g, grp in s["groups"].items():
            lines.append(f"  {g} ({grp['n_cases']} cases): cleared " + ", ".join(f"{a} {n}" for a, n in grp["cleared"].items())
                         + f"; best sample {grp['best_sample']}, median sample {grp['median_sample']}; any baseline "
                         f"{grp['any_baseline']}, every baseline {grp['every_baseline']}")
            for c in grp["pairs"]:
                lines.append(f"    {g}: " + _pair_line(c))
        for f in s["fisher"]:
            lines.append(f"  Fisher exact, {f['outcome']}: partial {f['partial'][0]}/{f['partial'][1]} vs yes "
                         f"{f['yes'][0]}/{f['yes'][1]}, p {f['fisher_exact_p']:.3f}")
        lines.append(f"  cleared by A6 and no baseline sample: {s['cases_a6_only']}; by a baseline sample and not A6: "
                     f"{s['cases_baseline_only']}")
        lines.append(f"R1e decision: {s['verdict']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default=str(LABELS_CSV), help="rule-discovery labels (R1e groups)")
    args = ap.parse_args()
    results, duplicates = merge_results([load_results(Path(p)) for p in args.results])
    rep = report(results, permutations=args.permutations, seed=args.seed, duplicates=duplicates,
                 labels=load_labels(Path(args.labels)))
    rep["inputs"] = [str(p) for p in args.results]
    print(format_report(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
