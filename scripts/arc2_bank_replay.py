#!/usr/bin/env python3
"""Replay selection rules on the ARC-AGI-2 output bank (P3) and keep the reuse ledger (E17). CPU only.

    python scripts/arc2_bank_replay.py                                   # the registered rules against b0
    python scripts/arc2_bank_replay.py --rule q27_agree --rule null_jitter1:7
    python scripts/arc2_bank_replay.py --tasks 0934a4d8,136b0064 --no-ledger
    python scripts/arc2_bank_replay.py --sealed --final-pick --rule v7_direct   # the one final-pick check

Every rule is replayed on the same outputs (pool/ of both bank parts plus the 27B bank) and compared with the base
(default b0: B0 as submitted): gained / lost / net outputs, the one-sided sign test on the discordants, the
task-level paired difference with a cluster bootstrap by task, and both gates (v1 as pre-registered: net >= 4 and
p < 0.05; v2 as proposed after a check: net >= 3 and lost <= 1). A rule is a candidate only when its row is in
experiments/arc2-bank/RULES.md (E7); otherwise its number is a screening result.

The reuse ledger (experiments/arc2-bank/replay_ledger.jsonl) gets one line per replay: rule, rule fingerprint
(source hash), bank fingerprint, result, comparison, registration. The runner prints how many distinct rules have
been screened on this bank so far and the expected false positives at each gate (the pooled false-positive rates of
the level-preserving null families from experiments/arc2-bank/null_calibration.json when present, else the nominal
0.05 and the v2 rate unknown).

Sealed tasks (pool_sealed/, 19 tasks) are refused unless --sealed and --final-pick are both given; a sealed read is
written to the ledger with sealed=true. Use it once, for the final pick.

The GS1 rule (gs_low; GS1 revision 1 dropped gs_all) reads Glimmer's scores of the pool candidates: --glimmer
(default: the GS1 kernel's archived unsealed file when present) and, with --sealed --final-pick only, --glimmer-sealed
(default: its sealed file). The ledger line records the score files and their sha256; a sealed read also prints
gained / lost on the sealed tasks.

The DC1 rules (dc_fit, dc_long) read the decision model's probabilities the same way: --decider (default: the DC1
kernel's archived unsealed file when present) and, with --sealed --final-pick only, --decider-sealed.

C2 (experiments/arc2-bank/C2_RULE.md): --q27 takes several 27B directories, merged by task and sample index (the
bank's q27d plus the C2 kernels' q27d; a sealed q27d_sealed/ is refused), and --as NAME logs the one --rule under
that registered name:

    python scripts/arc2_bank_replay.py --rule oracle --q27 <bank q27d> <c2a q27d> <c2b q27d> --as oracle_c2

The ledger line then carries rule = the --as name, rule_run = the rule replayed, the 27B directories (q27_dirs) and
the merged bank's fingerprint.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (BANK_PARTS, BANK_SEALED, DC1_PROBS, DC1_PROBS_SEALED, GS1_SCORES,  # noqa: E402
                            GS1_SCORES_SEALED, Q27_BANK, RULES, SCHEMA, SealedAccessError, compare, get_rule,
                            load_bank, replay)

BANK_DIR = ROOT / "experiments" / "arc2-bank"
RULES_FILE = BANK_DIR / "RULES.md"
LEDGER = BANK_DIR / "replay_ledger.jsonl"
NULL_CALIBRATION = BANK_DIR / "null_calibration.json"
DEFAULT_RULES = ("b0_kgmon", "b0_probmul3", "v7_direct", "q27_first", "q27_agree", "oracle")
NOMINAL_ALPHA = 0.05


def registered_rules(path: Path = RULES_FILE) -> dict[str, dict]:
    """{rule: {date, family, status}} from the pre-registration table (rows '| YYYY-MM-DD | name | ... |')."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([\w:.-]+)\s*\|\s*([\w-]+)\s*\|(.*)\|\s*([^|]*)\|\s*$", line)
        if m:
            out[m.group(2)] = {"date": m.group(1), "family": m.group(3), "status": m.group(5).strip()}
    return out


def read_ledger(path: Path = LEDGER) -> list[dict]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def false_positive_rates(path: Path = NULL_CALIBRATION) -> dict:
    """{gate_v1, gate_v2, net_needed_fp5, net_needed_fp1, source} from the null calibration's pooled block (the
    level-preserving families), else the nominal level."""
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            pooled = doc.get("pooled_level_preserving") or doc.get("pooled") or {}
            v1 = pooled.get("fp_gate_v1", pooled.get("fpr_gate_v1"))
            v2 = pooled.get("fp_gate_v2", pooled.get("fpr_gate_v2"))
            if v1 is not None or v2 is not None:
                return {"gate_v1": v1, "gate_v2": v2, "net_needed_fp5": pooled.get("net_needed_fp5"),
                        "net_needed_fp1": pooled.get("net_needed_fp1"), "source": str(path)}
        except (ValueError, AttributeError):
            pass
    return {"gate_v1": NOMINAL_ALPHA, "gate_v2": None, "net_needed_fp5": None, "net_needed_fp1": None,
            "source": "nominal 0.05 (no null calibration file)"}


def screened_so_far(rows: list[dict], bank_fp: str) -> int:
    """Distinct candidate-rule fingerprints replayed on this bank before now (nulls, bounds and the base apart)."""
    return len({r.get("rule_fingerprint") for r in rows
                if r.get("bank_fingerprint") == bank_fp and r.get("family") not in ("null", "oracle")
                and r.get("rule") != r.get("base")})


def fmt_p(p) -> str:
    return "-" if p is None else f"{p:.3f}"


def sha256_of(path: str) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def glimmer_files(args: argparse.Namespace) -> list[str]:
    """The GS1 score files this replay reads: --glimmer, and --glimmer-sealed only on the sealed read."""
    files = [p for p in [getattr(args, "glimmer", None) or ""] if p]
    if args.sealed and getattr(args, "glimmer_sealed", None):
        files.append(args.glimmer_sealed)
    for p in files:
        if not Path(p).is_file():
            raise FileNotFoundError(f"GS1 scores file {p} not found")
    return files


def decider_files(args: argparse.Namespace) -> list[str]:
    """The DC1 probability files this replay reads: --decider, and --decider-sealed only on the sealed read."""
    files = [p for p in [getattr(args, "decider", None) or ""] if p]
    if args.sealed and getattr(args, "decider_sealed", None):
        files.append(args.decider_sealed)
    for p in files:
        if not Path(p).is_file():
            raise FileNotFoundError(f"DC1 probabilities file {p} not found")
    return files


def q27_dirs_of(args: argparse.Namespace) -> list[Path]:
    """The 27B directories of this replay: --q27 (one or several), default the q27 bank."""
    q = getattr(args, "q27", None)
    dirs = list(q) if isinstance(q, (list, tuple)) else ([q] if q else [])
    return [Path(d) for d in dirs if d] or [Q27_BANK]


def run(args: argparse.Namespace) -> dict:
    pools = [Path(p) for p in args.pools] if args.pools else list(BANK_PARTS)
    q27_dirs = q27_dirs_of(args)
    as_name = getattr(args, "as_name", "") or ""
    names = list(dict.fromkeys(args.rule or DEFAULT_RULES))
    if as_name and len(names) != 1:
        raise ValueError(f"--as {as_name} logs one ledger line: give exactly one --rule (got {len(names)})")
    if args.sealed:
        if not args.final_pick:
            raise SealedAccessError("--sealed needs --final-pick: the sealed 19 tasks are read once, for the final pick")
        pools += [p for p in BANK_SEALED if p.is_dir()]
    tasks = [t for t in args.tasks.split(",") if t] if args.tasks else None
    gfiles = glimmer_files(args)
    dfiles = decider_files(args)
    bank = load_bank(pools, q27_dirs, final_pick=bool(args.final_pick), tasks=tasks,
                     glimmer=gfiles or None, decider=dfiles or None)
    glimmer = {"files": gfiles, "sha256": {p: sha256_of(p) for p in gfiles},
               "tasks_scored": len(bank.glimmer)} if gfiles else None
    decider = {"files": dfiles, "sha256": {p: sha256_of(p) for p in dfiles},
               "tasks_with_probabilities": len(bank.decider)} if dfiles else None
    sealed_ids = {t for t, task in bank.tasks.items() if task.sealed}
    bank_fp = bank.fingerprint()
    reg = registered_rules(Path(args.rules_file))
    rows_before = read_ledger(Path(args.ledger)) if not args.no_ledger else []
    n_screened = screened_so_far(rows_before, bank_fp)
    fpr = false_positive_rates(Path(args.null_calibration))

    base_rule = get_rule(args.base)
    base = replay(bank, base_rule)
    results, lines = [], []
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for name in names:
        rule = get_rule(name)
        res = replay(bank, rule)
        cmp = compare(base, res, n_boot=args.n_boot, seed=args.seed)
        logged = as_name or name          # the ledger name: the --as row of RULES.md, else the rule's own
        registered = f"registered {reg[logged]['date']} ({reg[logged]['status']})" if logged in reg else ""
        if rule.family == "null":
            status = "null (calibration)"
        elif rule.family == "oracle" or rule.uses_labels:
            status = "bound (reads labels)" + (f"; {registered}" if registered else "")
        elif registered:
            status = registered
        else:
            status = "NOT PRE-REGISTERED: screening only"
        row = {"schema": SCHEMA, "utc": utc, "rule": logged, "rule_run": name, "logged_as": as_name or None,
               "family": rule.family, "description": rule.description,
               "rule_fingerprint": res.rule_fingerprint, "bank_fingerprint": bank_fp, "base": base.rule,
               "q27_dirs": [str(d) for d in q27_dirs],
               "n_tasks": res.n_tasks, "n_outputs": res.n_outputs, "n_right": res.n_right, "n_right_1": res.n_right_1,
               "score": round(res.score, 4), "task_score": round(res.task_score, 4),
               "gained": cmp["gained"], "lost": cmp["lost"], "net": cmp["net"], "sign_test_p": cmp["sign_test_p"],
               "task_mean_delta": cmp["task_level"]["mean_delta"], "task_ci95": cmp["task_level"]["bootstrap"].get("ci95"),
               "gate_v1": cmp["gate_v1"]["pass"], "gate_v2": cmp["gate_v2"]["pass"], "registration": status,
               "sealed": bool(args.sealed), "tasks_subset": bool(tasks), "sources": bank.sources,
               "gained_keys": cmp["gained_keys"], "lost_keys": cmp["lost_keys"]}
        if glimmer is not None:
            row["glimmer"] = glimmer
        if decider is not None:
            row["decider"] = decider
        if args.sealed:
            task_of = {k: v["task_id"] for k, v in res.per_output.items()}
            g_s = [k for k in cmp["gained_keys"] if task_of.get(k) in sealed_ids]
            l_s = [k for k in cmp["lost_keys"] if task_of.get(k) in sealed_ids]
            row["sealed_split"] = {"n_tasks": len(sealed_ids), "gained": len(g_s), "lost": len(l_s),
                                   "net": len(g_s) - len(l_s), "gained_keys": g_s, "lost_keys": l_s}
        results.append(row)
        lines.append(f"| {logged}{'' if logged == name else f' (= {name})'} | {res.n_right}/{res.n_outputs} | "
                     f"{100 * res.score:.2f} | {100 * res.task_score:.2f} | "
                     f"{cmp['gained']} | {cmp['lost']} | {cmp['net']:+d} | {fmt_p(cmp['sign_test_p'])} | "
                     f"{'PASS' if cmp['gate_v1']['pass'] else 'no'} | {'PASS' if cmp['gate_v2']['pass'] else 'no'} | {status} |")
    seen = {x.get("rule_fingerprint") for x in rows_before if x.get("bank_fingerprint") == bank_fp}
    new_screened = n_screened + len({r["rule_fingerprint"] for r in results if r["family"] not in ("null", "oracle")
                                     and r["rule"] != base.rule and r["rule_fingerprint"] not in seen})
    exp = {g: (None if fpr.get(g) is None else round(new_screened * float(fpr[g]), 2)) for g in ("gate_v1", "gate_v2")}
    report = {"schema": SCHEMA, "utc": utc, "bank_fingerprint": bank_fp, "sources": bank.sources,
              "q27_dirs": [str(d) for d in q27_dirs], "logged_as": as_name or None,
              "solutions_source": bank.solutions_source, "n_tasks": len(bank.tasks), "n_outputs": bank.n_outputs,
              "n_q27_tasks": len(bank.q27), "sealed_read": bool(args.sealed), "base": base.summary(),
              "rules": results, "screened_rules_on_this_bank": new_screened,
              "false_positive_rates": fpr, "expected_false_positives": exp, "glimmer": glimmer, "decider": decider}
    print(f"bank {bank_fp}: {len(bank.tasks)} tasks, {bank.n_outputs} outputs, 27B samples on {len(bank.q27)} tasks"
          f"{' [SEALED READ: final pick]' if args.sealed else ''}; solutions: {bank.solutions_source}")
    print(f"base {base.rule}: {base.n_right}/{base.n_outputs} right, score {100 * base.score:.2f}, task score "
          f"{100 * base.task_score:.2f}")
    print("| rule | right | score | task score | gained | lost | net | sign p | gate v1 | gate v2 | registration |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    if glimmer is not None:
        print(f"GS1 scores: {', '.join(gfiles)} ({glimmer['tasks_scored']} tasks with scores)")
    elif any(r["family"] == "gs1" for r in results):
        print("GS1 scores: none loaded (--glimmer): the gs1 rules fall back to B0's order on every output")
    if decider is not None:
        print(f"DC1 probabilities: {', '.join(dfiles)} ({decider['tasks_with_probabilities']} tasks)")
    elif any(r["family"] == "dc1" for r in results):
        print("DC1 probabilities: none loaded (--decider): the dc1 rules fall back to B0's order on every output")
    if args.sealed:
        for r in results:
            s = r["sealed_split"]
            print(f"sealed tasks ({s['n_tasks']}), {r['rule']}: gained {s['gained']}, lost {s['lost']}, net {s['net']:+d}")
    print(f"rules screened on this bank (distinct, excluding base, nulls, bounds): {new_screened}; expected false "
          f"positives: gate v1 {exp['gate_v1']} (rate {fpr['gate_v1']}), gate v2 {exp['gate_v2']} (rate {fpr['gate_v2']}); "
          f"net a null reaches at <= 5% / 1%: {fpr.get('net_needed_fp5')} / {fpr.get('net_needed_fp1')}; "
          f"rates from {fpr['source']}")
    if not args.no_ledger:
        Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.ledger).open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"ledger: {len(results)} line(s) appended to {args.ledger}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"report written to {args.json}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", action="append", help="rule name (repeatable); default: " + ", ".join(DEFAULT_RULES)
                    + "; registered: " + ", ".join(sorted(RULES)) + "; null: <family>:<seed>")
    ap.add_argument("--base", default="b0")
    ap.add_argument("--pools", nargs="*", help="pool directories (default: both bank parts' pool/)")
    ap.add_argument("--q27", nargs="*", default=None, metavar="DIR",
                    help="27B directories merged by task and sample index (default: the q27 bank); C2: the bank's "
                         "q27d and the C2 kernels' q27d")
    ap.add_argument("--as", dest="as_name", default="", metavar="NAME",
                    help="log the one --rule under this registered RULES.md name (e.g. --rule oracle --as oracle_c2)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (subset)")
    ap.add_argument("--sealed", action="store_true", help="also read pool_sealed/ (needs --final-pick)")
    ap.add_argument("--final-pick", action="store_true", help="the one sealed read of the final pick")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules-file", default=str(RULES_FILE))
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--null-calibration", default=str(NULL_CALIBRATION))
    ap.add_argument("--json", default="", help="write the full report here")
    ap.add_argument("--glimmer", default=str(GS1_SCORES) if GS1_SCORES.is_file() else "",
                    help="GS1 scores of the unsealed pools (default: the GS1 archive's file when present)")
    ap.add_argument("--glimmer-sealed", default=str(GS1_SCORES_SEALED) if GS1_SCORES_SEALED.is_file() else "",
                    help="GS1 scores of the sealed pools (read only with --sealed --final-pick)")
    ap.add_argument("--decider", default=str(DC1_PROBS) if DC1_PROBS.is_file() else "",
                    help="DC1 probabilities of the unsealed pools (default: the DC1 archive's file when present)")
    ap.add_argument("--decider-sealed", default=str(DC1_PROBS_SEALED) if DC1_PROBS_SEALED.is_file() else "",
                    help="DC1 probabilities of the sealed pools (read only with --sealed --final-pick)")
    args = ap.parse_args(argv)
    try:
        run(args)
    except SealedAccessError as exc:
        print(f"refused: {exc}")
        return 2
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
