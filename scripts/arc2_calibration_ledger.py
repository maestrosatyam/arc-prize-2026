#!/usr/bin/env python3
"""Calibration ledger (P5, E6, E8): every submitted ARC-AGI-2 build's bank score beside its board reading.

    python scripts/arc2_calibration_ledger.py                      # read the ledgers, replay, write the calibration ledger
    python scripts/arc2_calibration_ledger.py --write-submissions-ledger   # also put expected_board / bank_points into
                                                                           # submissions/ledger.json entries

Rows come from submissions/ledger.json (track arc2, official submissions) joined with experiments/arc2-bank/builds.json,
which maps each entry to the replay rule that reproduces its selection on the bank (or marks it not replayable). The
bank score is the rule's replay on the bank (eval/arc2_bank.py), in board points (100 x outputs right / outputs) and
as the task-weighted number (the hidden-mix proxy). With at least four (bank, board) pairs the mapping
board = a + b x bank is fitted by least squares; the residual SD is the prediction SD, and the leave-one-out
predictions (E6) say whether the mapping predicts at all. Every row then carries an expected board (E8), pending
submissions included. With fewer pairs the ledger reports the sign agreement of bank and board differences between
pairs instead, and no expected board.

Board run noise for reference: identical B0 copies read 28.06-32.22 on the board,
about 1.8 points SD; a mapping cannot predict closer than that.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import BANK_PARTS, Q27_BANK, get_rule, load_bank, replay  # noqa: E402

SUBMISSIONS = ROOT / "submissions" / "ledger.json"
BUILDS = ROOT / "experiments" / "arc2-bank" / "builds.json"
OUT = ROOT / "experiments" / "arc2-bank" / "calibration_ledger.json"
SCHEMA = "arc2_calibration_ledger.v1"
MIN_PAIRS = 4
BOARD_RUN_SD = 1.8
BOARD_RUN_SD_SOURCE = "identical B0 copies read 28.06-32.22 on the board"


# ----------------------------------------------------------------------------
# the mapping

def ols(xs: list[float], ys: list[float]) -> dict | None:
    """board = a + b x bank by least squares; residual SD with n - 2 degrees of freedom; None below 3 points."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / (n - 2)) if n > 2 else float("nan")
    syy = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in resid) / syy if syy > 0 else None
    return {"a": round(a, 4), "b": round(b, 4), "n": n, "residual_sd": round(sd, 3),
            "r2": None if r2 is None else round(r2, 4)}


def leave_one_out(xs: list[float], ys: list[float]) -> dict:
    """Each pair predicted from the others (E6): predictions, errors, RMSE; None where a fit is impossible."""
    preds, errs = [], []
    for i in range(len(xs)):
        fit = ols(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
        if fit is None:
            preds.append(None)
            continue
        p = fit["a"] + fit["b"] * xs[i]
        preds.append(round(p, 2))
        errs.append(ys[i] - p)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else None
    return {"predictions": preds, "rmse": None if rmse is None else round(rmse, 3), "n": len(errs)}


def sign_agreement(pairs: list[dict]) -> dict:
    """For every two pairs: do the bank difference and the board difference have the same sign?"""
    rows, agree, n = [], 0, 0
    for p, q in itertools.combinations(pairs, 2):
        db, dk = q["board"] - p["board"], q["bank_points"] - p["bank_points"]
        same = (db > 0) == (dk > 0) if db != 0 and dk != 0 else None
        rows.append({"from": p["entry_id"], "to": q["entry_id"], "board_delta": round(db, 2),
                     "bank_delta": round(dk, 2), "same_sign": same})
        if same is not None:
            n += 1
            agree += int(same)
    return {"n_comparisons": n, "agree": agree, "rows": rows}


# ----------------------------------------------------------------------------
# the ledger

def submissions(path: Path = SUBMISSIONS) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    ents = doc["entries"] if isinstance(doc, dict) else doc
    return sorted((e for e in ents if e.get("track") == "arc2" and e.get("kind") == "official_submission"),
                  key=lambda e: e.get("utc_timestamp", ""))


def build_ledger(subs: list[dict], builds: dict, bank) -> dict:
    by_entry = {b["entry_id"]: b for b in builds.get("builds", [])}
    cache: dict[str, object] = {}
    rows = []
    for e in subs:
        b = by_entry.get(e["entry_id"], {"rule": None, "replayable": False, "note": "not in builds.json"})
        board = e.get("public_score")
        board = float(board) if board is not None else None
        row = {"entry_id": e["entry_id"], "utc": e.get("utc_timestamp"), "slug": e.get("notebook_slug"),
               "submission_id": e.get("submission_id"), "board": board, "rule": b.get("rule"),
               "replayable": bool(b.get("replayable")), "approx": bool(b.get("approx")), "note": b.get("note", ""),
               "bank_points": None, "bank_task_points": None, "bank_right": None, "bank_outputs": None,
               "expected_board": None, "loo_expected_board": None}
        if b.get("replayable") and b.get("rule"):
            if b["rule"] not in cache:
                cache[b["rule"]] = replay(bank, get_rule(b["rule"]))
            res = cache[b["rule"]]
            row.update(bank_points=round(100 * res.score, 2), bank_task_points=round(100 * res.task_score, 2),
                       bank_right=res.n_right, bank_outputs=res.n_outputs)
        rows.append(row)
    pairs = [r for r in rows if r["board"] is not None and r["bank_points"] is not None]
    xs, ys = [r["bank_points"] for r in pairs], [r["board"] for r in pairs]
    fit = ols(xs, ys) if len(pairs) >= MIN_PAIRS else None
    loo = leave_one_out(xs, ys) if len(pairs) >= MIN_PAIRS else None
    if fit:
        for r in rows:
            if r["bank_points"] is not None:
                r["expected_board"] = round(fit["a"] + fit["b"] * r["bank_points"], 2)
        for r, p in zip(pairs, loo["predictions"]):
            r["loo_expected_board"] = p
    status = (f"fitted on {len(pairs)} pairs" if fit else
              f"insufficient pairs: {len(pairs)} of {MIN_PAIRS} needed; sign agreement reported instead")
    return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bank_fingerprint": bank.fingerprint(), "bank": {"n_tasks": len(bank.tasks), "n_outputs": bank.n_outputs,
                                                             "solver": builds.get("bank_solver")},
            "board_run_sd": BOARD_RUN_SD, "board_run_sd_source": BOARD_RUN_SD_SOURCE, "min_pairs": MIN_PAIRS,
            "status": status, "n_pairs": len(pairs), "fit": fit, "leave_one_out": loo,
            "sign_agreement": sign_agreement(pairs) if len(pairs) >= 2 else None, "rows": rows}


def render(led: dict) -> str:
    out = [f"calibration ledger: {len(led['rows'])} submissions, {led['n_pairs']} (bank, board) pairs; {led['status']}",
           "| entry | board | bank pts | bank task pts | rule | expected | LOO expected | note |", "|---|---|---|---|---|---|---|---|"]
    for r in led["rows"]:
        out.append(f"| {r['entry_id']} | {'pending' if r['board'] is None else r['board']} | "
                   f"{'-' if r['bank_points'] is None else r['bank_points']} | "
                   f"{'-' if r['bank_task_points'] is None else r['bank_task_points']} | "
                   f"{r['rule'] or '-'}{' (approx)' if r['approx'] else ''} | "
                   f"{'-' if r['expected_board'] is None else r['expected_board']} | "
                   f"{'-' if r['loo_expected_board'] is None else r['loo_expected_board']} | {r['note'][:70]} |")
    if led["fit"]:
        f, l = led["fit"], led["leave_one_out"]
        out.append(f"fit: board = {f['a']} + {f['b']} x bank (n {f['n']}, residual SD {f['residual_sd']} points, "
                   f"R2 {f['r2']}); leave-one-out RMSE {l['rmse']} points on {l['n']} pairs; board run SD "
                   f"{led['board_run_sd']} points for reference")
    sa = led.get("sign_agreement")
    if sa:
        out.append(f"sign agreement of bank and board differences between pairs: {sa['agree']} of {sa['n_comparisons']}")
        for r in sa["rows"]:
            out.append(f"  {r['from']} -> {r['to']}: board {r['board_delta']:+.2f}, bank {r['bank_delta']:+.2f}, "
                       f"{'same sign' if r['same_sign'] else ('opposite' if r['same_sign'] is False else 'tie')}")
    return "\n".join(out)


def write_into_submissions(led: dict, path: Path = SUBMISSIONS) -> int:
    """expected_board and bank_points into the matching submissions ledger entries; returns the count changed."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    ents = doc["entries"] if isinstance(doc, dict) else doc
    by_id = {r["entry_id"]: r for r in led["rows"]}
    n = 0
    for e in ents:
        r = by_id.get(e.get("entry_id"))
        if r is None or r["bank_points"] is None:
            continue
        new = {"bank_points": r["bank_points"], "expected_board": r["expected_board"]}
        if any(e.get(k) != v for k, v in new.items()):
            e.update(new)
            n += 1
    if n:
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submissions", default=str(SUBMISSIONS))
    ap.add_argument("--builds", default=str(BUILDS))
    ap.add_argument("--pools", nargs="*")
    ap.add_argument("--q27", default="")
    ap.add_argument("--json", default=str(OUT))
    ap.add_argument("--write-submissions-ledger", action="store_true",
                    help="put bank_points and expected_board into submissions/ledger.json entries")
    args = ap.parse_args(argv)
    bank = load_bank([Path(p) for p in args.pools] if args.pools else list(BANK_PARTS), args.q27 or Q27_BANK)
    builds = json.loads(Path(args.builds).read_text(encoding="utf-8"))
    led = build_ledger(submissions(Path(args.submissions)), builds, bank)
    print(render(led))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(led, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written to {args.json}")
    if args.write_submissions_ledger:
        print(f"submissions ledger: {write_into_submissions(led, Path(args.submissions))} entries updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
