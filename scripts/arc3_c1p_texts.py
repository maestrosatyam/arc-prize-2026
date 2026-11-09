#!/usr/bin/env python3
"""C1P control texts (run a3-c1p-placebo, arm A7cF): R1f's length-matched control text for every C1 level start.

Rule: reports/research/ARC3_C1P_PLACEBO_RULE_2026-10-29.md (frozen f4c1658; revision 1 at 1b8666f, before any text or
build). The texts come from R1f's control rule applied unchanged (kaggle/arc3/c1p/c1p_control.py, a verbatim copy of the
R1f builder's select_control); none is edited by hand.

Checks (any failure stops the script, nothing is written):
  - the pinned inputs: C1's level list (experiments/arc3-c1/levels.jsonl, 7d7abd6), rules.json, the R1f case bundle;
  - the pipeline reproduces the 35 R1 cases' oracle.json control entries (message byte for byte, and every field) from
    the R1f bundle R1f, R1g and T1 ran on;
  - the levels with no core-claim subset within 20% of the insight's length (select_control raises) are exactly the 11
    revision 1 names; they are left out and recorded;
  - every kept message is within r1_recovery's 1,500-character cap and its claims are in rules.json.

Outputs (experiments/arc3-c1p/): texts.jsonl (one row per kept level in C1's list order: case_id, game, level, kernel,
fallback, the control entry), excluded.jsonl (the 11 and why), check.json (counts, fallback levels and their shared
words, the 35-case reproduction, hashes).

Usage: envs/arc3/bin/python -I scripts/arc3_c1p_texts.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "kaggle" / "arc3" / "c1"), str(ROOT / "kaggle" / "arc3" / "c1p")]
import c1_oracle  # noqa: E402
import c1p_control  # noqa: E402

OUT = ROOT / "experiments" / "arc3-c1p"
C1_LEVELS = ROOT / "experiments" / "arc3-c1" / "levels.jsonl"
C1_LEVELS_SHA256 = "1c8a0dff01f4a5ec07068f2ca5640ee60e34fa672734095911b498c51a21d6ab"
RULES = ROOT / "experiments" / "arc3-rules" / "rules.json"
RULES_SHA256 = "0bce2aaf6408285b337fcd35b5ffe2fbe1b887e36379d103d77b23dcaff89784"
R1F_BUNDLE = ROOT / "kaggle" / "arc3" / "r1" / "r1f" / "dataset" / "r1_cases.tar.gz"
R1F_BUNDLE_SHA256 = "e50cdc86c06807f137bcf5b52013532cd0c8872180fc7baae5fa8da24383b303"
R1_BUILDER = ROOT / "kaggle" / "arc3" / "r1" / "build_r1_notebook.py"
R1_BUILDER_SHA256 = "f837829a7bb3ede3db5552c8e2b882dcd162e7a2e578c32ae7be80cadfcb876f"
# revision 1: no length-matched control (select_control raises); left out of the run and the reading
NO_CONTROL = ("dc22_L2", "ft09_L3", "ft09_L6", "lp85_L1", "tn36_L1", "tr87_L2", "tr87_L3", "tr87_L4", "tu93_L1",
              "wa30_L7", "wa30_L8")
KERNELS = ("k1", "k2", "k3")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r1f_oracles() -> dict[str, dict]:
    """case id -> its oracle.json in the R1f bundle."""
    out = {}
    with tarfile.open(R1F_BUNDLE, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith("/oracle.json"):
                out[m.name.split("/", 1)[0]] = json.loads(tar.extractfile(m).read())
    return out


def texts_sha256(kept: list[dict]) -> str:
    """The control texts as the run uses them: [case_id, message] in list order."""
    return hashlib.sha256(json.dumps([[r["case_id"], r["control"]["message"]] for r in kept],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def build() -> dict:
    assert sha256_file(C1_LEVELS) == C1_LEVELS_SHA256, "C1's levels.jsonl differs from the committed list"
    assert sha256_file(RULES) == RULES_SHA256, "rules.json differs from the pinned version"
    assert sha256_file(R1F_BUNDLE) == R1F_BUNDLE_SHA256, "R1f bundle differs"
    builder = {"path": str(R1_BUILDER.relative_to(ROOT)), "present": R1_BUILDER.is_file(),
               "sha256": sha256_file(R1_BUILDER) if R1_BUILDER.is_file() else None}
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    levels = [json.loads(l) for l in C1_LEVELS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(levels) == 157

    # the 35 R1 cases: the control entry recomputed from rules.json against R1f's oracle.json
    repro = []
    for cid, o in sorted(r1f_oracles().items()):
        got = c1p_control.control_entry(rules, o["game"], int(o["level"]))
        want = o["control"]
        repro.append({"case_id": cid, "level": f"{o['game']}_L{int(o['level'])}",
                      "message_bytes_equal": got["message"].encode("utf-8") == want["message"].encode("utf-8"),
                      "entry_equal": got == want, "fallback": bool(want["shared_words"])})
    bad = [r for r in repro if not (r["message_bytes_equal"] and r["entry_equal"])]
    if len(repro) != 35 or bad:
        raise SystemExit(f"R1f control reproduction failed: {len(repro)} cases, mismatched {bad}")

    kept, excluded, problems = [], [], []
    for lv in levels:
        cid = lv["case_id"]
        try:
            control = c1p_control.control_entry(rules, lv["game"], int(lv["level"]))
        except ValueError as exc:
            excluded.append({"case_id": cid, "game": lv["game"], "level": lv["level"], "kernel": lv["kernel"],
                             "insight_chars": len(lv["insight"]["text"]),
                             "reason": f"no control: {exc} (select_control, R1f's tolerance)"})
            continue
        insight = c1_oracle.game_terms(c1_oracle.strip_action_sequences(
            rules[lv["game"]]["levels"][str(lv["level"])]["key_insight"]))
        if insight != lv["insight"]["text"]:
            problems.append(f"{cid}: the insight text the control is matched to differs from C1's")
        if not control["text"].strip() or not control["claim_ids"]:
            problems.append(f"{cid}: empty control text or no claim id")
        elif len(control["message"]) > c1_oracle.ORACLE_MAX_MESSAGE_CHARS:
            problems.append(f"{cid}: control message {len(control['message'])} chars")
        elif "missing" in control["claims_status"].values():
            problems.append(f"{cid}: control claim id not in rules.json: {control['claim_ids']}")
        kept.append({"case_id": cid, "game": lv["game"], "level": lv["level"], "kernel": lv["kernel"],
                     "insight_chars": len(lv["insight"]["text"]), "fallback": bool(control["shared_words"]),
                     "control": control})
    if sorted(r["case_id"] for r in excluded) != sorted(NO_CONTROL):
        problems.append(f"levels with no control {sorted(r['case_id'] for r in excluded)} differ from revision 1's "
                        f"{sorted(NO_CONTROL)}")
    if problems:
        raise SystemExit("C1P texts check failed:\n  " + "\n  ".join(problems))
    return {"kept": kept, "excluded": excluded, "repro": repro, "builder": builder}


def write(res: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    kept, excluded = res["kept"], res["excluded"]
    tb = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in kept).encode("utf-8")
    (out / "texts.jsonl").write_bytes(tb)
    (out / "excluded.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in excluded), encoding="utf-8")
    fb = [r for r in kept if r["fallback"]]
    run_levels = {r["case_id"] for r in kept}
    summary = {
        "levels": len(kept), "excluded": len(excluded), "c1_levels": len(kept) + len(excluded),
        "texts_jsonl_sha256": hashlib.sha256(tb).hexdigest(), "texts_sha256": texts_sha256(kept),
        "fallback_fewest_shared_words": len(fb), "clean": len(kept) - len(fb),
        "fallback_levels": {r["case_id"]: r["control"]["shared_words"] for r in fb},
        "excluded_levels": {r["case_id"]: r["insight_chars"] for r in excluded},
        "length_ratio": {"min": min(r["control"]["length_ratio"] for r in kept),
                         "max": max(r["control"]["length_ratio"] for r in kept)},
        "inputs": {"c1 levels.jsonl": C1_LEVELS_SHA256, "rules.json": RULES_SHA256, "r1f r1_cases.tar.gz": R1F_BUNDLE_SHA256,
                   "r1 builder (select_control source)": res["builder"],
                   "c1_oracle.py": sha256_file(ROOT / "kaggle" / "arc3" / "c1" / "c1_oracle.py"),
                   "c1p_control.py": sha256_file(ROOT / "kaggle" / "arc3" / "c1p" / "c1p_control.py")},
        "r1f_reproduction": {"cases": len(res["repro"]),
                             "message_bytes_equal": sum(r["message_bytes_equal"] for r in res["repro"]),
                             "entry_equal": sum(r["entry_equal"] for r in res["repro"]), "cases_detail": res["repro"]},
        "r1_levels_in_run": sorted({r["level"] for r in res["repro"]} & run_levels),
        "kernels": {k: {"levels": sum(1 for r in kept if r["kernel"] == k),
                        "games": sorted({r["game"] for r in kept if r["kernel"] == k})} for k in KERNELS},
        "by_game": dict(sorted(collections.Counter(r["game"] for r in kept).items())),
    }
    (out / "check.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    s = write(build(), Path(a.out))
    print(json.dumps({k: v for k, v in s.items() if k not in ("r1f_reproduction", "fallback_levels")}, indent=1,
                     ensure_ascii=False))
    print(f"r1f reproduction: {s['r1f_reproduction']['message_bytes_equal']}/35 control messages byte-equal, "
          f"{s['r1f_reproduction']['entry_equal']}/35 entries equal; {s['levels']} levels kept, {s['excluded']} without "
          f"a control, {s['fallback_fewest_shared_words']} fallback; texts sha256 {s['texts_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
