"""R1z pre-run check (experiments/arc3-lawlib/R1Z_RULE.md): wm_laws fitted on the 35 stuck states' own frames.

Each recovery case (experiments/arc3-recovery/cases/<case>/) is replayed in the local engine from its recorded action
prefix; the transitions on the case's target level (what the agent had seen of that level at the cut) are passed to
`wm_fit_all` exactly as the sandbox would pass them (the tools' record format, 8 s budget). Per case: transitions,
the composed model's match rate, the best single law at 0.9 or better with at least 3 transitions checked, the
kinds in the model, the fit's wall time and whether the budget ran out. The rule's validity line asks for a law at
0.9 or better on at least 10 of the 35 levels; this is that count on the cut-point frames, before any run.

    python -I scripts/arc3_lawlib_cases.py [--workers 8] [--out experiments/arc3-lawlib]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments" / "arc3-recovery" / "cases"
MODULE = ROOT / "kaggle" / "arc3" / "lawlib" / "wm_laws.py"
ENV_DIR = Path("~/data/arc3/environment_files")
HEX = "0123456789abcdef"
KEYS = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
        "ACTION7": "UNDO"}


def _wm():
    spec = importlib.util.spec_from_file_location("wm_laws", MODULE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rows_of(frame) -> list:
    return ["".join(HEX[int(v)] for v in r) for r in frame.frame[-1]]


def replay(case_dir: Path):
    """Transition records for the case's prefix (the tools' format) and the case's meta."""
    import arc_agi
    from arcengine import GameAction, GameState
    meta = json.loads((case_dir / "meta.json").read_text())
    prefix = json.loads((case_dir / "prefix_actions.json").read_text())
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=str(ENV_DIR.expanduser()))
    env = arcade.make(meta["game_id"])
    f = env.reset()
    recs = []
    for i, a in enumerate(prefix):
        name = a["name"]
        before, lc0 = rows_of(f), int(f.levels_completed)
        if name == "RESET":
            if i == 0:
                continue
            f = env.step(GameAction.RESET)
            act = "RESET"
        elif name == "ACTION6":
            d = a.get("data") or {}
            f = env.step(GameAction.ACTION6, data={"x": int(d["x"]), "y": int(d["y"])})
            act = ("MOUSE", int(d["y"]), int(d["x"]))
        else:
            f = env.step(GameAction.from_name(name))
            act = KEYS.get(name, name)
        recs.append({"action": act, "before": before, "after": rows_of(f), "level": lc0 + 1,
                     "level_completed": int(f.levels_completed) > lc0, "game_over": f.state == GameState.GAME_OVER,
                     "reset": act == "RESET"})
    return meta, recs


def run(case_dir: str) -> dict:
    logging.disable(logging.WARNING)
    W = _wm()
    case_dir = Path(case_dir)
    t0 = time.time()
    meta, recs = replay(case_dir)
    on_level = [r for r in recs if r["level"] == meta["target_level"]]
    fits = W.wm_fit_all(recs, max_seconds=8.0)
    rep = dict(W.wm_last_fit)
    model = fits[0] if fits and fits[0].get("role") == "model" else None
    laws = [f for f in fits if f.get("role") == "law"]
    good = [f for f in laws if f["match_rate"] >= 0.9 and f["checked"] >= 3]
    goals = [f for f in fits if f.get("role") == "goal"]
    return {"case": case_dir.name, "game": meta["game_id"].split("-")[0], "level": meta["target_level"],
            "prefix": len(recs), "transitions": rep.get("transitions", 0), "on_level_records": len(on_level),
            "model_rate": model["match_rate"] if model else None, "model_checked": model["checked"] if model else 0,
            "model_kinds": sorted({l["kind"] for l in model["laws"]}) if model else [],
            "model": W.wm_describe(model)[:300] if model else None,
            "best_law": W.wm_describe(good[0])[:200] if good else None,
            "best_law_rate": good[0]["match_rate"] if good else (laws[0]["match_rate"] if laws else None),
            "laws_at_0_9": len(good), "goals": len(goals),
            "fit_s": rep.get("elapsed_s"), "timed_out": rep.get("timed_out"), "families_cut": rep.get("families_cut"),
            "per_kind_s": rep.get("per_kind_s"),
            "wall_s": round(time.time() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "arc3-lawlib"))
    a = ap.parse_args(argv)
    cases = sorted(str(p) for p in CASES.iterdir() if (p / "prefix_actions.json").exists())
    with Pool(a.workers) as pool:
        rows = pool.map(run, cases)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cases_fit.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    with_law = sum(1 for r in rows if r["laws_at_0_9"])
    model90 = sum(1 for r in rows if (r["model_rate"] or 0) >= 0.9)
    lines = ["| track | case | transitions on level | model match | model kinds | laws at 0.9 (>= 3 checked) | "
             "fit s | families cut |", "|---|---|---:|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append("| ARC-3 | %s | %d | %s | %s | %d | %.2f | %s |" % (
            r["case"], r["transitions"], "%.2f of %d" % (r["model_rate"], r["model_checked"])
            if r["model_rate"] is not None else "none", ", ".join(r["model_kinds"]) or "-", r["laws_at_0_9"],
            r["fit_s"] or 0, ", ".join(r["families_cut"] or []) or "-"))
    summary = {"cases": n, "with_law_at_0_9": with_law, "model_at_0_9": model90,
               "max_fit_s": max((r["fit_s"] or 0) for r in rows),
               "over_budget": sum(1 for r in rows if r["timed_out"])}
    (out / "cases_fit.md").write_text("\n".join(lines) + "\n\n" + json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
