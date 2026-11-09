#!/usr/bin/env python3
"""Decode-many replay (experiments/arc2-bank/DM_RULE.md section 3): every narrower search setting rebuilt on CPU from
the superset run's logs (dm/<task_id>.json.gz, written by kaggle/arc2/bank/build_dm_notebook.py's arc_solver.py).

    python scripts/arc2_dm_replay.py --run experiments/runs/kaggle-A2-dm-k1-v1 [--run ...] --out experiments/arc2-bank/dm

Settings: views {16, 32} x floor {0.2, 0.1} x rescoring {8, 16, 32}; the ranking (kgmon, probmul3) is applied by the
reader on each setting's pool. Under the bank's caps (1,200 s per task, 540 s per DFS batch), per task:
  calls      a DFS call exists at floor f iff it popped a prefix below -log f (logged: every call pops below -log 0.1;
             call_lo02 marks those below -log 0.2); its time is its interval to the next call (the last: to the DFS
             end); the batch's prefill is the time to its first call
  DFS cap    calls run while the emulated DFS time at their start, from the end of the prefill call (the solver's
             start_time), is below 540 s
  beams      the batch's beams with score below -log f whose completing call ran (-1 = the prefill)
  rescoring  a grid's first r scores; its time (setup of the bank's 8, the share of the other 24's setup, the first
             r / 4 calls) charged at the batch where the setting first finds the grid
  other      the batch's remaining wall (tokenising, decoding beams, files), charged in full
  task cap   a batch is decoded only while training plus the emulated seconds so far are at most 1,200 s
"Coverage short": the setting would decode a batch the run never decoded, or would run a DFS past where the run's
own DFS cap (1,620 s) cut it. Each setting's pool is written in the bank's format (arc2_pool_audit.v1) with the
emulated seconds as elapsed_s, for eval/arc2_bank.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "arc2_dm_replay.v1"
VIEWS = (16, 32)
FLOORS = (0.2, 0.1)
RESCORES = (8, 16, 32)
TASK_CAP_S = 1200.0
DFS_CAP_S = 540.0
RUN_DFS_CAP_S = 1620.0
RUN_FLOOR = 0.1
LO_SCORE = -math.log(0.2)
OUT = ROOT / "experiments" / "arc2-bank" / "dm"
KAGGLE_SOLUTIONS_COPY = "_kaggle_solutions_audit_tasks.json"


def setting_name(views: int, floor: float, r: int) -> str:
    return f"v{views}_f{floor:g}_r{r}"


SETTINGS = [(v, f, r) for v in VIEWS for f in FLOORS for r in RESCORES]


def load_log(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# one DFS batch

def emulate_dfs(batch: dict, floor: float, cap_s: float = DFS_CAP_S, run_cap_s: float = RUN_DFS_CAP_S) -> dict:
    """The batch's DFS at ``floor`` under ``cap_s``: {time_s, last_call (calls with index below it ran), cut, short}."""
    starts = [u / 1e6 for u in batch.get("call_us") or []]
    lo = batch.get("call_lo02") or [1] * len(starts)
    dfs_s = float(batch["dfs_s"])
    if len(lo) != len(starts):
        raise ValueError(f"batch {batch.get('i')}: {len(starts)} call starts, {len(lo)} flags")
    prefill = starts[0] if starts else dfs_s
    t = prefill
    stop_at = len(starts)
    for c, s in enumerate(starts):
        if floor < 0.2 - 1e-12 or lo[c]:           # the call exists at this floor
            if t - prefill >= cap_s:              # the solver's clock starts after the prefill call
                stop_at = c
                break
            nxt = starts[c + 1] if c + 1 < len(starts) else dfs_s
            t += nxt - s
    cut = stop_at < len(starts)
    run_cut = dfs_s >= run_cap_s
    return {"time_s": t, "last_call": stop_at, "cut": cut, "short": run_cut and not cut}


def beam_kept(beam: dict, floor: float, last_call: int) -> bool:
    if not beam["score"] < -math.log(floor):
        return False
    c = beam.get("call")
    if c is None:
        raise ValueError(f"beam without its completing call: {beam}")
    return c < last_call


def rescore_seconds(grid: dict, r: int) -> float:
    s8, s24 = (grid.get("setup_s") or [0.0, 0.0])[:2]
    return float(s8) + float(s24) * (r - 8) / 24 + sum(grid["call_s"][: r // 4])


# ----------------------------------------------------------------------------
# one task

def batch_walls(log: dict) -> list[float]:
    bs = log["batches"]
    out = []
    for i, b in enumerate(bs):
        end = bs[i + 1]["t0_s"] if i + 1 < len(bs) else float(log["elapsed_s"])
        out.append(end - b["t0_s"])
    return out


def emulate_task(log: dict, views: int, floor: float, r: int, *, task_cap_s: float = TASK_CAP_S,
                 dfs_cap_s: float = DFS_CAP_S, run_dfs_cap_s: float = RUN_DFS_CAP_S) -> dict:
    """One setting on one task: the beams it keeps (with their emulated find time) and its seconds."""
    n_a = int(log["n_batches"]["a"])
    n_b = int(log["n_batches"]["b"])
    wanted = n_a if views == 16 else n_a + n_b
    decoded = log["batches"]
    walls = batch_walls(log)
    grids = log["grids"]
    t = float(decoded[0]["t0_s"]) if decoded else float(log["train_s"])   # training and the view set-up
    seen: set[int] = set()
    beams, stop, short, dfs_cut, n_dec = [], "exhausted", [], 0, 0
    for i in range(wanted):
        if t > task_cap_s:
            stop = "per_task_cap"
            break
        if i >= len(decoded):
            short.append(f"batch {i} not decoded by the run")
            break
        b = decoded[i]
        if int(b["i"]) != i:
            raise ValueError(f"{log['task_id']}: batch {b['i']} at position {i}")
        n_dec += 1
        e = emulate_dfs(b, floor, dfs_cap_s, run_dfs_cap_s)
        dfs_cut += int(e["cut"])
        if e["short"]:
            short.append(f"batch {i}: the run's DFS cap cut it first")
        t_batch = t + (b["call_us"][0] / 1e6 if b.get("call_us") else 0.0)
        resc = 0.0
        for bm in b["beams"]:
            if not beam_kept(bm, floor, e["last_call"]):
                continue
            g = int(bm["grid"])
            if g not in seen:
                seen.add(g)
                resc += rescore_seconds(grids[g], r)
            beams.append({"view": bm["view"], "out": int(bm["out"]), "score": float(bm["score"]), "grid": g,
                          "batch": i, "order": len(beams), "found_s": round(t_batch, 3)})
        other = walls[i] - float(b["dfs_s"]) - float(b.get("rescore_s") or 0.0) - float((b.get("exact_check") or {}).get("s") or 0.0)
        t += e["time_s"] + max(other, 0.0) + resc
    if stop == "exhausted" and dfs_cut:
        stop = "dfs_cap"
    return {"beams": beams, "elapsed_s": round(t, 3), "stop_reason": stop, "coverage_short": short,
            "batches_decoded": n_dec, "batches_total": wanted, "dfs_batches_cut_540": dfs_cut}


# ----------------------------------------------------------------------------
# ranking, as arc_pool_export.rank_output / arc_decoder (kgmon, probmul3)

def _hashable(grid) -> tuple:
    return tuple(tuple(int(x) for x in row) for row in grid)


def rank_output(samples: list[dict]) -> tuple[list[dict], list]:
    """samples: [{key, beam_score, score_aug, solution, order, found_s}] in discovery order. (candidates in kgmon order
    with the export's fields, the selected pair)."""
    groups: dict[tuple, list[dict]] = {}
    for s in samples:
        groups.setdefault(_hashable(s["solution"]), []).append(s)

    def kg(m):
        return len(m) - sum(sum(x["score_aug"]) / len(x["score_aug"]) for x in m) / len(m)

    def pm(m):
        return (sum(3 - x["beam_score"] for x in m)
                + sum(sum(3 - s for s in x["score_aug"]) for x in m) / len(m))

    items = list(groups.items())
    by_kg = sorted(items, key=lambda it: kg(it[1]), reverse=True)
    by_pm = sorted(items, key=lambda it: pm(it[1]), reverse=True)
    pm_rank = {h: (r + 1, pm(m)) for r, (h, m) in enumerate(by_pm)}
    cands = []
    for r, (h, m) in enumerate(by_kg):
        first = min(m, key=lambda x: x["order"])
        cands.append({
            "grid": [list(row) for row in h], "rank_kgmon": r + 1, "score_kgmon": kg(m),
            "rank_probmul3": pm_rank[h][0], "score_probmul3": pm_rank[h][1], "n_beams": len(m),
            "score_aug": list(m[0]["score_aug"]),
            "score_aug_mean": sum(sum(x["score_aug"]) / len(x["score_aug"]) for x in m) / len(m),
            "first_found_order": first["order"], "first_found_s": first["found_s"],
            "beams": [{"key": x["key"], "beam_score": x["beam_score"],
                       "aug_mean": sum(x["score_aug"]) / len(x["score_aug"]), "order": x["order"],
                       "elapsed_s": x["found_s"]} for x in m],
        })
    selected = [c["grid"] for c in cands[:2]]
    return cands, selected


def pool_doc(log: dict, em: dict, r: int, n_test: int, setting: str) -> dict:
    grids = log["grids"]
    per_out: dict[int, list[dict]] = {}
    for bm in em["beams"]:
        g = grids[bm["grid"]]
        j = int(g["bk"].rsplit("_", 1)[1])
        per_out.setdefault(j, []).append({"key": f"{bm['view']}.out{bm['out']}", "beam_score": bm["score"],
                                          "score_aug": list(g["score_aug"][:r]), "solution": g["grid"],
                                          "order": bm["order"], "found_s": bm["found_s"]})
    outputs = []
    for j in range(n_test):
        cands, sel = rank_output(per_out.get(j, []))
        outputs.append({"test_index": j, "n_beams": len(per_out.get(j, [])), "n_unique": len(cands),
                        "rank_check_ok": True,
                        "margin_kgmon": (cands[0]["score_kgmon"] - cands[1]["score_kgmon"]) if len(cands) > 1 else None,
                        "candidates": cands, "selected": sel,
                        "submitted": {"attempt_1": sel[0] if sel else [[0]],
                                      "attempt_2": sel[1] if len(sel) > 1 else [[0]]}})
    run = {"elapsed_s": em["elapsed_s"], "stop_reason": em["stop_reason"], "loop_stop": em["stop_reason"],
           "batches_total": em["batches_total"], "batches_decoded": em["batches_decoded"],
           "dfs_batches_cut_540": em["dfs_batches_cut_540"], "dfs_batches_cut_end_time": 0,
           "per_task_cap_s": TASK_CAP_S, "dfs_cap_s": DFS_CAP_S, "end_time_reached": False,
           "coverage_short": em["coverage_short"], "setting": setting, "replayed": True,
           "train_s": log["train_s"], "rank": log.get("rank")}
    return {"schema": "arc2_pool_audit.v1", "task_id": log["task_id"], "n_test": n_test,
            "selection": "decode-many replay: kgmon order (arc_decoder.score_kgmon formula)", "run": run,
            "outputs": outputs}


# ----------------------------------------------------------------------------
# a run (one or more kernel archives)

def task_logs(run_dirs: list[Path], run_tasks: dict[Path, list[str]] | None = None) -> dict[str, Path]:
    """Task -> log path. With run_tasks, a run's logs count only for the tasks run in it: a log cut short in one
    kernel (the proof's e87109e9) does not clash with the full run of that task in another."""
    out = {}
    for d in run_dirs:
        keep = None if run_tasks is None else set(run_tasks[d])
        for p in sorted((Path(d) / "dm").glob("*.json.gz")):
            t = p.name[: -len(".json.gz")]
            if keep is not None and t not in keep:
                continue
            if t in out:
                raise ValueError(f"{t} logged in two runs: {out[t]} and {p}")
            out[t] = p
    return out


def replay(run_dirs: list[Path], out_dir: Path, *, tasks: list[str] | None = None,
           n_tests: dict[str, int] | None = None, run_tasks: dict[Path, list[str]] | None = None) -> dict:
    """Write out_dir/pools/<setting>/<task>.json for every setting and task logged; return a summary."""
    logs = task_logs(run_dirs, run_tasks)
    if tasks is not None:
        logs = {t: p for t, p in logs.items() if t in set(tasks)}
    sols = {}
    for d in run_dirs:
        kag = Path(d) / "pool" / KAGGLE_SOLUTIONS_COPY
        if kag.is_file():
            sols.update(json.loads(kag.read_text(encoding="utf-8")))
    summary = {"schema": SCHEMA, "runs": [str(d) for d in run_dirs], "settings": {}, "tasks": sorted(logs)}
    for v, f, r in SETTINGS:
        name = setting_name(v, f, r)
        d = out_dir / "pools" / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        per = {}
        for t, p in sorted(logs.items()):
            log = load_log(p)
            n_test = int(log.get("n_test") or (n_tests or {})[t])
            em = emulate_task(log, v, f, r)
            (d / f"{t}.json").write_text(json.dumps(pool_doc(log, em, r, n_test, name), separators=(",", ":")),
                                         encoding="utf-8")
            per[t] = {"elapsed_s": em["elapsed_s"], "stop_reason": em["stop_reason"],
                      "coverage_short": bool(em["coverage_short"]), "n_beams": len(em["beams"])}
        if sols:
            (d / KAGGLE_SOLUTIONS_COPY).write_text(json.dumps({t: sols[t] for t in logs if t in sols}), encoding="utf-8")
        (d / "_run.json").write_text(json.dumps({"schema": SCHEMA, "run": {"setting": name, "views": v, "floor": f,
                                                                          "rescore_n": r, "replayed": True}}),
                                     encoding="utf-8")
        summary["settings"][name] = {"views": v, "floor": f, "rescore_n": r, "tasks": per,
                                     "seconds": round(sum(x["elapsed_s"] for x in per.values()), 1),
                                     "coverage_short": sorted(t for t, x in per.items() if x["coverage_short"])}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="a kernel archive holding dm/ and pool/")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    s = replay([Path(r) for r in args.run], out)
    (out / "dm_replay.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    for name, x in s["settings"].items():
        print(f"{name}: {len(x['tasks'])} tasks, {x['seconds']:.0f} s, coverage short {len(x['coverage_short'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
