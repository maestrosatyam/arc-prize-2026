"""Development-time calibration of DiverseSelector parameters (B3: calibrated on dev/tune only).

``fit_weights(records)`` grid-searches (tau, w_symbolic, w_loo, w_neural) to maximise pass@2 on labelled records
``{"key": base_key, "candidates": [Candidate, ...], "label": Grid}``; ties broken by higher pass@1, then by the
parameter set closest to the defaults (first in grid order). It returns a ``Calibration`` that
``write_calibration`` stores as a versioned JSON (schema ``arc2.selector_calibration.v1``) with the fitted
parameters, the search grid, the record count and a sha256 of the record keys, so a later reader can tell which
development set produced it. ``load_calibration`` builds a selector from the JSON; the selector never sees labels.

Labels are only read by this module's CLI through ``eval/data`` solutions for the ``dev``/``tune`` split ids; the
holdout ids are refused (splits/holdout_policy.md).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc2.io.types import Candidate, Grid
from arc2.selection.diverse import DEFAULT_PARAMS, DiverseSelector
from arc2.selection.metrics import aggregate, output_metrics

__all__ = ["DEFAULT_GRID", "Calibration", "fit_weights", "load_calibration", "write_calibration"]

SCHEMA = "arc2.selector_calibration.v1"
DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "tau": (0.1, 0.05, 0.2, 0.3, 0.5),
    "w_symbolic": (1000.0, 10.0, 2.0, 0.0),
    "w_loo": (1.0, 0.0, 5.0),
    "w_neural": (1.0,),
}


@dataclass
class Calibration:
    params: dict[str, float]
    pass_at_1: float
    pass_at_2: float
    n_records: int
    records_sha256: str
    grid: dict[str, list[float]]
    n_evaluated: int
    baseline: dict[str, Any] = field(default_factory=dict)  # metrics of the default parameters on the same records
    version: str = ""
    created_utc: str = ""
    notes: str = ""

    def selector(self, **kw: Any) -> DiverseSelector:
        return DiverseSelector.from_params(self.params, calibration_version=self.version or None, **kw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": self.version,
            "created_utc": self.created_utc,
            "params": self.params,
            "pass_at_1": self.pass_at_1,
            "pass_at_2": self.pass_at_2,
            "n_records": self.n_records,
            "records_sha256": self.records_sha256,
            "grid": self.grid,
            "n_evaluated": self.n_evaluated,
            "baseline_default_params": self.baseline,
            "notes": self.notes,
        }


def _records_sha(records: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(str(r["key"]).encode())
        h.update(b"\0")
    return h.hexdigest()


def evaluate_params(params: Mapping[str, float], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sel = DiverseSelector.from_params(params)
    rows = []
    for r in records:
        cands: list[Candidate] = list(r["candidates"])
        ti = cands[0].test_index if cands else 0
        a1, a2 = sel.select(cands, ti)
        rows.append(output_metrics([a1, a2], [c.grid for c in cands if c.test_index == ti], r["label"]))
    return aggregate(rows)


def fit_weights(records: Sequence[Mapping[str, Any]], grid: Mapping[str, Iterable[float]] | None = None, version: str = "", notes: str = "") -> Calibration:
    """Grid-search DiverseSelector parameters maximising pass@2 (then pass@1) on labelled ``records``."""
    if not records:
        raise ValueError("fit_weights needs at least one labelled record")
    for r in records:
        if not {"key", "candidates", "label"} <= set(r):
            raise ValueError("record needs key, candidates, label")
    g = {k: [float(v) for v in (grid or DEFAULT_GRID)[k]] for k in DEFAULT_PARAMS}
    names = list(DEFAULT_PARAMS)
    best: tuple[tuple[float, float], dict[str, float], dict[str, Any]] | None = None
    n_eval = 0
    for values in itertools.product(*(g[n] for n in names)):
        params = dict(zip(names, values))
        if params["tau"] <= 0 or params["tau"] > 1:
            continue
        m = evaluate_params(params, records)
        n_eval += 1
        key = (m["pass_at_2"], m["pass_at_1"])
        if best is None or key > best[0]:
            best = (key, params, m)
    assert best is not None
    _, params, m = best
    return Calibration(
        params=params,
        pass_at_1=m["pass_at_1"],
        pass_at_2=m["pass_at_2"],
        n_records=len(records),
        records_sha256=_records_sha(records),
        grid=g,
        n_evaluated=n_eval,
        baseline=evaluate_params(DEFAULT_PARAMS, records),
        version=version or f"b3-cal-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        created_utc=datetime.now(UTC).isoformat(),
        notes=notes,
    )


def write_calibration(cal: Calibration, path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cal.as_dict(), fh, indent=1, sort_keys=True)
    os.replace(tmp, p)
    return p


def load_calibration(path: str | os.PathLike[str]) -> Calibration:
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {obj.get('schema')!r}")
    params = {k: float(obj["params"][k]) for k in DEFAULT_PARAMS}
    return Calibration(
        params=params,
        pass_at_1=float(obj.get("pass_at_1", 0.0)),
        pass_at_2=float(obj.get("pass_at_2", 0.0)),
        n_records=int(obj.get("n_records", 0)),
        records_sha256=str(obj.get("records_sha256", "")),
        grid={k: [float(x) for x in v] for k, v in obj.get("grid", {}).items()},
        n_evaluated=int(obj.get("n_evaluated", 0)),
        baseline=dict(obj.get("baseline_default_params", {})),
        version=str(obj.get("version", "")),
        created_utc=str(obj.get("created_utc", "")),
        notes=str(obj.get("notes", "")),
    )


# ----------------------------------------------------------------------------------------------
# CLI: calibrate on an exported Kaggle candidate pool (neural) + optional symbolic JSON, dev/tune only
# ----------------------------------------------------------------------------------------------


def _label_safe_ids(split_names: Sequence[str], manifest_path: Path) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    ids: list[str] = []
    for s in split_names:
        if s == "holdout":
            raise SystemExit("refusing to calibrate on the sealed holdout split")
        if s not in manifest["splits"]:
            raise SystemExit(f"unknown split {s!r}")
        ids.extend(manifest["splits"][s])
    return sorted(set(ids))


def build_records(pool: Mapping[str, list[Candidate]], solutions: Mapping[str, list[Grid]], task_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Labelled records for every test output of ``task_ids`` that has a non-empty pool."""
    wanted = set(task_ids)
    records = []
    for bk, cands in sorted(pool.items()):
        tid, _, idx = bk.rpartition("_")
        if tid not in wanted or tid not in solutions or not cands:
            continue
        i = int(idx)
        if i >= len(solutions[tid]):
            continue
        records.append({"key": bk, "candidates": list(cands), "label": solutions[tid][i]})
    return records


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Fit DiverseSelector parameters on dev/tune pools and write a versioned calibration JSON.")
    ap.add_argument("--neural-dir", help="directory of baseline pickles (/kaggle/inference_outputs export, untarred)")
    ap.add_argument("--symbolic-json", help="symbolic_candidates.json written by write_symbolic_json")
    ap.add_argument("--solutions", default=str(root / "eval" / "data" / "arc-agi_evaluation-solutions.json"))
    ap.add_argument("--splits", default="dev,tune", help="comma-separated split names from splits/arc2_eval_split_v1.json (holdout refused)")
    ap.add_argument("--manifest", default=str(root / "splits" / "arc2_eval_split_v1.json"))
    ap.add_argument("--out", required=True, help="calibration JSON path (versioned by file name)")
    ap.add_argument("--version", default="")
    args = ap.parse_args(argv)
    if not args.neural_dir and not args.symbolic_json:
        raise SystemExit("need --neural-dir and/or --symbolic-json")

    from arc2.decoding.pool import (
        PoolStats,
        load_nvarc_outputs,
        load_symbolic_json,
        merge_pools,
        neural_to_candidate,
    )

    stats = PoolStats()
    pools: list[dict[str, list[Candidate]]] = []
    if args.symbolic_json:
        pools.append(load_symbolic_json(args.symbolic_json, stats))
    if args.neural_dir:
        neural = load_nvarc_outputs(args.neural_dir, stats)
        pools.append({bk: [neural_to_candidate(nc) for nc in cs] for bk, cs in neural.items()})
    pool = merge_pools(*pools)
    ids = _label_safe_ids([s.strip() for s in args.splits.split(",") if s.strip()], Path(args.manifest))
    with open(args.solutions, "r", encoding="utf-8") as fh:
        solutions = json.load(fh)
    records = build_records(pool, solutions, ids)
    if not records:
        raise SystemExit("no labelled records for the requested splits; nothing to fit")
    cal = fit_weights(records, version=args.version, notes=f"splits={args.splits} neural_dir={args.neural_dir} symbolic_json={args.symbolic_json} pool_stats={stats.as_dict()}")
    out = write_calibration(cal, args.out)
    print(json.dumps({"out": str(out), "params": cal.params, "pass_at_1": cal.pass_at_1, "pass_at_2": cal.pass_at_2, "n_records": cal.n_records, "default_params": cal.baseline}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
