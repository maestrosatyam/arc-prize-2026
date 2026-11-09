"""Unified candidate pool reader (A2-04).

Two sources feed the two-answer selector (arc2/selection):

(a) symbolic branch: ``Candidate`` lists produced by ``arc2.solvers.symbolic.SymbolicSolver`` (evidence keys per
    arc2/evidence/builder.py), or the same candidates serialised to ``symbolic_candidates.json`` by a CPU process
    on Kaggle (format defined by ``write_symbolic_json`` / ``load_symbolic_json`` below);

(b) the public baseline notebook's neural outputs. Structure read from the notebook code
    (mikelou1/arc-agi2-lb33-89-minimal-perfpatch, scriptVersionId 335137615, cells ``%%writefile arc_solver.py``
    and ``%%writefile arc_decoder.py``), verbatim facts:

    * ``arc_solver.py`` ``worker()`` writes, per augmented sub-key, the file
      ``/kaggle/inference_outputs/<subkey>`` where ``subkey = "<task_id>_<test_index>.<mod>.<mod>...ex<perm>"``
      (the ``ArcDataset`` key after ``split_multi_replies`` and ``augment``). The file is ``bz2.BZ2File`` +
      ``pickle.dump`` of a Python ``list`` of dicts, one per decoded beam:
          {"beam_score": float,          # DFS NLL of the beam (sum of -log p over generated tokens; lower is better)
           "score_aug":  list[float],    # calc_scores over 8 augmentations of (task, this grid); each an NLL
           "solution":   numpy.ndarray}  # the decoded grid mapped back to the original frame (invert_mod)
      Files are only written when ``len(decoded_result) > 0``.
    * ``arc_decoder.py`` ``ArcDecoder.load_decoded_results(store)``: ``base_key = filename.split(".")[0]``;
      every sample of every file is merged into ``decoded_results[base_key][f"{filename}.out{i}"]``.
    * ``score_sum`` groups samples by ``tuple(map(tuple, solution))`` (exact grid), scores each group with
      ``getter_kgmon(group) = len(group) - mean(mean(g["score_aug"]) for g in group)`` and sorts descending.
      ``sorted`` is stable, so ties keep ``os.listdir`` order (not deterministic). This module reproduces the score
      and breaks ties deterministically (lower best beam_score, then grid bytes); that is the only deviation.
    * ``ArcDataset.fill_submission`` splits the base key on ``"_"`` and writes ``g.tolist()`` of ``v[:2]`` into
      ``attempt_1``/``attempt_2``; keys with no file keep the ``[[0]]`` default from ``get_submission``.

Every grid entering a pool is validated with ``arc2.io.types.is_valid_grid``; invalid grids are rejected and
counted in ``PoolStats``. Pools are deduplicated by exact grid per base key. Nothing here reads labels.
"""
from __future__ import annotations

import bz2
import json
import math
import os
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc2.io.types import Candidate, Grid, is_valid_grid

__all__ = [
    "NeuralCandidate",
    "PoolStats",
    "base_key",
    "candidates_to_pool",
    "group_samples",
    "load_nvarc_outputs",
    "load_symbolic_json",
    "merge_pools",
    "neural_to_candidate",
    "pool_from_decoded_results",
    "split_base_key",
    "write_symbolic_json",
]

SYMBOLIC_JSON_SCHEMA = "arc2.symbolic_candidates.v1"


@dataclass(frozen=True)
class NeuralCandidate:
    """One distinct grid from the baseline's neural outputs for one base key (``<task>_<test_index>``)."""

    base_key: str
    grid: Grid
    beam_score: float  # best (lowest) DFS NLL among the hits
    score_aug: tuple[float, ...]  # per-hit mean augmentation NLL, one entry per hit (order = hit order)
    n_hits: int  # number of decoded beams (over all augmented sub-keys) that produced exactly this grid
    aug_mods: tuple[str, ...]  # the sub-key suffixes (file name after the base key) that produced it

    @property
    def mean_score_aug(self) -> float:
        return float(sum(self.score_aug) / len(self.score_aug))

    @property
    def score_kgmon(self) -> float:
        """Baseline ranking score: ``getter_kgmon`` = hit count minus mean augmentation NLL."""
        return float(self.n_hits - self.mean_score_aug)


@dataclass
class PoolStats:
    files: int = 0
    samples: int = 0
    rejected_invalid_grid: int = 0
    rejected_malformed: int = 0
    duplicates_merged: int = 0
    base_keys: int = 0
    distinct_grids: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "samples": self.samples,
            "rejected_invalid_grid": self.rejected_invalid_grid,
            "rejected_malformed": self.rejected_malformed,
            "duplicates_merged": self.duplicates_merged,
            "base_keys": self.base_keys,
            "distinct_grids": self.distinct_grids,
            "errors": list(self.errors),
        }


# ----------------------------------------------------------------------------------------------
# keys
# ----------------------------------------------------------------------------------------------


def base_key(task_id: str, test_index: int) -> str:
    return f"{task_id}_{int(test_index)}"


def split_base_key(key: str) -> tuple[str, int]:
    """Inverse of ``base_key`` (the baseline's ``fill_submission`` does ``k.split('_')``)."""
    tid, _, idx = key.rpartition("_")
    if not tid or not idx.isdigit():
        raise ValueError(f"not a base key: {key!r}")
    return tid, int(idx)


def _grid_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid)


def _to_grid(solution: Any) -> Grid | None:
    """numpy array or nested list -> Grid (list of lists of int), or None if not convertible."""
    if hasattr(solution, "tolist"):
        solution = solution.tolist()
    if not isinstance(solution, list):
        return None
    try:
        grid = [[int(v) for v in row] for row in solution]
    except (TypeError, ValueError):
        return None
    if not is_valid_grid(grid):
        return None
    # bool cells and non-integral floats (lossy int()) are rejected
    for r_in, r_out in zip(solution, grid):
        for a, b in zip(r_in, r_out):
            if isinstance(a, bool) or (isinstance(a, float) and a != b):
                return None
    return grid


def _finite_float(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ----------------------------------------------------------------------------------------------
# neural outputs (baseline pickle format)
# ----------------------------------------------------------------------------------------------


def group_samples(key: str, samples: Iterable[tuple[str, Mapping[str, Any]]], stats: PoolStats | None = None) -> list[NeuralCandidate]:
    """Group raw baseline samples ``(file_name, sample_dict)`` of one base key by exact grid.

    Returns the distinct grids ranked exactly like the baseline (``score_kgmon`` descending), ties broken
    deterministically by lower best ``beam_score`` then grid content. Malformed samples (missing keys, non-finite or
    empty scores) and invalid grids are dropped and counted.
    """
    st = stats if stats is not None else PoolStats()
    groups: dict[tuple, dict[str, Any]] = {}
    order: list[tuple] = []
    for name, sample in samples:
        st.samples += 1
        if not isinstance(sample, Mapping) or not {"beam_score", "score_aug", "solution"} <= set(sample):
            st.rejected_malformed += 1
            continue
        beam = _finite_float(sample["beam_score"])
        aug_raw = sample["score_aug"]
        if hasattr(aug_raw, "tolist"):
            aug_raw = aug_raw.tolist()
        aug = [_finite_float(a) for a in aug_raw] if isinstance(aug_raw, (list, tuple)) else []
        if beam is None or not aug or any(a is None for a in aug):
            st.rejected_malformed += 1
            continue
        grid = _to_grid(sample["solution"])
        if grid is None:
            st.rejected_invalid_grid += 1
            continue
        gk = _grid_key(grid)
        mod = name.split(".", 1)[1] if "." in name else ""
        if gk in groups:
            st.duplicates_merged += 1
            g = groups[gk]
            g["beam"] = min(g["beam"], beam)
            g["aug"].append(sum(aug) / len(aug))  # type: ignore[arg-type]
            g["mods"].append(mod)
        else:
            groups[gk] = {"grid": grid, "beam": beam, "aug": [sum(aug) / len(aug)], "mods": [mod]}  # type: ignore[arg-type]
            order.append(gk)
    out = [
        NeuralCandidate(
            base_key=key,
            grid=groups[gk]["grid"],
            beam_score=groups[gk]["beam"],
            score_aug=tuple(groups[gk]["aug"]),
            n_hits=len(groups[gk]["aug"]),
            aug_mods=tuple(groups[gk]["mods"]),
        )
        for gk in order
    ]
    out.sort(key=lambda c: (-c.score_kgmon, c.beam_score, _grid_key(c.grid)))
    return out


def load_nvarc_outputs(directory: str | os.PathLike[str], stats: PoolStats | None = None) -> dict[str, list[NeuralCandidate]]:
    """Read every file of the baseline's ``/kaggle/inference_outputs`` directory.

    ``{base_key: [NeuralCandidate, ...]}`` ranked best first per base key. A file that fails to decode is skipped
    and recorded in ``stats.errors`` (the baseline would crash; a selector must not). Files are read in sorted name
    order so the result is deterministic.
    """
    st = stats if stats is not None else PoolStats()
    d = Path(directory)
    per_key: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    if not d.is_dir():
        st.errors.append(f"not a directory: {d}")
        return {}
    for name in sorted(os.listdir(d)):
        p = d / name
        if not p.is_file():
            continue
        st.files += 1
        try:
            with bz2.BZ2File(p) as fh:
                outputs = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - corrupt file must not kill the selection step
            st.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(outputs, list):
            st.errors.append(f"{name}: pickle is {type(outputs).__name__}, expected list")
            continue
        bk = name.split(".")[0]
        per_key.setdefault(bk, []).extend((name, s) for s in outputs)
    pool = {bk: group_samples(bk, samples, st) for bk, samples in per_key.items()}
    pool = {bk: cs for bk, cs in pool.items() if cs}
    st.base_keys = len(pool)
    st.distinct_grids = sum(len(v) for v in pool.values())
    return pool


def pool_from_decoded_results(decoded_results: Mapping[str, Mapping[str, Mapping[str, Any]]], stats: PoolStats | None = None) -> dict[str, list[NeuralCandidate]]:
    """Same as ``load_nvarc_outputs`` but from an in-memory ``ArcDecoder.decoded_results``
    (``{base_key: {"<file>.out<i>": sample}}``)."""
    st = stats if stats is not None else PoolStats()
    pool: dict[str, list[NeuralCandidate]] = {}
    for bk, samples in decoded_results.items():
        items = [(k.rsplit(".out", 1)[0], s) for k, s in samples.items()]
        cs = group_samples(bk, items, st)
        if cs:
            pool[bk] = cs
    st.base_keys = len(pool)
    st.distinct_grids = sum(len(v) for v in pool.values())
    return pool


def neural_to_candidate(nc: NeuralCandidate) -> Candidate:
    """Contract ``Candidate`` with evidence ``model_score`` (= score_kgmon), ``aug_support`` (= n_hits) and
    ``demo_consistent: None`` (no executable program; the neural branch cannot certify demos)."""
    tid, ti = split_base_key(nc.base_key)
    return Candidate(
        task_id=tid,
        test_index=ti,
        grid=[list(r) for r in nc.grid],
        source="neural",
        evidence={
            "demo_consistent": None,
            "program": None,
            "program_len": None,
            "loo_consistent": None,
            "aug_support": nc.n_hits,
            "model_score": nc.score_kgmon,
            "beam_score": nc.beam_score,
            "mean_score_aug": nc.mean_score_aug,
            "n_hits": nc.n_hits,
        },
        cost_s=0.0,
        lineage={"branch": "neural", "baseline": "nvarc-mikelou1-v335137615", "aug_mods": list(nc.aug_mods)},
    )


# ----------------------------------------------------------------------------------------------
# Candidate lists (symbolic branch) and the Kaggle symbolic JSON
# ----------------------------------------------------------------------------------------------


def candidates_to_pool(cands: Iterable[Candidate], stats: PoolStats | None = None) -> dict[str, list[Candidate]]:
    """``{base_key: [Candidate, ...]}`` in input order, invalid grids rejected, exact duplicates (per base key)
    dropped (the first occurrence is kept; inputs are never mutated)."""
    st = stats if stats is not None else PoolStats()
    pool: dict[str, list[Candidate]] = {}
    seen: dict[str, dict[tuple, int]] = {}
    for c in cands:
        st.samples += 1
        if not is_valid_grid(c.grid):
            st.rejected_invalid_grid += 1
            continue
        bk = base_key(c.task_id, c.test_index)
        gk = _grid_key(c.grid)
        idx = seen.setdefault(bk, {})
        if gk in idx:
            st.duplicates_merged += 1
            continue
        idx[gk] = len(pool.setdefault(bk, []))
        pool[bk].append(c)
    st.base_keys = len(pool)
    st.distinct_grids = sum(len(v) for v in pool.values())
    return pool


def merge_pools(*pools: Mapping[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    """Concatenate pools per base key, keeping the first occurrence of each exact grid (earlier pools win)."""
    merged: dict[str, list[Candidate]] = {}
    for pool in pools:
        for bk, cs in pool.items():
            have = {_grid_key(c.grid) for c in merged.get(bk, [])}
            for c in cs:
                gk = _grid_key(c.grid)
                if gk in have:
                    continue
                have.add(gk)
                merged.setdefault(bk, []).append(c)
    return merged


_SYMBOLIC_EVIDENCE_KEYS = ("demo_consistent", "program", "program_len", "program_cost", "n_programs", "loo_consistent", "shape_rule", "shape_plausible")


def write_symbolic_json(cands: Iterable[Candidate], path: str | os.PathLike[str]) -> dict[str, Any]:
    """Serialise symbolic candidates for the Kaggle final cell: ``{"schema": ..., "candidates": {base_key: [...]}}``.
    Each entry: ``{"grid", "source", "cost_s", "evidence": {<_SYMBOLIC_EVIDENCE_KEYS>}}``. Returns the object."""
    out: dict[str, list[dict[str, Any]]] = {}
    for bk, cs in candidates_to_pool(cands).items():
        out[bk] = [
            {
                "grid": [list(r) for r in c.grid],
                "source": c.source,
                "cost_s": float(c.cost_s),
                "evidence": {k: c.evidence.get(k) for k in _SYMBOLIC_EVIDENCE_KEYS},
            }
            for c in cs
        ]
    obj = {"schema": SYMBOLIC_JSON_SCHEMA, "candidates": out}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, p)
    return obj


def load_symbolic_json(path: str | os.PathLike[str], stats: PoolStats | None = None) -> dict[str, list[Candidate]]:
    """Read ``write_symbolic_json`` output (missing file -> empty pool, recorded in ``stats.errors``).
    ``demo_consistent`` is kept as written; it is never promoted to True here."""
    st = stats if stats is not None else PoolStats()
    p = Path(path)
    if not p.is_file():
        st.errors.append(f"symbolic json missing: {p}")
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError) as exc:
        st.errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(obj, dict) or obj.get("schema") != SYMBOLIC_JSON_SCHEMA or not isinstance(obj.get("candidates"), dict):
        st.errors.append(f"{p.name}: unexpected schema {obj.get('schema') if isinstance(obj, dict) else type(obj).__name__}")
        return {}
    st.files += 1
    cands: list[Candidate] = []
    for bk, entries in obj["candidates"].items():
        try:
            tid, ti = split_base_key(bk)
        except ValueError:
            st.rejected_malformed += 1
            continue
        if not isinstance(entries, list):
            st.rejected_malformed += 1
            continue
        for e in entries:
            if not isinstance(e, dict) or "grid" not in e:
                st.rejected_malformed += 1
                continue
            ev = e.get("evidence") if isinstance(e.get("evidence"), dict) else {}
            evidence = {k: ev.get(k) for k in _SYMBOLIC_EVIDENCE_KEYS}
            evidence["demo_consistent"] = evidence["demo_consistent"] is True
            evidence.update({"aug_support": None, "model_score": None})
            cands.append(
                Candidate(
                    task_id=tid,
                    test_index=ti,
                    grid=e["grid"],
                    source=str(e.get("source", "symbolic")),
                    evidence=evidence,
                    cost_s=float(e.get("cost_s", 0.0) or 0.0),
                    lineage={"branch": "symbolic", "file": p.name},
                )
            )
    return candidates_to_pool(cands, st)

