"""ARC-AGI-2 output bank: loaders, replay rules, scoring and paired comparison. CPU only, no GPU, no network.

The bank is B0's candidate pool on the public evaluation tasks (kaggle/arc2/bank, parts 1 and 2: every decoded grid
with its scores, beams and discovery time) plus the 27B direct samples on the shortest quarter (the q27 bank). A
*rule* turns a task's pool (and its 27B samples) into two attempts per test output; replaying a rule on the bank
scores it on the same outputs every time, with no run noise.

    from eval.arc2_bank import load_bank, RULES, replay, compare
    bank = load_bank()                      # pool/ of both parts + the 27B bank; the sealed 19 tasks are refused
    base = replay(bank, RULES["b0"])
    new = replay(bank, RULES["v7_direct"])
    compare(base, new)                      # gained / lost / net, sign test, task-paired cluster bootstrap, gates

Sealed tasks: a directory named ``pool_sealed`` or ``q27d_sealed`` (or one carrying a ``_sealed.json`` manifest) is
read only with ``final_pick=True``; every sealed file is checked against the manifest's sha256
(kaggle/arc2/bank/README.md; C2_RULE.md section 4, item 7 for the 27B sealed block).

Several 27B directories (C2, experiments/arc2-bank/C2_RULE.md: the bank's samples 0 and 1 on the band beside the C2
kernels' samples 2 and 3 there and 0 and 1 beyond) are merged by task and sample index:

    bank = load_bank(q27_dir=[Q27_BANK, c2a_q27d, c2b_q27d])     # or q27_dirs=[...]

A (task, index) present in two directories is an error; the merge rules take a task's samples in ascending index
order (the rule's "sample order 0..3"), and nothing assumes the indices are 0 and 1.

Rules are plain functions ``rule(task, bank, rng) -> [(attempt_1, attempt_2) per test output]`` registered in
``RULES``; the null families (``null_pair:<seed>`` etc.) are built by ``get_rule``. Scores use the official formula
(eval/arc2_score.py: outputs right in either attempt over all outputs); the task-level view (mean over tasks of
the fraction of outputs right) is reported beside it.

GS1 (experiments/arc2-glimmer-scorer/GS1_RULE.md, with its Revision 1): a second model's scores of the pool
candidates, read from a JSON file (``load_bank(glimmer=...)``, ``load_glimmer_scores``), schema ``arc2_gs1_scores.v1``:

    {"schema": "arc2_gs1_scores.v1", "split": "unsealed" | "sealed", "sealed": false | true,
     "views": ["identity"],
     "tasks": {"<task_id>": [{"test_index": 0, "candidates": [
         {"grid": [[...]], "sources": ["b0:<rank_kgmon>", "q27:<sample>"],
          "views": {"<view>": {"mean": <mean log-likelihood per answer token>, "sum": ..., "n_answer_tokens": ...,
                               "prompt_tokens": ..., "seconds": ..., "roundtrip_ok": true}, ...},
          "mean": <the average over the file's views of their means>}, ...]}, ...]}}

One entry per distinct candidate grid of a scored test output (B0's pool and the 27B's grids); natural-log
likelihoods, higher is better. Revision 1 (2026-10-28 00:43 IST): one view, identity (GS1_VIEWS), and only
the low-margin outputs are scored. The Glimmer score of a candidate is the mean over GS1_VIEWS of the views' means,
i.e. the identity view's mean (recomputed here; a candidate with the view missing or not finite has no score). A file
whose "views" differ from GS1_VIEWS is refused; a sealed file is refused without ``final_pick``.

B0R (experiments/arc2-bank/B0R_RULE.md): three rankings of B0's own candidates from the pool's signals (n_beams,
score_aug, beam_score, the two built-in ranks), each keeping kgmon's first and choosing attempt 2 (b0r_zbeam,
b0r_amax, b0r_borda, and their _v7 forms inside v7's merge). They are in RULES but not in the replay runner's default
list; b0r_check is the rule's field check (floor item 1).
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from eval.arc2_score import grids_equal, is_valid_grid, score_submission

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
BANK_PARTS = (RUNS / "kaggle-A2-bank-part1-v1" / "pool", RUNS / "kaggle-A2-bank-part2-v1" / "pool")
BANK_SEALED = (RUNS / "kaggle-A2-bank-part1-v1" / "pool_sealed", RUNS / "kaggle-A2-bank-part2-v1" / "pool_sealed")
Q27_BANK = RUNS / "kaggle-A2-q27-bank-v1" / "q27d"
CHALLENGES = ROOT / "eval" / "data" / "arc-agi_evaluation-challenges.json"
LOCAL_SOLUTIONS = ROOT / "eval" / "data" / "arc-agi_evaluation-solutions.json"
KAGGLE_SOLUTIONS_COPY = "_kaggle_solutions_audit_tasks.json"
SEALED_DIRNAME = "pool_sealed"
Q27_SEALED_DIRNAME = "q27d_sealed"   # C2: the 27B samples of the sealed 19 tasks, with a _sealed.json manifest
SEALED_DIRNAMES = (SEALED_DIRNAME, Q27_SEALED_DIRNAME)
SEALED_MANIFEST = "_sealed.json"
PLACEHOLDER = [[0]]
BAND_FRACTION = 0.25          # the hybrid's band: the shortest ceil(0.25 n) tasks by task JSON length
# the 7 evaluation tasks that overlap ARC-AGI-1 (splits/holdout_policy.md, "Known contamination")
ARC1_OVERLAP = ("0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329", "db695cfb")
SCHEMA = "arc2_bank_replay.v1"
# GS1: Glimmer's scores of the pool candidates (the scoring kernel's outputs, archived by the queue)
GS1_SCHEMA = "arc2_gs1_scores.v1"
GS1_VIEWS = ("identity",)     # GS1 revision 1 (2026-10-28): one view; the original text had four
GS1_RUN = RUNS / "kaggle-A2-GS1-v1" / "gs1"
GS1_SCORES = GS1_RUN / "gs1_scores_unsealed.json"
GS1_SCORES_SEALED = GS1_RUN / "sealed" / "gs1_scores_sealed.json"
# DC1: an open decision model's choice probabilities over B0's pool (experiments/arc2-decider/DC1_RULE.md)
DC1_SCHEMA = "arc2_dc1_probs.v1"
DC1_RUN = RUNS / "kaggle-A2-DC1-v1" / "dc1"
DC1_PROBS = DC1_RUN / "dc1_probs_unsealed.json"
DC1_PROBS_SEALED = DC1_RUN / "sealed" / "dc1_probs_sealed.json"


class SealedAccessError(RuntimeError):
    """A sealed pool was opened without the final-pick flag."""


# ----------------------------------------------------------------------------
# data

@dataclass
class Output:
    task_id: str
    test_index: int
    candidates: list[dict]            # B0's candidates: grid, rank_kgmon, score_kgmon, rank_probmul3, ... (pool order)
    attempt_1: Any                    # as B0 submitted (placeholder when B0 had nothing)
    attempt_2: Any
    margin_kgmon: float | None = None
    n_unique: int = 0
    n_beams: int = 0
    has_pool: bool = False

    @property
    def key(self) -> str:
        return f"{self.task_id}_{self.test_index}"

    def ranked(self, key: str = "rank_kgmon") -> list[dict]:
        """Candidates with a rank under ``key``, best first (ties by pool order)."""
        with_rank = [(c.get(key), i, c) for i, c in enumerate(self.candidates) if c.get(key) is not None]
        return [c for _, _, c in sorted(with_rank, key=lambda t: (t[0], t[1]))]


@dataclass
class Task:
    task_id: str
    outputs: list[Output]
    run: dict
    size: int | None = None           # characters of the task JSON (the hybrid's route key); None without challenges
    sealed: bool = False
    file_sha256: str = ""

    @property
    def elapsed_s(self) -> float | None:
        v = self.run.get("elapsed_s")
        return float(v) if v is not None else None

    @property
    def stop_reason(self) -> str | None:
        return self.run.get("stop_reason") or self.run.get("loop_stop")


@dataclass
class Bank:
    tasks: dict[str, Task]
    solutions: dict[str, list]
    q27: dict[str, dict[int, dict]] = field(default_factory=dict)   # task -> sample -> {grids, status, finish, ...}
    sources: list[str] = field(default_factory=list)
    solutions_source: str = ""
    includes_sealed: bool = False
    sizes_all: dict[str, int] = field(default_factory=dict)   # task JSON length of every task in the challenges file
    glimmer: dict[str, dict[int, dict[str, dict]]] = field(default_factory=dict)   # GS1: task -> output -> grid key -> record
    glimmer_sources: list[str] = field(default_factory=list)
    decider: dict[str, dict[int, dict]] = field(default_factory=dict)   # DC1: task -> output -> {fit_4096, probs}
    decider_sources: list[str] = field(default_factory=list)

    @property
    def task_ids(self) -> list[str]:
        return sorted(self.tasks)

    def outputs(self) -> Iterable[Output]:
        for t in self.task_ids:
            yield from self.tasks[t].outputs

    @property
    def n_outputs(self) -> int:
        return sum(len(t.outputs) for t in self.tasks.values())

    def route_order(self) -> list[str]:
        """This bank's task ids shortest task JSON first, ties by id (hybrid_core.route_order); no size: last."""
        return sorted(self.tasks, key=lambda t: (self.tasks[t].size is None, self.tasks[t].size or 0, t))

    def route_order_all(self) -> list[str]:
        """The route over the whole challenges file (120 public tasks), the hybrid's own order; falls back to the
        bank's tasks when no challenges were loaded."""
        if not self.sizes_all:
            return self.route_order()
        return sorted(self.sizes_all, key=lambda t: (self.sizes_all[t], t))

    def band(self, fraction: float = BAND_FRACTION) -> list[str]:
        """The hybrid's band: the shortest ceil(fraction x n) tasks of the whole route (30 of 120) that are in this
        bank. The 27B bank sampled exactly these tasks."""
        order = self.route_order_all()
        n = len(order)
        nb = min(n, int(math.ceil(round(fraction * n, 9)))) if n else 0
        return [t for t in order[:nb] if t in self.tasks]

    def fingerprint(self) -> str:
        """sha256 over the task files and the 27B samples: two replays on the same bank share it."""
        h = hashlib.sha256()
        for t in self.task_ids:
            h.update(f"{t}:{self.tasks[t].file_sha256}\n".encode())
        for t in sorted(self.q27):
            for k in sorted(self.q27[t]):
                h.update(f"q27:{t}:{k}:{json.dumps(self.q27[t][k].get('grids'), sort_keys=True)}\n".encode())
        return h.hexdigest()[:16]

    def truth(self, out: Output):
        sol = self.solutions.get(out.task_id)
        return sol[out.test_index] if sol is not None and out.test_index < len(sol) else None


def task_size(task: dict) -> int:
    """The hybrid's route key: characters of the task's JSON."""
    return len(json.dumps(task))


def load_challenges(path: Path | str = CHALLENGES) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_of(d: Path) -> dict | None:
    m = d / SEALED_MANIFEST
    return json.loads(m.read_text(encoding="utf-8")) if m.is_file() else None


def is_sealed_dir(d: Path) -> bool:
    """A sealed directory: named pool_sealed or q27d_sealed, or carrying a _sealed.json manifest."""
    d = Path(d)
    return d.name in SEALED_DIRNAMES or (d / SEALED_MANIFEST).is_file()


def load_pools(dirs: Iterable[Path | str], *, final_pick: bool = False,
               tasks: Iterable[str] | None = None) -> tuple[dict[str, Task], dict[str, list], list[str]]:
    """(tasks by id, Kaggle solutions found beside the pools, source dirs). A sealed directory is refused without
    ``final_pick``; its files must match the manifest's sha256."""
    keep = set(tasks) if tasks is not None else None
    out: dict[str, Task] = {}
    sols: dict[str, list] = {}
    sources = []
    for d in dirs:
        d = Path(d)
        sealed = is_sealed_dir(d)
        if sealed and not final_pick:
            raise SealedAccessError(f"{d} is a sealed pool: replays read pool/ only; pass final_pick=True for the one "
                                    f"final-pick check (kaggle/arc2/bank/README.md)")
        manifest = _manifest_of(d) if sealed else None
        sources.append(str(d))
        kag = d / KAGGLE_SOLUTIONS_COPY
        if kag.is_file():
            sols.update(json.loads(kag.read_text(encoding="utf-8")))
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            raw = p.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            doc = json.loads(raw.decode("utf-8"))
            tid = doc.get("task_id", p.stem)
            if keep is not None and tid not in keep:
                continue
            if manifest is not None:
                want = (manifest.get("sha256") or {}).get(tid)
                if want and want != sha:
                    raise SealedAccessError(f"{p} does not match the sealed manifest ({SEALED_MANIFEST})")
            outs = []
            for o in doc.get("outputs") or []:
                sub = o.get("submitted") if isinstance(o.get("submitted"), dict) else {}
                sel = list(o.get("selected") or []) + [PLACEHOLDER, PLACEHOLDER]
                a1 = sub.get("attempt_1", sel[0]) if sub else sel[0]
                a2 = sub.get("attempt_2", sel[1]) if sub else sel[1]
                cands = [c for c in (o.get("candidates") or []) if is_valid_grid(c.get("grid"))]
                outs.append(Output(tid, int(o.get("test_index", len(outs))), cands, a1, a2,
                                   o.get("margin_kgmon"), int(o.get("n_unique") or len(cands)),
                                   int(o.get("n_beams") or 0), bool(cands)))
            run = dict(doc.get("run") or {})
            out[tid] = Task(tid, outs, run, None, sealed, sha)
    return out, sols, sources


def _read_q27(d: Path, *, final_pick: bool = False
              ) -> tuple[dict[str, dict[int, dict]], dict[str, list], set[tuple[str, int]]]:
    """One 27B directory: (samples with grids by task and sample index, the Kaggle solutions copy beside it, every
    (task, index) that has a record, grids or not). A sealed directory (``q27d_sealed/``, or any carrying a
    ``_sealed.json`` manifest) is refused without ``final_pick``; with it every per-task file must match the
    manifest's sha256."""
    d = Path(d)
    if not d.is_dir():
        raise FileNotFoundError(f"27B directory {d} not found")
    sealed = is_sealed_dir(d)
    if sealed and not final_pick:
        raise SealedAccessError(f"{d} is a sealed 27B directory: replays read the unsealed samples only; pass "
                                f"final_pick=True for the one final-pick check (C2_RULE.md section 4, item 7)")
    manifest = _manifest_of(d) if sealed else None
    q27: dict[str, dict[int, dict]] = {}
    seen: set[tuple[str, int]] = set()

    def take(s: dict) -> None:
        tid = s.get("task_id")
        if not tid:
            return
        k = int(s.get("sample") or 0)
        seen.add((tid, k))
        grids = s.get("grids")
        if k in q27.get(tid, {}) or not grids:
            return
        q27.setdefault(tid, {})[k] = {"grids": grids, "status": s.get("status"), "finish": s.get("finish"),
                                      "gen_tokens": s.get("gen_tokens"), "wall_s": s.get("wall_s")}

    records: list[dict] = []
    for rp in sorted(d.glob("*.json")):
        if rp.name.startswith("_"):
            continue
        raw = rp.read_bytes()
        if manifest is not None:
            shas = manifest.get("sha256") or {}
            want = shas.get(rp.stem) or shas.get(rp.name)   # B0's pool_sealed keys by task id, C2's q27d_sealed by file
            if want and want != hashlib.sha256(raw).hexdigest():
                raise SealedAccessError(f"{rp} does not match the sealed manifest ({SEALED_MANIFEST})")
        try:
            records.append(json.loads(raw.decode("utf-8")))
        except ValueError:
            continue
    p = d / "samples.jsonl"
    if p.is_file():
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                take(s)
    else:
        for rec in records:
            for s in rec.get("samples") or []:
                if isinstance(s, dict):
                    take(dict(s, task_id=s.get("task_id") or rec.get("task_id")))
    kag = d / KAGGLE_SOLUTIONS_COPY
    sols = json.loads(kag.read_text(encoding="utf-8")) if kag.is_file() else {}
    return q27, sols, seen


def load_q27(d: Path | str, *, final_pick: bool = False) -> tuple[dict[str, dict[int, dict]], dict[str, list]]:
    """The 27B bank: {task: {sample: {grids, status, finish, gen_tokens, wall_s}}} from samples.jsonl (per task and
    sample the first record with grids; a line cut mid-write is skipped), else from the ``samples`` lists of the
    per-task records (``*.json``; samples.jsonl is not tracked in git), and the Kaggle solutions copy beside it.
    A sealed 27B directory is refused without ``final_pick`` (see ``_read_q27``)."""
    q27, sols, _ = _read_q27(Path(d), final_pick=final_pick)
    return q27, sols


def load_q27_dirs(dirs: Iterable[Path | str], *, final_pick: bool = False
                  ) -> tuple[dict[str, dict[int, dict]], dict[str, list], list[str]]:
    """Several 27B directories merged by task and sample index (C2: the bank's samples 0 and 1 beside the C2
    kernels' 2 and 3 on the band and 0 and 1 beyond). A (task, index) with a record in two directories raises
    ValueError: a sample is banked once. Returns (samples, solutions, source directories)."""
    q27: dict[str, dict[int, dict]] = {}
    sols: dict[str, list] = {}
    sources: list[str] = []
    owner: dict[tuple[str, int], str] = {}
    for d in dirs:
        d = Path(d)
        got, s, seen = _read_q27(d, final_pick=final_pick)
        for key in sorted(seen):
            if key in owner:
                raise ValueError(f"27B sample {key[0]} index {key[1]} is in both {owner[key]} and {d}: a sample is "
                                 f"banked once (C2 samples carry indices 2 and 3 on the band, 0 and 1 beyond)")
            owner[key] = str(d)
        for t, per in got.items():
            q27.setdefault(t, {}).update(per)
        for t, v in s.items():
            sols.setdefault(t, v)
        sources.append(str(d))
    return q27, sols, sources


def grid_key(grid) -> str:
    """A grid's identity for score lookups (compact JSON)."""
    return json.dumps(grid, separators=(",", ":"))


def load_glimmer_scores(path: Path | str, *, final_pick: bool = False) -> dict[str, dict[int, dict[str, dict]]]:
    """GS1 scores file -> {task: {test_index: {grid_key: candidate record}}}. A file marked sealed is refused without
    ``final_pick`` (the one sealed read of the final pick)."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != GS1_SCHEMA:
        raise ValueError(f"{p}: schema {doc.get('schema')!r}, want {GS1_SCHEMA!r}")
    if list(doc.get("views") or []) != list(GS1_VIEWS):
        raise ValueError(f"{p}: views {doc.get('views')}, want {list(GS1_VIEWS)} (GS1 revision 1)")
    if (doc.get("sealed") or doc.get("split") == "sealed") and not final_pick:
        raise SealedAccessError(f"{p} holds scores of the sealed tasks: read only with final_pick=True "
                                f"(--sealed --final-pick, once)")
    out: dict[str, dict[int, dict[str, dict]]] = {}
    for tid, outs in (doc.get("tasks") or {}).items():
        for o in outs or []:
            per = out.setdefault(tid, {}).setdefault(int(o["test_index"]), {})
            for c in o.get("candidates") or []:
                per[grid_key(c["grid"])] = c
    return out


def glimmer_score(rec: dict | None) -> float | None:
    """The GS1 score of one candidate record: the mean over GS1_VIEWS (revision 1: identity only) of the mean
    log-likelihood per answer token; None when a view is missing or not finite."""
    if not rec:
        return None
    views = rec.get("views") or {}
    vals = []
    for v in GS1_VIEWS:
        m = (views.get(v) or {}).get("mean")
        if m is None:
            return None
        m = float(m)
        if not math.isfinite(m):
            return None
        vals.append(m)
    return sum(vals) / len(vals)


def load_decider_probs(path: Path | str, *, final_pick: bool = False) -> dict[str, dict[int, dict]]:
    """DC1 probabilities file -> {task: {test_index: {"fit_4096": bool, "probs": {grid_key: mean probability or
    None}}}}. A file marked sealed is refused without ``final_pick`` (the one sealed read of the final pick)."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != DC1_SCHEMA:
        raise ValueError(f"{p}: schema {doc.get('schema')!r}, want {DC1_SCHEMA!r}")
    if doc.get("twin"):
        raise ValueError(f"{p}: a twin's file (three outputs), not the run's")
    if (doc.get("sealed") or doc.get("split") == "sealed") and not final_pick:
        raise SealedAccessError(f"{p} holds probabilities of the sealed tasks: read only with final_pick=True "
                                f"(--sealed --final-pick, once)")
    out: dict[str, dict[int, dict]] = {}
    for tid, outs in (doc.get("tasks") or {}).items():
        for o in outs or []:
            out.setdefault(tid, {})[int(o["test_index"])] = {
                "fit_4096": bool(o.get("fit_4096")), "tier": o.get("tier"),
                "probs": {grid_key(c["grid"]): c.get("prob_mean") for c in o.get("candidates") or []}}
    return out


def load_bank(pool_dirs: Iterable[Path | str] | None = None,
              q27_dir: Path | str | Iterable[Path | str] | None = Q27_BANK, *,
              challenges: Path | str | dict | None = CHALLENGES, solutions: Path | str | dict | None = None,
              final_pick: bool = False, tasks: Iterable[str] | None = None,
              glimmer: Path | str | dict | Iterable[Path | str | dict] | None = None,
              decider: Path | str | dict | Iterable[Path | str | dict] | None = None,
              q27_dirs: Iterable[Path | str] | None = None) -> Bank:
    """The bank: the pool directories (default: both parts' pool/) and the 27B directories. ``q27_dir``: one 27B
    directory or a list of them (``q27_dirs`` is the same as a keyword and takes precedence), merged by task and
    sample index (``load_q27_dirs``); a directory that does not exist is skipped, a sealed one is refused without
    ``final_pick``, a (task, index) in two of them is an error. Solutions: the Kaggle copies beside the pools and
    the 27B directories, then ``solutions`` (path or dict), then eval/data's local file. ``glimmer``: GS1 scores (a
    path, a loaded dict, or a list of them; a sealed scores file needs ``final_pick``). ``decider``: DC1
    probabilities, the same way."""
    dirs = list(pool_dirs) if pool_dirs is not None else list(BANK_PARTS)
    tasks_by_id, sols, sources = load_pools(dirs, final_pick=final_pick, tasks=tasks)
    if q27_dirs is not None:
        q_list = list(q27_dirs)
    elif q27_dir is None or isinstance(q27_dir, (str, Path)):
        q_list = [q27_dir] if q27_dir else []
    else:
        q_list = list(q27_dir)
    q_list = [Path(d) for d in q_list if d and Path(d).is_dir()]
    q27: dict[str, dict[int, dict]] = {}
    if q_list:
        q27, q_sols, q_sources = load_q27_dirs(q_list, final_pick=final_pick)
        for t, s in q_sols.items():
            sols.setdefault(t, s)
        sources.extend(q_sources)
        q27 = {t: v for t, v in q27.items() if tasks is None or t in set(tasks)}
    src = "Kaggle solutions copies beside the pools" if sols else ""
    if solutions is not None:
        extra = solutions if isinstance(solutions, dict) else json.loads(Path(solutions).read_text(encoding="utf-8"))
        for t, s in extra.items():
            sols.setdefault(t, s)
        src = (src + "; " if src else "") + ("given solutions" if isinstance(solutions, dict) else str(solutions))
    missing = [t for t in tasks_by_id if t not in sols]
    if missing and LOCAL_SOLUTIONS.is_file():
        local = json.loads(LOCAL_SOLUTIONS.read_text(encoding="utf-8"))
        for t in missing:
            if t in local:
                sols[t] = local[t]
        src = (src + "; " if src else "") + f"{LOCAL_SOLUTIONS.name} for {len(missing)} tasks"
    sizes_all = {}
    if challenges is not None:
        ch = challenges if isinstance(challenges, dict) else (load_challenges(challenges) if Path(challenges).is_file() else {})
        sizes_all = {t: task_size(task) for t, task in ch.items()}
        for t, task in tasks_by_id.items():
            if t in sizes_all:
                task.size = sizes_all[t]
    gl: dict[str, dict[int, dict[str, dict]]] = {}
    gl_sources: list[str] = []
    if glimmer is not None:
        items = list(glimmer) if isinstance(glimmer, (list, tuple)) else [glimmer]
        for it in items:
            got = it if isinstance(it, dict) else load_glimmer_scores(it, final_pick=final_pick)
            gl_sources.append("given scores" if isinstance(it, dict) else str(it))
            for t, outs in got.items():
                if t not in tasks_by_id:
                    continue
                for j, per in outs.items():
                    gl.setdefault(t, {}).setdefault(int(j), {}).update(per)
    dc: dict[str, dict[int, dict]] = {}
    dc_sources: list[str] = []
    if decider is not None:
        items = list(decider) if isinstance(decider, (list, tuple)) else [decider]
        for it in items:
            got = it if isinstance(it, dict) else load_decider_probs(it, final_pick=final_pick)
            dc_sources.append("given probabilities" if isinstance(it, dict) else str(it))
            for t, outs in got.items():
                if t in tasks_by_id:
                    for j, rec in outs.items():
                        dc.setdefault(t, {})[int(j)] = rec
    return Bank(tasks_by_id, sols, q27, sources, src, any(t.sealed for t in tasks_by_id.values()), sizes_all, gl,
                gl_sources, decider=dc, decider_sources=dc_sources)


# ----------------------------------------------------------------------------
# rules

Attempts = list[tuple[Any, Any]]
RuleFn = Callable[[Task, Bank, random.Random], Attempts]


@dataclass
class Rule:
    name: str
    fn: RuleFn
    family: str                      # "b0", "merge", "null", "oracle"
    description: str
    uses_labels: bool = False        # an oracle reads the solutions: a bound, never a candidate rule

    def fingerprint(self) -> str:
        try:
            src = inspect.getsource(self.fn)
        except (OSError, TypeError):
            src = repr(self.fn)
        return hashlib.sha256(f"{self.name}\n{src}".encode()).hexdigest()[:16]


def _pair(a1, a2) -> tuple[Any, Any]:
    return (a1 if is_valid_grid(a1) else PLACEHOLDER, a2 if is_valid_grid(a2) else PLACEHOLDER)


def _top2(cands: list[dict]) -> tuple[Any, Any]:
    grids = [c["grid"] for c in cands]
    return _pair(grids[0] if grids else PLACEHOLDER, grids[1] if len(grids) > 1 else PLACEHOLDER)


def rule_b0(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0 as submitted: its kgmon top two (the entry's phase 1 and the base of every comparison)."""
    return [_pair(o.attempt_1, o.attempt_2) for o in task.outputs]


def rule_b0_kgmon(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """The kgmon top two recomputed from the pool (equals b0 when the pool export is consistent)."""
    return [_top2(o.ranked("rank_kgmon")) if o.has_pool else _pair(o.attempt_1, o.attempt_2) for o in task.outputs]


def rule_b0_probmul3(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0's alternative ranking: the probmul3 top two."""
    return [_top2(o.ranked("rank_probmul3")) if o.has_pool else _pair(o.attempt_1, o.attempt_2) for o in task.outputs]


def _q27_grids(bank: Bank, task: Task, j: int, order: list[int] | None = None) -> list[tuple[int, Any]]:
    """[(sample, grid)] valid 27B grids for output j in sample order (or ``order``)."""
    per = bank.q27.get(task.task_id) or {}
    keys = order if order is not None else sorted(per)
    out = []
    for k in keys:
        g = per.get(k, {}).get("grids")
        if isinstance(g, list) and j < len(g) and is_valid_grid(g[j]):
            out.append((k, g[j]))
    return out


def merge_v7(task: Task, bank: Bank, order: list[int] | None = None) -> Attempts:
    """hybrid_core.merge: attempt_1 is never replaced; attempt_2 = the first valid 27B grid in sample order that
    differs from attempt_1; nothing changes when no sample differs."""
    out = []
    for j, o in enumerate(task.outputs):
        a1, a2 = _pair(o.attempt_1, o.attempt_2)
        use = next((g for _, g in _q27_grids(bank, task, j, order) if g != a1), None)
        out.append((a1, use if use is not None else a2))
    return out


def rule_v7_direct(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """Hybrid v7's merge on the bank: B0's attempts, attempt_2 replaced by the first 27B sample that differs from
    attempt_1 (tasks with 27B samples: the shortest quarter)."""
    return merge_v7(task, bank)


def rule_q27_first(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """27B first: attempt_1 = the first valid 27B grid, attempt_2 = B0's attempt_1 (B0's attempt_2 when equal);
    outputs without a 27B grid keep B0."""
    out = []
    for j, o in enumerate(task.outputs):
        a1, a2 = _pair(o.attempt_1, o.attempt_2)
        grids = [g for _, g in _q27_grids(bank, task, j)]
        if not grids:
            out.append((a1, a2))
            continue
        q = grids[0]
        out.append((q, a1 if a1 != q else a2))
    return out


def rule_q27_agree(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """Two-sample agreement (E15): when the 27B's samples agree on an output their grid goes first and B0's
    attempt_1 second; otherwise v7's merge."""
    base = merge_v7(task, bank)
    out = []
    for j, o in enumerate(task.outputs):
        grids = [g for _, g in _q27_grids(bank, task, j)]
        if len(grids) >= 2 and all(g == grids[0] for g in grids[1:]):
            a1, a2 = _pair(o.attempt_1, o.attempt_2)
            out.append((grids[0], a1 if a1 != grids[0] else a2))
        else:
            out.append(base[j])
    return out


def rule_oracle(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """Coverage bound (reads the labels): a correct grid anywhere in the pool or the 27B samples goes first."""
    out = []
    for j, o in enumerate(task.outputs):
        truth = bank.truth(o)
        pool = [c["grid"] for c in o.candidates] + [g for _, g in _q27_grids(bank, task, j)]
        hit = next((g for g in pool if truth is not None and grids_equal(g, truth)), None)
        a1, a2 = _pair(o.attempt_1, o.attempt_2)
        out.append((hit, a1 if a1 != hit else a2) if hit is not None else (a1, a2))
    return out


def rule_q27_any(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """27B-only coverage bound (reads the labels): the first correct 27B sample's grid goes first (B0's attempt_1
    second), else B0's attempts as submitted. Its attempt-1 hits on outputs B0 misses are the 27B's own coverage;
    ``q27_hit_keys`` gives that coverage alone."""
    out = []
    for j, o in enumerate(task.outputs):
        truth = bank.truth(o)
        hit = next((g for _, g in _q27_grids(bank, task, j) if truth is not None and grids_equal(g, truth)), None)
        a1, a2 = _pair(o.attempt_1, o.attempt_2)
        out.append((hit, a1 if a1 != hit else a2) if hit is not None else (a1, a2))
    return out


# GS1 (experiments/arc2-glimmer-scorer/GS1_RULE.md, frozen 2026-10-27 23:31 IST with bank gate v2; Revision 1
# frozen 2026-10-28 00:43 IST): Glimmer as a second judge of B0's pool. The family is gs_low alone (gs_all dropped by
# revision 1), fixed before any score exists; weights 1:1, nothing tuned.

def _zscores(values: list[float]) -> list[float]:
    """(v - mean) / population SD within one output's candidates; all 0 when the SD is 0."""
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / n)
    return [0.0] * n if sd <= 0 else [(v - m) / sd for v in values]


def gs1_ranking(o: Output, bank: Bank) -> list[dict] | None:
    """B0's pool candidates (those with a kgmon rank) ranked by z(B0's kgmon score) + z(Glimmer score), the z-scores
    taken over this output's candidates, weight 1:1; ties keep B0's order. None (B0's order stands) when the output has
    fewer than two B0 candidates or any of them lacks either score. The 27B's grids carry no B0 score and are not
    ranked here (their Glimmer scores are recorded by the analysis only)."""
    cands = o.ranked("rank_kgmon")
    if len(cands) < 2:
        return None
    per = (bank.glimmer.get(o.task_id) or {}).get(o.test_index) or {}
    b0s, gls = [], []
    for c in cands:
        s, g = c.get("score_kgmon"), glimmer_score(per.get(grid_key(c["grid"])))
        if s is None or g is None or not math.isfinite(float(s)):
            return None
        b0s.append(float(s))
        gls.append(g)
    zb, zg = _zscores(b0s), _zscores(gls)
    order = sorted(range(len(cands)), key=lambda i: (-(zb[i] + zg[i]), i))
    return [cands[i] for i in order]


def margin_a2s1(o: Output) -> float:
    """B0's top-two kgmon margin as A2-S1 reads it (experiments/arc2-q27/scripts/arc2_s1_merge_analyze.py): a missing
    margin counts as -inf when B0 has no candidate and +inf when it has one."""
    if o.margin_kgmon is not None:
        return float(o.margin_kgmon)
    return -math.inf if not (o.n_unique or o.candidates) else math.inf


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return math.nan
    if n % 2:
        return s[n // 2]
    a, b = s[n // 2 - 1], s[n // 2]
    return a if a == b or not (math.isfinite(a) and math.isfinite(b)) else (a + b) / 2


def low_margin_medians(bank: Bank) -> dict[str, float]:
    """The median margin (margin_a2s1) per split: {"unsealed": ..., "sealed": ...} over that split's outputs."""
    out = {}
    for name, sealed in (("unsealed", False), ("sealed", True)):
        vals = [margin_a2s1(o) for t in bank.task_ids if bank.tasks[t].sealed == sealed for o in bank.tasks[t].outputs]
        if vals:
            out[name] = _median(vals)
    return out


def low_margin_keys(bank: Bank) -> set[str]:
    """A2-S1's low-margin set, per split (GS1 revision 1: on the unsealed bank and, separately, on the sealed 19): the
    outputs whose margin (margin_a2s1) is below the median over the outputs of their own split. An unsealed-only bank
    gives the same set as one median over all its outputs."""
    med = low_margin_medians(bank)
    return {o.key for t in bank.task_ids for o in bank.tasks[t].outputs
            if margin_a2s1(o) < med["sealed" if bank.tasks[t].sealed else "unsealed"]}


def rule_gs_low(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """GS1 gs_low (the family's one rule, revision 1): on A2-S1's low-margin outputs (margin below its split's
    median), the top two of B0's pool by z(kgmon score) + z(Glimmer score) (gs1_ranking); B0's attempts everywhere
    else and where the ranking does not apply."""
    low = low_margin_keys(bank)
    out = []
    for o in task.outputs:
        r = gs1_ranking(o, bank) if o.key in low else None
        out.append(_top2(r) if r is not None else _pair(o.attempt_1, o.attempt_2))
    return out


# DC1 (experiments/arc2-decider/DC1_RULE.md, drafted 2026-10-28, revised after a check,
# not yet frozen): an open Jev-style decision model, zero-shot, as the judge of B0's pool. The family is dc_fit,
# dc_long, dc_keep1 and dc_keep1_long, fixed before any probability exists; the candidate's score is its mean choice
# probability over two option orders; nothing tuned. Outputs over 16,384 tokens carry no probabilities (B0 stands).

def dc1_ranking(o: Output, bank: Bank, *, fit_only: bool) -> list[dict] | None:
    """B0's pool candidates (those with a kgmon rank) ranked by the decider's mean probability, ties in B0's order.
    None (B0's order stands) with fewer than two candidates, without the output's record, when any candidate lacks a
    probability, or (fit_only) when the output's request did not fit the 4,096-token window in both orders."""
    cands = o.ranked("rank_kgmon")
    if len(cands) < 2:
        return None
    rec = (bank.decider.get(o.task_id) or {}).get(o.test_index)
    if rec is None or (fit_only and not rec.get("fit_4096")):
        return None
    ps = [(rec.get("probs") or {}).get(grid_key(c["grid"])) for c in cands]
    if any(p is None or not math.isfinite(float(p)) for p in ps):
        return None
    order = sorted(range(len(cands)), key=lambda i: (-float(ps[i]), i))
    return [cands[i] for i in order]


def rule_dc_fit(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """DC1 dc_fit: on outputs whose request fits the decider's trained 4,096-token window in both orders, the top two
    of B0's pool by the decider's mean probability; B0's attempts everywhere else."""
    out = []
    for o in task.outputs:
        r = dc1_ranking(o, bank, fit_only=True)
        out.append(_top2(r) if r is not None else _pair(o.attempt_1, o.attempt_2))
    return out


def rule_dc_long(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """DC1 dc_long: dc_fit's ranking on every output with two or more B0 candidates (the 16,384 window where the
    request exceeds 4,096); B0's attempts where the ranking does not apply."""
    out = []
    for o in task.outputs:
        r = dc1_ranking(o, bank, fit_only=False)
        out.append(_top2(r) if r is not None else _pair(o.attempt_1, o.attempt_2))
    return out


def _keep1(o: Output, r: list[dict] | None) -> tuple[Any, Any]:
    """Attempt 1 stays B0's attempt 1; attempt 2 is the decider's top candidate, or its second when the top is B0's
    attempt 1. B0's attempts when the ranking does not apply."""
    if r is None:
        return _pair(o.attempt_1, o.attempt_2)
    a1 = grid_key(o.attempt_1)
    second = next((c["grid"] for c in r if grid_key(c["grid"]) != a1), None)
    return _pair(o.attempt_1, second if second is not None else o.attempt_2)


def rule_dc_keep1(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """DC1 dc_keep1: on outputs that fit the 4,096-token window in both orders, B0's attempt 1 kept and attempt 2 the
    decider's top candidate other than it; B0's attempts elsewhere."""
    return [_keep1(o, dc1_ranking(o, bank, fit_only=True)) for o in task.outputs]


def rule_dc_keep1_long(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """DC1 dc_keep1_long: dc_keep1 on every output with two or more B0 candidates and probabilities (fit_4096 and
    long_16384 tiers); B0's attempts elsewhere."""
    return [_keep1(o, dc1_ranking(o, bank, fit_only=False)) for o in task.outputs]


# B0R (experiments/arc2-bank/B0R_RULE.md, frozen 2026-10-29 15:41 IST at 166b24a):
# three rankings of B0's own candidates from signals already in the pool, each keeping kgmon's first as attempt 1
# and choosing attempt 2 among the other candidates by its score (higher first, ties by kgmon rank). The _v7 forms
# apply v7's merge to the rule's two attempts (G2). Nothing tuned; time-neutral (the solver has every score).

B0R_TOL = 1e-3                # floor item 1: |recomputed - exported| for kgmon and probmul3


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def download_gaps(pool: Path, logs: Path | None = None) -> list[str]:
    """Tasks a kernel's own audit (pool/_run.json: its task_ids less its missing; run.dm.logs) says it wrote but the
    archive lacks. A download made the moment the kernel turns COMPLETE can miss the last files written (the
    decode-many proof's e87109e9, Oct 30); such an archive is re-fetched before it is read."""
    meta_path = pool / "_run.json"
    if not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gaps = {t for t in set(meta.get("task_ids") or []) - set(meta.get("missing") or []) if not (pool / f"{t}.json").is_file()}
    if logs is not None:
        gaps |= {name.split(".")[0] for name in ((meta.get("run") or {}).get("dm") or {}).get("logs") or []
                 if not (logs / name).is_file()}
    return sorted(gaps)


def b0r_check(c: dict, n_aug: int = 8) -> list[str]:
    """B0R floor item 1 on one candidate: the field meanings the rules rely on. [] when they hold, else the failed
    fields. kgmon = n_beams - score_aug_mean; probmul3 = sum over beams of (3 - beam_score) + mean over beams of the
    sum of (3 - s) over the beam's n_aug augmentation scores (= 3 n_aug - n_aug x the beam's aug_mean; the bank: 8;
    decode-many's replayed pools: 8, 16 or 32)."""
    bad = []
    v, aug, a, beams = c.get("n_beams"), c.get("score_aug"), c.get("score_aug_mean"), c.get("beams") or []
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        bad.append("n_beams")
    if not (isinstance(aug, list) and len(aug) == n_aug and all(_finite(s) for s in aug)):
        bad.append("score_aug")
    if not _finite(a):
        bad.append("score_aug_mean")
    if (not beams or len(beams) != v
            or any(not _finite(b.get("beam_score")) or not _finite(b.get("aug_mean")) for b in beams)):
        bad.append("beams")
    if c.get("rank_kgmon") is None or c.get("rank_probmul3") is None:
        bad.append("ranks")
    if bad:
        return bad
    if not _finite(c.get("score_kgmon")) or abs(v - a - c["score_kgmon"]) > B0R_TOL:
        bad.append("kgmon")
    pm = sum(3 - b["beam_score"] for b in beams) + sum(n_aug * (3 - b["aug_mean"]) for b in beams) / len(beams)
    if not _finite(c.get("score_probmul3")) or abs(pm - c["score_probmul3"]) > B0R_TOL:
        bad.append("probmul3")
    return bad


def _b0r_zbeam(cands: list[dict]) -> list[float]:
    """z(v) + z(-a) + z(-b) over the output's candidates: v = n_beams, a = score_aug_mean, b = the best (min)
    beam_score."""
    zv = _zscores([float(c["n_beams"]) for c in cands])
    za = _zscores([-float(c["score_aug_mean"]) for c in cands])
    zb = _zscores([-min(float(b["beam_score"]) for b in c["beams"]) for c in cands])
    return [x + y + z for x, y, z in zip(zv, za, zb)]


def _b0r_amax(cands: list[dict]) -> list[float]:
    """v - a_max: n_beams minus the worst (max) of the 8 augmentation scores."""
    return [float(c["n_beams"]) - max(float(s) for s in c["score_aug"]) for c in cands]


def _b0r_borda(cands: list[dict]) -> list[float]:
    """-(rank_kgmon + rank_probmul3)."""
    return [-float(c["rank_kgmon"] + c["rank_probmul3"]) for c in cands]


B0R_SCORES: dict[str, Callable[[list[dict]], list[float]]] = {"zbeam": _b0r_zbeam, "amax": _b0r_amax,
                                                               "borda": _b0r_borda}


def b0r_ranking(o: Output, score: str) -> list[dict] | None:
    """kgmon's first, then the output's other candidates by the score (higher first, ties by kgmon rank). None (B0's
    attempts stand) without candidates or when any candidate fails the field check (floor item 1)."""
    cands = o.ranked("rank_kgmon")
    if not cands or any(b0r_check(c) for c in cands):
        return None
    s = B0R_SCORES[score](cands)
    rest = sorted(range(1, len(cands)), key=lambda i: (-s[i], i))
    return [cands[0]] + [cands[i] for i in rest]


def b0r_pairs(task: Task, score: str) -> Attempts:
    """B0's attempt 1 kept; attempt 2 = the score's best candidate other than it (dc_keep1's helper)."""
    return [_keep1(o, b0r_ranking(o, score)) for o in task.outputs]


def merge_v7_pairs(task: Task, bank: Bank, pairs: Attempts) -> Attempts:
    """merge_v7 on given attempts: attempt 1 kept; attempt 2 = the first valid 27B grid in sample order that differs
    from attempt 1; unchanged when no sample differs."""
    out = []
    for j, (a1, a2) in enumerate(pairs):
        use = next((g for _, g in _q27_grids(bank, task, j) if g != a1), None)
        out.append((a1, use if use is not None else a2))
    return out


def rule_b0r_zbeam(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0R zbeam: attempt 2 = the best other candidate by z(n_beams) + z(-score_aug_mean) + z(-best beam_score)."""
    return b0r_pairs(task, "zbeam")


def rule_b0r_amax(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0R amax: attempt 2 = the best other candidate by n_beams - max(score_aug)."""
    return b0r_pairs(task, "amax")


def rule_b0r_borda(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0R borda: attempt 2 = the best other candidate by -(rank_kgmon + rank_probmul3)."""
    return b0r_pairs(task, "borda")


def rule_b0r_zbeam_v7(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0R zbeam inside v7's merge (G2)."""
    return merge_v7_pairs(task, bank, b0r_pairs(task, "zbeam"))


def rule_b0r_amax_v7(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0R amax inside v7's merge (G2)."""
    return merge_v7_pairs(task, bank, b0r_pairs(task, "amax"))


def rule_b0r_borda_v7(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """B0R borda inside v7's merge (G2)."""
    return merge_v7_pairs(task, bank, b0r_pairs(task, "borda"))


def b0r_code_fingerprint(score: str) -> str:
    """sha256 (16 hex) over the source of everything a B0R rule's attempts depend on (the rule fingerprint hashes the
    rule function alone)."""
    fns = (b0r_check, B0R_SCORES[score], b0r_ranking, b0r_pairs, _keep1, merge_v7_pairs, _zscores, _q27_grids)
    return hashlib.sha256("\n".join(inspect.getsource(f) for f in fns).encode()).hexdigest()[:16]


# null families (E12): rules that carry no information, for the gate's calibration

# ----------------------------------------------------------------------------
# TCP, the time-cut merged projection (experiments/arc2-bank/S1_RULE.md section 5; DM_RULE.md uses it). A slower B0
# moves the entry's B0 phase end later by its extra hidden-mix wall and the 27B loses that much of its window.

TCP_PHASE1_END_S = 33480                                 # v7-vr2's phase1_end_s, 9 h 18 m
TCP_COLLECT_END_S = 40800                                # v7-vr2's collect_end_s, 11 h 20 m
TCP_WINDOW_S = TCP_COLLECT_END_S - TCP_PHASE1_END_S      # C = 7,320 s


def tcp_e13(bank: Bank, seconds: dict[str, float] | None = None, tasks: Iterable[str] | None = None,
            cap_s: float | None = None) -> dict:
    """E13 (scripts/arc2_bank_profile.py fit_ols and project, its constants unchanged) on per-task seconds (default:
    each task's run elapsed_s), fitted on size and outputs over ``tasks`` (default: all with a size): the mean
    hidden-mix wall and the cut fractions at 33,480 and 40,800 s. ``cap_s``: the clip of each predicted task time,
    by default E13's 1,200 s; TS_RULE.md section 5 clips G+T at its own maximum, 2,400 s."""
    import importlib
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    prof = importlib.import_module("arc2_bank_profile")
    ids = sorted(tasks) if tasks is not None else bank.task_ids
    rows = []
    for t in ids:
        task = bank.tasks.get(t)
        y = (seconds or {}).get(t) if seconds is not None else (task.elapsed_s if task else None)
        if task is None or task.size is None or y is None:
            continue
        rows.append((task.size, len(task.outputs), float(y)))
    model = prof.fit_ols([[a, b] for a, b, _ in rows], [y for *_, y in rows], ["size", "n_outputs"])
    sizes = list(bank.sizes_all.values()) or [a for a, _, _ in rows]
    proj = prof.project(sizes, model, n_tasks=prof.HIDDEN_TASKS, outputs_per_task=prof.HIDDEN_OUTPUTS_PER_TASK,
                        ranks=4, overhead_s=600.0, startup_s=prof.STARTUP_S,
                        cap_s=float(prof.PER_TASK_CAP_S if cap_s is None else cap_s),
                        budgets=(TCP_PHASE1_END_S, TCP_COLLECT_END_S), n_resample=200, seed=0)
    return {"wall_s": proj["wall_s"]["mean"], "cut_33480": proj["budgets"][str(TCP_PHASE1_END_S)]["cut_fraction_mean"],
            "cut_40800": proj["budgets"][str(TCP_COLLECT_END_S)]["cut_fraction_mean"], "n_rows": len(rows),
            "coef": [round(c, 6) for c in model["coef"]], "sum_s": round(sum(y for *_, y in rows), 1),
            "cap_s": float(prof.PER_TASK_CAP_S if cap_s is None else cap_s)}


def tcp_keep(bank: Bank, k: float) -> dict[str, dict[int, dict]]:
    """The 27B samples kept at fraction k of the window: the entry's order (v7-vr2, stage-major, shortest first:
    every band task's first sample, then every second), while the running sum of their wall_s stays <= k x the sum
    over all of them; the later ones are dropped."""
    route = [t for t in bank.route_order_all() if t in bank.q27]
    idx = sorted({j for t in route for j in bank.q27[t]})
    order = [(t, j) for j in idx for t in route if j in bank.q27[t]]
    total = sum(float(bank.q27[t][j].get("wall_s") or 0.0) for t, j in order)
    kept: dict[str, dict[int, dict]] = {}
    used = 0.0
    for t, j in order:
        w = float(bank.q27[t][j].get("wall_s") or 0.0)
        if used + w > k * total + 1e-9:
            break
        used += w
        kept.setdefault(t, {})[j] = bank.q27[t][j]
    return kept


def tcp(variant: Bank, ref_e13: dict, seconds: dict[str, float] | None = None,
        tasks: Iterable[str] | None = None, cap_s: float | None = None) -> dict:
    """TCP of a variant against a reference's E13 (S1: the bank's run; decode-many: s0 replayed from the run). The
    variant's bank comes back with the 27B samples cut to k and, when its extra time exceeds the window, B0's
    attempts removed on the longest round(f x n) tasks in route order."""
    e = tcp_e13(variant, seconds, tasks, cap_s)
    delta = max(0.0, e["wall_s"] - ref_e13["wall_s"])
    k = max(0.0, 1.0 - delta / TCP_WINDOW_S)
    f = max(0.0, e["cut_40800"] - ref_e13["cut_33480"]) if delta > TCP_WINDOW_S else 0.0
    kept = tcp_keep(variant, k)
    order = variant.route_order()
    n_cut = int(round(f * len(order)))
    cut = order[len(order) - n_cut:] if n_cut else []
    new_tasks = dict(variant.tasks)
    for t in cut:
        old = variant.tasks[t]
        new_tasks[t] = copy.copy(old)
        new_tasks[t].outputs = [Output(o.task_id, o.test_index, [], PLACEHOLDER, PLACEHOLDER) for o in old.outputs]
    cut_bank = copy.copy(variant)
    cut_bank.tasks, cut_bank.q27 = new_tasks, kept
    n_all = sum(len(v) for v in variant.q27.values())
    return {"e13": e, "ref_wall_s": ref_e13["wall_s"], "delta_s": round(delta, 1), "k": round(k, 4),
            "samples_kept": sum(len(v) for v in kept.values()), "samples_all": n_all,
            "b0_cut_fraction": round(f, 4), "b0_cut_tasks": cut, "bank": cut_bank}


def null_pair(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """Two distinct candidates drawn at random from the pool (B0's attempts when the pool is empty)."""
    out = []
    for o in task.outputs:
        grids = [c["grid"] for c in o.candidates]
        if len(grids) < 2:
            out.append(_pair(o.attempt_1, o.attempt_2) if not grids else _pair(grids[0], o.attempt_2))
            continue
        i, j = rng.sample(range(len(grids)), 2)
        out.append(_pair(grids[i], grids[j]))
    return out


def null_rerank(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """A random permutation of the kgmon ranking; its top two."""
    out = []
    for o in task.outputs:
        cands = o.ranked("rank_kgmon")
        if not cands:
            out.append(_pair(o.attempt_1, o.attempt_2))
            continue
        perm = list(cands)
        rng.shuffle(perm)
        out.append(_top2(perm))
    return out


def null_jitter(task: Task, bank: Bank, rng: random.Random, sigma: float = 0.5) -> Attempts:
    """kgmon scores jittered by Gaussian noise (sigma 0.5 in score units, the size of a typical margin); top two
    by the jittered score. Close calls flip, clear ones stay: the tie-break null."""
    out = []
    for o in task.outputs:
        cands = [c for c in o.candidates if c.get("score_kgmon") is not None]
        if not cands:
            out.append(_pair(o.attempt_1, o.attempt_2))
            continue
        scored = sorted(cands, key=lambda c: -(float(c["score_kgmon"]) + rng.gauss(0.0, sigma)))
        out.append(_top2(scored))
    return out


def null_jitter1(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """kgmon scores jittered with sigma 1.0 (see null_jitter)."""
    return null_jitter(task, bank, rng, 1.0)


def null_jitter2(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """kgmon scores jittered with sigma 2.0 (see null_jitter)."""
    return null_jitter(task, bank, rng, 2.0)


def null_q27swap(task: Task, bank: Bank, rng: random.Random) -> Attempts:
    """v7's merge with the 27B samples in random order (which differing sample fills attempt_2 is chance)."""
    per = bank.q27.get(task.task_id) or {}
    order = sorted(per)
    rng.shuffle(order)
    return merge_v7(task, bank, order)


NULL_FAMILIES: dict[str, RuleFn] = {"null_pair": null_pair, "null_rerank": null_rerank, "null_jitter": null_jitter,
                                    "null_jitter1": null_jitter1, "null_jitter2": null_jitter2,
                                    "null_q27swap": null_q27swap}

RULES: dict[str, Rule] = {r.name: r for r in (
    Rule("b0", rule_b0, "b0", "B0 as submitted (kgmon top two); the base"),
    Rule("b0_kgmon", rule_b0_kgmon, "b0", "kgmon top two recomputed from the pool (consistency check)"),
    Rule("b0_probmul3", rule_b0_probmul3, "b0", "probmul3 top two from the pool"),
    Rule("v7_direct", rule_v7_direct, "merge", "hybrid v7: attempt_2 = first 27B sample that differs from attempt_1"),
    Rule("q27_first", rule_q27_first, "merge", "27B's first grid as attempt_1, B0's attempt_1 second"),
    Rule("q27_agree", rule_q27_agree, "merge", "E15: agreeing 27B samples first, else v7's merge"),
    Rule("oracle", rule_oracle, "oracle", "coverage bound: any correct grid in the pool or samples first", True),
    Rule("q27_any", rule_q27_any, "oracle", "27B-only coverage bound: any correct 27B sample first, else B0's attempts "
         "(reads labels; never adoptable)", True),
    Rule("gs_low", rule_gs_low, "gs1", "GS1 rev. 1: B0's pool by z(kgmon) + z(Glimmer identity-view log-likelihood), "
         "1:1, on A2-S1's low-margin outputs only (median per split); B0's order elsewhere"),
    Rule("dc_fit", rule_dc_fit, "dc1", "DC1: B0's pool by the decision model's mean choice probability (two option "
         "orders) where the request fits its 4,096-token window; B0's order elsewhere"),
    Rule("dc_long", rule_dc_long, "dc1", "DC1: B0's pool by the decision model's mean choice probability (two option "
         "orders) on every output with two or more candidates (16,384 window where needed)"),
    Rule("dc_keep1", rule_dc_keep1, "dc1", "DC1: B0's attempt 1 kept; attempt 2 the decision model's top candidate other "
         "than it, where the request fits 4,096 tokens; B0's attempts elsewhere"),
    Rule("dc_keep1_long", rule_dc_keep1_long, "dc1", "DC1: dc_keep1 on every output with two or more candidates and "
         "probabilities (16,384 window where needed)"),
    Rule("b0r_zbeam", rule_b0r_zbeam, "b0r", "B0R: kgmon's first kept; attempt 2 the best other candidate by "
         "z(n_beams) + z(-score_aug_mean) + z(-best beam_score)"),
    Rule("b0r_amax", rule_b0r_amax, "b0r", "B0R: kgmon's first kept; attempt 2 the best other candidate by "
         "n_beams - max(score_aug)"),
    Rule("b0r_borda", rule_b0r_borda, "b0r", "B0R: kgmon's first kept; attempt 2 the best other candidate by "
         "-(rank_kgmon + rank_probmul3)"),
    Rule("b0r_zbeam_v7", rule_b0r_zbeam_v7, "b0r", "B0R zbeam's attempts inside v7's merge (G2)"),
    Rule("b0r_amax_v7", rule_b0r_amax_v7, "b0r", "B0R amax's attempts inside v7's merge (G2)"),
    Rule("b0r_borda_v7", rule_b0r_borda_v7, "b0r", "B0R borda's attempts inside v7's merge (G2)"),
)}
GS1_RULES = ("gs_low",)       # GS1 revision 1: gs_low alone (gs_all dropped before any score existed)
DC1_RULES = ("dc_fit", "dc_long", "dc_keep1", "dc_keep1_long")
B0R_RULES = ("b0r_zbeam", "b0r_amax", "b0r_borda")   # B0R_RULE.md; each with its _v7 form; not in the default list


def get_rule(name: str) -> Rule:
    """A registered rule, or a null rule ``<family>:<seed>`` (seeded, reproducible)."""
    if name in RULES:
        return RULES[name]
    fam, _, seed = name.partition(":")
    if fam in NULL_FAMILIES and seed.lstrip("-").isdigit():
        return Rule(name, NULL_FAMILIES[fam], "null", f"{NULL_FAMILIES[fam].__doc__.strip().splitlines()[0]} (seed {seed})")
    raise KeyError(f"unknown rule {name!r}; registered: {', '.join(sorted(RULES))}; null families: "
                   f"{', '.join(sorted(NULL_FAMILIES))} as <family>:<seed>")


def rule_seed(rule: Rule) -> int:
    _, _, seed = rule.name.partition(":")
    return int(seed) if seed.lstrip("-").isdigit() else 0


# ----------------------------------------------------------------------------
# replay and scoring

@dataclass
class Result:
    rule: str
    n_tasks: int
    n_outputs: int
    n_right: int
    n_right_1: int
    score: float                      # official: outputs right in either attempt / outputs
    task_score: float                 # mean over tasks of the fraction of outputs right
    per_output: dict[str, dict]       # key -> {task_id, test_index, c1, c2, right}
    submission: dict = field(default_factory=dict, repr=False)
    bank_fingerprint: str = ""
    rule_fingerprint: str = ""

    def right_keys(self) -> set[str]:
        return {k for k, v in self.per_output.items() if v["right"]}

    def per_task(self) -> dict[str, float]:
        acc: dict[str, list[bool]] = {}
        for v in self.per_output.values():
            acc.setdefault(v["task_id"], []).append(bool(v["right"]))
        return {t: sum(h) / len(h) for t, h in acc.items()}

    def summary(self) -> dict:
        return {"rule": self.rule, "n_tasks": self.n_tasks, "n_outputs": self.n_outputs, "n_right": self.n_right,
                "n_right_1": self.n_right_1, "score": round(self.score, 4), "task_score": round(self.task_score, 4),
                "bank_fingerprint": self.bank_fingerprint, "rule_fingerprint": self.rule_fingerprint}


def build_submission(bank: Bank, rule: Rule, rng: random.Random | None = None) -> dict[str, list[dict]]:
    """{task_id: [{attempt_1, attempt_2} per test output]} from the rule over the bank's tasks."""
    rng = rng if rng is not None else random.Random(rule_seed(rule))
    sub = {}
    for t in bank.task_ids:
        task = bank.tasks[t]
        pairs = rule.fn(task, bank, rng)
        if len(pairs) != len(task.outputs):
            raise ValueError(f"rule {rule.name} returned {len(pairs)} pairs for {len(task.outputs)} outputs of {t}")
        sub[t] = [{"attempt_1": copy.deepcopy(a1), "attempt_2": copy.deepcopy(a2)} for a1, a2 in pairs]
    return sub


def score_bank(bank: Bank, submission: dict, name: str = "") -> Result:
    """Score a submission on the bank's outputs with the official formula (eval/arc2_score.py)."""
    sols = {t: bank.solutions[t] for t in bank.task_ids if t in bank.solutions}
    rep = score_submission(submission, sols) if sols else {"per_output": [], "n_outputs": 0, "n_solved": 0}
    per = {}
    for r in rep["per_output"]:
        per[f"{r['task_id']}_{r['test_index']}"] = {"task_id": r["task_id"], "test_index": r["test_index"],
                                                     "c1": bool(r["correct_1"]), "c2": bool(r["correct_2"]),
                                                     "right": bool(r["correct"])}
    n_out = len(per)
    by_task: dict[str, list[bool]] = {}
    for v in per.values():
        by_task.setdefault(v["task_id"], []).append(v["right"])
    n_right = sum(v["right"] for v in per.values())
    n1 = sum(v["c1"] for v in per.values())
    task_score = (sum(sum(h) / len(h) for h in by_task.values()) / len(by_task)) if by_task else 0.0
    return Result(name, len(by_task), n_out, n_right, n1, n_right / n_out if n_out else 0.0, task_score, per,
                  submission, bank.fingerprint())


def replay(bank: Bank, rule: Rule | str, rng: random.Random | None = None) -> Result:
    rule = get_rule(rule) if isinstance(rule, str) else rule
    res = score_bank(bank, build_submission(bank, rule, rng), rule.name)
    res.rule_fingerprint = rule.fingerprint()
    return res


# ----------------------------------------------------------------------------
# statistics (pure Python: the kaggle-sim environment has numpy but no scipy)

def binom_tail(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if n <= 0:
        return 1.0
    k = max(0, int(k))
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test_p(gained: int, lost: int) -> float | None:
    """One-sided sign test on the discordant outputs: P(gains >= observed | fair coin); None without discordants."""
    n = gained + lost
    return binom_tail(gained, n) if n else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    return m, (math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0)


def cluster_bootstrap(per_task_delta: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample tasks with replacement; the distribution of the mean per-task delta (task-level, each task 1)."""
    rng = random.Random(seed)
    tasks = sorted(per_task_delta)
    if not tasks:
        return {"n_tasks": 0}
    vals = [per_task_delta[t] for t in tasks]
    if n_boot <= 0:
        return {"n_tasks": len(tasks), "n_boot": 0, "mean": round(sum(vals) / len(vals), 5)}
    means = []
    for _ in range(n_boot):
        pick = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(pick) / len(pick))
    m, sd = mean_sd(means)
    return {"n_tasks": len(tasks), "n_boot": n_boot, "mean": round(sum(vals) / len(vals), 5), "sd": round(sd, 5),
            "ci95": [round(percentile(means, 0.025), 5), round(percentile(means, 0.975), 5)],
            "p_le_0": round(sum(1 for v in means if v <= 0) / n_boot, 4)}


# the gates as pre-registered and as revised after a check (proposed revision)
GATE_V1 = {"net_min": 4, "p_max": 0.05}
GATE_V2 = {"net_min": 3, "lost_max": 1}


def compare(base: Result, new: Result, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison on the shared outputs: gained / lost / net, the one-sided sign test on the discordants,
    the task-level paired difference with a cluster bootstrap by task, and both gate readings."""
    shared = sorted(set(base.per_output) & set(new.per_output))
    gained = [k for k in shared if new.per_output[k]["right"] and not base.per_output[k]["right"]]
    lost = [k for k in shared if base.per_output[k]["right"] and not new.per_output[k]["right"]]
    net = len(gained) - len(lost)
    p = sign_test_p(len(gained), len(lost))
    bt, nt = base.per_task(), new.per_task()
    delta = {t: nt[t] - bt[t] for t in bt if t in nt}
    changed = {t: d for t, d in delta.items() if abs(d) > 1e-12}
    boot = cluster_bootstrap(delta, n_boot, seed)
    gate_v1 = net >= GATE_V1["net_min"] and p is not None and p < GATE_V1["p_max"]
    gate_v2 = net >= GATE_V2["net_min"] and len(lost) <= GATE_V2["lost_max"]
    return {"base": base.rule, "new": new.rule, "n_shared_outputs": len(shared),
            "base_right": sum(1 for k in shared if base.per_output[k]["right"]),
            "new_right": sum(1 for k in shared if new.per_output[k]["right"]),
            "gained": len(gained), "lost": len(lost), "net": net, "gained_keys": gained, "lost_keys": lost,
            "sign_test_p": None if p is None else round(p, 5),
            "task_level": {"n_tasks": len(delta), "n_changed": len(changed),
                           "mean_delta": round(sum(delta.values()) / len(delta), 5) if delta else None,
                           "base_task_score": round(base.task_score, 4), "new_task_score": round(new.task_score, 4),
                           "bootstrap": boot},
            "gate_v1": {"rule": "net >= 4 and sign test p < 0.05", "pass": bool(gate_v1)},
            "gate_v2": {"rule": "net >= 3 and lost <= 1 (proposed revision)", "pass": bool(gate_v2)}}


def bootstrap_board(result: Result, n_tasks: int = 120, n_boot: int = 4000, seed: int = 0) -> dict:
    """E1: the spread of a board-sized reading of this rule. Tasks are resampled with replacement to ``n_tasks``
    (the public board is about half of 240 hidden tasks); the score is output-weighted as on the board."""
    rng = random.Random(seed)
    by_task: dict[str, list[bool]] = {}
    for v in result.per_output.values():
        by_task.setdefault(v["task_id"], []).append(v["right"])
    tasks = sorted(by_task)
    if not tasks:
        return {"n_tasks": 0}
    scores, outs = [], []
    for _ in range(n_boot):
        pick = [by_task[tasks[rng.randrange(len(tasks))]] for _ in range(n_tasks)]
        n_out = sum(len(h) for h in pick)
        scores.append(100.0 * sum(sum(h) for h in pick) / n_out)
        outs.append(n_out)
    m, sd = mean_sd(scores)
    return {"rule": result.rule, "n_tasks_resampled": n_tasks, "n_boot": n_boot, "mean_points": round(m, 2),
            "sd_points": round(sd, 2), "ci95_points": [round(percentile(scores, 0.025), 2), round(percentile(scores, 0.975), 2)],
            "mean_outputs": round(sum(outs) / len(outs), 1)}


def bootstrap_difference(base: Result, new: Result, n_tasks: int = 120, n_boot: int = 4000, seed: int = 0) -> dict:
    """E1: paired resample of tasks; the spread of the board-point difference new - base and the minimum detectable
    difference (the 95th percentile of |difference| under task resampling)."""
    rng = random.Random(seed)
    bt: dict[str, list[bool]] = {}
    nt: dict[str, list[bool]] = {}
    for k, v in base.per_output.items():
        if k in new.per_output:
            bt.setdefault(v["task_id"], []).append(v["right"])
            nt.setdefault(v["task_id"], []).append(new.per_output[k]["right"])
    tasks = sorted(bt)
    if not tasks:
        return {"n_tasks": 0}
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(tasks)) for _ in range(n_tasks)]
        n_out = sum(len(bt[tasks[i]]) for i in idx)
        b = sum(sum(bt[tasks[i]]) for i in idx)
        n = sum(sum(nt[tasks[i]]) for i in idx)
        diffs.append(100.0 * (n - b) / n_out)
    m, sd = mean_sd(diffs)
    return {"base": base.rule, "new": new.rule, "n_tasks_resampled": n_tasks, "n_boot": n_boot,
            "mean_points": round(m, 2), "sd_points": round(sd, 2),
            "ci95_points": [round(percentile(diffs, 0.025), 2), round(percentile(diffs, 0.975), 2)],
            "p_le_0": round(sum(1 for d in diffs if d <= 0) / n_boot, 4)}


# ----------------------------------------------------------------------------
# views used by the profile and the calibration ledger

def quartiles_by_size(bank: Bank) -> dict[str, list[str]]:
    """The bank's task ids by task-length quartile of the whole route (120 tasks): q1 = shortest quarter (the
    hybrid's band)."""
    order = bank.route_order_all()
    n = len(order)
    out = {}
    for q in range(4):
        out[f"q{q + 1}"] = [t for t in order[(q * n) // 4:((q + 1) * n) // 4] if t in bank.tasks]
    return out


def pool_hit_keys(bank: Bank) -> set[str]:
    """Output keys where some B0 pool candidate equals the truth (B0's pool coverage, no 27B samples)."""
    out = set()
    for o in bank.outputs():
        truth = bank.truth(o)
        if truth is not None and any(grids_equal(c["grid"], truth) for c in o.candidates):
            out.add(o.key)
    return out


def q27_hit_keys(bank: Bank, samples: Iterable[int] | None = None) -> set[str]:
    """Output keys where some 27B sample's grid equals the truth (the 27B-only union; ``samples`` restricts the
    sample indices, e.g. (0, 1) for the bank's two)."""
    keep = set(samples) if samples is not None else None
    out = set()
    for t in bank.task_ids:
        task = bank.tasks[t]
        for j, o in enumerate(task.outputs):
            truth = bank.truth(o)
            if truth is None:
                continue
            for k, g in _q27_grids(bank, task, j):
                if (keep is None or k in keep) and grids_equal(g, truth):
                    out.add(o.key)
                    break
    return out


def subset_score(result: Result, task_ids: Iterable[str]) -> dict:
    ids = set(task_ids)
    per = [v for v in result.per_output.values() if v["task_id"] in ids]
    n = len(per)
    return {"n_tasks": len({v["task_id"] for v in per}), "n_outputs": n, "n_right": sum(v["right"] for v in per),
            "score": round(sum(v["right"] for v in per) / n, 4) if n else None}
