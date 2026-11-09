"""Training-set loading (label-free, through arc2.io.loader), the meta-train / meta-val / held-out split, and
the sampler that mixes generated and real tasks into training sequences."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc2.io.loader import load_kaggle_challenges  # noqa: E402

from arcssm.encoding import Encoded, encode, encoded_len, random_aug  # noqa: E402
from arcssm.generators import sample_task  # noqa: E402
from arcssm.util import atomic_write_json  # noqa: E402

TRAIN_CHALLENGES = ROOT / "eval" / "data" / "arc-agi_training-challenges.json"
TAGS = ROOT / "experiments" / "arc2-grammar" / "tasks.json"
SPLIT_PATH = HERE / "splits.json"
SYNTH_DIR = ROOT / "experiments" / "arc2-scene-grammar-eval" / "synthetic" / "tasks"


def to_np(g) -> np.ndarray:
    return np.asarray(g, dtype=np.int8)


def load_training() -> dict:
    """{task_id: {"train": [(in, out)], "test": [in]}} as int8 arrays; test outputs are never present."""
    views = load_kaggle_challenges(str(TRAIN_CHALLENGES))
    return {tid: {"train": [(to_np(p.input), to_np(p.output)) for p in v.train],
                  "test": [to_np(g) for g in v.test_inputs]} for tid, v in views.items()}


def max_side(task) -> int:
    gs = [g for p in task["train"] for g in p] + list(task["test"])
    return max(max(g.shape) for g in gs)


def _h(seed: int, tid: str) -> str:
    return hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()


def _stratified_pick(ids_by_stratum: dict, n_total: int, seed: int) -> list[str]:
    total = sum(len(v) for v in ids_by_stratum.values())
    quotas = {k: n_total * len(v) / total for k, v in ids_by_stratum.items()}
    base = {k: int(q) for k, q in quotas.items()}
    rest = n_total - sum(base.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - base[k]), k))[:rest]:
        base[k] += 1
    out = []
    for k, ids in ids_by_stratum.items():
        out += sorted(ids, key=lambda t: _h(seed, t))[: base[k]]
    return sorted(out)


def make_split(seed: int = 0, n_heldout: int = 200, n_val: int = 40) -> dict:
    tags = json.loads(TAGS.read_text())["tags"]
    tasks = load_training()
    strata: dict = {}
    for tid in sorted(tasks):
        t = tags[tid]
        strata.setdefault(f"{t['change']}|{'inherited' if t['arc1_inherited'] else 'new'}", []).append(tid)
    held = set(_stratified_pick(strata, n_heldout, seed))
    rest = {k: [t for t in v if t not in held] for k, v in strata.items()}
    val = set(_stratified_pick(rest, n_val, seed + 1))
    split = {
        "schema": "arc2_mamba_split.v1",
        "seed": seed,
        "stratified_by": "experiments/arc2-grammar/tasks.json tags[change] x tags[arc1_inherited]",
        "heldout": sorted(held),
        "meta_val": sorted(val),
        "meta_train": sorted(t for t in tasks if t not in held and t not in val),
        "strata": {k: {"n": len(v), "heldout": sum(t in held for t in v), "meta_val": sum(t in val for t in v)}
                   for k, v in sorted(strata.items())},
    }
    split["sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    return split


def load_split() -> dict:
    if not SPLIT_PATH.exists():
        atomic_write_json(SPLIT_PATH, make_split())
    return json.loads(SPLIT_PATH.read_text())


def load_synthetic() -> dict:
    out = {}
    for f in sorted(SYNTH_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        out[f.stem] = {"train": [(to_np(p["input"]), to_np(p["output"])) for p in d["train"]],
                       "test": [to_np(p["input"]) for p in d["test"]],
                       "test_out": [to_np(p["output"]) for p in d["test"]]}
    return out


# ----------------------------------------------------------------------------------------------
# Training sequences
# ----------------------------------------------------------------------------------------------


def fit_pairs(pairs, query, max_len: int, query_out_shape):
    """Drop context pairs from the end until the sequence fits; None when even one context pair does not fit."""
    pairs = list(pairs)
    while pairs and encoded_len(pairs, query[0] if isinstance(query, tuple) else query, query_out_shape) > max_len:
        pairs.pop()
    return pairs if pairs else None


def real_example(rng, task, max_len: int, aug: bool = True) -> Encoded | None:
    """One train pair as the query, the others (shuffled, augmented) as context."""
    pairs = list(task["train"])
    if len(pairs) < 2:
        return None
    a = random_aug(rng) if aug else None
    if a is not None:
        pairs = [(a.fwd(i), a.fwd(o)) for i, o in pairs]
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    q = pairs[-1]
    ctx = fit_pairs(pairs[:-1], q, max_len, q[1].shape)
    if ctx is None:
        return None
    return encode(ctx, q[0], q[1])


def generated_example(rng, max_len: int, max_side_: int) -> tuple[str, Encoded]:
    while True:
        fam, pairs = sample_task(rng, max_side=max_side_)
        q = pairs[-1]
        ctx = fit_pairs(pairs[:-1], q, max_len, q[1].shape)
        if ctx is not None:
            a = random_aug(rng)
            ctx = [(a.fwd(i), a.fwd(o)) for i, o in ctx]
            return fam, encode(ctx, a.fwd(q[0]), a.fwd(q[1]))


class Mixture:
    """Draws training sequences: generated tasks with probability p_gen, else a real meta-train task."""

    def __init__(self, real_tasks: dict, seed: int, p_gen: float = 0.5, max_len: int = 2048, max_side_gen: int = 12,
                 max_side_real: int = 15):
        self.rng = np.random.default_rng(seed)
        self.real = [t for t in real_tasks.values() if max_side(t) <= max_side_real and len(t["train"]) >= 2]
        self.p_gen, self.max_len, self.max_side_gen = p_gen, max_len, max_side_gen

    def draw(self) -> tuple[str, Encoded]:
        while True:
            if self.rng.random() < self.p_gen:
                return generated_example(self.rng, self.max_len, self.max_side_gen)
            t = self.real[int(self.rng.integers(len(self.real)))]
            e = real_example(self.rng, t, self.max_len)
            if e is not None:
                return "real", e

    def batches(self, tokens_per_batch: int, pool: int = 64):
        """Yield lists of (source, Encoded) grouped by similar length, each list under tokens_per_batch padded."""
        while True:
            items = sorted((self.draw() for _ in range(pool)), key=lambda x: len(x[1]))
            batch, longest = [], 0
            out = []
            for it in items:
                n = len(it[1])
                if batch and max(longest, n) * (len(batch) + 1) > tokens_per_batch:
                    out.append(batch)
                    batch, longest = [], 0
                batch.append(it)
                longest = max(longest, n)
            if batch:
                out.append(batch)
            for i in self.rng.permutation(len(out)):
                yield out[i]
