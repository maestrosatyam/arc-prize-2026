"""Build hash-grouped split manifests for ARC-AGI-2 (S03, evaluator-side; reads NO test labels).

Outputs (default --out splits/):
  arc2_eval_split_v1.json      dev / tune / holdout (target 40/40/40) over the 120 evaluation tasks,
                               deterministic by sha256(task_id + 'arc2-split-v1'), groups kept together.
  arc2_training_overlap_v1.json  per training task: ARC-AGI-1 id match, ARC-AGI-1 content-hash match,
                               structural group id; plus evaluation tasks that structurally match
                               training or ARC-AGI-1 tasks.

Structural near-duplicate grouping: each train INPUT grid is canonicalized under the 8 dihedral
symmetries combined with color relabeling by first appearance (row-major) so that any color
permutation maps to the same form; the lexicographically smallest of the 8 forms is the grid's
canonical string. A task's canonical key is sha256 of the sorted multiset of its train-input
canonical strings. Tasks sharing a key are one group (union over the training, evaluation and
ARC-AGI-1 pools). Only train inputs are used, never test grids or outputs.

Usage:
  python scripts/arc2_build_splits.py --repo third_party/ARC-AGI-2 --arc1 /tmp/arc-agi-1-clone/ARC-AGI --out splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc2.io.loader import iter_task_files, task_from_dict  # noqa: E402
from arc2.io.types import Grid, TaskView  # noqa: E402
from arc2.io.writer import atomic_write_json  # noqa: E402

SPLIT_SALT = "arc2-split-v1"
SPLIT_NAMES = ("dev", "tune", "holdout")
TARGET_SIZES = {"dev": 40, "tune": 40, "holdout": 40}


# ----------------------------------------------------------------------------------------------
# Canonicalization
# ----------------------------------------------------------------------------------------------


def _rot90(g: Grid) -> Grid:
    h, w = len(g), len(g[0])
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _flip_h(g: Grid) -> Grid:
    return [list(reversed(row)) for row in g]


def dihedral_variants(g: Grid) -> list[Grid]:
    """The 8 symmetries of the square: 4 rotations of g and 4 rotations of its mirror."""
    out: list[Grid] = []
    for base in (g, _flip_h(g)):
        cur = base
        for _ in range(4):
            out.append(cur)
            cur = _rot90(cur)
    return out


def color_canonical(g: Grid) -> str:
    """Relabel colors by order of first appearance (row-major) and serialize as 'HxW:digits'."""
    mapping: dict[int, int] = {}
    chars: list[str] = []
    for row in g:
        for v in row:
            if v not in mapping:
                mapping[v] = len(mapping)
            chars.append(str(mapping[v]))
    return f"{len(g)}x{len(g[0])}:{''.join(chars)}"


def grid_canonical(g: Grid) -> str:
    """Lexicographically smallest color-canonical serialization over the 8 dihedral variants."""
    return min(color_canonical(v) for v in dihedral_variants(g))


def task_structural_key(task: TaskView) -> str:
    """sha256 of the sorted multiset of canonical train-input strings (labels never used)."""
    forms = sorted(grid_canonical(p.input) for p in task.train)
    return hashlib.sha256("\n".join(forms).encode("utf-8")).hexdigest()


def task_content_hash(raw: dict) -> str:
    """sha256 of the task content (train + test pairs, labels included), insensitive to pair order.

    ARC-AGI-2 re-shuffled the train pair order of every task it inherited from ARC-AGI-1, so a hash
    of the raw file would match nothing; the multiset of pairs is what identifies a copied task.
    Used only to detect copies of ARC-AGI-1 tasks; this is not exposed to solvers.
    """
    def pairs(section: str) -> list[str]:
        return sorted(json.dumps(p, sort_keys=True, separators=(",", ":")) for p in raw.get(section, []))

    body = json.dumps({"train": pairs("train"), "test": pairs("test")}, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def split_sort_key(task_id: str) -> str:
    return hashlib.sha256((task_id + SPLIT_SALT).encode("utf-8")).hexdigest()


def manifest_hash(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------
# Loading (label-free views for structure; raw content hashes only for ARC-1 identity)
# ----------------------------------------------------------------------------------------------


def _load_pool(directory: Path) -> tuple[dict[str, TaskView], dict[str, str]]:
    views: dict[str, TaskView] = {}
    content: dict[str, str] = {}
    for tid, p in iter_task_files(directory):
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        views[tid] = task_from_dict(tid, raw)
        content[tid] = task_content_hash(raw)
    return views, content


def _git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def assign_splits(
    eval_ids: Iterable[str],
    group_of: dict[str, str],
    targets: dict[str, int] = TARGET_SIZES,
) -> dict[str, list[str]]:
    """Deterministic group-aware assignment.

    Groups are ordered by the smallest split_sort_key among their members. Each group goes to the
    first split (dev, tune, holdout order) whose remaining capacity fits the whole group; if none
    fits, the split with the most remaining capacity takes it. Groups are never divided.
    """
    members: dict[str, list[str]] = defaultdict(list)
    for tid in eval_ids:
        members[group_of[tid]].append(tid)
    ordered_groups = sorted(members, key=lambda gid: min(split_sort_key(t) for t in members[gid]))
    remaining = dict(targets)
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for gid in ordered_groups:
        tids = sorted(members[gid], key=split_sort_key)
        fit = [s for s in SPLIT_NAMES if remaining[s] >= len(tids)]
        target = fit[0] if fit else max(SPLIT_NAMES, key=lambda s: (remaining[s], -SPLIT_NAMES.index(s)))
        out[target].extend(tids)
        remaining[target] -= len(tids)
    for name in SPLIT_NAMES:
        out[name].sort()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(_ROOT / "third_party" / "ARC-AGI-2"))
    ap.add_argument("--arc1", required=True, help="path to a clone of https://github.com/fchollet/ARC-AGI")
    ap.add_argument("--out", default=str(_ROOT / "splits"))
    args = ap.parse_args(argv)
    repo, arc1, out = Path(args.repo), Path(args.arc1), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_views, train_content = _load_pool(repo / "data" / "training")
    eval_views, eval_content = _load_pool(repo / "data" / "evaluation")
    arc1_views: dict[str, TaskView] = {}
    arc1_content: dict[str, str] = {}
    arc1_split_of: dict[str, str] = {}
    for split in ("training", "evaluation"):
        v, c = _load_pool(arc1 / "data" / split)
        for tid in v:
            if tid in arc1_views:
                raise SystemExit(f"ARC-AGI-1 id {tid} appears in both splits")
            arc1_split_of[tid] = split
        arc1_views.update(v)
        arc1_content.update(c)

    # Structural keys over all pools.
    key_of: dict[str, str] = {}
    pool_of: dict[str, str] = {}
    for pool, views in (("arc2_training", train_views), ("arc2_evaluation", eval_views), ("arc1", arc1_views)):
        for tid, view in views.items():
            qualified = f"{pool}:{tid}"
            key_of[qualified] = task_structural_key(view)
            pool_of[qualified] = pool
    groups: dict[str, list[str]] = defaultdict(list)
    for q, key in key_of.items():
        groups[key].append(q)
    group_id_of_key = {key: key[:16] for key in groups}

    # ARC-AGI-1 overlap of the 1000 training tasks.
    arc1_content_to_id: dict[str, str] = {}
    for tid, h in arc1_content.items():
        arc1_content_to_id.setdefault(h, tid)
    training_rows: dict[str, dict] = {}
    n_id, n_content, n_struct = 0, 0, 0
    for tid in sorted(train_views):
        key = key_of[f"arc2_training:{tid}"]
        id_match = tid in arc1_views
        content_match = arc1_content_to_id.get(train_content[tid])
        struct_matches = sorted(q.split(":", 1)[1] for q in groups[key] if pool_of[q] == "arc1")
        n_id += int(id_match)
        n_content += int(content_match is not None)
        n_struct += int(bool(struct_matches))
        training_rows[tid] = {
            "arc1_id_match": id_match,
            "arc1_split": arc1_split_of.get(tid),
            "arc1_content_match": content_match,
            "arc1_structural_matches": struct_matches,
            "group_id": group_id_of_key[key],
            "n_train": len(train_views[tid].train),
            "n_test": len(train_views[tid].test_inputs),
        }
    n_arc1_overlap = sum(
        1
        for r in training_rows.values()
        if r["arc1_id_match"] or r["arc1_content_match"] is not None or r["arc1_structural_matches"]
    )

    # Evaluation-task leakage flags and grouping among the 120.
    eval_rows: dict[str, dict] = {}
    eval_group_of: dict[str, str] = {}
    for tid in sorted(eval_views):
        key = key_of[f"arc2_evaluation:{tid}"]
        eval_group_of[tid] = group_id_of_key[key]
        eval_rows[tid] = {
            "group_id": group_id_of_key[key],
            "arc1_id_match": tid in arc1_views,
            "arc1_content_match": arc1_content_to_id.get(eval_content[tid]),
            "training_structural_matches": sorted(
                q.split(":", 1)[1] for q in groups[key] if pool_of[q] == "arc2_training"
            ),
            "arc1_structural_matches": sorted(q.split(":", 1)[1] for q in groups[key] if pool_of[q] == "arc1"),
            "n_train": len(eval_views[tid].train),
            "n_test": len(eval_views[tid].test_inputs),
        }

    splits = assign_splits(eval_views.keys(), eval_group_of)
    eval_group_members: dict[str, list[str]] = defaultdict(list)
    for tid, gid in eval_group_of.items():
        eval_group_members[gid].append(tid)

    split_manifest = {
        "schema": "arc2_eval_split.v1",
        "salt": SPLIT_SALT,
        "ordering": "sha256(task_id + salt), groups ordered by their minimum key, assigned in dev/tune/holdout order",
        "source_repo_sha": _git_sha(repo),
        "arc1_repo_sha": _git_sha(arc1),
        "n_tasks": len(eval_views),
        "target_sizes": TARGET_SIZES,
        "sizes": {name: len(ids) for name, ids in splits.items()},
        "n_test_outputs": {
            name: sum(len(eval_views[t].test_inputs) for t in ids) for name, ids in splits.items()
        },
        "splits": splits,
        "group_of": eval_group_of,
        "groups": {gid: sorted(m) for gid, m in sorted(eval_group_members.items())},
        "n_groups": len(eval_group_members),
        "n_multi_member_groups": sum(1 for m in eval_group_members.values() if len(m) > 1),
        "eval_tasks": eval_rows,
    }
    split_manifest["manifest_sha256"] = manifest_hash(split_manifest)
    atomic_write_json(out / "arc2_eval_split_v1.json", split_manifest, indent=1)

    training_manifest = {
        "schema": "arc2_training_overlap.v1",
        "source_repo_sha": _git_sha(repo),
        "arc1_repo": "https://github.com/fchollet/ARC-AGI",
        "arc1_repo_sha": _git_sha(arc1),
        "arc1_n_tasks": len(arc1_views),
        "n_training_tasks": len(training_rows),
        "n_arc1_id_matches": n_id,
        "n_arc1_content_matches": n_content,
        "n_arc1_structural_matches": n_struct,
        "n_arc1_overlap_any": n_arc1_overlap,
        "n_training_groups": len({r["group_id"] for r in training_rows.values()}),
        "n_training_multi_member_groups": sum(
            1
            for key, m in groups.items()
            if sum(1 for q in m if pool_of[q] == "arc2_training") > 1
        ),
        "n_eval_tasks_structurally_in_training": sum(
            1 for r in eval_rows.values() if r["training_structural_matches"]
        ),
        "n_eval_tasks_in_arc1": sum(
            1
            for r in eval_rows.values()
            if r["arc1_id_match"] or r["arc1_content_match"] is not None or r["arc1_structural_matches"]
        ),
        "training_tasks": training_rows,
    }
    training_manifest["manifest_sha256"] = manifest_hash(training_manifest)
    atomic_write_json(out / "arc2_training_overlap_v1.json", training_manifest, indent=1)

    summary = {k: v for k, v in split_manifest.items() if k not in ("splits", "group_of", "groups", "eval_tasks")}
    summary["training_overlap"] = {k: v for k, v in training_manifest.items() if k != "training_tasks"}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
