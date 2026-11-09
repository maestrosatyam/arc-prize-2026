"""Generate Kaggle-schema challenge/solution files under eval/data/ from the pinned ARC-AGI-2 repo clone.

Usage:
    python scripts/arc2_build_eval_data.py --repo third_party/ARC-AGI-2 --out eval/data

Writes arc-agi_training-challenges.json, arc-agi_training-solutions.json,
arc-agi_evaluation-challenges.json, arc-agi_evaluation-solutions.json and eval/data/MANIFEST.json
(source SHA, counts, sha256 of each file). Solutions files are evaluator-side only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc2.io.loader import kaggle_to_repo_format, load_raw_task_dir, repo_to_kaggle_format  # noqa: E402
from arc2.io.writer import atomic_write_json  # noqa: E402

SPLITS = ("training", "evaluation")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(_ROOT / "third_party" / "ARC-AGI-2"))
    ap.add_argument("--out", default=str(_ROOT / "eval" / "data"))
    args = ap.parse_args(argv)
    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"source_repo": str(repo), "source_sha": _repo_sha(repo), "files": {}, "splits": {}}
    for split in SPLITS:
        raw = load_raw_task_dir(repo / "data" / split)
        challenges, solutions = repo_to_kaggle_format(raw)
        # Round-trip check: converting back must reproduce the validated input exactly.
        back = kaggle_to_repo_format(challenges, solutions)
        if back != {tid: raw[tid] for tid in sorted(raw)}:
            raise SystemExit(f"{split}: round-trip mismatch between repo and kaggle formats")
        ch_path = out / f"arc-agi_{split}-challenges.json"
        so_path = out / f"arc-agi_{split}-solutions.json"
        atomic_write_json(ch_path, challenges)
        atomic_write_json(so_path, solutions)
        n_outputs = sum(len(v) for v in solutions.values())
        manifest["splits"][split] = {
            "n_tasks": len(challenges),
            "n_test_outputs": n_outputs,
            "n_multi_test_tasks": sum(1 for v in solutions.values() if len(v) > 1),
        }
        for p in (ch_path, so_path):
            manifest["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        print(f"{split}: {len(challenges)} tasks, {n_outputs} test outputs -> {ch_path.name}, {so_path.name}")
    atomic_write_json(out / "MANIFEST.json", manifest, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
