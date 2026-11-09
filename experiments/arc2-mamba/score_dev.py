"""Score dev predictions (dev_eval.py output) through scripts/arc2_eval_local.py --split dev with ReplaySolver and
print aggregate metrics only (per-task dev results are not read).

    ../../envs/mamba/bin/python score_dev.py --dev-dir runs/kaggle-check-v1/bundle/experiments/arc2-mamba/runs/check/dev_eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ("official_score", "pass_at_1", "pass_at_2", "n_outputs", "n_solved", "n_tasks", "shape_acc")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--out", default=str(HERE / "runs" / "dev_scores"))
    args = ap.parse_args(argv)
    res = {}
    for pred in sorted(Path(args.dev_dir).glob("*/dev_*_pred.json")):
        arm, mode = pred.parent.name, pred.stem.split("_")[1]
        out = Path(args.out) / f"{arm}_{mode}"
        if not (out / "run_record.json").exists():
            env = dict(os.environ, PYTHONPATH=str(HERE), ARC2_MAMBA_REPLAY=str(pred.resolve()), OMP_NUM_THREADS="2")
            subprocess.run([str(ROOT / "envs" / "mamba" / "bin" / "python"), str(ROOT / "scripts" / "arc2_eval_local.py"),
                            "--solver", "arcssm.solver:ReplaySolver", "--split", "dev", "--per-task-seconds", "5",
                            "--out", str(out)], check=True, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        metric = json.loads((out / "run_record.json").read_text())["metric"]
        n_pred = len(json.loads(pred.read_text()))
        res[f"{arm}:{mode}"] = {"tasks_predicted": n_pred, **{k: metric.get(k) for k in KEYS}}
        print(f"{arm}:{mode}", json.dumps(res[f"{arm}:{mode}"]))
    (Path(args.out) / "dev_scores.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
