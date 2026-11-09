"""Solvers for scripts/arc2_eval_local.py (contract: arc2/io/types.py). Run with PYTHONPATH including
experiments/arc2-mamba, e.g.

    PYTHONPATH=experiments/arc2-mamba ARC2_MAMBA_REPLAY=experiments/arc2-mamba/runs/mamba/htest_icl_pred.json \
        envs/mamba/bin/python scripts/arc2_eval_local.py --solver arcssm.solver:ReplaySolver --split train \
        --per-task-seconds 5 --out experiments/arc2-mamba/runs/mamba/host_htest_icl

ReplaySolver returns predictions computed earlier from label-free TaskViews (evaluate.py); MambaSolver runs a
trained arm (ARC2_MAMBA_MODEL, ARC2_MAMBA_TTT=0|1) inside the deadline.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from arc2.io.types import Candidate, TaskView


def _cand(task_id, i, grid, source, rank):
    return Candidate(task_id=task_id, test_index=i, grid=[list(map(int, r)) for r in grid], source=source,
                     evidence={"rank": rank})


class ReplaySolver:
    name = "arc2_mamba_replay"

    def __init__(self, seed: int = 0) -> None:
        path = os.environ["ARC2_MAMBA_REPLAY"]
        with open(path, "r", encoding="utf-8") as fh:
            self.pred = json.load(fh)
        self.name = f"arc2_mamba_replay:{os.path.basename(path)}"

    def propose(self, task: TaskView, deadline_s: float) -> list[Candidate]:
        out = []
        for i, attempts in enumerate(self.pred.get(task.task_id, [])):
            for rank, g in enumerate(attempts or []):
                if g:
                    out.append(_cand(task.task_id, i, g, self.name, rank))
        return out


class MambaSolver:
    name = "arc2_mamba"

    def __init__(self, seed: int = 0) -> None:
        import torch

        from arcssm.util import limit_threads

        limit_threads(int(os.environ.get("ARC2_MAMBA_THREADS", "2")))
        blob = torch.load(os.environ["ARC2_MAMBA_MODEL"], weights_only=False)
        from arcssm.models import build

        self.model = build(blob["cfg"])
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.ttt = os.environ.get("ARC2_MAMBA_TTT", "0") == "1"
        self.ttt_lr = float(os.environ.get("ARC2_MAMBA_TTT_LR", "1e-3"))
        self.max_side = int(os.environ.get("ARC2_MAMBA_TTT_MAX_SIDE", "15"))
        self.seed = seed
        self.name = f"arc2_mamba:{blob['cfg']['arch']}:{'ttt' if self.ttt else 'icl'}"

    def propose(self, task: TaskView, deadline_s: float) -> list[Candidate]:
        import torch

        from arcssm.infer import predict, ttt_predict

        pairs = [(np.asarray(p.input, dtype=np.int8), np.asarray(p.output, dtype=np.int8)) for p in task.train]
        tests = [np.asarray(g, dtype=np.int8) for g in task.test_inputs]
        big = max(max(max(g.shape) for p in pairs for g in p), max(max(g.shape) for g in tests))
        seed = int.from_bytes(task.task_id.encode()[:8].ljust(8, b"\0"), "little") % (2**32)
        if self.ttt and big <= self.max_side:
            torch.manual_seed(seed)
            preds, _ = ttt_predict(self.model, pairs, tests, task.task_id, np.random.default_rng(seed), lr=self.ttt_lr)
        else:
            preds = []
            for k, x in enumerate(tests):
                if time.monotonic() >= deadline_s:
                    break
                preds.append(predict(self.model, pairs, x, f"{task.task_id}:{k}"))
        out = []
        for i, p in enumerate(preds):
            for rank, g in enumerate((p or {}).get("top", [])):
                out.append(_cand(task.task_id, i, g, self.name, rank))
        return out
