"""Training-step throughput of the candidate configurations on this CPU (2 threads), written to bench.json.

    OMP_NUM_THREADS=2 nice -n 19 ../../envs/mamba/bin/python bench.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arcssm.configs import ARMS  # noqa: E402
from arcssm.models import build, n_params  # noqa: E402
from arcssm.util import atomic_write_json  # noqa: E402

torch.set_num_threads(2)


def bench(cfg, b, L, steps=3):
    torch.manual_seed(0)
    m = build(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    tok = torch.randint(0, 16, (b, L))
    row = torch.randint(0, 32, (b, L))
    col = torch.randint(0, 32, (b, L))
    role = torch.randint(0, 4, (b, L))
    times = []
    for i in range(steps + 1):
        t0 = time.process_time()
        logits = m(tok, row, col, role)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), tok.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i:
            times.append(time.process_time() - t0)
    return n_params(m), sum(times) / len(times)


def main():
    out = {"threads": torch.get_num_threads(), "torch": torch.__version__, "rows": []}
    lengths = [int(x) for x in os.environ.get("BENCH_L", "512,1024,2048").split(",")]
    for name, cfg in ARMS.items():
        for L in lengths:
            b = max(1, 4096 // L)
            p, cpu = bench(cfg, b, L)
            row = {"arm": name, "params": p, "batch": b, "L": L, "cpu_s_per_step": round(cpu, 3),
                   "tokens_per_cpu_s": round(b * L / cpu)}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
    atomic_write_json(HERE / "bench.json", out)


if __name__ == "__main__":
    main()
