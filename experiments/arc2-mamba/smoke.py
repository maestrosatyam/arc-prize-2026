"""GPU readiness smoke test for the SSM experiment; writes runs/smoke/smoke_summary.json and prints
"smoke: PASS" or "smoke: FAIL".

Checks: environment; the chunked scan against the sequential reference on the device; step decoding against the
parallel forward pass for every GPU arm (float32) and bfloat16-vs-float32 agreement; training throughput and peak
memory per arm at 1024 / 4096 / 8192 tokens per sequence (eager, blocks recomputed in backward); task-generation
throughput of the DataLoader workers; a short end-to-end run of every arm in parallel (gpu_launch.py) with a few
evaluation items per stage. Without CUDA (the CPU twin) the same steps run on CPU at toy sizes.

    ARC2_MAMBA_MAX_LEN=8192 python smoke.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from arcssm.configs import ARMS, GPU_ARMS  # noqa: E402
from arcssm.models import build, n_params, ssd_chunked, ssd_reference  # noqa: E402
from arcssm.util import atomic_write_json  # noqa: E402

OUT = HERE / "runs" / "smoke"
CUDA = torch.cuda.is_available()
DEV = torch.device("cuda:0" if CUDA else "cpu")


def env_report():
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = v.strip()
    except OSError:
        pass
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": CUDA, "n_gpu": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_mem_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                           for i in range(torch.cuda.device_count())],
            "bf16": bool(CUDA and torch.cuda.is_bf16_supported()), "cpu_count": os.cpu_count(), "mem": mem,
            "numpy": np.__version__}


def check_scan():
    g = torch.Generator(device="cpu").manual_seed(0)
    b, L, H, P, N = 2, 300, 4, 16, 32
    X = torch.randn(b, L, H, P, generator=g).to(DEV)
    A = (-torch.rand(b, L, H, generator=g) * 0.5).to(DEV)
    B = torch.randn(b, L, H, N, generator=g).to(DEV)
    C = torch.randn(b, L, H, N, generator=g).to(DEV)
    y_ref, h_ref = ssd_reference(X, A, B, C)
    out = {}
    for chunk in (32, 64):
        y, h = ssd_chunked(X, A, B, C, chunk=chunk)
        out[f"chunk{chunk}_max_abs_err_y"] = float((y - y_ref).abs().max())
        out[f"chunk{chunk}_max_abs_err_h"] = float((h - h_ref).abs().max())
    scale = float(y_ref.abs().max())
    out["y_scale"] = scale
    out["pass"] = all(v < 1e-3 * max(scale, 1.0) for k, v in out.items() if k.startswith("chunk"))
    return out


def check_step(arms):
    out = {}
    for name, cfg in arms.items():
        torch.manual_seed(0)
        m = build(cfg).to(DEV).eval()
        g = torch.Generator().manual_seed(1)
        L, k = 48, 30
        tok = torch.randint(0, 16, (2, L), generator=g).to(DEV)
        row = torch.randint(0, 32, (2, L), generator=g).to(DEV)
        col = torch.randint(0, 32, (2, L), generator=g).to(DEV)
        role = torch.randint(0, 4, (2, L), generator=g).to(DEV)
        with torch.no_grad():
            full = m(tok, row, col, role)
            logits, st = m(tok[:, :k], row[:, :k], col[:, :k], role[:, :k], return_state=True)
            steps = []
            for t in range(k, L):
                lt, st = m.step(tok[:, t], row[:, t], col[:, t], role[:, t], st)
                steps.append(lt)
            err = float((torch.stack(steps, 1) - full[:, k:]).abs().max())
            r = {"step_vs_forward_max_abs_err": err, "logit_scale": float(full.abs().max())}
            if CUDA:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    half = m(tok, row, col, role).float()
                r["bf16_rel_err"] = float((half - full).abs().max() / full.abs().max())
                r["bf16_argmax_agree"] = float((half.argmax(-1) == full.argmax(-1)).float().mean())
        r["pass"] = err < 1e-2 * max(r["logit_scale"], 1.0)
        out[name] = r
        del m
    return out


def bench(arms, lengths, tokens_per_batch, compile_at=None, steps=5, grad_ckpt=False):
    from arcssm.infer import amp

    out = []
    for name, cfg in arms.items():
        for L in lengths:
            for compiled in ([False, True] if compile_at == L else [False]):
                torch.manual_seed(0)
                m = build(cfg).to(DEV)
                m.grad_ckpt = grad_ckpt
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                fwd = torch.compile(m) if compiled else m
                b = max(1, tokens_per_batch // L)
                tok = torch.randint(0, 16, (b, L), device=DEV)
                row = torch.randint(0, 32, (b, L), device=DEV)
                rec = {"arm": name, "params": n_params(m), "L": L, "batch": b, "compiled": compiled, "grad_ckpt": grad_ckpt}
                try:
                    if CUDA:
                        torch.cuda.reset_peak_memory_stats(DEV)
                    t_first = time.perf_counter()
                    times = []
                    for i in range(steps + 2):
                        t0 = time.perf_counter()
                        with amp(DEV):
                            logits = fwd(tok, row, row, torch.zeros_like(tok))
                        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 16), tok.reshape(-1))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        if CUDA:
                            torch.cuda.synchronize(DEV)
                        if i == 0:
                            rec["first_step_s"] = round(time.perf_counter() - t_first, 2)
                        if i >= 2:
                            times.append(time.perf_counter() - t0)
                    rec["s_per_step"] = round(float(np.median(times)), 4)
                    rec["tokens_per_s"] = round(b * L / float(np.median(times)))
                    if CUDA:
                        rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(DEV) / 2**30, 2)
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                out.append(rec)
                print(json.dumps(rec), flush=True)
                del m, opt, fwd
                if CUDA:
                    torch.cuda.empty_cache()
    return out


def data_bench(worker_counts, seconds, tokens_per_batch, max_len):
    from arcssm.data import load_split, load_training
    from train import _Stream

    split, tasks = load_split(), load_training()
    real = {t: tasks[t] for t in split["meta_train"]}
    out = []
    for w in worker_counts:
        dl = torch.utils.data.DataLoader(_Stream(real, 0, 0.5, max_len, 20, 30, tokens_per_batch), batch_size=None,
                                         num_workers=w, prefetch_factor=2 if w else None)
        it = iter(dl)
        next(it)
        t0, n_b, n_tok = time.perf_counter(), 0, 0
        while time.perf_counter() - t0 < seconds:
            _, nt, _ = next(it)
            n_b += 1
            n_tok += nt
        dt = time.perf_counter() - t0
        out.append({"workers": w, "batches_per_s": round(n_b / dt, 2), "tokens_per_s": round(n_tok / dt)})
        print(json.dumps(out[-1]), flush=True)
        del it, dl
    return out


def mini_pipeline():
    if CUDA:
        n = torch.cuda.device_count()
        arms = ["transformer_g", "hybrid_g", "mamba_g", "hybrid_g:1"][: max(1, n)]
        cmd = [sys.executable, str(HERE / "gpu_launch.py"), "--group", "smoke", "--arms", ",".join(arms), "--",
               "--train-tokens", "4000000", "--train-seconds", "300", "--deadline-s", "900", "--limit", "4",
               "--tune-tasks", "2", "--val-every", "1000000000"]
        runs = [HERE / "runs" / "smoke" / (a.replace(":", "_s") if ":" in a else f"{a}_s0") for a in arms]
    else:
        cmd = [sys.executable, str(HERE / "gpu_pipeline.py"), "--arm", "mamba", "--run-dir",
               str(HERE / "runs" / "smoke" / "mamba_s0"), "--device", "cpu", "--train-seconds", "20",
               "--train-tokens", "80000", "--deadline-s", "300", "--limit", "2", "--tune-tasks", "1", "--max-len", "2048",
               "--tokens-per-batch", "4096", "--max-side-gen", "10", "--max-side-real", "12", "--workers", "1",
               "--ttt-grid", "0,2", "--ttt-bs", "2", "--val-every", "1000000000"]
        runs = [HERE / "runs" / "smoke" / "mamba_s0"]
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=HERE).returncode
    out = {"cmd": cmd, "returncode": rc, "seconds": round(time.monotonic() - t0, 1), "runs": {}}
    ok = rc == 0
    if not CUDA:  # dev_eval.py on the CPU model just trained (finds <arm>/model.pt under the smoke directory)
        dev_cmd = [sys.executable, str(HERE / "dev_eval.py"), "--models", "mamba_s0", "--input-root",
                   str(HERE / "runs" / "smoke"), "--out", str(HERE / "runs" / "smoke" / "dev_eval"), "--device", "cpu",
                   "--deadline-s", "90", "--ttt-grid", "0,2", "--ttt-bs", "2", "--n-aug", "2"]
        drc = subprocess.run(dev_cmd, cwd=HERE).returncode
        ds = HERE / "runs" / "smoke" / "dev_eval" / "dev_summary.json"
        out["dev_eval"] = {"returncode": drc, "summary": json.loads(ds.read_text()) if ds.exists() else None}
        ok = ok and drc == 0 and ds.exists()
    for run in runs:
        r = {}
        for f in ("model.pt", "train_summary.json", "tf_eval.json", "ttt_lr.json"):
            r[f] = (run / f).exists()
        for f in ("eval_icl.jsonl", "eval_ttt.jsonl"):
            r[f] = len((run / f).read_text().splitlines()) if (run / f).exists() else 0
        errors = []
        if (run / "pipeline.jsonl").exists():
            for line in (run / "pipeline.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row.get("event") in ("error", "abort"):
                    errors.append({k: row.get(k) for k in ("stage", "reason", "traceback")})
        r["errors"] = errors
        if (run / "train_summary.json").exists():
            s = json.loads((run / "train_summary.json").read_text())
            r["train"] = {"steps": s["state"]["step"], "tokens": s["state"]["tokens"],
                          "wall": round(s["state"]["wall_used"]), "final_val": s["final_val"],
                          "device_name": s.get("device_name")}
        r["pass"] = (all(r[f] for f in ("model.pt", "train_summary.json", "tf_eval.json", "ttt_lr.json"))
                     and r["eval_icl.jsonl"] > 0 and r["eval_ttt.jsonl"] > 0 and not errors)
        ok = ok and r["pass"]
        out["runs"][run.name] = r
    out["pass"] = ok
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "max_len_env": os.environ.get("ARC2_MAMBA_MAX_LEN")}
    steps = [("env", env_report), ("scan", check_scan)]
    if CUDA:
        steps += [("step", lambda: check_step(GPU_ARMS)),
                  ("bench", lambda: bench(GPU_ARMS, [1024, 4096, 8192], 32768, grad_ckpt=True)),
                  ("data", lambda: data_bench([2, 4, 8], 20, 32768, 8192))]
    else:
        small = {k: dataclasses.replace(v, d_model=64, n_layers=2, n_heads=4, headdim=16, d_state=16, chunk=16,
                                        attn_layers=(1,) if v.arch == "hybrid" else ())
                 for k, v in GPU_ARMS.items()}
        steps += [("step", lambda: check_step(small)),
                  ("bench", lambda: bench({"mamba": ARMS["mamba"]}, [256], 2048, compile_at=None, steps=2)),
                  ("data", lambda: data_bench([1], 5, 4096, 2048))]
    steps.append(("mini_pipeline", mini_pipeline))
    ok = True
    for name, fn in steps:
        t0 = time.monotonic()
        try:
            res = fn()
        except Exception:  # noqa: BLE001
            res = {"pass": False, "traceback": traceback.format_exc()}
        summary[name] = res
        summary[f"{name}_seconds"] = round(time.monotonic() - t0, 1)
        if isinstance(res, dict) and res.get("pass") is False:
            ok = False
        if name == "step" and isinstance(res, dict) and not all(r.get("pass") for r in res.values()):
            ok = False
        if name == "bench" and any("error" in r and not r["compiled"] for r in res):
            ok = False  # a torch.compile failure is recorded but optional
        print(f"== {name} ({summary[f'{name}_seconds']} s): " + json.dumps(res)[:3000], flush=True)
        atomic_write_json(OUT / "smoke_summary.json", summary)
    summary["pass"] = ok
    summary["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(OUT / "smoke_summary.json", summary)
    print("smoke: PASS" if ok else "smoke: FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
