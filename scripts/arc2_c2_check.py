#!/usr/bin/env python3
"""Validity floor of experiment C2 (ARC-2) on one kernel's archive: C2_RULE.md section 4, items 2, 3, 4, 5 and 7.
CPU only; the queue's post-command (exit 1 on any failed item).

  item 2  complete: every expected task has a record with complete == true and two sample records, no sample with
          status error or deadline, a transcript with reply texts per sample; _run.json tasks_complete == the
          number of expected tasks and invalid empty (the tasks a problem or the invalid list names are "report
          only": the floor scope of 2026-10-29, section 4's "affected tasks")
  item 3  schema: the records load with eval.arc2_bank.load_q27 (arc2_q27_direct.v1); each task's sample indices
          are exactly {start, start + 1} (start 2 on the band tasks, else 0); no sealed id in q27d/; the tasks at
          start 2 are among the bank's 30 and no task at start 0 is; the two kernels' task lists are disjoint
  item 4  caps: every answer request max_tokens 49,152 (36,864 on a context-overflow retry) with seed 1000 k + 7,
          every rescue 8,192 with seed 1000 k + 8 (k the sample index); no sample ended by the deadline or the kill
          (deadline-ended samples list their tasks: "report only")
  item 5  settings: _config.json equals the bank's on the sampling and server keys, the model files and the
          wheelhouse; _run.json's run settings and _environment.json's wheelhouse pins likewise
  item 7  sealed block: when the kernel carries sealed ids, q27d_sealed/ holds exactly their records (complete, two
          samples, transcripts) with a _sealed.json manifest whose sha256 match, no solutions copy; the loader
          refuses it; no sealed block when none is expected. No grid or accuracy of a sealed task is read or printed.

Usage: arc2_c2_check.py ARCHIVE_DIR --kernel c2a|c2b [--provenance PATH] [--tasks JSON]
                        [--bank-config experiments/runs/kaggle-A2-q27-bank-v1/q27d/_config.json] [--json OUT]

The expected task set comes from --tasks (a JSON file or literal of the provenance's shape: {"task_ids": [...],
"sample_start": {id: 2}, "sealed_ids": [...]}), else from kaggle/arc2/bank/<kernel>/PROVENANCE.json "task_set".
The other kernel's provenance (when present) gives the disjointness check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc2_bank import (Q27_BANK, Q27_SEALED_DIRNAME, SEALED_MANIFEST, SealedAccessError,  # noqa: E402
                            is_sealed_dir, load_q27)

SCHEMA = "arc2_c2_check.v1"
KERNELS = ("c2a", "c2b")
Q27_SCHEMA = "arc2_q27_direct.v1"
BANK_CONFIG = Q27_BANK / "_config.json"
CONFIG_KEYS = ("thinking_token_budget", "max_tokens", "rescue_tokens", "template_kwargs", "temperature", "top_p",
               "top_k", "max_model_len", "server_ladder", "concurrency", "max_num_seqs")
FILES_KEYS = ("model_glob", "overlay", "wheelhouse_dir")
RUN_META_KEYS = ("temperature", "top_p", "top_k", "template_kwargs", "max_tokens", "rescue_tokens",
                 "thinking_token_budget", "concurrency")
N_SAMPLES = 2
ANSWER_MAX_TOKENS = (49152, 36864)      # 36,864 only on a context-overflow retry
RESCUE_MAX_TOKENS = 8192
SEED_BASE, RESCUE_SEED_OFFSET = 7, 1    # seed 1000 k + 7, rescue seed + 1
BAD_STATUSES = ("error", "deadline")


FLOOR_SCOPE = ("2026-10-29 14:18 IST, C2 section 4 -- only the tasks a failed item "
               "names are report only (section 4's \"affected tasks\"), not the failed kernel's whole list (section 7); "
               "the gate reads on the rest")


def _task_of(problem: str) -> str | None:
    """The task id a per-task problem string names ("<task>: ..." or "<task>#<k>: ..."), else None."""
    head = problem.split(":", 1)[0].split("#", 1)[0].strip()
    return head if head and " " not in head and not head.startswith("_") else None


def provenance_path(kernel: str) -> Path:
    return ROOT / "kaggle" / "arc2" / "bank" / kernel / "PROVENANCE.json"


def _normalise_task_set(doc: dict) -> dict:
    return {"task_ids": [str(t) for t in doc.get("task_ids") or []],
            "sample_start": {str(t): int(v) for t, v in (doc.get("sample_start") or {}).items()},
            "sealed_ids": [str(t) for t in doc.get("sealed_ids") or []]}


def load_task_set(spec: str | None, provenance: str | None, kernel: str) -> tuple[dict, str]:
    """({task_ids, sample_start, sealed_ids}, source) from --tasks (file or literal) or the kernel's provenance."""
    if spec:
        p = Path(spec)
        doc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else json.loads(spec)
        doc = doc.get("task_set", doc) if isinstance(doc, dict) and "task_ids" not in doc else doc
        return _normalise_task_set(doc), str(p) if p.is_file() else "--tasks literal"
    p = Path(provenance) if provenance else provenance_path(kernel)
    if not p.is_file():
        raise FileNotFoundError(f"provenance {p} not found: give --provenance or --tasks")
    doc = json.loads(p.read_text(encoding="utf-8")).get("task_set")
    if not doc or not doc.get("task_ids"):
        raise ValueError(f"{p} has no task_set.task_ids")
    return _normalise_task_set(doc), str(p)


def other_task_set(kernel: str, provenance: str | None) -> dict | None:
    """The other kernel's task set from its provenance beside this one's (default location), when present."""
    other = next((k for k in KERNELS if k != kernel), None)
    if other is None:
        return None
    p = (Path(provenance).parent.parent / other / "PROVENANCE.json") if provenance else provenance_path(other)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8")).get("task_set")
    except ValueError:
        return None
    return _normalise_task_set(doc) if doc else None


def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_deadline(s: dict) -> bool:
    return s.get("status") == "deadline" or str(s.get("status_detail") or "").startswith("skipped")


def _expected_indices(t: str, start: dict[str, int]) -> set[int]:
    k0 = start.get(t, 0)
    return {k0 + i for i in range(N_SAMPLES)}


def check_records(d: Path, ids: list[str], start: dict[str, int], *, transcripts: bool = True
                  ) -> tuple[dict[str, dict], list[str], list[str], list[str]]:
    """Per-task records under d for ids: (records, completeness problems, index problems, request problems)."""
    records: dict[str, dict] = {}
    complete, indices, requests = [], [], []
    for t in ids:
        p = d / f"{t}.json"
        doc = _read(p) if p.is_file() else None
        if doc is None:
            complete.append(f"{t}: no record" if not p.is_file() else f"{t}: record unreadable")
            continue
        records[t] = doc
        if doc.get("schema") != Q27_SCHEMA:
            indices.append(f"{t}: schema {doc.get('schema')!r}")
        samples = [s for s in doc.get("samples") or [] if isinstance(s, dict)]
        if doc.get("complete") is not True:
            complete.append(f"{t}: complete {doc.get('complete')!r} (stop_reason {doc.get('stop_reason')!r})")
        if doc.get("n_sample_records") != N_SAMPLES or len(samples) != N_SAMPLES:
            complete.append(f"{t}: {doc.get('n_sample_records')} sample records ({len(samples)} samples), want {N_SAMPLES}")
        got = {int(s.get("sample") or 0) for s in samples}
        want = _expected_indices(t, start)
        if got != want:
            indices.append(f"{t}: sample indices {sorted(got)}, want {sorted(want)}")
        status_keys = {int(k) for k in (doc.get("sample_status") or {})} if doc.get("sample_status") else set()
        if status_keys and status_keys != got:
            indices.append(f"{t}: sample_status keys {sorted(status_keys)} differ from the samples {sorted(got)}")
        for s in samples:
            k = int(s.get("sample") or 0)
            if s.get("status") in BAD_STATUSES or _is_deadline(s):
                complete.append(f"{t}#{k}: status {s.get('status')!r} ({str(s.get('status_detail') or '')[:60]})")
            if transcripts:
                tp = d / "transcripts" / f"{t}.{k}.json"
                tdoc = _read(tp) if tp.is_file() else None
                if tdoc is None:
                    complete.append(f"{t}#{k}: transcript {tp.name} missing or unreadable")
                elif s.get("n_requests") and not (isinstance(tdoc.get("texts"), list) and tdoc["texts"]):
                    complete.append(f"{t}#{k}: transcript without reply texts")
            reqs = s.get("requests") or []
            for i, r in enumerate(reqs):
                kind, mt, seed = r.get("kind"), r.get("max_tokens"), r.get("seed")
                if kind == "rescue":
                    if mt != RESCUE_MAX_TOKENS:
                        requests.append(f"{t}#{k} rescue max_tokens {mt}, want {RESCUE_MAX_TOKENS}")
                    if seed != 1000 * k + SEED_BASE + RESCUE_SEED_OFFSET:
                        requests.append(f"{t}#{k} rescue seed {seed}, want {1000 * k + SEED_BASE + RESCUE_SEED_OFFSET}")
                elif kind == "answer":
                    if mt not in ANSWER_MAX_TOKENS:
                        requests.append(f"{t}#{k} answer max_tokens {mt}, want {ANSWER_MAX_TOKENS[0]} "
                                        f"({ANSWER_MAX_TOKENS[1]} on a context-overflow retry)")
                    elif mt == ANSWER_MAX_TOKENS[1] and not any(x.get("context_overflow") for x in reqs[:i]):
                        requests.append(f"{t}#{k} answer max_tokens {mt} without a context-overflow retry before it")
                    if seed != 1000 * k + SEED_BASE:
                        requests.append(f"{t}#{k} answer seed {seed}, want {1000 * k + SEED_BASE}")
                else:
                    requests.append(f"{t}#{k} request kind {kind!r}")
    return records, complete, indices, requests


def sealed_ids_in(d: Path, sealed: set[str]) -> list[str]:
    """Sealed ids that appear in an unsealed directory: record files, samples.jsonl lines, transcripts."""
    found = set()
    for t in sealed:
        if (d / f"{t}.json").is_file() or any((d / "transcripts").glob(f"{t}.*.json")):
            found.add(t)
    jl = d / "samples.jsonl"
    if jl.is_file():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if s.get("task_id") in sealed:
                found.add(s["task_id"])
    return sorted(found)


def check_archive(archive: Path, kernel: str, task_set: dict, *, bank_config: Path = BANK_CONFIG,
                  other: dict | None = None) -> dict:
    archive = Path(archive)
    q27d = archive / "q27d"
    sealed_dir = archive / Q27_SEALED_DIRNAME
    ids = task_set["task_ids"]
    start = task_set["sample_start"]
    sealed = set(task_set["sealed_ids"])
    items: list[dict] = []
    report_only: set[str] = set()

    def add(n: int, name: str, problems: list[str], ok_reason: str, details: dict | None = None) -> None:
        items.append({"item": n, "name": name, "pass": not problems,
                      "reason": ok_reason if not problems else f"{len(problems)} problem(s): " + "; ".join(problems[:12])
                      + (" ..." if len(problems) > 12 else ""), "problems": problems, "details": details or {}})

    if not q27d.is_dir():
        for n, name in ((2, "complete"), (3, "schema"), (4, "caps"), (5, "settings")):
            add(n, name, [f"{q27d} is not a directory"], "")
        add(7, "sealed", [f"{q27d} is not a directory"], "")
        return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "archive": str(archive),
                "kernel": kernel, "rule": "experiments/arc2-bank/C2_RULE.md section 4 (frozen 24ef327)",
                "task_set": {"n_tasks": len(ids), "n_at_start_2": sum(1 for t in ids if start.get(t, 0) == 2),
                             "n_sealed": len(sealed)},
                "bank_config": str(bank_config), "items": items, "pass": False, "report_only_tasks": []}

    records, p_complete, p_indices, p_requests = check_records(q27d, ids, start)
    run = _read(q27d / "_run.json") if (q27d / "_run.json").is_file() else None

    # item 2: complete
    p2 = list(p_complete)
    if run is None:
        p2.append("_run.json missing or unreadable")
    else:
        if run.get("tasks_complete") != len(ids):
            p2.append(f"_run.json tasks_complete {run.get('tasks_complete')}, want {len(ids)}")
        if run.get("invalid"):
            p2.append(f"_run.json invalid: {sorted(run['invalid'])[:8]}")
    # the affected tasks (FLOOR_SCOPE): every task a completeness problem or the run's invalid list names
    affected2 = {t for t in (_task_of(x) for x in p_complete) if t in set(ids)}
    affected2 |= {t for t in ((run or {}).get("invalid") or {}) if t in set(ids)}
    report_only.update(affected2)
    n_tr = sum(1 for _ in (q27d / "transcripts").glob("*.json")) if (q27d / "transcripts").is_dir() else 0
    add(2, "complete", p2, f"{len(records)} tasks complete with {N_SAMPLES} sample records each, {n_tr} transcripts, "
        f"_run.json tasks_complete {run.get('tasks_complete') if run else None}, invalid empty",
        {"n_tasks": len(ids), "n_records": len(records), "n_transcripts": n_tr,
         "invalid": (run or {}).get("invalid") or {}, "report_only": sorted(affected2)})

    # item 3: schema, loader, indices, sealed ids, band ids, disjointness
    p3 = list(p_indices)
    try:
        loaded, _ = load_q27(q27d)
        n_loaded = sum(len(v) for v in loaded.values())
        missing_loaded = [t for t in ids if t in records and any(s.get("grids") for s in records[t].get("samples") or [])
                          and t not in loaded]
        if missing_loaded:
            p3.append(f"load_q27 holds no samples for {missing_loaded[:8]}")
    except (SealedAccessError, ValueError, OSError) as exc:
        n_loaded = 0
        p3.append(f"load_q27 failed: {exc}")
    if run is not None:
        if set(run.get("task_ids") or []) != set(ids):
            extra = sorted(set(run.get("task_ids") or []) - set(ids))
            miss = sorted(set(ids) - set(run.get("task_ids") or []))
            p3.append(f"_run.json task_ids differ from the expected list (extra {extra[:6]}, missing {miss[:6]})")
        if run.get("n_samples_per_task") != N_SAMPLES:
            p3.append(f"_run.json n_samples_per_task {run.get('n_samples_per_task')}, want {N_SAMPLES}")
    found_sealed = sealed_ids_in(q27d, sealed)
    if found_sealed:
        p3.append(f"sealed ids in q27d/: {found_sealed[:8]}")
    bank_cfg = _read(bank_config) if Path(bank_config).is_file() else None
    bank_ids = set((bank_cfg or {}).get("task_ids") or [])
    at2 = {t for t in ids if start.get(t, 0) == 2}
    at0 = {t for t in ids if start.get(t, 0) == 0}
    odd = {t for t in ids if start.get(t, 0) not in (0, 2)}
    if odd:
        p3.append(f"sample_start not 0 or 2 for {sorted(odd)[:6]}")
    if bank_ids:
        if at2 - bank_ids:
            p3.append(f"tasks at sample start 2 outside the bank's 30: {sorted(at2 - bank_ids)[:6]}")
        if at0 & bank_ids:
            p3.append(f"bank band tasks at sample start 0 (would duplicate the bank's samples): {sorted(at0 & bank_ids)[:6]}")
        if sealed & bank_ids:
            p3.append(f"sealed ids among the bank's 30: {sorted(sealed & bank_ids)[:6]}")
    if sealed & set(ids):
        p3.append(f"sealed ids in the task list: {sorted(sealed & set(ids))[:6]}")
    if other is not None:
        both = set(ids) & set(other["task_ids"])
        if both:
            p3.append(f"tasks in both kernels: {sorted(both)[:6]}")
        both_s = (sealed | set(ids)) & (set(other["sealed_ids"]) | set(other["task_ids"]))
        if both_s - both:
            p3.append(f"ids in both kernels (sealed or scored): {sorted(both_s - both)[:6]}")
    add(3, "schema", p3, f"load_q27 holds {n_loaded} samples; indices {{start, start+1}} on {len(ids)} tasks "
        f"({len(at2)} at 2, {len(at0)} at 0); no sealed id in q27d/"
        + (f"; disjoint from the other kernel's {len(other['task_ids'])} tasks" if other else ""),
        {"n_loaded_samples": n_loaded, "n_at_start_2": len(at2), "n_at_start_0": len(at0),
         "bank_ids_known": bool(bank_ids), "other_kernel_checked": other is not None})

    # item 4: caps, seeds, deadline, kill
    p4 = list(p_requests)
    deadline_tasks = sorted(t for t, doc in records.items()
                            if any(_is_deadline(s) for s in doc.get("samples") or [] if isinstance(s, dict)))
    if deadline_tasks:
        report_only.update(deadline_tasks)
        p4.append(f"deadline-ended samples on {len(deadline_tasks)} task(s), report only: {deadline_tasks[:10]}")
    killed = sorted(t for t, doc in records.items() if doc.get("stop_reason") == "killed")
    if killed:
        report_only.update(killed)
        p4.append(f"killed before every sample on {len(killed)} task(s), report only: {killed[:10]}")
    meta = (run or {}).get("run") or {}
    for flag in ("deadline_reached", "hard_stop"):
        if meta.get(flag):
            p4.append(f"_run.json run.{flag} is true")
    n_req = sum(len(s.get("requests") or []) for doc in records.values() for s in doc.get("samples") or [])
    n_rescue = sum(1 for doc in records.values() for s in doc.get("samples") or []
                   for r in s.get("requests") or [] if r.get("kind") == "rescue")
    add(4, "caps", p4, f"{n_req} requests ({n_rescue} rescues) at the bank's caps and seeds; no deadline-ended or "
        f"killed sample; deadline_reached {meta.get('deadline_reached')}, hard_stop {meta.get('hard_stop')}",
        {"n_requests": n_req, "n_rescues": n_rescue, "deadline_tasks": deadline_tasks, "killed_tasks": killed})

    # item 5: settings
    p5 = []
    cfg = _read(q27d / "_config.json") if (q27d / "_config.json").is_file() else None
    if bank_cfg is None:
        p5.append(f"bank config {bank_config} missing or unreadable")
    if cfg is None:
        p5.append("_config.json missing or unreadable")
    if bank_cfg is not None and cfg is not None:
        for k in CONFIG_KEYS + FILES_KEYS:
            if cfg.get(k) != bank_cfg.get(k):
                p5.append(f"_config.json {k}: {json.dumps(cfg.get(k))[:80]} differs from the bank's "
                          f"{json.dumps(bank_cfg.get(k))[:80]}")
        for k in RUN_META_KEYS:
            if k in meta and meta.get(k) != bank_cfg.get(k):
                p5.append(f"_run.json run.{k}: {json.dumps(meta.get(k))[:60]} differs from the bank's "
                          f"{json.dumps(bank_cfg.get(k))[:60]}")
    env = _read(q27d / "_environment.json") if (q27d / "_environment.json").is_file() else None
    bank_env_p = Path(bank_config).parent / "_environment.json"
    bank_env = _read(bank_env_p) if bank_env_p.is_file() else None
    if env is not None and bank_env is not None:
        pins = sorted((env.get("wheelhouse_lock") or {}).get("pins") or [])
        bpins = sorted((bank_env.get("wheelhouse_lock") or {}).get("pins") or [])
        if pins != bpins:
            p5.append(f"_environment.json wheelhouse pins {pins} differ from the bank's {bpins}")
        if env.get("server") != bank_env.get("server"):
            p5.append(f"_environment.json server {env.get('server')!r} differs from the bank's {bank_env.get('server')!r}")
    elif env is None:
        p5.append("_environment.json missing or unreadable")
    add(5, "settings", p5, f"_config.json equals the bank's on {', '.join(CONFIG_KEYS + FILES_KEYS)}; run settings and "
        f"wheelhouse pins {'checked' if bank_env is not None else 'not compared (no bank environment file)'}",
        {"config_keys": list(CONFIG_KEYS + FILES_KEYS), "environment_compared": env is not None and bank_env is not None})

    # item 7: sealed block
    p7 = []
    det7: dict = {"expected": sorted(sealed), "present": sealed_dir.is_dir()}
    if not sealed:
        if sealed_dir.is_dir() and any(p.suffix == ".json" and not p.name.startswith("_") for p in sealed_dir.iterdir()):
            p7.append(f"unexpected sealed block {sealed_dir} (no sealed ids expected)")
        add(7, "sealed", p7, "no sealed block expected and none present", det7)
    else:
        if not sealed_dir.is_dir():
            p7.append(f"{sealed_dir} missing")
        else:
            manifest = _read(sealed_dir / SEALED_MANIFEST) if (sealed_dir / SEALED_MANIFEST).is_file() else None
            if manifest is None:
                p7.append(f"{SEALED_MANIFEST} missing or unreadable")
            else:
                if set(manifest.get("task_ids") or []) != sealed:
                    p7.append(f"manifest task_ids differ from the expected sealed ids "
                              f"(extra {sorted(set(manifest.get('task_ids') or []) - sealed)[:6]}, "
                              f"missing {sorted(sealed - set(manifest.get('task_ids') or []))[:6]})")
                # the kernel's manifest keys every file by its path under the block ("<task>.json",
                # "transcripts/<task>.<k>.json", "samples.jsonl", "_run.json"); a task-id key is accepted too
                shas = manifest.get("sha256") or {}
                for t in sorted(sealed):
                    f = sealed_dir / f"{t}.json"
                    if not f.is_file():
                        p7.append(f"sealed record {t}.json missing")
                        continue
                    want = shas.get(f"{t}.json") or shas.get(t)
                    if not want:
                        p7.append(f"manifest has no sha256 for {t}")
                    elif hashlib.sha256(f.read_bytes()).hexdigest() != want:
                        p7.append(f"sha256 of {t}.json differs from the manifest")
                for rel, want in sorted(shas.items()):
                    f = sealed_dir / rel if "/" in rel or "." in rel else sealed_dir / f"{rel}.json"
                    if not f.is_file():
                        p7.append(f"manifest lists {rel}, not in the block")
                    elif hashlib.sha256(f.read_bytes()).hexdigest() != want:
                        p7.append(f"sha256 of {rel} differs from the manifest")
                det7["n_manifest_files"] = len(shas)
            extra = sorted(p.stem for p in sealed_dir.glob("*.json") if not p.name.startswith("_") and p.stem not in sealed)
            if extra:
                p7.append(f"files of unexpected tasks in the sealed block: {extra[:6]}")
            sol = sorted(str(p.relative_to(sealed_dir)) for p in sealed_dir.rglob("*") if "solutions" in p.name.lower())
            if sol:
                p7.append(f"solutions copy in the sealed block: {sol[:4]}")
            if not is_sealed_dir(sealed_dir):
                p7.append("the loader does not recognise the sealed block as sealed")
            else:
                try:
                    load_q27(sealed_dir)
                    p7.append("load_q27 read the sealed block without final_pick")
                except SealedAccessError:
                    pass
                except (ValueError, OSError) as exc:
                    p7.append(f"load_q27 on the sealed block: {exc}")
            # completeness of the sealed records without reading grids or truth: flags, indices, statuses, transcripts
            _, s_complete, s_indices, s_requests = check_records(sealed_dir, sorted(sealed), {})
            p7 += [f"sealed {x}" for x in s_complete + s_indices + s_requests]
            det7["n_files"] = sum(1 for p in sealed_dir.glob("*.json") if not p.name.startswith("_"))
        add(7, "sealed", p7, f"{len(sealed)} sealed records under {sealed_dir.name}/ match {SEALED_MANIFEST}; complete, "
            f"indices 0 and 1, transcripts present; no solutions copy; the loader refuses the block", det7)

    ok = all(it["pass"] for it in items)
    return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "archive": str(archive),
            "kernel": kernel, "rule": "experiments/arc2-bank/C2_RULE.md section 4 (frozen 24ef327)",
            "task_set": {"n_tasks": len(ids), "n_at_start_2": len(at2), "n_sealed": len(sealed)},
            "bank_config": str(bank_config), "items": items, "pass": ok, "report_only_tasks": sorted(report_only),
            "floor_scope": FLOOR_SCOPE}


def print_report(rep: dict) -> None:
    for it in rep["items"]:
        print(f"item {it['item']} {it['name']}: {'PASS' if it['pass'] else 'FAIL'} {it['reason']}")
    if rep["report_only_tasks"]:
        print(f"report only ({len(rep['report_only_tasks'])} tasks): {' '.join(rep['report_only_tasks'])}")
    n = sum(1 for it in rep["items"] if it["pass"])
    print(f"C2 check {rep['kernel']} {rep['archive']}: {'PASS' if rep['pass'] else 'FAIL'} ({n} of {len(rep['items'])} items)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", help="the kernel's archive (holds q27d/ and, for c2a, q27d_sealed/)")
    ap.add_argument("--kernel", required=True, choices=KERNELS)
    ap.add_argument("--provenance", default="", help="PROVENANCE.json with task_set (default: the kernel's)")
    ap.add_argument("--tasks", default="", help="JSON file or literal {task_ids, sample_start, sealed_ids} (overrides)")
    ap.add_argument("--bank-config", default=str(BANK_CONFIG), help="the bank's _config.json to compare with")
    ap.add_argument("--json", default="", help="write the report here")
    args = ap.parse_args(argv)
    try:
        task_set, source = load_task_set(args.tasks or None, args.provenance or None, args.kernel)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    other = None if args.tasks else other_task_set(args.kernel, args.provenance or None)
    rep = check_archive(Path(args.archive), args.kernel, task_set, bank_config=Path(args.bank_config), other=other)
    rep["task_set"]["source"] = source
    print(f"C2 validity floor, kernel {args.kernel}: {len(task_set['task_ids'])} tasks "
          f"({rep['task_set']['n_at_start_2']} at sample start 2), {len(task_set['sealed_ids'])} sealed ids; task set from "
          f"{source}; bank config {args.bank_config}")
    print_report(rep)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
