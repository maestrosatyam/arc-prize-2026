"""Phase 2 of the ARC-AGI-2 hybrid notebook with the verify-and-refine harness (arc_harness, solver "vr") in place of
version 7's direct mode.

    python hybrid_vr.py run <config.json>      as hybrid_p2.py run: install wait, server ladder, smoke request and
                                               server shutdown are hybrid_p2's; the answering loop is the harness

The harness asks for the test grids and a transform(grid) program in one reply, runs the program on the
demonstrations (the verification bit), rescues replies cut in their reasoning, gives an unverified first sample one
feedback round, and spends what is left on second samples; selection puts verified programs first. Every task in
the challenges file is written: tasks outside the route keep B0's entry, routed tasks get B0's pool merged with the
harness's candidates (cfg["vr"]["merge"]). submission.json is rewritten atomically every write_every_s seconds, so a
kill leaves the last valid file.

Scheduling (cfg["route"] is shortest task JSON first): group 0 = the shortest ceil(n/4) tasks, every stage (first
sample, repair, second samples) before group 1's first samples; within a group stage-major, shortest first. The
harness's own deadline guard clips every request to what the per-stream rate allows before the collection deadline.

The model's programs run under cfg["vr"]["sandbox_python"] (the image's Python, with numpy and scipy): this process
runs under the serving Python, whose numpy lives only in the vLLM site.
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PLACEHOLDER = [[0]]


def b0_pool(path, tasks):
    """B0's submission as the harness's external pool {task: [[attempt_1, attempt_2] per test]}; placeholders out."""
    try:
        with open(path) as f:
            sub = json.load(f)
    except Exception:
        return {}
    pool = {}
    for tid, task in tasks.items():
        ent = sub.get(tid)
        if not isinstance(ent, list):
            continue
        per_test = []
        for j in range(len(task["test"])):
            e = ent[j] if j < len(ent) and isinstance(ent[j], dict) else {}
            per_test.append([g for g in (e.get("attempt_1"), e.get("attempt_2")) if g is not None and g != PLACEHOLDER])
        pool[tid] = per_test
    return pool


def read_samples(path):
    """The harness's samples.jsonl by task; a line cut by a kill mid-write is skipped."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                s = json.loads(line)
            except ValueError:
                continue
            out.setdefault(s["task_id"], []).append(s)
    return out


def full_submission(tasks, route, samples, pool, b0_path, merge):
    """Every task: B0's entry; routed tasks: the harness's two attempts over B0's pool."""
    from arc_harness.selection import two_attempts
    try:
        with open(b0_path) as f:
            sub = json.load(f)
    except Exception:
        sub = {}
    sub = {t: sub.get(t) for t in tasks}
    for tid in route:
        if tid in tasks:
            sub[tid] = two_attempts(tasks[tid], samples.get(tid, []), "vr", pool.get(tid), merge, True)
    return sub


def run_vr(cfg, st=None):
    """hybrid_p2.run_routed's replacement; returns the summary keys hybrid_p2.run reports."""
    vr = cfg["vr"]
    os.environ["ARC_SANDBOX_PYTHON"] = vr["sandbox_python"]
    from arc_harness.budget import Budget, Caps
    from arc_harness.data import load_kaggle
    from arc_harness.llm import VLLMChat
    from arc_harness.solvers import Config, Runner
    from arc_harness.submission import write_submission

    t0 = time.time()
    tasks = load_kaggle(cfg["challenges"])
    route = [t for t in cfg["route"] if t in tasks]
    quarter = set(route[:-(-len(route) // 4)])
    pool = b0_pool(cfg["b0_submission"], tasks)
    out = os.path.join(cfg["out_dir"], "vr")
    llm = VLLMChat(cfg["base_url"], cfg["model"], top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20),
                   template_kwargs=cfg.get("template_kwargs"))
    caps = Caps.from_mean(vr["mean_tokens"], vr["caps"])
    budget = Budget(vr["mean_tokens"], max(1, len(route)), deadline=float(cfg["collect_end_epoch"]),
                    reserve_s=vr["reserve_s"], tokens_per_s=vr["tokens_per_s"],
                    max_in_flight=int(cfg.get("concurrency", 20)), caps=caps)
    conf = Config(solver="vr", temperature=cfg.get("temperature", 1.0), merge=vr["merge"], log=True,
                  write_every_s=vr["write_every_s"], max_resamples=vr["max_resamples"], only_ids=route,
                  quarter_ids=quarter, think1_band=vr.get("think1_band"), min_useful_band=vr.get("min_useful_band"))
    print(f"vr: {len(route)} tasks routed ({len(quarter)} in group 0), {len(tasks)} written; caps {caps}; "
          f"deadline in {float(cfg['collect_end_epoch']) - time.time():.0f} s; merge {vr['merge']}", flush=True)

    class HybridRunner(Runner):
        def maybe_write(self, force):
            now = time.time()
            if not force and now - self._last_write < self.cfg.write_every_s:
                return
            self._last_write = now
            with self.lock:
                snapshot = {tid: list(v) for tid, v in self.samples.items()}
            sub = full_submission(tasks, route, snapshot, pool, cfg["b0_submission"], self.cfg.merge)
            write_submission(sub, tasks, cfg["submission"])
            print(f"vr | merge written: {sum(len(v) for v in snapshot.values())} samples", flush=True)

    runner = HybridRunner({t: tasks[t] for t in route}, llm, budget, conf, out, None)
    summary = runner.run()
    answered = sum(1 for t in route if any(g is not None for s in runner.samples.get(t, []) for g in s["grids"]))
    verified = sum(1 for t in route if any(s["verify"]["passed"] for s in runner.samples.get(t, [])))
    wall = time.time() - t0
    b = summary["budget"]
    print(f"vr done: {summary['n_samples']} samples, {answered} tasks with a grid, {verified} with a verified "
          f"program; {b['spent']} generated tokens in {wall:.0f} s ({b['spent'] / max(1.0, wall):.1f} tok/s)",
          flush=True)
    return {"started": summary["n_samples"], "skipped": [], "answered_tasks": answered,
            "merge": {"replaced": None, "verified_tasks": verified}, "agent_wall_s": round(wall, 1),
            "gen_tok_per_s": round(b["spent"] / max(1.0, wall), 1)}


if __name__ == "__main__":
    try:
        mode, path = sys.argv[1], sys.argv[2]
        with open(path) as f:
            config = json.load(f)
        import hybrid_p2
        if mode == "install":
            hybrid_p2.install(config)
        else:
            hybrid_p2.run_routed = run_vr
            hybrid_p2.run(config)
    except BaseException:
        print(traceback.format_exc(), flush=True)
    sys.exit(0)
