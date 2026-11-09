"""Solvers. Each turns a set of tasks into Sample records and a submission, under a Budget.

  direct   the model writes the grids; cut samples are rescued; spare budget buys second samples (shortest first)
  poetiq   the Appendix A loop (one expert, up to 10 rounds) with each task given the mean budget B
  vr       verify-and-refine:
             s1      grid + program in one reply; the program is run on the demonstrations (the verification bit)
             rescue  a cut sample gets a thinking-off turn with the tail of its reasoning
             repair  an unverified sample gets one Poetiq feedback round (new user turn, capped thinking)
             s2      spare budget buys independent second samples on still-unverified tasks, shortest first
           selection: verified programs first, then self-consistent grids (selection.py, policy "vr")

A Sample record (one per model reply that could carry an answer):
  task_id, sample_id, stage, parent, finish ("own" | "cut" | "rescued" | "length" | "error"),
  gen_tokens, prompt_tokens, latency_s, grids (per test input), program, verify {ran, passed, score, n_pass,
  program_outputs, feedback}, text (answer), reasoning_tail (last 16k chars), seed
Correctness is never computed here; the solvers do not see solutions.

Scheduling: a priority heap of jobs run by a thread pool (up to budget.max_in_flight in flight). Each task is a
chain (s1 -> repair -> s2 -> s2 ...), one request in flight per task, so no two samples of one task race; the heap
orders chains across tasks by (stage, task size), shortest first within a stage.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .augment import IDENTITY, Aug, invert_sample, random_aug
from .budget import Budget
from .data import task_size
from .grids import hardcoded_risk, parse_outputs, parse_program
from .llm import LLM, Completion
from .prompts import direct_program_prompt, direct_prompt, poetiq_prompt, repair_prompt, rescue_prompt, think_hint
from .sandbox import grade
from .selection import two_attempts
from .submission import identity_submission, write_submission

REASONING_TAIL_CHARS = 16000


@dataclass
class Config:
    solver: str = "vr"                   # direct | poetiq | vr
    temperature: float = 1.0
    style: str = "diagram"               # task text style
    shuffle_demos: bool = False          # new demonstration order per sample (Poetiq does this every round)
    augment_s2: bool = True              # second samples see a dihedral + colour-permuted view, mapped back
    max_resamples: int = 3               # extra independent samples a task may get from spare budget
    repair: bool = True                  # vr: one feedback round for unverified samples
    repair_only_with_program: bool = False
    poetiq_rounds: int = 10
    poetiq_experts: int = 1
    selection: str = ""                  # "" = the solver's own policy
    merge: str = "R0"                    # with an external pool
    use_validator: bool = True
    seed: int = 0
    write_every_s: float = 20.0
    max_errors_in_a_row: int = 40
    think_hint: bool = False             # tell the model its thinking budget in the prompt
    log: bool = True
    # hybrid scheduling (all optional; the defaults keep the single-group behaviour)
    only_ids: Optional[List[str]] = None     # sample only these tasks; every task is still written (external pool)
    quarter_ids: Optional[set] = None        # group 0: every stage of these tasks goes before group 1's first samples
    think1_band: Optional[int] = None        # first-sample thinking cap for tasks outside quarter_ids
    min_useful_band: Optional[int] = None    # smallest grant for their first sample (default: half its cap)
    extra: dict = field(default_factory=dict)


class Runner:
    def __init__(self, tasks: Dict[str, dict], llm: LLM, budget: Budget, cfg: Config, out_dir: str,
                 external: Optional[Dict[str, List[List[list]]]] = None):
        self.tasks = tasks
        self.llm = llm
        self.budget = budget
        self.cfg = cfg
        self.out_dir = out_dir
        self.external = external or {}
        os.makedirs(out_dir, exist_ok=True)
        self.samples: Dict[str, List[dict]] = {tid: [] for tid in tasks}
        self.sizes = {tid: task_size(t) for tid, t in tasks.items()}
        self.lock = threading.Lock()
        self.seq = 0
        self.heap: List[tuple] = []
        self.sub_path = os.path.join(out_dir, "submission.json")
        self.samples_path = os.path.join(out_dir, "samples.jsonl")
        self._last_write = 0.0
        self.events: List[dict] = []
        self.errors_in_a_row = 0
        self.policy = cfg.selection or {"direct": "grid_only", "poetiq": "poetiq", "vr": "vr"}[cfg.solver]
        write_submission(identity_submission(tasks), tasks, self.sub_path)   # valid from the first second
        if os.path.exists(self.samples_path):                                 # a rerun into the same dir: keep
            os.replace(self.samples_path, self.samples_path + f".prev-{int(time.time())}")   # the old file aside
        open(self.samples_path, "w").close()

    # ------------------------------------------------------------------ scheduling
    def push(self, priority: tuple, fn: Callable):
        with self.lock:
            self.seq += 1
            heapq.heappush(self.heap, (priority, self.seq, fn))

    def pop(self):
        with self.lock:
            return heapq.heappop(self.heap) if self.heap else None

    def run(self) -> dict:
        self.seed_jobs()
        futures = set()
        with ThreadPoolExecutor(max_workers=self.budget.max_in_flight) as pool:
            while True:
                while len(futures) < self.budget.max_in_flight:
                    item = self.pop()
                    if item is None:
                        break
                    futures.add(pool.submit(self._guard, item[2]))
                if not futures:
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    for pr, fn in (f.result() or []):
                        self.push(pr, fn)
                self.maybe_write(force=False)
                rs = self.budget.remaining_seconds()
                if (rs is not None and rs <= 0) or self.budget.remaining_tokens() <= 0:
                    with self.lock:
                        self.heap.clear()
        self.maybe_write(force=True)
        summary = {"budget": self.budget.summary(), "n_samples": sum(len(v) for v in self.samples.values()),
                   "config": self.cfg.__dict__, "policy": self.policy}
        with open(os.path.join(self.out_dir, "run.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        return summary

    def _guard(self, fn):
        try:
            return fn()
        except Exception as e:  # a failed job must never kill the run
            self.log_event({"error": f"{type(e).__name__}: {e}"})
            return []

    def log_event(self, ev: dict):
        ev["t"] = round(time.time() - self.budget.start, 1)
        with self.lock:
            self.events.append(ev)
        if self.cfg.log:
            print(json.dumps(ev), flush=True)

    # ------------------------------------------------------------------ model calls
    def call(self, tid: str, stage: str, prompt: str, think_cap: int, answer_cap: int, thinking: bool,
             seed: int, reserve: int = 0, min_useful: int = 1500) -> Optional[Completion]:
        want = (think_cap + answer_cap) if thinking else answer_cap
        granted = self.budget.grant(want, min_useful=min_useful, reserve=reserve)
        if granted <= 0:
            self.log_event({"task": tid, "stage": stage, "skipped": "no budget"})
            return None
        self.budget.begin(granted)
        t0 = time.time()
        comp = self.llm.generate([{"role": "user", "content": prompt}], granted, thinking=thinking,
                                 temperature=self.cfg.temperature, seed=seed,
                                 timeout_s=self.budget.timeout_for(granted))
        comp.latency_s = time.time() - t0
        self.budget.charge(comp.tokens_out, comp.tokens_in, comp.latency_s, granted)
        comp.meta["granted"] = granted
        if comp.finish == "error":
            self.log_event({"task": tid, "stage": stage, "error": comp.error})
            self._on_error()
        else:
            self.errors_in_a_row = 0
        return comp

    def _on_error(self):
        """A server hiccup must not drain the job heap in seconds: back off, and stop after many in a row."""
        with self.lock:
            self.errors_in_a_row += 1
            n = self.errors_in_a_row
        time.sleep(min(60, 5 * n))
        if n >= self.cfg.max_errors_in_a_row:
            self.log_event({"fatal": f"{n} model errors in a row; stopping new work"})
            with self.lock:
                self.heap.clear()

    def record(self, task: dict, stage: str, parent: Optional[str], comp: Completion, with_program: bool,
               seed: int, aug: Aug = IDENTITY) -> dict:
        """task is the view the model saw (possibly augmented); the record is stored in the original frame."""
        tid = task["id"]
        n_test = len(task["test"])
        grids = parse_outputs(comp.text, n_test) if comp.text else [None] * n_test
        program = parse_program(comp.text) if (with_program and comp.text) else None
        verify = {"ran": False, "passed": False, "score": 0.0, "n_pass": 0, "program_outputs": [None] * n_test,
                  "feedback": ""}
        if program:
            g = grade(program, task["train"], task["test"])
            verify = {"ran": True, "passed": g.passed, "score": g.score, "n_pass": g.n_pass,
                      "program_outputs": g.test_outputs, "feedback": g.feedback, "errors": g.errors}
        if comp.finish == "error":
            finish = "error"
        elif comp.cut_in_thinking:
            finish = "cut"
        elif comp.finish == "length":
            finish = "length"
        elif stage.startswith("rescue"):
            finish = "rescued"
        else:
            finish = "own"
        with self.lock:
            sid = f"{tid}-{stage}-{len(self.samples[tid])}"
            s = {"task_id": tid, "sample_id": sid, "stage": stage, "parent": parent, "finish": finish,
                 "gen_tokens": comp.tokens_out, "prompt_tokens": comp.tokens_in, "granted": comp.meta.get("granted"),
                 "latency_s": round(comp.latency_s, 1), "grids": grids, "program": program, "verify": verify,
                 "text": comp.text[-20000:], "reasoning_tail": comp.reasoning[-REASONING_TAIL_CHARS:],
                 "reasoning_chars": len(comp.reasoning), "seed": seed, "with_program": with_program}
            invert_sample(aug, s)
            self.samples[tid].append(s)
            with open(self.samples_path, "a") as f:
                f.write(json.dumps(s) + "\n")
        self.log_event({"task": tid, "stage": stage, "finish": finish, "tokens": comp.tokens_out,
                        "grids": sum(g is not None for g in grids), "program": bool(program),
                        "verified": verify["passed"], "score": round(verify["score"], 2)})
        return s

    def maybe_write(self, force: bool):
        now = time.time()
        if not force and now - self._last_write < self.cfg.write_every_s:
            return
        self._last_write = now
        sub = {}
        with self.lock:
            snapshot = {tid: list(v) for tid, v in self.samples.items()}
        for tid, task in self.tasks.items():
            sub[tid] = two_attempts(task, snapshot[tid], self.policy, self.external.get(tid), self.cfg.merge,
                                    self.cfg.use_validator)
        write_submission(sub, self.tasks, self.sub_path)

    # ------------------------------------------------------------------ job graphs
    def _group(self, tid: str) -> int:
        q = self.cfg.quarter_ids
        return 0 if q is None or tid in q else 1

    def _prio(self, tid: str, stage_rank: int) -> tuple:
        """Heap priority: (group, stage, task size). One group (the default) orders stage-major, shortest first."""
        return (self._group(tid), stage_rank, self.sizes[tid])

    def seed_jobs(self):
        ids = self.tasks if self.cfg.only_ids is None else [t for t in self.cfg.only_ids if t in self.tasks]
        order = sorted(ids, key=lambda t: self.sizes[t])
        for tid in order:
            if self.cfg.solver == "poetiq":
                self.push(self._prio(tid, 0), lambda tid=tid: self.job_poetiq_round(tid, 0, [], 0))
            else:
                self.push(self._prio(tid, 0), lambda tid=tid: self.job_sample(tid, "s1", 0))

    def _next_s2(self, tid: str, r: int):
        """The next link of a task's chain: second sample r+1, if allowed."""
        nr = r + 1
        if nr > self.cfg.max_resamples:
            return []
        return [(self._prio(tid, 2 + nr), lambda: self.job_sample(tid, "s2", nr))]

    @staticmethod
    def _aug_seed(tid: str, seed: int) -> int:
        return int(hashlib.sha256(f"{tid}:{seed}".encode()).hexdigest()[:8], 16)

    def _task_done(self, tid: str) -> bool:
        """vr: a verified answer exists. direct: two samples agree on every test output."""
        with self.lock:
            ss = list(self.samples[tid])
        if self.cfg.solver == "vr":
            # done = a verified program whose test output the model's own written grid agrees with (or no grid
            # was written); a verified program that contradicts the written grid is not trusted on its own
            for s in ss:
                if not s["verify"]["passed"] or hardcoded_risk(s.get("program") or ""):
                    continue
                po = s["verify"]["program_outputs"]
                if all(g is None or (j < len(po) and g == po[j]) for j, g in enumerate(s["grids"])):
                    return True
            return False
        n_test = len(self.tasks[tid]["test"])
        for j in range(n_test):
            grids = [json.dumps(s["grids"][j]) for s in ss if s["grids"][j] is not None]
            if not grids or max(grids.count(g) for g in set(grids)) < 2:
                return False
        return True

    def job_sample(self, tid: str, stage: str, r: int, retry: int = 0):
        """One independent sample (s1 or s2) and the next link of the task's chain."""
        task = self.tasks[tid]
        if stage == "s2" and self._task_done(tid):
            return []
        with_program = self.cfg.solver == "vr"
        seed = self.cfg.seed * 1000 + r * 10 + retry
        dseed = seed if (self.cfg.shuffle_demos and r > 0) else None
        aug = random_aug(self._aug_seed(tid, seed)) if (stage == "s2" and self.cfg.augment_s2) else IDENTITY
        view = aug.apply_task(task) if not aug.is_identity else task
        prompt = (direct_program_prompt if with_program else direct_prompt)(view, dseed, self.cfg.style)
        caps = self.budget.caps
        think1 = caps.think1 if self._group(tid) == 0 else (self.cfg.think1_band or caps.think1)
        if self.cfg.think_hint:
            prompt += think_hint(think1)
        # a sample that might be cut needs its rescue affordable too; a second sample, or a first sample outside
        # the quarter, is not worth starting with less than half its cap
        min_useful = 1500 if (stage == "s1" and self._group(tid) == 0) else (think1 + caps.answer) // 2
        if stage == "s1" and self._group(tid) == 1 and self.cfg.min_useful_band is not None:
            min_useful = self.cfg.min_useful_band
        comp = self.call(tid, stage, prompt, think1, caps.answer, True, seed, reserve=caps.rescue,
                         min_useful=min_useful)
        if comp is None:
            return []
        s = self.record(view, stage, None, comp, with_program, seed, aug)
        if s["finish"] == "error":
            if stage == "s1" and retry < 2:     # a server error is not a sample: try the first sample again
                return [(self._prio(tid, 0), lambda: self.job_sample(tid, "s1", r, retry + 1))]
            return self._next_s2(tid, r)
        if s["finish"] == "cut":
            rs = self.rescue(view, s, with_program, seed, aug)
            if rs is None:
                return []
            s = rs
        if stage == "s1":
            follow = self.after_sample(task, s)   # the repair job continues the chain itself
            return follow if follow else self._next_s2(tid, 0)
        return self._next_s2(tid, r)

    def rescue(self, task: dict, cut: dict, with_program: bool, seed: int, aug: Aug = IDENTITY) -> Optional[dict]:
        prompt = rescue_prompt(task, cut["reasoning_tail"], with_program, self.cfg.style)
        comp = self.call(task["id"], "rescue", prompt, 0, self.budget.caps.rescue, False, seed)
        if comp is None:
            return None
        stage = {"s1": "rescue", "repair": "rescue_repair", "s2": "rescue_s2"}.get(cut["stage"], "rescue")
        return self.record(task, stage, cut["sample_id"], comp, with_program, seed, aug)

    def after_sample(self, task: dict, s: dict):
        """vr: queue a repair round if the sample is not verified."""
        caps = self.budget.caps
        if self.cfg.solver != "vr" or not self.cfg.repair or not caps.repair_enabled:
            return []
        if s["verify"]["passed"] or s["finish"] == "error" or s["stage"] in ("repair", "rescue_repair"):
            return []
        if self.cfg.repair_only_with_program and not s["program"]:
            return []
        if s["stage"] in ("s2", "rescue_s2"):
            return []   # one repair per task, on the first sample
        tid = task["id"]
        return [(self._prio(tid, 1), lambda: self.job_repair(tid, s))]

    def job_repair(self, tid: str, s: dict):
        task = self.tasks[tid]
        if self._task_done(tid):
            return []
        v = s["verify"]
        feedback = v["feedback"] if v["ran"] else "(no runnable program was found in your reply)"
        prompt = repair_prompt(task, s["program"], feedback, v["score"] if v["ran"] else 0.0, s["grids"],
                               self.cfg.style)
        caps = self.budget.caps
        seed = s["seed"] + 500
        comp = self.call(tid, "repair", prompt, caps.think2, caps.answer, True, seed, reserve=caps.rescue,
                         min_useful=max(1500, caps.think2 // 2))
        if comp is None:
            return []
        rs = self.record(task, "repair", s["sample_id"], comp, True, seed)
        if rs["finish"] == "cut":
            self.rescue(task, rs, True, seed)
        return self._next_s2(tid, 0)

    def job_poetiq_round(self, tid: str, it: int, tries: List[dict], spent: int):
        """One round of the Poetiq loop for one task; the task's own budget is the mean B."""
        task = self.tasks[tid]
        per_task = self.budget.mean_tokens_per_task
        if it >= self.cfg.poetiq_rounds or spent >= per_task:
            return []
        seed = self.cfg.seed * 1000 + it
        prompt = poetiq_prompt(task, seed, tries)
        caps = self.budget.caps
        think = min(caps.think1, max(1500, per_task - spent - caps.answer))
        comp = self.call(tid, f"poetiq_r{it}", prompt, think, caps.answer, True, seed, reserve=caps.rescue)
        if comp is None:
            return []
        s = self.record(task, f"poetiq_r{it}", None, comp, True, seed)
        spent += comp.tokens_out
        if s["finish"] == "cut":
            rs = self.rescue(task, s, True, seed)
            if rs is None:
                return []
            spent += rs["gen_tokens"]
            s = rs
        if s["verify"]["passed"]:
            return []
        if s["program"]:
            tries = tries + [{"code": s["program"], "feedback": s["verify"]["feedback"], "score": s["verify"]["score"]}]
        return [(self._prio(tid, 0), lambda: self.job_poetiq_round(tid, it + 1, tries, spent))]


def load_samples(path: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                out.setdefault(s["task_id"], []).append(s)
    return out
