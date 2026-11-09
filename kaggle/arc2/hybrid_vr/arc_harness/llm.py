"""LLM backends.

  VLLMChat      an OpenAI-compatible vLLM server (the Kaggle path); thinking on/off via chat_template_kwargs
  ScriptedLLM   a python callable, for tests
  OracleLLM     a simulator driven by the solutions, for CPU end-to-end checks of the pipeline and of the
                budget curve plumbing; it does NOT measure anything about a real model

All return a Completion. tokens_out counts reasoning + answer tokens; that is the budget unit.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .augment import IDENTITY, Aug
from .grids import grid_text

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


@dataclass
class Completion:
    text: str
    reasoning: str
    tokens_out: int
    tokens_in: int
    finish: str                      # "stop" | "length" | "error"
    cut_in_thinking: bool
    latency_s: float = 0.0
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


class LLM:
    name = "base"

    def generate(self, messages: List[dict], max_tokens: int, thinking: bool = True,
                 temperature: float = 1.0, seed: Optional[int] = None, timeout_s: Optional[float] = None) -> Completion:
        raise NotImplementedError


class VLLMChat(LLM):
    """OpenAI-compatible chat completions against a local vLLM server.

    Qwen3-family thinking is switched with chat_template_kwargs={"enable_thinking": ...}. If the server runs with a
    reasoning parser, reasoning arrives in message.reasoning_content; otherwise it is split from <think> tags here.
    The thinking budget is enforced by the caller as max_tokens = thinking_cap + answer_cap, and a reply that ends
    with finish_reason=="length" and no answer content is reported as cut_in_thinking (the caller then rescues).
    """
    name = "vllm"

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "qwen", timeout_s: float = 3600,
                 top_p: float = 0.95, top_k: int = 20, extra: Optional[dict] = None, retries: int = 3,
                 system: Optional[str] = None, template_kwargs: Optional[dict] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.top_p = top_p
        self.top_k = top_k
        self.extra = extra or {}              # extra body fields, e.g. {"reasoning_effort": "low"}
        self.template_kwargs = template_kwargs or {}   # extra chat_template_kwargs (model-specific effort knobs)
        self.system = system                  # optional system prompt
        self.retries = retries

    def generate(self, messages, max_tokens, thinking=True, temperature=1.0, seed=None, timeout_s=None) -> Completion:
        if self.system:
            messages = [{"role": "system", "content": self.system}] + list(messages)
        body = {
            "model": self.model, "messages": messages, "max_tokens": int(max_tokens),
            "temperature": temperature, "top_p": self.top_p, "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": bool(thinking), **self.template_kwargs},
        }
        if seed is not None:
            body["seed"] = int(seed)
        body.update(self.extra)
        t0 = time.time()
        last_err = None
        timeout = timeout_s or self.timeout_s
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.load(r)
                return self._parse(resp, time.time() - t0)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode()[:500]
                except Exception:
                    pass
                last_err = f"HTTP {e.code}: {detail}"
                if e.code == 400 and "context length" in detail and body["max_tokens"] > 2048:
                    # prompt + max_tokens over the server's max_model_len (large tasks at the band's cap): retry at
                    # 0.75x, as the public direct mode does
                    body["max_tokens"] = int(body["max_tokens"] * 0.75)
                    continue
                if 400 <= e.code < 500:
                    break                        # a bad request will not get better by retrying
                time.sleep(min(30, 2 ** attempt))
            except (TimeoutError, socket.timeout) as e:
                last_err = f"timeout after {timeout:.0f}s"
                break                            # the server is still generating; do not pile a second request on
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(30, 2 ** attempt))
        return Completion(text="", reasoning="", tokens_out=0, tokens_in=0, finish="error",
                          cut_in_thinking=False, latency_s=time.time() - t0, error=last_err)

    @staticmethod
    def _parse(resp: dict, latency: float) -> Completion:
        ch = resp["choices"][0]
        msg = ch.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and "<think>" in content:
            m = THINK_RE.search(content)
            if m:
                reasoning = m.group(1)
                content = content[m.end():]
            else:  # cut inside the think block
                reasoning = content.split("<think>", 1)[1]
                content = ""
        usage = resp.get("usage", {})
        finish = ch.get("finish_reason") or "stop"
        cut = finish == "length" and not content.strip()
        return Completion(text=content, reasoning=reasoning, tokens_out=int(usage.get("completion_tokens", 0)),
                          tokens_in=int(usage.get("prompt_tokens", 0)), finish=finish, cut_in_thinking=cut,
                          latency_s=latency, meta={"id": resp.get("id")})

    def ping(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/models", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False


class ScriptedLLM(LLM):
    """fn(messages, thinking, max_tokens) -> str | (text, reasoning). Tokens are counted as len/4."""
    name = "scripted"

    def __init__(self, fn: Callable):
        self.fn = fn
        self.calls = 0

    def generate(self, messages, max_tokens, thinking=True, temperature=1.0, seed=None, timeout_s=None) -> Completion:
        self.calls += 1
        out = self.fn(messages, thinking, max_tokens)
        text, reasoning = (out, "") if isinstance(out, str) else out
        n = (len(text) + len(reasoning)) // 4
        return Completion(text=text, reasoning=reasoning, tokens_out=n, tokens_in=len(messages[-1]["content"]) // 4,
                          finish="stop", cut_in_thinking=False)


class OracleLLM(LLM):
    """A deterministic simulator of a reasoning model, keyed on the test input shown in the prompt.

    Each task gets a hidden "tokens needed" T from its id. With a thinking cap C >= T the reply finishes on its own
    and is right with probability p_own; with C < T the reply is cut (the caller rescues it, and the rescue is right
    with a probability that grows with C/T). A repair turn (the prompt carries example feedback) succeeds with
    probability p_repair. Wrong grids are near misses (a few cells) or wrong shapes. Programs are lookup tables so
    that the sandbox path, verification and the verified-but-wrong case are all exercised.

    This is for exercising the harness on CPU. Its numbers say nothing about any real model.
    """
    name = "oracle"

    def __init__(self, tasks: Dict[str, dict], p_own: float = 0.8, p_rescue_max: float = 0.45,
                 p_repair: float = 0.35, p_program: float = 0.75, p_verified_wrong: float = 0.08, seed: int = 0,
                 t_min: int = 4000, t_max: int = 160000):
        self.index = {}
        for t in tasks.values():
            self.index[grid_text(t["test"][0]["input"])] = t
        self.p_own, self.p_rescue_max, self.p_repair = p_own, p_rescue_max, p_repair
        self.p_program, self.p_verified_wrong = p_program, p_verified_wrong
        self.seed = seed
        self.t_min, self.t_max = t_min, t_max
        self.lock = threading.Lock()
        self.calls = 0

    def tokens_needed(self, tid: str) -> int:
        h = int(hashlib.sha256((tid + "oracle").encode()).hexdigest(), 16) % 10_000 / 10_000
        return int(self.t_min + (h ** 1.6) * (self.t_max - self.t_min))

    def _find(self, prompt: str):
        """The task shown in the prompt and the view (Aug) it is shown in, or (None, None)."""
        m = re.search(r"(?:Test input #1|Challenge #1)\n(?:<Diagram>\n)?(.*?)(?:\n</Diagram>)?\n(?:\n|$)", prompt, re.S)
        if not m:
            return None, None
        key = m.group(1).strip()
        if key in self.index:
            return self.index[key], IDENTITY
        seen = [[int(c) for c in row.split()] for row in key.splitlines()]
        for t in self.index.values():
            for rot in range(4):
                for flip in (False, True):
                    geo = Aug(rot, flip)
                    g = geo.apply(t["test"][0]["input"])
                    if len(g) != len(seen) or len(g[0]) != len(seen[0]):
                        continue
                    perm, ok = {}, True
                    for ra, rb in zip(g, seen):
                        for a, b in zip(ra, rb):
                            if perm.setdefault(a, b) != b:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok and perm.get(0, 0) == 0 and len(set(perm.values())) == len(perm):
                        full = list(range(10))
                        used = set(perm.values())
                        free = [c for c in range(10) if c not in used]
                        for c in range(10):
                            if c in perm:
                                full[c] = perm[c]
                            elif c != 0 or 0 not in used:
                                full[c] = free.pop(0) if c != 0 else 0
                        return t, Aug(rot, flip, tuple(full))
        return None, None

    def generate(self, messages, max_tokens, thinking=True, temperature=1.0, seed=None, timeout_s=None) -> Completion:
        with self.lock:
            self.calls += 1
            n = self.calls
        prompt = messages[-1]["content"]
        task, aug = self._find(prompt)
        if task is None:
            return Completion(text="I cannot parse this task.", reasoning="", tokens_out=20, tokens_in=len(prompt) // 4,
                              finish="stop", cut_in_thinking=False)
        rng = random.Random(f"{self.seed}:{task['id']}:{n}:{seed}")
        T = self.tokens_needed(task["id"])
        is_rescue = "<your_reasoning_so_far>" in prompt
        is_repair = "Results on the examples" in prompt or "EXISTING PARTIAL" in prompt
        wants_program = "transform(grid" in prompt
        answer_cap = 6000
        think_cap = max_tokens - answer_cap if thinking else 0

        if thinking and not is_repair and think_cap < T:
            # cut inside thinking: no answer at all
            return Completion(text="", reasoning="partial reasoning " * (think_cap // 8), tokens_out=max_tokens,
                              tokens_in=len(prompt) // 4, finish="length", cut_in_thinking=True)
        if is_repair:
            p_right = self.p_repair if rng.random() < 0.9 else 0.0
            spent = min(think_cap, T // 3) if thinking else 500
        elif is_rescue:
            frac = min(1.0, len(prompt) / 1.0 / max(1, T))  # crude: more of the tail, more progress
            p_right = self.p_rescue_max * (0.5 + 0.5 * frac)
            spent = 1500
        else:
            p_right = self.p_own
            spent = min(think_cap, T) if thinking else 800
        right = rng.random() < p_right
        grids = []
        for te in task["test"]:
            truth = aug.apply(te["output"])          # the answer in the frame the prompt shows
            grids.append(truth if right else self._wrong(truth, rng))
        text = "\n".join(f"<output_{k}>\n{grid_text(g)}\n</output_{k}>" for k, g in enumerate(grids, 1))
        tokens = spent + sum(len(grid_text(g)) // 3 for g in grids)
        if wants_program and rng.random() < self.p_program:
            passes = rng.random() < (0.9 if right else 0.25)
            prog_test = grids
            if passes and right and rng.random() < self.p_verified_wrong:
                prog_test = [self._wrong(g, rng) for g in grids]  # passes the examples, wrong on the test
                text = "\n".join(f"<output_{k}>\n{grid_text(g)}\n</output_{k}>" for k, g in enumerate(prog_test, 1))
            text += "\n\n```python\n" + self._lookup_program(aug.apply_task(task), passes, prog_test, rng) + "\n```"
            tokens += 900
        return Completion(text=text, reasoning="reasoning " * (spent // 10), tokens_out=tokens,
                          tokens_in=len(prompt) // 4, finish="stop", cut_in_thinking=False)

    @staticmethod
    def _wrong(truth, rng):
        g = [row[:] for row in truth]
        if rng.random() < 0.25 and len(g) > 1:
            return g[:-1]  # wrong shape
        for _ in range(rng.randint(1, 3)):
            i, j = rng.randrange(len(g)), rng.randrange(len(g[0]))
            g[i][j] = (g[i][j] + rng.randint(1, 9)) % 10
        return g

    @staticmethod
    def _lookup_program(task, passes, test_grids, rng):
        table = {}
        for k, p in enumerate(task["train"]):
            out = p["output"]
            if not passes and k == 0:
                out = OracleLLM._wrong(out, rng)
            table[json.dumps(p["input"])] = out
        for te, g in zip(task["test"], test_grids):
            table[json.dumps(te["input"])] = g
        return ("import json\nimport numpy as np\nTABLE = " + json.dumps(table) +
                "\ndef transform(grid):\n    return np.array(TABLE[json.dumps(grid.tolist())])")
