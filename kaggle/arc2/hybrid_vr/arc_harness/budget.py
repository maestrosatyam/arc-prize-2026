"""Token and wall-clock budget.

The budget unit is generated tokens (reasoning + answer), summed over every model call. A run is given a mean
budget per task B; the pool is B x n_tasks. Per-stage caps are derived from B so that one knob gives the 15k /
30k / 60k / 120k curve:

  think1   first-sample thinking cap     clamp(0.68 B,  8k, 90k)   (41k at B=60k, the measured sweet spot)
  think2   repair-round thinking cap     clamp(0.30 B,  6k, 24k)   (repair is skipped below B=20k)
  answer   answer part after thinking    6k  (grids for up to 2 tests plus a program)
  rescue   thinking-off rescue cap       8k

A deadline guard clips every grant so that the request can finish before the deadline minus a reserve at the
measured per-stream rate (aggregate rate / requests in flight).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def clamp(x, lo, hi):
    return max(lo, min(hi, int(x)))


@dataclass
class Caps:
    think1: int
    think2: int
    answer: int = 6000
    rescue: int = 8000
    repair_enabled: bool = True

    @staticmethod
    def from_mean(b: int, overrides: Optional[dict] = None) -> "Caps":
        c = Caps(think1=clamp(0.68 * b, 8000, 90000), think2=clamp(0.30 * b, 6000, 24000),
                 repair_enabled=b >= 20000)
        for k, v in (overrides or {}).items():
            setattr(c, k, v)
        return c


@dataclass
class Budget:
    mean_tokens_per_task: int
    n_tasks: int
    deadline: Optional[float] = None          # epoch seconds; None = no wall-clock limit
    reserve_s: float = 900.0                  # kept free before the deadline for writing the submission
    tokens_per_s: float = 334.0               # aggregate, initial estimate; updated from observations
    max_in_flight: int = 20
    caps: Caps = field(default_factory=lambda: Caps(41000, 18000))
    spent: int = 0
    prefill: int = 0
    calls: int = 0
    _obs_tokens: int = 0
    _obs_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    in_flight: int = 0
    committed: int = 0                        # tokens granted to requests still in flight

    def __post_init__(self):
        self.pool = int(self.mean_tokens_per_task) * int(self.n_tasks)
        self.start = time.time()

    def remaining_tokens(self) -> int:
        return max(0, self.pool - self.spent)

    def remaining_seconds(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return self.deadline - time.time() - self.reserve_s

    def per_stream_rate(self) -> float:
        """Tokens/s one request can expect: the aggregate shared by a full batch. Conservative in the tail of a
        run (fewer requests go faster), right at saturation, never optimistic at the start."""
        return self.tokens_per_s / max(1, self.max_in_flight)

    def grant(self, want: int, min_useful: int = 2000, reserve: int = 0) -> int:
        """Tokens this request may generate now: the cap, clipped by the pool and by the deadline.
        `reserve` is kept back (e.g. for the rescue a cut sample will need), in tokens and in time.
        Returns 0 when fewer than min_useful tokens can be afforded."""
        with self._lock:
            g = min(int(want), self.remaining_tokens() - self.committed - int(reserve))
            rs = self.remaining_seconds()
            if rs is not None:
                g = min(g, int(rs * self.per_stream_rate()) - int(reserve))
            return g if g >= min_useful else 0

    def begin(self, granted: int = 0):
        with self._lock:
            self.in_flight += 1
            self.committed += int(granted)

    def charge(self, tokens_out: int, tokens_in: int = 0, seconds: float = 0.0, granted: int = 0):
        with self._lock:
            self.spent += int(tokens_out)
            self.prefill += int(tokens_in)
            self.calls += 1
            self.in_flight = max(0, self.in_flight - 1)
            self.committed = max(0, self.committed - int(granted))
            elapsed = time.time() - self.start
            if seconds > 1.0 and elapsed > 600 and self.spent > 200_000:   # real servers only, once saturated
                self.tokens_per_s = self.spent / elapsed                 # the observed aggregate rate

    def timeout_for(self, granted: int, floor_s: float = 600.0) -> float:
        """A read timeout for one request: three times the expected stream time at a full batch, at least floor_s."""
        return max(floor_s, 3.0 * granted / max(1.0, self.per_stream_rate()))

    def summary(self) -> dict:
        return {"pool": self.pool, "spent": self.spent, "prefill": self.prefill, "calls": self.calls,
                "mean_per_task": self.spent / max(1, self.n_tasks), "elapsed_s": time.time() - self.start,
                "tokens_per_s_est": self.tokens_per_s}
