"""Atomic writes, resource guards and small helpers shared by training and evaluation."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    torch.save(obj, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_jsonl(path: str | Path, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


class LoadGuard:
    """Pause work while the 5-minute load average stays above ``limit`` (checked every ``every_s`` seconds).

    ``check()`` returns the seconds spent paused. Work resumes once the 1- and 5-minute averages fall below
    ``resume``; ``on_pause`` (e.g. a checkpoint) runs once before a pause.
    """

    def __init__(self, limit: float = 10.0, resume: float = 8.0, every_s: float = 30.0, log=print):
        self.limit, self.resume, self.every_s, self.log = limit, resume, every_s, log
        self._last = 0.0
        self._high_since: float | None = None
        self.paused_s = 0.0

    def check(self, on_pause=None) -> float:
        now = time.monotonic()
        if now - self._last < self.every_s:
            return 0.0
        self._last = now
        l1, l5, _ = os.getloadavg()
        if l5 <= self.limit:
            self._high_since = None
            return 0.0
        if self._high_since is None:
            self._high_since = now
            return 0.0
        if now - self._high_since < 120:
            return 0.0
        if on_pause is not None:
            on_pause()
        self.log(f"load {l1:.1f}/{l5:.1f} above {self.limit}: pausing")
        t0 = time.monotonic()
        while True:
            time.sleep(30)
            l1, l5, _ = os.getloadavg()
            if l1 < self.resume and l5 < self.resume:
                break
        dt = time.monotonic() - t0
        self.paused_s += dt
        self._high_since = None
        self.log(f"load {l1:.1f}/{l5:.1f}: resuming after {dt:.0f}s")
        return dt


def limit_threads(n: int = 2) -> None:
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
