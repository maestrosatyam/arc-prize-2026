"""Grid text formatting and parsing of grids from model output."""
from __future__ import annotations

import json
import re
from typing import List, Optional

Grid = List[List[int]]


def grid_text(g: Grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def grid_shape(g) -> Optional[tuple]:
    if not isinstance(g, list) or not g or not all(isinstance(r, list) for r in g):
        return None
    w = len(g[0])
    if w == 0 or any(len(r) != w for r in g):
        return None
    return (len(g), w)


def is_grid(g) -> bool:
    s = grid_shape(g)
    if s is None or s[0] > 30 or s[1] > 30:
        return False
    return all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 9 for r in g for c in r)


def grids_equal(a, b) -> bool:
    return is_grid(a) and is_grid(b) and a == b


def cell_accuracy(pred: Grid, truth: Grid) -> float:
    """Share of right cells; 0 if the shapes differ."""
    if grid_shape(pred) != grid_shape(truth):
        return 0.0
    n = sum(len(r) for r in truth)
    hit = sum(1 for pr, tr in zip(pred, truth) for p, t in zip(pr, tr) if p == t)
    return hit / n if n else 0.0


_ROW_RE = re.compile(r"^\s*[0-9](?:[ ,]+[0-9])*\s*$")


def _parse_block(text: str) -> Optional[Grid]:
    """Parse a block of digit rows ('0 1 2' per line, or '012', or JSON-ish) into a grid."""
    text = text.strip()
    if not text:
        return None
    # JSON list of lists
    if text.startswith("["):
        try:
            g = json.loads(text)
            return g if is_grid(g) else None
        except json.JSONDecodeError:
            pass
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("|").strip()
        if not line:
            continue
        if _ROW_RE.match(line):
            rows.append([int(c) for c in re.findall(r"[0-9]", line)])
        elif re.fullmatch(r"[0-9]{1,30}", line):
            rows.append([int(c) for c in line])
        else:
            return None
    return rows if is_grid(rows) else None


# Answer layouts the prompts ask for:  <output_1> ... </output_1>  (one tag per test input)
_TAG_RE = re.compile(r"<output_(\d+)>\s*(.*?)\s*</output_\1>", re.S)
_FENCE_RE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.S)


def parse_outputs(text: str, n_test: int) -> List[Optional[Grid]]:
    """Parse the test output grids from a reply. Each test output is parsed on its own.

    Order of preference: <output_k> tags, then the last n_test digit blocks (fenced or bare).
    """
    out: List[Optional[Grid]] = [None] * n_test
    tags = {int(k): _parse_block(v) for k, v in _TAG_RE.findall(text)}
    if tags:
        for k in range(n_test):
            out[k] = tags.get(k + 1)
        if any(o is not None for o in out):
            return out
    blocks = [b for b in _FENCE_RE.findall(text) if _parse_block(b) is not None]
    if not blocks:
        blocks = _bare_blocks(text)
    grids = [_parse_block(b) for b in blocks]
    grids = [g for g in grids if g is not None]
    if grids:
        tail = grids[-n_test:]
        for k, g in enumerate(tail):
            out[k] = g
    return out


def _bare_blocks(text: str) -> List[str]:
    """Maximal runs of lines that look like grid rows."""
    blocks, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if s and (_ROW_RE.match(s) or re.fullmatch(r"[0-9]{1,30}", s)):
            cur.append(s)
        else:
            if cur:
                blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


_PY_RE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def parse_program(text: str) -> Optional[str]:
    """The last fenced block (```python, ```py or bare ```) that defines transform()."""
    cands = [m for m in _PY_RE.findall(text) if "def transform" in m]
    return cands[-1] if cands else None


_LITERAL_RE = re.compile(r"\[\s*(?:-?\d+\s*,\s*)+-?\d+\s*\]")


def hardcoded_risk(program: str, min_ints: int = 40) -> bool:
    """A program that carries long integer-list literals is probably a lookup table of the demonstrations: it
    passes them without encoding the rule. Verification by such a program is not trusted as evidence."""
    n = sum(len(re.findall(r"-?\d+", m)) for m in _LITERAL_RE.findall(program))
    return n >= min_ints
