"""Cheap hard checks on a candidate test output, inferred from the demonstrations.

Two rules, each applied only when every demonstration obeys it:
  shape:   same as input | integer up/down scale of the input | transposed input
  palette: output colours are a subset of the test input's colours plus the colours any demonstration introduces
A candidate that breaks a rule is demoted in ranking, never dropped (the rules are checked on the public sets
by scripts/check_validator.py: they must never reject a true output there).
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .grids import Grid, grid_shape

Shape = Tuple[int, int]


def _colors(g: Grid) -> Set[int]:
    return {c for r in g for c in r}


def predict_shape(train: List[dict], test_input: Grid) -> Optional[Shape]:
    ins = [grid_shape(p["input"]) for p in train]
    outs = [grid_shape(p["output"]) for p in train]
    ti = grid_shape(test_input)
    if not train or ti is None or None in ins or None in outs:
        return None
    if all(i == o for i, o in zip(ins, outs)):
        return ti
    if all(o == (i[1], i[0]) for i, o in zip(ins, outs)) and any(i[0] != i[1] for i in ins):
        return (ti[1], ti[0])
    # (a constant output shape is NOT trusted: on the public sets it wrongly rejects true outputs,
    #  e.g. eval 38007db0 and train c909285e, where the constant was a coincidence of the demonstrations)
    # integer scale factors, constant across demonstrations
    for num, den in [(k, 1) for k in range(2, 11)] + [(1, k) for k in range(2, 11)]:
        if all(o[0] * den == i[0] * num and o[1] * den == i[1] * num for i, o in zip(ins, outs)):
            if (ti[0] * num) % den == 0 and (ti[1] * num) % den == 0:
                return (ti[0] * num // den, ti[1] * num // den)
    return None


def allowed_colors(train: List[dict], test_input: Grid) -> Set[int]:
    introduced: Set[int] = set()
    for p in train:
        introduced |= _colors(p["output"]) - _colors(p["input"])
    return _colors(test_input) | introduced


def check(train: List[dict], test_input: Grid, cand: Grid) -> dict:
    """Returns {"shape_ok": bool|None, "palette_ok": bool, "hard_ok": bool}. None = rule not applicable."""
    ps = predict_shape(train, test_input)
    shape_ok = None if ps is None else (grid_shape(cand) == ps)
    palette_ok = _colors(cand) <= allowed_colors(train, test_input)
    return {"shape_ok": shape_ok, "palette_ok": palette_ok, "hard_ok": (shape_ok is not False) and palette_ok}
