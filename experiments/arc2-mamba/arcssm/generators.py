"""Seeded generators of ARC-format tasks: 27 families (26 single rules and two-step compositions) with task-level
parameters.

``make_task(rng, family, n_pairs, max_side)`` draws the task parameters once, then a fresh input per pair, and
returns ``n_pairs`` (input, output) pairs as int8 arrays, or None when the draw is degenerate (invalid output,
outputs equal to inputs in every pair, or repeated inputs). The families are written for this experiment;
the nine tasks in experiments/arc2-scene-grammar-eval/synthetic are not generated here (FAMILY_NOTES lists
the conceptual overlaps).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from arcssm.encoding import dihedral

Grid = np.ndarray

FAMILY_NOTES = {
    "gravity": "related to synthetic 08_gravity_more_objects (cells fall; the synthetic task moves whole objects)",
    "fill_enclosed": "related to synthetic 09_fill_enclosed_bigger",
    "rays": "related to synthetic 03_rays_east",
    "count_bar": "related to synthetic 06_count_bar",
    "compose": "crop/dihedral/colour-map compositions relate to synthetic 05_crop_turn_recolour",
}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def components(g: Grid, bg: int = 0, by_colour: bool = True):
    """4-connected components of non-background cells; returns a list of lists of (r, c)."""
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == bg or seen[r, c]:
                continue
            col = g[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != bg:
                        if by_colour and g[ny, nx] != col:
                            continue
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(cells)
    return out


def rand_shape(rng, max_h: int, max_w: int, kind: str | None = None):
    """Cells (relative) of a random connected shape within max_h x max_w."""
    kind = kind or rng.choice(["blob", "rect", "line", "pixel", "blob"])
    if kind == "pixel":
        return [(0, 0)]
    if kind == "rect":
        h, w = int(rng.integers(1, max_h + 1)), int(rng.integers(1, max_w + 1))
        return [(r, c) for r in range(h) for c in range(w)]
    if kind == "line":
        n = int(rng.integers(2, max(3, max(max_h, max_w) + 1)))
        if rng.random() < 0.5:
            return [(0, c) for c in range(min(n, max_w))]
        return [(r, 0) for r in range(min(n, max_h))]
    n = int(rng.integers(2, max(3, max_h * max_w // 2 + 2)))
    cells = {(0, 0)}
    for _ in range(4 * n):
        if len(cells) >= n:
            break
        r, c = list(cells)[int(rng.integers(len(cells)))]
        dr, dc = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(4))]
        nr, nc = r + dr, c + dc
        if 0 <= nr < max_h and 0 <= nc < max_w and nr >= 0 and nc >= 0:
            cells.add((nr, nc))
    return sorted(cells)


def place_objects(rng, h, w, n, colours, bg=0, max_obj=4, kinds=None, gap=True, tries=60):
    g = np.full((h, w), bg, dtype=np.int8)
    placed = []
    for _ in range(n):
        for _t in range(tries):
            kind = None if kinds is None else kinds[int(rng.integers(len(kinds)))]
            cells = rand_shape(rng, min(max_obj, h), min(max_obj, w), kind)
            sh = max(r for r, _ in cells) + 1
            sw = max(c for _, c in cells) + 1
            if sh > h or sw > w:
                continue
            r0, c0 = int(rng.integers(0, h - sh + 1)), int(rng.integers(0, w - sw + 1))
            ok = True
            for r, c in cells:
                y, x = r0 + r, c0 + c
                rng_y = range(max(0, y - 1), min(h, y + 2)) if gap else range(y, y + 1)
                rng_x = range(max(0, x - 1), min(w, x + 2)) if gap else range(x, x + 1)
                if any(g[yy, xx] != bg for yy in rng_y for xx in rng_x):
                    ok = False
                    break
            if not ok:
                continue
            colour = colours[int(rng.integers(len(colours)))]
            for r, c in cells:
                g[r0 + r, c0 + c] = colour
            placed.append([(r0 + r, c0 + c) for r, c in cells])
            break
    return g, placed


def noise_grid(rng, h, w, density, colours, bg=0):
    g = np.full((h, w), bg, dtype=np.int8)
    m = rng.random((h, w)) < density
    g[m] = np.asarray(colours, dtype=np.int8)[rng.integers(len(colours), size=int(m.sum()))]
    return g


def pick_colours(rng, k, exclude=(0,)):
    pool = [c for c in range(10) if c not in exclude]
    return [int(x) for x in rng.choice(pool, size=k, replace=False)]


def side(rng, lo, hi):
    return int(rng.integers(lo, max(lo, hi) + 1))


# ----------------------------------------------------------------------------------------------
# Families: each returns (sample_input(rng) -> grid, transform(grid) -> grid | None)
# ----------------------------------------------------------------------------------------------


def objects_input(rng, S, n_lo=1, n_hi=4, colours=None, min_side=3, **kw):
    colours = colours or pick_colours(rng, int(rng.integers(1, 4)))

    def sample(r):
        h, w = side(r, min_side, S), side(r, min_side, S)
        n = int(r.integers(n_lo, n_hi + 1))
        return place_objects(r, h, w, n, colours, **kw)[0]

    return sample


def fam_geom(rng, S):
    k = int(rng.integers(1, 8))
    sample = objects_input(rng, S) if rng.random() < 0.6 else (
        lambda r: noise_grid(r, side(r, 2, S), side(r, 2, S), r.uniform(0.2, 0.7), list(range(1, 10))))
    return sample, lambda g: dihedral(g, k)


def fam_colour_map(rng, S):
    n = int(rng.integers(1, 4))
    src = pick_colours(rng, n, exclude=())
    dst = [int(rng.choice([c for c in range(10) if c != s])) for s in src]
    table = np.arange(10, dtype=np.int8)
    for s, d in zip(src, dst):
        table[s] = d
    keep = [int(c) for c in rng.permutation([c for c in range(1, 10) if c not in src])][: int(rng.integers(0, 3))]
    pal = sorted(set(src + keep) - {0}) or [1]

    def sample(r):
        return noise_grid(r, side(r, 2, S), side(r, 2, S), r.uniform(0.3, 1.0), pal)

    return sample, lambda g: table[g]


def fam_translate(rng, S):
    dy, dx = 0, 0
    while dy == 0 and dx == 0:
        dy, dx = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    wrap = bool(rng.random() < 0.4)

    def tf(g):
        if wrap:
            return np.roll(g, (dy, dx), axis=(0, 1))
        h, w = g.shape
        out = np.zeros_like(g)
        ys, xs = np.nonzero(g)
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        out[ny[ok], nx[ok]] = g[ys[ok], xs[ok]]
        return out

    return objects_input(rng, S, 1, 3), tf


def fam_gravity(rng, S):
    k = int(rng.integers(4))  # rotate so that gravity is "down" in the rotated frame

    def tf(g):
        a = np.rot90(g, k)
        out = np.zeros_like(a)
        for c in range(a.shape[1]):
            col = a[:, c][a[:, c] != 0]
            if len(col):
                out[-len(col):, c] = col
        return np.rot90(out, -k)

    sample = lambda r: noise_grid(r, side(r, 3, S), side(r, 3, S), r.uniform(0.1, 0.4), pick_colours(r, 2))
    return sample, tf


def fam_crop_bbox(rng, S):
    def tf(g):
        ys, xs = np.nonzero(g)
        if len(ys) == 0:
            return None
        return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    return objects_input(rng, S, 1, 1, min_side=5, max_obj=5), tf


def fam_upscale(rng, S):
    k = int(rng.integers(2, 4))
    small = max(2, S // k)
    sample = lambda r: noise_grid(r, side(r, 2, small), side(r, 2, small), r.uniform(0.3, 0.8), pick_colours(r, 3))
    return sample, lambda g: np.kron(g, np.ones((k, k), dtype=g.dtype))


def fam_downscale(rng, S):
    k = int(rng.integers(2, 4))
    small = max(2, S // k)

    def sample(r):
        g = noise_grid(r, side(r, 2, small), side(r, 2, small), r.uniform(0.3, 0.8), pick_colours(r, 3))
        return np.kron(g, np.ones((k, k), dtype=g.dtype))

    return sample, lambda g: g[::k, ::k]


def fam_tile(rng, S):
    m, n = 1, 1
    while m * n == 1:
        m, n = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    mirror = bool(rng.random() < 0.4)
    small_h, small_w = max(2, S // m), max(2, S // n)

    def tf(g):
        rows = []
        for i in range(m):
            row = []
            for j in range(n):
                t = g
                if mirror and i % 2:
                    t = t[::-1, :]
                if mirror and j % 2:
                    t = t[:, ::-1]
                row.append(t)
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0)

    sample = lambda r: noise_grid(r, side(r, 2, min(5, small_h)), side(r, 2, min(5, small_w)), r.uniform(0.3, 0.7),
                                  pick_colours(r, 2))
    return sample, tf


def fam_mirror_concat(rng, S):
    axis = int(rng.integers(3))

    def tf(g):
        if axis == 0:
            return np.concatenate([g, g[:, ::-1]], axis=1)
        if axis == 1:
            return np.concatenate([g, g[::-1, :]], axis=0)
        top = np.concatenate([g, g[:, ::-1]], axis=1)
        return np.concatenate([top, top[::-1, :]], axis=0)

    half = max(2, S // 2)
    sample = lambda r: noise_grid(r, side(r, 2, half), side(r, 2, half), r.uniform(0.3, 0.7), pick_colours(r, 3))
    return sample, tf


def fam_fill_enclosed(rng, S):
    fill = int(rng.integers(1, 10))
    wall = int(rng.choice([c for c in range(1, 10) if c != fill]))

    def sample(r):
        h, w = side(r, 5, S), side(r, 5, S)
        g = np.zeros((h, w), dtype=np.int8)
        for _ in range(int(r.integers(1, 4))):
            bh, bw = int(r.integers(3, max(4, min(7, h) + 1))), int(r.integers(3, max(4, min(7, w) + 1)))
            if bh > h or bw > w:
                continue
            y, x = int(r.integers(0, h - bh + 1)), int(r.integers(0, w - bw + 1))
            g[y, x:x + bw] = wall
            g[y + bh - 1, x:x + bw] = wall
            g[y:y + bh, x] = wall
            g[y:y + bh, x + bw - 1] = wall
        if r.random() < 0.5:
            g[(r.random((h, w)) < 0.05) & (g == 0)] = wall
        return g

    def tf(g):
        h, w = g.shape
        outside = np.zeros((h, w), dtype=bool)
        q = deque((y, x) for y in range(h) for x in range(w)
                  if (y in (0, h - 1) or x in (0, w - 1)) and g[y, x] == 0)
        for y, x in q:
            outside[y, x] = True
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not outside[ny, nx] and g[ny, nx] == 0:
                    outside[ny, nx] = True
                    q.append((ny, nx))
        out = g.copy()
        out[(g == 0) & ~outside] = fill
        return out

    return sample, tf


def fam_outline(rng, S):
    interior = int(rng.integers(0, 10))

    def tf(g):
        out = g.copy()
        h, w = g.shape
        for y in range(h):
            for x in range(w):
                if g[y, x] == 0:
                    continue
                nb = [g[y + dy, x + dx] if 0 <= y + dy < h and 0 <= x + dx < w else 0
                      for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
                if all(v != 0 for v in nb):
                    out[y, x] = interior
        return out

    return objects_input(rng, S, 1, 3, kinds=["rect", "rect", "blob"], max_obj=6, min_side=4), tf


def fam_rays(rng, S):
    k = int(rng.integers(4))
    src = int(rng.integers(1, 10))
    ray = src if rng.random() < 0.5 else int(rng.choice([c for c in range(1, 10) if c != src]))

    def tf(g):
        a = np.rot90(g, k).copy()
        h, w = a.shape
        out = a.copy()
        for y, x in zip(*np.nonzero(a == src)):
            for xx in range(x + 1, w):
                if a[y, xx] != 0:
                    break
                out[y, xx] = ray
        return np.rot90(out, -k)

    def sample(r):
        h, w = side(r, 4, S), side(r, 4, S)
        g = np.zeros((h, w), dtype=np.int8)
        n = int(r.integers(1, 4))
        ys, xs = r.integers(0, h, n), r.integers(0, w, n)
        g[ys, xs] = src
        if r.random() < 0.5:
            other = int(r.choice([c for c in range(1, 10) if c not in (src, ray)]))
            m = (r.random((h, w)) < 0.05) & (g == 0)
            g[m] = other
        return g

    return sample, tf


def fam_recolour_by_size(rng, S):
    largest = bool(rng.random() < 0.6)
    target = int(rng.integers(1, 10))
    base = int(rng.choice([c for c in range(1, 10) if c != target]))

    def tf(g):
        comps = components(g)
        if len(comps) < 2:
            return None
        sizes = sorted(len(c) for c in comps)
        pick = sizes[-1] if largest else sizes[0]
        if sizes.count(pick) > 1:
            return None
        out = g.copy()
        for c in comps:
            if len(c) == pick:
                for y, x in c:
                    out[y, x] = target
        return out

    return objects_input(rng, S, 2, 4, colours=[base], max_obj=4, min_side=5), tf


def fam_keep_largest(rng, S):
    crop = bool(rng.random() < 0.5)

    def tf(g):
        comps = components(g)
        if not comps:
            return None
        sizes = sorted(len(c) for c in comps)
        if len(sizes) > 1 and sizes[-1] == sizes[-2]:
            return None
        best = max(comps, key=len)
        out = np.zeros_like(g)
        for y, x in best:
            out[y, x] = g[y, x]
        if crop:
            ys, xs = zip(*best)
            return out[min(ys):max(ys) + 1, min(xs):max(xs) + 1]
        return out

    return objects_input(rng, S, 2, 4, max_obj=4, min_side=5), tf


def fam_denoise(rng, S):
    def tf(g):
        h, w = g.shape
        out = g.copy()
        for y in range(h):
            for x in range(w):
                if g[y, x] == 0:
                    continue
                same = any(0 <= y + dy < h and 0 <= x + dx < w and g[y + dy, x + dx] == g[y, x]
                           for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                if not same:
                    out[y, x] = 0
        return out

    def sample(r):
        h, w = side(r, 5, S), side(r, 5, S)
        cols = pick_colours(r, 2)
        g, _ = place_objects(r, h, w, int(r.integers(1, 4)), cols[:1], kinds=["rect"], max_obj=5)
        m = (r.random((h, w)) < 0.08) & (g == 0)
        g[m] = cols[1] if r.random() < 0.5 else cols[0]
        return g

    return sample, tf


def fam_count_bar(rng, S):
    vertical = bool(rng.random() < 0.5)
    colour = int(rng.integers(1, 10))

    def tf(g):
        n = len(components(g))
        if n == 0 or n > 9:
            return None
        bar = np.full((1, n), colour, dtype=g.dtype)
        return bar.T if vertical else bar

    return objects_input(rng, S, 1, 6, max_obj=3, min_side=4), tf


def fam_majority(rng, S):
    k = int(rng.choice([1, 1, 2, 3]))

    def tf(g):
        vals, counts = np.unique(g[g != 0], return_counts=True)
        if len(vals) == 0 or (counts == counts.max()).sum() > 1:
            return None
        return np.full((k, k), vals[counts.argmax()], dtype=g.dtype)

    sample = lambda r: noise_grid(r, side(r, 3, S), side(r, 3, S), r.uniform(0.3, 0.8), pick_colours(r, 3))
    return sample, tf


def fam_symmetrize(rng, S):
    axis = int(rng.integers(2))

    def tf(g):
        a = g if axis == 0 else g.T
        out = a.copy()
        w = a.shape[1]
        out[:, w - w // 2:] = a[:, : w // 2][:, ::-1]
        return out if axis == 0 else out.T

    sample = lambda r: noise_grid(r, side(r, 3, S), side(r, 3, S), r.uniform(0.3, 0.7), pick_colours(r, 3))
    return sample, tf


def fam_bbox_frame(rng, S):
    frame = int(rng.integers(1, 10))

    def tf(g):
        out = g.copy()
        h, w = g.shape
        for cells in components(g, by_colour=False):
            ys, xs = zip(*cells)
            y0, y1, x0, x1 = min(ys) - 1, max(ys) + 1, min(xs) - 1, max(xs) + 1
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if (y in (y0, y1) or x in (x0, x1)) and 0 <= y < h and 0 <= x < w and out[y, x] == 0:
                        out[y, x] = frame
        return out

    cols = [c for c in range(1, 10) if c != frame]
    return objects_input(rng, S, 1, 2, colours=[int(rng.choice(cols))], max_obj=3, min_side=5), tf


def fam_connect(rng, S):
    def tf(g):
        out = g.copy()
        h, w = g.shape
        for y in range(h):
            xs = np.nonzero(g[y])[0]
            for a, b in zip(xs[:-1], xs[1:]):
                if g[y, a] == g[y, b]:
                    out[y, a:b + 1] = np.where(out[y, a:b + 1] == 0, g[y, a], out[y, a:b + 1])
        for x in range(w):
            ys = np.nonzero(g[:, x])[0]
            for a, b in zip(ys[:-1], ys[1:]):
                if g[a, x] == g[b, x]:
                    out[a:b + 1, x] = np.where(out[a:b + 1, x] == 0, g[a, x], out[a:b + 1, x])
        return out

    def sample(r):
        h, w = side(r, 5, S), side(r, 5, S)
        g = np.zeros((h, w), dtype=np.int8)
        for c in pick_colours(r, int(r.integers(1, 4))):
            if r.random() < 0.5:
                y = int(r.integers(h))
                x0, x1 = sorted(r.choice(w, 2, replace=False))
                g[y, x0] = g[y, x1] = c
            else:
                x = int(r.integers(w))
                y0, y1 = sorted(r.choice(h, 2, replace=False))
                g[y0, x] = g[y1, x] = c
        return g

    return sample, tf


def fam_invert(rng, S):
    fg = int(rng.integers(1, 10))

    def tf(g):
        out = np.where(g == fg, 0, np.where(g == 0, fg, g)).astype(g.dtype)
        return out

    sample = lambda r: noise_grid(r, side(r, 2, S), side(r, 2, S), r.uniform(0.2, 0.6), [fg])
    return sample, tf


def fam_halves_logic(rng, S):
    op = ["and", "or", "xor", "nor"][int(rng.integers(4))]
    sep = int(rng.integers(1, 10))
    ca, cb, cout = pick_colours(rng, 3, exclude=(0, sep))
    vertical = bool(rng.random() < 0.5)
    hs = max(2, (S - 1) // 2)

    def sample(r):
        h, w = side(r, 2, min(S, 6)), side(r, 2, hs)
        a = noise_grid(r, h, w, 0.5, [ca])
        b = noise_grid(r, h, w, 0.5, [cb])
        s = np.full((h, 1), sep, dtype=np.int8)
        g = np.concatenate([a, s, b], axis=1)
        return g.T.copy() if vertical else g

    def tf(g):
        a_ = g.T if vertical else g
        w = (a_.shape[1] - 1) // 2
        a, b = a_[:, :w] != 0, a_[:, w + 1:] != 0
        m = {"and": a & b, "or": a | b, "xor": a ^ b, "nor": ~(a | b)}[op]
        out = np.where(m, cout, 0).astype(g.dtype)
        return out.T.copy() if vertical else out

    return sample, tf


def fam_move_to_wall(rng, S):
    k = int(rng.integers(4))

    def tf(g):
        a = np.rot90(g, k)
        ys, _ = np.nonzero(a)
        if len(ys) == 0:
            return None
        shift = a.shape[0] - 1 - ys.max()
        return np.rot90(np.roll(a, shift, axis=0), -k)

    return objects_input(rng, S, 1, 1, max_obj=4, min_side=4), tf


def fam_draw_border(rng, S):
    c = int(rng.integers(1, 10))

    def tf(g):
        out = g.copy()
        out[0, :] = out[-1, :] = c
        out[:, 0] = out[:, -1] = c
        return out

    sample = lambda r: noise_grid(r, side(r, 3, S), side(r, 3, S), r.uniform(0.0, 0.3), pick_colours(r, 2, exclude=(0, c)))
    return sample, tf


def fam_colour_by_count(rng, S):
    """Recolour each object by its cell count (1..4 -> four task-level colours)."""
    table = pick_colours(rng, 4)
    base = int(rng.choice([c for c in range(1, 10) if c not in table]))

    def tf(g):
        out = g.copy()
        for cells in components(g):
            n = len(cells)
            if n > 4:
                return None
            for y, x in cells:
                out[y, x] = table[n - 1]
        return out

    return objects_input(rng, S, 2, 5, colours=[base], max_obj=2, min_side=4), tf


def fam_flood_row(rng, S):
    """Each row (or column) that contains a marker pixel is filled with the marker colour."""
    axis = int(rng.integers(2))

    def tf(g):
        a = g if axis == 0 else g.T
        out = a.copy()
        for y in range(a.shape[0]):
            nz = a[y][a[y] != 0]
            if len(nz) == 1:
                out[y, :] = nz[0]
        return out if axis == 0 else out.T.copy()

    def sample(r):
        h, w = side(r, 3, S), side(r, 3, S)
        g = np.zeros((h, w), dtype=np.int8)
        n = int(r.integers(1, max(2, h // 2)))
        for y in r.choice(h, size=min(n, h), replace=False):
            g[y, int(r.integers(w))] = int(r.integers(1, 10))
        return g if axis == 0 else g.T.copy()

    return sample, tf


SIMPLE_FOR_COMPOSE = ["geom", "colour_map", "translate", "upscale", "gravity", "crop_bbox", "draw_border"]


def fam_compose(rng, S):
    a, b = rng.choice(SIMPLE_FOR_COMPOSE, size=2, replace=False)
    sa, ta = FAMILIES[str(a)](rng, S)
    _, tb = FAMILIES[str(b)](rng, S)

    def tf(g):
        x = ta(g)
        return None if x is None else tb(x)

    return sa, tf


FAMILIES = {
    "geom": fam_geom,
    "colour_map": fam_colour_map,
    "translate": fam_translate,
    "gravity": fam_gravity,
    "crop_bbox": fam_crop_bbox,
    "upscale": fam_upscale,
    "downscale": fam_downscale,
    "tile": fam_tile,
    "mirror_concat": fam_mirror_concat,
    "fill_enclosed": fam_fill_enclosed,
    "outline": fam_outline,
    "rays": fam_rays,
    "recolour_by_size": fam_recolour_by_size,
    "keep_largest": fam_keep_largest,
    "denoise": fam_denoise,
    "count_bar": fam_count_bar,
    "majority": fam_majority,
    "symmetrize": fam_symmetrize,
    "bbox_frame": fam_bbox_frame,
    "connect": fam_connect,
    "invert": fam_invert,
    "halves_logic": fam_halves_logic,
    "move_to_wall": fam_move_to_wall,
    "draw_border": fam_draw_border,
    "colour_by_count": fam_colour_by_count,
    "flood_row": fam_flood_row,
    "compose": fam_compose,
}


def valid(g) -> bool:
    return g is not None and g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30


def make_task(rng: np.random.Generator, family: str, n_pairs: int, max_side: int = 10, tries: int = 20):
    for _ in range(tries):
        sample, tf = FAMILIES[family](rng, max_side)
        pairs = []
        for _p in range(4 * n_pairs):
            gi = np.ascontiguousarray(sample(rng)).astype(np.int8)
            if not valid(gi):
                continue
            go = tf(gi)
            if not valid(go):
                continue
            go = np.ascontiguousarray(go).astype(np.int8)
            if max(go.shape) > max_side * 3:
                continue
            if any(gi.shape == p[0].shape and np.array_equal(gi, p[0]) for p in pairs):
                continue
            pairs.append((gi, go))
            if len(pairs) == n_pairs:
                break
        if len(pairs) < n_pairs:
            continue
        if all(i.shape == o.shape and np.array_equal(i, o) for i, o in pairs):
            continue
        return pairs
    return None


def sample_task(rng: np.random.Generator, max_side: int = 10, n_lo: int = 3, n_hi: int = 5):
    """A random family, random number of pairs; the last pair is the query."""
    names = list(FAMILIES)
    while True:
        fam = names[int(rng.integers(len(names)))]
        S = side(rng, 4, max_side)
        pairs = make_task(rng, fam, int(rng.integers(n_lo, n_hi + 1)), S)
        if pairs is not None:
            return fam, pairs
