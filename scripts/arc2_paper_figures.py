"""Figures for the ARC-AGI-2 Solution Writeup (reports/paper/ARC2_WRITEUP.md) and the ARC-AGI-2 paper
(reports/paper/arc2/PAPER_ARC2.md).

Reads repository files only and writes reports/paper/figures/arc2_*: the Writeup's arc2_cover.png (unchanged), and the
paper's five figures as vector PDF (for the ACL template) and 300 dpi PNG (for the HTML build), 6.3 in wide with 8-10 pt
text and no title inside: arc2_account, arc2_difficulty, arc2_pool_picks, arc2_selection, arc2_board. Needs matplotlib.

    python3 -I scripts/arc2_paper_figures.py

Figures and inputs:
  arc2_cover.png        where each output ends up, B0 alone, the merge, and the merge by task-length quartile
                        experiments/arc2-bank/profile.json (rules b0, v7_direct and oracle on the 101 unsealed bank
                        tasks, 145 outputs, overall and by task-length quartile); B0_POOL_RIGHT below (B0's own pool
                        holds the right grid on 46 of 145 outputs; copied from experiments/arc2-bank/C2_RULE.md line 13
                        and B0R_RULE.md line 27)
  arc2_account          the paper's Figure 1: the same account as arc2_cover, at print size
  arc2_difficulty.png   the merge's account by the release's player difficulty label (1-5)
                        experiments/arc2-bank/account.json (scripts/arc2_paper_account.py)
  arc2_pool_picks.png   generation interventions: the right grid in the pool against submitted and right, each against
                        its own reference on the same outputs
                        experiments/arc2-bank/dm_result.json, ts_result.json, s1_result.json, s3_result.json,
                        c2_result.json
  arc2_selection.png    selection rules against the base and the pool ceiling
                        experiments/arc2-bank/b0r_result.json, amax_result.json, profile.json,
                        null_calibration.json; experiments/runs/kaggle-A2-GS1-v1/gs1_analysis.json,
                        experiments/runs/kaggle-A2-DC1-v1/dc1_analysis.json
  arc2_board.png        public scores of the ARC-AGI-2 entries, with the predictions committed before scoring
                        submissions/ledger.json; experiments/arc2-bank/bootstrap.json (task-sampling SD of one public
                        reading); PREDICTIONS below (copied from the prediction files named there)

Each output is counted once: submitted and right; right grid in the candidate pool but not submitted; right grid
never generated. Colours: slots 1-2 of the validated categorical palette (blue, orange) plus a grey remainder; blue
always marks submitted and right, orange the right grid in the pool. Segments of 3 or more carry their count; the
paper's tables give every count.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "experiments/arc2-bank"
PROFILE = BANK / "profile.json"
FIGS = ROOT / "reports/paper/figures"
OUT = FIGS / "arc2_cover.png"
B0_POOL_RIGHT = 46

# Predictions committed before each submission was scored (the expected public reading):
#   S1     experiments/arc2-bank/S1_ENTRY_PREDICTION_2026-10-31.md, H1 (the offline tools' time-cut projection)
#   zbeam  experiments/arc2-bank/ZBEAM_ENTRY_PREDICTION_2026-10-31.md ("about 33.8")
#   amax   experiments/arc2-bank/AMAX_ENTRY_PREDICTION_2026-11-02.md, H0 (no change, "about 32.4")
PREDICTIONS = {"57837368": 26.9, "57853811": 33.8, "57887325": 32.4}

# Board entries in submission order: (ledger entry ids, label)
ENTRIES = [
    (["sub-20261019-arc2-b0-v1"], "B0 (128 training sequences)"),
    (["sub-20261022-arc2-b0-half"], "H1: B0 with 64"),
    (["sub-20261023-arc2-hybrid-size"], "v1: H1, then the 27B direct"),
    (["sub-20261024-arc2-hybrid-v2"], "v2: v1 with second samples"),
    (["sub-20261025-arc2-hybrid-v6nb"], "v6nb: B0 cut to fit"),
    (["sub-20261026-arc2-hybrid-v7vr", "sub-20261030-arc2-hybrid-v7vr"], "v7-vr: verify and refine (final)"),
    (["sub-20261027-arc2-hybrid-v7"], "v7: v2's B0, 27B direct"),
    (["sub-20261028-arc2-hybrid-v7vr2", "sub-20261029-arc2-hybrid-v7vr2"], "v7-vr2: written grids first"),
    (["sub-20261031-arc2-hybrid-v7vr2-s1"], "S1: v7-vr2, 128 sequences"),
    (["sub-20261101-arc2-hybrid-v7vr2-zbeam"], "zbeam: v7-vr2 + picker (final)"),
    (["sub-20261102-arc2-hybrid-v7vr2-amax"], "amax: v7-vr2 + picker"),
]

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#d3d1ca"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rows() -> list[tuple[str, int, int, int]]:
    """(label, right, in pool not submitted, never generated) per bar."""
    p = load(PROFILE)
    groups = p["breakdown"]["groups"]
    allr = groups["all"]["rules"]
    n = groups["all"]["n_outputs"]
    out = [
        ("B0 alone (4B, test-time training)", allr["b0"]["n_right"], B0_POOL_RIGHT - allr["b0"]["n_right"],
         n - B0_POOL_RIGHT),
        ("B0 + 27B, v7's direct merge", allr["v7_direct"]["n_right"],
         allr["oracle"]["n_right"] - allr["v7_direct"]["n_right"], n - allr["oracle"]["n_right"]),
    ]
    names = {"q1": "shortest quarter", "q2": "second quarter", "q3": "third quarter", "q4": "longest quarter"}
    for q in ("q1", "q2", "q3", "q4"):
        g = groups[q]
        r = g["rules"]
        out.append((f"merge, {names[q]} ({g['n_outputs']})", r["v7_direct"]["n_right"],
                    r["oracle"]["n_right"] - r["v7_direct"]["n_right"], g["n_outputs"] - r["oracle"]["n_right"]))
    assert out[0][1] + out[0][2] + out[0][3] == n == 145, out[0]
    return out


def style(plt) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})


def bare(ax, keep_left: bool = False) -> None:
    for s in ("top", "right") + (() if keep_left else ("left",)):
        ax.spines[s].set_visible(False)


def titles(fig, title: str, sub: str) -> None:
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.975)
    fig.text(0.02, 0.9, sub, ha="left", fontsize=10, color=INK2)


def stacked(ax, data, ys, heights, label_x) -> None:
    for (label, right, pool, never), y, hh in zip(data, ys, heights):
        left = 0
        for val, col, ink in ((right, BLUE, "white"), (pool, ORANGE, INK), (never, GREY, INK)):
            if val:
                ax.barh(y, val, left=left, height=hh, color=col, edgecolor=SURFACE, linewidth=2)
                if val >= 3:
                    ax.text(left + val / 2, y, str(val), ha="center", va="center", fontsize=9, color=ink)
            left += val
        ax.text(label_x, y, label, ha="right", va="center", fontsize=9.5, color=INK)


def account_legend(ax, plt, x: float) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREY)]
    ax.legend(handles, ["submitted and right", "right grid in the pool, not submitted", "right grid never generated"],
              loc="lower center", bbox_to_anchor=(x, 1.0), ncol=3, frameon=False, fontsize=9,
              handlelength=1.2, columnspacing=1.4)


def cover(plt) -> None:
    data = rows()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ys = [6.4, 5.2, 3.4, 2.5, 1.6, 0.7]
    h = [0.9, 0.9, 0.7, 0.7, 0.7, 0.7]
    stacked(ax, data, ys, h, -2)
    ax.text(-2, 4.25, "By task length (outputs)", ha="right", va="center", fontsize=9, color=INK2, style="italic")
    ax.set_xlim(0, 145)
    ax.set_ylim(0.1, 7.1)
    ax.set_yticks([])
    bare(ax)
    ax.set_xlabel("public evaluation outputs (101 tasks, 145 outputs)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    account_legend(ax, plt, 0.32)
    titles(fig, "Nine in ten missed ARC-AGI-2 answers were never generated",
           "Where each output ends up, B0 alone and the B0 + 27B merge, on the offline output bank")
    fig.subplots_adjust(left=0.33, right=0.98, top=0.78, bottom=0.12)
    fig.savefig(OUT, facecolor=SURFACE)
    plt.close(fig)
    print(OUT.relative_to(ROOT), [d[1:] for d in data])


# ---- Paper figures (ACL specification): true printed width, 8-10 pt text, vector PDF plus a 300 dpi PNG for the
# HTML build, no title inside the figure (the paper's caption carries it), descriptive labels. The misranked segment
# carries a 45-degree hatch so it reads in grayscale print. arc2_cover.png above stays as the Writeup's figure.

WIDTH = 6.3
PAPER_RC = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
            "axes.labelsize": 8.5, "xtick.color": INK2, "xtick.labelsize": 8, "ytick.color": INK,
            "legend.fontsize": 8, "pdf.fonttype": 42, "hatch.linewidth": 0.9}
PT_PER_IN = 72


def save(fig, name: str) -> list[Path]:
    outs = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = FIGS / f"{name}.{ext}"
        fig.savefig(out, facecolor=SURFACE, **kw)
        outs.append(out)
    return outs


def paper_stacked(ax, data, ys, height, label_x, pt_per_unit) -> None:
    """Stacked account bars; a count sits inside a segment only where it fits with padding."""
    for (label, right, pool, never), y in zip(data, ys):
        left = 0
        for val, col, ink, hatch in ((right, BLUE, "white", None), (pool, ORANGE, INK, "////"),
                                     (never, GREY, INK, None)):
            if val:
                ax.barh(y, val, left=left, height=height, color=col, edgecolor=SURFACE, linewidth=1.2, hatch=hatch)
                if val * pt_per_unit >= len(str(val)) * 5.2 + 6:
                    ax.text(left + val / 2, y, str(val), ha="center", va="center", fontsize=8, color=ink,
                            bbox=None if hatch is None else dict(boxstyle="round,pad=0.28", fc=ORANGE, ec="none"))
            left += val
        ax.text(label_x, y, label, ha="right", va="center", fontsize=8.5, color=INK)


def paper_account_legend(ax, plt, x: float) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor=SURFACE),
               plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor=SURFACE, hatch="////"),
               plt.Rectangle((0, 0), 1, 1, facecolor=GREY, edgecolor=SURFACE)]
    ax.legend(handles, ["submitted and right", "misranked: in the pool, not submitted", "missing: never generated"],
              loc="lower center", bbox_to_anchor=(x, 1.02), ncol=3, frameon=False, handlelength=1.4,
              handleheight=1.0, columnspacing=1.2)


def account(plt) -> None:
    data = rows()
    labels = ["B0 alone", "B0 + 27B merge", "shortest quarter", "second quarter", "third quarter", "longest quarter"]
    data = [(f"{lab} ({d[1] + d[2] + d[3]})" if i >= 2 else lab,) + d[1:] for i, (lab, d) in enumerate(zip(labels, data))]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    left_in, right_in = 1.45, 0.08
    ys = [5.6, 4.7, 3.1, 2.3, 1.5, 0.7]
    pt_per_unit = (WIDTH - left_in - right_in) * PT_PER_IN / 145
    paper_stacked(ax, data, ys, 0.66, -2, pt_per_unit)
    ax.text(-2, 3.85, "merge, by task length", ha="right", va="center", fontsize=8, color=INK2, style="italic")
    ax.set_xlim(0, 145)
    ax.set_ylim(0.2, 6.05)
    ax.set_yticks([])
    bare(ax)
    ax.set_xlabel("outputs (101 public evaluation tasks, 145 outputs)")
    paper_account_legend(ax, plt, 0.42)
    fig.subplots_adjust(left=left_in / WIDTH, right=1 - right_in / WIDTH, top=0.86, bottom=0.17)
    outs = save(fig, "arc2_account")
    plt.close(fig)
    print([o.relative_to(ROOT) for o in outs], [d[1:] for d in data])


def difficulty(plt) -> None:
    acc = load(BANK / "account.json")["groups"]
    data = []
    for d in range(1, 6):
        g = acc[f"difficulty_{d}"]
        m = g["v7_direct"]
        data.append((f"{d}  ({g['n_tasks']} tasks, {g['n_outputs']} outputs)", m["right"], m["pool"], m["never"]))
    assert sum(r[1] for r in data) == 45 and sum(r[2] for r in data) == 10 and sum(r[3] for r in data) == 90
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    left_in, right_in = 1.85, 0.08
    ys = [4.4, 3.5, 2.6, 1.7, 0.8]
    pt_per_unit = (WIDTH - left_in - right_in) * PT_PER_IN / 63
    paper_stacked(ax, data, ys, 0.66, -0.9, pt_per_unit)
    ax.text(-0.9, 5.15, "player difficulty", ha="right", va="center", fontsize=8, color=INK2, style="italic")
    ax.set_xlim(0, 63)
    ax.set_ylim(0.3, 5.4)
    ax.set_yticks([])
    bare(ax)
    ax.set_xlabel("outputs, B0 + 27B merge")
    paper_account_legend(ax, plt, 0.33)
    fig.subplots_adjust(left=left_in / WIDTH, right=1 - right_in / WIDTH, top=0.84, bottom=0.2)
    outs = save(fig, "arc2_difficulty")
    plt.close(fig)
    print([o.relative_to(ROOT) for o in outs], [d[1:] for d in data])


def pool_picks_rows() -> list[dict]:
    """Per intervention: label, outputs, (pool base, pool new), (picks base, picks new), time ratio, reference."""
    dm = load(BANK / "dm_result.json")
    ts = load(BANK / "ts_result.json")
    s1 = load(BANK / "s1_result.json")
    s3 = load(BANK / "s3_result.json")
    c2 = load(BANK / "c2_result.json")
    s0 = dm["lines"]["v16_f0.2_r8_kgmon"]
    out = []
    for key, label in (("v16_f0.1_r8_kgmon", "B0 search, floor 0.1 instead of 0.2"),
                       ("v32_f0.2_r8_kgmon", "B0 search, 32 views instead of 16"),
                       ("v32_f0.1_r8_kgmon", "B0 search, both changes")):
        ln = dm["lines"][key]
        out.append({"label": label, "n": ln["alone"]["all"]["n"], "pool": (s0["oracle"], ln["oracle"]),
                    "picks": (ln["alone"]["all"]["base_right"], ln["alone"]["all"]["new_right"]),
                    "time": ln["time_ratio_to_s0"], "ref": "same run"})
    gt2 = ts["lines"]["G+T2"]
    out.append({"label": "B0 search + temperature samples", "n": gt2["alone"]["all"]["n"],
                "pool": (ts["recorded"]["oracle"]["g"], ts["recorded"]["oracle"]["gt2"]),
                "picks": (gt2["alone"]["all"]["base_right"], gt2["alone"]["all"]["new_right"]),
                "time": gt2["time_ratio_to_g"], "ref": "same run"})
    for res, line, label in ((s1, s1["line1"], "B0, 128 training sequences"),
                             (s3, s3["line1"], "B0, colour 0 fixed")):
        rec = res["recorded"]
        new = line.get("s1_right", line.get("s3_right"))
        out.append({"label": label, "n": res["n_outputs"],
                    "pool": (rec["b0_pool_oracle"]["bank"], rec["b0_pool_oracle"].get("s1", rec["b0_pool_oracle"].get("s3"))),
                    "picks": (line["base_right"], new), "time": rec["time"]["ratio_sum"], "ref": "store"})
    out.append({"label": "27B, two more samples per task", "n": c2["headline"]["n_outputs"],
                "pool": (c2["rules"]["bank"]["oracle"], c2["rules"]["c2"]["oracle"]),
                "picks": (c2["rules"]["bank"]["v7_direct"], c2["rules"]["c2"]["v7_direct"]),
                "time": None, "ref": "store"})
    return out


def dumbbell(ax, y, a, b, col, marker, dy=0.0) -> None:
    ax.plot([a, b], [y + dy, y + dy], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    ax.scatter([a], [y + dy], s=26, marker=marker, facecolor=SURFACE, edgecolor=col, linewidth=1.4, zorder=3)
    ax.scatter([b], [y + dy], s=26, marker=marker, facecolor=col, edgecolor=SURFACE, linewidth=1.0, zorder=4)


def pool_picks(plt) -> None:
    data = pool_picks_rows()
    fig, ax = plt.subplots(figsize=(WIDTH, 3.15))
    left_in = 2.25
    ys = list(range(len(data)))[::-1]
    for r, y in zip(data, ys):
        (pa, pb), (ka, kb) = r["pool"], r["picks"]
        dumbbell(ax, y, pa, pb, ORANGE, "o", 0.18)
        dumbbell(ax, y, ka, kb, BLUE, "s", -0.18)
        ax.text(max(pa, pb) + 0.7, y + 0.18, f"{pa} → {pb}", va="center", fontsize=8, color=INK)
        ax.text(max(ka, kb) + 0.7, y - 0.18, f"{ka} → {kb}", va="center", fontsize=8, color=INK)
        note = (f"time ×{r['time']:.2f}, " if r["time"] else "") + f"{r['n']} outputs"
        ax.text(33.4, y + 0.14, r["label"], ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(33.4, y - 0.24, note, ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xlim(34, 65.5)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_yticks([])
    bare(ax)
    ax.set_xlabel("outputs")
    handles = [plt.Line2D([0], [0], color=ORANGE, lw=1.6, marker="o", markersize=5, markerfacecolor=ORANGE),
               plt.Line2D([0], [0], color=BLUE, lw=1.6, marker="s", markersize=5, markerfacecolor=BLUE),
               plt.Line2D([0], [0], color=INK2, lw=0, marker="o", markersize=5, markerfacecolor=SURFACE,
                          markeredgewidth=1.2)]
    ax.legend(handles, ["right grid in the pool", "submitted and right", "reference → intervention"],
              loc="lower center", bbox_to_anchor=(0.34, 1.0), ncol=3, frameon=False, handlelength=1.8,
              columnspacing=1.2)
    fig.subplots_adjust(left=left_in / WIDTH, right=0.985, top=0.9, bottom=0.13)
    outs = save(fig, "arc2_pool_picks")
    plt.close(fig)
    print([o.relative_to(ROOT) for o in outs], [(r["label"], r["pool"], r["picks"], r["time"]) for r in data])


def selection_rows() -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict]:
    b0r = load(BANK / "b0r_result.json")
    amax = load(BANK / "amax_result.json")
    prof = load(PROFILE)["rules"]
    nul = load(BANK / "null_calibration.json")
    gs1 = load(ROOT / "experiments/runs/kaggle-A2-GS1-v1/gs1_analysis.json")["gate"]
    dc1 = load(ROOT / "experiments/runs/kaggle-A2-DC1-v1/dc1_analysis.json")["gate"]
    alone = [("z-score picker", b0r["G1"]["right"]),
             ("views − worst picker", amax["G1"]["right"]),
             ("30B likelihood re-ranking", gs1["gs_low"]["n_right"]),
             ("2B judge, keep attempt 1", dc1["dc_keep1"]["n_right"]),
             ("2B judge, keep 1, all lengths", dc1["dc_keep1_long"]["n_right"]),
             ("2B judge, within its window", dc1["dc_fit"]["n_right"]),
             ("2B judge, all lengths", dc1["dc_long"]["n_right"])]
    merged = [("z-score picker", b0r["G2"]["right"]),
              ("views − worst picker", amax["G2"]["right"]),
              ("27B grid first", prof["q27_first"]["n_right"]),
              ("27B first if its samples agree", prof["q27_agree"]["n_right"])]
    ref = {"b0": prof["b0"]["n_right"], "b0_pool": B0_POOL_RIGHT, "merge": prof["v7_direct"]["n_right"],
           "merge_pool": prof["oracle"]["n_right"],
           "null_max": nul["pooled_level_preserving"]["net"]["max"]}
    return alone, merged, ref


def selection(plt) -> None:
    alone, merged, ref = selection_rows()
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.7), sharex=True,
                             gridspec_kw={"height_ratios": [len(alone) + 0.75, len(merged) + 0.75], "hspace": 0.3})
    left_in = 2.0
    for ax, data, base, pool, name in ((axes[0], alone, ref["b0"], ref["b0_pool"], "(a) B0 alone"),
                                       (axes[1], merged, ref["merge"], ref["merge_pool"], "(b) B0 + 27B merge")):
        ys = list(range(len(data)))[::-1]
        ax.axvspan(base, base + ref["null_max"], color=GREY, alpha=0.6, lw=0, zorder=0)
        ax.axvline(base, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.axvline(pool, color=ORANGE, lw=1.8, zorder=1)
        for (label, v), y in zip(data, ys):
            ax.plot([base, v], [y, y], color="#b9b7b0", lw=1.4, zorder=2)
            ax.scatter([v], [y], s=24, color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            right_side = v >= base
            ax.text(v + (0.55 if right_side else -0.55) + (ref["null_max"] if v in (base, base + 1) and right_side else 0),
                    y, str(v), ha="left" if right_side else "right", va="center", fontsize=8, color=INK)
            ax.text(21.3, y, label, ha="right", va="center", fontsize=8.5, color=INK)
        top = len(data) - 0.2
        ax.text(pool + 0.4, top, f"ceiling {pool}", ha="left", va="center", fontsize=8, color=INK)
        ax.text(base - 0.4, top, f"base {base}", ha="right", va="center", fontsize=8, color=INK2)
        ax.text(21.3, top, name, ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
        ax.set_ylim(-0.6, len(data) + 0.15)
        ax.set_yticks([])
        bare(ax)
    axes[1].set_xlim(22, 61)
    axes[1].set_xlabel("outputs submitted and right, of 145")
    handles = [plt.Line2D([0], [0], color=BLUE, lw=0, marker="o", markersize=5),
               plt.Line2D([0], [0], color=INK2, lw=1.0, ls=(0, (3, 2))),
               plt.Line2D([0], [0], color=ORANGE, lw=1.8),
               plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.6)]
    axes[0].legend(handles, ["rule", "base", "pool ceiling", "largest gain in 800 null replays"],
                   loc="lower center", bbox_to_anchor=(0.36, 1.0), ncol=4, frameon=False, handlelength=1.6,
                   columnspacing=1.1)
    fig.subplots_adjust(left=left_in / WIDTH, right=0.985, top=0.91, bottom=0.12)
    outs = save(fig, "arc2_selection")
    plt.close(fig)
    print([o.relative_to(ROOT) for o in outs], alone, merged, ref)


LEADERBOARD_LABELS = ["B0, public notebook (128 sequences)", "B0 at 64 training sequences",
                      "+ 27B, one direct sample", "+ second 27B samples, short tasks",
                      "B0 cut to fit (¼ training, ½ views)", "27B with verify and refine (final)",
                      "27B direct, deadline caps", "verify and refine, written grids first",
                      "the same, B0 at 128 sequences", "the same + z-score picker (final)",
                      "the same + views − worst picker"]


def board_rows() -> tuple[list[tuple[str, list[float], list[str]]], float]:
    led = {e["entry_id"]: e for e in load(ROOT / "submissions/ledger.json")["entries"]}
    out = []
    assert len(LEADERBOARD_LABELS) == len(ENTRIES)
    for (ids, _), label in zip(ENTRIES, LEADERBOARD_LABELS):
        scores = [float(led[i]["public_score"]) for i in ids]
        subs = [str(led[i]["submission_id"]) for i in ids]
        out.append((label, scores, subs))
    sd = load(BANK / "bootstrap.json")["boards"]
    return out, sd


def board(plt) -> None:
    data, boards = board_rows()
    sd = next(b["sd_points"] for b in boards["b0"] if b["n_tasks_resampled"] == 120)
    assert abs(sd - 3.70) < 0.01, sd
    fig, ax = plt.subplots(figsize=(WIDTH, 3.35))
    left_in = 2.3
    ys = list(range(len(data)))[::-1]
    for (label, scores, subs), y in zip(data, ys):
        for s in scores:
            ax.plot([s - sd, s + sd], [y, y], color=GREY, lw=1.8, solid_capstyle="round", zorder=1)
        if len(scores) == 2:
            ax.plot(scores, [y, y], color=BLUE, lw=1.6, zorder=2)
        ax.scatter(scores, [y] * len(scores), s=24, color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        txt = " and ".join(f"{s:.2f}" for s in scores) + (" (identical)" if len(scores) == 2 else "")
        ax.text(max(scores) + sd + 0.4, y, txt, va="center", fontsize=8, color=INK)
        for sub in subs:
            if sub in PREDICTIONS:
                ax.scatter([PREDICTIONS[sub]], [y], s=28, marker="D", facecolor=SURFACE, edgecolor=INK2,
                           linewidth=1.2, zorder=4)
        ax.text(22.7, y, label, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(23, 47.5)
    ax.set_xticks([25, 30, 35, 40, 45])
    ax.set_ylim(-0.7, len(data) - 0.4)
    ax.set_yticks([])
    bare(ax)
    ax.set_xlabel("public leaderboard score (points, public half of the hidden tasks)")
    handles = [plt.Line2D([0], [0], color=BLUE, lw=0, marker="o", markersize=5),
               plt.Line2D([0], [0], color=GREY, lw=1.8),
               plt.Line2D([0], [0], color=INK2, lw=0, marker="D", markersize=5, markerfacecolor=SURFACE,
                          markeredgewidth=1.2)]
    ax.legend(handles, ["score", f"± one task-sampling SD ({sd:.2f})", "prediction committed before scoring"],
              loc="lower center", bbox_to_anchor=(0.32, 1.0), ncol=3, frameon=False, handlelength=1.6)
    fig.subplots_adjust(left=left_in / WIDTH, right=0.985, top=0.91, bottom=0.13)
    outs = save(fig, "arc2_board")
    plt.close(fig)
    print([o.relative_to(ROOT) for o in outs], [(d[0], d[1]) for d in data])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        style(plt)
        cover(plt)
    with plt.rc_context(PAPER_RC):
        account(plt)
        difficulty(plt)
        pool_picks(plt)
        selection(plt)
        board(plt)


if __name__ == "__main__":
    main()
