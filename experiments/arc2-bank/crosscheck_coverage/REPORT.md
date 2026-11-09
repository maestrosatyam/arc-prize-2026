# ARC-2 cross-check: public candidate pools against the bank's account

Thu 5 Nov 2026. `scripts/arc2_crosscheck_coverage.py` → `result.json` (deterministic). Inputs: datasets `markrowset/arc-agi-ttt-candidate-pools` and `markrowset/arc-ttt-coverage-code`, read as JSON only, and the scorer of notebook `farmountain/arc2-coverage-not-selection`.

## Pools

Each pool covers the full public evaluation (120 tasks, 172 outputs), and none of them is B0. Four come from the release's rebuild of the NVARC recipe (same 4B checkpoint), and these keep only the grids that survived the deadline:
- **submitted config**: 48 views, wall matched.
- **per-input deadlines**: submitted config plus three scheduling flags.
- **16 views**: the release's hidden-entry configuration.
- **fewer views**: an older run, not wall matched.

The fifth, **full-clock notebook**, holds every grid decoded by the public notebook `koushikrudra/failed-in-aimo`. Like our bank, it keeps every decoded grid.

## Instrument check (172 outputs)

| Track | Configuration | a | Coverage | Selected | Efficiency | Empty | Tasks covered |
|---|---|---|---|---|---|---|---|
| ARC-2 | full-clock notebook | 0.5 | 30.8% | 26.2% | 85% | 14 | 32/120 |
| ARC-2 | per-input deadlines | 0.02 | 26.7% | 20.9% | 78% | 34 | 30/120 |
| ARC-2 | 16 views | 0.02 | 25.6% | 20.3% | 80% | 65 | 27/120 |
| ARC-2 | fewer views | 0.02 | 23.3% | 17.4% | 75% | 73 | 24/120 |
| ARC-2 | submitted config | 0.02 | 24.4% | 19.8% | 81% | 53 | 27/120 |

This is what the notebook prints. Fit: coverage = 31.5 − 1.12 pp per 10 empty pools, R² 0.86. Two deviations:
1. **The scorer ignores the `count` field.** Four pools store one row per grid, so the count it computes is always 1. The ranking is then file order at a = 0 and mean NLL at any a > 0, and grids without NLLs rank first. The pools' own selector reproduces every count the release publishes (for example 42 in pool, 35 hit and 53 empty for submitted config). The notebook's scorer gives 1, 1 and 4 fewer selected outputs than that selector for submitted config, 16 views and per-input deadlines. Coverage and the fit do not change; the efficiency band becomes 75–87%.
2. **`eval/data` has 167 outputs, not 172.** It lacks the first output of five bank tasks, so the scorer would misalign those tasks. We used the bank's competition copies, plus `eval/data` for the other 19 tasks, whose output counts match. Those 19 tasks enter these aggregates only.

## Account on the bank's 145 outputs

| Track | System (a*) | Right | In pool, not submitted | Never generated | Right at a = 0 | Right, own selector | Empty |
|---|---|---|---|---|---|---|---|
| ARC-2 | B0, bank | 40 | 6 | 99 | – | – | 21 |
| ARC-2 | v7 direct merge | 45 | 10 | 90 | – | – | – |
| ARC-2 | B0 pool, their scorer (0.5) | 40 | 6 | 99 | 34 | 40 | 21 |
| ARC-2 | submitted config (0.02) | 29 | 8 | 108 | 25 | 30 | 41 |
| ARC-2 | 16 views (0.02) | 28 | 8 | 109 | 19 | 29 | 55 |
| ARC-2 | per-input deadlines (0.02) | 30 | 8 | 107 | 25 | 32 | 26 |
| ARC-2 | fewer views (0.02) | 24 | 9 | 112 | 21 | 24 | 60 |
| ARC-2 | full-clock notebook (0.5) | 37 | 6 | 102 | 32 | 37 | 13 |

The a* values are the same as on 172 outputs. The full-clock notebook's own submission also gives 37 / 6 / 102. Seventy-eight outputs are in none of the seven pools: the five above, B0's and the 27B samples.

## Agreement with B0's pool

| Track | Their pool | Coverage: both / theirs only / ours only / neither (McNemar p) | Top-2 right, same split (p) |
|---|---|---|---|
| ARC-2 | submitted config | 33 / 4 / 13 / 95 (0.049) | 26 / 3 / 14 / 102 (0.013) |
| ARC-2 | 16 views | 30 / 6 / 16 / 93 (0.053) | 23 / 5 / 17 / 100 (0.017) |
| ARC-2 | full-clock notebook | 35 / 8 / 11 / 91 (0.65) | 30 / 7 / 10 / 98 (0.63) |

Outputs only submitted config covers: 2b83f449_0, 78332cb0_0, 80a900e0_0, d35bdbdc_1. Outputs only B0 covers: 135a2760_0, 36a08778_1, 4a21e3da_1, 7b0280bc_0, 800d221b_0, 8e5c0c38_0, 981571dc_0, a6f40cea_0, b5ca7ac4_0, d8e07eb2_0, d8e07eb2_1, dfadab01_0, fc7cae8d_0. The other lists are in `result.json`.

## Their attribution claims

1. **Selection efficiency is flat.** Agreed. B0's efficiency is 87% and the merge's 82%. Their scorer on B0's pool returns 40.
2. **Coverage is driven by empty pools.** This holds between configurations. On the 145 outputs the slope is −1.2 pp per 10 empty pools (R² 0.88), and B0 sits about one standard error above the line. Empty pools explain 9 of the 17 coverage disagreements with submitted config, but only 1 of the 19 with the full-clock notebook. They do not explain the level of loss: 78 of B0's 99 never-generated outputs have a non-empty pool. Even their own fit, extrapolated to zero empty pools, leaves 68% of outputs uncovered.

## What it means

Independent pools reproduce our account on the same outputs. Never generated is 102–112 for their pools against 99 for B0, and in pool but not submitted is 6–9 against 6. The generation-bound reading does not depend on our pipeline or our scorer. The two retained runs cover fewer outputs than B0 (p ≈ 0.05), so pools from the same recipe are not interchangeable. Any citation of the notebook's efficiency figures should say they come from a mean-NLL ranker.
