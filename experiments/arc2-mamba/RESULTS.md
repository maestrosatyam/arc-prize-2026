# ARC-AGI-2 with selective state-space models: results

Run 2026-10-22, CPU only (2 threads, nice 19), torch 2.14.1+cpu.
Split splits.json sha256 8b8d1f33. Design, hypotheses and the go rule:
DESIGN.md. Numbers below come from results.json (analyze.py), stats.json (stats.py), runs/*/tf_eval.json and
runs/*/val_log.jsonl; tables are printed by report.py.

## Verdict

Update 2026-10-23: the Kaggle GPU runs (RESULTS_GPU.md) qualify the ordering below.
- At about 21M parameters and equal training tokens, the Mamba-2/attention hybrid is at least as good as a
  Transformer whose learning rate is tuned, and better on exact held-out outputs under TTT.
- At equal GPU time with the plain-PyTorch scan, the tuned Transformer solves more.
- Dev is 0/62 for the best hybrids.

The CPU comparison below stands as measured at 0.7M parameters and equal CPU time.

**No-go for a Mamba-based ARC-AGI-2 entry before 2026-11-02.**

- **Pure Mamba-2 trails a matched Transformer.** At 0.67M parameters and 7200 CPU seconds each, it is worse on
  every held-out measure, with and without TTT. On held-out outputs with grids <= 15x15 the Transformer solves
  15 and Mamba 7 with TTT (sign test p 0.02). Cell accuracy is 9.4 points lower (95% CI -12.5 to -6.3), and
  teacher-forced NLL is 0.078 per token higher (CI 0.045 to 0.115). The gap persists at equal training tokens.
- **The hybrid (5 Mamba-2 layers + 1 attention layer) cannot be separated from the Transformer once both use TTT.**
  - With TTT: 13 vs 15 exact outputs (p 0.75), cell accuracy -1.8 points (CI -4.8 to +1.3).
  - Without TTT it is worse: cell accuracy -5.9 points (CI -9.9 to -2.4), NLL +0.077 (CI 0.053 to 0.101).
  - The best outcome this supports is parity with attention, not a gain.
- **No arm solves any ARC-AGI-2-only held-out test output** (0/53 for every run). Every held-out solve is an
  ARC-AGI-1-inherited task except f1bcbc2c (new, solved as a held-out pair only). The dev split was therefore not evaluated
  (DESIGN.md section 6: dev only if held-out results justify it).
- **Go rule (DESIGN.md section 2):**
  - Clause 1 (Mamba or hybrid at least level with the Transformer under TTT): fails for Mamba. For the hybrid it
    holds only within noise; point estimates are below on both measures.
  - Clause 2 (a solve beyond single-transform tasks): met, e.g. 44f52bb0 (shrink:summary), dae9d2b5
    (shrink:panels), d10ecb37 (shrink:crop).
  - Taken with the time budget, the rule gives no-go:
    - B0 already uses about 10.6 h of the 12 h on 4xL4.
    - A hybrid at parity would only help as a cheaper backbone for B0-style search, which needs a model as accurate
      as B0's. That requires ARC fine-tuning at a scale not available before the deadline.

## Training (equal budget: 7200 process-CPU seconds per arm)

| Arm | Params | Steps | Sequences | Tokens | Wall s | Paused s (load guard) | gen tf-exact | gen tok-acc | meta-val tf-exact | meta-val tok-acc |
|---|---|---|---|---|---|---|---|---|---|---|
| mamba | 668,048 | 3779 | 39,014 | 16,762,214 | 5005 | 1380 | 14/160 | 0.859 | 0/19 | 0.824 |
| transformer | 667,264 | 7323 | 75,558 | 32,421,885 | 3630 | 0 | 45/160 | 0.918 | 1/19 | 0.915 |
| gru | 705,280 | 2963 | 30,478 | 13,123,674 | 3747 | 0 | 5/160 | 0.803 | 0/19 | 0.735 |
| hybrid | 722,616 | 4741 | 48,873 | 20,987,014 | 3619 | 0 | 35/160 | 0.903 | 0/19 | 0.861 |

gen = 160 generated validation tasks (fixed seed, all families); meta-val = 19 real meta-val tasks that fit.
tf-exact = every query-output token correct under teacher forcing (a proxy; decoded exact on generated tasks was not
measured).

Equal training tokens (validation points nearest to the same token count, from val_log.jsonl):

| Arm | Tokens | gen loss | gen tf-exact | meta-val loss | meta-val tok-acc |
|---|---|---|---|---|---|
| mamba | 10.3M | 0.432 | 11/160 | 0.506 | 0.799 |
| transformer | 11.2M | 0.316 | 32/160 | 0.314 | 0.896 |
| hybrid | 10.5M | 0.384 | 15/160 | 0.441 | 0.850 |
| gru | 9.8M | 0.537 | 5/160 | 0.661 | 0.725 |
| mamba | 16.8M | 0.368 | 14/160 | 0.461 | 0.824 |
| transformer | 15.5M | 0.285 | 31/160 | 0.287 | 0.905 |
| hybrid | 15.7M | 0.290 | 31/160 | 0.420 | 0.847 |
| gru (final) | 13.1M | 0.516 | 5/160 | 0.653 | 0.735 |

## Held-out training tasks, last train pair as query (hpair, 200 queries)

| Run | exact@2 | exact@1 | shape@1 | cell acc | attempted | <=10 | 11-15 | 16-30 | new | inherited | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy input (no model) | 1/200 | 1/200 | 141/200 | 73.3% | 200/200 | 1/72 | 0/49 | 0/79 | 0/48 | 1/152 | 0 |
| mamba, in context | 2/200 | 1/200 | 118/200 | 51.2% | 194/200 | 2/72 | 0/49 | 0/79 | 0/48 | 2/152 | 448 |
| transformer, in context | 7/200 | 6/200 | 123/200 | 50.1% | 194/200 | 5/72 | 1/49 | 1/79 | 1/48 | 6/152 | 494 |
| gru, in context | 1/200 | 1/200 | 90/200 | 49.1% | 194/200 | 1/72 | 0/49 | 0/79 | 0/48 | 1/152 | 214 |
| hybrid, in context | 2/200 | 1/200 | 119/200 | 50.5% | 194/200 | 2/72 | 0/49 | 0/79 | 0/48 | 2/152 | 507 |
| mamba-130m-hf, text prompt | 1/72 | 1/72 | 13/72 | 15.7% | 72/72 | 1/72 | - | - | 0/14 | 1/58 | 2162 |

Columns <=10 ... inherited are exact@2 by largest grid side and by ARC-AGI-1 inheritance. Six queries do not fit in
2048 tokens with one context pair; they count as wrong. Cell accuracy = attempt-1 cells equal at the same position
over truth cells. Over all sizes every model is below copying the input (73.3%), mostly from wrong shapes on large
grids. The pretrained LM was run on <=10x10 tasks only.

TTT runs only on tasks whose grids are all <= 15x15, so in-context and TTT are compared on those 121 queries:

| Run | exact@2 | exact@1 | shape@1 | cell acc | changed cells correct | unchanged cells correct | CPU s |
|---|---|---|---|---|---|---|---|
| copy input (no model) | 1/121 | 1/121 | 87/121 | 68.1% | 0 by definition | 6392/6392 | 0 |
| mamba, in context | 2/121 | 1/121 | 105/121 | 65.6% | 382/1834 (20.8%) | 5447/6392 (85.2%) | 138 |
| mamba, TTT | 2/121 | 1/121 | 106/121 | 68.7% | 511/1834 (27.9%) | 5240/6392 (82.0%) | 773 |
| transformer, in context | 6/121 | 5/121 | 107/121 | 76.2% | 550/1834 (30.0%) | 6178/6392 (96.7%) | 149 |
| transformer, TTT | 7/121 | 5/121 | 114/121 | 78.1% | 626/1834 (34.1%) | 6102/6392 (95.5%) | 623 |
| gru, in context | 1/121 | 1/121 | 83/121 | 65.2% | 243/1834 (13.2%) | 3928/6392 (61.5%) | 82 |
| gru, TTT | 2/121 | 1/121 | 87/121 | 67.7% | 448/1834 (24.4%) | 3933/6392 (61.5%) | 3572 |
| hybrid, in context | 2/121 | 1/121 | 107/121 | 70.3% | 346/1834 (18.9%) | 5692/6392 (89.0%) | 125 |
| hybrid, TTT | 8/121 | 4/121 | 112/121 | 76.3% | 585/1834 (31.9%) | 5868/6392 (91.8%) | 929 |

Changed / unchanged cells: the 86 same-shape queries among the 121, split by whether the truth cell differs from
the query input (a wrong-shape prediction scores 0 on both).

## Held-out training tasks, real test outputs (htest, host evaluator)

Predictions from label-free TaskViews, scored by scripts/arc2_eval_local.py --split train with
arcssm.solver:ReplaySolver (runs/*/host_htest_*), restricted to the 200 held-out ids (209 test outputs). Every
solve is among the run's own candidates (none from the evaluator's fallback grids).

| Run | exact@2 | exact@1 | attempted | <=10 | 11-15 | 16-30 | new | inherited | CPU s |
|---|---|---|---|---|---|---|---|---|---|
| mamba, in context | 2/209 | 2/209 | 198/209 | 2/68 | 0/48 | 0/93 | 0/53 | 2/156 | 508 |
| transformer, in context | 6/209 | 5/209 | 198/209 | 4/68 | 2/48 | 0/93 | 0/53 | 6/156 | 676 |
| gru, in context | 1/209 | 1/209 | 198/209 | 1/68 | 0/48 | 0/93 | 0/53 | 1/156 | 300 |
| hybrid, in context | 3/209 | 1/209 | 198/209 | 3/68 | 0/48 | 0/93 | 0/53 | 3/156 | 622 |
| mamba, TTT | 5/209 | 4/209 | 116/209 | 4/68 | 1/48 | 0/93 | 0/53 | 5/156 | 1210 |
| transformer, TTT | 8/209 | 7/209 | 116/209 | 5/68 | 3/48 | 0/93 | 0/53 | 8/156 | 951 |
| gru, TTT | 3/209 | 2/209 | 116/209 | 3/68 | 0/48 | 0/93 | 0/53 | 3/156 | 5539 |
| hybrid, TTT | 5/209 | 3/209 | 116/209 | 4/68 | 1/48 | 0/93 | 0/53 | 5/156 | 1341 |
| mamba-130m-hf, text prompt | 1/209 | 1/209 | 68/209 | 1/68 | - | - | 0/53 | 1/156 | 2370 |

No in-context run solves a 16-30 output, so the TTT rows lose nothing in exact@2 by skipping them.

## Synthetic suite (9 tasks, 1 test output each)

| Run | exact@2 | shape@1 | cell acc | solved | CPU s |
|---|---|---|---|---|---|
| copy input (no model) | 0/9 | 7/9 | 88.1% | none | 0 |
| mamba, in context | 1/9 | 7/9 | 79.2% | 06_count_bar | 18 |
| transformer, in context | 2/9 | 6/9 | 89.2% | 03_rays_east, 09_fill_enclosed_bigger | 23 |
| gru, in context | 1/9 | 3/9 | 77.3% | 06_count_bar | 11 |
| hybrid, in context | 1/9 | 7/9 | 77.3% | 06_count_bar | 18 |
| mamba, TTT | 1/9 | 7/9 | 62.0% | 06_count_bar | 77 |
| transformer, TTT | 3/9 | 6/9 | 74.1% | 01_marker_colour, 03_rays_east, 09_fill_enclosed_bigger | 64 |
| gru, TTT | 0/9 | 5/9 | 62.0% | none | 353 |
| hybrid, TTT | 2/9 | 7/9 | 70.5% | 01_marker_colour, 06_count_bar | 88 |
| mamba-130m-hf, text prompt | 0/9 | 0/9 | 26.9% | none | 600 |

TTT skips 04_shape_legend (grids up to 15x20), which scores 0 of its 300 cells; that accounts for most of the
lower TTT cell accuracy. Three of the four synthetic tasks solved by any run have a related generator family
(rays, fill enclosed, count bar; generators.FAMILY_NOTES).

## Paired comparisons with the Transformer (stats.json)

Same queries, grids <= 15x15. Exact counts are hpair + htest outputs; intervals are task-level bootstrap 95%.

| Comparison | exact: arm / transformer (arm-only, transformer-only, sign p) | cell acc difference | changed-cell acc difference |
|---|---|---|---|
| mamba, in context | 4 / 12 (1, 9, 0.021) | -10.6 pts [-14.5, -7.1] | -9.2 pts [-18.6, -0.5] |
| mamba, TTT | 7 / 15 (1, 9, 0.021) | -9.4 pts [-12.5, -6.3] | -6.3 pts [-15.3, +3.2] |
| hybrid, in context | 5 / 12 (2, 9, 0.065) | -5.9 pts [-9.9, -2.4] | -11.1 pts [-18.9, -3.3] |
| hybrid, TTT | 13 / 15 (4, 6, 0.754) | -1.8 pts [-4.8, +1.3] | -2.2 pts [-10.2, +6.1] |
| gru, in context | 2 / 12 (1, 11, 0.006) | -11.0 pts [-15.4, -6.3] | -16.7 pts [-28.3, -6.6] |
| gru, TTT | 5 / 15 (1, 11, 0.006) | -10.4 pts [-13.9, -6.9] | -9.7 pts [-18.4, -0.8] |

Teacher-forced NLL per token on hpair (193 tasks that fit, no TTT; tf_eval.py), difference to the Transformer:
mamba +0.078 [0.045, 0.115], hybrid +0.077 [0.053, 0.101], gru +0.182 [0.137, 0.231].

## Size

Teacher-forced NLL per token on hpair by largest grid side (tf_eval.json):

| Arm | <=10 | 11-15 | 16-30 | all (193 tasks) | token acc | tf-exact |
|---|---|---|---|---|---|---|
| mamba | 0.494 | 0.418 | 0.575 | 0.523 | 0.815 | 2/193 |
| transformer | 0.278 | 0.275 | 0.567 | 0.444 | 0.859 | 5/193 |
| hybrid | 0.369 | 0.344 | 0.643 | 0.521 | 0.834 | 1/193 |
| gru | 0.640 | 0.583 | 0.640 | 0.626 | 0.773 | 1/193 |

- Summed over the nine runs, the 66 exact held-out outputs (hpair + htest) split 55 at <=10, 10 at 11-15 and 1 at
  16-30. The denominators per run are 140 at <=10, 97 at 11-15 and 172 at 16-30.
- Grids above 15 use row and column positions never seen in training (generated tasks <= 12, real tasks
  <= 15). There the Transformer's advantage disappears (0.567 vs 0.575) and the hybrid is worst.
- Exact accuracy is near zero above 10x10 for every arm (H4 holds).

## TTT details

- Learning rate picked on 8 meta-val tasks (tune_ttt.py): 1e-3 for Mamba, 3e-4 for Transformer, GRU and hybrid.
- Steps chosen by leave-one-pair-out (counts over all TTT records, grid {0, 4, 8, 16}):
  - mamba 0:49, 4:47, 8:62, 16:83
  - transformer 0:24, 4:22, 8:52, 16:143
  - gru 0:18, 4:19, 8:39, 16:165
  - hybrid 0:15, 4:20, 8:50, 16:156
- 16 steps, the upper end of the grid, is the most frequent choice for every arm, so the grid's cap may limit TTT.
  Longer TTT was not run.
- TTT raises changed-cell accuracy for every arm (+4.1 to +13.0 points on the 121 queries), lowers unchanged-cell
  accuracy for Mamba and the Transformer, and raises it for the hybrid.

## Pretrained Mamba LM (option 4)

state-spaces/mamba-130m-hf (revision 1e76775f) with a few-shot text prompt (digits separated by spaces, one row per
line), greedy, attempt 2 from the transposed task:
- **Solves:** 1/72 hpair and 1/68 htest outputs (<=10x10), 0/9 synthetic. Both solves are 44f52bb0, whose output is a
  single cell taking one of two colours across the task's pairs.
- **Output quality:** 124 of 298 generations parse as a rectangular grid; shape@1 13/72; changed cells 7/1048.
- **Cost:** 5131 CPU seconds; the plain-PyTorch fallback of transformers' Mamba implementation is used (mamba-ssm
  not installed).

H5 holds: no evidence of few-shot transduction at this size.

## Hypotheses

| | Outcome |
|---|---|
| H1 learnability | Met in the teacher-forced proxy: every arm reaches non-zero tf-exact on generated tasks (5-45 of 160); decoded exact on generated tasks not measured |
| H2 Mamba not worse than Transformer | Rejected for Mamba (all held-out measures, both modes, intervals exclude 0 except changed cells under TTT). Hybrid: worse in context, not separable under TTT |
| H3 TTT helps every arm | Exact@2 on the 121 queries: mamba 2->2, transformer 6->7, gru 1->2, hybrid 2->8; htest outputs 2->5, 6->8, 1->3, 3->5. Changed-cell accuracy up for every arm. Gains are small in absolute terms |
| H4 size | Holds |
| H5 pretrained 130M | Holds |

## Limitations

- **Scale:** one seed per arm, 0.7M parameters, 7200 CPU seconds. Exact counts are small; the paired intervals
  are the main evidence.
- **Training data and positions:** training grids are <= 15 (real) and <= 12 (generated). Results on larger grids
  measure position extrapolation as much as the architectures.
- **Throughput:** the Mamba arm trained on half the Transformer's tokens at equal CPU time; plain-PyTorch Mamba-2
  on CPU is slower per token than SDPA attention at these lengths (DESIGN.md section 7). At equal tokens the gap
  remains (training table).
- **TTT grid:** the step grid stops at 16, which most tasks chose.
- **ARC-AGI-2-only material:** 48 hpair queries and 53 htest outputs are ARC-AGI-2-only. One query is solved
  (f1bcbc2c, by the Transformer and the hybrid) and no htest output.

## Reproduction

```
cd experiments/arc2-mamba
python tests/test_scan.py && python tests/test_data.py && python tests/test_infer.py
./run_all.sh          # 4 x 7200 CPU-s training, in-context eval, TTT lr pick, TTT eval (synth, htest)
./run_stage2.sh       # TTT on hpair, teacher-forced eval, mamba-130m probe (<=10x10)
python pretrained_probe.py --sets synth --max-side 20
python analyze.py && python stats.py && python report.py
```

Wall time: about 8.5 h (including a 23-minute load pause). Run artefacts are in runs/ (model.pt per arm, 2.6-2.8 MB). Resume
checkpoints were removed after training.
