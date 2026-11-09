# ARC-AGI-2 with selective state-space models: Kaggle GPU runs

Setup and rules: DESIGN.md section 10. CPU results: RESULTS.md. Numbers from results_gpu_small.json
(analyze_gpu.py); htest scored only by scripts/arc2_eval_local.py (ReplaySolver) on held-out ids.

## Overall reading (2026-10-23, after the check run)

- **Per training token**, the Mamba-2/attention hybrid (about 21.8M parameters, 2 of 12 layers attention) is at
  least as good as a Transformer of the same size, and better on exact held-out outputs under TTT. This holds when
  the Transformer's learning rate is tuned: 38 and 47 against 24 exact outputs, with NLL level or slightly better.
- **Per GPU-hour with the plain-PyTorch scan**, a tuned Transformer trained 3x longer solves more held-out outputs
  (55 against 38 and 47).
- Pure Mamba-2 is level with an untuned Transformer per token.
- **Every arm scores 0/62 on dev.** No-go for an ARC-AGI-2 entry; the open questions are a tuned hybrid and fused
  scan kernels.
- The full run's larger gap (hybrid 38-47 vs Transformer 8-9 exact) was mostly the Transformer's learning rate (1e-3).

## Short run (a2-ssm-small-v1, 2026-10-22, 124 min on 4x L4)

About 21M parameters per arm, one L4 each, 36M training tokens per arm (equal tokens, not equal time), lr 1e-3,
bfloat16, blocks recomputed in backward. Evaluation on held-out tasks and synthetic tasks with all grids <= 15x15;
TTT steps chosen from {0, 8, 16} by leave-one-pair-out, batch 4, learning rate 3e-4 (picked on meta-val for every
arm). Every arm finished all stages; every item was evaluated by every arm (409 records per mode, of which the
16-30 items are skipped by design).

### Verdict

- **At this scale the CPU ordering reverses.** The hybrid (10 Mamba-2 + 2 attention layers) is ahead of the
  Transformer at equal tokens in both seeds:
  - teacher-forced NLL per token on hpair: -0.069 [-0.085, -0.055] and -0.028 [-0.046, -0.011];
  - exact outputs under TTT (hpair + htest, <= 15x15): 20 vs 10 (sign p 0.021) and 15 vs 10 (p 0.27);
  - cell accuracy under TTT: +1.7 [-2.0, +5.9] and -1.2 [-5.4, +2.9] points (not separable).
- **Pure Mamba-2 is level with the Transformer** (NLL +0.010 [-0.016, +0.036]; exact 12 vs 10 under TTT, p 0.77).
- **The seeds differ by about as much as the arms.** Hybrid seed 1 ends near where seed 0 stood at 24M tokens
  (meta-val loss 0.287 vs 0.259; generated tf-exact 35 vs 27 at 24M and 42 at 30M), so one-seed comparisons are weak.
- **Cost:** equal tokens took 1097 s of training for the Transformer and 3641 s for the hybrid (4365 s for Mamba).
  The plain-PyTorch scan is 3-4x slower per token. At equal GPU time in this implementation the Transformer would
  have seen about 3x the tokens; that comparison was not run.
- **ARC-AGI-2-only material is still out of reach:**
  - 0/53 new htest outputs for every arm and mode;
  - new hpair: Transformer with TTT 1/48 (f1bcbc2c), all others 0/48.
- **Undertrained:** all curves are still falling at 36M tokens (about 1.7 tokens per parameter).
- **Rule check** (DESIGN.md section 10, fixed before the results): every arm finished (1); the hybrid's cell-accuracy
  interval under TTT includes 0, and hybrid TTT exact on htest (9) exceeds its CPU counterpart (5) (2). The 90M-token
  run is launched, revised:
  - two seeds each of the Transformer and the hybrid, with pure Mamba dropped;
  - in-context evaluation on all grid sizes, smallest first.

### Training (36M tokens per arm)

| Arm | Params | Steps | Train wall s | gen tf-exact | meta-val tf-exact | meta-val tok-acc | hpair NLL/token (no TTT) |
|---|---|---|---|---|---|---|---|
| transformer_g s0 | 21,023,232 | 2229 | 1097 | 24/160 | 0/40 | 0.910 | 0.303 |
| hybrid_g s0 | 21,827,808 | 2229 | 3641 | 46/160 | 1/40 | 0.928 | 0.234 |
| hybrid_g s1 | 21,827,808 | 2231 | 3634 | 35/160 | 0/40 | 0.909 | 0.275 |
| mamba_g s0 | 21,544,688 | 2229 | 4365 | 25/160 | 0/40 | 0.873 | 0.313 |

Meta-val loss along training (6M-token steps):

| Arm | 6M | 12M | 18M | 24M | 30M | 36M |
|---|---|---|---|---|---|---|
| transformer_g s0 | 0.655 | 0.418 | 0.383 | 0.329 | 0.306 | 0.295 |
| hybrid_g s0 | 0.647 | 0.510 | 0.379 | 0.259 | 0.225 | 0.217 |
| hybrid_g s1 | 0.641 | 0.527 | 0.427 | 0.332 | 0.310 | 0.287 |
| mamba_g s0 | 0.612 | 0.505 | 0.443 | 0.401 | 0.379 | 0.360 |

The Transformer is ahead at 12M tokens; hybrid seed 0 passes it at 18M, hybrid seed 1 only at 36M; Mamba stays
behind from 12M on.

### Held-out training tasks, last train pair as query, grids <= 15x15 (121 queries)

| Run | exact@2 | exact@1 | shape@1 | cell acc | changed cells correct | unchanged cells correct |
|---|---|---|---|---|---|---|
| transformer_g s0, in context | 1/121 | 1/121 | 99/121 | 69.3% | 431/1834 (23.5%) | 5865/6392 (91.8%) |
| hybrid_g s0, in context | 6/121 | 3/121 | 107/121 | 75.6% | 327/1834 (17.8%) | 6193/6392 (96.9%) |
| hybrid_g s1, in context | 3/121 | 2/121 | 104/121 | 72.8% | 343/1834 (18.7%) | 6053/6392 (94.7%) |
| mamba_g s0, in context | 2/121 | 1/121 | 106/121 | 71.9% | 419/1834 (22.8%) | 5591/6392 (87.5%) |
| transformer_g s0, TTT | 4/121 | 3/121 | 112/121 | 76.7% | 517/1834 (28.2%) | 6085/6392 (95.2%) |
| hybrid_g s0, TTT | 11/121 | 6/121 | 109/121 | 78.4% | 700/1834 (38.2%) | 5963/6392 (93.3%) |
| hybrid_g s1, TTT | 6/121 | 6/121 | 105/121 | 75.5% | 588/1834 (32.1%) | 5604/6392 (87.7%) |
| mamba_g s0, TTT | 5/121 | 4/121 | 106/121 | 73.3% | 619/1834 (33.8%) | 5424/6392 (84.9%) |
| CPU transformer (0.7M), TTT | 7/121 | 5/121 | 114/121 | 78.1% | 626/1834 (34.1%) | 6102/6392 (95.5%) |
| CPU hybrid (0.7M), TTT | 8/121 | 4/121 | 112/121 | 76.3% | 585/1834 (31.9%) | 5868/6392 (91.8%) |
| copy input (no model) | 1/121 | 1/121 | 87/121 | 68.1% | 0 by definition | 6392/6392 |

### Held-out test outputs (host evaluator, 209 outputs; TTT and in-context limited to <= 15x15)

| Run | exact@2 | <=10 | 11-15 | new | inherited |
|---|---|---|---|---|---|
| transformer_g s0, in context | 4/209 | 3/68 | 1/48 | 0/53 | 4/156 |
| hybrid_g s0, in context | 3/209 | 3/68 | 0/48 | 0/53 | 3/156 |
| hybrid_g s1, in context | 3/209 | 3/68 | 0/48 | 0/53 | 3/156 |
| mamba_g s0, in context | 3/209 | 3/68 | 0/48 | 0/53 | 3/156 |
| transformer_g s0, TTT | 6/209 | 5/68 | 1/48 | 0/53 | 6/156 |
| hybrid_g s0, TTT | 9/209 | 5/68 | 4/48 | 0/53 | 9/156 |
| hybrid_g s1, TTT | 9/209 | 7/68 | 2/48 | 0/53 | 9/156 |
| mamba_g s0, TTT | 7/209 | 5/68 | 2/48 | 0/53 | 7/156 |
| CPU transformer (0.7M), TTT | 8/209 | 5/68 | 3/48 | 0/53 | 8/156 |
| CPU hybrid (0.7M), TTT | 5/209 | 4/68 | 1/48 | 0/53 | 5/156 |

### Synthetic suite (9 tasks)

| Run | in context | TTT |
|---|---|---|
| transformer_g s0 | 0/9 | 2/9 (01, 03) |
| hybrid_g s0 | 3/9 (03, 06, 09) | 4/9 (01, 03, 06, 09) |
| hybrid_g s1 | 1/9 (06) | 2/9 (01, 06) |
| mamba_g s0 | 1/9 (01) | 1/9 (01) |

### Paired comparisons with transformer_g s0 (same 121 hpair queries and htest outputs, <= 15x15)

| Comparison | exact outputs arm / transformer (arm-only, transformer-only, sign p) | cell acc difference | changed-cell acc difference |
|---|---|---|---|
| hybrid s0, in context | 9 / 5 (6, 2, 0.29) | +6.3 pts [+1.7, +11.8] | -5.7 pts [-12.9, +1.0] |
| hybrid s1, in context | 6 / 5 (3, 2, 1.0) | +3.5 pts [-0.3, +8.1] | -4.8 pts [-11.2, +2.2] |
| mamba s0, in context | 5 / 5 (3, 3, 1.0) | +2.6 pts [-2.0, +8.1] | -0.7 pts [-10.0, +7.5] |
| hybrid s0, TTT | 20 / 10 (13, 3, 0.021) | +1.7 pts [-2.0, +5.9] | +10.0 pts [+3.3, +17.2] |
| hybrid s1, TTT | 15 / 10 (9, 4, 0.27) | -1.2 pts [-5.4, +2.9] | +3.9 pts [-1.0, +8.9] |
| mamba s0, TTT | 12 / 10 (7, 5, 0.77) | -3.4 pts [-7.0, +0.4] | +5.6 pts [-2.5, +14.2] |

Teacher-forced NLL per token on hpair (200 tasks, no TTT), difference to the Transformer: hybrid s0 -0.069
[-0.085, -0.055], hybrid s1 -0.028 [-0.046, -0.011], mamba +0.010 [-0.016, +0.036].

TTT steps chosen (0 / 8 / 16): transformer 31 / 86 / 124, hybrid s0 54 / 98 / 89, hybrid s1 51 / 95 / 95,
mamba 66 / 96 / 79.

## Full run (a2-ssm-main-v2, 2026-10-22, 217 min on 4x L4)

Two seeds each of transformer_g and hybrid_g, 90M training tokens per arm, otherwise as the short run; in-context
evaluation on all grid sizes (smallest first), TTT on <= 15x15. Every arm finished every stage; all 409 items per
mode were done by every arm (results_gpu_main.json).

### Verdict

- **The hybrid's lead replicates across seeds and widens with training.** Both hybrid seeds beat both Transformer
  seeds; the two Transformer seeds agree with each other.
  - Teacher-forced NLL per token on hpair (200 tasks, no TTT), against transformer s0: hybrid s0 -0.072
    [-0.090, -0.056], hybrid s1 -0.084 [-0.101, -0.069], transformer s1 -0.006 [-0.013, +0.000].
  - Exact outputs under TTT (hpair + htest, <= 15x15): hybrid 38 and 47, Transformer 9 and 8 (sign p < 0.001 for
    each hybrid seed against transformer s0).
  - Held-out test outputs under TTT (host evaluator): hybrid 22/209 and 23/209, Transformer 4/209 and 5/209.
- **TTT helps the hybrid much more than the Transformer.** Held-out test outputs go from 12 to 22 and from 6 to 23 for
  the hybrids, but from 4 to 4 and from 4 to 5 for the Transformers. The 21M Transformer with TTT is below the 0.7M
  CPU Transformer (hpair <= 15x15: 5 and 3 vs 7 of 121).
- **First signs on ARC-AGI-2-only material, all from hybrid seed 1:**
  - 1/53 new htest outputs (f1bcbc2c) and 3/48 new hpair queries;
  - every other run has at most 1/48 new hpair and 0/53 new htest.
- **Every GPU model now beats copying the input over all sizes.** hpair cell accuracy in context: hybrid 77.7% and
  79.8%, Transformer 77.7% and 76.8%; copy 73.3%. On large grids (16-30, in context only) exact outputs are rare:
  - hpair: hybrid s0 3/79 and s1 2/79, Transformer s1 1/79 and s0 0/79;
  - htest: 1/93 for hybrid s0 only.
- **Equal tokens is not equal cost in this implementation.** Training took 2726 s per Transformer and about 9050 s
  per hybrid (3.3x, the plain-PyTorch scan). The Transformers' final meta-val loss (0.238, 0.230) is where the hybrids
  stood at roughly 27-32M tokens (between their 23M and 34M points: 0.314-0.224 and 0.255-0.195), and the
  Transformer curves are still falling. A Transformer given
  the hybrid's GPU time (about 300M tokens) was not run, so the per-GPU-hour comparison in this implementation is
  open. With fused scan kernels the hybrid's cost per token would be closer to the Transformer's (not measured here).
- **Not tuned per arm:** every arm used lr 1e-3, warmup 2%, cosine decay. A Transformer-specific learning rate could
  narrow the gap; not tested.

### Training (90M tokens per arm)

| Arm | Steps | Train wall s | gen tf-exact | meta-val loss | meta-val tok-acc | hpair NLL/token | TTT lr |
|---|---|---|---|---|---|---|---|
| transformer_g s0 | 5568 | 2726 | 33/160 | 0.238 | 0.927 | 0.260 | 3e-4 |
| transformer_g s1 | 5574 | 2728 | 34/160 | 0.230 | 0.928 | 0.254 | 1e-3 |
| hybrid_g s0 | 5568 | 9030 | 65/160 | 0.155 | 0.950 | 0.188 | 3e-4 |
| hybrid_g s1 | 5574 | 9066 | 62/160 | 0.145 | 0.950 | 0.176 | 3e-4 |

Meta-val loss along training (about 11M-token steps):

| Arm | 11M | 23M | 34M | 45M | 56M | 68M | 79M | 90M |
|---|---|---|---|---|---|---|---|---|
| transformer_g s0 | 0.471 | 0.370 | 0.338 | 0.298 | 0.273 | 0.251 | 0.243 | 0.238 |
| transformer_g s1 | 0.481 | 0.367 | 0.302 | 0.275 | 0.256 | 0.238 | 0.232 | 0.230 |
| hybrid_g s0 | 0.512 | 0.314 | 0.224 | 0.190 | 0.167 | 0.155 | 0.151 | 0.155 |
| hybrid_g s1 | 0.539 | 0.255 | 0.195 | 0.182 | 0.160 | 0.148 | 0.147 | 0.145 |

### Held-out training tasks, last train pair as query, grids <= 15x15 (121 queries)

| Run | exact@2 | exact@1 | shape@1 | cell acc | changed cells correct | unchanged cells correct |
|---|---|---|---|---|---|---|
| transformer_g s0, in context | 4/121 | 3/121 | 103/121 | 74.9% | 494/1834 (26.9%) | 6119/6392 (95.7%) |
| transformer_g s1, in context | 5/121 | 3/121 | 104/121 | 76.9% | 424/1834 (23.1%) | 6267/6392 (98.0%) |
| hybrid_g s0, in context | 9/121 | 7/121 | 106/121 | 78.2% | 440/1834 (24.0%) | 6189/6392 (96.8%) |
| hybrid_g s1, in context | 12/121 | 8/121 | 110/121 | 79.3% | 584/1834 (31.8%) | 6147/6392 (96.2%) |
| transformer_g s0, TTT | 5/121 | 2/121 | 109/121 | 76.4% | 588/1834 (32.1%) | 6154/6392 (96.3%) |
| transformer_g s1, TTT | 3/121 | 1/121 | 108/121 | 75.9% | 472/1834 (25.7%) | 6060/6392 (94.8%) |
| hybrid_g s0, TTT | 16/121 | 14/121 | 113/121 | 80.6% | 725/1834 (39.5%) | 6071/6392 (95.0%) |
| hybrid_g s1, TTT | 24/121 | 17/121 | 113/121 | 82.4% | 850/1834 (46.3%) | 6131/6392 (95.9%) |

All 200 hpair queries in context (all sizes): exact@2 transformer 4 and 6, hybrid 12 and 14; cell accuracy 77.7%,
76.8%, 77.7%, 79.8% (copy 73.3%); new (ARC-AGI-2-only) queries 0/48, 0/48, 1/48, 1/48.

### Held-out test outputs (host evaluator, 209 outputs)

| Run | exact@2 | <=10 | 11-15 | 16-30 | new | inherited |
|---|---|---|---|---|---|---|
| transformer_g s0, in context | 4/209 | 3/68 | 1/48 | 0/93 | 0/53 | 4/156 |
| transformer_g s1, in context | 4/209 | 3/68 | 1/48 | 0/93 | 0/53 | 4/156 |
| hybrid_g s0, in context | 12/209 | 8/68 | 3/48 | 1/93 | 0/53 | 12/156 |
| hybrid_g s1, in context | 6/209 | 6/68 | 0/48 | 0/93 | 1/53 | 5/156 |
| transformer_g s0, TTT | 4/209 | 3/68 | 1/48 | 0/93 | 0/53 | 4/156 |
| transformer_g s1, TTT | 5/209 | 4/68 | 1/48 | 0/93 | 0/53 | 5/156 |
| hybrid_g s0, TTT | 22/209 | 16/68 | 6/48 | 0/93 | 0/53 | 22/156 |
| hybrid_g s1, TTT | 23/209 | 16/68 | 7/48 | 0/93 | 1/53 | 22/156 |

TTT runs only on <= 15x15 tasks, so its 16-30 column is 0 by design (in context, 1/93 in total).

### Synthetic suite (9 tasks)

| Run | in context | TTT |
|---|---|---|
| transformer_g s0 | 1/9 (09) | 3/9 (01, 03, 09) |
| transformer_g s1 | 1/9 (09) | 2/9 (01, 09) |
| hybrid_g s0 | 4/9 (01, 03, 06, 09) | 4/9 (01, 03, 06, 09) |
| hybrid_g s1 | 4/9 (01, 03, 06, 09) | 4/9 (01, 03, 06, 09) |

### Paired comparisons with transformer_g s0 (same queries and outputs, <= 15x15)

| Comparison | exact outputs arm / transformer s0 (arm-only, ref-only, sign p) | cell acc difference | changed-cell acc difference |
|---|---|---|---|
| transformer s1, in context | 9 / 8 (3, 2, 1.0) | +2.0 pts [+0.0, +4.1] | -3.8 pts [-9.4, +0.1] |
| hybrid s0, in context | 20 / 8 (14, 2, 0.004) | +3.3 pts [+0.0, +6.9] | -2.9 pts [-11.9, +5.3] |
| hybrid s1, in context | 18 / 8 (13, 3, 0.021) | +4.4 pts [+0.8, +8.3] | +4.9 pts [-3.0, +12.9] |
| transformer s1, TTT | 8 / 9 (2, 3, 1.0) | -0.5 pts [-4.3, +4.2] | -6.3 pts [-15.0, +1.4] |
| hybrid s0, TTT | 38 / 9 (31, 2, <0.001) | +4.2 pts [-0.4, +9.3] | +7.5 pts [+0.3, +15.9] |
| hybrid s1, TTT | 47 / 9 (39, 1, <0.001) | +5.9 pts [+1.6, +10.9] | +14.3 pts [+5.9, +24.3] |

TTT steps chosen (0 / 8 / 16): transformer s0 33 / 74 / 134, transformer s1 95 / 57 / 89, hybrid s0 63 / 104 / 74,
hybrid s1 58 / 119 / 64.

## Check run (a2-ssm-check-v1, 2026-10-22/23, 182 min on 4x L4)

Arms and rules: DESIGN.md section 10 (fixed before the results).
Transformer arms are seed 0 (same initialisation and data order as transformer_g s0 of the full run); comparisons use
the full run's arms (results_gpu_check.json; runs/combined_check links both runs).

### Verdict

- **Dev (rule: >= 5 of 62 with TTT): 0/62 for both hybrids, in context and with TTT.**
  - Shape accuracy 73-77%; every output had two model attempts (no evaluator fallbacks).
  - The no-go for an ARC-AGI-2 entry stands.
- **Transformer rule (an arm with hpair NLL <= 0.188 and >= 38 TTT exact outputs): not met by any single arm, so by
  the rule the hybrids' lead stands.** The rule's two parts were met by different arms:
  - lr 3e-4 at 90M: NLL 0.192, 24 exact;
  - lr 1e-3 at equal time: NLL 0.190, 22 exact;
  - lr 3e-4 at equal time: NLL 0.228, 55 exact.
- **Most of the full run's gap was the Transformer's learning rate.**
  - At the same 90M tokens, lr 3e-4 instead of 1e-3 lowers the Transformer's hpair NLL from 0.260 to 0.192 and
    raises its TTT exact outputs from 9 to 24 (held-out test outputs 4 -> 13 of 209).
  - The hybrids (lr 1e-3, not tuned) stay ahead of this Transformer on TTT exact: 38 vs 24 (p 0.013) and 47 vs 24
    (p < 0.001).
  - On NLL they are level or slightly ahead: -0.004 [-0.015, +0.008] and -0.015 [-0.024, -0.008].
- **At equal GPU time in this implementation the Transformer wins on exact outputs.** It trained for 8700 s (285M
  tokens at lr 3e-4) against the hybrids' about 9050 s.
  - TTT exact: 55 vs 38 (p 0.008) and 55 vs 47 (p 0.18).
  - Held-out test outputs: 29/209 vs 22 and 23.
  - Synthetic: 5/9 vs 4/9.
  - The hybrids keep a lower hpair NLL (-0.040, -0.051), because this Transformer overfits the real training tasks
    after about 143M tokens: meta-val loss is lowest at 0.145 there and rises to 0.183, while generated-task loss keeps
    falling.
  - At lr 1e-3 the equal-time Transformer reaches NLL 0.190 but only 22 TTT exact.
- **Reading:** per training token the hybrid (Mamba-2 + 2 attention layers) is at least as good as a tuned
  Transformer, and better on exact outputs under TTT. Per GPU-hour with the plain-PyTorch scan it is not. Neither
  architecture solves dev tasks at this scale. The hybrid's learning rate was not tuned; a tuned hybrid could move
  either comparison.

### Training

| Arm | lr | Tokens | Train wall s | gen tf-exact | meta-val loss (final / best) | hpair NLL | TTT lr |
|---|---|---|---|---|---|---|---|
| transformer_g s0 (full run) | 1e-3 | 90M | 2726 | 33/160 | 0.238 / 0.238 | 0.260 | 3e-4 |
| transformer_g s0, lr 3e-4 | 3e-4 | 90M | 2729 | 54/160 | 0.169 / 0.169 | 0.192 | 3e-4 |
| transformer_g s0, equal time | 1e-3 | 286M | 8700 | 48/160 | 0.169 / 0.169 | 0.190 | 3e-4 |
| transformer_g s0, equal time | 3e-4 | 285M | 8700 | 105/160 | 0.183 / 0.145 (143M) | 0.228 | 3e-4 |
| hybrid_g s0 (full run) | 1e-3 | 90M | 9030 | 65/160 | 0.155 / 0.151 | 0.188 | 3e-4 |
| hybrid_g s1 (full run) | 1e-3 | 90M | 9066 | 62/160 | 0.145 / 0.145 | 0.176 | 3e-4 |

### Held-out, grids <= 15x15 (121 hpair queries; htest 116 outputs)

| Run | hpair exact@2 in context | hpair exact@2 TTT | hpair cell acc TTT | changed cells TTT | htest exact@2 TTT (of 209) |
|---|---|---|---|---|---|
| transformer_g s0 (lr 1e-3, 90M) | 4 | 5 | 76.4% | 588/1834 (32.1%) | 4 |
| transformer_g s0, lr 3e-4, 90M | 7 | 11 | 79.6% | 755/1834 (41.2%) | 13 |
| transformer_g s0, lr 1e-3, equal time | 7 | 8 | 80.3% | 694/1834 (37.8%) | 14 |
| transformer_g s0, lr 3e-4, equal time | 11 | 26 | 83.1% | 963/1834 (52.5%) | 29 |
| hybrid_g s0 (90M) | 9 | 16 | 80.6% | 725/1834 (39.5%) | 22 |
| hybrid_g s1 (90M) | 12 | 24 | 82.4% | 850/1834 (46.3%) | 23 |

ARC-AGI-2-only material with TTT: htest 0/53 for every Transformer arm, 1/53 for hybrid s1; hpair 3/48 for the
equal-time lr 3e-4 Transformer and hybrid s1, at most 1/48 otherwise.

### Paired comparisons of the hybrids with the lr 3e-4 Transformers (same items, <= 15x15)

| Comparison | TTT exact (sign p) | TTT cell acc | TTT changed cells | hpair NLL (no TTT) |
|---|---|---|---|---|
| hybrid s0 vs Transformer lr 3e-4, 90M | 38 vs 24 (0.013) | +1.0 pts [-1.7, +3.7] | -1.6 pts [-9.6, +5.8] | -0.004 [-0.015, +0.008] |
| hybrid s1 vs Transformer lr 3e-4, 90M | 47 vs 24 (<0.001) | +2.8 pts [+0.3, +5.5] | +5.2 pts [-3.8, +14.2] | -0.015 [-0.024, -0.008] |
| hybrid s0 vs Transformer lr 3e-4, equal time | 38 vs 55 (0.008) | -2.5 pts [-6.1, +1.1] | -13.0 pts [-23.0, -2.9] | -0.040 [-0.059, -0.023] |
| hybrid s1 vs Transformer lr 3e-4, equal time | 47 vs 55 (0.18) | -0.8 pts [-3.9, +2.5] | -6.2 pts [-16.1, +3.9] | -0.051 [-0.069, -0.036] |

### Dev (40 tasks, 62 outputs; scripts/arc2_eval_local.py --split dev, ReplaySolver; aggregate only)

| Run | exact@2 | shape (either attempt) |
|---|---|---|
| hybrid_g s0, in context | 0/62 | 72.6% |
| hybrid_g s0, TTT | 0/62 | 77.4% |
| hybrid_g s1, in context | 0/62 | 75.8% |
| hybrid_g s1, TTT | 0/62 | 77.4% |
