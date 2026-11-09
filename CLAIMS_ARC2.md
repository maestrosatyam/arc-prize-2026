# Claims table: Missing or Misranked? (ARC-AGI-2)

This table maps each number in the ARC-AGI-2 paper, "Missing or Misranked? Two Ways a Sampling Solver Loses Right
Answers" (`papers/missing-or-misranked.pdf`), to the file in this repository that holds it, with paths relative to the
repository root. The bank scripts run on CPU from the repository root (`python3 -I scripts/...`); "recorded" means the
number is read from the committed file, and "not in this release" marks a source that is not shipped, with the reason.
Public leaderboard scores are Kaggle's, kept per entry in `submissions/ledger.json` (`entry_id`, `submission_id`,
`public_score`).

## Abstract

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | v7-vr scored 34.31 on the public leaderboard | `submissions/ledger.json`, `sub-20261026-arc2-hybrid-v7vr`, submission 57646090 | recorded (Kaggle) |
| ARC-2 | 29.31 resubmitted unchanged | `submissions/ledger.json`, `sub-20261030-arc2-hybrid-v7vr`, submission 57785824 | recorded (Kaggle) |
| ARC-2 | The bank: 101 public evaluation tasks, 145 test outputs | `experiments/arc2-bank/account.json`, key `bank` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Direct merge: 45 right, 10 misranked, 90 missing | `experiments/arc2-bank/account.json`, `groups.all.v7_direct` | `python3 -I scripts/arc2_bank_profile.py` (right 45; pool 55 as rule `oracle`) |
| ARC-2 | A perfect selector would add at most 10 | `experiments/arc2-bank/profile.json`, `rules.oracle.n_right` 55 against `rules.v7_direct.n_right` 45 | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | 85% of the 145 would take 124 | 0.85 × 145 = 123.25, rounded up | recorded |
| ARC-2 | Five public pools of B0's recipe give a similar split | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.pools` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` (the public datasets of reference [K1], downloaded) |
| ARC-2 | Wider B0 search, temperature samples, more 27B samples added 4, 6 and 6 right grids to the pool | `experiments/arc2-bank/dm_result.json`, `lines.v16_f0.2_r8_kgmon.oracle` 48 against `lines.v32_f0.1_r8_kgmon.oracle` 52; `experiments/arc2-bank/ts_result.json`, `recorded.oracle` (g 45, gt2 51); `experiments/arc2-bank/c2_result.json`, `rules` (oracle 55 to 61) | recorded |
| ARC-2 | ...and changed the count submitted right by −3, 0 and −2 | `experiments/arc2-bank/dm_result.json`, `lines.<setting>.alone.all.new_right` (42 to 39); `experiments/arc2-bank/ts_result.json`, `lines.G+T2.alone.all` (39 to 39); `experiments/arc2-bank/c2_result.json`, `rules` (v7_direct 45 to 43) | recorded |
| ARC-2 | No rule registered in advance gained more than 2 outputs | `experiments/arc2-bank/b0r_result.json`, `G2.net` 2 (the best, zbeam in the merge) | `python3 -I scripts/arc2_b0r_read.py --rule b0r_zbeam` |
| ARC-2 | As much as 800 null replays reached | `experiments/arc2-bank/null_calibration.json`, `pooled_level_preserving` (`n_seeds` 800, `net.max` 2) | `python3 -I scripts/arc2_bank_null.py --seeds 200` |
| ARC-2 | The judges lost outputs | `experiments/runs/kaggle-A2-GS1-v1/gs1_analysis.json`, `gate.gs_low`; `experiments/runs/kaggle-A2-DC1-v1/dc1_analysis.json`, `gate` | recorded |
| ARC-2 | A second generator lifted B0's 40 to 45 | `experiments/arc2-bank/profile.json`, `rules.b0.n_right`, `rules.v7_direct.n_right` | `python3 -I scripts/arc2_bank_profile.py` |

## 1. Introduction

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | "Out of Reach" finds most of the loss from ARC-AGI-1 in coverage | not in this release: author-reported, reference [K1] | recorded |
| ARC-2 | The bank: 101 public evaluation tasks | `experiments/arc2-bank/account.json`, key `bank` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | A fixed 12 hours of compute | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, settings `hard_stop_s` (11 h 30 min, inside Kaggle's 12-hour limit) | recorded |
| ARC-2 | Predictions committed before each entry from Oct 31 | `DECISION_RULES_ARC2.md` | recorded |
| ARC-2 | Decision rule: net gain of at least 3 outputs with at most 1 lost | `DECISION_RULES_ARC2.md`; `experiments/arc2-bank/null_calibration.json`, key `gates` | recorded (rules frozen before each run) |
| ARC-2 | It sits above the largest gain of 800 null replays | `experiments/arc2-bank/null_calibration.json`, `pooled_level_preserving.net.max` 2 | `python3 -I scripts/arc2_bank_null.py --seeds 200` |
| ARC-2 | 19 tasks sealed for one final check | `experiments/arc2-bank/b0r_zbeam_sealed_check.json`, `result.sealed_19`; pools in `experiments/runs/kaggle-A2-bank-part1-v1/pool_sealed` and `experiments/runs/kaggle-A2-bank-part2-v1/pool_sealed` | recorded |
| ARC-2 | Figure 1: of the merge's 100 misses, 90 missing and 10 misranked | `experiments/arc2-bank/account.json`, `groups.all.v7_direct` | `python3 -I scripts/arc2_paper_figures.py` |

## 2. Setting and system

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | Two attempts per test output; 240 hidden tasks; four NVIDIA L4 GPUs; 12-hour limit | `kaggle/arc2/hybrid_vr/v7vr/kernel-metadata.json` (hardware); `experiments/arc2-bank/profile.json`, `arguments.hidden_tasks` 240 | recorded |
| ARC-2 | B0 trains on 128 sequences in the public notebook, 64 in the entry | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, first cell; `kaggle/arc2/hybrid_vr/PATCHES.md` | recorded |
| ARC-2 | 16 views (8 dihedral transforms × 2 colour permutations), probability floor 0.2, 8 rescoring augmentations, top two by kgmon | `experiments/arc2-bank/dm_result.json`, `lines.v16_f0.2_r8_kgmon.setting` (the bank's setting) | recorded |
| ARC-2 | 1,200 s task cap, including training, checked between batches | `experiments/arc2-bank/profile.json`, `throughput.cap` (`cap_s` 1200; capped tasks ran to 1465.4 s) | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | B0 starts no new task after 9 h 18 min; 27B requests until 11 h 15 min; hard stop 11 h 30 min | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, settings `phase1_end_s`, `issue_end_s`, `hard_stop_s` | recorded |
| ARC-2 | 27B: FP8 on vLLM with multi-token prediction, 81,920-token context, highest reasoning effort | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, `Q27_CONFIG` (`max_model_len`, `template_kwargs`, `server_ladder`) | recorded |
| ARC-2 | One repair round after a failed check; one rescue request after a cut reply; merge order | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, first cell; `kaggle/arc2/hybrid_vr/hybrid_vr.py` | recorded |
| ARC-2 | v7's direct merge, which the bank replays | `eval/arc2_bank.py`, rule `v7_direct` | `python3 -I scripts/arc2_bank_replay.py --rule v7_direct` |
| ARC-2 | Table 1, B0: 30.56 | `submissions/ledger.json`, `sub-20261019-arc2-b0-v1`, submission 57387075 | recorded (Kaggle) |
| ARC-2 | Table 1, H1: 28.89 | `submissions/ledger.json`, `sub-20261022-arc2-b0-half`, submission 57518886 | recorded (Kaggle) |
| ARC-2 | Table 1, v1: 31.39 | `submissions/ledger.json`, `sub-20261023-arc2-hybrid-size`, submission 57541839 | recorded (Kaggle) |
| ARC-2 | Table 1, v2: 31.81 | `submissions/ledger.json`, `sub-20261024-arc2-hybrid-v2`, submission 57582551 | recorded (Kaggle) |
| ARC-2 | Table 1, v6nb: 28.61 | `submissions/ledger.json`, `sub-20261025-arc2-hybrid-v6nb`, submission 57610305 | recorded (Kaggle) |
| ARC-2 | Table 1, v7-vr: 34.31; 29.31 (identical) | `submissions/ledger.json`, `sub-20261026-arc2-hybrid-v7vr` (57646090) and `sub-20261030-arc2-hybrid-v7vr` (57785824) | recorded (Kaggle) |
| ARC-2 | Table 1, v7: 30.97 | `submissions/ledger.json`, `sub-20261027-arc2-hybrid-v7`, submission 57682082 | recorded (Kaggle) |
| ARC-2 | Table 1, v7-vr2: 32.64; 32.22 (identical) | `submissions/ledger.json`, `sub-20261028-arc2-hybrid-v7vr2` (57716508) and `sub-20261029-arc2-hybrid-v7vr2` (57752087) | recorded (Kaggle) |
| ARC-2 | Table 1, S1: 128 training sequences, B0 phase 5,630 s longer; 31.81 | `submissions/ledger.json`, `sub-20261031-arc2-hybrid-v7vr2-s1`, submission 57837368; `DECISION_RULES_ARC2.md` | recorded (Kaggle) |
| ARC-2 | Table 1, zbeam: 32.64 | `submissions/ledger.json`, `sub-20261101-arc2-hybrid-v7vr2-zbeam`, submission 57853811 | recorded (Kaggle) |
| ARC-2 | Table 1, amax: 29.31 (after close) | `submissions/ledger.json`, `sub-20261102-arc2-hybrid-v7vr2-amax`, submission 57887325 | recorded (Kaggle) |

## 3. Where the misses are

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | B0 at 64 training sequences and 16 views on the 120 public tasks; 19 sealed; 101 tasks and 145 outputs in the bank | `experiments/arc2-bank/account.json`, key `bank`; `experiments/arc2-bank/b0r_zbeam_sealed_check.json`, `result` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Per candidate: beam score, 8 augmentation scores, kgmon rank | `experiments/runs/kaggle-A2-bank-part1-v1/pool`, `experiments/runs/kaggle-A2-bank-part2-v1/pool` | recorded |
| ARC-2 | The band: 30 shortest tasks (44 outputs); 55 samples on 29 of the 30 at a 49,152-token cap | `experiments/arc2-bank/c2_result.json`, `pools.bank` (`n_q27_tasks` 29, `n_q27_samples` 55) and `which_lever.band`; samples in `experiments/runs/kaggle-A2-q27-bank-v1/q27d` | recorded |
| ARC-2 | Table 2, all: B0 40 / 6 / 99; merge 45 / 10 / 90 | `experiments/arc2-bank/account.json`, `groups.all` | `python3 -I scripts/arc2_bank_profile.py` (right and merged pool per group) |
| ARC-2 | Table 2, shortest quarter (30, 44): 12 / 3 / 29; 17 / 7 / 20 | `experiments/arc2-bank/account.json`, `groups.q1` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Table 2, second quarter (30, 42): 11 / 1 / 30 both | `experiments/arc2-bank/account.json`, `groups.q2` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Table 2, third quarter (22, 30): 11 / 2 / 17 both | `experiments/arc2-bank/account.json`, `groups.q3` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Table 2, longest quarter (19, 29): 6 / 0 / 23 both | `experiments/arc2-bank/account.json`, `groups.q4` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Table 2, difficulty 1 (4, 5): 4 / 0 / 1; 5 / 0 / 0 | `experiments/arc2-bank/account.json`, `groups.difficulty_1` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | Table 2, difficulty 2 (18, 22): 13 / 1 / 8 both | `experiments/arc2-bank/account.json`, `groups.difficulty_2` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | Table 2, difficulty 3 (45, 63): 16 / 3 / 44; 17 / 7 / 39 | `experiments/arc2-bank/account.json`, `groups.difficulty_3` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | Table 2, difficulty 4 (25, 38): 6 / 1 / 31; 7 / 1 / 30 | `experiments/arc2-bank/account.json`, `groups.difficulty_4` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | Table 2, difficulty 5 (9, 17): 1 / 1 / 15; 3 / 1 / 13 | `experiments/arc2-bank/account.json`, `groups.difficulty_5` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | Table 2, five public pools: 24 to 37 / 6 to 9 / 102 to 112 | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.pools` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | Shortest quarter: merged pool 24 right grids against B0's 15; merge 17 against 12 | `experiments/arc2-bank/c2_result.json`, `which_lever.band` (`oracle_bank` 24, `b0_pool_right` 15); `experiments/arc2-bank/account.json`, `groups.q1` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | 7 of the merge's 10 misranked outputs in the shortest quarter | `experiments/arc2-bank/account.json`, `groups.q1.v7_direct.pool` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Never-generated share of the merge's misses: 8 of 9, 39 of 46, 30 of 31, 13 of 14 at difficulty 2 to 5 | `experiments/arc2-bank/account.json`, `groups.difficulty_2` to `groups.difficulty_5` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | RQ1: 90 of the merge's 100 misses and 99 of B0's 105 never generated | `experiments/arc2-bank/account.json`, `groups.all` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | RQ1: perfect selection 55 of 145, against 124 for 85% | `experiments/arc2-bank/profile.json`, `rules.oracle.n_right` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | RQ1: at difficulty 4 and 5, 43 of 45 misses missing | `experiments/arc2-bank/account.json`, `groups.difficulty_4`, `groups.difficulty_5` | recorded (labels: the team's own difficulty labels, not in this release) |
| ARC-2 | Five pools: four rebuilt to the deadline, one full-clock notebook | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.pools[].name`; `experiments/arc2-bank/crosscheck_coverage/REPORT.md` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | 102 to 112 missing and 6 to 9 misranked, against B0's 99 and 6 | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.pools`, `account_145.ours.b0` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | Their scorer on B0's pool returns 40 | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.b0_pool_release_scorer.right` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | 78 outputs missing from all seven pools | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.union_of_five_pools.covered_with_merge` (67 of 145 covered) | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | Rebuilt submitted configuration covers 37 against B0's 46 (McNemar p 0.049) | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.pools` (`covered`) and `agreement_145["submitted config"].coverage` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | Recurrent transducers (0.7M to 21M parameters) solved none of 53 new-task outputs, except one hybrid seed with 1 | `experiments/arc2-mamba/RESULTS.md`; `experiments/arc2-mamba/RESULTS_GPU.md` | recorded |
| ARC-2 | Program search: eight learned priors, none solved any of 85 held-out outputs; three priors scored 0 on two 40-task splits at 10 s per task | not in this release: the earlier solver's run records, not shipped | recorded |
| ARC-2 | Program search oracle check: 1,191 programs and no projection proposal | not in this release: the earlier solver's run records, not shipped | recorded |
| ARC-2 | RQ2: neither earlier solver solved more than 1 output from tasks new in ARC-AGI-2 | `experiments/arc2-mamba/RESULTS_GPU.md`; program-search records not in this release | recorded |

## 4. Why the right grid is never generated

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | Audit of B0 (128 sequences, 40 tasks, 65 outputs): 12 submitted right, attempt 1 alone 12, in the pool 15, never generated 50 | `experiments/runs/kaggle-A2-pool-audit-v1/analysis.json` (`selected_pass_at_2`, `selected_pass_at_1`, `coverage`); `experiments/arc2-pool-audit/audit_v1_analysis.txt` | recorded |
| ARC-2 | Every audit task stopped on the 0.2 floor; median 601 s, longest 1,379 s | `experiments/arc2-pool-audit/audit_v1_analysis.txt`, lines 28 and 29 | recorded |
| ARC-2 | Bank: 94 of 101 tasks ended on the floor, 7 at the 1,200 s cap, median 469 s | `experiments/arc2-bank/profile.json`, `throughput.stop_reason`, `throughput.elapsed_s.p50` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Floor 0.05: pool 15 to 19, submission 12 to 14, 1.66 times as long, 20 of 40 tasks at the cap, 10.6 h to 17.6 h | `experiments/arc2-pool-audit/w1_compare.txt`; `experiments/runs/kaggle-A2-W1-v1/compare.json` | recorded |
| ARC-2 | Unchanged repeat of the audit: 19 and 16 | `experiments/arc2-pool-audit/noise_v2_vs_v1.txt`; `experiments/runs/kaggle-A2-pool-audit-v2/compare_v1.json` | recorded |
| ARC-2 | Superset search (32 views, floor 0.1, 32 augmentations): pool 48 to 50 (×1.37), 49 (×1.43), 52 (×1.76) | `experiments/arc2-bank/dm_result.json`, `lines.<setting>.oracle` and `time_ratio_to_s0` for `v16_f0.2_r8_kgmon`, `v16_f0.1_r8_kgmon`, `v32_f0.2_r8_kgmon`, `v32_f0.1_r8_kgmon`; rule `DECISION_RULES_ARC2.md` | recorded (search kernels' archives not shipped) |
| ARC-2 | B0's attempts: 42, 39, 42 and 39 | `experiments/arc2-bank/dm_result.json`, `lines.<setting>.alone.all.new_right` | recorded |
| ARC-2 | Projected merge 48 to 36, 38 and 31 | `experiments/arc2-bank/dm_result.json`, `lines.<setting>.tcp.all.new_right` | recorded |
| ARC-2 | No setting reached a net gain on the choosing half | `experiments/arc2-bank/dm_result.json`, `selection`, `outcome` | recorded |
| ARC-2 | Temperature samples: 6 outputs the search never generated; pool 45 to 51 on 142 outputs | `experiments/arc2-bank/ts_result.json`, `recorded.oracle`, `recorded.right_grids_only_t_found`; rule `DECISION_RULES_ARC2.md` | recorded |
| ARC-2 | B0's attempts stayed at 39 (one gained, one lost), 1.99 times B0's time; one round: pool 47 at 1.50 times | `experiments/arc2-bank/ts_result.json`, `lines.G+T2.alone.all`, `lines.G+T2.time_ratio_to_g`, `recorded.oracle.gt1`, `lines.G+T1.time_ratio_to_g` | recorded |
| ARC-2 | 128 training sequences: 36 against 40 at 1.17 times the time | `experiments/arc2-bank/s1_result.json`, `line1`, `recorded.time.ratio_sum` | recorded |
| ARC-2 | Colour 0 fixed: 39 | `experiments/arc2-bank/s3_result.json`, `line1` | recorded |
| ARC-2 | B0's model adapted on 80 other tasks: 15 of 65 | `experiments/arc2-bank/b0r_dev.json`, `table.kaggle-A2-adapt-dev-v1` (`base_right`); `experiments/runs/kaggle-A2-adapt-dev-v1/submission.json` | `python3 -I scripts/arc2_b0r_dev.py` |
| ARC-2 | Halving training to 64 sequences kept 15 of 65 at 18% less time per task | `experiments/arc2-pool-audit/h1_compare.txt`; `experiments/runs/kaggle-A2-H1-v1/compare.json` | recorded |
| ARC-2 | 27B first sample right on 17 of 44 outputs in the shortest quarter, 2 of 46 beyond | not in this release: per-length table from 27B kernel runs, not shipped | recorded |
| ARC-2 | 40 development tasks: 70 of 80 samples needed a rescue; 97k generated tokens per task | not in this release: 27B kernel run, not shipped | recorded |
| ARC-2 | 50 bank tasks beyond the band: 92 of 100 samples at the 49k cap cut and rescued | `experiments/arc2-bank/c2_check_c2a.json`, `items` 2 (`n_transcripts` 100) and 4 (`n_rescues` 92) | recorded |
| ARC-2 | 98,304-token cap, next 20 tasks (26 outputs): right on 8; two 49k samples: 4 as a union | `experiments/arc2-bank/SIDE_BY_SIDE.md`, line 22 | recorded |
| ARC-2 | Table 3: own finish 8 of 10; cut and rescued 0 of 9; no grid 0 of 7; 18 misses (9 cut, 7 no grid, 2 wrong) | not in this release: 27B kernel run log, not shipped | recorded |
| ARC-2 | Cost: 79.6k tokens per task, 221.5 tokens/s with 12 in flight, 7.2 ks for 20 tasks | not in this release: 27B kernel run log, not shipped | recorded |
| ARC-2 | Full saved reasoning instead of its last 16,000 characters: the same 14 of 25 | not in this release: 27B kernel run, not shipped | recorded |
| ARC-2 | Worked programs in the rescue: net −3 over 196 paired draws | `experiments/arc2-q27/RESCUE_REPLAY_RESULTS.md`; rule `DECISION_RULES_ARC2.md` | recorded |
| ARC-2 | Forced answer at a 40,960-token budget: 9 of 25 against 14 and 10 | not in this release: 27B kernel runs, not shipped | recorded |
| ARC-2 | 27B's two direct samples right on 9 outputs, 7 of them missed by B0 in both runs; union 23 against 16 | not in this release: 27B kernel run, not shipped | recorded |
| ARC-2 | Solved tasks at ranks 9 to 31 of 40 by B0's confidence | not in this release: 27B kernel run analysis, not shipped | recorded |
| ARC-2 | 20 unseen tasks (25 outputs): B0 9, 27B 14, merge 16 (9 gained, 2 lost) | not in this release: 27B and B0 kernel runs, not shipped | recorded |
| ARC-2 | 27B as a program-writing agent: 3 of 65, all also B0's; budget used up on 38 of 40 tasks | not in this release: 27B kernel run, not shipped | recorded |
| ARC-2 | Glimmer 30B: 8 of 25 (variants 5 to 11) against 9 to 14 for the 27B | not in this release: kernel runs, not shipped | recorded |
| ARC-2 | Flash-Next: 4 of 25 at 99 tokens/s against the 27B's 334 | `experiments/arc2-bank/SIDE_BY_SIDE.md`, FN-L4 row | recorded |
| ARC-2 | RQ3: floor on 94 of 101 tasks; 92 of 100 samples beyond the band cut | `experiments/arc2-bank/profile.json`, `throughput.stop_reason`; `experiments/arc2-bank/c2_check_c2a.json`, `items` 4 | `python3 -I scripts/arc2_bank_profile.py` |

## 5. Selection against its ceiling

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | Ceiling: 6 for B0 alone (46 in its pool, 40 submitted), 10 for the merge (55, 45) | `experiments/arc2-bank/b0r_result.json`, `rank_of_truth.covered` 46; `experiments/arc2-bank/profile.json`, `rules` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | kgmon ranks the right grid first on 29, second 11, third 4, fourth 1, lower 1 | `experiments/arc2-bank/b0r_result.json`, `rank_of_truth.kgmon` | `python3 -I scripts/arc2_b0r_read.py --rule b0r_zbeam` |
| ARC-2 | B0's attempt 1 alone right on 29, both attempts 40 | `experiments/arc2-bank/profile.json`, `rules.b0.n_right_1`, `rules.b0.n_right` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Common decision rule: net +3 or more, at most 1 lost; first rules (Oct 27): net +4 and a sign test | `experiments/arc2-bank/null_calibration.json`, `gates`; `DECISION_RULES_ARC2.md` | recorded (frozen before the run) |
| ARC-2 | Six development pools (390 outputs): zbeam 90, amax 89, borda 87 against kgmon's 86 | `experiments/arc2-bank/b0r_dev.json`, `choice.pooled`; rule `DECISION_RULES_ARC2.md` | `python3 -I scripts/arc2_b0r_dev.py` |
| ARC-2 | zbeam on the bank: 41 alone (2 gained, 1 lost), 47 merged (2 gained, 0 lost) | `experiments/arc2-bank/b0r_result.json`, `G1`, `G2` | `python3 -I scripts/arc2_b0r_read.py --rule b0r_zbeam` |
| ARC-2 | amax changed attempt 2 on 29 outputs, right on none; met its own entry rule | `experiments/arc2-bank/amax_result.json`, `attempt2_changes`, `gate`, `gate_pass`; rule `DECISION_RULES_ARC2.md` | `python3 -I scripts/arc2_amax_read.py` |
| ARC-2 | Glimmer 30B likelihood on the 72 least-sure outputs: 35 (1 gained, 6 lost) | `experiments/runs/kaggle-A2-GS1-v1/gs1_analysis.json`, `gate.gs_low`; rule `DECISION_RULES_ARC2.md` | recorded |
| ARC-2 | 2B decision model: 34 to 36 under three rules, 25 under the fourth; top pick right on 15 of 42 against B0's 25 | `experiments/runs/kaggle-A2-DC1-v1/dc1_analysis.json`, `gate`, `recorded`; rule `DECISION_RULES_ARC2.md` | recorded |
| ARC-2 | 27B checking B0's top two: 10 of 25 against 14 (12 against 16 as attempt 2); 6 of 9 against 5; 3 against 7 | not in this release: 27B kernel run, not shipped | recorded |
| ARC-2 | Merge orderings leave 45; attempt-1 hits 33 and 32 against 29 | `experiments/arc2-bank/profile.json`, `rules.q27_first`, `rules.q27_agree`, `rules.v7_direct` (`n_right`, `n_right_1`) | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Program checks: positive by 1 to 2 outputs | `experiments/arc2-q27/VR_BANK_RESULTS.md`; `experiments/arc2-bank/V7VR_REPEAT_2026-10-31.md` (18 and 19 against 17); the 25-output replay (15 against 14) not in this release: 27B replay run, not shipped | recorded |
| ARC-2 | 20 routing tasks: verified programs on 6 outputs, 5 right; preferring them 15 of 25 against 14 | not in this release: 27B replay run, not shipped | recorded |
| ARC-2 | Entry as first built replayed at 13 against 16; v7-vr2 restored 16 | not in this release: replay note, not shipped | recorded |
| ARC-2 | 30 band tasks (44 outputs): v7-vr 17 and 19, v7-vr2 18 and 19, direct merge 17, B0 12 | `experiments/arc2-bank/V7VR_REPEAT_2026-10-31.md`; `experiments/arc2-q27/VR_BANK_RESULTS.md` | recorded |
| ARC-2 | Verified answers right on 25 of 33 | `experiments/arc2-q27/VR_BANK_RESULTS.md` | recorded |
| ARC-2 | Replay validity failed: 8 of 132 records timed out | `experiments/arc2-q27/VR_BANK_RESULTS.md`; rule `DECISION_RULES_ARC2.md` | recorded |
| ARC-2 | Two more 27B samples: merged pool 55 to 61, merge 45 to 43; 4 right attempts replaced, 2 gained | `experiments/arc2-bank/c2_result.json`, `headline`, `rules`, `which_lever`; rule `DECISION_RULES_ARC2.md` | recorded (sample kernels not shipped) |
| ARC-2 | Scored on 139 outputs; three tasks with a grid larger than 30 × 30 left out | `experiments/arc2-bank/c2_result.json`, `headline.report_only_tasks`; `experiments/arc2-bank/c2_check_c2a.json`, `experiments/arc2-bank/c2_check_c2b.json`, `items` 2 | recorded |
| ARC-2 | 800 replays of four null rules: no seed reached net +3; the largest +2 | `experiments/arc2-bank/null_calibration.json`, `pooled_level_preserving` | `python3 -I scripts/arc2_bank_null.py --seeds 200` |
| ARC-2 | Best rule, zbeam in the merge, at +2 | `experiments/arc2-bank/b0r_result.json`, `G2.net` | `python3 -I scripts/arc2_b0r_read.py --rule b0r_zbeam` |
| ARC-2 | RQ4: ceiling 10 outputs; best gain 2 | `experiments/arc2-bank/profile.json`, `rules`; `experiments/arc2-bank/b0r_result.json`, `G2` | `python3 -I scripts/arc2_bank_profile.py` |

## 6. Measuring without the leaderboard

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | v7-vr resubmitted byte for byte: 34.31 and 29.31 | `submissions/ledger.json`, submissions 57646090 and 57785824 | recorded (Kaggle) |
| ARC-2 | v7-vr2: 32.64 and 32.22 | `submissions/ledger.json`, submissions 57716508 and 57752087 | recorded (Kaggle) |
| ARC-2 | Four byte-identical B0 copies scored 28.06 to 32.22 | ours: `submissions/ledger.json`, submission 57387075 (30.56); the three by others: not in this release (other teams' Kaggle scores) | recorded |
| ARC-2 | Difference of two hybrid scores: SD about 3.2 (2.6 for B0-only copies) | `experiments/arc2-bank/V7VR_REPEAT_2026-10-31.md` | recorded |
| ARC-2 | Task sampling: SD 3.70 points for one public score (4,000 resamples of 120 tasks) | `experiments/arc2-bank/bootstrap.json`, `boards.b0[0].sd_points` | `python3 -I scripts/arc2_bank_bootstrap.py --n-boot 4000` |
| ARC-2 | Two builds must differ by about 4.4 points (1.96 × 2.25) | `experiments/arc2-bank/bootstrap.json`, `differences.v7_direct[0]` (`sd_points`, `mdd95_points`) | `python3 -I scripts/arc2_bank_bootstrap.py --n-boot 4000` |
| ARC-2 | Verify and refine about +1 over the direct merge, not +3.34 | `experiments/arc2-bank/V7VR_REPEAT_2026-10-31.md`; `submissions/ledger.json` (34.31 against 30.97) | recorded |
| ARC-2 | 616 views: the grid set above 0.2 differed in 54% | `experiments/arc2-pool-audit/SPEED_NOTES.md` | recorded |
| ARC-2 | Rerun of the bank setting: 39 against 39 on 142 outputs (2 gained, 2 lost); same candidates on 38 of 145, same attempts on 80 | `experiments/arc2-bank/ts_result.json`, `g.against_bank_b0`, `g.candidates_equal`, `g.selections_equal` | recorded |
| ARC-2 | Second rerun: 42 against 40 on 145 (3 gained, 1 lost) | `experiments/arc2-bank/dm_result.json`, `s0.alone_against_bank_b0` | recorded |
| ARC-2 | Null replays: net +2 in under 5% of seeds, net +3 in under 1% | `experiments/arc2-bank/null_calibration.json`, `pooled_level_preserving` (`net_needed_fp5`, `net_needed_fp1`) | `python3 -I scripts/arc2_bank_null.py --seeds 200` |
| ARC-2 | Sealed check of zbeam: net −1 on the 19 (0 gained, 1 lost); net 0 on all 172 | `experiments/arc2-bank/b0r_zbeam_sealed_check.json`, `result.sealed_19`, `result.all_172` | recorded (the seal is spent) |
| ARC-2 | Calibration: leaderboard ≈ 8.84 + 0.727 × bank, residual SD 0.42, leave-one-out 0.52 on three; four replays, two distinct bank values | `experiments/arc2-bank/calibration_ledger.json`, `fit`, `leave_one_out`, `rows`; `experiments/arc2-bank/builds.json` | `python3 -I scripts/arc2_calibration_ledger.py` |
| ARC-2 | S1: projected about 26.9; scored 31.81 | `DECISION_RULES_ARC2.md`; `submissions/ledger.json`, submission 57837368 | recorded |
| ARC-2 | zbeam: about 33.8; scored 32.64 | `DECISION_RULES_ARC2.md`; `submissions/ledger.json`, submission 57853811 | recorded |
| ARC-2 | amax: about 32.4; scored 29.31, 0.3 below the no-change band; attempt 2 right on none of 29 changed outputs | `DECISION_RULES_ARC2.md`; `submissions/ledger.json`, submission 57887325; `experiments/arc2-bank/amax_result.json`, `attempt2_changes` | recorded |
| ARC-2 | Development replays overstated gains: zbeam +4 of 390 became +1 of 145; amax +3 became 0 | `experiments/arc2-bank/b0r_dev.json`, `choice.pooled`; `experiments/arc2-bank/b0r_result.json`, `G1`; `experiments/arc2-bank/amax_result.json`, `G1` | `python3 -I scripts/arc2_b0r_dev.py` |
| ARC-2 | Figure 5: public scores with one task-sampling SD | `submissions/ledger.json`; `experiments/arc2-bank/bootstrap.json` | `python3 -I scripts/arc2_paper_figures.py` |

## 7. Where the 12 hours went

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | B0 had the machine until 9 h 18 min | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, settings `phase1_end_s` | recorded |
| ARC-2 | 128 sequences projected to 10.6 h for 240 tasks; 8.84 h at 64 | `experiments/arc2-pool-audit/h1_compare.txt` | recorded |
| ARC-2 | Throughput model: 9.96 h at 64 against the 9.30 h phase, cutting about 10 of 240 tasks | `experiments/arc2-bank/profile.json`, `throughput.reading`, `throughput.projection` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | v6nb scored 28.61, 3.20 below v2 | `submissions/ledger.json`, submissions 57610305 and 57582551 | recorded (Kaggle) |
| ARC-2 | 27B: about 334 tokens/s with 20 in flight, about 45k tokens per task, the shortest 52 tasks by 11.5 h | not in this release: 27B kernel run analysis, not shipped | recorded |
| ARC-2 | More 27B samples: band pool 24 to 28, beyond 31 to 33; 10 of the 27B's 12 right grids beyond the band duplicate B0's | `experiments/arc2-bank/c2_result.json`, `which_lever.band`, `which_lever.beyond` | recorded |
| ARC-2 | Hybrid against a Transformer at equal training tokens: 38 and 47 against 24 | `experiments/arc2-mamba/RESULTS_GPU.md`; `experiments/arc2-mamba/results_gpu_check_ref_transformer_g_s0_lr3e-4.json` | recorded |
| ARC-2 | One Transformer seed at the same training time: 55 against 38 (p 0.008) and 47 (p 0.18) | `experiments/arc2-mamba/RESULTS_GPU.md`; `experiments/arc2-mamba/results_gpu_check_ref_transformer_g_s0_eqtime_lr3e-4.json` | recorded |
| ARC-2 | No top-10 final notebook public on Nov 3; public notebooks gave 11 to 12 hours on 4×L4 to one model | not in this release: public notebooks read from code, not run | recorded |
| ARC-2 | Rank 2: second solver +6 to 7 points; 200k to 300k tokens per conversation; about 46M tokens per run | not in this release: author-reported, reference [K2] | recorded |
| ARC-2 | Rank 4: two tries of up to 200k tokens | not in this release: author-reported, reference [K3] | recorded |
| ARC-2 | Rank 9: two passes that vote; enough tokens (256k) | not in this release: author-reported, reference [K4] | recorded |
| ARC-2 | Those budgets are four to six times the 27B's 49k cap | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, `Q27_CONFIG.max_tokens` 49152; budgets from references [K2] to [K4] | recorded |
| ARC-2 | The same 27B: 26.81 with two 56k attempts, 44.03 after fine-tuning on its own traces | not in this release: author-reported, reference [K6] | recorded |
| ARC-2 | Ranks 2 and 4 serve Flash-Next at 3 and 2 bits | not in this release: author-reported, references [K2] and [K3] | recorded |
| ARC-2 | Our Flash-Next serving reached 99 tokens/s | `experiments/arc2-bank/SIDE_BY_SIDE.md`, FN-L4 row | recorded |
| ARC-2 | RQ5: checks gained 1 to 2 outputs | `experiments/arc2-q27/VR_BANK_RESULTS.md`; `experiments/arc2-bank/V7VR_REPEAT_2026-10-31.md` | recorded |

## 8. Related work

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | "Out of Reach": five NVARC-style pools, efficiency flat, most of the loss in coverage | not in this release: author-reported, reference [K1] | recorded |
| ARC-2 | On our outputs its pools reproduce our account | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145`; `experiments/arc2-bank/crosscheck_coverage/REPORT.md` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | The pools' own selector selects 1 to 4 more than the release's ranker | `experiments/arc2-bank/crosscheck_coverage/result.json`, `instrument`, `account_145.pools[].count_field` | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |
| ARC-2 | 78 of B0's 99 missing outputs have non-empty pools | `experiments/arc2-bank/crosscheck_coverage/result.json`, `account_145.ours.b0` (`never` 99, `empty` 21) | `python3 -I scripts/arc2_crosscheck_coverage.py --pools <pools> --code <code>` |

## 9. Discussion

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | Shortest quarter: pool holds the right grid for 24 of 44 outputs, 7 misranked; elsewhere 31 of 101 and 3 | `experiments/arc2-bank/account.json`, `groups.q1` to `groups.q4` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Fisher p 0.08, two-sided | computed from `experiments/arc2-bank/account.json` (Fisher exact test on [[7, 17], [3, 28]]) | recorded |
| ARC-2 | A second model with its own attempt added 5 outputs | `experiments/arc2-bank/profile.json`, `rules.b0.n_right` 40, `rules.v7_direct.n_right` 45 | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | More samples without a check lost 2 | `experiments/arc2-bank/c2_result.json`, `rules` (v7_direct 45 to 43) | recorded |
| ARC-2 | A refiner added 4.6 points (95% CI 1.7 to 8.3) | not in this release: author-reported, reference [K2] | recorded |
| ARC-2 | Rank 4 answers only from a function that passes every training pair | not in this release: author-reported, reference [K3] | recorded |
| ARC-2 | Gating on verification: about 87% right | not in this release: author-reported, reference [K5] | recorded |
| ARC-2 | Our verified answers: 25 of 33 | `experiments/arc2-q27/VR_BANK_RESULTS.md` | recorded |

## 10. Conclusion

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | 90 of the merge's 100 misses never generated | `experiments/arc2-bank/account.json`, `groups.all.v7_direct` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | Selection had at most 10 outputs to recover | `experiments/arc2-bank/profile.json`, `rules.oracle`, `rules.v7_direct` | `python3 -I scripts/arc2_bank_profile.py` |
| ARC-2 | The 12 hours | `kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb`, settings `hard_stop_s` | recorded |

## Appendix B. Two earlier solvers

| Track | Claim | Source in this repository | How to recompute |
|---|---|---|---|
| ARC-2 | Recurrent study: about 0.7M parameters on CPU, 7,200 CPU s per model, one seed | `experiments/arc2-mamba/RESULTS.md` | recorded |
| ARC-2 | About 21M parameters on GPU, one or two seeds, 36M to 285M training tokens | `experiments/arc2-mamba/RESULTS_GPU.md`; `experiments/arc2-mamba/results_gpu_main.json` | recorded |
| ARC-2 | Table B1: CPU 0.7M, 237 held-out outputs up to 15×15: GRU 5, Mamba-2 7, hybrid 13 against 15 | `experiments/arc2-mamba/RESULTS.md`; `experiments/arc2-mamba/results.json`; `experiments/arc2-mamba/stats.json` | recorded |
| ARC-2 | Table B1: 53 held-out outputs from tasks new in ARC-AGI-2: 0; one GPU hybrid seed 1 | `experiments/arc2-mamba/RESULTS.md`; `experiments/arc2-mamba/RESULTS_GPU.md` | recorded |
| ARC-2 | Table B1: GPU hybrids, two seeds, 0 and 0 of 62 on dev | `experiments/arc2-mamba/RESULTS_GPU.md` | recorded |
| ARC-2 | Table B1: equal tokens, 38 and 47 against 24 | `experiments/arc2-mamba/RESULTS_GPU.md`; `experiments/arc2-mamba/results_gpu_check_ref_transformer_g_s0_lr3e-4.json` | recorded |
| ARC-2 | Table B1: equal GPU time, 38 and 47 against 55 (p 0.008, 0.18) | `experiments/arc2-mamba/RESULTS_GPU.md`; `experiments/arc2-mamba/results_gpu_check_ref_transformer_g_s0_eqtime_lr3e-4.json` | recorded |
| ARC-2 | Program search: eight learned priors of 2,205 to 6,705 parameters, reading 244 pooled statistics | not in this release: the earlier solver's records, not shipped | recorded |
| ARC-2 | Priors named the program kind of held-out generated tasks with 71% to 78% accuracy | not in this release: the earlier solver's records, not shipped | recorded |
| ARC-2 | Table B1: 85 validation outputs from tasks new in ARC-AGI-2: 0 for each prior | not in this release: the earlier solver's records, not shipped | recorded |
| ARC-2 | Table B1: logistic, state-space and JEPA-inspired priors at 10 s per task: 0 of 62 (dev) and 0 of 53 (tune) | not in this release: the earlier solver's run records, not shipped | recorded |
| ARC-2 | Logistic prior's answers matched the unguided search on all 233 new training tasks | not in this release: the earlier solver's records, not shipped | recorded |
| ARC-2 | Table B1: oracle family and perception on one training task: 1,191 programs, no projection proposal | not in this release: the earlier solver's records, not shipped | recorded |
| ARC-2 | Dev and tune are 40-task splits of the public evaluation set | `splits/holdout_policy.md`; `splits/arc2_eval_split_v1.json` | recorded |
