# ARC-AGI-2 decision rules and predictions

Each ARC-AGI-2 decision rule below was written and frozen before its run, and is given with its frozen date and,
where recorded, the commit it was frozen at. Selection rules decide by replays on the output bank (`eval/arc2_bank.py`, `RULES` and
`get_rule`: 101 public evaluation tasks, 145 test outputs, 19 more tasks sealed) against a named base, under gate v1
(net >= 4 and one-sided sign test p < 0.05 on the discordant outputs) or gate v2 (net >= 3 and lost <= 1 inside a
pre-registered family); the generation rules set their own thresholds, quoted with each. Outcomes are copied from the
result files in this repository, named under each rule.

## 1. The bank's first replay rules

**Question:** Does any registered way of choosing the two attempts from B0's pool and the 27B's samples get more
outputs right than B0 as submitted?

**Compared:** `b0` (B0 as submitted: kgmon top two; the base), `b0_kgmon` (kgmon top two recomputed from the pool;
must equal `b0`), `b0_probmul3` (B0's alternative ranking, probmul3, top two), `v7_direct` (hybrid v7's merge:
attempt_2 = first 27B sample that differs from attempt_1; the entry's own rule), `q27_first` (27B's first grid as
attempt_1, B0's attempt_1 second), `q27_agree` (when the two 27B samples agree, their grid first and B0's attempt_1
second; else v7's merge), `oracle` (coverage bound: any correct grid in the pool or the samples first; reads labels,
never adoptable). Null rules `null_<family>:<seed>` calibrate the gate and need no row.

**Measure:** right outputs (pass@2) of the 145 unsealed outputs, with Kaggle's solutions copy as truth; gained, lost
and net against the base on the same outputs; the exact one-sided sign test on the discordant outputs.

**Decision rule:** A rule is scored on the bank only after its row is in the pre-registration table, dated; a rule
without a row is marked `not pre-registered` in the reuse ledger (`replay_ledger.jsonl`) and its result is a
screening number, not a candidate for adoption. Gate v1, as pre-registered: net >= 4 and one-sided sign test
p < 0.05 on the discordant outputs; every replay also reports gate v2. Each row's status moves registered ->
screened -> adopted / declined and is never deleted.

**Frozen:** 2026-10-27 (the rows' registration date).

**Outcome** (`replay_ledger.jsonl`; base `b0`, 40 of 145):

| Track | Rule | Right of 145 | Gained | Lost | Net | Sign test p | Gate v1 |
|---|---|---:|---:|---:|---:|---:|---|
| ARC-2 | b0_kgmon | 40 | 0 | 0 | 0 | | no (equals b0) |
| ARC-2 | b0_probmul3 | 40 | 0 | 0 | 0 | | no |
| ARC-2 | v7_direct | 45 | 7 | 2 | 5 | 0.08984 | no |
| ARC-2 | q27_first | 45 | 7 | 2 | 5 | 0.08984 | no |
| ARC-2 | q27_agree | 45 | 7 | 2 | 5 | 0.08984 | no |
| ARC-2 | oracle (bound) | 55 | 15 | 0 | 15 | 3e-05 | bound, not adoptable |

Null calibration (`null_calibration.json`): 800 replays of the four level-preserving null families (`null_jitter`,
`null_jitter1`, `null_jitter2`, `null_q27swap`) gave net from -5 to +2; false-positive rate 0.0 under gate v1 and
0.0 under gate v2; the net needed for a 5% false-positive rate is 2, for 1% it is 3.

**Files:** `eval/arc2_bank.py`, `scripts/arc2_bank_replay.py`, `scripts/arc2_bank_null.py`,
`experiments/arc2-bank/replay_ledger.jsonl`, `experiments/arc2-bank/null_calibration.json`,
`experiments/arc2-bank/profile.json`.

## 2. VR-bank: the entry's verify-and-refine stages on the band

**Question:** Measured on the entry's own samples against truth, does the repair round earn its tokens, is the
verified-first merge precise enough, and are there enough cut samples to replay the rescue request?

**Compared:** v7-vr2's phase 2 unchanged (solver `vr`, merge `verified_first`, the entry's competition caps: mean
60,000 tokens per task; think1 43,152, think2 18,000, answer 6,000, rescue 8,192), in two independent replicates A
and B on the bank's 30 band tasks (the 30 shortest; 44 outputs), each merged over B0's attempts (`b0`); `b0` and
`v7_direct` reported beside.

**Measure:** net right outputs of the repair round: an output (task, test index, replicate) that a repair or
rescue_repair sample has right while neither the replicate's first sample (s1 and its rescue) nor B0's pool (its two
attempts) has it right, counted over A and B; verified-clean precision (over test outputs of samples whose program
passed every demonstration and carries no lookup table, the share whose program output is right); cut samples with a
saved reasoning tail.

**Decision rule:**
- Validity (otherwise: no decision, report only): both replicates ran s1 on at least 27 of 30 tasks; median s1
  completion tokens at least 1,000; empty or zero-token replies at most 5% of requests; no sealed task read.
- 1, repair round. Decided only with at least 20 repair rounds over A and B; otherwise "too few", report only (v8
  keeps v7-vr2's setting). 4 or more net right outputs of the repair round: the repair round stays in hybrid v8 at its
  current caps; 0 or 1: the repair round is dropped from v8 and its tokens go to second samples on the shortest
  quarter; 2 or 3: report only (v8 keeps v7-vr2's setting).
- 2, verified-first merge. Decided only with at least 10 verified-clean program outputs over A and B; otherwise
  report only. Precision at least 90%: `verified_first` stays in v8; precision below 75%: v8 merges with `R0` (B0's
  attempt_1 kept; the 27B's best answer takes attempt_2); between: report only (v8 keeps `verified_first`).
- 3, paired rescue replay go/no-go. The replay is built only with at least 15 cut samples carrying a saved reasoning
  tail over A and B; with fewer it runs on what there is and is report only.

**Frozen:** 2026-10-27 20:15 IST, as written at 8c553a6.

**Outcome:** Validity failed on one condition, so every rule is report only: empty or zero-token replies were 8 of 66
records in replicate A and 0 of 66 in B (8 of 132, 6.1%, over both), against at most 5%. All 8 are first-sample
rescues in replicate A that ended on the client timeout with no reply, with the two replicates running at once on
one server. As the rules would read if valid: (1) 18 repair rounds over A and B, fewer than 20, so "too few"; net
right outputs 1 (dbff022c[0], replicate A); (2) verified-clean precision 25 of 33 outputs, 75.8%, the 75-90% band, so
report only, and v8 keeps `verified_first`; (3) 64 cut samples with a saved reasoning tail, against a minimum of 15,
so the replay can be built. Reported, no rule, right of 44 (pass@2): B0 alone 12; `v7_direct` 17; v7-vr2's phase 2
over B0, replicate A 18 and replicate B 19.

**Files:** `experiments/arc2-q27/VR_BANK_RESULTS.md`, `scripts/arc2_vrbank_analyze.py`.

## 3. GS1: Glimmer as a second judge of B0's candidates

**Question:** Does Glimmer 30B's likelihood of each candidate grid, added to B0's own score, put more right grids in
B0's two attempts?

**Compared:** `gs_low` (family `gs1`) against `b0`. On the outputs where kgmon's top-two margin is below the median
of its own split (the unsealed outputs and the sealed 19 separately; no candidate = -inf, one = +inf), B0's pool
candidates ranked by z(kgmon score) + z(Glimmer score), z over the output's candidates, 1:1, ties in B0's order; top
two. Glimmer score = mean log-likelihood per answer token in one view (identity). Fewer than two B0 candidates or a
score missing, and every other output: B0's attempts. 27B-only grids are not ranked. `gs_all` (every output, four
views) was withdrawn by Revision 1 before any score existed.

**Measure:** right outputs of the 145 unsealed outputs against `b0` (40); gained and lost.

**Decision rule:**
- Validity: the CPU twin passed; every candidate of the 101 unsealed tasks scored; each view's inverse restores the
  grid exactly; score ties are at most 5% of candidates.
- Gate v2: net 3 or more and lost 1 or fewer, inside the family (gs_low alone under Revision 1). A rule that passes
  must then agree in sign on the sealed 19 tasks, once.
- Time gate (Revision 1): B0's projected p90 end from the bank's throughput model (E13); the scorer's projected time
  for this scope on the 240-task hidden run must fit within 43,200 s with 1,800 s spare. If it cannot fit, the result
  is report only.

**Frozen:** 2026-10-27 23:31 IST; Revision 1, which governs where it differs, frozen 2026-10-28 00:43 IST at 85a54e3,
before any score existed.

**Outcome:** "GS1: gate v2 not passed (gs_low net -5, lost 6): declined, report only". `gs_low` 35 of 145 (gained 1,
lost 6, net -5, sign test p 0.99219). Validity held: 72 low-margin outputs, 452 candidates, 0 missing, 0 ties. Time
gate held: projected end 41,088.1 s against the 43,200 s limit with 1,800 s spare.

**Files:** `experiments/runs/kaggle-A2-GS1-v1/gs1_analysis.json`, `scripts/arc2_gs1_analyze.py`,
`eval/arc2_bank.py` (`gs_low`), `experiments/arc2-bank/replay_ledger.jsonl`.

## 4. Rescue replay: worked examples in the entry's rescue request

**Question:** Does a block of worked examples raise right answers when it is added to v7-vr2's own rescue request?

**Compared:** two paired arms: `plain`, v7-vr2's rescue request byte for byte (`arc_harness.prompts.rescue_prompt`),
and `block`, the same request with the example block in front of it in the same user message and one line added to
its program instruction ("Make counts, coordinates and tie-breaks exact before writing the program."). The block
holds the five nearest of 674 sandbox-checked training-task programs, each shown as its train pairs, then a `Rule:`
line where one exists, then its program. Inputs: every VR-bank sample that finished cut with a saved reasoning tail.
Decoding as v7-vr2's rescue call; two samples per request per arm, seeds 101 and 102 in both arms, paired by draw.

**Measure:** a unit is (cut sample, test index, draw k); its answer in an arm is what that one rescued sample would
put first under the entry's ranking, and it is right when it equals the truth. A gain: block right and plain not; a
loss: plain right and block not.

**Decision rule:**
- Validity (otherwise no decision, report only): at least 15 cut samples with a saved tail over A and B; median
  completion tokens at least 5 in each arm; empty or zero-token replies at most 5% of requests per arm; no sealed task
  read.
- Block helps: gains minus losses at least 3, and the one-sided sign test on the discordant units gives p at most
  0.10, and block's verified-clean wrong answers are at most plain's plus 1. Then the replay is the evidence for adding
  the block to hybrid v8's rescue request.
- Block does not help: gains minus losses at most 0. v8's rescue request stays v7-vr2's.
- Otherwise: report only; v8's rescue request stays v7-vr2's.

**Frozen:** 2026-10-27 23:51 IST, as written at cbcbf70.

**Outcome:** Valid; block does not help. 64 cut samples with a saved tail; median completion tokens 2,199.5 (plain)
and 1,565 (block); no empty or zero-token replies; no sealed task. On 196 paired units: gains 4, losses 7, net -3;
one-sided sign test p 0.89; block's verified-clean wrong answers +4 over plain's. Right: plain 31, block 28 of 196.
Net at most 0 means block does not help: v8's rescue request stays v7-vr2's.

**Files:** `experiments/arc2-q27/RESCUE_REPLAY_RESULTS.md`, `scripts/arc2_rescue_replay_analyze.py`.

## 5. DC1: an open decision model as the judge of B0's candidates

**Question:** Does an open 2B decision model, used zero-shot to choose among B0's candidates, put more right grids in
B0's two attempts?

**Compared:** family `dc1` against `b0`, B0's pool ranked by the decision model's mean choice probability over two
option orders (a seeded shuffle and its reverse); the 27B's grids are not ranked.
- `dc_fit`: on outputs whose request fits its 4,096-token window in both orders, the two highest-scoring candidates;
  ties in B0's order; B0's attempts elsewhere.
- `dc_long`: the same ranking on every output with two or more B0 candidates, the 16,384 window where the request
  exceeds 4,096.
- `dc_keep1`: B0's attempt 1 kept; attempt 2 the decision model's top candidate, or its second when the top is B0's
  attempt 1, on outputs whose request fits 4,096 tokens in both orders; B0's attempts elsewhere.
- `dc_keep1_long`: `dc_keep1` on every output with two or more B0 candidates and probabilities (16,384 window where
  needed; over 16,384 B0's attempts stand).

**Measure:** right outputs of the 145 unsealed outputs against `b0` (40); gained and lost.

**Decision rule:**
- Validity: the CPU twin passed; the pools match the bank (unsealed: 101 tasks, 145 outputs, 828 B0 candidates;
  sealed: 19 tasks, 27 outputs, 152 B0 candidates); every checkpoint and base file matches its sha256; every unsealed
  output with 2 or more candidates outside the over_16384 tier is answered in both orders, the unsealed file is
  complete, and the only refusals are 4,096-window refusals that were answered at 16,384; each answer's probabilities
  sum to 1 within 0.01.
- Gate: adoption is under v2: net 3 or more and lost 1 or fewer, inside family dc1 (four rules). A rule that passes
  must then agree in sign on the sealed 19 tasks, once. Passing still adopts nothing. Entering an entry needs its own
  time gate, pre-registered then.

**Frozen:** 2026-10-28 23:25 IST (17:55 UTC), as written at b80d5ce.

**Outcome:** Validity PASS: 112 of 112 eligible unsealed outputs answered in both orders (50 fit_4096, 62
long_16384, 0 over_16384). Gate v2: no rule passes. DROP.

| Track | Rule | Right of 145 | Gained | Lost | Net | Gate v2 |
|---|---|---:|---:|---:|---:|---|
| ARC-2 | b0 (base) | 40 | | | | |
| ARC-2 | dc_fit | 34 | 0 | 6 | -6 | no |
| ARC-2 | dc_long | 25 | 2 | 17 | -15 | no |
| ARC-2 | dc_keep1 | 36 | 0 | 4 | -4 | no |
| ARC-2 | dc_keep1_long | 35 | 2 | 7 | -5 | no |

Recorded, no decision: on the 42 outputs with the right grid in B0's pool, the top pick is right for B0 on 25 and for
the decision model on 15.

**Files:** `experiments/runs/kaggle-A2-DC1-v1/dc1_analysis.json`, `scripts/arc2_dc1_analyze.py`,
`eval/arc2_bank.py` (`DC1_RULES`), `experiments/arc2-bank/replay_ledger.jsonl`.

## 6. C2: the sample-count ceiling

**Question:** Does more 27B sampling, which a faster second phase (v9) would buy, raise the ceiling at all?

**Compared:** `oracle` on the C2 pool (B0's pool and all 27B samples: four on the band, two beyond; logged as
`oracle_c2`) against `oracle` on the bank pool (B0's pool and the 27B's samples 0 and 1 on the band; 55 of 145).
Recorded: `v7_direct` and `q27_first` on the C2 pool (`v7_direct_c2`, `q27_first_c2`). The new samples: samples 2 and
3 (seeds 2007 and 3007) on the 30 band tasks and samples 0 and 1 (seeds 7 and 1007) on the 71 unsealed tasks beyond
the band, with the bank's prompts, caps and sampling.

**Measure:** the oracle (any correct grid in the pool) on the 145 unsealed outputs, with Kaggle's solutions copy as
truth.

**Decision rule:**
- Gate: oracle at C2 >= 60 of 145 (at least +5 outputs). Below 60 the answer is "no".
- Oracle at C2 >= 60: v9's faster second phase buys samples where line 2 says they pay (per task on the band, or a
  wider band); the pick-rate line says how much the current rule would keep, and a selection change goes through the
  replay gate as its own rule.
- Oracle at C2 < 60: more 27B sampling does not raise the ceiling enough; v9's time goes to the band's width only if
  line 2(b) alone is at or above +4, else to finishing samples and to B0.
- Line 2, recorded with the result, not a gate: (a) the band, four samples against two on the 44 band outputs, below
  +2 says more samples per task is not where v9's time goes; (b) beyond the band, at or above +4 says a wider band is.
- Validity floor: every item, else the affected tasks are "report only" and no threshold decision is read.

**Frozen:** 2026-10-29 00:12 IST (18:42 UTC Oct 28), as written at 24ef327.

**Outcome:** Gate PASS: oracle 61 on the C2 pool against 55 on the bank pool (+6), read on 139 outputs of 98 tasks;
the 3 tasks whose samples wrote grids over 30 × 30 are report only, and none of their outputs is right in any pool
(55 and 61 on all 145). Line 2: the band 24 to 28 (+4) of 44 outputs; beyond the band 31 to 33 (+2) of 95 outputs;
neither guidance line holds. Recorded: `v7_direct` 45 on the bank pool and 43 on the C2 pool (pick rate 10 of 13,
0.7692, against 14 of 24, 0.5833); `q27_first` 45 and 43. Task bootstrap of the oracle difference: 6, 95% interval
1 to 12.

**Files:** `experiments/arc2-bank/c2_result.json`, `experiments/arc2-bank/c2_check_c2a.json`,
`experiments/arc2-bank/c2_check_c2b.json`, `scripts/arc2_c2_read.py`, `scripts/arc2_c2_check.py`,
`experiments/arc2-bank/replay_ledger.jsonl`.

## 7. B0R: B0's own ranking signals

**Question:** Can a ranking of B0's own candidates, built only from signals already in the pools, put more right
grids in B0's two attempts than kgmon does, alone and inside v7's merge?

**Compared:** three rules that keep kgmon's first as attempt 1 and choose attempt 2 as the best other candidate,
ties by kgmon rank: `b0r_zbeam` (z(n_beams) + z(-score_aug_mean) + z(-best beam_score), z over the output's
candidates), `b0r_amax` (n_beams - max(score_aug)), `b0r_borda` (-(rank_kgmon + rank_probmul3)); their merged forms
`b0r_zbeam_v7`, `b0r_amax_v7`, `b0r_borda_v7` (v7's merge applied to the rule's two attempts). Against `b0` and
`v7_direct`.

**Measure:** on 6 development pools (40 dev tasks, 390 outputs), each rule's top two against kgmon's: right, gained,
lost. On the bank, right outputs of 145; held out, the 80 outputs of the 61 tasks outside the dev set.

**Decision rule:**
- Choice on the dev pools: a rule qualifies when its pooled net is at least +3 and its net is at least 0 on at least
  5 of the 6 pools. The chosen rule is the qualifier with the highest pooled net, then fewer lost, then the table's
  order. No qualifier: the study ends with "kgmon stays", and the bank is not read.
- Bank reading of the chosen rule: G1, the rule against `b0` (40), gate v2: net >= +3 and lost <= 1. G2,
  `<rule>_v7` against `v7_direct` (45), gate v2. G3, on the 61 non-dev tasks (80 outputs), net >= 0 in G1's and in
  G2's comparison.
- G2 and G3 pass: entry candidate (the ranking joins Stage 1's replay grid, and an entry build can go to the
  leaderboard under its own rule). G1 and G3 pass, G2 fails: the ranking joins Stage 1's replay grid only; no entry on
  its own. The chosen rule fails, or no qualifier on dev: kgmon stays; Stage 1's grid carries kgmon and probmul3. A
  pass on the 145 with G3 failed is reported as "gain on the dev tasks only" and decides nothing.
- A rule proposed for an entry gets the sealed sign check once, at the final pick.

**Frozen:** 2026-10-29 15:41 IST, revision 1 as written at 166b24a.

**Outcome:** Dev choice: `b0r_zbeam` 90 of 390 against kgmon's 86 (+9 -5, net +4, 5 of 6 pools >= 0), chosen;
`b0r_amax` 89 (+3 -0, net +3, 6 of 6 pools), qualified, not chosen; `b0r_borda` 87 (+1 -0, net +1), not qualified.
Bank: G1 41 against 40 (+2 -1, net +1), gate v2 no; G2 47 against 45 (+2 -0, net +2), gate v2 no; G3 PASS (net 0
and +1 on the 80 held-out outputs). "kgmon stays; Stage 1's grid carries kgmon and probmul3". The sealed check of
`b0r_zbeam`, read once as a report: net -1 on the sealed 19 (0 gained, 1 lost) and net 0 on all 172 outputs (2
gained, 2 lost); it was the bank's first sealed read, so no untouched sealed check remains for any rule.

**Files:** `experiments/arc2-bank/b0r_dev.json`, `experiments/arc2-bank/b0r_result.json`,
`experiments/arc2-bank/b0r_zbeam_sealed_check.json`, `scripts/arc2_b0r_dev.py`, `scripts/arc2_b0r_read.py`,
`eval/arc2_bank.py` (`B0R_RULES`), `experiments/arc2-bank/replay_ledger.jsonl`.

## 8. S3: colour 0 fixed in every colour permutation

**Question:** Do colour permutations that keep colour 0 (the usual background) fixed raise B0's solved outputs,
against the bank's B0 on the same tasks?

**Compared:** a B0 run with one change, every colour permutation drawn as `[0] + (1 + permutation of 0..8)` in
training, search and rescoring, against the bank's B0 (`b0` on each pool); line 2, `v7_direct` replayed with S3's
pool and the bank's 27B samples against `v7_direct` on the bank.

**Measure:** pass@2 of each arm's own two attempts on the N outputs of the tasks S3 ran; exact one-sided McNemar on
the per-output discordance (`sign_test_p`).

**Decision rule:**
- Line 1, B0 alone, gain (gate): net >= ceil(5 N / 145) and p <= 0.05. No gain: net <= floor(N / 145), i.e.
  net <= 0 below 145 outputs. Otherwise: report.
- Line 2, entry line (gate): gate v2, net >= +3 and lost <= 1; unscaled, the tasks S3 did not run counting as
  unchanged.
- Gain and entry gate pass: entry candidate, a v7-vr2 + S3 build under its own rule, decided separately. Gain, entry
  gate fails: S3 is a B0 training lever for Stage 1; no entry on its own. No gain: S3 is not a lever on this evidence;
  recorded. Otherwise: reported; no decision.

**Frozen:** 2026-10-29 16:48 IST, as drafted at 922727f.

**Outcome:** All 101 tasks run (145 outputs), none report only. Line 1: 39 against 40 (+5 -6, net -1, p 0.72559;
net needed 5, no gain at or below 1): "no gain". Line 2: 44 against 45 (+5 -6, net -1), gate v2 no. "S3 is not a
lever on this evidence; recorded". Recorded: B0's pool oracle 44 against 46; time ratio 1.014 (sum), neutral.

**Files:** `experiments/arc2-bank/s3_result.json`, `scripts/arc2_s3_read.py`.

## 9. S1: 128 training sequences

**Question:** Do 128 augmented training sequences per task (the public notebook's setting) instead of the entry's 64
raise B0's solved outputs, against the bank's B0 on the same tasks, and does a gain survive the 27B time it costs?

**Compared:** a B0 run with one change, `augment(n=16, shfl_keys=True, seed=1)` (8 geometric views x 16 colour
permutations = 128 sequences), against the bank's B0 (`b0`); line 2, the time-cut merged projection of `v7_direct`
on S1's pool against `v7_direct` on the bank.

**Measure:** line 1 as S3's. The time-cut merged projection (TCP): B0's phase ends at 9 h 18 m (33,480 s) and the
27B's window is C = 7,320 s; a slower variant moves B0's phase end later by Δ = max(0, W_v - W_ref), from the bank's
throughput model (E13) on the hidden mix; the 27B keeps its samples, in the entry's order, up to k = max(0, 1 - Δ / C)
of their summed time; `v7_direct` is replayed on the variant's pool with the kept samples.

**Decision rule:**
- Line 1, B0 alone (gate). Gain: net >= ceil(5 N / 145) and p <= 0.05; with all 101 tasks run, net >= +5. No gain:
  net <= floor(N / 145), i.e. net <= +1 with all run. Otherwise: report.
- Line 2, entry line (gate): the time-cut merged projection against `v7_direct` on the bank (45 with all run) passes
  gate v2: net >= +3 and lost <= 1.
- Gain and pass: entry candidate, decode-many's entry template with 128 sequences and B0's phase end moved by Δ, and a ~12 h
  commit run, under its own rule. Gain and fail: S1 is a training lever that does not pay for its time in this entry.
  No gain: S1 is not a lever on this evidence; recorded. Otherwise: reported; no decision.

**Frozen:** 2026-10-29 22:36 IST (17:06 UTC Oct 29), with its TCP section shared with decode-many, at 2de2fc9.

**Outcome:** All 101 tasks run (145 outputs). Line 1: 36 against 40 (+4 -8, net -4, p 0.927): "no gain". Line 2:
37 against 45 (+5 -13, net -8), gate v2 no; Δ 5,629.9 s, k 0.2309, 11 of 55 27B samples kept. "S1 is not a lever on
this evidence; recorded". Recorded: time ratio 1.172 (sum) and 1.214 (median); B0's pool oracle 44 against 46.

**Files:** `experiments/arc2-bank/s1_result.json`, `scripts/arc2_s1_read.py`, `scripts/arc2_bank_profile.py`.

## 10. DM: decode-many, one superset search replayed on CPU

**Question:** From one training of each task's LoRA, which decoding setting (views, search floor, rescoring count,
ranking) gets the most right outputs, and does any setting beat the bank's setting by enough to pay for its time in
the entry?

**Compared:** 24 settings, views {16, 32} x floor {0.2, 0.1} x rescoring {8, 16, 32} x ranking {kgmon, probmul3}
(`v16_f0.2_r8_kgmon` and so on), each replayed under the bank's caps from one superset search (32 views, floor 0.1,
32 rescoring augmentations) on the same adapters; s0 = (16, 0.2, 8, kgmon), the bank's setting, is the reference.

**Measure:** each setting's TCP merged line (S1's) against s0's, with W_ref the E13 wall on s0's emulated seconds;
half A holds the tasks whose sha256 of "dm-half:" + task id has an even first byte, half B the rest.

**Decision rule:**
- Choose on half A: among the 23 settings other than s0, the highest net on half A's outputs in the TCP merged line
  against s0's; ties go to fewer lost, then the smaller Δ, then grid order. If no setting reaches net >= +1 on A,
  there is no winner and the bank's decoding stays.
- Confirm on half B: the winner's TCP merged line on half B: net >= +1 and lost <= 1.
- Qualify: confirmed, and the winner's TCP merged line on all 145 outputs passes gate v2 against s0's: net >= +3 and
  lost <= 1.
- A qualifying winner: an entry draft (v7-vr2 with the setting, B0's phase end moved by its Δ) and a commit run, under
  its own rule. Confirmed, not qualified: recorded; no entry. No winner, or not confirmed: the bank's decoding stays.
- Time emulation: s0's emulated seconds sum within 0.90 to 1.10 of the bank's on the tasks run, and s0 "coverage
  short" on at most 2 of them; else every TCP line is report only, and nothing qualifies.

**Frozen:** 2026-10-29 22:36 IST (17:06 UTC Oct 29), before the build and before the proof, at 2de2fc9.

**Outcome:** Valid: s0's time ratio 1.0026, no task coverage short. "no winner on half A: the bank's decoding stays";
the best setting on half A (16 views, floor 0.2, 16 rescorings, kgmon) reached net 0. Recorded: s0 against the bank's
B0, 42 against 40 (+3 -1); merged, 48 against 45 (+3 -0). Pool oracle 48 at s0 and 52 at 32 views, floor 0.1 and 8
rescorings, where B0 alone gave 39. The proof run before the main run checked the path only and fed no decision.

**Files:** `experiments/arc2-bank/dm_result.json`, `experiments/arc2-bank/dm_proof_result.json`,
`experiments/arc2-bank/dm/main/dm_replay.json`, `experiments/arc2-bank/dm/proof/dm_replay.json`,
`scripts/arc2_dm_replay.py`, `scripts/arc2_dm_read.py`.

## 11. TS: temperature sampling as a second B0 search

**Question:** On the same adapters, does a temperature-sampling search after the bank's search add right outputs, as
ranked by kgmon, and does any gain survive the time it costs in the entry?

**Compared:** G, a fresh run of the bank's search (the reference); `G+T1`, G's beams plus one round of samples at
temperature 1.0 per view (16 per output); `G+T2`, two rounds (32 per output); T1 and T2 alone recorded.

**Measure:** the TCP merged line against G's on halves A and B (as decode-many's), with E13's per-task clip at
2,400 s for G+T; the sampling line, G+T2 against G, kgmon both, B0 alone, exact McNemar.

**Decision rule:**
- Choose on half A: between G+T1 and G+T2, the higher net on half A's outputs in the TCP merged line against G's;
  ties go to fewer lost, then the smaller Δ, then T1. If neither reaches net >= +1 on A, there is no winner.
- Confirm on half B: the winner's TCP merged line on half B: net >= +1 and lost <= 1.
- Qualify: confirmed, and the winner's TCP merged line on all outputs passes gate v2 against G's: net >= +3 and
  lost <= 1.
- Sampling line: gain: net >= ceil(5 N / 145) and p <= 0.05 (+5 with all run); no gain: net <= floor(N / 145) (+1
  with all run); otherwise: report.
- A qualifying winner: an entry draft (v7-vr2 with T after each task's search, B0's phase end moved by its Δ) and a
  commit run, under its own rule. Confirmed, not qualified: recorded; no entry. No winner, or not confirmed: the
  bank's decoding stays. Sampling line gain: temperature sampling adds right outputs to B0's DFS at about its own
  search time; no gain: temperature sampling is not a lever for B0 on this evidence; with decode-many, this closes the
  sampling question; otherwise: reported.
- Whole run: G's seconds sum within 0.90 to 1.10 of the bank's, else every TCP line is report only and nothing
  qualifies; T failed on at most 2 of the tasks run, else every T line is report only.

**Frozen:** 2026-10-30 09:33 IST (04:03 UTC Oct 30), at 7734025.

**Outcome:** Valid: G's time ratio 1.0073, T failed on 0 tasks; 101 tasks, 142 outputs. "no winner on half A: the
bank's decoding stays": G+T1 and G+T2 each net -3 (5 lost) on half A. Sampling line: G+T2 39 against G's 39 (+1 -1,
net 0, p 0.75): no gain. Recorded: pool oracle G 45, G+T1 47, G+T2 51; time ratio to G 1.5046 (G+T1) and 1.9865
(G+T2); G against the bank's B0, 39 against 39 (+2 -2).

**Files:** `experiments/arc2-bank/ts_result.json`, `scripts/arc2_ts_read.py`.

## 12. AMAX: the amax picker as a learning entry

**Question:** Learning only: does re-picking B0's attempt 2 by amax put at least as many right grids in B0's pair as
kgmon on the bank, alone and inside v7's merge?

**Compared:** `b0r_amax` against `b0` (G1) and `b0r_amax_v7` against `v7_direct` (G2), as frozen in B0R (kgmon's
first stays attempt 1; attempt 2 is the best other candidate by n_beams - max(score_aug), higher first, ties by kgmon
rank).

**Measure:** right outputs of the 145 unsealed outputs; the outputs with 3+ candidates where amax's attempt 2
differs from kgmon's.

**Decision rule:**
- Gate: net >= 0 in G1 and in G2. Pass: build the entry and one commit run. Fail: stop; no build; the result is
  reported.
- No-change stop (added before any number): if amax's attempt 2 equals kgmon's on every bank output with 3+
  candidates (0 differ), stop before the build; the result is reported, even though the gate reads net 0 as a pass.
  With 1 or more differing, the gate stands as written.
- Gate v2, G3 (the 80 outputs of the 61 tasks outside dev) and the bootstraps are recorded beside, no decision. No
  sealed read: the seal is spent.
- The leaderboard score, if submitted, is recorded beside zbeam's. Neither decides a ranking rule.

**Frozen:** 2026-11-01 23:17 IST (17:47 UTC Nov 1), with its one addition (the no-change stop) before any amax bank
number, at 87163c3c.

**Outcome:** G1 40 against 40 (+0 -0) and G2 45 against 45 (+0 -0): the gate passes. Attempt 2 differs from kgmon's
on 29 of the 102 outputs with 3+ candidates, so the no-change stop does not apply. Beside: G3 25 against 25 and 27
against 27. "build: the amax entry and one commit run; the submission is a separate decision".

**Files:** `experiments/arc2-bank/amax_result.json`, `scripts/arc2_amax_read.py`,
`experiments/arc2-bank/replay_ledger.jsonl`.

## Predictions committed before scoring

Each prediction was committed before its entry was submitted. The base for all three is v7-vr2's two readings, 32.64
and 32.22 (mean 32.43). Public scores are in `submissions/ledger.json`.

| Track | Entry | Predicted band or value | Frozen | Public score |
|---|---|---|---|---|
| ARC-2 | S1: v7-vr2 with 128 training sequences, B0's phase end moved by 5,630 s | About 26.9 if the offline tools are right (bank B0 -4, time-cut projection -8 merged); about 31.7 if the time model is too harsh; above 32.4 if more training helps on hidden tasks. SD of one reading about 4.0. Bands: at or below 29.3, consistent with the offline tools; between 29.3 and 34.5, the time model looks too harsh or the bank's B0 loss does not carry to hidden tasks; at or above 34.5, above v7-vr2's best reading by about one reading's SD | 2026-10-31 14:35 IST | 31.81 |
| ARC-2 | zbeam: v7-vr2 with `b0r_zbeam` choosing B0's attempt 2 | About 33.8 (32.43 + 1.36); SD of one reading about 3. About 28 to 40 is consistent with the bank; at or above 36.4, a first hint that the picker carries to the hidden tasks; at or below 29.3, a second hint against the bank's B0 lines | 2026-10-31 | 32.64 |
| ARC-2 | amax: v7-vr2 with `b0r_amax` choosing B0's attempt 2 | About 32.4 if the bank is right; about 33.2 if the dev gain carries; SD of one reading about 2.8. Bands: 29.6 to 35.2, consistent with no change; above 35.2, a hint that the picker carries to hidden tasks; below 29.6, more likely run noise than the picker | 2026-11-01 23:56 IST | 29.31 (scored after the competition closed) |

## The v7-vr repeat reading

v7-vr was submitted twice with nothing changed (the same notebook version, cells and inputs) and scored 34.31, then
29.31: 5.00 points, 6 tasks of 120. Replaying both builds' ranking code on the VR-bank's two replicates (44 outputs
of the 30 shortest public evaluation tasks; `experiments/arc2-bank/v7vr_repeat_replay.py`) gave 17 and 19 for v7-vr,
18 and 19 for v7-vr2, and 12 for B0; under v7-vr's rule 8 outputs were right in only one replicate, 5 tasks flipped.
With the 27B's draw included, a difference of two hybrid readings has an SD of about 3.2 points, so 5.00 is about
1.6 SD (about 1 in 9) and 34.31 was the outlier; against v7's 30.97, the means put v7-vr at +0.84 and v7-vr2 at
+1.46, not the +3.34 first read from 34.31 alone.
