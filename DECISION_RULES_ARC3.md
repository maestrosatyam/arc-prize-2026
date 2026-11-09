# ARC-AGI-3 decision rules

This file records the decision rule of every ARC-AGI-3 run behind the paper The Idea Gap: the first continuation run (R1) was exploratory, and every run after it had its arms and decision rule written down before it ran. A reading is stated as met, not met or report, in the words of its rule, and no threshold was changed after a result. Outcomes cite the result files in this repository; dates are IST (UTC+5:30); a continuation starts from the restore point of one of the 35 stuck states, and "mechanics-only" (`partial` in the files) means the 22 of those 35 levels where the agent had the game's mechanics right.

## R1: probe-first and timeout repair (pilot)

- **Question**: On the 35 failed levels, from the same point in each, does a probe-first prompt or a tool-timeout repair let the unchanged model clear the level more often than the unchanged harness at an identical budget?
- **Arms**: A0 unchanged; A1 probe-first (a notice asking for two hypotheses and a probe of at most 6 actions before long plans; batches capped at 6 actions until a read-back); A2 timeout repair (tool timeout 120 s instead of 30 s; queued actions cancelled on timeout; functions kept).
- **Budget**: 80 actions or 15 minutes per continuation, whichever comes first; stop at the clearing action.
- **Decision rule**: none. The design fixed an analysis plan (each arm paired with A0 by case: discordant counts, exact two-sided McNemar p) and no threshold; the paper treats this run as exploratory.
- **Frozen**: design 2026-10-19, at 9bdc7fa8.
- **Outcome**: A0 9/35, A1 5/35 (discordant 4/8, McNemar exact p 0.388), A2 6/35 (discordant 3/6, p 0.508); 16/35 cases cleared by at least one arm (16.53 expected from independent draws); "arms do not matter" permutation p 0.5207.
- **Files**: `experiments/runs/kaggle-R1-pilot-v4/r1_summary.txt`, `experiments/runs/kaggle-R1-pilot-v4/r1_results.jsonl`, `experiments/arc3-recovery/r1_v4_analysis.txt`.

## R1b: second unchanged sample and fresh start

- **Question**: Is the outcome from a restore point a draw that the arms do not move, or is the restored conversation itself the handicap?
- **Arms**: A0b unchanged, second sample; A3 fresh start (no conversation, no retained functions; one line stating the level and actions so far).
- **Budget**: 80 actions or 15 minutes per continuation.
- **Decision rule**: no numeric threshold; the reading written before the run: "A0b vs A0 is the discordance between two draws of the same arm from the same state: the reproducibility floor." "If A1, A2 and A3 vs A0 are no more discordant than A0b vs A0, and the permutation test over the five arms does not reject, the arms are indistinguishable from re-running the control." "A3 above both A0 samples by more than the A0-A0b difference, with few reverse discordances, would say the restored conversation is a handicap."
- **Frozen**: 2026-10-19, at c24b851f.
- **Outcome**: A0b 11/35; A0b vs A0 discordant 4/2 (6 of 35), McNemar exact p 0.688. A1 vs A0 discordant 4/8 (12 of 35), A2 vs A0 3/6 (9 of 35). A3 1/35, vs A0 discordant 0/8, p 0.008. Permutation over the five arms p 0.0074. The paper reads A3 as "below (p 0.008); context carries progress".
- **Files**: `experiments/runs/kaggle-R1b-v1/r1_summary.txt`, `experiments/runs/kaggle-R1b-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1_r1b_analysis.txt`.

## R1c: trimmed conversation and no retained functions

- **Question**: A3 dropped the conversation and the retained functions at once; which of the two carries the loss, and does a context trim of the size the harness itself makes cost levels?
- **Arms**: A4 trim (conversation cut to its most recent 45% of messages, as one of P0's own trims does); A5 no functions (full conversation, no retained functions).
- **Budget**: 80 actions or 15 minutes per continuation.
- **Decision rule**: predictions written before the run:
  - "A4 ~ A0 (A4 discordance with each baseline sample no larger than the A0-A0b floor, 6/35, and A4's clear count between "cleared by both samples" and "cleared by either"): one P0-sized trim at the restore point is harmless; the dropped turns carry nothing the model uses on this level."
  - "A4 << A0 (below "cleared by both", reverse discordance against both samples): the trim costs levels; a trim-preserving change (a summary of the dropped turns, or keeping key observations) is the next candidate."
  - "A5 tells which half of A3's loss is which. A5 ~ A0: the retained functions do not matter and the conversation carries A3's loss. A5 near A3 (about 1/35): the functions carry it. In between: both."
- **Frozen**: 2026-10-20, at 9afde836.
- **Outcome**: A4 7/35 (vs A0 discordant 2/4, p 0.688; vs A0b 1/5, p 0.219; cleared by both samples 7, by either 13). A5 6/35 (vs A0 2/5, p 0.453; vs A0b 1/6, p 0.125). No verdict line is recorded; the paper reads both arms as "inside" the unchanged range.
- **Files**: `experiments/runs/kaggle-R1c-v1/r1_summary.txt`, `experiments/runs/kaggle-R1c-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1_r1b_r1c_analysis.txt`.

## R1d: unchanged harness, doubled budget

- **Question**: "Are the remaining failures time-limited (more time on the same state clears more levels) or competence-limited (it does not)?"
- **Arms**: A0L unchanged.
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: "gain = A0L clears at 30 min - A0L clears at 15 min."
  - "gain >= 5: failed levels are time-limited; the next work is time allocation in the 9-hour run (P0 failures mostly ended in scheduler pauses that were never resumed)."
  - "gain <= 2: failed levels are competence-limited; harness work stops."
  - "gain 3 or 4: inconclusive, reported as such."
  - Internal control: A0L's own 15-minute clears (within 80 actions and 900 s) "should sit within the reproducibility floor (A0 vs A0b: 9 vs 11, 6/35 discordant)". "If A0L at 15 min falls outside the floor against both baseline samples, the clock difference changed play, and the 30-minute count is read against A0L's own 15-minute count only."
- **Frozen**: 2026-10-20, at c5ff7ce7.
- **Outcome**: "R1d decision (gain >= 5 time-limited, <= 2 competence-limited): time-limited: next work is time allocation in the 9-hour run"; cleared within the R1 budget 5, within its own 14, gain 9. The internal control failed: A0L at 15 min 5 against A0's 9 (discordant 2/6, p 0.289) and A0b's 11 (0/6, p 0.031). The paper therefore reads time only against independent samples at each budget.
- **Files**: `experiments/runs/kaggle-R1d-v1/r1_summary.txt`, `experiments/runs/kaggle-R1d-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1_all_analysis.txt`.

## R1e: repertoire sweep

- **Question**: Does a prompt that lists the game's observed effects per action and asks for untested uses of them raise clears on the mechanics-only levels?
- **Arms**: A6 repertoire sweep (every 25 actions without progress, the observed effects per action are listed and the agent asked to try known mechanics in new roles); A0c unchanged, third sample.
- **Budget**: 80 actions or 15 minutes per continuation.
- **Decision rule**: "Decision (fixed before the run): build a full 25-game P0+A6 notebook for a leaderboard reading if
  - A6 clears >= the best baseline sample + 4 on the 35 cases, or
  - A6 clears >= the best baseline sample + 3 on the 22 `partial` cases and A6 on the 11 `yes` cases is not below the median of the three samples there;

  otherwise stop."
- **Frozen**: 2026-10-21, at f0cf3c64.
- **Outcome**: "R1e decision: no promotion: A6 does not beat the baseline samples by the margin (all cases: A6 8 vs best sample 11 (needs +4): not met; partial cases: A6 4 vs best sample 3 (needs +3): not met; yes cases: A6 3 vs median sample 6: loss)". A0c 6/35.
- **Files**: `experiments/runs/kaggle-R1e-v1/r1_summary.txt`, `experiments/runs/kaggle-R1e-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1_all_analysis_with_r1e.txt`.

## R1f: told the idea at the standard budget

- **Question**: Told the level's key idea at the restore point, does the agent clear the mechanics-only levels at the standard budget?
- **Arms**: A7 told the idea (the level's key idea in a message at the restore); A7w its win condition only; A7c placebo (a length-matched block of true but irrelevant rules); A0d unchanged, fourth sample.
- **Budget**: 80 actions or 15 minutes per continuation.
- **Decision rule**: A7 needs 8 of the 22 mechanics-only cases; short of that, the reading "revises the account towards representation or goal inference", split by A7w: "towards representation (A7w not above A7)".
- **Frozen**: before the run of 2026-10-21.
- **Outcome**: "R1f decision: A7 cleared 6 of 22 partial cases (needs 8 of 22): revises the account towards representation or goal inference; A7w 2 on the partial cases: towards representation (A7w not above A7)". Overall A7 12/35, A7w 7/35, A7c 10/35, A0d 5/35; content effect A7 against A7c 12 vs 10, discordant 7/5, p 0.774.
- **Files**: `experiments/runs/kaggle-R1f-v1/r1_summary.txt`, `experiments/runs/kaggle-R1f-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1g_v2_analysis.txt`.

## R1g: told the idea at the doubled budget, run twice

- **Question**: Is R1f's ceiling budget-bound: told the idea at the doubled budget, does the agent clear the mechanics-only levels?
- **Arms**: A7L told the idea; A0Lb unchanged; both run twice (the second run is the repeat).
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: A7L on the 22 mechanics-only cases: "budget-bound at >= 11 of 22, partly at 7-10, not at <= 6". Repeat: the told arm's repeat clears 10 or more of 22.
- **Frozen**: before the run of 2026-10-21; the repeat's line before the repeat of 2026-10-22.
- **Outcome**: run 1 A7L 20/35 (12 of 22 mechanics-only), A0Lb 14/35; run 2 A7L 23/35 (12 of 22), A0Lb 18/35. Recorded: "R1g decision: A7L cleared 12 of 22 partial cases (budget-bound at >= 11 of 22, partly at 7-10, not at <= 6): the R1f ceiling was budget-bound; the account returns to generation plus budget". Repeat line met (12 and 12).
- **Files**: `experiments/runs/kaggle-R1g-v1/r1_summary.txt`, `experiments/runs/kaggle-R1g-v1/r1_results.jsonl`, `experiments/runs/kaggle-R1g-v2/r1_summary.txt`, `experiments/runs/kaggle-R1g-v2/r1_results.jsonl`, `experiments/arc3-recovery/r1g_v2_analysis.txt`.

## R1h: the idea with its objects located, on the residual seven

- **Question**: On the seven states that no unchanged or idea arm cleared at the standard or doubled budget, does a paragraph locating on the board each object the idea names convert them?
- **Arms**: A8L the idea plus a paragraph locating its objects; A7Lr the idea repeated.
- **Budget**: 160 actions or 30 minutes per continuation; 7 cases.
- **Decision rule**: "grounding at A8L >= A7Lr + 3, planning or grain at |A8L - A7Lr| <= 1".
- **Frozen**: before the run of 2026-10-22.
- **Outcome**: "R1h decision: A8L cleared 1 and A7Lr 0 of 7 cases (grounding at A8L >= A7Lr + 3, planning or grain at |A8L - A7Lr| <= 1): the remaining failures are planning under budget or the grain of the insight text".
- **Files**: `experiments/runs/kaggle-R1h-v1/r1_summary.txt`, `experiments/runs/kaggle-R1h-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1n_analysis.txt`.

## R1i: low reasoning effort

- **Question**: Does reasoning effort low, instead of the template's default (the highest), change clears?
- **Arms**: A8l low effort; A0e unchanged, fifth sample.
- **Budget**: 80 actions or 15 minutes per continuation.
- **Decision rule**: as recorded with the outcome, three lines: all cases, A8l "needs >= 15" of 35; partial cases, A8l "needs >= 7" of 22; `yes` cases, A8l against the median sample. The effort lever closes when "no effort arm meets the rule".
- **Frozen**: before the run of 2026-10-22.
- **Outcome**: "A8l not promoted (all cases: A8l 7 of 35 (needs >= 15): not met; partial cases: A8l 2 of 22 (needs >= 7): not met; yes cases: A8l 5 of 11 vs median sample 4: no loss)"; "R1i decision: effort closed as a lever: no effort arm meets the rule". A0e 7/35.
- **Files**: `experiments/runs/kaggle-R1i-v1/r1_summary.txt`, `experiments/runs/kaggle-R1i-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1i_analysis.txt`.

## R1j: trick library

- **Question**: Does a library of 57 game-independent patterns from the other games' rules, ranked by observed effects, raise clears?
- **Arms**: A9 trick library; A0Lc unchanged (doubled-budget sample).
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: "promote if A9 >= 22 of 35 (best unchanged 30-minute sample 18 + 4) or >= 12 of 22 partial."
- **Frozen**: before the run of 2026-10-22.
- **Outcome**: "A9: 16 of 35, 8 partial; not run 0. Verdict: report and stop." A0Lc 17/35, 8 of 22.
- **Files**: `experiments/runs/kaggle-R1j-v1/r1_summary.txt`, `experiments/runs/kaggle-R1j-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1j_analysis.txt`.

## R1k: four times the budget on the residual seven

- **Question**: On the residual seven, does four times the budget clear them, with or without the idea?
- **Arms**: A0X unchanged; A7X told the idea.
- **Budget**: 320 actions or 60 minutes per continuation; 7 cases.
- **Decision rule**: "budget at A7X >= 3; not budget at A7X <= 1 and A0X <= 1".
- **Frozen**: before the run of 2026-10-22.
- **Outcome**: "R1k decision: A7X cleared 1 and A0X 3 of 7 cases (budget at A7X >= 3; not budget at A7X <= 1 and A0X <= 1): between the bands, no reading; A0X cleared as many as A7X (3 vs 1): budget alone suffices on those cases".
- **Files**: `experiments/runs/kaggle-R1k-v1/r1_summary.txt`, `experiments/runs/kaggle-R1k-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1n_analysis.txt`.

## R1l: the 27B of the same family as the agent

- **Question**: Does the 27B model of the same family as the agent's model, used as the agent, clear more of the stuck states?
- **Arms**: A10l low effort; A10x full effort; A10m medium effort (on a server with a bf16 KV cache and prefix caching off).
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**:
  - A10l, A10x: "promote if either arm >= 20/35 or >= 11/22 mechanics-only; close the model lever if both <= 15/35; otherwise report only."
  - A10m: "promote (a full 27B ARC-AGI-3 entry) if A10m >= 20/35 or >= 11/22 mechanics-only; if A10m <= 5/35 the server-setting and effort explanations are ruled out and the model lever stays closed; between, report only."
- **Frozen**: A10l and A10x before their runs of 2026-10-22; A10m 2026-10-23, at 2663e20e.
- **Outcome**: "Decision: close (A10l 0/35 (0/22 mechanics-only), A10x 0/35 (0/22 mechanics-only): both <= 15/35; the model lever is closed for ARC-AGI-3)"; "Decision A10m: ruled_out (A10m 1/35 (0/22 mechanics-only): <= 5/35; the server-setting and effort explanations are ruled out and the model lever stays closed)".
- **Files**: `experiments/runs/kaggle-R1l-low-v1/r1_summary.txt`, `experiments/runs/kaggle-R1l-xhigh-v1/r1_summary.txt`, `experiments/runs/kaggle-R1l-medium-v1/r1_summary.txt`, their `r1_results.jsonl`, `experiments/arc3-recovery/r1l_analysis.txt`, `experiments/arc3-recovery/r1l_medium_analysis.txt`.

## R1m: restart option at the doubled budget, run twice

- **Question**: Does exposing the game's RESET action (the restart option) raise clears on the stuck states?
- **Arms**: A11r restart option; run twice.
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: "helpful if >= 21/35; harmful if <= 11/35; otherwise neutral." For the two runs together, written before the second: "helpful if the two runs together clear >= 41/70."
- **Frozen**: 2026-10-23, at 5cdb4310; the pooled line the same day, before the second run.
- **Outcome**: run 1 "neutral (A11r 20/35 between 12 and 20)"; run 2 "helpful (A11r 21/35 >= 21)"; pooled "run 1 20/35, run 2 21/35, together 41/70", "Decision: helpful".
- **Files**: `experiments/runs/kaggle-R1m-reset-v1/r1_summary.txt`, `experiments/runs/kaggle-R1m-reset-v2/r1_summary.txt`, their `r1_results.jsonl`, `experiments/arc3-recovery/r1m_analysis.txt`, `experiments/arc3-recovery/r1m_v2_analysis.txt`.

## R1n: the engine as a simulator

- **Question**: On the 22 mechanics-only levels, does a sandbox function that steps a copy of the real engine raise clears, alone or with the idea?
- **Arms**: A12S simulator; A12SI simulator with the idea.
- **Budget**: 160 actions or 30 minutes per continuation; 22 levels.
- **Decision rule**: A12S "binds at >= 12" (against the best unchanged doubled-budget sample's 8); A12SI "binds at >= 16" (against A7L's 12).
- **Frozen**: before the run of 2026-10-23.
- **Outcome**: "R1n decision: A12S cleared 9 of 22 partial cases against the best unchanged doubled-budget sample's 8 (binds at >= 12); A12SI 16 against A7L's 12 (binds at >= 16): given the idea, trial-action cost binds". The idea given the simulator: 16 vs 9, discordant 8/1, p 0.039.
- **Files**: `experiments/runs/kaggle-R1n-v1/r1_summary.txt`, `experiments/runs/kaggle-R1n-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1n_analysis.txt`.

## R1o: restart nudge

- **Question**: Does a prompt line telling the agent to restart after no progress add to the restart option?
- **Arms**: A12n restart option with the nudge as the first system-prompt line.
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: "helpful if >= 21/35; harmful if <= 11/35; otherwise neutral; and against R1m: the nudge adds value only if A12n clears >= 3 more than A11r's mean."
- **Frozen**: 2026-10-23, at 2663e20e.
- **Outcome**: "Decision: neutral (A12n 20/35 between 12 and 20)"; "Against R1m: no added value (A12n 20/35 vs A11r mean 20.5 over 2 run(s) [20, 21]; needs >= 23.5)".
- **Files**: `experiments/runs/kaggle-R1o-nudge-v1/r1_summary.txt`, `experiments/runs/kaggle-R1o-nudge-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1o_analysis.txt`.

## R1p: restart option at the standard budget

- **Question**: Does the restart option's gain hold at the standard budget?
- **Arms**: A11q restart option.
- **Budget**: 80 actions or 15 minutes per continuation.
- **Decision rule**: "helpful if >= 14/35 (3 above the best 15-minute unchanged sample, 11); harmful if <= 4/35; otherwise neutral."
- **Frozen**: 2026-10-23, at 75c3ff0c.
- **Outcome**: "Decision: neutral (A11q 10/35 between 5 and 13)".
- **Files**: `experiments/runs/kaggle-R1p-reset15-v1/r1_summary.txt`, `experiments/runs/kaggle-R1p-reset15-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1p_analysis.txt`.

## R1q: restart option with the told idea

- **Question**: Do the restart option and the told idea add?
- **Arms**: A13 restart option with the told idea.
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: "additive if A13 >= 26/35 (3 above the best single-lever run, 23); otherwise not shown additive; harmful if A13 <= 17/35 (3 below the single-lever runs' lowest, 20)."
- **Frozen**: 2026-10-23, at 75c3ff0c.
- **Outcome**: "Decision: not shown additive (A13 23/35 between 18 and 25)".
- **Files**: `experiments/runs/kaggle-R1q-insight-reset-v1/r1_summary.txt`, `experiments/runs/kaggle-R1q-insight-reset-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1q_analysis.txt`.

## R1r: restart option at four times the budget, run twice

- **Question**: Does four times the budget compound the restart option's gain?
- **Arms**: A11h restart option; run twice.
- **Budget**: 320 actions or 60 minutes per continuation.
- **Decision rule**: "time compounds the RESET gain if A11h >= 25/35 (about 4.7 above the 30-minute RESET mean, 20.3); otherwise not shown."
- **Frozen**: before the run of 2026-10-23.
- **Outcome**: run 1 "compounds (A11h 26/35 >= 25 (30-minute RESET mean 20.3))"; run 2 "compounds (A11h 25/35 >= 25 (30-minute RESET mean 20.3))".
- **Files**: `experiments/runs/kaggle-R1r-reset60-v1/r1_summary.txt`, `experiments/runs/kaggle-R1r-reset60-v2/r1_summary.txt`, their `r1_results.jsonl`, `experiments/arc3-recovery/r1r_analysis.txt`, `experiments/arc3-recovery/r1r_v2_analysis.txt`.

## R1s: unchanged at four times the budget

- **Question**: How much of R1r's gain is time alone?
- **Arms**: A0H unchanged.
- **Budget**: 320 actions or 60 minutes per continuation.
- **Decision rule**: "">= 22 time explains most of R1r's gain; <= 18 RESET adds value"; between: report only."
- **Frozen**: before the run of 2026-10-24.
- **Outcome**: "Decision: between: report only (18 < A0H 21/35 < 22)"; recorded beside, report only: time share of R1r's gain over the unchanged 30-minute mean 0.54.
- **Files**: `experiments/runs/kaggle-R1s-unch60-v1/r1_summary.txt`, `experiments/runs/kaggle-R1s-unch60-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1s_analysis.txt`.

## R1t: told the idea at four times the budget

- **Question**: Does four times the budget compound the told idea?
- **Arms**: A14 told the idea.
- **Budget**: 320 actions or 60 minutes per continuation.
- **Decision rule**: "time compounds the insight if A14 >= 26/35 (3 above the insight's best 30-min run, 23); no time effect if A14 <= 21; between: report only."
- **Frozen**: before the run of 2026-10-24.
- **Outcome**: "Decision: time compounds the insight (A14 29/35 >= 26)"; 17 of 22 mechanics-only; against A0H 29 vs 21, discordant 10/2, p 0.039.
- **Files**: `experiments/runs/kaggle-R1t-insight60-v1/r1_summary.txt`, `experiments/runs/kaggle-R1t-insight60-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1t_analysis.txt`.

## R1u: second checkpoint, unchanged

- **Question**: Does the ThinkingCap checkpoint of the agent's model spend fewer tokens per request and make more actions per minute without losing clears?
- **Arms**: A15 second checkpoint, unchanged.
- **Budget**: 320 actions or 30 minutes per continuation.
- **Decision rule** (as revised before the run): "full public run if completion tokens per request <= 0.8 x 2,225 (1,780), actions per case-minute >= 1.2 x R1s A0H's median, and clears >= 15/35; drop if tokens per request fall by less than 10% (> 2,002.5) or clears <= 12; otherwise report only".
- **Frozen**: 2026-10-24, at 0b0e44f4.
- **Outcome**: "Decision (revised): report only (actions per case-minute 5.342 vs 1.2 x 4.949 = 5.938 (below))". A15 17/35, 7 of 22 mechanics-only; completion tokens per request 1478.8 (0.665 x 2225).
- **Files**: `experiments/runs/kaggle-R1u-thinkingcap-v2/r1_summary.txt`, `experiments/runs/kaggle-R1u-thinkingcap-v2/r1_results.jsonl`, `experiments/arc3-recovery/r1u_analysis.txt`.

## R1v: second checkpoint, told the idea

- **Question**: Does the told idea's effect replicate on the ThinkingCap checkpoint?
- **Arms**: A16 second checkpoint with the idea, against R1u's A15.
- **Budget**: 320 actions or 30 minutes per continuation.
- **Decision rule**: "A16 >= 21/35: the insight effect replicates on ThinkingCap; <= 19: not replicated; 20: inconclusive".
- **Frozen**: 2026-10-24, at 5c041172.
- **Outcome**: "Decision: replicates (A16 24/35)"; against A15 17/35: A16 only 7, A15 only 0, p 0.016.
- **Files**: `experiments/runs/kaggle-R1v-tcinsight-v1/r1_summary.txt`, `experiments/runs/kaggle-R1v-tcinsight-v1/r1_results.jsonl`, `experiments/arc3-recovery/r1v_analysis.txt`.

## R1x: rule card

- **Question**: Does a written protocol to keep a one-sentence rule, predict before each batch and rewrite the rule after a failure raise clears?
- **Arms**: A20 rule card (one message: keep a one-sentence rule, predict before each batch, rewrite after a failure).
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: "A20 >= 21/35 or >= 12/22 mechanics-only: helpful; <= 11/35: harmful; otherwise neutral (report only); baseline the unchanged 30-minute samples 14, 14, 18, 17 (mean 15.75), no new baseline run".
- **Frozen**: before the run of 2026-10-27.
- **Outcome**: "neutral"; "A20 14/35, mechanics-only 6/22".
- **Files**: `experiments/runs/kaggle-R1x-rulecard-v1/r1_summary.txt`, `experiments/runs/kaggle-R1x-rulecard-v1/rulecard_decision.json`, `experiments/runs/kaggle-R1x-rulecard-v1/r1_results.jsonl`.

## WM2: world-model tools with verification enforced

- **Question**: With tools to write a step function, verify it on recorded transitions and plan with it, and with actions refused after a failure until a step function is verified, does the agent clear more stuck states?
- **Arms**: WM2 tools (write, verify and plan; after a failure, actions refused until a step function passes verification on the level).
- **Budget**: 160 actions or 30 minutes per continuation; tool timeout 60 s.
- **Decision rule**: validity (otherwise no decision, report only): the CPU twin passed; at least 33 of 35 case-arms ran to a clear, their action budget or their wall budget; the replayed prefixes match the recorded boards; the trigger fired on at least one level.
  - "**Promote** if WM2 clears at least 22 of 35 (four above the best unchanged sample, 18), or at least 11 of the 22 `partial` cases (three above the best, 8)."
  - "**Harmful** if WM2 clears 11 or fewer of 35."
  - "Otherwise: report."
- **Frozen**: 2026-10-27, as written at 8c553a6.
- **Outcome**: "report": 16 of 35, 8 of 22 `partial`; validity met (35 of 35 ran to a clear or budget; the gate closed on 28 levels). Recorded beside: 5 clears within 80 actions / 900 s; 10 of 90 verify calls returned ok.
- **Files**: `experiments/runs/kaggle-WM2-stuck-v1/r1_summary.txt`, `experiments/runs/kaggle-WM2-stuck-v1/wm2_decision.json`, `experiments/runs/kaggle-WM2-stuck-v1/r1_results.jsonl`.

## R1y: proposer

- **Question**: Does a separate request at each failure, asking the same model for three uses of the mechanics seen, each with a test of at most 5 actions, raise clears?
- **Arms**: A21 proposer (at each failure the same model proposes three uses of the mechanics seen, each with a test of at most 5 actions; the reply enters the agent's next prompt).
- **Budget**: 160 actions or 30 minutes per continuation; the proposer's time counts against the case's own 30 minutes.
- **Decision rule**: validity (otherwise no decision, report only): the CPU twin passed; at least 33 of 35 case-arms ended by a clear, the action budget or the wall budget; the replayed prefixes match; the proposer fired in at least 25 cases; proposer errors at most 10% of its calls.
  - "Helpful at 21 or more of 35, or 12 or more of the 22 partial cases."
  - "Harmful at 11 or fewer of 35."
  - "Otherwise neutral."
  - Entry line: a full-run entry if "not harmful (12 or more of 35) and at least 5 within 80 / 900".
- **Frozen**: 2026-10-27, at f334881e.
- **Outcome**: "no decision (validity failed): report only"; "A21 17/35, partial 6/22". The proposer fired in 24 cases (25 needed). Lever band recorded beside: "neutral"; entry line recorded beside: 17 clears (12 needed), 7 within 80 actions / 900 s (5 needed).
- **Files**: `experiments/runs/kaggle-R1y-proposer-v1/r1_summary.txt`, `experiments/runs/kaggle-R1y-proposer-v1/r1y_decision.json`, `experiments/runs/kaggle-R1y-proposer-v1/r1_results.jsonl`.

## R1z: fitted law library

- **Question**: Does a library of parameterised law kinds, fitted to the level's own transitions and offered as tools, raise clears, and does testing the proposer's tests in the best fitted law add to it?
- **Arms**: A22L law library (parameterised law kinds fitted to the level's transitions, as tools, on WM2's base); A22P the library plus the proposer, with each proposed test run in the best fitted law before the agent sees it.
- **Budget**: 160 actions or 30 minutes per continuation; tool timeout 60 s.
- **Decision rule**: validity (otherwise no decision, report only): the CPU twin passed; at least 33 of 35 case-arms per arm ran to a clear, their action budget or their wall budget; the replayed prefixes match the recorded boards; the library fitted at least one law at match rate 0.9 or better on at least 10 of the 35 levels; for A22P, the proposer fired on at least 20 levels.
  - "**Promote** an arm to the entry path if it clears at least 10 of 35 within 80 actions / 900 s (the entry's per-level share; unchanged 5-6, WM2 5)."
  - "**Paper result only** if an arm clears 22 or more of 35 at 30 minutes but fewer than 10 within budget."
  - "**Harmful** if an arm clears 11 or fewer of 35 at 30 minutes."
  - "Otherwise: report."
  - "**Additivity** (report only): A22P against R1y's A21, additive if at least 3 above A21 on the same 35."
- **Frozen**: 2026-10-28, at bb4d902a; revisions 3 to 5 (scoped planning over the best single law at 0.9; A22P run only if R1y is not harmful; A22P's proposer at reasoning effort medium) fixed the same day before the run, with no threshold changed.
- **Outcome**: A22L "no decision (validity failed): report only"; lever band recorded beside "report" ("A22L 20/35 at 30 min, 4 within 80 actions / 900 s"). A22P "no decision (validity failed): report only"; lever band recorded beside "harmful" ("A22P 10/35 at 30 min, 3 within 80 actions / 900 s"). Validity failed on the law check: a law at 0.9 from the model's own fits on 4 levels (10 needed). Additivity: A22P 10 against A21 17, not additive; A22P against A22L 0 only-A22P, 10 only-A22L, p 0.002.
- **Files**: `experiments/runs/kaggle-R1z-lawlib-v1/r1_summary.txt`, `experiments/runs/kaggle-R1z-lawlib-v1/r1z_decision.json`, `experiments/runs/kaggle-R1z-lawlib-v1/r1_results.jsonl`.

## T1: placebo at the doubled budget

- **Question**: At the doubled budget, does a same-length text of true but irrelevant rules recover the told idea's gain?
- **Arms**: A7cL placebo (A7c's text, byte for byte).
- **Budget**: 160 actions or 30 minutes per continuation.
- **Decision rule**: validity: twin passed; at least 33/35 ended by clear, action budget or wall budget; prefixes match; placebo texts R1f's A7c byte for byte. "P = A7cL clears of 35 at 30 min: P <= 17 content effect supported; P >= 20 content effect not supported at 30 min; 18-19 report".
- **Frozen**: 2026-10-28, at 37219cf.
- **Outcome**: "report"; "P = 18/35 at 30 min (8 within 80 actions / 900 s)"; 9 of 22 mechanics-only.
- **Files**: `experiments/runs/kaggle-A3-T1-placebo30-v1/r1_summary.txt`, `experiments/runs/kaggle-A3-T1-placebo30-v1/t1_decision.json`, `experiments/runs/kaggle-A3-T1-placebo30-v1/r1_results.jsonl`.

## C1: told the idea from every verified level start

- **Question**: Does the told idea convert levels across the public set from a clean level start, not only at stuck states?
- **Arms**: A0F unchanged, from the level start with no earlier conversation; A7F told the idea as the first message. Levels: every public level whose key-idea claim is engine-checked (159 before the run; 157 on the final list), both arms paired by level.
- **Budget**: 160 actions or 30 minutes per level start.
- **Decision rule**: validity (otherwise no decision, report only): the CPU twin passes; at least 95% of level starts replay to their hash (failures excluded from both arms); at least 95% of case-arms per arm end by a clear, the action budget or the wall budget; the idea list and texts equal the committed list; both arms of each level in the same run. "G = A7F clears minus A0F clears over the N levels in the run; exact two-sided McNemar on the discordant pairs."
  - "**Generalises:** G ≥ 10 and p ≤ 0.01. The told insight converts levels across the public set, not only at stuck states."
  - "**Confined:** G ≤ 3. The effect does not show from a clean start on the full set. It is specific to stuck states, or the unchanged arm is near ceiling; the band table shows which."
  - "**Otherwise:** report."
- **Frozen**: 2026-10-29, at d72c696.
- **Outcome**: "generalises": N 157, A0F 90, A7F 116, G 26, discordant 35 (A7F only) against 9 (A0F only), p 0.00010604466626773501; validity met.
- **Files**: `experiments/runs/kaggle-A3-C1-levelstarts-v1/c1_decision.json`, `experiments/runs/kaggle-A3-C1-levelstarts-v1/level_map.tsv`, `experiments/runs/kaggle-A3-C1-levelstarts-v1/k1/r1_summary.txt` (and `k2`, `k3`).

## C1P: placebo from level starts

- **Question**: On C1's levels, does the level's idea convert more levels than a same-length block of true rules about the same game?
- **Arms**: A7cF placebo from starts (R1f's control-text rule applied to each level, as the first message), against C1's A7F and A0F.
- **Budget**: 160 actions or 30 minutes per level start.
- **Decision rule**: validity (otherwise no decision, report only): the CPU twin passes; at least 95% of level starts replay to their hash; at least 95% of case-arms end by a clear, the action budget or the wall budget; the placebo texts equal the committed list and the 35 stuck cases match R1f's A7c byte for byte; every cell other than the arm, its messages and the level list identical to C1's. "P = A7cF clears over the N valid levels. The content effect C = A7F − P, with C1's A7F, compared level by level with an exact two-sided McNemar test."
  - "**Content supported:** C ≥ 10 and p ≤ 0.01. The level's insight converts more levels than a same-length block of true rules about the same game."
  - "**Content not supported:** C ≤ 3. A rule-shaped message about the game recovers about all of C1's gain. The paper narrows "told the insight" to "told rules of the game"."
  - "**Otherwise: report.** The paper states P beside A0F and A7F."
  - Revision 1, before any text or build, no threshold changed: the 11 levels with no core-claim subset within 20% of the idea's length are left out, so N = 146.
- **Frozen**: 2026-10-29, at f4c1658.
- **Outcome**: "report": N 146, P 91, A7F 107, A0F 81, C 16, discordant 26 (A7F only) against 10 (A7cF only), p 0.011330984183587134. Recorded beside: message effect P − A0F 10 (p 0.121), 0.385 of C1's gain.
- **Files**: `experiments/runs/kaggle-A3-C1P-placebo-v1/c1p_decision.json`, `experiments/runs/kaggle-A3-C1P-placebo-v1/k1/r1_summary.txt` (and `k2`, `k3`).

## C1S: self-written idea from level starts

- **Question**: Does an idea the agent's own model writes at each level start convert levels at C1's scale?
- **Arms**: A7SF self-written from starts (before play, the agent's own model writes the level's idea from the start frame, the level number, the available actions and the one-step effect of each; greedy, thinking off, at most 256 new tokens, cut at 400 characters; that text is the first message), against C1's A0F.
- **Budget**: 160 actions or 30 minutes per level start; a clear counts only if its actions plus the probe cost fit within 160.
- **Decision rule**: validity (otherwise no decision, report only): the CPU twin passes; at least 95% of level starts replay to their hash; at least 95% of levels get a non-empty generated text; at least 95% of case-arms end by a clear, the action budget or the wall budget; cells identical to C1's except the arm, the introduction, the bundle and the generation step. "S = A7SF clears, counted that way. M = S − A0F, with C1's A0F on the same levels, compared level by level with an exact two-sided McNemar test."
  - "**Self-insight helps:** M ≥ 10 and p ≤ 0.01."
  - "**No help:** M ≤ 3."
  - "**Otherwise:** report."
  - An entry draft needs both "helps" and "A7SF's probe-priced measured value ... above A0F's".
- **Frozen**: 2026-10-29, at 99ea2c9.
- **Outcome**: "no help": N 157, S 76, A0F 90, M −14, discordant 12 (S only) against 26 (A0F only), p 0.03355243967962451; lost to probe cost 0. Entry draft: "no (helps: False; probe-priced measured value 21.78 vs A0F 35.96)".
- **Files**: `experiments/runs/kaggle-A3-C1S-selfinsight-v1/c1s_decision.json`, `experiments/runs/kaggle-A3-C1S-selfinsight-v1/k1/r1_summary.txt` (and `k2`, `k3`).

## TCF: the second checkpoint on full public games

- **Question**: "Was Oct 29's 16.53 the model or the day?" (16.53 is the leaderboard reading of the entry with the ThinkingCap checkpoint.)
- **Arms**: TCF, the ThinkingCap checkpoint in P0's harness on the public games, no tools; against P0's 48.96.
- **Budget**: full games, not continuations: 25 public games × 2 passes (50 game-passes); everything else as P0's public-game run.
- **Decision rule**: validity: the twin passes; the checkpoint assembles to the same layout; the server loads with no error; at least 48 of 50 game-passes end with a scorecard. "D = TCF's mean game score minus P0's (48.96), with a game-paired permutation test on per-game means."
  - "**Weaker on full games:** D ≤ −5 and p ≤ 0.05. ThinkingCap is closed, and Oct 29's reading is taken as the model."
  - "**Not weaker:** D ≥ −2. Oct 29 was likely the day."
  - "**Otherwise:** report."
  - Revision 2, the same day, before any data: "weaker" stands "only if TCF's median generated tokens per game-pass is at least 80% of P0's"; "Below 80% the line reads "report (time-bound)"."
- **Frozen**: 2026-10-29, at 731383e.
- **Outcome**: "report (time-bound)": TCF mean 25.71, D −23.25, p 0.0 (the paper: below 0.0001); band without the guard "weaker on full games"; median generated tokens per game-pass 113,956.5 against P0's 171,437, ratio 0.665, guard 0.8 not met.
- **Files**: `experiments/runs/kaggle-A3-TCF-public-v1/tcf_decision.json`, `experiments/runs/kaggle-A3-TCF-public-v1/summary.txt`.

## SIF: self-written idea in full games (gated on C1S)

- **Question**: "Does the agent writing its own insight at each new level raise full-game scores once its probes are paid in real moves?"
- **Arms**: SIF, the FP8 build on the public games; at each new level the harness takes the probes in the real game (counted as moves), the served model writes the idea with C1S's generation step, and the text is prepended as one user message.
- **Budget**: full games: 25 public games × 4 passes (2 if 4 could not be placed).
- **Decision rule**: "It runs only if C1S ... reads "helps" on the probe-priced line and its entry-draft line holds. If C1S reads anything else, SIF is cancelled before any GPU use." "D = SIF's mean game score minus P0's (48.96), game-paired permutation."
  - "**Entry candidate:** D ≥ +4 and p ≤ 0.05." An entry is then drafted under its own rule.
  - "**No gain:** D ≤ 0."
  - "**Otherwise:** report."
- **Frozen**: 2026-10-29, at 731383e.
- **Outcome**: not run. C1S read "no help", so the gate was not met.
- **Files**: none (no run); the gate's input is `experiments/runs/kaggle-A3-C1S-selfinsight-v1/c1s_decision.json`.

## OIF: the level's idea in full public games

- **Question**: What is the told idea worth in competition score on full games, where levels follow one another, time is shared and every action counts?
- **Arms**: OIF, P0 on the public games with the level's key idea (C1's text, byte for byte) shown once at the first model call in each of the 157 listed levels; against P0's 48.96.
- **Budget**: full games: 25 public games × 4 passes (100 game-passes) at P0's per-game wall budget (29,018 s).
- **Decision rule**: validity (otherwise report only): the twin passes; the server loads with no error; at least 96 of 100 game-passes end with a scorecard; every level entered logs whether a message was shown and its text hash; at least 95% of listed levels entered show their message exactly once. "D = OIF's mean game score minus P0's 48.96 ..., with a game-paired permutation on per-game means ...: two-sided, 20,000 draws."
  - "**Insight raises full-game score:** D ≥ +5 and p ≤ 0.05."
  - "**No measurable gain:** D ≤ +1."
  - "**Otherwise:** report."
- **Frozen**: 2026-10-29, at acba2fb.
- **Outcome**: "insight raises full-game score": OIF 66.06, P0 48.96, D 17.11, p 0.0 (the paper: below 0.0001); shown exactly once on 554 of 554 listed levels entered.
- **Files**: `experiments/runs/kaggle-A3-OIF-public-v1/oif_decision.json`, `experiments/runs/kaggle-A3-OIF-public-v1/summary.txt`.

## Think-aloud alignment (human data, descriptive)

- **Question**: When do humans in the public think-aloud dataset ("How Humans Learn Unfamiliar Worlds: A Think-Aloud Dataset on ARC-AGI-3", arXiv:2610.34523, CC BY 4.0) recognise a level's key mechanism, and how does that line up with this agent's told-idea clears?
- **Arms**: none; two primary comparisons fixed before any outcome: P1, humans' actions to recognition (M2) at the 25 stuck levels against the other 132 C1 levels; P2, human M2 at C1's converted levels (A0F 0, A7F 1) against C1's both-clear levels. A third primary uses data not in this release and is left out here.
- **Budget**: none; the dataset's humans played with no action cap.
- **Decision rule**: descriptive, no decision. "A sentence of the form "X differs from Y" or "X goes with Y" is written only for the three primaries ..., and only when the 95% interval of the median difference (or of rho) excludes 0; otherwise the report says "no difference shown" with the interval." Agreement floor: below a Cohen's kappa of 0.40 on the key-mechanism map, the measures over all mechanisms introduced at the level are read first.
- **Frozen**: 2026-10-29, at acc8d87; revision 1 (the game-identity check run on every human trajectory) before any recognition outcome was computed.
- **Outcome**: kappa 0.7982, above the floor. P1 "no difference shown": median M2 83 against 31.75, difference 51.25, 95% interval −9.75 to 99.5. P2 "differs": median M2 58 against 23.5, difference 34.5, 95% interval 6.25 to 55.0.
- **Files**: `experiments/arc3-think-aloud/RESULT.json`, `experiments/arc3-think-aloud/levels.csv`.

## FPG: the FP8 build on full public games

- **Question**: Does the FP8 build of the agent's model lose game skill against the W4A16 build on which P0's 48.96 was measured?
- **Arms**: FPG, P0's public-game setup with the FP8 build of the same model in place of the W4A16 build; no idea message, no keep-alive, no load wait.
- **Budget**: full games: 25 public games × 4 passes (100 game-passes) at P0's per-game wall budget (29,018 s).
- **Decision rule**: validity (otherwise report only): the twin passes (model path, build checks, settings); the server loads with no error; at least 96 of 100 game-passes end with a scorecard. "D = FPG's mean game score minus P0's 48.96 ..., with a game-paired permutation on per-game means: two-sided, 20,000 draws."
  - "**FP8 loses game skill:** D ≤ −5 and p ≤ 0.05."
  - "**No measurable loss:** D ≥ −2."
  - "**Otherwise:** report."
- **Frozen**: 2026-10-30, at e672dbe.
- **Outcome**: "report": FPG 44.21, P0 48.96, D −4.75, p 0.163.
- **Files**: `experiments/runs/kaggle-A3-FPG-public-v1/fpg_decision.json`, `experiments/runs/kaggle-A3-FPG-public-v1/summary.txt`.

## TCI60: second checkpoint, told the idea, 60 minutes

- **Question**: Does more time add to the told idea on the ThinkingCap checkpoint, as actions and time together did on the W4A16 build? (Here time changes alone: R1v already ran at 320 actions.)
- **Arms**: A16 second checkpoint with the idea (R1v's arm), with 3,600 s per case.
- **Budget**: 320 actions or 60 minutes per continuation.
- **Decision rule**: validity (otherwise report only): the twin passes; 35 of 35 case-arms end, with no oracle errors; the idea shown in 35 of 35; cells identical to R1v's except the time budget. "TCI60's clears of 35, paired with R1v's on the same 35 states."
  - "**Time adds on ThinkingCap:** TCI60 ≥ 28 (+4 of the 9 that time can reach ...)." (R1v missed 11 of 35: 9 on the wall budget, 2 on the action budget.)
  - "**No added effect:** TCI60 ≤ 25 (≤ +1)."
  - "**Otherwise (26 or 27):** report."
  - "The exact McNemar p (two-sided) against R1v is recorded beside."
- **Frozen**: 2026-10-30, at 7475f7d.
- **Outcome**: "no added effect": "A16 22/35"; R1v 24; p against R1v 0.625.
- **Files**: `experiments/runs/kaggle-A3-TCI60-v1/tci60_decision.json`, `experiments/runs/kaggle-A3-TCI60-v1/r1_summary.txt`, `experiments/runs/kaggle-A3-TCI60-v1/r1_results.jsonl`.

## HL: hint library in full games (HLA, HLB)

- **Question**: Does a library of general hints help when it is built only from other games, kept in view on every model call, and refreshed at each level start and when the model is stuck?
- **Arms**: FPG with a 365-entry pool (R1j's 57 patterns with 308 further tricks, leave-one-game-out) in the system prompt, picked by the harness from cue tags (HLB) or by the model (HLA); up to 15 hints and 2,000 characters; against FPG's 44.21.
- **Budget**: full games: 25 public games × 4 passes (100 game-passes), as FPG; lookups at each level start, then after 25 actions or 10 minutes without a clear, at most 4 per level.
- **Decision rule**: validity (otherwise report only): the twin passes; the server loads with no error; at least 96 of 100 game-passes end with a scorecard; every level entered logs its lookups; on at least 95% of levels entered the picked hints are in the system prompt at the level's last model call; HLA: at least 90% of its calls return a valid id list. "Per arm: D = the arm's mean game score minus FPG's 44.21 .... The test is game-paired on per-game means, two-sided, with 20,000 draws, as OIF and FPG. With two arms, each is read at p ≤ 0.025."
  - "**Entry candidate:** D ≥ +5 and p ≤ 0.025. If both arms qualify, the one with the larger D."
  - "**No measurable gain:** D ≤ +1."
  - "**Otherwise:** report."
- **Frozen**: 2026-10-30, at a561b9f.
- **Outcome**: HLA "no measurable gain": 36.79, D −7.42, p 0.0293 (836 of 910 calls valid, 0.9187). HLB "no measurable gain": 41.27, D −2.94, p 0.3756.
- **Files**: `experiments/runs/kaggle-A3-HLA-public-v1/hl_decision.json`, `experiments/runs/kaggle-A3-HLA-public-v1/summary.txt`, `experiments/runs/kaggle-A3-HLB-public-v1/hl_decision.json`, `experiments/runs/kaggle-A3-HLB-public-v1/summary.txt`.

## HLS and HLC: hint arms with the prompt front fixed

- **Question**: With the prompt front kept fixed (so the server's prefix cache holds), do the hints cost anything?
- **Arms**: HLS, up to 15 of the 57 patterns placed once in the system prompt at the game-pass's first call, no further lookups; HLC, HLB's picks at the start of the lookup turn's message, system prompt unchanged; against FPG's 44.21.
- **Budget**: full games: 25 public games × 4 passes (100 game-passes), as FPG.
- **Decision rule**: validity (otherwise report only): the twin passes (system prompt constant within each game-pass; the arm's section placed as specified); the server loads with no error; at least 96 of 100 game-passes end with a scorecard. "D = the arm's mean game score minus FPG's 44.21 ..., game-paired on per-game means, two-sided, with 20,000 draws."
  1. "**Cache gate:** prefix-cache hit at least 92% and at least 19,100 actions (95% of FPG's). An arm that fails is not eligible." "If an arm reaches 92% but not 19,100 actions ... Report."
  2. "**Score gate:** D ≥ -2: eligible. D ≤ -5: dropped. The hints cost points in this form. Between: report, a separate decision."
  3. "**Pick:** the eligible arm with the larger D. If the two are within 2.0 points, HLS, which has fewer parts ...."
  4. "**Gain claim:** only at D ≥ +5 and p ≤ 0.025, as in the HL rule. The gate above is "not worse", not "better"."
  - Leaderboard prediction for the pick, written before submission (base FP8 28.90; HLB 23.67): "At or above 27.0: on hidden games the hints cost nothing once the cache is fixed." "At or below 24.0: the hints cost points on hidden games beyond throughput." "Between: unresolved on one reading."
- **Frozen**: 2026-11-01, at f2c4ab2f.
- **Outcome**: HLS "score gate: eligible": 42.29, D −1.91, p 0.5505; cache gate pass (hit 0.9354, 20,369 actions); no gain claim. HLC "score gate: eligible": 49.35, D 5.15, p 0.1157; cache gate pass (hit 0.9348, 21,291 actions); no gain claim (p above 0.025). Pick under the rule: HLC, "the eligible arm with the larger D". Leaderboard readings: HLC 26.51, HLS 29.13; no reading against the prediction bands is recorded in the result files.
- **Files**: `experiments/runs/kaggle-A3-HLS-public-v1/hl_decision.json`, `experiments/runs/kaggle-A3-HLS-public-v1/summary.txt`, `experiments/runs/kaggle-A3-HLC-public-v1/hl_decision.json`, `experiments/runs/kaggle-A3-HLC-public-v1/summary.txt`, `submissions/ledger.json`.

## HLS pattern texts: competition-rule check

- **Question**: Do the competition rules allow selecting HLS with its embedded pattern texts?
- **Arms**: none; a reading of the ARC-AGI-3 competition rules, required by the HLS and HLC rule before HLS is selected. The 57 patterns HLS shows the model are our own text, written from the 25 public games' source with no game name, colour, coordinate or count.
- **Budget**: none.
- **Decision rule**: none; the check states which competition rules apply to the texts.
- **Frozen**: 2026-11-02 (date of the check).
- **Outcome**: "No rule blocks selecting HLS with the embedded texts." One interpretation, "validation dataset" in rule 3.4.b, is recorded, not resolved. "This is a reading of the rules, not legal advice."
- **Files**: `kaggle/arc3/hl/entry/hls/arc-agi-3-fp8-hls.ipynb` (the entry with the texts).
