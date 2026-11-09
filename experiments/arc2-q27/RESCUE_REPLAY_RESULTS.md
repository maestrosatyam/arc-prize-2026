# Rescue replay results (run kaggle-A2-rescue-replay-v1, 2026-10-28)

Rule: `RESCUE_REPLAY_RULE.md` (frozen 2026-10-27, as written in cbcbf70). Kernel
`arc-agi-2-rescue-replay` v1, built from 6d5d72f.
- Twin PASS. The run took 84 min on 4xL4, to the window's stop.
- The server came up on option set 0 after 549 s.
- Analysis: `scripts/arc2_rescue_replay_analyze.py` → `experiments/runs/kaggle-A2-rescue-replay-v1/replay_decision.json`.

## Reading: valid; block does not help

Validity holds on every count:
- 64 cut samples with a saved tail, against a minimum of 15.
- Median completion tokens 2,199.5 (plain) and 1,565 (block); the minimum is 5.
- No empty or zero-token replies in either arm.
- No sealed task.

Rule result: 196 paired units (64 cut samples × test outputs × 2 draws). Gains 4, losses 7, net −3; one-sided sign test p 0.89; block's verified-clean wrong answers +4 over plain's. Net at most 0 means **block does not help: the rescue request stays v7-vr2's.**

| per arm (196 units) | plain | block |
|---|---:|---:|
| right (the entry's ranking of the one rescued sample) | 31 | 28 |
| right in the written grid or the program output | 38 | 32 |
| programs that ran | 123 | 126 |
| verified-clean outputs, right / all | 26 / 41 (63%) | 23 / 42 (55%) |
| median completion tokens | 2,199.5 | 1,565 |
| examples shown; with a Rule line | 0 | 640; 536 |

- Gains and losses by the cut sample's stage: s1 2 and 4, repair 2 and 3, s2 0 and 0. By replicate: A 3 and 1, B 1 and 6.
- Check against the entry: the VR-bank's own recorded rescue of the same cut samples (one sample, scored the same way) is right on 13 of 74 units (17.6%). Plain here is 31 of 196 (15.8%), so the replay reproduces the entry's rescue rate.
- Both paired arms finished 2,553 s after the replay started, inside the window.

## Report-only arms: no usable reading

- **R1** (36 release hypotheses with "fails on example k", harness repair prompt, thinking on, 24,000 tokens):
  - 20 of 36 had a record when the window closed. The rest had not finished.
  - All 20 ran to the token cap inside their thinking ("cut"), so none wrote an answer.
  - The arm sent no rescue turn after a cut, unlike the harness's own repair round. That is a design gap of this run's report arm, not a reading of the repair round.
- **R2** (29 unverified programs): not reached before the window closed.

## Limits

- Units within a cut sample and the two replicates of a task are correlated, so p is optimistic. The reading would not change: net is negative.
- Public evaluation tasks, read before. This chooses a rescue request; it is not a score.
