# VR-bank results (run kaggle-A2-VRbank-v1, 2026-10-27/28)

Rule: `VR_BANK_RULE.md` (frozen 2026-10-27). Kernel `arc-agi-2-vrbank` v1.
- Twin PASS. The run took 3 h 17 m on 4xL4.
- The vLLM server came up on option set 0 after 498 s.
- Neither replicate was cut by the window. Both used their full token pool (1.79 M and 1.80 M of 1.8 M).
- Analysis: `scripts/arc2_vrbank_analyze.py` → `experiments/runs/kaggle-A2-VRbank-v1/vrbank_decision.json`.

## Validity: failed (one condition), so every rule is report only

| condition | needed | A | B |
|---|---|---|---|
| s1 on tasks | >= 27 of 30 | 30 | 30 |
| median s1 completion tokens | >= 1,000 | 49,152 | 49,152 |
| empty or zero-token replies | <= 5% of records | 8 of 66 | 0 of 66 |
| sealed task read | none | none | none |

Over both replicates the empty-reply share is 8 of 132 records, 6.1%.
- All 8 are first-sample rescues in replicate A. Each ended on the client timeout after 1,558-2,200 s with no reply.
- Cause, a build choice of this run, not the entry's behaviour:
  - The two replicates ran at once with 20 requests in flight each, against a server that serves 20 at a time.
  - The short rescue requests queued behind 49k-token first samples.
  - The harness's request timeout is sized for 20 in flight on its own server (3 x tokens / per-stream rate), as in the entry, and does not allow for queueing.
  - The entry runs one harness on its server and would not queue.
  - Server log: no error, abort or context overflow. 20 requests waited beside 20 running for most of the run (up to 31 waiting at the start). The 8 failures are client timeouts, not a server fault.

## Rules as they would read if valid (all report only now)

1. **Repair round.** 18 repair rounds over A and B (A 3, B 15), fewer than the 20 the rule needs, so "too few".
   - Net right outputs: 1 (dbff022c[0], replicate A).
   - 438,442 repair tokens, which is 2.28 net right outputs per million.
   - The count is not independent of the timeouts. After a timed-out rescue, a task's chain goes to second samples instead of a repair round: A ran 3 repairs and 6 second samples, B ran 15 repairs and none.
   - Without the timeouts the count would likely have passed 20, and the reading cannot be inferred.
2. **Verified-first merge.** Verified-clean precision is 25 of 33 outputs, 75.8%.
   - By stage: s1 15/19, rescue 8/12, rescue_repair 2/2.
   - That is the 75-90% band, so report only.
3. **Rescue replay go/no-go.** 64 cut samples with a saved reasoning tail, against a minimum of 15, so the replay can be built.

## Reported, no rule

| | outputs right of 44 (pass@2) |
|---|---|
| B0 alone (the bank's attempts) | 12 |
| v7 direct merge (bank rule `v7_direct`) | 17 |
| v7-vr2 phase 2 over B0, replicate A | 18 |
| v7-vr2 phase 2 over B0, replicate B | 19 |

- Repair effect on the entry's answers: +1 gained, 0 lost.
- Second samples (A only, 6 plus 5 rescues; B spent its pool on repairs): 0 net right; effect 0 gained, 1 lost.
- Rescue precision: 14 of 74 rescued outputs right (19%).
- A and B disagree on 9 outputs.
- B0's file equals the bank's rule b0.
- Tokens per stage:
  - s1: A 1.39 M, B 1.38 M.
  - Repair: A 63 k, B 360 k.
  - s2 (A only): 295 k.
  - Rescues: 23-60 k per replicate.

## Limits

- The pool (60,000 x 30) binds: s1 alone takes about 1.39 M of 1.8 M per replicate. That leaves about 400 k per replicate, about 12-16 repair rounds at 24-32 k each. So the rule's 20 rounds are reached only when both replicates spend the rest on repairs.
- Two replicates on 30 shortest public tasks. The pass@2 rows are two draws, not a score.
