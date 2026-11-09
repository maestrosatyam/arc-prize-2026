# B0 speed analysis for W2

Question: can B0's per-task work be made cheaper without changing what it produces, enough that the 0.05 floor (W1)
projects to <= 11 h on 240 hidden tasks with 4 ranks? Answer: no. The output-preserving changes available save at most
a few tens of percent of DFS time at the 0.05 floor; the target needs about 2.9x on the whole search. W2 and W3 were
not built. Analysis only, no GPU used.

Sources: run logs `experiments/runs/kaggle-A2-pool-audit-v1/arc-agi-2-pool-audit.log` and
`kaggle-A2-W1-v1/arc-agi-2-w1-wide-search.log` (per-line timestamps of each rank's `allocated .. for training`,
`decoding`, `scoring`, `finished` lines), the two runs' pools, and the solver as the audit notebook writes it.
Phases per task: TTT = previous task end (or rank start) to `allocated .. for training`; DFS = `decoding` to the
batch's first `scoring` line (or the next event); rescoring = first `scoring` to the next `decoding` / end.

## Where the time goes

| per task (40 tasks) | audit (0.2) | W1 (0.05) |
|---|---|---|
| mean task s | 626 | 1041 |
| TTT s (of which `train_runtime`) | 192 (183) | 193 (182) |
| DFS s | 386 (62 %) | 736 (71 %) |
| rescoring s | 51 (8 %) | 117 (11 %) |
| DFS batches decoded / candidates rescored | 260 / 477 | 206 / 1150 |
| tasks at the 1200 s cap | 0 | 20 (mean 1340 s) |

- On the 206 batches both runs decoded, W1's DFS takes 2.81x the audit's (median per batch 2.74x; 3 batches hit 540 s).
- W1's search (DFS + rescoring) is 852 s/task as measured; scaled to all 260 batches about 1318 s/task.
- Rescoring: 4.3 s per new candidate (8 sequences of ~2200 tokens, full forward), about 35 TFLOP/s on L4: compute-bound.
- TTT outside `trainer.train()`: 6.2 s per task (trainer construction, tokenization); first task per rank +12-94 s.
- Rank tails: audit 840 rank-s idle after a rank's last task (about 210 s wall), W1 2997 rank-s (about 750 s).
  Ranks start 228 s apart (serial unsloth import, B0's patching workaround).

## Required saving

11 h = 39 600 s. With the audit's measured overhead (636 s), mean task time must be <= 649 s (642 s with W1's
1070 s overhead). TTT (192 s) is untouched by any output-preserving change, so the 0.05 search must fit in <= 457 s
per task, against 852 s measured (cap-truncated) and about 1318 s for all batches: a 1.9x to 2.9x speed-up of the
search. The 1.9x end is not reachable as a mean: the 20 capped W1 tasks alone contribute 670 s to the 40-task mean,
so a faster search only helps once it lets those tasks finish all batches before the cap, which needs close to 2.9x.

## Candidates

| | change | output-preserving | expected saving | risk |
|---|---|---|---|---|
| a | DFS: merge 4-row batches with equal prefix length into 8 or 16 rows | rows independent in exact arithmetic; decode numerics change with batch shape | 0.05: <= 18 % (8 rows) / <= 27 % (16 rows) of DFS steps, 0-15 % of DFS time after per-step cost; 0.2: 0-30 % of DFS | memory (KV 147 KB/token/row, x2 during DFS); 540 s DFS cap covers more rows |
| b | KV reuse / redundant passes in turbo_dfs | - | about 0: nothing redundant | - |
| c1 | rescoring batched across candidates | yes | about 0: compute-bound, same FLOPs | - |
| c2 | rescoring with the 8 augmented queries' KV cached per test output | no (different attention path) | about 85 % of rescoring: 7 % of audit time, 11 % of W1's | needs a non-unsloth forward for multi-token steps on a cache |
| d | longest-first task queue | yes | <= 1.5 % of a full run (tail is about one task per rank on 60 tasks) | low |
| e | TTT overhead | yes | <= 1 % (6.2 s of 192 s per task is outside training) | low, not worth the change |
| f | merge the LoRA adapter into the weights for decoding (restore base weights per task) | no (bf16 rounding of merged weights) | 10-35 % per decode step (1 GB less read, ~750 fewer kernels) | unsloth must honour `merged`; base copy per task |

(a) in detail. `turbo_dfs` runs the 4 views of a batch in lockstep: each step pops the next candidate token of every
row, so the 4 rows share forward passes only along aligned sibling indices, mainly the greedy path. Merging k batches
removes at most k-1 greedy paths per merged group. A greedy path is at most the audit's whole per-batch step count,
which is <= 1/2.81 = 36 % of W1's lockstep steps; so merging all 4 batches of an output (possible only when the two
orientations have equal prefix length, e.g. square grids) removes <= 27 % of W1's steps, 2 batches <= 18 %, even if a
16-row step cost the same as a 4-row step. Estimated per-step cost at L about 2000 tokens: weights 8.3 GB (base +
r=256 LoRA) plus KV 4 x L x 147 KB per step, so 8 rows about 1.1-1.2x and 16 rows 1.3-1.6x a 4-row step. Branches of
different views sit at different token depths (the uncertain cells move under rotation and colour permutation), so
they rarely align. Only frontier batching (independent branches of any row packed into one forward, each with its own
KV slot, masked attention and per-row positions) could reach 3x; that bypasses unsloth's paged decode path, is not
bit-identical, changes which work the 540 s / 1200 s caps cut, and cannot be tested without a GPU.

(b) The DFS already reuses the cache: in unsloth's decode path each layer writes the new token at index `seq_len` of a
per-call buffer and attends over `[:seq_len + 1]`, so siblings overwrite their predecessor's slot (stack discipline);
the full-prefix forward resets the buffer, and the prefix is computed once per batch (about 3 % of DFS time at 0.2,
1 % at 0.05, cost model). The 4 views in a batch are different augmentations: no shared prefix. Taken from the
unsloth 2025.9.7 source, not checked against the installed copy.

Best case, everything stacked (a at 16 rows, c2, d, f at their upper estimates; two of the four not output-preserving):
DFS about 1171 s / (1.37 x 1.35) = 633 s, rescoring 22 s, TTT 192 s: about 850 s per task, 14.4 h projected. With only
the output-preserving changes (a, d): about 1200 s per task uncapped; with the cap binding, W1's 17.6 h less a few
percent. Neither reaches 11 h.

## Run-to-run nondeterminism (for any later identity test)

The audit and W1 decode the same views; beams with p >= 0.2 are found identically by both floors in exact arithmetic.
On 616 views decoded by both runs (tasks with a DFS batch cut at 540 s excluded): the p >= 0.2 grid set differs in 331
(54 %; 264 grids only in the audit, 254 only in W1); for the 461 common (view, grid) pairs the beam score never matches
exactly (median |delta| 0.157 nats, p90 0.75); the 8 augmentation scores never match (median of the per-candidate max
|delta| 4.9 nats). Test-time training is not bit-reproducible, and the rescoring augmentations are seeded by
`hash(bk) % 1024**2`, which Python randomizes per process unless PYTHONHASHSEED is set (inferred from the score
spread; the environment variable was not inspected). Consequences: "same selected attempts as the audit" cannot be
tested across runs; output preservation of a decoder change can only be checked inside one run (both decoders on the
same adapter and views); and B0's own selection varies with the per-process hash seed.

## Outcome

No safe speed-up fits W1's widening into 11 h, so W2 (and W3) were not built and no GPU was used.

Not verified: per-step decode cost on L4 (estimated, not measured; the bound on (a) does not depend on it); the
unsloth decode-path details in (b); the hash-seed explanation of the rescoring-score spread.
