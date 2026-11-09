# ARC-AGI-3 game rules from source, and rule discovery in P0's failed levels

Rules of the public ARC-AGI-3 games, written from the competition's game source files
(`environment_files/<game>/<version>/<game>.py`, arc_agi 0.9.9), with every statement tied to source lines
and, where possible, to a reproducible engine check. Used to label whether the P0 agent discovered each
failed level's key insight (`../arc3-failure-annotation/labels_rule_A.csv`).

Game source is competition data and is not stored here; files quote at most short snippets with line numbers.
Each file records the sha256 of the source it was written from (all 25 checked against the competition files).

## Files

| file | content |
|---|---|
| `<game>.md` | one per game: header (version, levels, sha256, baseline actions, what ACTION1-7 do in this game), core rules, one section per level (layout, what is new, key insight, winning strategy), engine-check table, not-verified list |
| `rules.json` | `{game: {game_id, version, sha256, n_levels, actions, core, levels: {L: {new, key_insight, win, engine_checked}}, claims: [...], engine_checks, status, unverified}}` |
| `claim_results.tsv` | result of re-running every claim's check on the merged `rules.json` (claim id, PASS / inspect / source-only) |
| `STATUS.md` | per-game coverage and claim counts; regenerated at each commit |
| `../../scripts/arc3_rules_probe.py` | engine helper: reach a level, run actions, per-step diffs; `claim <id>` re-runs a recorded check |
| `../../scripts/arc3_rules_merge.py` | merges per-game fragments into `rules.json`, writes/checks `labels_rule_A.csv`, cross-tabs, `STATUS.md` |

## Claims and check status

Every rule statement in a `.md` file ends with `[L<a>-<b>; <claim id>; engine | source-only]`. Claim ids are
stable: `<game>.core.c<n>` for core rules, `<game>.L<n>.k<m>` for level statements, where `k1` is the level's
key insight and `k2` its win condition / strategy. `rules.json` lists every claim with its text, source lines,
status and, for engine claims, the exact check: start (`human` = replay a human winning session's earlier
levels, carrying cross-level state and the true `levels_completed`; `set_level` = jump to the level on a fresh
game), action tokens, and the expected outcome.

Settling a disputed claim:

    export ARC3_ENV_DIR=<environment_files dir> ARC3_HUMAN_JSON=<human_levels.json>
    python scripts/arc3_rules_probe.py claim sk48.L3.k1        # one claim; 'sk48.L3' = all claims of a level; 'all'
    python scripts/arc3_rules_probe.py run sk48 --level 3 --via-human --actions "1,1,4,6:32:10,3x5" --show-every

`claim` prints per-step board diffs (cells changed, bbox, colour transitions), `levels_completed` and state,
then judges the recorded expectation: `level_up` (cleared on the last action, not before), `no_level_up`,
`game_over`, `no_game_over`. Claims recorded without an expectation print `inspect`: the writer read the
trace against the claim's `note` (what to look for); a checker re-reads it. `ARC3_HUMAN_JSON` is built by
`python scripts/arc3_rules_probe.py extract --parquet-dir <HF shards> --out human_levels.json` from the human
play data (`../arc3-human-data/`; not stored here, no license stated): per game the shortest winning session
that replays frame-exactly on the competition engine, else (bp35, cn04, lf52, s5i5, vc33) the shortest whose
win reproduces, split into levels by the engine's own `levels_completed`.

## Method

One writer per group of 2-3 games read the game class (step logic, win/lose, level list) and wrote the rules;
key insights and winning strategies were then tested on the engine. For each failed level in `cases.csv`, the
key insight (`k1`) and the win condition (`k2`) were engine-checked from the human prefix, normally by a
winning sequence built from the insight (or the human's own sequence for the level) plus, where cheap, a
contrast check showing that the naive alternative does not clear (e.g. tr87 L6 rules assigned to the wrong
source symbols, ka59 L6 blast ridden without pushing the bomb first, cn04 L6 one stack left short). Rule
writing preceded reading the dossiers; no key insight was changed after reading them.

## Status (2026-10-20)

All 25 public games are complete for every level (183 levels): the 19 games in `cases.csv` (144 levels) and
cd82, dc22, ft09, lp85, sb26, vc33 (39 levels).

Claims over the 25 games in `rules.json`: 794, of which 629 engine-checked and 165 source-only. Re-running
every check: 547 PASS, 82 inspect, 0 FAIL (`claim_results.tsv`; the 19 `cases.csv` games were re-run together
on the merged file, the 6 others per game). For the 19 `cases.csv` games: 694 claims, 538 engine-checked (307
from a human prefix, 231 via `set_level`), 456 PASS, 82 inspect, 156 source-only. Claims on the 25 failed
(game, level) pairs: 101 (85 PASS, 5 inspect, 11 source-only); every failed level has its `k1` and `k2`
engine-checked.

Typical source-only statements: HUD colours and layout, absence of randomness, mechanics no level uses,
undo details, and in some games the key insights of levels that were not failed (backed only by the human
winning sequence, e.g. bp35 L3-L9, lf52 L3/L5-L10, sk48 L4-L8, cn04 L2-L5). Each file's "Not verified"
section lists them.

## Limits

- Written from source by one writer per game, single pass; engine checks confirm outcomes of action sequences,
  not the wording of a rule. To be spot-checked by a human against replays.
- `inspect` claims (82) rest on the writer's reading of a printed trace.
- Key insights are one formulation of what a level requires; a level can admit other routes. Where only the
  human sequence was checked, the stated insight is the writer's reading of that sequence and the source.
- `set_level` starts do not carry cross-level state (e.g. facing direction in sc25); claims that depend on it use
  the human prefix.

## Rule discovery in P0's failed level attempts (labelling pass A)

`../arc3-failure-annotation/labels_rule_A.csv`, one row per failed level attempt (35). Labels were drafted per
game by the same writer, after that game's rules file was finished, from the dossiers only; every
`evidence_quote` is checked verbatim against its dossier by `arc3_rules_merge.py labels/check`. Definitions:
`yes` = the agent stated the level's key insight (`k1`, or an equivalent) as its working belief at some turn on
the level; `partial` = core mechanics right, key insight missing or half-stated; `no` = core mechanics for the
level wrong or never stated; `unclear` = dossier too thin. Single labeller; treat counts as one reading.

| rule_discovered | stratum A (n=10) | stratum B (n=25) | all (n=35) |
|---|---|---|---|
| yes | 3 | 8 | 11 |
| partial | 6 | 16 | 22 |
| no | 1 | 1 | 2 |
| unclear | 0 | 0 | 0 |

Key insight stated at any turn (even as one hypothesis among several): 12/35 (the 11 `yes` plus lf52_p2_L4_F,
where lockstep platform motion appears only as an untested hypothesis at turn 42). Of the 11 `yes`, 10 acted on
the insight afterwards; tu93_p2_L8_F simulated the exact 21-move winning plan at turn 7 and the pass ended
(job deadline) before it acted. All 11 `yes` passes ended by scheduler stop or job deadline with the level
uncleared.

Reading of the `partial` group: in 22 of 35 the mechanics were right and the missing piece was the level's
specific insight, typically a mechanic in the game's own repertoire that the agent ruled out or never tested
(ka59 blasts as transport, m0r0 the clickable box as a button-holder, wa30 the player-to-helper relay, sc25 the
teleport pad cycle, cn04 stretching pieces with ACTION5, g50t a third recording wiping all ghosts).

### Cross-tabulation with failure_category

Read only after `labels_rule_A.csv` was written and committed (commit 81a73e9).

| failure_category (labels_A.csv) | yes | partial | no | total |
|---|---|---|---|---|
| modeling | 5 | 9 | 1 | 15 |
| planning-execution | 3 | 5 | 1 | 9 |
| harness-runtime | 2 | 3 | 0 | 5 |
| goal inference | 0 | 4 | 0 | 4 |
| exploration | 1 | 0 | 0 | 1 |
| perception | 0 | 1 | 0 | 1 |

| failure_category (labels_B.csv) | yes | partial | no | total |
|---|---|---|---|---|
| modeling | 5 | 6 | 2 | 13 |
| harness-runtime | 4 | 6 | 0 | 10 |
| planning-execution | 1 | 4 | 0 | 5 |
| goal inference | 0 | 4 | 0 | 4 |
| exploration | 1 | 2 | 0 | 3 |

`unclear` is 0 in every row and is omitted. The two readings do not line up one-to-one: 5 of the 15 pass-A
`modeling` failures (and 5 of 13 pass-B) are cases where the key insight was discovered, the modelling error
lying elsewhere (e.g. bp35_p1 spike "halo", re86_p3 square placement, ka59_p1 budget); goal-inference failures
are all `partial`; harness-runtime failures split 2/3 (A) and 4/6 (B) between `yes` and `partial`.

### Discovered but not cleared (examples)

- tu93_p2_L8_F (A modeling, B harness-runtime): "solution probably involves luring the chaser out and around"
  (turn 4); the exact lure-and-loop plan was simulated to the goal at turn 7; the pass hit the job deadline
  before executing it.
- lf52_p1_L2_F (A, B exploration): "The box is now at pixel (51,37) = lattice (7,5), carrying the peg, right
  next to panel 2's socket" (turn 39); stopped by the scheduler one select-and-jump short of the win.
- tr87_p1_L6_F (A, B harness-runtime): two-level rule chain worked out by turn 5 ("requirement Y(W) = ybar[2]
  = ybar[3] = x12"); the full plan ran blind in a loop cut twice by the 30 s tool timeout, then the pass was
  stopped with the rules partly set.

### Never discovered (examples)

- bp35_p3_L2_F (`no`; A, B modeling): "clicking the band plant at (33,45) in world A killed the diver (the
  blocks fell onto it)" (turn 31); a falling-blocks/tide model replaced the actual spike hazard; five deaths in
  the same shaft.
- tu93_p0_L9_F (`no`; A planning-execution, B modeling): "sweepers ARE lethal, and side-entry does NOT
  absorb" (turn 31); the follower was modelled as a shortest-path pursuer and patrollers as unkillable, so
  searches found no plan. 23 middle turns are thinned in the dossier; least certain row.
- ka59_p3_L6_F (`partial`, insight never stated; A modeling, B exploration): "So the top is definitely sealed
  for the player." (turn 17); the downward blast wave was described but its use as transport was never inferred.
