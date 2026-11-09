# arc_harness in hybrid v7-vr: source and patches

Source: the team's `arc-harness` repository (zip received 2026-10-25), package `arc_harness/` and `tests/`, copied
unchanged except for the patches below. Licence: MIT-0 (Null Hypothesis Enthusiasts); Poetiq-derived parts MIT
(`NOTICES.md`). Built into the Kaggle kernels by `build_v7vr.py` (v7vr/) and `build_v7vr2.py` (v7vr2/); driver `hybrid_vr.py`.

| File | Patch | Why |
|---|---|---|
| `sandbox.py` | The interpreter that runs the model's programs comes from `ARC_SANDBOX_PYTHON` (default `sys.executable`) | In the hybrid the harness runs under the serving Python 3.12, whose numpy lives only in the vLLM site; `-I` hides it, so every program would fail on `import numpy`. The notebook points it at the image's Python |
| `prompts.py` | numpy imported only when the demonstration order is shuffled | The serving Python may not import numpy at all |
| `llm.py` | HTTP 400 "maximum context length": retry at 0.75x `max_tokens` (up to the retry count) | The harness otherwise drops the task; the public direct mode retries the same way |
| `solvers.py` | `Config.only_ids` (sample only these tasks), `Config.quarter_ids` (group 0: every stage of the shortest quarter before the rest's first samples), `Config.think1_band`, priorities `(group, stage, size)`; a first sample outside the quarter needs half its cap to start | Our measured yields: the shortest quarter earns most per token (first sample, then second sample), the rest little at a 49k cap (brief 4.3). With one group (the defaults) the order is the original stage-major one |
| `selection.py` (v7-vr2) | `key_vr`: the model's written grids before outputs that only an unverified program produced (`Group.written`); verified programs keep first place; the rest of the order unchanged | As built, an unverified program's output outranked the same reply's written grid on the demonstration-score tie-breakers, which only program candidates carry. CPU replay (`replay_v7vr2.py`, `v7vr2/REPLAY.md`): 13/25 as built, 16/25 fixed, none lost |

Tests: `tests/` pass unchanged (18 of 18, network off); v7-vr2 adds `tests/test_ranking_v7vr2.py` (6). Phase 2 checked locally against the hybrid's stub server
(first sample, rescue, repair, rescue of a repair, augmented second samples, the kill of a hung request, final merge
over B0's pool; tasks outside the route unchanged).
