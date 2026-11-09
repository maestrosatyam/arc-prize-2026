# ARC Prize 2026: Null Hypothesis Enthusiasts

Null Hypothesis Enthusiasts is an independent research collective for unusually curious people to pursue important questions with people they might never otherwise meet.

Code, data, entry notebooks and papers for the team's ARC Prize 2026 entries on ARC-AGI-2 and ARC-AGI-3. The
papers' analyses rerun from this repository on CPU. The entries and the ARC-AGI-3 arms also need model weights,
third-party wheels and game files that are not redistributed here (see "Not included"); the weights and wheels are
public on Kaggle at the versions pinned below.

## Papers

| Track | Paper | PDF | Kaggle writeup |
|---|---|---|---|
| ARC-AGI-2 | Missing or Misranked? Two Ways a Sampling Solver Loses Right Answers | [papers/missing-or-misranked.pdf](papers/missing-or-misranked.pdf) | ARC-AGI-2 Solution Writeup |
| ARC-AGI-3 | The Idea Gap: Why Open LLM Agents Fail New Environments | [papers/the-idea-gap.pdf](papers/the-idea-gap.pdf) | ARC Prize 2026 Paper Track |

## Entries (final selections)

| Track | Entry | Kaggle notebook | Submission | Public score | In this repository |
|---|---|---|---|---|---|
| ARC-AGI-2 | v7-vr | [purushottamanurag/arc-agi-2-hybrid-v7vr](https://www.kaggle.com/code/purushottamanurag/arc-agi-2-hybrid-v7vr) | 57646090 | 34.31 (identical resubmission 57785824: 29.31) | `kaggle/arc2/hybrid_vr/v7vr/` |
| ARC-AGI-2 | zbeam | [purushottamanurag/arc-agi-2-hybrid-v7vr2-zbeams](https://www.kaggle.com/code/purushottamanurag/arc-agi-2-hybrid-v7vr2-zbeams) | 57853811 | 32.64 | `kaggle/arc2/hybrid_vr/v7vr2x/zbeams/` |
| ARC-AGI-3 | FP8 | [drdiwakarmishra/arc-agi-3-p0-ka2-fp8ple](https://www.kaggle.com/code/drdiwakarmishra/arc-agi-3-p0-ka2-fp8ple) | 57785925 | 30.26 (same version, 57716633: 27.54) | `kaggle/arc3/a3l1/entry/` |
| ARC-AGI-3 | HLS | [drdiwakarmishra/arc-agi-3-fp8-hls](https://www.kaggle.com/code/drdiwakarmishra/arc-agi-3-fp8-hls) | 57887330 | 29.13 | `kaggle/arc3/hl/entry/hls/` |

Each folder holds the submitted notebook and its `kernel-metadata.json` (inputs, hardware, container image). Each
notebook has the same cells as version 1 on Kaggle, the version that was scored. The papers list every other
submission; `submissions/ledger.json` lists every submission with its public score.

| Track | Notebook file | sha256 (first 8) | Cells |
|---|---|---|---|
| ARC-AGI-2 | `arc-agi-2-hybrid-v7vr.ipynb` | b705ef40 | 21 |
| ARC-AGI-2 | `arc-agi-2-hybrid-v7vr2-zbeams.ipynb` | 8450420b | 22 |
| ARC-AGI-3 | `arc-agi-3-p0-ka2-fp8ple.ipynb` | beebc0b8 | 25 |
| ARC-AGI-3 | `arc-agi-3-fp8-hls.ipynb` | 4f758454 | 27 |

### Running an entry

**On Kaggle.** Open the public notebook and choose Copy and Edit, or push the folder with the Kaggle CLI
(`kaggle kernels push -p <folder>` after changing `id` in `kernel-metadata.json` to your own account). The metadata
attaches the inputs below. Hardware: four NVIDIA L4 GPUs for ARC-AGI-2 (12 hours) and one NVIDIA RTX PRO 6000 for
ARC-AGI-3. Submitted to the competition, the notebook reruns the entry on the hidden tasks. Run interactively, the
ARC-AGI-2 notebooks take a short commit-run path on 5 public evaluation tasks.

**Elsewhere.** The notebooks read their inputs from `/kaggle/input/<slug>/` and the competition data from
`/kaggle/input/<competition>/`. Download each input below at the version given (the Kaggle CLI needs a free account)
and place it at that path. Each model page names its upstream source and licence.

| Track | Input | Kind | Version | Used for |
|---|---|---|---|---|
| ARC-AGI-2 | [sorokin/qwen3_4b_grids15_sft139](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139) | model, Transformers/bfloat16 | 1 | B0's Qwen3-4B grid model |
| ARC-AGI-2 | [mikedan7/qwen3-8-27b-fp8-official](https://www.kaggle.com/models/mikedan7/qwen3-8-27b-fp8-official) | model, PyTorch/hf-fp8 | 1 | the 27B (Qwen3.8-27B-FP8) |
| ARC-AGI-2 | [sorokin/pip-install-unsloth-flash-patch](https://www.kaggle.com/code/sorokin/pip-install-unsloth-flash-patch) | notebook | 2 | B0's install step |
| ARC-AGI-2 | [mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3](https://www.kaggle.com/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3) | dataset | 1 | vLLM 0.19 wheels |
| ARC-AGI-2 | [keithtyser/taaf-duck-qwen38-serving-v1](https://www.kaggle.com/datasets/keithtyser/taaf-duck-qwen38-serving-v1) | dataset | 2 | serving overlay |
| ARC-AGI-2 | [purushottamanurag/cpython-3-12-standalone-bin](https://www.kaggle.com/datasets/purushottamanurag/cpython-3-12-standalone-bin) | dataset | 1 | CPython 3.12.15 (python-build-standalone 20261003, install_only, x86_64 Linux) |
| ARC-AGI-3 | [dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple](https://www.kaggle.com/models/dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple) | model, Transformers/default | 1 | the agent's model (FP8 build) |
| ARC-AGI-3 | [dfranzen/albucino-qwen3-8-flash-next-drafter](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter) | model, Transformers/default | 1 | draft head for speculative decoding |
| ARC-AGI-3 | [dfranzen/pennyroyal-v253](https://www.kaggle.com/datasets/dfranzen/pennyroyal-v253) | dataset | 1 | serving wheelhouse |
| ARC-AGI-3 | [dfranzen/taaf-kaggle-source-bundle-copy](https://www.kaggle.com/datasets/dfranzen/taaf-kaggle-source-bundle-copy) | dataset | 1 | the agent harness |

The ARC-AGI-2 base method is the public notebook
[mikelou1/arc-agi2-lb33-89-minimal-perfpatch](https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch)
(the NVARC 2025 solution). The ARC-AGI-3 agent is D. Franzen's public Milestone-2 notebook, derived from J. Cottaar's
Duck notebook and the Tufa Labs Duck harness, run unmodified apart from a start-up keep-alive and the FP8 model build.
The HLS entry adds a hint library: up to 15 cross-game patterns from a pool of 365 hint texts, which is public as
the dataset
[maestrosatyam/arc-agi-3-hint-texts](https://www.kaggle.com/datasets/maestrosatyam/arc-agi-3-hint-texts) (CC0).

### Entry code

The submitted notebooks are complete and self-contained. For ARC-AGI-2, `kaggle/arc2/hybrid_vr/` also holds the
hybrid's code as plain files: `arc_harness/` is the verify-and-refine harness, `hybrid_vr.py` the phase-2 code,
`PATCHES.md` lists every change to the base notebook, and `NOTICES.md` the harness's MIT-licensed parts.

## Analysis code and data

All paths are relative to the repository root. The analysis runs on CPU with Python 3.12:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### Repository map

| Track | Path | What it holds |
|---|---|---|
| ARC-AGI-2 | `eval/` | the bank's loader, rules and scorer (`eval/arc2_bank.py`); ARC-AGI-2 task data in `eval/data/`, reformatted by `scripts/arc2_build_eval_data.py` |
| ARC-AGI-2 | `experiments/runs/kaggle-A2-bank-part1-v1/`, `kaggle-A2-bank-part2-v1/`, `kaggle-A2-q27-bank-v1/q27d/` | the bank: B0's candidate pools (`pool/`, and `pool_sealed/` for the held-out tasks), the 27B samples with transcripts, and the ground truth (`_kaggle_solutions_audit_tasks.json`) |
| ARC-AGI-2 | `experiments/arc2-bank/` | readings (`*_result.json`, `profile.json`, `bootstrap.json`, `null_calibration.json`), the replay ledger (`replay_ledger.jsonl`), the side-by-side and repeat readings, and the decode-many and temperature pools (`dm/`, `ts/`) |
| ARC-AGI-2 | `experiments/arc2-pool-audit/` and `experiments/runs/kaggle-A2-{pool-audit,H1,H2,W1,adapt-dev}-*` | the public-pool audit and its inputs |
| ARC-AGI-2 | `experiments/arc2-q27/`, `experiments/runs/kaggle-A2-DC1-v1/`, `kaggle-A2-GS1-v1/` | results of the 27B verify-and-refine bank, the rescue replay, the decider and the 30B scorer |
| ARC-AGI-2 | `experiments/arc2-mamba/` | the recurrent study (Appendix B of the paper): code, results and statistics |
| ARC-AGI-2 | `scripts/arc2_*.py`, `arc2/`, `splits/` | replay, null, bootstrap, profile, calibration, account, figure and reading scripts, the package they import, and the development split |
| ARC-AGI-2 | `DECISION_RULES_ARC2.md` | every decision rule fixed before its run, and the predictions committed before scoring |
| ARC-AGI-3 | `experiments/arc3-rules/` | rules for the 25 public games, written from the game source, with their engine checks (`rules.json`, one file per game) |
| ARC-AGI-3 | `experiments/arc3-recovery/` | the 35 case bundles (`cases/`), their replay check, and each continuation run's analysis |
| ARC-AGI-3 | `kaggle/arc3/r1/r1-changes.patch`, `kaggle/arc3/c1/base_r1g_v2/` | the restore patch for the harness, and the arm notebook that includes the told-idea arm |
| ARC-AGI-3 | `experiments/arc3-failure-annotation/` | the failure labels (yes, partial, no) and the 22-level test set |
| ARC-AGI-3 | `experiments/arc3-c1/`, `arc3-c1p/`, `arc3-c1s/` | the 157 level starts (`levels.jsonl`), the placebo texts and the self-written ideas |
| ARC-AGI-3 | `experiments/runs/kaggle-R1*`, `kaggle-WM2-*`, `kaggle-A3-*` | per-run summaries (`r1_summary.txt`), decisions (`*_decision.json`), per-case results (`r1_results.jsonl`) and public-game readings (`benchmark.json`) |
| ARC-AGI-3 | `DECISION_RULES_ARC3.md` | every decision rule fixed before its run, with its outcome |
| ARC-AGI-3 | `experiments/arc3-human-data/`, `experiments/arc3-think-aloud/` | derived human counts (replay checks, recognition counts) |

`CLAIMS_ARC2.md` maps each number in the ARC-AGI-2 paper to its file here. Working notes, build scripts, kernel
plans and run logs are not part of this release.

### Rerunning

ARC-AGI-2 bank readings, from the repository root:

```bash
python3 -I scripts/arc2_bank_profile.py                 # the account: where right outputs are lost
python3 -I scripts/arc2_bank_replay.py --rule v7_direct  # one rule replayed on the bank (appends to the replay ledger)
python3 -I scripts/arc2_bank_null.py --seeds 200         # null replays
python3 -I scripts/arc2_bank_bootstrap.py --n-boot 4000  # task-sampling bootstrap
python3 -I scripts/arc2_paper_figures.py                 # the paper's figures
```

ARC-AGI-3 readings:

```bash
python3 -I scripts/arc3_recovery_analyze.py experiments/runs/kaggle-R1b-v1/r1_results.jsonl   # one or more runs
python3 -I scripts/arc3_c1_analyze.py --archive experiments/runs/kaggle-A3-C1-levelstarts-v1
python3 -I scripts/analysis_arc3_kaggle.py                                                     # public-game runs
```

Replaying the 35 cases and checking the rules on the engine needs the ARC-AGI-3 public game files, which are
competition data and are not included here. Pass their folder with `--env-dir`, for example
`python3 -I scripts/arc3_recovery_prepare.py replay-check --env-dir <game files>`. Running an arm needs the
third-party harness (the Kaggle dataset `dfranzen/taaf-kaggle-source-bundle-copy`) with `kaggle/arc3/r1/r1-changes.patch`
applied, and a GPU. The arms ran on Kaggle.

Some readers check their inputs against rule documents, run logs or commits of the development repository that are
not included. For those, the committed outputs next to them are the record.


## Licences

- **Code we wrote: MIT-0** (`LICENSE`). This includes our code in the four Kaggle notebooks, which Kaggle labels
  Apache 2.0 by default. Both licences apply, and MIT-0 asks nothing in return.
- **Data we made: CC BY 4.0** (`LICENSE-DATA.md`): the rules, case bundles, test sets, level starts, bank pools and
  samples, decision rules, result files, figures and papers.
- **Third-party parts** keep their own licences: `licenses/NOTICES.md`, `licenses/APACHE-2.0.txt` and
  `kaggle/arc2/hybrid_vr/NOTICES.md`.
  ARC-AGI-2 tasks are Apache-2.0, the ARC-AGI-3 engine is MIT, Qwen3.8-27B is Apache-2.0, and the Qwen3.8-Flash-Next
  family is under the Qwen Community License 1.0.

## Not included

Model weights; third-party wheels; the ARC-AGI-3 game files and frame imagery; the source of the third-party agent
harness (the restore patch here quotes it only as diff context, and the public notebooks this work builds on embed it under
Apache-2.0); raw third-party human-play and think-aloud datasets (derived counts only); and the team's own play
records, apart from the published hint texts. The 157 level starts in `experiments/arc3-c1/levels.jsonl` are action
prefixes from the team's own play (each game played individually, by hand), checked on the engine.

## Citation

```bibtex
@misc{dwivedi2026missing,
  title  = {Missing or Misranked? Two Ways a Sampling Solver Loses Right Answers},
  author = {Dwivedi, Satyam and Kumari, Nishi and Pandey, Avinash and Purushottam, Anurag and Verma, Rakhi},
  year   = {2026},
  note   = {ARC Prize 2026, ARC-AGI-2 Solution Writeup. Vaikhari AI; Null Hypothesis Enthusiasts},
  url    = {https://github.com/maestrosatyam/arc-prize-2026}
}
@misc{dwivedi2026ideagap,
  title  = {The Idea Gap: Why Open LLM Agents Fail New Environments},
  author = {Dwivedi, Satyam and Kumari, Nishi and Mishra, Diwakar and Pandey, Avinash and Purushottam, Anurag and Rajput, Pawan and Tripathi, Piyush and Verma, Rakhi},
  year   = {2026},
  note   = {ARC Prize 2026 Paper Track. Vaikhari AI; Null Hypothesis Enthusiasts},
  url    = {https://github.com/maestrosatyam/arc-prize-2026}
}
```

## Version

The tag `v1.0` is the repository as submitted with the Kaggle writeups on 9 November 2026.
