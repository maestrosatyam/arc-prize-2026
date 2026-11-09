"""arc_harness: a verify-and-refine harness around an open-weights reasoning model for ARC-AGI-2.

Layout
  data.py        task loading (Kaggle json, GitHub dirs, ARC-AGI-1, ConceptARC), holdout split
  grids.py       grid text formatting and grid parsing from model output
  sandbox.py     run the model's transform() in a subprocess; grade on demonstrations (Poetiq feedback)
  validator.py   cheap hard checks on a candidate output (shape, palette) from the demonstrations
  prompts.py     prompts for direct, direct+program, repair and Poetiq
  llm.py         LLM backends: vLLM (OpenAI-compatible), scripted, oracle simulator
  budget.py      token/time budget accounting and per-stage caps
  solvers.py     direct, poetiq, vr (verify-and-refine) solvers; all emit Sample records
  selection.py   rank candidates into attempt_1 / attempt_2, merge with an external pool
  submission.py  always-valid submission.json writer
  score.py       local scorer
  loss.py        attribute each missed output to a loss category
  run.py         CLI entry point
  ablate.py      replay-based component ablation
"""

__version__ = "0.1.0"
