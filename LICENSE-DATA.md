# Data licence

The data in this repository that the team made is licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/legalcode

This covers:
- ARC-AGI-2: the bank's candidate pools and 27B samples (`experiments/runs/kaggle-A2-bank-part1-v1/`,
  `kaggle-A2-bank-part2-v1/`, `kaggle-A2-q27-bank-v1/q27d/`), the decision rules as fixed, and the result files under
  `experiments/arc2-bank/`, `experiments/arc2-pool-audit/` and `experiments/runs/`;
- ARC-AGI-3: the game rules and their engine checks, the 35 case bundles, the 22-level test set, the level starts,
  the decision rules as fixed and the result files under `experiments/`;
- the figures (`reports/paper/figures/`) and the two papers (`papers/`).

Attribution: Satyam Dwivedi and Null Hypothesis Enthusiasts, ARC Prize 2026 (https://github.com/maestrosatyam/arc-prize-2026).

Third-party material keeps its own licence:
- the ARC-AGI-2 tasks, including the reformatted copy in `eval/data/` and the bank's ground truth, are Apache-2.0
  (ARC Prize Foundation; see `eval/data/README.md` and `licenses/APACHE-2.0.txt`);
- the ARC-AGI-3 engine and scorer packages are MIT. The game files are competition data and are not included here.
  The rules cite the game source by line number.

The hint texts inside the HLS entry notebook (`kaggle/arc3/hl/entry/hls/`) are CC0 1.0, as published in the
Kaggle dataset maestrosatyam/arc-agi-3-hint-texts.

Code is licensed separately under MIT-0 (`LICENSE`). Third-party code and models are listed in `licenses/NOTICES.md`.
