# ARC-AGI-2 task data (reformatted)

These four files are the ARC-AGI-2 training and public evaluation tasks from
https://github.com/arcprize/ARC-AGI-2 at commit f3283f727488ad98fe575ea6a5ac981e4a188e49, Copyright 2019 ARC Prize
Foundation, licensed under the Apache License 2.0 (`licenses/APACHE-2.0.txt`).

Change: `scripts/arc2_build_eval_data.py` merged the one-file-per-task JSON files into the Kaggle competition's file
layout (one challenges file and one solutions file per split). The grids are unchanged. `MANIFEST.json` records the
source commit and each file's sha256. To rebuild them, clone that commit to `third_party/ARC-AGI-2` and run
`python3 -I scripts/arc2_build_eval_data.py --repo third_party/ARC-AGI-2 --out eval/data`.

The bank's ground truth (`experiments/runs/kaggle-A2-bank-part*-v1/pool*/_kaggle_solutions_audit_tasks.json` and
`experiments/runs/kaggle-A2-q27-bank-v1/q27d/_kaggle_solutions_audit_tasks.json`) comes from the competition's copy of
the same public evaluation solutions, which has extra test outputs on five tasks. It is under the same licence.
