"""Two-answer selection (A2-10). Selectors implement the frozen ``Selector`` protocol: ``select(candidates,
test_index) -> (Grid, Grid)`` and never read labels. Calibration (``fit_weights``) is a development-time step that
reads labels on dev/tune only and writes a versioned JSON the selector loads at inference."""
from arc2.selection.calibration import (
    Calibration,
    fit_weights,
    load_calibration,
    write_calibration,
)
from arc2.selection.diverse import DiverseSelector
from arc2.selection.metrics import aggregate, output_metrics, selection_gap
from arc2.selection.rank import RankSelector, rank_candidates
from arc2.selection.structure import cell_disagreement, structurally_different

__all__ = [
    "Calibration",
    "DiverseSelector",
    "RankSelector",
    "aggregate",
    "cell_disagreement",
    "fit_weights",
    "load_calibration",
    "output_metrics",
    "rank_candidates",
    "selection_gap",
    "structurally_different",
    "write_calibration",
]
