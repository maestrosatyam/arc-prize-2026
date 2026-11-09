"""Decoding-side candidate ingestion (A2-04): unified candidate pool over the symbolic branch and the public
baseline's neural outputs. Must not import from eval/ or splits/ label files."""
from arc2.decoding.pool import (
    NeuralCandidate,
    PoolStats,
    base_key,
    candidates_to_pool,
    group_samples,
    load_nvarc_outputs,
    load_symbolic_json,
    merge_pools,
    neural_to_candidate,
    pool_from_decoded_results,
    split_base_key,
    write_symbolic_json,
)

__all__ = [
    "NeuralCandidate",
    "PoolStats",
    "base_key",
    "candidates_to_pool",
    "group_samples",
    "load_nvarc_outputs",
    "load_symbolic_json",
    "merge_pools",
    "neural_to_candidate",
    "pool_from_decoded_results",
    "split_base_key",
    "write_symbolic_json",
]
