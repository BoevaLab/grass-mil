"""Cross-fold niche replication: per-fold clustering, matching into metaclusters,
whole-dataset transfer and a shared label-free basis."""

from .basis import run_shared_basis
from .match import (
    constrained_average_linkage,
    hellinger_matrix,
    match_metaclusters,
    scan_threshold,
)
from .records import cluster_fold_models
from .transfer import run_transfer

__all__ = [
    "cluster_fold_models",
    "constrained_average_linkage",
    "hellinger_matrix",
    "match_metaclusters",
    "run_shared_basis",
    "run_transfer",
    "scan_threshold",
]
