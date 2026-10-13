"""Evaluation statistics for held-out scores (AUROC CIs, DeLong, operating points)."""

from .operating_points import (
    confusion_counts,
    flag_top_within_fold,
    likelihood_ratios,
    metrics_from_confusion,
    operating_point,
    operating_point_bootstrap,
    ppv_npv_at_prevalence,
    threshold_points,
)
from .stats import (
    auroc,
    bootstrap_auroc_ci,
    delong_test,
    per_fold_auroc,
    rank_within_fold,
    stratified_auroc,
    z_within_fold,
)

__all__ = [
    "auroc",
    "bootstrap_auroc_ci",
    "confusion_counts",
    "delong_test",
    "flag_top_within_fold",
    "likelihood_ratios",
    "metrics_from_confusion",
    "operating_point",
    "operating_point_bootstrap",
    "per_fold_auroc",
    "ppv_npv_at_prevalence",
    "rank_within_fold",
    "stratified_auroc",
    "threshold_points",
    "z_within_fold",
]
