"""Cross-cutting contracts.

Invariants and payload shapes shared between the model, inference and
interpretability layers. Kept in one dependency-light module so any layer can
import it without creating a cycle -- ``grass_mil.models`` and
``grass_mil.inference`` already import each other through the supervised
module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["validate_attention_width"]


def validate_attention_width(*, attention_width: int, num_classes: int) -> None:
    """Require one attention channel per class.

    Bag pooling computes ``L_c = sum_i A[i,c] * l[i,c]``, and the attribution
    identity that the interpretability suite rests on is only exact when each
    class has its own attention distribution. A binary head therefore needs
    **two** attention channels, not one shared channel broadcast across both:
    a shared channel cannot express a niche that pushes toward one class and
    away from the other.
    """
    if attention_width != num_classes:
        raise ValueError(
            f"Attention emits {attention_width} channel(s) but the head emits "
            f"{num_classes} class logit(s). Attention is per-class: set "
            f"model.attention.n_classes to {num_classes}."
        )


@dataclass(frozen=True)
class EmbeddingSet:
    matrix: np.ndarray
    feature_columns: List[str]
    id_column: str = "instance_id"


@dataclass(frozen=True)
class InterpretabilityDataset:
    instance_table: pd.DataFrame
    bag_table: Optional[pd.DataFrame] = None
    spatial_table: Optional[pd.DataFrame] = None
    id_column: str = "instance_id"
    bag_id_column: str = "bag_id"
    cell_type_column: Optional[str] = None
    condition_column: Optional[str] = None


@dataclass(frozen=True)
class ReductionResult:
    method: str
    embedding: np.ndarray
    params: Dict[str, Any]
    fitted_object: Any = None


@dataclass(frozen=True)
class ClusteringResult:
    method: str
    labels: np.ndarray
    params: Dict[str, Any]
    fitted_object: Any = None


@dataclass(frozen=True)
class NicheSummary:
    niche_labels: np.ndarray
    composition: pd.DataFrame
    enrichment: pd.DataFrame
    niche_counts: pd.Series
    weighted_scores: Optional[pd.Series] = None
    mean_scores: Optional[pd.Series] = None
    attention_present: Optional[pd.Series] = None
    attention_lift_present: Optional[pd.Series] = None


@dataclass(frozen=True)
class NeighborhoodEnrichmentResult:
    enrichment: pd.DataFrame
    observed: pd.DataFrame
    expected: pd.DataFrame
    pvalues: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class FiltrationCurvesResult:
    thresholds: np.ndarray
    curves: Dict[str, Dict[str, np.ndarray]]


@dataclass(frozen=True)
class ReportSection:
    title: str
    description: str
    figure_html: Optional[str] = None
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginResult:
    name: str
    payload: Dict[str, Any]
    sections: List[ReportSection]


@dataclass(frozen=True)
class NicheTransferBundle:
    embedding_columns: List[str]
    id_column: str
    cluster_on_pca: bool
    pca_model: Any = None
    knn_model: Any = None
    label_values: np.ndarray = field(default_factory=lambda: np.asarray([]))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportBundle:
    dataset: InterpretabilityDataset
    reduction: Optional[ReductionResult]
    niche_feature_reduction: Optional[ReductionResult]
    clustering: Optional[ClusteringResult]
    niche_summary: Optional[NicheSummary]
    plugin_results: Dict[str, PluginResult]
    artifacts_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
