from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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
class ClusterSummary:
    cluster_labels: np.ndarray
    composition: pd.DataFrame
    enrichment: pd.DataFrame
    cluster_counts: pd.Series
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
class ReportBundle:
    dataset: InterpretabilityDataset
    reduction: Optional[ReductionResult]
    clustering: Optional[ClusteringResult]
    cluster_summary: Optional[ClusterSummary]
    plugin_results: Dict[str, PluginResult]
    artifacts_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
