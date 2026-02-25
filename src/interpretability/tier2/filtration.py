from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.interpretability.contracts import FiltrationCurvesResult


def compute_filtration_curves(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    thresholds: np.ndarray,
    cluster_column: str = "cluster_label",
    cell_type_column: str = "cell_type",
    id_column: str = "instance_id",
    distance_column: str = "distance",
    scale_within_cluster: bool = True,
) -> FiltrationCurvesResult:
    if cluster_column not in node_table.columns:
        raise ValueError(f"Missing cluster column {cluster_column!r}.")
    if cell_type_column not in node_table.columns:
        raise ValueError(f"Missing cell_type column {cell_type_column!r}.")
    if distance_column not in spatial_table.columns:
        raise ValueError(f"Missing distance column {distance_column!r} in spatial table.")
    if {"source_id", "target_id"} - set(spatial_table.columns):
        raise ValueError("spatial_table must include source_id/target_id columns.")

    nodes = node_table[[id_column, cluster_column, cell_type_column]].copy()
    nodes[id_column] = nodes[id_column].astype(str)
    edges = spatial_table.copy()
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    edges = edges.merge(nodes, left_on="source_id", right_on=id_column, how="inner")

    clusters = sorted(nodes[cluster_column].astype(str).unique())
    cell_types = sorted(nodes[cell_type_column].astype(str).unique())
    thresholds = np.asarray(thresholds, dtype=float)
    curves: dict[str, dict[str, np.ndarray]] = {
        c: {ct: np.zeros_like(thresholds, dtype=float) for ct in cell_types} for c in clusters
    }

    for cluster in clusters:
        c_edges = edges[edges[cluster_column].astype(str) == str(cluster)]
        for i, thr in enumerate(thresholds):
            selected = c_edges[c_edges[distance_column] <= float(thr)]
            counts = selected[cell_type_column].astype(str).value_counts()
            for ct in cell_types:
                curves[cluster][ct][i] = float(counts.get(ct, 0.0))
        if scale_within_cluster:
            for ct in cell_types:
                vmax = float(curves[cluster][ct].max())
                if vmax > 0:
                    curves[cluster][ct] = curves[cluster][ct] / vmax

    return FiltrationCurvesResult(thresholds=thresholds, curves=curves)


def _serialize_filtration_curves(result: FiltrationCurvesResult) -> pd.DataFrame:
    rows = []
    for cluster, ct_map in result.curves.items():
        for cell_type, values in ct_map.items():
            for idx, val in enumerate(values):
                rows.append(
                    {
                        "cluster_label": cluster,
                        "cell_type": cell_type,
                        "threshold": float(result.thresholds[idx]),
                        "value": float(val),
                    }
                )
    return pd.DataFrame(rows)
