from __future__ import annotations

import numpy as np
import pandas as pd

from grass_mil.interpretability.contracts import FiltrationCurvesResult


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
    if id_column not in node_table.columns:
        raise ValueError(f"Missing id column {id_column!r}.")
    if distance_column not in spatial_table.columns:
        raise ValueError(f"Missing distance column {distance_column!r} in spatial table.")
    if {"source_id", "target_id"} - set(spatial_table.columns):
        raise ValueError("spatial_table must include source_id/target_id columns.")

    nodes = node_table[[id_column, cluster_column, cell_type_column]].copy()
    nodes[id_column] = nodes[id_column].astype(str)
    if nodes[id_column].duplicated().any():
        raise ValueError(
            f"Node table contains duplicate IDs in {id_column!r}; "
            "filtration requires unique node identifiers."
        )
    nodes[cluster_column] = nodes[cluster_column].astype(str)
    nodes[cell_type_column] = nodes[cell_type_column].astype(str)
    node_to_cell_type = nodes.set_index(id_column)[cell_type_column].to_dict()
    node_to_cluster = nodes.set_index(id_column)[cluster_column].to_dict()

    edges = spatial_table.copy()
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    edges = edges[edges["source_id"].isin(node_to_cluster.keys())].copy()
    edges[distance_column] = edges[distance_column].astype(float)
    edges["source_cluster"] = edges["source_id"].map(node_to_cluster)

    clusters = sorted(nodes[cluster_column].astype(str).unique())
    cell_types = sorted(nodes[cell_type_column].astype(str).unique())
    cell_type_to_idx = {name: idx for idx, name in enumerate(cell_types)}
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1:
        raise ValueError("thresholds must be a 1D array.")
    if thresholds.size == 0:
        raise ValueError("thresholds must include at least one value.")
    sort_order = np.argsort(thresholds)
    thresholds_sorted = thresholds[sort_order]
    curves: dict[str, dict[str, np.ndarray]] = {
        c: {ct: np.zeros_like(thresholds, dtype=float) for ct in cell_types} for c in clusters
    }

    for cluster in clusters:
        c_edges = edges[edges["source_cluster"] == str(cluster)]
        per_ct_diff = np.zeros((len(cell_types), thresholds_sorted.shape[0] + 1), dtype=float)
        for _, subgraph_edges in c_edges.groupby("source_id", sort=False):
            source_ids = subgraph_edges["source_id"].to_numpy(dtype=str)
            target_ids = subgraph_edges["target_id"].to_numpy(dtype=str)
            distances = subgraph_edges[distance_column].to_numpy(dtype=float)
            if distances.size == 0:
                continue
            edge_order = np.argsort(distances)
            source_ids = source_ids[edge_order]
            target_ids = target_ids[edge_order]
            distances = distances[edge_order]

            first_threshold_by_node: dict[str, int] = {}
            for src_id, tgt_id, distance in zip(source_ids, target_ids, distances):
                start_idx = int(np.searchsorted(thresholds_sorted, distance, side="left"))
                if start_idx >= int(thresholds_sorted.shape[0]):
                    continue
                prev_src = first_threshold_by_node.get(src_id)
                if prev_src is None or start_idx < prev_src:
                    first_threshold_by_node[src_id] = start_idx
                prev_tgt = first_threshold_by_node.get(tgt_id)
                if prev_tgt is None or start_idx < prev_tgt:
                    first_threshold_by_node[tgt_id] = start_idx

            for node_id, start_idx in first_threshold_by_node.items():
                cell_type = node_to_cell_type.get(node_id)
                if cell_type is None:
                    continue
                ct_idx = cell_type_to_idx.get(cell_type)
                if ct_idx is None:
                    continue
                per_ct_diff[ct_idx, start_idx] += 1.0

        per_ct_sorted = np.cumsum(per_ct_diff[:, :-1], axis=1)
        per_ct_original = np.zeros_like(per_ct_sorted)
        per_ct_original[:, sort_order] = per_ct_sorted
        for ct_idx, ct in enumerate(cell_types):
            curves[cluster][ct] = per_ct_original[ct_idx].astype(float, copy=False)
        if scale_within_cluster:
            for ct in cell_types:
                vmax = float(curves[cluster][ct].max())
                if vmax > 0:
                    curves[cluster][ct] = curves[cluster][ct] / vmax

    return FiltrationCurvesResult(thresholds=thresholds, curves=curves)
