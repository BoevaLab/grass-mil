from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.interpretability.contracts import NeighborhoodEnrichmentResult


def _coerce_edges(spatial_table: pd.DataFrame) -> pd.DataFrame:
    required = {"source_id", "target_id"}
    if not required.issubset(spatial_table.columns):
        raise ValueError("spatial_table must include ['source_id', 'target_id'].")
    frame = spatial_table.copy()
    if "weight" not in frame.columns:
        frame["weight"] = 1.0
    return frame


def run_neighborhood_enrichment(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    label_column: str,
    id_column: str = "instance_id",
    n_perms: int = 0,
    random_state: int = 42,
) -> NeighborhoodEnrichmentResult:
    if label_column not in node_table.columns:
        raise ValueError(f"Missing label column {label_column!r}.")
    nodes = node_table[[id_column, label_column]].copy()
    nodes[id_column] = nodes[id_column].astype(str)

    edges = _coerce_edges(spatial_table)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    edges = edges.merge(nodes, left_on="source_id", right_on=id_column, how="inner")
    edges = edges.merge(
        nodes, left_on="target_id", right_on=id_column, how="inner", suffixes=("_src", "_dst")
    )

    observed = (
        edges.groupby([f"{label_column}_src", f"{label_column}_dst"])["weight"]
        .sum()
        .unstack(fill_value=0.0)
    )
    categories = sorted(set(observed.index) | set(observed.columns))
    observed = observed.reindex(index=categories, columns=categories, fill_value=0.0)

    src_freq = (
        edges[f"{label_column}_src"]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )
    dst_freq = (
        edges[f"{label_column}_dst"]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )
    total_weight = float(edges["weight"].sum())
    expected = pd.DataFrame(
        np.outer(src_freq.values, dst_freq.values) * total_weight,
        index=categories,
        columns=categories,
    )
    enrichment = (observed - expected) / expected.replace(0.0, np.nan)
    enrichment = enrichment.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pvalues: Optional[pd.DataFrame] = None
    if int(n_perms) > 0:
        rng = np.random.default_rng(int(random_state))
        perm_scores = np.zeros((int(n_perms), len(categories), len(categories)), dtype=float)
        source = edges[f"{label_column}_src"].to_numpy()
        target = edges[f"{label_column}_dst"].to_numpy()
        w = edges["weight"].to_numpy()
        for i in range(int(n_perms)):
            shuffled = rng.permutation(target)
            tmp = (
                pd.DataFrame({"s": source, "t": shuffled, "w": w})
                .groupby(["s", "t"])["w"]
                .sum()
                .unstack(fill_value=0.0)
                .reindex(index=categories, columns=categories, fill_value=0.0)
            )
            perm_scores[i] = tmp.values
        obs = observed.values
        p = (perm_scores >= obs[None, :, :]).mean(axis=0)
        pvalues = pd.DataFrame(p, index=categories, columns=categories)

    return NeighborhoodEnrichmentResult(
        enrichment=enrichment,
        observed=observed,
        expected=expected,
        pvalues=pvalues,
    )


def run_diff_neighborhood_enrichment(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    label_column: str,
    condition_column: str,
    id_column: str = "instance_id",
    n_perms: int = 0,
    random_state: int = 42,
) -> dict[str, NeighborhoodEnrichmentResult]:
    if condition_column not in node_table.columns:
        raise ValueError(f"Missing condition column {condition_column!r}.")
    outputs: dict[str, NeighborhoodEnrichmentResult] = {}
    for cond, sub_nodes in node_table.groupby(condition_column):
        outputs[str(cond)] = run_neighborhood_enrichment(
            sub_nodes,
            spatial_table,
            label_column=label_column,
            id_column=id_column,
            n_perms=n_perms,
            random_state=random_state,
        )
    return outputs
