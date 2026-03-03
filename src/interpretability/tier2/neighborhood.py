from __future__ import annotations

import warnings
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from src.interpretability.contracts import NeighborhoodEnrichmentResult


def _coerce_edges(spatial_table: pd.DataFrame) -> pd.DataFrame:
    required = {"source_id", "target_id"}
    if not required.issubset(spatial_table.columns):
        raise ValueError("spatial_table must include ['source_id', 'target_id'].")
    return spatial_table.copy()


def _prepare_edges_with_labels(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    label_column: str,
    id_column: str,
    undirected: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    if id_column not in node_table.columns:
        raise ValueError(f"Missing id column {id_column!r}.")
    nodes = node_table[[id_column, label_column]].copy()
    nodes[id_column] = nodes[id_column].astype(str)
    if nodes[id_column].duplicated().any():
        raise ValueError(
            f"Node table contains duplicate IDs in {id_column!r}; "
            "neighborhood enrichment requires unique node identifiers."
        )
    label_map = nodes.set_index(id_column)[label_column].astype(str)

    edges = _coerce_edges(spatial_table)
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)

    if undirected:
        rev = edges.rename(columns={"source_id": "target_id", "target_id": "source_id"})
        edges = pd.concat([edges, rev], axis=0, ignore_index=True)

    edges = edges[
        edges["source_id"].isin(label_map.index) & edges["target_id"].isin(label_map.index)
    ]
    edges = edges.copy()
    edges["source_label"] = edges["source_id"].map(label_map)
    edges["target_label"] = edges["target_id"].map(label_map)
    return edges, label_map


def _observed_matrix(edges: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    observed = edges.groupby(["source_label", "target_label"]).size().unstack(fill_value=0.0)
    return observed.reindex(index=categories, columns=categories, fill_value=0.0)


def _analytical_expected_and_std(
    n_edges: int,
    node_labels: pd.Series,
    categories: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_nodes = int(node_labels.shape[0])
    if n_nodes == 0 or n_edges == 0:
        zeros = pd.DataFrame(0.0, index=categories, columns=categories)
        return zeros, zeros
    cluster_counts = node_labels.value_counts().reindex(categories, fill_value=0.0).astype(float)
    probs = cluster_counts.values / float(n_nodes)
    pair_probs = np.outer(probs, probs)
    expected = float(n_edges) * pair_probs
    var = float(n_edges) * pair_probs * (1.0 - pair_probs)
    std = np.sqrt(np.clip(var, a_min=0.0, a_max=None))
    return (
        pd.DataFrame(expected, index=categories, columns=categories),
        pd.DataFrame(std, index=categories, columns=categories),
    )


def _zscore_enrichment(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    std: pd.DataFrame,
) -> pd.DataFrame:
    z = (observed - expected) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _perm_counts(
    edge_stub: pd.DataFrame,
    node_ids: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    *,
    n_perms: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    n_labels = len(categories)
    out = np.zeros((int(n_perms), n_labels, n_labels), dtype=float)
    for i in range(int(n_perms)):
        shuffled_labels = rng.permutation(labels)
        shuffled_map = pd.Series(shuffled_labels, index=node_ids)
        perm_edges = edge_stub.copy()
        perm_edges["source_label"] = perm_edges["source_id"].map(shuffled_map)
        perm_edges["target_label"] = perm_edges["target_id"].map(shuffled_map)
        out[i] = _observed_matrix(perm_edges, categories).values
    return out


def _two_sided_empirical_pvalues(
    observed: pd.DataFrame,
    permuted_samples: np.ndarray,
) -> pd.DataFrame:
    if permuted_samples.ndim != 3:
        raise ValueError("Expected permutation samples with shape (n_perms, n_labels, n_labels).")
    if permuted_samples.shape[1:] != observed.shape:
        raise ValueError(
            "Permutation sample shape mismatch for empirical p-value computation: "
            f"{permuted_samples.shape[1:]} vs {observed.shape}."
        )
    center = permuted_samples.mean(axis=0)
    obs_dev = np.abs(observed.values - center)
    perm_dev = np.abs(permuted_samples - center[None, :, :])
    extreme = np.sum(perm_dev >= obs_dev[None, :, :], axis=0)
    p = (extreme + 1.0) / (permuted_samples.shape[0] + 1.0)
    return pd.DataFrame(np.clip(p, 0.0, 1.0), index=observed.index, columns=observed.columns)


def run_neighborhood_enrichment(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    label_column: str,
    id_column: str = "instance_id",
    n_perms: int = 0,
    random_state: int = 42,
    undirected: bool = False,
    categories: Optional[list[str]] = None,
    warn_analytical: bool = True,
) -> NeighborhoodEnrichmentResult:
    """Compute unweighted neighborhood enrichment with Squidpy-style z-scores.

    Behavior:
    - Edge `weight` is ignored. Observed values are raw edge counts by label pair.
    - If `undirected=True`, reverse edges are explicitly mirrored (`u->v` and `v->u`).
    - If `n_perms > 0`, expected/std are estimated from label permutations and `pvalues`
      are two-sided empirical p-values from the same permutation null.
    - If `n_perms == 0`, expected/std use an analytical approximation and a warning is emitted.
    """
    if label_column not in node_table.columns:
        raise ValueError(f"Missing label column {label_column!r}.")
    perms = int(n_perms)
    if perms < 0:
        raise ValueError("n_perms must be >= 0.")
    if perms == 0 and warn_analytical:
        warnings.warn(
            "run_neighborhood_enrichment is using analytical expected/std (n_perms=0). "
            "Set n_perms>0 for Squidpy-style permutation expected/std and empirical p-values.",
            UserWarning,
            stacklevel=2,
        )
    edges, node_labels = _prepare_edges_with_labels(
        node_table,
        spatial_table,
        label_column=label_column,
        id_column=id_column,
        undirected=undirected,
    )
    if categories is None:
        category_list = sorted(node_labels.astype(str).unique().tolist())
    else:
        category_list = [str(c) for c in categories]

    observed = _observed_matrix(edges, category_list)
    n_edges = int(edges.shape[0])
    expected, std = _analytical_expected_and_std(n_edges, node_labels, category_list)
    enrichment = _zscore_enrichment(observed, expected, std)

    pvalues: Optional[pd.DataFrame] = None
    if perms > 0:
        node_ids = node_labels.index.to_numpy()
        base_labels = node_labels.to_numpy()
        edge_stub = edges[["source_id", "target_id"]].copy()
        perm_counts = _perm_counts(
            edge_stub,
            node_ids,
            base_labels,
            category_list,
            n_perms=perms,
            random_state=int(random_state),
        )
        perm_mean = perm_counts.mean(axis=0)
        perm_std = perm_counts.std(axis=0, ddof=0)
        expected = pd.DataFrame(perm_mean, index=category_list, columns=category_list)
        std = pd.DataFrame(perm_std, index=category_list, columns=category_list)
        enrichment = _zscore_enrichment(observed, expected, std)
        pvalues = _two_sided_empirical_pvalues(observed, perm_counts)

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
    undirected: bool = False,
    warn_analytical: bool = True,
) -> dict[str, NeighborhoodEnrichmentResult]:
    """Compute pairwise differential neighborhood enrichment between conditions.

    Differential enrichment is `zscore(condition_a) - zscore(condition_b)`.
    Per-condition enrichments are computed in analytical mode for stability/speed; when
    `n_perms > 0`, differential p-values are estimated by permuting condition labels.
    """
    if condition_column not in node_table.columns:
        raise ValueError(f"Missing condition column {condition_column!r}.")
    perms = int(n_perms)
    if perms < 0:
        raise ValueError("n_perms must be >= 0.")
    if warn_analytical:
        warnings.warn(
            "run_diff_neighborhood_enrichment computes per-condition baselines with the "
            "analytical neighborhood model; n_perms controls only differential p-value permutations.",
            UserWarning,
            stacklevel=2,
        )
    category_list = sorted(node_table[label_column].astype(str).unique().tolist())
    per_condition: dict[str, NeighborhoodEnrichmentResult] = {}
    for cond, sub_nodes in node_table.groupby(condition_column):
        per_condition[str(cond)] = run_neighborhood_enrichment(
            sub_nodes,
            spatial_table,
            label_column=label_column,
            id_column=id_column,
            n_perms=0,
            random_state=random_state,
            undirected=undirected,
            categories=category_list,
            warn_analytical=False,
        )

    cond_keys = sorted(per_condition.keys())
    outputs: dict[str, NeighborhoodEnrichmentResult] = {}
    for left, right in combinations(cond_keys, 2):
        left_res = per_condition[left]
        right_res = per_condition[right]
        diff_enrichment = left_res.enrichment - right_res.enrichment
        diff_observed = left_res.observed - right_res.observed
        diff_expected = left_res.expected - right_res.expected

        pvalues: Optional[pd.DataFrame] = None
        if perms > 0:
            rng = np.random.default_rng(int(random_state))
            n_labels = len(category_list)
            perm_scores = np.zeros((perms, n_labels, n_labels), dtype=float)
            shuffled_conditions = node_table[condition_column].astype(str).to_numpy()
            for i in range(perms):
                perm_nodes = node_table.copy()
                perm_nodes[condition_column] = rng.permutation(shuffled_conditions)
                perm_left = perm_nodes[perm_nodes[condition_column].astype(str) == left]
                perm_right = perm_nodes[perm_nodes[condition_column].astype(str) == right]
                left_perm_res = run_neighborhood_enrichment(
                    perm_left,
                    spatial_table,
                    label_column=label_column,
                    id_column=id_column,
                    n_perms=0,
                    random_state=random_state,
                    undirected=undirected,
                    categories=category_list,
                    warn_analytical=False,
                )
                right_perm_res = run_neighborhood_enrichment(
                    perm_right,
                    spatial_table,
                    label_column=label_column,
                    id_column=id_column,
                    n_perms=0,
                    random_state=random_state,
                    undirected=undirected,
                    categories=category_list,
                    warn_analytical=False,
                )
                perm_scores[i] = (left_perm_res.enrichment - right_perm_res.enrichment).values
            obs = diff_enrichment.values
            perm_dev = np.abs(perm_scores)
            extreme = np.sum(perm_dev >= np.abs(obs)[None, :, :], axis=0)
            pvals = (extreme + 1.0) / (perm_scores.shape[0] + 1.0)
            pvalues = pd.DataFrame(
                np.clip(pvals, 0.0, 1.0),
                index=diff_enrichment.index,
                columns=diff_enrichment.columns,
            )

        outputs[f"{left}_{right}"] = NeighborhoodEnrichmentResult(
            enrichment=diff_enrichment,
            observed=diff_observed,
            expected=diff_expected,
            pvalues=pvalues,
        )
    return outputs
