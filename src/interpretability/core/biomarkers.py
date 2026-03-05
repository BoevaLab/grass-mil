from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.interpretability.contracts import ClusterSummary


def _resolve_variance_estimator(variance_estimator: str) -> int:
    key = str(variance_estimator).strip().lower()
    if key in {"unbiased", "sample", "n-1"}:
        return 1
    if key in {"biased", "population", "n"}:
        return 0
    raise ValueError("Unsupported variance_estimator. Expected one of: " "'unbiased' or 'biased'.")


def _infer_composition_matrix(
    table: pd.DataFrame,
    *,
    cell_type_column: Optional[str],
    composition_prefix: str,
) -> pd.DataFrame:
    comp_cols = [c for c in table.columns if c.startswith(composition_prefix)]
    if comp_cols:
        return table[comp_cols].astype(float).copy()

    raise ValueError(
        f"No composition columns found with prefix {composition_prefix!r}. "
        "Explicit composition features are required."
    )


def cluster_biomarker_summary(
    table: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    cell_type_column: Optional[str] = None,
    composition_prefix: str = "comp_",
    variance_estimator: str = "unbiased",
) -> ClusterSummary:
    if len(table) != int(cluster_labels.shape[0]):
        raise ValueError("table rows and cluster_labels length mismatch.")

    comp = _infer_composition_matrix(
        table, cell_type_column=cell_type_column, composition_prefix=composition_prefix
    )
    labels = pd.Series(cluster_labels, name="cluster_label")

    ddof = _resolve_variance_estimator(variance_estimator)
    comp_mean = comp.mean(axis=0)
    comp_std = comp.std(axis=0, ddof=ddof).replace(0.0, 1.0)
    z_comp = (comp - comp_mean) / comp_std

    composition = comp.groupby(labels).mean().sort_index()
    enrichment = z_comp.groupby(labels).mean().sort_index()
    counts = labels.value_counts().sort_index()

    return ClusterSummary(
        cluster_labels=np.asarray(cluster_labels),
        composition=composition,
        enrichment=enrichment,
        cluster_counts=counts,
    )


def cluster_attention_summary(
    table: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    attention_column: str = "attention",
    score_column: str = "score",
    bag_id_column: str = "bag_id",
    cell_type_column: Optional[str] = None,
    composition_prefix: str = "comp_",
    variance_estimator: str = "unbiased",
    eps: float = 1e-8,
) -> ClusterSummary:
    summary = cluster_biomarker_summary(
        table,
        cluster_labels,
        cell_type_column=cell_type_column,
        composition_prefix=composition_prefix,
        variance_estimator=variance_estimator,
    )
    frame = table.copy()
    frame["cluster_label"] = cluster_labels
    for col in (attention_column, score_column, bag_id_column):
        if col not in frame.columns:
            raise ValueError(f"Missing required column {col!r} for attention summary.")
    frame[attention_column] = frame[attention_column].astype(float)
    frame[score_column] = frame[score_column].astype(float)

    weighted_frame = frame.assign(weighted=frame[attention_column] * frame[score_column])
    weighted = (
        weighted_frame.groupby("cluster_label")
        .apply(lambda g: float(g["weighted"].sum() / max(g[attention_column].sum(), eps)))
        .sort_index()
    )
    mean_scores = frame.groupby("cluster_label")[score_column].mean().sort_index()

    cluster_attn_in_bag = (
        frame.groupby([bag_id_column, "cluster_label"])[attention_column]
        .sum()
        .unstack(fill_value=0.0)
    )
    cluster_count_in_bag = (
        frame.groupby([bag_id_column, "cluster_label"]).size().unstack(fill_value=0)
    )
    present_mask = cluster_count_in_bag > 0

    attn_present = cluster_attn_in_bag.where(present_mask).mean(axis=0, skipna=True).sort_index()

    bag_totals = cluster_count_in_bag.sum(axis=1).replace(0, 1)
    abundance = cluster_count_in_bag.div(bag_totals, axis=0)
    lift = cluster_attn_in_bag / abundance.clip(lower=eps)
    lift_present = lift.where(present_mask).mean(axis=0, skipna=True).sort_index()

    return ClusterSummary(
        cluster_labels=summary.cluster_labels,
        composition=summary.composition,
        enrichment=summary.enrichment,
        cluster_counts=summary.cluster_counts,
        weighted_scores=weighted,
        mean_scores=mean_scores,
        attention_present=attn_present,
        attention_lift_present=lift_present,
    )


def cluster_survival_attention_summary(
    table: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    attention_column: str = "attention",
    hazard_column: str = "hazard",
    bag_id_column: str = "bag_id",
    cell_type_column: Optional[str] = None,
    composition_prefix: str = "comp_",
    variance_estimator: str = "unbiased",
    eps: float = 1e-8,
) -> ClusterSummary:
    renamed = table.copy()
    if hazard_column not in renamed.columns:
        raise ValueError(f"Missing hazard column {hazard_column!r}.")
    renamed["score"] = renamed[hazard_column]
    return cluster_attention_summary(
        renamed,
        cluster_labels,
        attention_column=attention_column,
        score_column="score",
        bag_id_column=bag_id_column,
        cell_type_column=cell_type_column,
        composition_prefix=composition_prefix,
        variance_estimator=variance_estimator,
        eps=eps,
    )
