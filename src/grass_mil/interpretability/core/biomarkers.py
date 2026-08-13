from __future__ import annotations

import numpy as np
import pandas as pd

from grass_mil.contracts import NicheSummary


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
    composition_prefix: str,
) -> pd.DataFrame:
    comp_cols = [c for c in table.columns if c.startswith(composition_prefix)]
    if comp_cols:
        return table[comp_cols].astype(float).copy()

    raise ValueError(
        f"No composition columns found with prefix {composition_prefix!r}. "
        "Explicit composition features are required."
    )


def niche_composition_summary(
    table: pd.DataFrame,
    niche_labels: np.ndarray,
    *,
    composition_prefix: str = "comp_",
    variance_estimator: str = "unbiased",
) -> NicheSummary:
    """Mean composition and z-scored enrichment per niche.

    Rows come out in niche-id order, which is dendrogram order: the pipeline
    renumbers niches by composition once, right after clustering, so nothing
    downstream has to re-derive or thread an ordering.
    """
    if len(table) != int(niche_labels.shape[0]):
        raise ValueError("table rows and niche_labels length mismatch.")

    comp = _infer_composition_matrix(table, composition_prefix=composition_prefix)
    labels = pd.Series(niche_labels, name="niche_label")

    ddof = _resolve_variance_estimator(variance_estimator)
    comp_mean = comp.mean(axis=0)
    comp_std = comp.std(axis=0, ddof=ddof).replace(0.0, 1.0)
    z_comp = (comp - comp_mean) / comp_std

    composition = comp.groupby(labels).mean().sort_index()
    enrichment = z_comp.groupby(labels).mean().sort_index()
    counts = labels.value_counts().sort_index()

    return NicheSummary(
        niche_labels=np.asarray(niche_labels),
        composition=composition,
        enrichment=enrichment,
        niche_counts=counts,
    )
