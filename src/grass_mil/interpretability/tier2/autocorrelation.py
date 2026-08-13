"""Global spatial autocorrelation (Moran's I).

Moran's I over a binary connectivity matrix :math:`W`:

.. math:: I = \\frac{N}{S_0} \\cdot \\frac{z^\\top W z}{z^\\top z},
          \\qquad S_0 = \\sum_{u,v} w_{uv}

with :math:`z` the mean-centred feature. Significance comes from a permutation
null: the feature is shuffled over nodes while the graph is held fixed.

Implemented natively over ``scipy.sparse`` rather than through squidpy. The
package stays installable without scanpy/numba, ``tier2.neighborhood`` already
sets that precedent, and the statistic is a few lines over a sparse matrix.

**Scope difference worth knowing:** the legacy report computes Moran's I over
*cells* inside pooled ego-graphs. Here the spatial table is root-instance level,
so this measures autocorrelation over the *instance* graph. That is a defensible
and arguably cleaner object, but it is not numerically the same statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["MoranResult", "morans_i", "run_morans_i", "diff_morans_i_vs_reference"]


@dataclass(frozen=True)
class MoranResult:
    """Per-group Moran's I with its permutation null.

    Attributes:
        statistic: group x feature observed I.
        expected: group x feature permutation mean.
        variance: group x feature permutation variance.
        pvalue: two-sided permutation p-value.
        qvalue: Benjamini-Hochberg adjusted p-value across all tests.
        n_nodes: node count per group.
    """

    statistic: pd.DataFrame
    expected: pd.DataFrame
    variance: pd.DataFrame
    pvalue: pd.DataFrame
    qvalue: pd.DataFrame
    n_nodes: pd.Series


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-FDR adjustment, ignoring NaNs."""
    flat = np.asarray(pvalues, dtype=float).ravel()
    out = np.full_like(flat, np.nan)
    finite = np.isfinite(flat)
    if not finite.any():
        return out.reshape(np.shape(pvalues))
    try:
        from scipy.stats import false_discovery_control

        out[finite] = false_discovery_control(flat[finite], method="bh")
    except Exception:
        values = flat[finite]
        order = np.argsort(values)
        ranked = values[order]
        n = ranked.size
        adjusted = ranked * n / np.arange(1, n + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        out[finite] = restored
    return out.reshape(np.shape(pvalues))


def morans_i(values: np.ndarray, adjacency) -> float:
    """Global Moran's I of ``values`` over a sparse connectivity matrix."""
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n < 2:
        return float("nan")
    s0 = float(adjacency.sum())
    if s0 <= 0:
        return float("nan")
    z = values - values.mean()
    denom = float(z @ z)
    if denom <= 0:
        # A constant feature has no variance to correlate.
        return float("nan")
    return float((n / s0) * float(z @ (adjacency @ z)) / denom)


def _adjacency_from_edges(
    edges: pd.DataFrame,
    node_ids: Sequence[str],
    *,
    source_column: str,
    target_column: str,
):
    from scipy.sparse import coo_matrix

    index = {str(node): position for position, node in enumerate(node_ids)}
    src = edges[source_column].astype(str).map(index)
    dst = edges[target_column].astype(str).map(index)
    keep = src.notna() & dst.notna()
    src = src[keep].to_numpy(dtype=np.int64)
    dst = dst[keep].to_numpy(dtype=np.int64)
    n = len(node_ids)
    if src.size == 0:
        return coo_matrix((n, n), dtype=float).tocsr()
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.size, dtype=float)
    adjacency = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0  # binary weights; collapse any duplicate edges
    return adjacency


def run_morans_i(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_column: str,
    id_column: str = "instance_id",
    source_column: str = "source_id",
    target_column: str = "target_id",
    n_perms: int = 100,
    random_state: int = 0,
    min_nodes: int = 3,
) -> MoranResult:
    """Moran's I per group, with a permutation null over node labels."""
    missing = [c for c in (*feature_columns, group_column, id_column) if c not in node_table]
    if missing:
        raise ValueError(f"node_table is missing required column(s): {missing}.")

    rng = np.random.default_rng(random_state)
    groups = sorted(node_table[group_column].unique())
    statistic: Dict[object, Dict[str, float]] = {}
    expected: Dict[object, Dict[str, float]] = {}
    variance: Dict[object, Dict[str, float]] = {}
    pvalue: Dict[object, Dict[str, float]] = {}
    counts: Dict[object, int] = {}

    for group in groups:
        subset = node_table[node_table[group_column] == group]
        counts[group] = int(len(subset))
        statistic[group] = {}
        expected[group] = {}
        variance[group] = {}
        pvalue[group] = {}
        if len(subset) < min_nodes:
            for feature in feature_columns:
                statistic[group][feature] = float("nan")
                expected[group][feature] = float("nan")
                variance[group][feature] = float("nan")
                pvalue[group][feature] = float("nan")
            continue

        node_ids = subset[id_column].astype(str).tolist()
        keep_ids = set(node_ids)
        edges = spatial_table[
            spatial_table[source_column].astype(str).isin(keep_ids)
            & spatial_table[target_column].astype(str).isin(keep_ids)
        ]
        adjacency = _adjacency_from_edges(
            edges, node_ids, source_column=source_column, target_column=target_column
        )

        for feature in feature_columns:
            values = subset[feature].to_numpy(dtype=float)
            observed = morans_i(values, adjacency)
            statistic[group][feature] = observed

            if n_perms <= 0 or not np.isfinite(observed):
                expected[group][feature] = float("nan")
                variance[group][feature] = float("nan")
                pvalue[group][feature] = float("nan")
                continue

            null = np.array(
                [morans_i(rng.permutation(values), adjacency) for _ in range(int(n_perms))],
                dtype=float,
            )
            null = null[np.isfinite(null)]
            if null.size == 0:
                expected[group][feature] = float("nan")
                variance[group][feature] = float("nan")
                pvalue[group][feature] = float("nan")
                continue
            expected[group][feature] = float(null.mean())
            variance[group][feature] = float(null.var(ddof=1)) if null.size > 1 else 0.0
            # Two-sided, with the observed value included so p is never zero.
            extreme = int(np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean())))
            pvalue[group][feature] = float((extreme + 1) / (null.size + 1))

    columns = list(feature_columns)
    stat_frame = pd.DataFrame(statistic).T.reindex(columns=columns)
    exp_frame = pd.DataFrame(expected).T.reindex(columns=columns)
    var_frame = pd.DataFrame(variance).T.reindex(columns=columns)
    p_frame = pd.DataFrame(pvalue).T.reindex(columns=columns)
    q_frame = pd.DataFrame(
        _benjamini_hochberg(p_frame.to_numpy()), index=p_frame.index, columns=p_frame.columns
    )
    return MoranResult(
        statistic=stat_frame,
        expected=exp_frame,
        variance=var_frame,
        pvalue=p_frame,
        qvalue=q_frame,
        n_nodes=pd.Series(counts).reindex(stat_frame.index),
    )


def diff_morans_i_vs_reference(
    result: MoranResult,
    *,
    reference_group: object = -1,
) -> pd.DataFrame:
    """Differential autocorrelation against a reference group.

    ``z = (I_k - I_ref) / sqrt(var_k + var_ref)``, using the permutation-null
    variances. The reference is typically the HDBSCAN noise cluster.
    """
    if reference_group not in result.statistic.index:
        raise ValueError(
            f"reference_group {reference_group!r} is not present; available: "
            f"{list(result.statistic.index)}."
        )
    ref_stat = result.statistic.loc[reference_group]
    ref_var = result.variance.loc[reference_group]

    rows: List[pd.Series] = []
    index: List[object] = []
    for group in result.statistic.index:
        if group == reference_group:
            continue
        delta = result.statistic.loc[group] - ref_stat
        pooled = np.sqrt(result.variance.loc[group].astype(float) + ref_var.astype(float))
        rows.append(delta / pooled.replace(0.0, np.nan))
        index.append(group)
    if not rows:
        return pd.DataFrame(columns=result.statistic.columns)
    return pd.DataFrame(rows, index=index)
