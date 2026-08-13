"""Exact attribution of a bag prediction to its instances.

Because the bag logit is an attention-weighted sum of instance logits with no
added bias,

.. math:: L_c = \\sum_{i \\in B_r} A_{i,c}\\, \\ell_{i,c},

each instance carries an exact additive contribution
:math:`M_{i,c} = A_{i,c} \\ell_{i,c}`. This is an identity, not an
approximation: it needs no baseline, no integration path, and no surrogate
model. :func:`cluster_attribution_summary` reports ``identity_residual`` so the
assumption is checked rather than trusted.

For binary tasks the reported margin removes the head's per-class bias, so it
reflects the instance-driven part of the decision:

.. math:: m_i = A_{i,1}(\\ell_{i,1} - \\beta_1) - A_{i,0}(\\ell_{i,0} - \\beta_0)

Cluster-level summaries aggregate within cluster x region cells and are
reported region-equal with percentile bootstrap intervals over regions, because
regions differ enormously in instance count and pooling over instances would
let one large region dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "AttributionResult",
    "cluster_attribution_summary",
    "instance_margin_contributions",
    "percentile_bootstrap_ci",
]


@dataclass(frozen=True)
class AttributionResult:
    """Per-instance contributions and their cluster-level summaries.

    Attributes:
        per_instance: One row per instance with its contribution columns.
        per_cluster: One row per cluster; each statistic carries ``_lo``/``_hi``
            percentile bootstrap bounds.
        identity_residual: ``max_r |sum_i M[i,c] - L_c|``. Non-zero means the
            additive identity does not hold and the attribution is invalid.
    """

    per_instance: pd.DataFrame
    per_cluster: pd.DataFrame
    identity_residual: float


def percentile_bootstrap_ci(
    values: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    n_boot: int = 200,
    q_lo: float = 2.5,
    q_hi: float = 97.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval over resampled units.

    Returns ``(point, lo, hi)``; all three are NaN for an empty input.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        finite &= np.isfinite(weights)
    values = values[finite]
    if weights is not None:
        weights = weights[finite]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    def _estimate(sample_values: np.ndarray, sample_weights: Optional[np.ndarray]) -> float:
        if sample_weights is None:
            return float(np.mean(sample_values))
        total = float(np.sum(np.abs(sample_weights)))
        if total <= 0:
            return float("nan")
        return float(np.sum(sample_weights * sample_values) / total)

    point = _estimate(values, weights)
    if values.size == 1 or n_boot <= 0:
        return point, point, point

    rng = rng or np.random.default_rng(0)
    draws = rng.integers(0, values.size, size=(int(n_boot), values.size))
    samples = np.array(
        [_estimate(values[idx], None if weights is None else weights[idx]) for idx in draws],
        dtype=float,
    )
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(samples, q_lo)), float(np.percentile(samples, q_hi))


def instance_margin_contributions(
    attention: np.ndarray,
    logits: np.ndarray,
    *,
    logit_bias: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """``M[i,c] = A[i,c] * (l[i,c] - beta_c)``.

    ``logit_bias`` is the head's per-class output bias. Removing it isolates the
    instance-driven component; pass ``None`` to keep the raw identity.
    """
    attention = np.asarray(attention, dtype=float)
    logits = np.asarray(logits, dtype=float)
    if attention.ndim == 1:
        attention = attention[:, None]
    if logits.ndim == 1:
        logits = logits[:, None]
    if attention.shape[0] != logits.shape[0]:
        raise ValueError(
            "attention and logits must have the same number of rows; got "
            f"{attention.shape[0]} and {logits.shape[0]}."
        )
    if attention.shape[1] not in (1, logits.shape[1]):
        raise ValueError(
            f"attention has {attention.shape[1]} channel(s), which is neither 1 nor the "
            f"{logits.shape[1]} class logit(s)."
        )

    centred = logits
    if logit_bias is not None:
        bias = np.asarray(logit_bias, dtype=float).reshape(1, -1)
        if bias.shape[1] != logits.shape[1]:
            raise ValueError(
                f"logit_bias has {bias.shape[1]} entries but there are {logits.shape[1]} classes."
            )
        centred = logits - bias
    return attention * centred


def _binary_margin(contributions: np.ndarray) -> np.ndarray:
    """Class-1-vs-0 margin per instance, or the single column when C == 1."""
    if contributions.shape[1] == 1:
        return contributions[:, 0]
    return contributions[:, 1] - contributions[:, 0]


def cluster_attribution_summary(
    table: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    attention_columns: Sequence[str],
    logit_columns: Sequence[str],
    bag_id_column: str = "bag_id",
    logit_bias: Optional[Sequence[float]] = None,
    n_bootstrap: int = 200,
    random_state: int = 0,
    margin_eps: float = 1e-6,
    bag_logits: Optional[Dict[str, Sequence[float]]] = None,
) -> AttributionResult:
    """Summarise each cluster's contribution to the bag decisions.

    Args:
        table: Instance table; must carry the attention, logit and bag columns.
        cluster_labels: One cluster id per row of ``table``.
        attention_columns: Per-class attention columns, or a single shared one.
        logit_columns: Per-class instance logit columns.
        bag_id_column: Column grouping instances into bags.
        logit_bias: Head output bias per class, removed from the margin.
        bag_logits: Optional bag logits keyed by bag id. When given, the
            additive identity is checked against them.
    """
    for column in (*attention_columns, *logit_columns, bag_id_column):
        if column not in table.columns:
            raise ValueError(f"Missing required column {column!r} for attribution.")
    if len(cluster_labels) != len(table):
        raise ValueError(
            f"cluster_labels has {len(cluster_labels)} entries for {len(table)} rows."
        )

    attention = table.loc[:, list(attention_columns)].to_numpy(dtype=float)
    logits = table.loc[:, list(logit_columns)].to_numpy(dtype=float)
    contributions = instance_margin_contributions(attention, logits, logit_bias=logit_bias)
    raw_contributions = instance_margin_contributions(attention, logits, logit_bias=None)

    per_instance = pd.DataFrame(
        {
            bag_id_column: table[bag_id_column].astype(str).to_numpy(),
            "cluster_label": np.asarray(cluster_labels),
        }
    )
    for index in range(contributions.shape[1]):
        per_instance[f"contribution_c{index}"] = contributions[:, index]
    per_instance["margin"] = _binary_margin(contributions)

    identity_residual = _identity_residual(
        per_instance[bag_id_column].to_numpy(), raw_contributions, bag_logits
    )

    summary = _summarise_clusters(
        per_instance,
        attention=attention,
        bag_id_column=bag_id_column,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        margin_eps=margin_eps,
    )
    return AttributionResult(
        per_instance=per_instance,
        per_cluster=summary,
        identity_residual=identity_residual,
    )


def _identity_residual(
    bag_ids: np.ndarray,
    contributions: np.ndarray,
    bag_logits: Optional[Dict[str, Sequence[float]]],
) -> float:
    """``max_r |sum_i M[i,c] - L_c|``, or NaN when bag logits are unavailable."""
    if not bag_logits:
        return float("nan")
    residual = 0.0
    frame = pd.DataFrame(contributions)
    frame["__bag"] = bag_ids
    for bag_id, group in frame.groupby("__bag"):
        expected = bag_logits.get(str(bag_id))
        if expected is None:
            continue
        got = group.drop(columns="__bag").to_numpy(dtype=float).sum(axis=0)
        expected_arr = np.asarray(expected, dtype=float).reshape(-1)
        width = min(got.size, expected_arr.size)
        residual = max(residual, float(np.max(np.abs(got[:width] - expected_arr[:width]))))
    return residual


def _summarise_clusters(
    per_instance: pd.DataFrame,
    *,
    attention: np.ndarray,
    bag_id_column: str,
    n_bootstrap: int,
    random_state: int,
    margin_eps: float,
) -> pd.DataFrame:
    frame = per_instance.copy()
    # A shared attention channel is broadcast; per-class attention is summed to
    # one selection weight per instance.
    frame["__attention"] = attention.sum(axis=1) if attention.ndim == 2 else attention

    bag_margin = frame.groupby(bag_id_column)["margin"].sum()
    bag_size = frame.groupby(bag_id_column).size()

    cell_margin = frame.groupby([bag_id_column, "cluster_label"])["margin"].sum()
    cell_count = frame.groupby([bag_id_column, "cluster_label"]).size()
    cell_attention = frame.groupby([bag_id_column, "cluster_label"])["__attention"].sum()

    rng = np.random.default_rng(random_state)
    rows = []
    for cluster in sorted(frame["cluster_label"].unique()):
        cluster_margin = cell_margin.xs(cluster, level="cluster_label")
        cluster_count = cell_count.xs(cluster, level="cluster_label")
        cluster_attn = cell_attention.xs(cluster, level="cluster_label")
        bags = cluster_margin.index

        totals = bag_margin.loc[bags].to_numpy(dtype=float)
        margins = cluster_margin.to_numpy(dtype=float)
        counts = cluster_count.to_numpy(dtype=float)
        sizes = bag_size.loc[bags].to_numpy(dtype=float)
        attn = cluster_attn.to_numpy(dtype=float)

        prevalence = counts / np.maximum(sizes, 1.0)
        # Abundance-corrected attention lift: 1 is neutral, >1 preferentially
        # attended relative to how common the cluster is in that bag.
        lift = np.divide(attn, prevalence, out=np.full_like(attn, np.nan), where=prevalence > 0)

        # Shares are fractions of the bag margin and sum to 1 across a full
        # partition, but an individual cluster is NOT bounded to [-1, 1]: it can
        # exceed 1 when another cluster pushes the opposite way. Regions whose
        # margin is near zero are excluded rather than clipped, since the ratio
        # is meaningless there.
        decisive = np.abs(totals) > margin_eps
        share = np.divide(margins, totals, out=np.full_like(margins, np.nan), where=decisive)
        consistency = np.where(decisive, np.sign(margins) == np.sign(totals), np.nan)

        row: Dict[str, object] = {"cluster_label": cluster, "n_regions": int(len(bags))}
        for name, values, weights in (
            ("margin_signed_share", share, None),
            ("margin_weighted", margins, np.abs(totals)),
            ("margin_sign_consistency", consistency, None),
            ("attention_lift", lift, None),
            ("prevalence", prevalence, None),
        ):
            point, lo, hi = percentile_bootstrap_ci(
                values, weights=weights, n_boot=n_bootstrap, rng=rng
            )
            row[name] = point
            row[f"{name}_lo"] = lo
            row[f"{name}_hi"] = hi
        row["n_instances"] = int(counts.sum())
        rows.append(row)

    return pd.DataFrame(rows).set_index("cluster_label").sort_index()
