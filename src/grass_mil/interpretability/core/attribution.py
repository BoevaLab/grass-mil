"""Exact attribution of a bag prediction to its instances.

Because the bag logit is an attention-weighted sum of instance logits with no
added bias,

.. math:: L_c = \\sum_{i \\in B_r} A_{i,c}\\, \\ell_{i,c},

each instance carries an exact additive contribution
:math:`M_{i,c} = A_{i,c} \\ell_{i,c}`. This is an identity, not an
approximation: it needs no baseline, no integration path, and no surrogate
model. :func:`niche_attribution_summary` reports ``identity_residual`` so the
assumption is checked rather than trusted.

The reported margin removes the head's per-class bias, so it reflects the
instance-driven part of the decision rather than the head's prior. It is a
one-vs-rest contrast, which for a binary head is exactly

.. math:: m_i = A_{i,1}(\\ell_{i,1} - \\beta_1) - A_{i,0}(\\ell_{i,0} - \\beta_0)

and generalises to any number of classes (see :func:`instance_ovr_margins`).

Niche-level summaries aggregate within niche x region cells and are
reported region-equal with percentile bootstrap intervals over regions, because
regions differ enormously in instance count and pooling over instances would
let one large region dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "AttributionResult",
    "niche_attribution_summary",
    "instance_margin_contributions",
    "instance_ovr_margins",
    "percentile_bootstrap_ci",
]


@dataclass(frozen=True)
class AttributionResult:
    """Per-instance contributions and their niche-level summaries.

    Attributes:
        per_instance: One row per instance with its contribution columns.
        per_niche: One row per niche; each statistic carries ``_lo``/``_hi``
            percentile bootstrap bounds.
        identity_residual: ``max_r |sum_i M[i,c] - L_c|``. Non-zero means the
            additive identity does not hold and the attribution is invalid.
    """

    per_instance: pd.DataFrame
    per_niche: pd.DataFrame
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


def instance_ovr_margins(contributions: np.ndarray) -> np.ndarray:
    """One-vs-rest margin per instance and class.

    For class :math:`c`, the margin contrasts that class's contribution against
    the mean of the others:

    .. math:: m_{i,c} = M_{i,c} - \\frac{1}{C-1} \\sum_{c' \\neq c} M_{i,c'}

    With ``C == 2`` this reduces exactly to :math:`M_{i,1} - M_{i,0}`, the
    classic binary margin, so binary results are unchanged. With ``C == 1``
    (regression, or Cox on the log-hazard scale) the contribution is already the
    margin.

    Returns:
        ``[n_instances, n_classes]``.
    """
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim == 1:
        contributions = contributions[:, None]
    n_classes = contributions.shape[1]
    if n_classes == 1:
        return contributions
    total = contributions.sum(axis=1, keepdims=True)
    rest_mean = (total - contributions) / (n_classes - 1)
    return contributions - rest_mean


def _resolve_focus_classes(n_classes: int, focus_classes) -> List[int]:
    """Which classes get a niche-level summary.

    Binary tasks summarise the positive class only: the class-0 margin is its
    exact negation, so reporting both is redundant. Multi-class tasks summarise
    every class, since no single contrast represents the decision.
    """
    if focus_classes is not None:
        resolved = [int(c) for c in focus_classes]
        for c in resolved:
            if c < 0 or c >= n_classes:
                raise ValueError(f"focus class {c} is out of range for {n_classes} class(es).")
        return resolved
    if n_classes <= 2:
        return [n_classes - 1]
    return list(range(n_classes))


def niche_attribution_summary(
    table: pd.DataFrame,
    niche_labels: np.ndarray,
    *,
    attention_columns: Sequence[str],
    logit_columns: Sequence[str],
    bag_id_column: str = "bag_id",
    logit_bias: Optional[Sequence[float]] = None,
    n_bootstrap: int = 200,
    random_state: int = 0,
    margin_eps: float = 1e-6,
    bag_logits: Optional[Dict[str, Sequence[float]]] = None,
    focus_classes: Optional[Sequence[int]] = None,
) -> AttributionResult:
    """Summarise each niche's contribution to the bag decisions.

    Args:
        table: Instance table; must carry the attention, logit and bag columns.
        niche_labels: One niche id per row of ``table``.
        attention_columns: Per-class attention columns, or a single shared one.
        logit_columns: Per-class instance logit columns.
        bag_id_column: Column grouping instances into bags.
        logit_bias: Head output bias per class, removed from the margin.
        bag_logits: Optional bag logits keyed by bag id. When given, the
            additive identity is checked against them.
        focus_classes: Classes to summarise. Defaults to the positive class for
            binary tasks (the class-0 margin is its exact negation) and to every
            class for multi-class tasks.
    """
    for column in (*attention_columns, *logit_columns, bag_id_column):
        if column not in table.columns:
            raise ValueError(f"Missing required column {column!r} for attribution.")
    if len(niche_labels) != len(table):
        raise ValueError(f"niche_labels has {len(niche_labels)} entries for {len(table)} rows.")

    attention = table.loc[:, list(attention_columns)].to_numpy(dtype=float)
    logits = table.loc[:, list(logit_columns)].to_numpy(dtype=float)
    contributions = instance_margin_contributions(attention, logits, logit_bias=logit_bias)
    raw_contributions = instance_margin_contributions(attention, logits, logit_bias=None)

    per_instance = pd.DataFrame(
        {
            bag_id_column: table[bag_id_column].astype(str).to_numpy(),
            "niche_label": np.asarray(niche_labels),
        }
    )
    margins = instance_ovr_margins(contributions)
    n_classes = int(contributions.shape[1])
    for index in range(n_classes):
        per_instance[f"contribution_c{index}"] = contributions[:, index]
        per_instance[f"margin_c{index}"] = margins[:, index]

    identity_residual = _identity_residual(
        per_instance[bag_id_column].to_numpy(), raw_contributions, bag_logits
    )

    summaries = []
    for class_index in _resolve_focus_classes(n_classes, focus_classes):
        frame = per_instance.copy()
        frame["margin"] = margins[:, class_index]
        part = _summarise_clusters(
            frame,
            attention=_attention_for_class(attention, class_index),
            bag_id_column=bag_id_column,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            margin_eps=margin_eps,
        )
        part["class_index"] = class_index
        summaries.append(part.set_index("class_index", append=True))
    summary = pd.concat(summaries).sort_index()
    return AttributionResult(
        per_instance=per_instance,
        per_niche=summary,
        identity_residual=identity_residual,
    )


def _attention_for_class(attention: np.ndarray, class_index: int) -> np.ndarray:
    """Attention weights driving one class, or the shared channel."""
    if attention.ndim == 1:
        return attention[:, None]
    if attention.shape[1] == 1:
        return attention
    return attention[:, class_index : class_index + 1]


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

    cell_margin = frame.groupby([bag_id_column, "niche_label"])["margin"].sum()
    cell_count = frame.groupby([bag_id_column, "niche_label"]).size()
    cell_attention = frame.groupby([bag_id_column, "niche_label"])["__attention"].sum()

    rng = np.random.default_rng(random_state)
    rows = []
    for niche in sorted(frame["niche_label"].unique()):
        cluster_margin = cell_margin.xs(niche, level="niche_label")
        cluster_count = cell_count.xs(niche, level="niche_label")
        cluster_attn = cell_attention.xs(niche, level="niche_label")
        bags = cluster_margin.index

        totals = bag_margin.loc[bags].to_numpy(dtype=float)
        margins = cluster_margin.to_numpy(dtype=float)
        counts = cluster_count.to_numpy(dtype=float)
        sizes = bag_size.loc[bags].to_numpy(dtype=float)
        attn = cluster_attn.to_numpy(dtype=float)

        prevalence = counts / np.maximum(sizes, 1.0)
        # Abundance-corrected attention lift: 1 is neutral, >1 preferentially
        # attended relative to how common the niche is in that bag.
        lift = np.divide(attn, prevalence, out=np.full_like(attn, np.nan), where=prevalence > 0)

        # Shares are fractions of the bag margin and sum to 1 across a full
        # partition, but an individual niche is NOT bounded to [-1, 1]: it can
        # exceed 1 when another niche pushes the opposite way. Regions whose
        # margin is near zero are excluded rather than clipped, since the ratio
        # is meaningless there.
        decisive = np.abs(totals) > margin_eps
        share = np.divide(margins, totals, out=np.full_like(margins, np.nan), where=decisive)
        consistency = np.where(decisive, np.sign(margins) == np.sign(totals), np.nan)

        row: Dict[str, object] = {"niche_label": niche, "n_regions": int(len(bags))}
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

    return pd.DataFrame(rows).set_index("niche_label").sort_index()
