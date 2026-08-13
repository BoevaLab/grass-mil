"""Contracts for the exact additive attribution.

The load-bearing property is the identity sum_i A[i,c] * l[i,c] == L_c. If it
does not hold exactly, every cluster-level attribution built on it is invalid,
so it is asserted directly rather than inferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.interpretability.core.attribution import (
    cluster_attribution_summary,
    instance_margin_contributions,
    percentile_bootstrap_ci,
)


def _instance_table(n_bags: int = 4, per_bag: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for bag in range(n_bags):
        attn_logits = rng.normal(size=(per_bag, 2))
        attention = np.exp(attn_logits) / np.exp(attn_logits).sum(axis=0, keepdims=True)
        logits = rng.normal(size=(per_bag, 2))
        for i in range(per_bag):
            rows.append(
                {
                    "bag_id": f"bag_{bag}",
                    "attention_c0": attention[i, 0],
                    "attention_c1": attention[i, 1],
                    "logit_0": logits[i, 0],
                    "logit_1": logits[i, 1],
                }
            )
    table = pd.DataFrame(rows)
    labels = np.array([i % 3 for i in range(len(table))])
    return table, labels


def test_contributions_sum_exactly_to_the_bag_logit() -> None:
    table, labels = _instance_table()

    bag_logits = {}
    for bag_id, group in table.groupby("bag_id"):
        attention = group[["attention_c0", "attention_c1"]].to_numpy()
        logits = group[["logit_0", "logit_1"]].to_numpy()
        bag_logits[bag_id] = (attention * logits).sum(axis=0)

    result = cluster_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        bag_logits=bag_logits,
        n_bootstrap=20,
    )
    assert result.identity_residual < 1e-10


def test_identity_residual_detects_a_broken_bag_logit() -> None:
    """A wrong bag logit must surface, not pass silently."""
    table, labels = _instance_table(n_bags=2)
    bad = {bag_id: np.array([99.0, -99.0]) for bag_id in table["bag_id"].unique()}

    result = cluster_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        bag_logits=bad,
        n_bootstrap=10,
    )
    assert result.identity_residual > 1.0


def test_bias_removal_shifts_the_margin_by_exactly_the_bias_term() -> None:
    attention = np.array([[0.25, 0.75], [0.75, 0.25]])
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    bias = [0.5, -0.25]

    raw = instance_margin_contributions(attention, logits)
    debiased = instance_margin_contributions(attention, logits, logit_bias=bias)

    expected_shift = attention * np.asarray(bias).reshape(1, -1)
    np.testing.assert_allclose(raw - debiased, expected_shift)


def test_margin_contributions_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        instance_margin_contributions(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="neither 1 nor"):
        instance_margin_contributions(np.zeros((3, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="logit_bias"):
        instance_margin_contributions(np.zeros((3, 2)), np.zeros((3, 2)), logit_bias=[1.0])


def test_cluster_summaries_are_bounded_and_carry_intervals() -> None:
    table, labels = _instance_table(n_bags=6, per_bag=8)
    result = cluster_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        n_bootstrap=50,
    )
    summary = result.per_cluster

    assert set(summary.index) == {0, 1, 2}
    # Note: margin_signed_share is NOT bounded to [-1, 1]. It is a fraction of
    # the bag margin, so a cluster pushing hard in one direction can exceed 1
    # when another cluster opposes it. The real invariant is that a full
    # partition sums to 1, covered separately below.
    assert np.isfinite(summary["margin_signed_share"]).all()
    assert summary["margin_sign_consistency"].between(0.0, 1.0).all()
    assert summary["prevalence"].between(0.0, 1.0).all()

    for column in ("attention_lift", "prevalence"):
        assert (summary[f"{column}_lo"] <= summary[column] + 1e-9).all()
        assert (summary[column] <= summary[f"{column}_hi"] + 1e-9).all()


def test_attention_lift_is_neutral_when_attention_matches_abundance() -> None:
    """Uniform attention over a bag gives every cluster a lift of 1."""
    per_bag = 6
    rows = []
    for bag in range(4):
        for i in range(per_bag):
            rows.append(
                {
                    "bag_id": f"bag_{bag}",
                    "attention_c0": 1.0 / per_bag,
                    "logit_0": float(i),
                }
            )
    table = pd.DataFrame(rows)
    labels = np.array([i % 2 for i in range(len(table))])

    result = cluster_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0"],
        logit_columns=["logit_0"],
        n_bootstrap=20,
    )
    np.testing.assert_allclose(result.per_cluster["attention_lift"].to_numpy(), 1.0, atol=1e-9)


def test_signed_share_across_clusters_sums_to_one_per_region() -> None:
    """Shares are fractions of the bag margin, so a full partition sums to 1."""
    table, labels = _instance_table(n_bags=1, per_bag=9)
    result = cluster_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        n_bootstrap=0,
    )
    assert result.per_cluster["margin_signed_share"].sum() == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=2.0, size=200)
    point, lo, hi = percentile_bootstrap_ci(values, n_boot=200, rng=rng)
    assert lo <= point <= hi
    assert point == pytest.approx(float(np.mean(values)))


def test_bootstrap_ci_handles_degenerate_inputs() -> None:
    empty = percentile_bootstrap_ci(np.array([]), n_boot=10)
    assert all(np.isnan(v) for v in empty)

    single = percentile_bootstrap_ci(np.array([3.0]), n_boot=10)
    assert single == (3.0, 3.0, 3.0)

    # NaNs are dropped rather than poisoning the estimate.
    point, _, _ = percentile_bootstrap_ci(np.array([1.0, np.nan, 3.0]), n_boot=10)
    assert point == pytest.approx(2.0)
