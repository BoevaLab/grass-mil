"""Contracts for the exact additive attribution.

The load-bearing property is the identity sum_i A[i,c] * l[i,c] == L_c. If it
does not hold exactly, every niche-level attribution built on it is invalid,
so it is asserted directly rather than inferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.interpretability.core.attribution import (
    niche_attribution_summary,
    instance_margin_contributions,
    instance_ovr_margins,
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

    result = niche_attribution_summary(
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

    result = niche_attribution_summary(
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
    result = niche_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        n_bootstrap=50,
    )
    summary = result.per_niche

    # Indexed by (niche_label, class_index). A binary head summarises the
    # positive class only: the class-0 margin is its exact negation.
    assert set(summary.index) == {(0, 1), (1, 1), (2, 1)}
    # Note: margin_signed_share is NOT bounded to [-1, 1]. It is a fraction of
    # the bag margin, so a niche pushing hard in one direction can exceed 1
    # when another niche opposes it. The real invariant is that a full
    # partition sums to 1, covered separately below.
    assert np.isfinite(summary["margin_signed_share"]).all()
    assert summary["margin_sign_consistency"].between(0.0, 1.0).all()
    assert summary["prevalence"].between(0.0, 1.0).all()

    for column in ("prevalence",):
        assert (summary[f"{column}_lo"] <= summary[column] + 1e-9).all()
        assert (summary[column] <= summary[f"{column}_hi"] + 1e-9).all()


def test_signed_share_across_clusters_sums_to_one_per_region() -> None:
    """Shares are fractions of the bag margin, so a full partition sums to 1."""
    table, labels = _instance_table(n_bags=1, per_bag=9)
    result = niche_attribution_summary(
        table,
        labels,
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        n_bootstrap=0,
    )
    assert result.per_niche["margin_signed_share"].sum() == pytest.approx(1.0, abs=1e-9)


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


def _multiclass_table(n_bags: int = 4, per_bag: int = 6, n_classes: int = 3, seed: int = 3):
    rng = np.random.default_rng(seed)
    rows = []
    for bag in range(n_bags):
        attn_logits = rng.normal(size=(per_bag, n_classes))
        attention = np.exp(attn_logits) / np.exp(attn_logits).sum(axis=0, keepdims=True)
        logits = rng.normal(size=(per_bag, n_classes))
        for i in range(per_bag):
            row = {"bag_id": f"bag_{bag}"}
            for c in range(n_classes):
                row[f"attention_c{c}"] = attention[i, c]
                row[f"logit_{c}"] = logits[i, c]
            rows.append(row)
    table = pd.DataFrame(rows)
    labels = np.array([i % 3 for i in range(len(table))])
    return table, labels


def test_ovr_margin_reduces_to_the_binary_margin_when_c_is_two() -> None:
    """The binary case must be unchanged by the OvR generalisation."""
    contributions = np.array([[0.2, 0.9], [-0.4, 0.1], [1.0, -1.0]])
    margins = instance_ovr_margins(contributions)

    np.testing.assert_allclose(margins[:, 1], contributions[:, 1] - contributions[:, 0])
    # Class 0 is the exact negation, which is why binary summarises one class.
    np.testing.assert_allclose(margins[:, 0], -margins[:, 1])


def test_ovr_margin_contrasts_each_class_against_the_mean_of_the_rest() -> None:
    contributions = np.array([[3.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    margins = instance_ovr_margins(contributions)

    # Instance 0: class 0 stands out against a zero mean of the others.
    assert margins[0, 0] == pytest.approx(3.0)
    assert margins[0, 1] == pytest.approx(-1.5)
    # Instance 1: perfectly balanced, so no class is favoured.
    np.testing.assert_allclose(margins[1], np.zeros(3), atol=1e-12)


def test_ovr_margin_is_a_single_column_for_scalar_heads() -> None:
    """Regression and Cox emit one logit; the contribution is the margin."""
    contributions = np.array([[0.7], [-0.2]])
    np.testing.assert_allclose(instance_ovr_margins(contributions), contributions)


def test_multiclass_attribution_summarises_every_class() -> None:
    table, labels = _multiclass_table(n_classes=3)
    attention_columns = [f"attention_c{c}" for c in range(3)]
    logit_columns = [f"logit_{c}" for c in range(3)]

    result = niche_attribution_summary(
        table,
        labels,
        attention_columns=attention_columns,
        logit_columns=logit_columns,
        n_bootstrap=20,
    )
    # One row per (niche, class): no class is silently ignored.
    assert set(result.per_niche.index) == {(c, k) for c in (0, 1, 2) for k in (0, 1, 2)}
    for c in range(3):
        assert f"margin_c{c}" in result.per_instance.columns
        assert f"contribution_c{c}" in result.per_instance.columns


def test_multiclass_identity_holds_for_every_class() -> None:
    table, labels = _multiclass_table(n_classes=4)
    attention_columns = [f"attention_c{c}" for c in range(4)]
    logit_columns = [f"logit_{c}" for c in range(4)]

    bag_logits = {}
    for bag_id, group in table.groupby("bag_id"):
        attention = group[attention_columns].to_numpy()
        logits = group[logit_columns].to_numpy()
        bag_logits[bag_id] = (attention * logits).sum(axis=0)

    result = niche_attribution_summary(
        table,
        labels,
        attention_columns=attention_columns,
        logit_columns=logit_columns,
        bag_logits=bag_logits,
        n_bootstrap=10,
    )
    assert result.identity_residual < 1e-10


def test_focus_classes_can_be_selected_explicitly() -> None:
    table, labels = _multiclass_table(n_classes=3)
    result = niche_attribution_summary(
        table,
        labels,
        attention_columns=[f"attention_c{c}" for c in range(3)],
        logit_columns=[f"logit_{c}" for c in range(3)],
        focus_classes=[2],
        n_bootstrap=10,
    )
    assert {k for _, k in result.per_niche.index} == {2}

    with pytest.raises(ValueError, match="out of range"):
        niche_attribution_summary(
            table,
            labels,
            attention_columns=[f"attention_c{c}" for c in range(3)],
            logit_columns=[f"logit_{c}" for c in range(3)],
            focus_classes=[7],
        )


def test_head_bias_is_read_from_a_checkpoint(tmp_path) -> None:
    """The bias must come from the trained model, not be transcribed by hand."""
    import torch

    from grass_mil.models.components.heads import GraphPredictionHead
    from grass_mil.models.training.checkpoint_init import read_graph_head_bias

    head = GraphPredictionHead(input_dim=4, output_dim=3, hidden_dim=4, num_layers=2)
    with torch.no_grad():
        head.net[-1].bias.copy_(torch.tensor([0.25, -0.5, 1.5]))

    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": {f"graph_head.{k}": v for k, v in head.state_dict().items()}}, ckpt)

    assert read_graph_head_bias(str(ckpt)) == pytest.approx([0.25, -0.5, 1.5])
    # An encoder-only checkpoint has no head to read.
    empty = tmp_path / "encoder.ckpt"
    torch.save({"state_dict": {"encoder.layers.0.weight": torch.zeros(2, 2)}}, empty)
    assert read_graph_head_bias(str(empty)) is None


def test_margin_plugin_reads_the_bias_from_a_checkpoint(tmp_path) -> None:
    import torch

    from grass_mil.contracts import InterpretabilityDataset
    from grass_mil.interpretability.plugins.base import PluginContext
    from grass_mil.interpretability.plugins.builtin import create_builtin_registry
    from grass_mil.models.components.heads import GraphPredictionHead

    table, labels = _instance_table(n_bags=3, per_bag=4)
    dataset = InterpretabilityDataset(
        instance_table=table, spatial_table=None, id_column="bag_id", bag_id_column="bag_id"
    )

    head = GraphPredictionHead(input_dim=4, output_dim=2, hidden_dim=4, num_layers=2)
    with torch.no_grad():
        head.net[-1].bias.copy_(torch.tensor([5.0, -5.0]))
    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": {f"graph_head.{k}": v for k, v in head.state_dict().items()}}, ckpt)

    plugin = create_builtin_registry().get("margin_attribution")
    context = PluginContext(state={"niche_labels": labels})
    unbiased = plugin.run(dataset, context, n_bootstrap=0)
    debiased = plugin.run(dataset, context, n_bootstrap=0, logit_bias_checkpoint=str(ckpt))

    # A large head bias must visibly change the margins once removed.
    assert not np.allclose(
        unbiased.payload["per_niche"]["margin_weighted"].to_numpy(),
        debiased.payload["per_niche"]["margin_weighted"].to_numpy(),
    )

    # An encoder-only checkpoint must fail loudly rather than silently
    # attributing with an un-removed bias.
    encoder_only = tmp_path / "encoder.ckpt"
    torch.save({"state_dict": {"encoder.layers.0.weight": torch.zeros(2, 2)}}, encoder_only)
    with pytest.raises(ValueError, match="No graph head bias found"):
        plugin.run(dataset, context, logit_bias_checkpoint=str(encoder_only))
