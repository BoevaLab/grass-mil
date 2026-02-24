from __future__ import annotations

import torch
import pytest

from src.inference.aggregation import aggregate_group_logits
from src.inference.schemas import BatchPredictionPayload


def test_aggregate_group_logits_mean() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["r1", "r1", "r2"],
        bag_logits=torch.tensor([[1.0], [3.0], [2.0]]),
        bag_targets=torch.tensor([[1.0], [1.0], [0.0]]),
        bag_attention=[torch.tensor([0.1, 0.9]), torch.tensor([0.9, 0.1]), None],
        row_region_ids=["region_a", "region_a", "region_b"],
        row_sample_ids=["sample_x", "sample_x", "sample_y"],
    )
    out = aggregate_group_logits(payload, mode="mean")
    assert out.bag_ids == ["r1", "r2"]
    assert torch.allclose(out.bag_logits[0], torch.tensor([2.0]))


def test_aggregate_group_logits_max() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["r1", "r1", "r2"],
        bag_logits=torch.tensor([[1.0], [3.0], [2.0]]),
        bag_targets=None,
        bag_attention=None,
        row_region_ids=["region_a", "region_a", "region_b"],
        row_sample_ids=["sample_x", "sample_x", "sample_y"],
    )
    out = aggregate_group_logits(payload, mode="max")
    assert out.bag_ids == ["r1", "r2"]
    assert torch.allclose(out.bag_logits[0], torch.tensor([3.0]))


def test_aggregate_group_logits_attention_weighted_uses_instance_softmax() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["r1", "r1", "r2"],
        bag_logits=torch.tensor([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]]),
        bag_targets=None,
        bag_attention=[torch.tensor([1.0]), torch.tensor([9.0]), torch.tensor([1.0])],
        row_region_ids=["region_a", "region_a", "region_b"],
        row_sample_ids=["sample_x", "sample_x", "sample_y"],
        instance_logits=torch.tensor([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]]),
        instance_attention_logits=torch.tensor([[1.0], [9.0], [1.0]]),
        instance_patch_ids=["r1", "r1", "r2"],
        instance_region_ids=["region_a", "region_a", "region_b"],
        instance_sample_ids=["sample_x", "sample_x", "sample_y"],
    )
    out = aggregate_group_logits(payload, mode="attention_weighted")
    assert out.bag_ids == ["r1", "r2"]
    assert torch.allclose(out.bag_logits[0], torch.tensor([2.9993, 29.9933]), atol=1e-3)
    assert torch.allclose(out.bag_logits[1], torch.tensor([2.0, 20.0]), atol=1e-6)
    assert out.bag_attention is not None
    assert len(out.bag_attention) == 2
    assert out.bag_attention[0] is not None
    assert torch.allclose(out.bag_attention[0], torch.tensor([0.000335, 0.999665]), atol=1e-4)


def test_aggregate_group_logits_mean_supports_region_scope() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["p0", "p1", "p2"],
        bag_logits=torch.tensor([[1.0], [3.0], [8.0]]),
        bag_targets=None,
        bag_attention=None,
        row_region_ids=["rA", "rA", "rB"],
        row_sample_ids=["s0", "s0", "s1"],
    )
    out = aggregate_group_logits(payload, mode="mean", bag_scope="region")
    assert out.bag_ids == ["rA", "rB"]
    assert torch.allclose(out.bag_logits[0], torch.tensor([2.0]))
    assert torch.allclose(out.bag_logits[1], torch.tensor([8.0]))


def test_attention_weighted_region_scope_with_subsample_and_seed() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["p0", "p1", "p2", "p3"],
        bag_logits=torch.tensor([[0.0], [0.0], [0.0], [0.0]]),
        bag_targets=None,
        bag_attention=None,
        row_region_ids=["rA", "rA", "rB", "rB"],
        row_sample_ids=["s0", "s0", "s1", "s1"],
        instance_logits=torch.tensor([[1.0], [2.0], [10.0], [20.0]]),
        instance_attention_logits=torch.tensor([[1.0], [2.0], [1.0], [2.0]]),
        instance_patch_ids=["p0", "p1", "p2", "p3"],
        instance_region_ids=["rA", "rA", "rB", "rB"],
        instance_sample_ids=["s0", "s0", "s1", "s1"],
    )
    out = aggregate_group_logits(
        payload,
        mode="attention_weighted",
        bag_scope="region",
        subsample_fraction=0.5,
        subsample_seed=7,
    )
    assert out.bag_ids == ["rA", "rB"]
    assert out.bag_logits.shape == (2, 1)


def test_attention_weighted_subsample_fraction_zero_raises() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["r1"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        row_region_ids=["region_a"],
        row_sample_ids=["sample_x"],
        instance_logits=torch.tensor([[1.0]]),
        instance_attention_logits=torch.tensor([[1.0]]),
        instance_patch_ids=["r1"],
        instance_region_ids=["region_a"],
        instance_sample_ids=["sample_x"],
    )
    with pytest.raises(ValueError, match="subsample_fraction=0.0"):
        aggregate_group_logits(
            payload,
            mode="attention_weighted",
            subsample_fraction=0.0,
        )
