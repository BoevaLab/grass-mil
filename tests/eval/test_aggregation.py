from __future__ import annotations

import torch

from src.inference.aggregation import aggregate_group_logits
from src.inference.schemas import BatchPredictionPayload


def test_aggregate_group_logits_mean() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["r1", "r1", "r2"],
        bag_logits=torch.tensor([[1.0], [3.0], [2.0]]),
        bag_targets=torch.tensor([[1.0], [1.0], [0.0]]),
        bag_attention={"r1": torch.tensor([0.1, 0.9])},
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
    )
    out = aggregate_group_logits(payload, mode="max")
    assert out.bag_ids == ["r1", "r2"]
    assert torch.allclose(out.bag_logits[0], torch.tensor([3.0]))
