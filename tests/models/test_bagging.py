"""Bag-pooling contracts, including the attribution identity.

The published MIL head pools instance logits with per-class attention:

    A[i,c] = softmax_i(alpha[i,c]),    L_c = sum_i A[i,c] * l[i,c]

The exactness of that sum is what the whole attribution suite rests on, so it
is asserted directly here rather than inferred from downstream behaviour.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from grass_mil.contracts import validate_attention_width
from grass_mil.models.training.bagging import aggregate_bag_logits_attention


class _FixedAttention(nn.Module):
    """Attention returning preset logits, so pooling is checked in isolation."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.attention_c = nn.Linear(1, logits.shape[1])

    def forward(self, x):
        return self.logits[: x.shape[0]], x


def _pool(*, logits: torch.Tensor, attn_logits: torch.Tensor):
    return aggregate_bag_logits_attention(
        logits=logits,
        embeddings=torch.zeros((logits.shape[0], 1)),
        attention=_FixedAttention(attn_logits),
        bag_groups={"bag": list(range(logits.shape[0]))},
        max_instances_per_bag=0,
        instance_sampling="all",
    )


def test_bag_logit_is_the_attention_weighted_sum_of_instance_logits() -> None:
    """The identity sum_i A[i,c] * l[i,c] == L_c, asserted exactly."""
    torch.manual_seed(0)
    logits = torch.randn(6, 2)
    attn_logits = torch.randn(6, 2)

    bag_logits, bag_ids, bag_attention, _ = _pool(logits=logits, attn_logits=attn_logits)

    assert bag_ids == ["bag"]
    attention = bag_attention["bag"]
    assert attention.shape == (6, 2)

    expected = (logits * attention).sum(dim=0, keepdim=True)
    torch.testing.assert_close(bag_logits, expected, rtol=0, atol=1e-12)


def test_attention_is_normalised_per_class_over_instances() -> None:
    torch.manual_seed(1)
    _, _, bag_attention, _ = _pool(logits=torch.randn(5, 3), attn_logits=torch.randn(5, 3))

    attention = bag_attention["bag"]
    # Each class column is a distribution over instances; columns are independent.
    torch.testing.assert_close(attention.sum(dim=0), torch.ones(3), rtol=0, atol=1e-6)


def test_per_class_attention_can_select_different_instances_per_class() -> None:
    """The point of per-class attention: class 0 and class 1 may disagree."""
    logits = torch.tensor([[1.0, 100.0], [2.0, 200.0]])
    # Class 0 concentrates on instance 0, class 1 on instance 1.
    attn_logits = torch.tensor([[20.0, 0.0], [0.0, 20.0]])

    bag_logits, _, _, _ = _pool(logits=logits, attn_logits=attn_logits)

    assert bag_logits[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert bag_logits[0, 1] == pytest.approx(200.0, abs=1e-2)


def test_pooling_rejects_a_shared_attention_channel() -> None:
    """One channel broadcast across classes cannot express opposing evidence.

    A niche that pushes toward one class and away from the other needs two
    independent attention distributions, so a binary head requires two
    channels rather than one.
    """
    with pytest.raises(ValueError, match="Attention emits 1 channel"):
        _pool(logits=torch.randn(4, 2), attn_logits=torch.randn(4, 1))


def test_pooling_rejects_attention_width_that_does_not_match_the_head() -> None:
    with pytest.raises(ValueError, match="Attention emits 3 channel"):
        _pool(logits=torch.randn(4, 2), attn_logits=torch.randn(4, 3))


def test_validate_attention_width_accepts_one_channel_per_class() -> None:
    validate_attention_width(attention_width=4, num_classes=4)
    # A scalar head (regression, Cox log-hazard) is one class, one channel.
    validate_attention_width(attention_width=1, num_classes=1)


@pytest.mark.parametrize("width", [0, 1, 2, 5])
def test_validate_attention_width_rejects_other_widths(width: int) -> None:
    with pytest.raises(ValueError):
        validate_attention_width(attention_width=width, num_classes=4)
