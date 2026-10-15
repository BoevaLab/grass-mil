"""Instance head, gated attention and logit-space attention pooling.

For a bag (unit) with instances ``i`` and classes ``c``, the instance head
gives logits ``l[i, c]``, the attention head gives scores ``a[i, c]``, and

.. math:: A_{i,c} = \\mathrm{softmax}_i(a_{i,c}), \\qquad L_c = \\sum_i A_{i,c}\\, l_{i,c}.

Pooling in logit space with one attention channel per class makes every bag
logit an exact sum of per-instance contributions ``A[i, c] l[i, c]``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import torch
from torch import nn

__all__ = ["HeadConfig", "AttentionConfig", "InstanceHead", "GatedAttention", "attention_pool", "pool_by_bag"]


@dataclass(frozen=True)
class HeadConfig:
    """Instance head: ``num_layers=1`` is linear, more layers make an MLP."""

    num_layers: int = 1
    hidden_dim: int | None = None
    dropout: float = 0.0
    act: str = "leaky_relu"

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> HeadConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model.head keys: {sorted(unknown)}")
        return cls(**cfg)


@dataclass(frozen=True)
class AttentionConfig:
    """Projected gated attention (``Linear+LeakyReLU`` projection, tanh/sigmoid gate)."""

    projection_dim: int = 32
    hidden_dim: int = 8
    dropout: float = 0.25
    init: str = "kaiming"

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> AttentionConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model.attention keys: {sorted(unknown)}")
        out = cls(**cfg)
        if out.init not in {"kaiming", "default"}:
            raise ValueError("model.attention.init must be 'kaiming' or 'default'.")
        return out


_ACT = {"leaky_relu": nn.LeakyReLU, "relu": nn.ReLU, "gelu": nn.GELU}


class InstanceHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cfg: HeadConfig) -> None:
        super().__init__()
        if cfg.num_layers < 1:
            raise ValueError("model.head.num_layers must be >= 1")
        hidden = cfg.hidden_dim or input_dim
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(cfg.num_layers - 1):
            layers += [nn.Linear(width, hidden), _ACT[cfg.act]()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            width = hidden
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    @property
    def is_linear(self) -> bool:
        return len(self.net) == 1

    @property
    def bias(self) -> torch.Tensor:
        return self.net[-1].bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedAttention(nn.Module):
    """Raw (pre-softmax) attention scores ``[n_instances, num_classes]``."""

    def __init__(self, input_dim: int, num_classes: int, cfg: AttentionConfig) -> None:
        super().__init__()
        p, h = cfg.projection_dim, cfg.hidden_dim
        self.projection = nn.Sequential(nn.Linear(input_dim, p), nn.LeakyReLU(), nn.Dropout(cfg.dropout))
        self.attention_a = nn.Sequential(nn.Linear(p, h), nn.Tanh(), nn.Dropout(cfg.dropout))
        self.attention_b = nn.Sequential(nn.Linear(p, h), nn.Sigmoid(), nn.Dropout(cfg.dropout))
        self.attention_c = nn.Linear(h, num_classes)
        if cfg.init == "kaiming":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.projection(x)
        return self.attention_c(self.attention_a(h) * self.attention_b(h))

    def space(self, x: torch.Tensor) -> torch.Tensor:
        """The low-dimensional pre-tanh activation, the attention head's own feature space.

        Computed without dropout, so it is the same in train and eval mode.
        """
        return self.attention_a[0](self.projection[1](self.projection[0](x)))


def attention_pool(logits: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool one bag: ``(bag_logits [C], attention weights [n, C])``."""
    if scores.shape != logits.shape:
        raise ValueError(
            f"Attention scores {tuple(scores.shape)} and instance logits {tuple(logits.shape)} "
            "must match: the attention needs one channel per class."
        )
    weights = torch.softmax(scores.float(), dim=0)
    return (weights * logits.float()).sum(dim=0), weights


def pool_by_bag(
    logits: torch.Tensor, scores: torch.Tensor, bag_ids: Sequence[str]
) -> tuple[list[str], torch.Tensor]:
    """Pool instances grouped by ``bag_ids``: sorted bag ids and ``[n_bags, C]`` logits."""
    groups: dict[str, list[int]] = {}
    for i, bag in enumerate(bag_ids):
        groups.setdefault(str(bag), []).append(i)
    names = sorted(groups)
    pooled = []
    for name in names:
        idx = torch.tensor(groups[name], device=logits.device)
        pooled.append(attention_pool(logits[idx], scores[idx])[0])
    return names, torch.stack(pooled) if pooled else logits.new_zeros((0, logits.shape[1]))
