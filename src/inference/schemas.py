from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class BatchPredictionPayload:
    """Normalized prediction payload emitted by collector utilities."""

    bag_ids: List[str]
    bag_logits: torch.Tensor
    bag_targets: Optional[torch.Tensor]
    bag_attention: Optional[List[Optional[torch.Tensor]]]


@dataclass(frozen=True)
class AggregatedPredictionPayload:
    """Prediction payload after optional group-level aggregation."""

    bag_ids: List[str]
    bag_logits: torch.Tensor
    bag_targets: Optional[torch.Tensor]
    bag_attention: Optional[List[Optional[torch.Tensor]]]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EmbeddingPayload:
    """Embeddings aligned to bag-level ids."""

    bag_ids: List[str]
    graph_embeddings: torch.Tensor
    node_embeddings: Optional[torch.Tensor]
    node_bag_ids: Optional[List[str]]
