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
    row_region_ids: Optional[List[Optional[str]]] = None
    row_sample_ids: Optional[List[Optional[str]]] = None
    instance_logits: Optional[torch.Tensor] = None
    instance_attention_logits: Optional[torch.Tensor] = None
    instance_patch_ids: Optional[List[str]] = None
    instance_region_ids: Optional[List[Optional[str]]] = None
    instance_sample_ids: Optional[List[Optional[str]]] = None


@dataclass(frozen=True)
class AggregatedPredictionPayload:
    """Prediction payload after optional group-level aggregation."""

    bag_ids: List[str]
    bag_logits: torch.Tensor
    bag_targets: Optional[torch.Tensor]
    bag_attention: Optional[List[Optional[torch.Tensor]]]
    metadata: Dict[str, Any]
    row_region_ids: Optional[List[Optional[str]]] = None
    row_sample_ids: Optional[List[Optional[str]]] = None


@dataclass(frozen=True)
class EmbeddingPayload:
    """Embeddings aligned to bag-level ids."""

    bag_ids: List[str]
    graph_embeddings: torch.Tensor
    node_embeddings: Optional[torch.Tensor]
    node_bag_ids: Optional[List[str]]


@dataclass(frozen=True)
class PreforwardSubsamplingDecision:
    """Decision record for preforward subsampling orchestration."""

    requested: bool
    effective: bool
    fraction: float
    seed: Optional[int]
    reason: str
    strategy_name: Optional[str] = None
    runtime_enabled: Optional[bool] = None
