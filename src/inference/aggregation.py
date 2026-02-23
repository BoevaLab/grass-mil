from __future__ import annotations

from typing import Dict, List

import torch

from src.inference.schemas import AggregatedPredictionPayload, BatchPredictionPayload


def _aggregate_rows(
    values: torch.Tensor, mode: str, weights: torch.Tensor | None
) -> torch.Tensor:
    if mode == "mean":
        return values.mean(dim=0, keepdim=True)
    if mode == "max":
        return values.max(dim=0, keepdim=True).values
    if mode == "attention_weighted":
        if weights is None:
            return values.mean(dim=0, keepdim=True)
        denom = weights.sum().clamp(min=1.0e-8)
        return (values * weights).sum(dim=0, keepdim=True) / denom
    raise ValueError(f"Unsupported aggregation mode '{mode}'.")


def aggregate_group_logits(
    payload: BatchPredictionPayload, *, mode: str
) -> AggregatedPredictionPayload:
    """Aggregate duplicated bag rows into one logit row per bag."""
    if mode == "none":
        return AggregatedPredictionPayload(
            bag_ids=payload.bag_ids,
            bag_logits=payload.bag_logits,
            bag_targets=payload.bag_targets,
            metadata={"mode": "none"},
        )

    grouped_indices: Dict[str, List[int]] = {}
    for idx, bag_id in enumerate(payload.bag_ids):
        grouped_indices.setdefault(str(bag_id), []).append(idx)

    aggregated_ids: List[str] = []
    aggregated_logits: List[torch.Tensor] = []
    aggregated_targets: List[torch.Tensor] = []
    used_attention = mode == "attention_weighted" and payload.bag_attention is not None

    for bag_id in sorted(grouped_indices):
        indices = grouped_indices[bag_id]
        idx = torch.tensor(indices, dtype=torch.long)
        bag_logits = payload.bag_logits.index_select(0, idx)
        weights = None
        if used_attention:
            attention_vals = payload.bag_attention.get(bag_id)
            if attention_vals is not None and attention_vals.numel() > 0:
                scalar_weight = float(attention_vals.float().mean().item())
                weights = torch.full_like(bag_logits, scalar_weight)

        agg_logits = _aggregate_rows(bag_logits, mode=mode, weights=weights)
        aggregated_ids.append(bag_id)
        aggregated_logits.append(agg_logits)

        if payload.bag_targets is not None:
            bag_targets = payload.bag_targets.index_select(0, idx)
            aggregated_targets.append(bag_targets[:1])

    target_tensor = torch.cat(aggregated_targets, dim=0) if aggregated_targets else None
    return AggregatedPredictionPayload(
        bag_ids=aggregated_ids,
        bag_logits=torch.cat(aggregated_logits, dim=0),
        bag_targets=target_tensor,
        metadata={"mode": mode, "used_attention": used_attention},
    )
