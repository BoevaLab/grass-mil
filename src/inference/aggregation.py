from __future__ import annotations

from typing import Dict, List, Optional

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
            bag_attention=payload.bag_attention,
            metadata={"mode": "none"},
        )

    grouped_indices: Dict[str, List[int]] = {}
    for idx, bag_id in enumerate(payload.bag_ids):
        grouped_indices.setdefault(str(bag_id), []).append(idx)

    aggregated_ids: List[str] = []
    aggregated_logits: List[torch.Tensor] = []
    aggregated_targets: List[torch.Tensor] = []
    aggregated_attention_rows: List[Optional[torch.Tensor]] = []
    used_attention = mode == "attention_weighted" and payload.bag_attention is not None
    if payload.bag_attention is not None and len(payload.bag_attention) != len(payload.bag_ids):
        raise ValueError(
            "Mismatch between bag_ids and attention rows for aggregation: "
            f"{len(payload.bag_ids)} ids vs {len(payload.bag_attention)} attention rows."
        )

    for bag_id in sorted(grouped_indices):
        indices = grouped_indices[bag_id]
        idx = torch.tensor(indices, dtype=torch.long)
        bag_logits = payload.bag_logits.index_select(0, idx)
        weights = None
        if used_attention:
            assert payload.bag_attention is not None
            row_weights = []
            for row_idx in indices:
                attention_vals = payload.bag_attention[row_idx]
                if attention_vals is None or attention_vals.numel() == 0:
                    row_weights.append(torch.tensor(1.0, dtype=bag_logits.dtype))
                else:
                    row_weights.append(
                        torch.tensor(
                            float(attention_vals.float().mean().item()),
                            dtype=bag_logits.dtype,
                        )
                    )
            weights = torch.stack(row_weights).unsqueeze(-1).expand_as(bag_logits)

        agg_logits = _aggregate_rows(bag_logits, mode=mode, weights=weights)
        aggregated_ids.append(bag_id)
        aggregated_logits.append(agg_logits)
        if payload.bag_attention is not None:
            selected_attention = next(
                (
                    payload.bag_attention[row_idx]
                    for row_idx in indices
                    if payload.bag_attention[row_idx] is not None
                ),
                None,
            )
            aggregated_attention_rows.append(selected_attention)

        if payload.bag_targets is not None:
            bag_targets = payload.bag_targets.index_select(0, idx)
            aggregated_targets.append(bag_targets[:1])

    target_tensor = torch.cat(aggregated_targets, dim=0) if aggregated_targets else None
    aggregated_attention: Optional[List[Optional[torch.Tensor]]] = None
    if payload.bag_attention is not None:
        aggregated_attention = aggregated_attention_rows

    return AggregatedPredictionPayload(
        bag_ids=aggregated_ids,
        bag_logits=torch.cat(aggregated_logits, dim=0),
        bag_targets=target_tensor,
        bag_attention=aggregated_attention,
        metadata={"mode": mode, "used_attention": used_attention},
    )
