from __future__ import annotations

from typing import Dict, List, Optional

import torch

from src.inference.schemas import AggregatedPredictionPayload, BatchPredictionPayload


def _aggregate_rows(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return values.mean(dim=0, keepdim=True)
    if mode == "max":
        return values.max(dim=0, keepdim=True).values
    raise ValueError(f"Unsupported aggregation mode '{mode}'.")


def _group_indices(ids: List[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, bag_id in enumerate(ids):
        grouped.setdefault(str(bag_id), []).append(idx)
    return grouped


def _normalize_group_ids(
    ids: Optional[List[Optional[str]]], *, expected_length: int, name: str
) -> List[str]:
    if ids is None:
        raise ValueError(
            f"Aggregation requires '{name}' to regroup predictions at the requested bag scope."
        )
    if len(ids) != expected_length:
        raise ValueError(
            f"Mismatch between rows and '{name}': {expected_length} rows vs {len(ids)} ids."
        )
    normalized: List[str] = []
    for idx, value in enumerate(ids):
        if value is None:
            raise ValueError(
                f"Missing '{name}' at row {idx}; cannot regroup predictions for this scope."
            )
        normalized.append(str(value))
    return normalized


def _row_group_ids(payload: BatchPredictionPayload, *, bag_scope: str) -> List[str]:
    if bag_scope == "patch":
        return [str(v) for v in payload.bag_ids]
    if bag_scope == "region":
        return _normalize_group_ids(
            payload.row_region_ids,
            expected_length=len(payload.bag_ids),
            name="row_region_ids",
        )
    if bag_scope == "sample":
        return _normalize_group_ids(
            payload.row_sample_ids,
            expected_length=len(payload.bag_ids),
            name="row_sample_ids",
        )
    raise ValueError(
        f"Unsupported bag_scope '{bag_scope}'. Expected one of ['patch', 'region', 'sample']."
    )


def _instance_group_ids(payload: BatchPredictionPayload, *, bag_scope: str) -> List[str]:
    if payload.instance_logits is None:
        raise ValueError(
            "attention_weighted aggregation requires instance_logits in prediction payload."
        )
    n_instances = int(payload.instance_logits.shape[0])
    if bag_scope == "patch":
        patch_ids = payload.instance_patch_ids
        if patch_ids is None:
            raise ValueError(
                "attention_weighted aggregation with bag_scope='patch' requires "
                "instance_patch_ids."
            )
        if len(patch_ids) != n_instances:
            raise ValueError(
                "Mismatch between instance_logits rows and instance_patch_ids: "
                f"{n_instances} vs {len(patch_ids)}."
            )
        return [str(v) for v in patch_ids]
    if bag_scope == "region":
        return _normalize_group_ids(
            payload.instance_region_ids,
            expected_length=n_instances,
            name="instance_region_ids",
        )
    if bag_scope == "sample":
        return _normalize_group_ids(
            payload.instance_sample_ids,
            expected_length=n_instances,
            name="instance_sample_ids",
        )
    raise ValueError(
        f"Unsupported bag_scope '{bag_scope}'. Expected one of ['patch', 'region', 'sample']."
    )


def _sample_indices(
    indices: List[int], *, subsample_fraction: float, generator: Optional[torch.Generator]
) -> List[int]:
    if subsample_fraction == 1.0 or len(indices) <= 1:
        return indices
    n_keep = int(len(indices) * subsample_fraction)
    if n_keep <= 0:
        n_keep = 1
    n_keep = min(n_keep, len(indices))
    if n_keep == len(indices):
        return indices
    perm = torch.randperm(len(indices), generator=generator).tolist()
    chosen = sorted(perm[:n_keep])
    return [indices[i] for i in chosen]


def _validate_subsample_fraction(subsample_fraction: float) -> None:
    if subsample_fraction == 0.0:
        raise ValueError(
            "aggregation.subsample_fraction=0.0 selects no instances and is invalid."
        )
    if not (0.0 <= subsample_fraction <= 1.0):
        raise ValueError(
            "aggregation.subsample_fraction must be within [0, 1]. "
            f"Got {subsample_fraction}."
        )


def aggregate_group_logits(
    payload: BatchPredictionPayload,
    *,
    mode: str,
    bag_scope: str = "patch",
    subsample_fraction: float = 1.0,
    subsample_seed: Optional[int] = None,
) -> AggregatedPredictionPayload:
    """Aggregate duplicated bag rows into one logit row per bag."""
    _validate_subsample_fraction(subsample_fraction)
    if mode == "none":
        return AggregatedPredictionPayload(
            bag_ids=payload.bag_ids,
            bag_logits=payload.bag_logits,
            bag_targets=payload.bag_targets,
            bag_attention=payload.bag_attention,
            metadata={"mode": "none", "bag_scope": bag_scope},
            row_region_ids=payload.row_region_ids,
            row_sample_ids=payload.row_sample_ids,
        )

    row_group_ids = _row_group_ids(payload, bag_scope=bag_scope)
    grouped_row_indices = _group_indices(row_group_ids)

    aggregated_ids: List[str] = []
    aggregated_logits: List[torch.Tensor] = []
    aggregated_targets: List[torch.Tensor] = []
    aggregated_attention_rows: List[Optional[torch.Tensor]] = []
    aggregated_row_region_ids: List[Optional[str]] = []
    aggregated_row_sample_ids: List[Optional[str]] = []
    used_attention = mode == "attention_weighted"
    generator = None
    if subsample_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(subsample_seed))
    if payload.bag_attention is not None and len(payload.bag_attention) != len(payload.bag_ids):
        raise ValueError(
            "Mismatch between bag_ids and attention rows for aggregation: "
            f"{len(payload.bag_ids)} ids vs {len(payload.bag_attention)} attention rows."
        )

    if mode == "attention_weighted":
        if payload.instance_attention_logits is None:
            raise ValueError(
                "attention_weighted aggregation requires instance_attention_logits "
                "in prediction payload."
            )
        instance_group_ids = _instance_group_ids(payload, bag_scope=bag_scope)
        grouped_instance_indices = _group_indices(instance_group_ids)
        if payload.instance_attention_logits.shape[0] != payload.instance_logits.shape[0]:
            raise ValueError(
                "Mismatch between instance_attention_logits and instance_logits rows: "
                f"{payload.instance_attention_logits.shape[0]} vs "
                f"{payload.instance_logits.shape[0]}."
            )
        for bag_id in sorted(grouped_instance_indices):
            indices = _sample_indices(
                grouped_instance_indices[bag_id],
                subsample_fraction=subsample_fraction,
                generator=generator,
            )
            idx = torch.tensor(indices, dtype=torch.long)
            instance_logits = payload.instance_logits.index_select(0, idx)
            attn_logits = payload.instance_attention_logits.index_select(0, idx).reshape(-1)
            attn_scores = torch.softmax(attn_logits, dim=0)
            agg_logits = (instance_logits * attn_scores.unsqueeze(-1)).sum(
                dim=0, keepdim=True
            )
            aggregated_ids.append(bag_id)
            aggregated_logits.append(agg_logits)
            aggregated_attention_rows.append(attn_scores.detach().cpu())

            row_indices = grouped_row_indices.get(bag_id, [])
            if row_indices:
                aggregated_row_region_ids.append(
                    payload.row_region_ids[row_indices[0]]
                    if payload.row_region_ids is not None
                    else None
                )
                aggregated_row_sample_ids.append(
                    payload.row_sample_ids[row_indices[0]]
                    if payload.row_sample_ids is not None
                    else None
                )
                if payload.bag_targets is not None:
                    target_idx = torch.tensor(row_indices, dtype=torch.long)
                    bag_targets = payload.bag_targets.index_select(0, target_idx)
                    aggregated_targets.append(bag_targets[:1])
            else:
                aggregated_row_region_ids.append(
                    bag_id if bag_scope == "region" else None
                )
                aggregated_row_sample_ids.append(
                    bag_id if bag_scope == "sample" else None
                )
    else:
        for bag_id in sorted(grouped_row_indices):
            indices = grouped_row_indices[bag_id]
            idx = torch.tensor(indices, dtype=torch.long)
            bag_logits = payload.bag_logits.index_select(0, idx)
            agg_logits = _aggregate_rows(bag_logits, mode=mode)
            aggregated_ids.append(bag_id)
            aggregated_logits.append(agg_logits)
            aggregated_row_region_ids.append(
                payload.row_region_ids[indices[0]]
                if payload.row_region_ids is not None
                else None
            )
            aggregated_row_sample_ids.append(
                payload.row_sample_ids[indices[0]]
                if payload.row_sample_ids is not None
                else None
            )
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
    if aggregated_attention_rows:
        aggregated_attention = aggregated_attention_rows

    return AggregatedPredictionPayload(
        bag_ids=aggregated_ids,
        bag_logits=torch.cat(aggregated_logits, dim=0),
        bag_targets=target_tensor,
        bag_attention=aggregated_attention,
        metadata={
            "mode": mode,
            "used_attention": used_attention,
            "bag_scope": bag_scope,
            "subsample_fraction": float(subsample_fraction),
            "subsample_seed": subsample_seed,
        },
        row_region_ids=aggregated_row_region_ids,
        row_sample_ids=aggregated_row_sample_ids,
    )
