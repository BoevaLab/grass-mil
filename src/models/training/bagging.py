from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

try:
    from torch_geometric.data import Data
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for training bagging helpers"
    ) from exc


def extract_bag_ids(
    batch: Data,
    *,
    bag_key: str,
    bag_fallback_key: str,
) -> List[str]:
    bag_values = getattr(batch, bag_key, None)
    if bag_values is None:
        bag_values = getattr(batch, bag_fallback_key, None)
    if bag_values is None:
        raise ValueError(
            f"Batch is missing both '{bag_key}' and '{bag_fallback_key}' for bagging."
        )
    if isinstance(bag_values, torch.Tensor):
        bag_values = bag_values.tolist()
    if isinstance(bag_values, (str, bytes)):
        bag_values = [bag_values]
    return [str(v) for v in bag_values]


def group_instance_indices_by_bag(bag_ids: Sequence[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, bag_id in enumerate(bag_ids):
        grouped.setdefault(bag_id, []).append(idx)
    return grouped


def maybe_sample_indices(
    indices: List[int],
    *,
    max_instances_per_bag: int,
    instance_sampling: str,
) -> List[int]:
    if max_instances_per_bag <= 0 or len(indices) <= max_instances_per_bag:
        return indices
    if instance_sampling == "all":
        return indices[:max_instances_per_bag]
    rand_perm = torch.randperm(len(indices))
    keep = rand_perm[:max_instances_per_bag].tolist()
    keep.sort()
    return [indices[i] for i in keep]


def aggregate_bag_logits_mean(
    logits: torch.Tensor,
    bag_groups: Dict[str, List[int]],
    *,
    max_instances_per_bag: int,
    instance_sampling: str,
) -> tuple[torch.Tensor, List[str], List[List[int]]]:
    bag_logits = []
    bag_ids = []
    bag_indices = []
    for bag_id, indices in bag_groups.items():
        selected = maybe_sample_indices(
            indices,
            max_instances_per_bag=max_instances_per_bag,
            instance_sampling=instance_sampling,
        )
        bag_ids.append(bag_id)
        bag_indices.append(selected)
        bag_logits.append(logits[selected].mean(dim=0, keepdim=True))
    return torch.cat(bag_logits, dim=0), bag_ids, bag_indices


def aggregate_bag_logits_attention(
    *,
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    attention: torch.nn.Module,
    bag_groups: Dict[str, List[int]],
    max_instances_per_bag: int,
    instance_sampling: str,
) -> tuple[torch.Tensor, List[str], Dict[str, torch.Tensor], List[List[int]]]:
    bag_logits = []
    bag_ids = []
    bag_attention = {}
    bag_indices = []
    for bag_id, indices in bag_groups.items():
        selected = maybe_sample_indices(
            indices,
            max_instances_per_bag=max_instances_per_bag,
            instance_sampling=instance_sampling,
        )
        sub_emb = embeddings[selected]
        sub_logits = logits[selected]
        attn_logits, _ = attention(sub_emb)
        attn_scores = torch.softmax(attn_logits.squeeze(-1), dim=0)
        weighted_logits = (sub_logits * attn_scores.unsqueeze(-1)).sum(
            dim=0, keepdim=True
        )
        bag_logits.append(weighted_logits)
        bag_ids.append(bag_id)
        bag_indices.append(selected)
        bag_attention[bag_id] = attn_scores
    return torch.cat(bag_logits, dim=0), bag_ids, bag_attention, bag_indices


def _find_label_index_by_name(label_names: Any, target_name: str) -> int:
    if isinstance(label_names, (list, tuple)):
        if target_name in label_names:
            return int(label_names.index(target_name))
    raise ValueError(
        f"Target '{target_name}' not found in graph_label_names={label_names}."
    )


def select_target_columns(
    graph_y: torch.Tensor,
    batch: Data,
    target_columns: Optional[Sequence[str]],
) -> torch.Tensor:
    if graph_y.ndim == 1:
        graph_y = graph_y.unsqueeze(-1)
    if not target_columns:
        return graph_y
    label_names = getattr(batch, "graph_label_names", None)
    # PyG collation may produce a list-of-lists (one per graph). Normalize to one
    # shared label-name list, but do not collapse plain list[str] labels.
    if (
        isinstance(label_names, list)
        and len(label_names) == graph_y.shape[0]
        and label_names
        and isinstance(label_names[0], (list, tuple))
    ):
        label_names = label_names[0]
    indices = [_find_label_index_by_name(label_names, name) for name in target_columns]
    return graph_y[:, indices]


def gather_bag_targets(
    *,
    batch: Data,
    bag_indices: List[List[int]],
    target_columns: Optional[Sequence[str]],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(batch, "graph_y"):
        raise ValueError("Batch is missing graph_y required for supervised training.")
    graph_y = select_target_columns(batch.graph_y.float(), batch, target_columns)
    graph_w = batch.graph_w.float() if hasattr(batch, "graph_w") else None
    bag_targets = torch.cat([graph_y[idxs[:1]] for idxs in bag_indices], dim=0)
    bag_weights = None
    if graph_w is not None:
        bag_weights = torch.cat([graph_w[idxs[:1]] for idxs in bag_indices], dim=0)
    return bag_targets, bag_weights
