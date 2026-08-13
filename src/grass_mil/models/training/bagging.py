from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from grass_mil.contracts import validate_attention_width

try:
    from torch_geometric.data import Data
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for training bagging helpers") from exc


def extract_bag_ids(
    batch: Data,
    *,
    bag_key: str,
    bag_fallback_key: str,
) -> List[str]:
    bag_values = _to_list(getattr(batch, bag_key, None))
    fallback_values = _to_list(getattr(batch, bag_fallback_key, None))
    if bag_values is None and fallback_values is None:
        raise ValueError(
            f"Batch is missing both '{bag_key}' and '{bag_fallback_key}' for bagging."
        )
    if bag_values is None:
        return [str(v) for v in fallback_values]
    if fallback_values is None:
        return [str(v) for v in bag_values]
    if len(bag_values) != len(fallback_values):
        raise ValueError(
            f"Batch key lengths mismatch for bagging: '{bag_key}' has {len(bag_values)} "
            f"values while fallback '{bag_fallback_key}' has {len(fallback_values)}."
        )
    region_values = _to_list(getattr(batch, "region_id", None))
    if region_values is not None and len(region_values) != len(bag_values):
        if len(region_values) == 1:
            region_values = region_values * len(bag_values)
        else:
            raise ValueError(
                "Batch key lengths mismatch for bagging: 'region_id' has "
                f"{len(region_values)} values while '{bag_key}' has {len(bag_values)}."
            )

    resolved: List[str] = []
    for idx, (primary, fallback) in enumerate(zip(bag_values, fallback_values)):
        region_value = region_values[idx] if region_values is not None else None
        use_fallback = _is_missing_value(primary)
        if use_fallback and _is_missing_value(fallback):
            raise ValueError(
                f"Both '{bag_key}' and '{bag_fallback_key}' are missing for at least one instance."
            )
        if bag_key == "patch_id":
            if use_fallback:
                if not _is_missing_value(region_value):
                    resolved.append(f"{str(fallback)}::{str(region_value)}")
                else:
                    resolved.append(str(fallback))
                continue
            if not _is_missing_value(fallback):
                if not _is_missing_value(region_value):
                    resolved.append(f"{str(fallback)}::{str(region_value)}::{str(primary)}")
                else:
                    resolved.append(f"{str(fallback)}::{str(primary)}")
                continue
            resolved.append(str(primary))
            continue
        if use_fallback:
            resolved.append(str(fallback))
            continue
        # Namespacing primary bag ids by fallback ids avoids accidental cross-sample
        # merges when primary ids (for example region_id) are only locally unique.
        if bag_key != bag_fallback_key and not _is_missing_value(fallback):
            resolved.append(f"{str(fallback)}::{str(primary)}")
            continue
        resolved.append(str(primary))
    return resolved


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
    """Pool instance logits into bag logits with gated attention.

    Attention is normalised over the instances of a bag, **independently per
    class**:

    .. math:: A_{i,c} = \\mathrm{softmax}_i(\\alpha_{i,c}), \\quad
              L_c = \\sum_{i \\in B} A_{i,c}\\, \\ell_{i,c}

    Two attention widths are legal. ``n_classes == num_classes`` is the
    published per-class form, and is what makes the attribution identity
    ``sum_i A[i,c] * l[i,c] == L_c`` exact. ``n_classes == 1`` keeps a single
    shared attention channel broadcast across the class logits.

    Returns:
        Bag logits, bag ids, per-bag attention of shape ``[n_instances,
        n_attention_classes]``, and the selected instance indices per bag.
    """
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
        if attn_logits.dim() == 1:
            attn_logits = attn_logits.unsqueeze(-1)
        validate_attention_width(
            attention_width=int(attn_logits.size(-1)),
            num_classes=int(sub_logits.size(-1)),
        )
        # Softmax down the instance axis, per class column.
        attn_scores = torch.softmax(attn_logits, dim=0)
        weighted_logits = (sub_logits * attn_scores).sum(dim=0, keepdim=True)
        bag_logits.append(weighted_logits)
        bag_ids.append(bag_id)
        bag_indices.append(selected)
        bag_attention[bag_id] = attn_scores
    return torch.cat(bag_logits, dim=0), bag_ids, bag_attention, bag_indices


def _find_label_index_by_name(label_names: Any, target_name: str) -> int:
    if isinstance(label_names, (list, tuple)):
        if target_name in label_names:
            return int(label_names.index(target_name))
    raise ValueError(f"Target '{target_name}' not found in graph_label_names={label_names}.")


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
    bag_targets: List[torch.Tensor] = []
    bag_weights = None
    bag_weight_rows: List[torch.Tensor] = []
    for idxs in bag_indices:
        bag_y = graph_y[idxs]
        ref_y = bag_y[:1]
        if (
            bag_y.shape[0] > 1
            and not torch.isclose(bag_y, ref_y.expand_as(bag_y), atol=0.0, rtol=0.0).all()
        ):
            raise ValueError(
                "Inconsistent graph_y values detected within the same bag. "
                "All instances in a bag must share one graph target."
            )
        bag_targets.append(ref_y)
        if graph_w is not None:
            bag_w = graph_w[idxs]
            ref_w = bag_w[:1]
            if (
                bag_w.shape[0] > 1
                and not torch.isclose(bag_w, ref_w.expand_as(bag_w), atol=0.0, rtol=0.0).all()
            ):
                raise ValueError("Inconsistent graph_w values detected within the same bag.")
            bag_weight_rows.append(ref_w)
    bag_targets_t = torch.cat(bag_targets, dim=0)
    if graph_w is not None:
        bag_weights = torch.cat(bag_weight_rows, dim=0)
    return bag_targets_t, bag_weights


def _to_list(values: Any) -> Optional[List[Any]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip().lower()
        return text in {"", "nan", "none", "null"}
    if isinstance(value, float):
        return value != value
    return False
