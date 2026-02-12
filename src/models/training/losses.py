from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def compute_supervised_loss(
    *,
    loss_fn: torch.nn.Module,
    task_cfg: Dict[str, Any],
    bag_logits: torch.Tensor,
    bag_targets: torch.Tensor,
    bag_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    target_type = task_cfg["target_type"]
    if target_type == "binary":
        if bag_targets.ndim == 1:
            bag_targets = bag_targets.unsqueeze(-1)
        return loss_fn(bag_logits, bag_targets, bag_weights)
    if target_type == "regression":
        return loss_fn(bag_logits, bag_targets, bag_weights)
    if bag_targets.ndim != 2 or bag_targets.shape[1] < 2:
        raise ValueError(
            "Survival task expects at least two target columns [time, event]."
        )
    return loss_fn(
        bag_logits.squeeze(-1),
        bag_targets[:, 0].float(),
        bag_targets[:, 1].float(),
    )


def build_mil_aux_targets(
    *,
    bag_targets: torch.Tensor,
    bag_indices: List[List[int]],
    bag_ids: Sequence[str],
    bag_attention: Dict[str, torch.Tensor],
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    aux_targets: List[torch.Tensor] = []
    aux_weights: List[torch.Tensor] = []
    for bag_pos, (bag_id, idxs) in enumerate(zip(bag_ids, bag_indices)):
        bag_target = bag_targets[bag_pos : bag_pos + 1]
        repeated_target = bag_target.repeat(len(idxs), 1)
        if target_mode == "graph_label":
            aux_targets.append(repeated_target)
            aux_weights.append(torch.ones_like(repeated_target))
            continue
        if target_mode == "attention_shaped_ti":
            attn = bag_attention[bag_id].reshape(-1, 1).to(repeated_target.device)
            target = 0.5 + (repeated_target - 0.5) * attn
            aux_targets.append(target)
            aux_weights.append(attn)
            continue
        raise ValueError(
            "node_aux.target_mode must be one of ['graph_label', 'attention_shaped_ti']"
        )
    return torch.cat(aux_targets, dim=0), torch.cat(aux_weights, dim=0)


def gather_instance_logits(
    patch_logits: torch.Tensor, bag_indices: List[List[int]]
) -> torch.Tensor:
    return torch.cat([patch_logits[idxs] for idxs in bag_indices], dim=0)


def compute_aux_node_loss(
    *,
    aux_logits: torch.Tensor,
    aux_targets: torch.Tensor,
    aux_weights: Optional[torch.Tensor],
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode not in {"bce", "weighted_bce"}:
        raise ValueError("node_aux.loss_mode must be one of ['bce', 'weighted_bce']")
    loss = F.binary_cross_entropy_with_logits(
        aux_logits, aux_targets.float(), reduction="none"
    )
    if loss_mode == "weighted_bce":
        if aux_weights is None:
            raise ValueError("weighted_bce requires non-null aux_weights.")
        loss = loss * aux_weights
    return loss.mean()


def compute_entropy_regularization(
    values: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    probs = values.float().clamp(min=eps, max=1.0)
    return -(probs * torch.log(probs + eps)).mean()


def compute_binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == target.float()).float().mean()
