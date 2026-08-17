from __future__ import annotations

from typing import Any, Dict, Optional

import torch


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
    if target_type == "categorical":
        if bag_logits.ndim != 2:
            raise ValueError(
                "Categorical task expects logits with shape [N, C] for cross-entropy."
            )
        targets = bag_targets.long().view(-1)
        return loss_fn(bag_logits, targets, bag_weights)
    if target_type == "regression":
        return loss_fn(bag_logits, bag_targets, bag_weights)
    if bag_targets.ndim != 2 or bag_targets.shape[1] < 2:
        raise ValueError("Survival task expects at least two target columns [time, event].")
    return loss_fn(
        bag_logits.squeeze(-1),
        bag_targets[:, 0].float(),
        bag_targets[:, 1].float(),
    )


def compute_binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == target.float()).float().mean()


def compute_categorical_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("Categorical accuracy expects logits with shape [N, C].")
    pred = torch.argmax(logits, dim=1)
    true = target.long().view(-1)
    return (pred == true).float().mean()
