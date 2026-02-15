from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _reshape_sample_weight(
    sample_weight: Optional[torch.Tensor],
    target_shape: torch.Size,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if sample_weight is None:
        return None
    w = sample_weight.to(device)
    if w.ndim == 1 and len(target_shape) > 1:
        w = w.unsqueeze(-1)
    return w


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weight = class_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(
                "WeightedCrossEntropyLoss expects logits with shape [N, C]"
            )
        target = target.long().view(-1)
        class_weight = (
            self.class_weight.to(logits.device)
            if self.class_weight is not None
            else None
        )
        loss = nn.functional.cross_entropy(
            logits, target, weight=class_weight, reduction="none"
        )
        if sample_weight is not None:
            sw = sample_weight.view(-1).to(loss.device)
            loss = loss * sw
        return loss.mean()


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"BCE logits and target must have identical shape, got {logits.shape} and {target.shape}"
            )
        pos_weight = (
            self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=pos_weight, reduction="none"
        )
        sw = _reshape_sample_weight(sample_weight, target.shape, logits.device)
        if sw is not None:
            loss = loss * sw
        return loss.mean()


class WeightedMSELoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = (pred - target) ** 2
        sw = _reshape_sample_weight(sample_weight, target.shape, pred.device)
        if sw is not None:
            loss = loss * sw
        return loss.mean()


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = nn.functional.huber_loss(
            pred, target, delta=self.delta, reduction="none"
        )
        sw = _reshape_sample_weight(sample_weight, target.shape, pred.device)
        if sw is not None:
            loss = loss * sw
        return loss.mean()


class CoxSGDLoss(nn.Module):
    """Pairwise Cox-style objective with optional random top-n pair pruning."""

    def __init__(self, top_n: int = 2, regularizer_weight: float = 0.05):
        super().__init__()
        self.top_n = top_n
        self.regularizer_weight = regularizer_weight

    def forward(
        self, y_pred: torch.Tensor, length: torch.Tensor, event: torch.Tensor
    ) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        length = length.view(-1)
        event = event.view(-1)
        if not (y_pred.shape[0] == length.shape[0] == event.shape[0]):
            raise ValueError("y_pred, length, and event must have same first dimension")

        n_samples = y_pred.shape[0]
        pair_mat = (length.view(1, -1) - length.view(-1, 1) > 0) * event.view(-1, 1)

        if self.top_n > 0 and n_samples > 1:
            top_k = min(int(self.top_n), n_samples - 1)
            if top_k <= 0:
                top_k = 1
            noise = 1 + torch.rand_like(pair_mat.float())
            p_with_rand = pair_mat.float() * noise
            rand_thr_ind = torch.argsort(p_with_rand, dim=1)[:, -(top_k + 1)]
            rand_thr = p_with_rand[torch.arange(n_samples), rand_thr_ind].view(-1, 1)
            pair_mat = pair_mat * (p_with_rand > rand_thr)

        valid_sample_is = torch.nonzero(pair_mat.sum(1)).flatten()
        if valid_sample_is.numel() == 0:
            return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

        pair_mat = pair_mat.clone()
        pair_mat[(valid_sample_is, valid_sample_is)] = 1

        score_diff = y_pred.view(1, -1) - y_pred.view(-1, 1)
        row_max = torch.max(score_diff, dim=1, keepdim=True).values
        exp_terms = torch.exp(score_diff - row_max) * pair_mat
        loss = row_max[:, 0][valid_sample_is] + torch.log(
            exp_terms.sum(1)[valid_sample_is]
        )
        regularizer = torch.abs(pair_mat.sum(0) * y_pred).sum()
        return loss.sum() + self.regularizer_weight * regularizer
