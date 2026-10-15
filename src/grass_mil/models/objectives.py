"""Bag-level objectives and metrics for the three target types.

* ``classification``: weighted cross-entropy over ``num_classes`` logits; AUROC
  (macro one-vs-rest beyond two classes), balanced accuracy and accuracy.
* ``regression``: mean squared error or Huber loss per target column; RMSE,
  MAE, R² and Pearson correlation (averaged over columns).
* ``survival``: a Cox partial-likelihood ranking loss on one risk logit,
  computed over all bags of an optimisation step; Harrell's concordance index.
  Targets are ``[time, event]``.

Every bag may carry a sample weight; class weights apply to classification.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "TARGET_TYPES",
    "output_dim",
    "Objective",
    "region_metrics",
    "concordance_index",
    "class_weights_from_labels",
]

TARGET_TYPES = ("classification", "regression", "survival")


def output_dim(task: Mapping[str, Any]) -> int:
    """Logits per instance for a task config."""
    kind = task["target_type"]
    if kind == "classification":
        n = int(task.get("num_classes", 2))
        if n < 2:
            raise ValueError("task.num_classes must be >= 2 (one logit per class).")
        return n
    if kind == "regression":
        return len(task.get("target_columns") or [None])
    if kind == "survival":
        return 1
    raise ValueError(f"task.target_type must be one of {TARGET_TYPES}, got '{kind}'.")


class Objective:
    """The loss of a stack of bag logits against their targets."""

    def __init__(self, task: Mapping[str, Any], class_weight: torch.Tensor | None = None) -> None:
        self.kind = task["target_type"]
        if self.kind not in TARGET_TYPES:
            raise ValueError(f"task.target_type must be one of {TARGET_TYPES}")
        self.class_weight = class_weight
        self.loss_name = str(task.get("loss", "mse"))
        self.huber_delta = float(task.get("huber_delta", 1.0))
        self.cox_regularization = float(task.get("cox_regularization", 0.0))
        if self.kind == "regression" and self.loss_name not in {"mse", "huber"}:
            raise ValueError("task.loss must be 'mse' or 'huber' for regression.")
        if self.kind == "survival" and len(task.get("target_columns") or []) != 2:
            raise ValueError("Survival tasks need task.target_columns: [time, event].")

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = logits.float()
        if self.kind == "classification":
            weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
            per_bag = F.cross_entropy(
                logits, targets[:, 0].long(), weight=weight, reduction="none"
            )
        elif self.kind == "regression":
            if self.loss_name == "huber":
                per_bag = F.huber_loss(
                    logits, targets, delta=self.huber_delta, reduction="none"
                ).mean(1)
            else:
                per_bag = F.mse_loss(logits, targets, reduction="none").mean(1)
        else:
            return cox_loss(
                logits[:, 0], targets[:, 0], targets[:, 1], weights, self.cox_regularization
            )
        if weights is not None:
            per_bag = per_bag * weights.view(-1).to(per_bag.device)
        return per_bag.mean()


def cox_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    weights: torch.Tensor | None = None,
    regularization: float = 0.0,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), averaged over events.

    ``regularization`` adds an L2 penalty on the risk scores, which keeps them
    centred (the partial likelihood is invariant to a shared shift).
    """
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order].float()
    w = (
        torch.ones_like(risk)
        if weights is None
        else weights.view(-1)[order].float().to(risk.device)
    )
    # Cumulative log-sum-exp over the risk set (everyone with time >= t_i).
    log_risk_set = torch.logcumsumexp(risk + torch.log(w), dim=0)
    n_events = (event * w).sum()
    if n_events <= 0:
        # No observed event in this step: nothing to rank. Keep the graph so the
        # optimiser step is a well-defined no-op.
        return (risk * 0.0).sum()
    loss = -((risk - log_risk_set) * event * w).sum() / n_events
    return loss + regularization * (risk**2).mean()


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse class frequency ``n / (k * count_c)``, normalised to mean 1 (absent classes get 1)."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    weights = np.ones(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C: the fraction of comparable pairs ordered by risk (ties count half)."""
    risk, time, event = (np.asarray(a, dtype=float).ravel() for a in (risk, time, event))
    comparable = (event[:, None] > 0) & (time[:, None] < time[None, :])
    n = comparable.sum()
    if n == 0:
        return float("nan")
    higher = risk[:, None] > risk[None, :]
    tied = risk[:, None] == risk[None, :]
    return float(((comparable & higher).sum() + 0.5 * (comparable & tied).sum()) / n)


def region_metrics(kind: str, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Metrics of bag (region) logits; ``loss`` is the unweighted objective."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    logits = logits.detach().float().cpu()
    targets = targets.detach().float().cpu()
    out: dict[str, float] = {}
    if kind == "classification":
        y = targets[:, 0].long().numpy()
        out["loss"] = float(F.cross_entropy(logits, torch.from_numpy(y)))
        prob = torch.softmax(logits, dim=1).numpy()
        pred = prob.argmax(1)
        out["acc"] = float((pred == y).mean())
        # Undefined on a single-class set: logged as NaN, which checkpoint
        # selection treats as "no improvement".
        out["bal_acc"] = out["auroc"] = float("nan")
        if len(np.unique(y)) > 1:
            out["bal_acc"] = float(balanced_accuracy_score(y, pred))
            try:
                out["auroc"] = float(
                    roc_auc_score(y, prob[:, 1])
                    if prob.shape[1] == 2
                    else roc_auc_score(
                        y,
                        prob,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(prob.shape[1])),
                    )
                )
            except ValueError:
                pass
    elif kind == "regression":
        err = logits - targets
        out["loss"] = float((err**2).mean())
        out["rmse"] = math.sqrt(out["loss"])
        out["mae"] = float(err.abs().mean())
        ss_tot = ((targets - targets.mean(0)) ** 2).sum(0)
        out["r2"] = out["pearson"] = float("nan")
        if bool((ss_tot > 0).all()):
            out["r2"] = float((1 - (err**2).sum(0) / ss_tot).mean())
            out["pearson"] = float(
                np.mean(
                    [
                        np.corrcoef(logits[:, j], targets[:, j])[0, 1]
                        for j in range(targets.shape[1])
                    ]
                )
            )
    else:
        out["loss"] = float(cox_loss(logits[:, 0], targets[:, 0], targets[:, 1]))
        out["c_index"] = concordance_index(
            logits[:, 0].numpy(), targets[:, 0].numpy(), targets[:, 1].numpy()
        )
    return out
