from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig

from src.inference.schemas import BatchPredictionPayload


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return float("nan")
    return float(num / den)


def _binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.long().view(-1)
    scores = scores.float().view(-1)
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_idx = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(sorted_idx, dtype=torch.float)
    ranks[sorted_idx] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    pos_rank_sum = ranks[pos_mask].sum().item()
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def _classification_binary(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> Dict[str, float]:
    scores = torch.sigmoid(logits.float().view(-1))
    y_true = targets.float().view(-1)
    y_pred = (scores >= threshold).float()

    tp = float(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = float(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = float(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = float(((y_pred == 0) & (y_true == 1)).sum().item())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    roc_auc = _binary_roc_auc(scores=scores, labels=y_true.long())
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _classification_multiclass(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    pred_labels = torch.argmax(logits, dim=1)
    true_labels = targets.long().view(-1)
    num_classes = int(logits.shape[1])

    acc = float((pred_labels == true_labels).float().mean().item())

    per_class_precision: List[float] = []
    per_class_recall: List[float] = []
    per_class_f1: List[float] = []
    for cls in range(num_classes):
        tp = float(((pred_labels == cls) & (true_labels == cls)).sum().item())
        fp = float(((pred_labels == cls) & (true_labels != cls)).sum().item())
        fn = float(((pred_labels != cls) & (true_labels == cls)).sum().item())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    def _nanmean(values: List[float]) -> float:
        valid = [v for v in values if not math.isnan(v)]
        if not valid:
            return float("nan")
        return float(sum(valid) / len(valid))

    return {
        "accuracy": acc,
        "precision": _nanmean(per_class_precision),
        "recall": _nanmean(per_class_recall),
        "f1": _nanmean(per_class_f1),
        "roc_auc": float("nan"),
    }


def _regression_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = logits.float().view(-1)
    y_true = targets.float().view(-1)
    err = preds - y_true

    mae = float(err.abs().mean().item())
    rmse = float(torch.sqrt((err**2).mean()).item())
    ss_res = float((err**2).sum().item())
    centered = y_true - y_true.mean()
    ss_tot = float((centered**2).sum().item())
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _survival_concordance_index(
    risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
) -> float:
    r = risk_scores.float().view(-1)
    t = times.float().view(-1)
    e = events.float().view(-1)

    concordant = 0.0
    ties = 0.0
    comparable = 0.0
    n = int(r.shape[0])
    for i in range(n):
        if e[i].item() <= 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if t[i].item() < t[j].item():
                comparable += 1.0
                if r[i].item() > r[j].item():
                    concordant += 1.0
                elif r[i].item() == r[j].item():
                    ties += 1.0
    if comparable == 0.0:
        return float("nan")
    return float((concordant + 0.5 * ties) / comparable)


def _survival_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if targets.ndim != 2 or targets.shape[1] < 2:
        raise ValueError("Survival metrics require target columns [time, event].")
    c_index = _survival_concordance_index(
        risk_scores=logits.view(-1),
        times=targets[:, 0],
        events=targets[:, 1],
    )
    return {"c_index": c_index}


def _compute_task_metrics(
    *, target_type: str, logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> Dict[str, float]:
    if target_type == "binary":
        return _classification_binary(logits=logits, targets=targets, threshold=threshold)
    if target_type == "regression":
        return _regression_metrics(logits=logits, targets=targets)
    if target_type == "survival":
        return _survival_metrics(logits=logits, targets=targets)
    if target_type == "categorical":
        return _classification_multiclass(logits=logits, targets=targets)
    raise ValueError(f"Unsupported target_type '{target_type}'.")


def compute_task_metrics(
    *,
    payload: BatchPredictionPayload,
    target_type: str,
    metrics_cfg: DictConfig,
) -> Dict[str, Any]:
    """Compute global and per-group metrics for a prediction payload."""
    if payload.bag_targets is None:
        return {"global": {}, "per_group": {}}

    threshold = float(metrics_cfg.categorical.threshold)
    global_metrics = _compute_task_metrics(
        target_type=target_type,
        logits=payload.bag_logits,
        targets=payload.bag_targets,
        threshold=threshold,
    )

    per_group: Dict[str, Dict[str, float]] = {}
    if bool(metrics_cfg.per_group):
        grouped_idx: Dict[str, List[int]] = {}
        for idx, bag_id in enumerate(payload.bag_ids):
            grouped_idx.setdefault(str(bag_id), []).append(idx)

        for bag_id, indices in grouped_idx.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            group_logits = payload.bag_logits.index_select(0, idx_tensor)
            group_targets = payload.bag_targets.index_select(0, idx_tensor)
            per_group[bag_id] = _compute_task_metrics(
                target_type=target_type,
                logits=group_logits,
                targets=group_targets,
                threshold=threshold,
            )

    return {"global": global_metrics, "per_group": per_group}
