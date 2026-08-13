from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from grass_mil.inference.schemas import BatchPredictionPayload


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return float("nan")
    return float(num / den)


def _binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y_true = labels.long().view(-1).detach().cpu().numpy()
    y_score = scores.float().view(-1).detach().cpu().numpy()
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        # Preserve previous behavior when only one class is present.
        return float("nan")


def _classification_binary(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> Dict[str, Any]:
    logits = logits.float()
    targets = targets.float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Binary metrics require logits and targets to have the same number of rows. "
            f"Got logits {tuple(logits.shape)} and targets {tuple(targets.shape)}."
        )
    if logits.shape[1] != targets.shape[1]:
        raise ValueError(
            "Binary metrics require matching task dimensions for logits and targets. "
            f"Got logits {tuple(logits.shape)} and targets {tuple(targets.shape)}."
        )

    per_task: Dict[str, Dict[str, float]] = {}
    for task_idx in range(int(logits.shape[1])):
        scores = torch.sigmoid(logits[:, task_idx].reshape(-1))
        y_true = targets[:, task_idx].reshape(-1)
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
        try:
            balanced_accuracy = float(
                balanced_accuracy_score(
                    y_true.detach().cpu().numpy().astype(int),
                    y_pred.detach().cpu().numpy().astype(int),
                )
            )
        except ValueError:
            balanced_accuracy = float("nan")
        per_task[f"task_{task_idx}"] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

    def _metric_mean(metric_name: str) -> float:
        values = [
            float(metrics[metric_name])
            for metrics in per_task.values()
            if not math.isnan(float(metrics[metric_name]))
        ]
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    result: Dict[str, Any] = {
        "accuracy": _metric_mean("accuracy"),
        "balanced_accuracy": _metric_mean("balanced_accuracy"),
        "precision": _metric_mean("precision"),
        "recall": _metric_mean("recall"),
        "f1": _metric_mean("f1"),
        "roc_auc": _metric_mean("roc_auc"),
    }
    if len(per_task) > 1:
        result["per_task"] = per_task
    return result


def _classification_multiclass(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    pred_labels = torch.argmax(logits, dim=1)
    true_labels = targets.long().view(-1)
    num_classes = int(logits.shape[1])

    acc = float((pred_labels == true_labels).float().mean().item())
    try:
        bal_acc = float(
            balanced_accuracy_score(
                true_labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy()
            )
        )
    except ValueError:
        bal_acc = float("nan")

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
        "balanced_accuracy": bal_acc,
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
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    *,
    chunk_size: int = 4096,
) -> float:
    """Harrell's concordance index.

    A pair is comparable when the earlier subject had an observed event; it is
    concordant when that subject also carries the higher risk, and tied
    contributes a half.

    Evaluated in row chunks of the pair matrix, which bounds peak memory at
    ``chunk_size * n`` booleans. The previous implementation was a Python
    double loop with two ``.item()`` calls per pair, i.e. a device
    synchronisation per pair.
    """
    r = risk_scores.float().view(-1)
    t = times.float().view(-1)
    e = events.float().view(-1)

    n = int(r.shape[0])
    if n == 0:
        return float("nan")

    concordant = 0.0
    ties = 0.0
    comparable = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        t_i = t[start:stop].unsqueeze(1)
        r_i = r[start:stop].unsqueeze(1)
        e_i = e[start:stop].unsqueeze(1)

        # Strict inequality on time also excludes the diagonal.
        is_comparable = (e_i > 0) & (t_i < t.unsqueeze(0))
        if not bool(is_comparable.any()):
            continue
        comparable += float(is_comparable.sum().item())
        concordant += float((is_comparable & (r_i > r.unsqueeze(0))).sum().item())
        ties += float((is_comparable & (r_i == r.unsqueeze(0))).sum().item())

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


def compute_task_metrics_from_tensors(
    *, target_type: str, logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> Dict[str, Any]:
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
    global_metrics = compute_task_metrics_from_tensors(
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
            per_group[bag_id] = compute_task_metrics_from_tensors(
                target_type=target_type,
                logits=group_logits,
                targets=group_targets,
                threshold=threshold,
            )

    return {"global": global_metrics, "per_group": per_group}
