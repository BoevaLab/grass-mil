from __future__ import annotations

import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from grass_mil.inference.metrics import compute_task_metrics
from grass_mil.inference.schemas import BatchPredictionPayload


def test_compute_binary_metrics() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["a", "b", "c", "d"],
        bag_logits=torch.tensor([[2.0], [-1.0], [1.5], [-2.0]]),
        bag_targets=torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        bag_attention=None,
    )
    cfg = OmegaConf.create({"categorical": {"threshold": 0.5}, "per_group": True})
    metrics = compute_task_metrics(payload=payload, target_type="binary", metrics_cfg=cfg)
    assert "accuracy" in metrics["global"]
    assert "balanced_accuracy" in metrics["global"]
    assert "roc_auc" in metrics["global"]
    assert metrics["global"]["accuracy"] >= 0.5
    assert "a" in metrics["per_group"]


def test_compute_regression_metrics() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["a", "b", "c"],
        bag_logits=torch.tensor([[1.0], [2.0], [3.0]]),
        bag_targets=torch.tensor([[1.1], [1.9], [3.2]]),
        bag_attention=None,
    )
    cfg = OmegaConf.create({"categorical": {"threshold": 0.5}, "per_group": False})
    metrics = compute_task_metrics(payload=payload, target_type="regression", metrics_cfg=cfg)
    assert "mae" in metrics["global"]
    assert "rmse" in metrics["global"]
    assert "r2" in metrics["global"]


def test_compute_survival_metrics() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["a", "b", "c", "d"],
        bag_logits=torch.tensor([[0.8], [0.1], [0.7], [0.2]]),
        bag_targets=torch.tensor([[2.0, 1.0], [4.0, 1.0], [3.0, 0.0], [5.0, 1.0]]),
        bag_attention=None,
    )
    cfg = OmegaConf.create({"categorical": {"threshold": 0.5}, "per_group": False})
    metrics = compute_task_metrics(payload=payload, target_type="survival", metrics_cfg=cfg)
    assert "c_index" in metrics["global"]


def test_compute_categorical_metrics() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["a", "b", "c", "d"],
        bag_logits=torch.tensor(
            [
                [4.0, 0.1, -1.0],
                [0.1, 3.0, 0.2],
                [-0.2, 0.1, 2.5],
                [0.5, 1.5, 0.2],
            ]
        ),
        bag_targets=torch.tensor([0, 1, 2, 1]),
        bag_attention=None,
    )
    cfg = OmegaConf.create({"categorical": {"threshold": 0.5}, "per_group": True})
    metrics = compute_task_metrics(payload=payload, target_type="categorical", metrics_cfg=cfg)
    assert "accuracy" in metrics["global"]
    assert "balanced_accuracy" in metrics["global"]
    assert "precision" in metrics["global"]
    assert "recall" in metrics["global"]
    assert "f1" in metrics["global"]
    assert "a" in metrics["per_group"]


def test_compute_binary_metrics_matches_sklearn_auc_with_ties() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["a", "b", "c", "d"],
        bag_logits=torch.tensor([[0.0], [0.0], [-1.0], [2.0]]),
        bag_targets=torch.tensor([[1.0], [0.0], [0.0], [1.0]]),
        bag_attention=None,
    )
    cfg = OmegaConf.create({"categorical": {"threshold": 0.5}, "per_group": False})
    metrics = compute_task_metrics(payload=payload, target_type="binary", metrics_cfg=cfg)
    expected = roc_auc_score(
        payload.bag_targets.view(-1).numpy(),
        torch.sigmoid(payload.bag_logits).view(-1).numpy(),
    )
    assert metrics["global"]["roc_auc"] == expected


def test_compute_binary_multitask_metrics_are_per_task_not_flattened() -> None:
    # Task 0 is perfect, task 1 is perfectly wrong; macro accuracy should be 0.5.
    payload = BatchPredictionPayload(
        bag_ids=["a", "b", "c", "d"],
        bag_logits=torch.tensor(
            [
                [8.0, 8.0],
                [-8.0, -8.0],
                [8.0, 8.0],
                [-8.0, -8.0],
            ]
        ),
        bag_targets=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        bag_attention=None,
    )
    cfg = OmegaConf.create({"categorical": {"threshold": 0.5}, "per_group": False})
    metrics = compute_task_metrics(payload=payload, target_type="binary", metrics_cfg=cfg)

    assert metrics["global"]["accuracy"] == 0.5
    assert metrics["global"]["balanced_accuracy"] == 0.5
    assert "per_task" in metrics["global"]
    assert metrics["global"]["per_task"]["task_0"]["accuracy"] == 1.0
    assert metrics["global"]["per_task"]["task_0"]["balanced_accuracy"] == 1.0
    assert metrics["global"]["per_task"]["task_1"]["accuracy"] == 0.0
    assert metrics["global"]["per_task"]["task_1"]["balanced_accuracy"] == 0.0
