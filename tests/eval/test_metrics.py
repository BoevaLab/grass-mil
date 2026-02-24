from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.inference.metrics import compute_task_metrics
from src.inference.schemas import BatchPredictionPayload


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
    metrics = compute_task_metrics(
        payload=payload, target_type="regression", metrics_cfg=cfg
    )
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
    metrics = compute_task_metrics(
        payload=payload, target_type="categorical", metrics_cfg=cfg
    )
    assert "accuracy" in metrics["global"]
    assert "precision" in metrics["global"]
    assert "recall" in metrics["global"]
    assert "f1" in metrics["global"]
    assert "a" in metrics["per_group"]
