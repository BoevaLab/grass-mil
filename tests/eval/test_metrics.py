from __future__ import annotations

import math

import pytest
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


def _reference_concordance_index(risk, time, event) -> float:
    """Straightforward O(n^2) reference for the vectorised implementation."""
    concordant = ties = comparable = 0.0
    n = len(risk)
    for i in range(n):
        if event[i] <= 0:
            continue
        for j in range(n):
            if i == j or not time[i] < time[j]:
                continue
            comparable += 1.0
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                ties += 1.0
    if comparable == 0.0:
        return float("nan")
    return (concordant + 0.5 * ties) / comparable


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_concordance_index_matches_the_reference_double_loop(seed: int) -> None:
    import numpy as np

    from grass_mil.inference.metrics import _survival_concordance_index

    rng = np.random.default_rng(seed)
    n = 50
    risk = rng.normal(size=n)
    time = rng.uniform(1.0, 100.0, size=n)
    event = (rng.uniform(size=n) > 0.4).astype(float)

    got = _survival_concordance_index(torch.tensor(risk), torch.tensor(time), torch.tensor(event))
    expected = _reference_concordance_index(risk, time, event)
    assert got == pytest.approx(expected, abs=1e-9)


def test_concordance_index_chunking_does_not_change_the_result() -> None:
    import numpy as np

    from grass_mil.inference.metrics import _survival_concordance_index

    rng = np.random.default_rng(7)
    n = 37
    risk = torch.tensor(rng.normal(size=n))
    time = torch.tensor(rng.uniform(1.0, 50.0, size=n))
    event = torch.tensor((rng.uniform(size=n) > 0.3).astype(float))

    whole = _survival_concordance_index(risk, time, event, chunk_size=1024)
    chunked = _survival_concordance_index(risk, time, event, chunk_size=4)
    assert whole == pytest.approx(chunked, abs=1e-12)


def test_concordance_index_edge_cases() -> None:
    from grass_mil.inference.metrics import _survival_concordance_index

    # No observed events: no comparable pairs at all.
    assert math.isnan(
        _survival_concordance_index(
            torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])
        )
    )
    # Perfect ordering: earlier event carries the higher risk.
    assert _survival_concordance_index(
        torch.tensor([3.0, 2.0, 1.0]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    ) == pytest.approx(1.0)
    # Fully reversed ordering.
    assert _survival_concordance_index(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    ) == pytest.approx(0.0)
    # All risks tied: every comparable pair counts a half.
    assert _survival_concordance_index(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    ) == pytest.approx(0.5)


def test_concordance_index_is_fast_for_large_cohorts() -> None:
    """The old double loop synchronised the device once per pair."""
    import time as timing

    import numpy as np

    from grass_mil.inference.metrics import _survival_concordance_index

    rng = np.random.default_rng(0)
    n = 4000
    risk = torch.tensor(rng.normal(size=n))
    time_col = torch.tensor(rng.uniform(1.0, 100.0, size=n))
    event = torch.tensor((rng.uniform(size=n) > 0.5).astype(float))

    started = timing.perf_counter()
    value = _survival_concordance_index(risk, time_col, event)
    elapsed = timing.perf_counter() - started

    assert 0.0 <= value <= 1.0
    assert elapsed < 5.0
