import pytest
import torch


def test_weighted_cross_entropy():
    from src.models.components.losses import WeightedCrossEntropyLoss

    loss_fn = WeightedCrossEntropyLoss()
    logits = torch.randn(12, 3)
    target = torch.randint(0, 3, (12,))
    weight = torch.rand(12)
    loss = loss_fn(logits, target, sample_weight=weight)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_weighted_bce():
    from src.models.components.losses import WeightedBCEWithLogitsLoss

    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.randn(12, 1)
    target = torch.randint(0, 2, (12, 1)).float()
    weight = torch.rand(12, 1)
    loss = loss_fn(logits, target, sample_weight=weight)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("loss_name", ["mse", "huber"])
def test_regression_losses(loss_name):
    from src.models.components.losses import WeightedHuberLoss, WeightedMSELoss

    pred = torch.randn(12, 2)
    target = torch.randn(12, 2)
    sample_weight = torch.rand(12)
    loss_fn = WeightedMSELoss() if loss_name == "mse" else WeightedHuberLoss(delta=1.0)
    loss = loss_fn(pred, target, sample_weight=sample_weight)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_coxsgd_loss():
    from src.models.components.losses import CoxSGDLoss

    loss_fn = CoxSGDLoss(top_n=2, regularizer_weight=0.01)
    y_pred = torch.randn(16, 1)
    length = torch.arange(16).float().flip(0)
    event = torch.randint(0, 2, (16,)).float()
    loss = loss_fn(y_pred, length, event)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_coxsgd_top_n_handles_tiny_batches():
    from src.models.components.losses import CoxSGDLoss

    loss_fn = CoxSGDLoss(top_n=10, regularizer_weight=0.01)
    y_pred = torch.randn(1, 1)
    length = torch.tensor([1.0])
    event = torch.tensor([1.0])
    loss = loss_fn(y_pred, length, event)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
