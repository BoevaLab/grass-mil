"""Training runs: the default regime, the other regimes, other target types, pretraining."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from grass_mil.models import SupervisedModule
from grass_mil.train import train
from tests.helpers.cohort import cohort_overrides, compose_cfg


def _train(cohort, tmp_path, *overrides):
    cfg = compose_cfg("train", cohort_overrides(cohort) + list(overrides), tmp_path)
    return cfg, *train(cfg)


def test_default_regime_is_finetune_with_a_probe_head(cohort, pretrained, tmp_path):
    cfg, metrics, objects = _train(cohort, tmp_path, f"model.init_from_ckpt={pretrained}")
    assert cfg.regime_name == "finetune"
    model = objects["model"]
    assert model.head.is_linear and model.finetune.head_init == "probe"
    assert {"val/auroc", "val/loss", "test/auroc"} <= set(metrics)
    preds = pd.read_csv(tmp_path / "test_predictions.csv")
    assert {"unit_id", "prob_0", "prob_1", "label", "patient_id"} <= set(preds.columns)
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics_per_epoch.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 and "val/auroc" in rows[0]
    # The last, smaller hyperbatch is stepped before the epoch is validated.
    n_train = len(objects["datamodule"].dataset_train)
    assert rows[0]["step"] == -(-n_train // cfg.finetune.hyperbatch_size)


def test_attention_only_trains_nothing_but_the_attention(cohort, pretrained, tmp_path):
    _, _, objects = _train(
        cohort, tmp_path, "regime=attention_only", f"model.init_from_ckpt={pretrained}"
    )
    model = objects["model"]
    ckpt = torch.load(pretrained, weights_only=True)["state_dict"]
    for name, value in model.encoder.state_dict().items():
        assert torch.equal(value.cpu(), ckpt[f"encoder.{name}"])
    assert not any(p.requires_grad for p in model.head.parameters())


@pytest.mark.parametrize("regime", ["finetune_reg", "mlp_head", "mlp_head_frozen"])
def test_other_regimes_run(cohort, pretrained, tmp_path, regime):
    cfg, metrics, objects = _train(
        cohort,
        tmp_path,
        f"regime={regime}",
        f"model.init_from_ckpt={pretrained}",
        "trainer.max_epochs=1",
    )
    assert cfg.regime_name == regime
    assert "test/auroc" in metrics
    assert objects["model"].head.is_linear == (regime == "finetune_reg")


def test_regression_and_survival_train_and_select_on_their_metric(cohort, tmp_path):
    _, metrics, _ = _train(
        cohort,
        tmp_path / "reg",
        "task=regression",
        "task.target_columns=[score]",
        "trainer.max_epochs=1",
    )
    assert "val/rmse" in metrics and "test/rmse" in metrics
    cfg, metrics, objects = _train(
        cohort,
        tmp_path / "surv",
        "task=survival",
        "task.target_columns=[time,event]",
        "trainer.max_epochs=2",
    )
    assert cfg.callbacks.model_checkpoint.monitor == "val/c_index"
    assert "test/c_index" in metrics
    trained = SupervisedModule.from_checkpoint(
        str(tmp_path / "surv" / "checkpoints" / "last.ckpt")
    )
    assert objects["model"].trainer.global_step > 0
    assert torch.isfinite(next(trained.head.parameters())).all()
    preds = pd.read_csv(tmp_path / "surv" / "test_predictions.csv")
    assert {"risk", "time", "event"} <= set(preds.columns)


def test_from_checkpoint_reproduces_the_trained_model(cohort, tmp_path):
    _, _, objects = _train(cohort, tmp_path, "trainer.max_epochs=1", "finetune.head_init=random")
    model = objects["model"].eval()
    rebuilt = SupervisedModule.from_checkpoint(str(tmp_path / "checkpoints" / "last.ckpt"))
    batch = next(iter(objects["datamodule"].test_dataloader()))
    with torch.no_grad():
        assert torch.allclose(model.cpu()(batch), rebuilt(batch), atol=1e-5)


def test_training_without_validation_split(cohort, tmp_path):
    _, metrics, _ = _train(
        cohort,
        tmp_path,
        "data.split.ratios=[0.8,0.0,0.2]",
        "trainer.max_epochs=1",
        "finetune.head_init=random",
    )
    assert not any(k.startswith("val/") for k in metrics)
    assert (tmp_path / "checkpoints" / "last.ckpt").exists()


def test_pretraining_checkpoint_initialises_the_encoder(pretrained):
    ckpt = torch.load(pretrained, weights_only=True)
    assert ckpt["input_spec"]["cell_types"] == ["A", "B", "C", "D"]
    assert any(k.startswith("target_encoder.") for k in ckpt["state_dict"])
    assert ckpt["hyper_parameters"]["pretrain"]["total_steps"] == 4
