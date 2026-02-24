from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from omegaconf import OmegaConf

from src.inference.schemas import AggregatedPredictionPayload, BatchPredictionPayload
from src.inference.predict import (
    _configure_preforward_subsampling,
    _needs_instance_payload,
    predict,
)
from src.train import train


@pytest.mark.slow
def test_predict_cli_smoke(cfg_train: DictConfig, cfg_predict: DictConfig, tmp_path: Path) -> None:
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
    HydraConfig().set_config(cfg_train)
    _, _ = train(cfg_train)

    ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
    assert ckpt_path.exists()

    with open_dict(cfg_predict):
        cfg_predict.ckpt_path = str(ckpt_path)
        cfg_predict.embeddings.extract_node = False
        cfg_predict.predict.output_subdir = "predict_artifacts"
    HydraConfig().set_config(cfg_predict)

    summary, _ = predict(cfg_predict)
    assert summary["row_count"] > 0

    output_dir = tmp_path / "predict_artifacts"
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "graph_embeddings.csv").exists()


def test_configure_preforward_subsampling_injects_runtime_and_resets_strategy() -> None:
    class _SamplerCfg:
        def __init__(self):
            self.runtime = {"enabled": True}

    class _DataModule:
        def __init__(self):
            self.sampler_config = _SamplerCfg()
            self.sampler_strategy = object()

    datamodule = _DataModule()
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "subsample_fraction": 0.5,
                "subsample_seed": 123,
            }
        }
    )
    changed = _configure_preforward_subsampling(datamodule, cfg)
    assert changed is True
    assert datamodule.sampler_config.runtime["subsample_fraction"] == 0.5
    assert datamodule.sampler_config.runtime["subsample_seed"] == 123
    assert datamodule.sampler_strategy is None


def test_configure_preforward_subsampling_noop_for_full_fraction() -> None:
    class _SamplerCfg:
        def __init__(self):
            self.runtime = {"enabled": True}

    class _DataModule:
        def __init__(self):
            self.sampler_config = _SamplerCfg()
            self.sampler_strategy = "keep"

    datamodule = _DataModule()
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "max",
                "subsample_fraction": 1.0,
                "subsample_seed": 7,
            }
        }
    )
    changed = _configure_preforward_subsampling(datamodule, cfg)
    assert changed is False
    assert "subsample_fraction" not in datamodule.sampler_config.runtime
    assert datamodule.sampler_strategy == "keep"


@pytest.mark.parametrize(
    ("cfg_dict", "expected"),
    [
        ({"aggregation": {"enabled": False, "mode": "mean"}}, False),
        ({"aggregation": {"enabled": True, "mode": "none"}}, False),
        ({"aggregation": {"enabled": True, "mode": "mean"}}, True),
        ({"aggregation": {"enabled": True, "mode": "max"}}, True),
        ({"aggregation": {"enabled": True, "mode": "attention_weighted"}}, True),
    ],
)
def test_needs_instance_payload(cfg_dict, expected) -> None:
    cfg = OmegaConf.create(cfg_dict)
    assert _needs_instance_payload(cfg) is expected


def test_predict_summary_includes_aggregation_metadata(monkeypatch, tmp_path: Path) -> None:
    import src.inference.predict as predict_module

    class _FakeDataModule:
        pass

    class _FakeModel:
        pass

    class _FakeTrainer:
        pass

    fake_datamodule = _FakeDataModule()
    fake_model = _FakeModel()
    fake_trainer = _FakeTrainer()

    def _fake_instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "fake.DataModule":
            return fake_datamodule
        if target == "fake.Model":
            return fake_model
        if target == "fake.Trainer":
            return fake_trainer
        raise AssertionError(f"Unexpected instantiate target: {target}")

    captured_summary_payload = {}

    def _fake_write_json(payload, path):
        captured_summary_payload.clear()
        captured_summary_payload.update(payload)

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(predict_module, "instantiate_loggers", lambda cfg: [])
    monkeypatch.setattr(predict_module, "log_hyperparameters", lambda obj: None)
    monkeypatch.setattr(
        predict_module,
        "collect_predictions",
        lambda **kwargs: BatchPredictionPayload(
            bag_ids=["r0"],
            bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
            bag_targets=None,
            bag_attention=None,
            row_region_ids=["r0"],
            row_sample_ids=["s0"],
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "aggregate_group_logits",
        lambda *args, **kwargs: AggregatedPredictionPayload(
            bag_ids=["r0"],
            bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
            bag_targets=None,
            bag_attention=None,
            metadata={"mode": "mean", "bag_scope": "region"},
            row_region_ids=["r0"],
            row_sample_ids=["s0"],
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "predictions_to_dataframe",
        lambda payload, include_targets, include_attention: pd.DataFrame(
            {"bag_id": ["r0"], "logit_0": [1.0]}
        ),
    )
    monkeypatch.setattr(predict_module, "write_json", _fake_write_json)

    cfg = OmegaConf.create(
        {
            "ckpt_path": "dummy.ckpt",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "bag_scope": "region",
                "subsample_fraction": 1.0,
                "subsample_seed": None,
            },
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {"enabled": False, "save": False, "extract_node": False},
            "task": {"target_type": "binary"},
        }
    )

    summary, _ = predict(cfg)
    assert "aggregation" in summary
    assert summary["aggregation"]["mode"] == "mean"
    assert summary["aggregation"]["bag_scope"] == "region"
    assert "preforward_subsampling_applied" in summary["aggregation"]
    assert captured_summary_payload["aggregation"]["mode"] == "mean"
