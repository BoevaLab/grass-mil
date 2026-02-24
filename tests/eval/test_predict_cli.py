from __future__ import annotations

from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from omegaconf import OmegaConf

from src.inference.predict import _configure_preforward_subsampling, predict
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
