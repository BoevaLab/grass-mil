from __future__ import annotations

from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.inference.predict import predict
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
