import os
import socket
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.train import (
    _filter_val_monitor_callbacks,
    _requires_zero_validation,
    train,
)
from tests.helpers.run_if import RunIf


def _can_bind_local_port() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


def test_train_test_mode_without_checkpoint_callback(cfg_train: DictConfig) -> None:
    """`test=True` training should not fail without checkpoint callback."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.callbacks = None
        cfg_train.test = True
    metric_dict, _ = train(cfg_train)
    assert "test/loss" in metric_dict


def test_requires_zero_validation_detects_loocv_only_when_val_ratio_zero() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "split": {
                    "loocv": {
                        "enabled": True,
                        "val_ratio": 0.0,
                    }
                }
            }
        }
    )
    assert _requires_zero_validation(cfg)

    cfg.data.split.loocv.val_ratio = 0.01
    assert not _requires_zero_validation(cfg)


def test_requires_zero_validation_detects_non_loocv_zero_val_split() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "split": {
                    "train_val_test_split": [0.8, 0.0, 0.2],
                }
            }
        }
    )
    assert _requires_zero_validation(cfg)

    cfg.data.split.train_val_test_split = [0.8, 0.1, 0.1]
    assert not _requires_zero_validation(cfg)


def test_filter_val_monitor_callbacks_removes_only_val_monitors() -> None:
    callbacks_cfg = OmegaConf.create(
        {
            "model_checkpoint": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "monitor": "val/loss",
            },
            "early_stopping": {
                "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                "monitor": "val/loss",
            },
            "model_summary": {
                "_target_": "lightning.pytorch.callbacks.ModelSummary",
                "max_depth": -1,
            },
        }
    )
    filtered, removed = _filter_val_monitor_callbacks(callbacks_cfg)

    assert set(removed) == {"model_checkpoint", "early_stopping"}
    assert "model_summary" in filtered
    assert "model_checkpoint" not in filtered
    assert "early_stopping" not in filtered


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    if not _can_bind_local_port():
        pytest.skip("Local port binding is not permitted in this environment")
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert any(name.startswith("epoch_") and name.endswith(".ckpt") for name in files)
    assert "last.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert "train/loss" in metric_dict_1
    assert "val/loss" in metric_dict_1
    assert "train/loss" in metric_dict_2
    assert "val/loss" in metric_dict_2
    assert torch.isfinite(metric_dict_1["train/loss"])
    assert torch.isfinite(metric_dict_1["val/loss"])
    assert torch.isfinite(metric_dict_2["train/loss"])
    assert torch.isfinite(metric_dict_2["val/loss"])
