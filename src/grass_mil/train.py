from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def _requires_zero_validation(cfg: DictConfig) -> bool:
    split_cfg = cfg.get("data", {}).get("split")
    if not split_cfg:
        return False

    loocv_cfg = split_cfg.get("loocv")
    if loocv_cfg and bool(loocv_cfg.get("enabled", False)):
        return float(loocv_cfg.get("val_ratio", 0.0)) == 0.0

    ratios = split_cfg.get("train_val_test_split")
    if ratios is None:
        return False
    try:
        return float(ratios[1]) == 0.0
    except Exception:
        return False


def _filter_val_monitor_callbacks(
    callbacks_cfg: Optional[DictConfig],
) -> Tuple[Optional[DictConfig], List[str]]:
    if not callbacks_cfg or not isinstance(callbacks_cfg, DictConfig):
        return callbacks_cfg, []

    filtered = OmegaConf.create({})
    removed: List[str] = []
    for name, cb_conf in callbacks_cfg.items():
        monitor = cb_conf.get("monitor") if isinstance(cb_conf, DictConfig) else None
        if isinstance(monitor, str) and monitor.startswith("val/"):
            removed.append(str(name))
            continue
        filtered[name] = cb_conf
    return filtered, removed


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train (and optionally test) using Hydra-composed config."""
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    if cfg.get("model") is None:
        raise ValueError("No model config provided. Set `model=...` in your Hydra config.")
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks_cfg = cfg.get("callbacks")
    if _requires_zero_validation(cfg):
        callbacks_cfg, removed = _filter_val_monitor_callbacks(callbacks_cfg)
        if removed:
            log.info(
                "Validation is disabled by configuration; skipping val-monitored callbacks: "
                f"{removed}"
            )
    callbacks: List[Callback] = instantiate_callbacks(callbacks_cfg)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = None
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_callback is not None:
            ckpt_path = checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
        else:
            log.warning(
                "Checkpoint callback is not configured. Using current weights for testing..."
            )
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Entry point for training."""
    extras(cfg)

    metric_dict, _ = train(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
