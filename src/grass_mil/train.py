"""``grass-mil-train``: one training run (fine-tuning, or ``regime=pretrain``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf, open_dict

from grass_mil.inference.predictions import prediction_frame
from grass_mil.runtime import EpochMetricsWriter, instantiate_group, start_run

__all__ = ["train", "main"]

log = logging.getLogger(__name__)


def _without_validation(cfg: DictConfig) -> DictConfig:
    """Drop the callbacks that monitor validation metrics and disable the loop."""
    cfg = cfg.copy()
    with open_dict(cfg):
        for name in list(cfg.callbacks or {}):
            monitor = (cfg.callbacks[name] or {}).get("monitor")
            if isinstance(monitor, str) and monitor.startswith("val/"):
                if name == "model_checkpoint":
                    cfg.callbacks[name].monitor = None
                    cfg.callbacks[name].save_top_k = 0
                else:
                    del cfg.callbacks[name]
        cfg.trainer.limit_val_batches = 0
    return cfg


def train(cfg: DictConfig) -> tuple[dict[str, float], dict[str, Any]]:
    """Train (and test) the configured model; returns the final metrics and the run objects."""
    start_run(cfg)
    output_dir = Path(str(cfg.paths.output_dir))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("fit")
    if len(datamodule.dataset_val) == 0 and cfg.trainer.get("limit_val_batches") != 0:
        log.info("The validation split is empty: training without validation.")
        cfg = _without_validation(cfg)

    model: L.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    callbacks = instantiate_group(cfg.get("callbacks"))
    if cfg.trainer.get("limit_val_batches") != 0:
        callbacks.append(EpochMetricsWriter(output_dir / "metrics_per_epoch.jsonl"))
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=instantiate_group(cfg.get("logger")) or False
    )

    if cfg.get("train"):
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}

    if cfg.get("test") and len(datamodule.dataset_test):
        checkpoint = trainer.checkpoint_callback
        best = checkpoint.best_model_path if checkpoint is not None and checkpoint.monitor else ""
        trainer.test(model, datamodule=datamodule, ckpt_path=best or None)
        metrics.update({k: float(v) for k, v in trainer.callback_metrics.items()})
        preds = model.test_predictions
        frame = prediction_frame(
            preds["unit_ids"],
            preds["logits"],
            target_type=model.task["target_type"],
            targets=preds["targets"],
            meta=datamodule.unit_table(datamodule.dataset_test),
        )
        frame.to_csv(output_dir / "test_predictions.csv", index=False)
        log.info(f"Test predictions written to {output_dir / 'test_predictions.csv'}")
    return metrics, {"cfg": cfg, "datamodule": datamodule, "model": model, "trainer": trainer}


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
