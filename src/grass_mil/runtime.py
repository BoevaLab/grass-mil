"""Shared plumbing of the command-line entry points."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "start_run",
    "instantiate_group",
    "resolve_device",
    "EpochMetricsWriter",
    "read_epoch_metrics",
]

log = logging.getLogger("grass_mil")


def start_run(cfg: DictConfig) -> None:
    """Seed every RNG and optionally print the composed config."""
    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True, verbose=False)
    if cfg.get("print_config"):
        log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))


def instantiate_group(group: DictConfig | None) -> list[Any]:
    """Instantiate every entry of a ``{name: {_target_: ...}}`` group (callbacks, loggers)."""
    if not group:
        return []
    return [hydra.utils.instantiate(conf) for conf in group.values() if conf and "_target_" in conf]


def resolve_device(name: str | None) -> torch.device:
    """``auto`` (CUDA when available), ``cpu``, ``cuda`` or ``cuda:<i>``."""
    name = str(name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class EpochMetricsWriter(L.Callback):
    """Append the logged ``train/*`` and ``val/*`` metrics of every epoch to a JSON-lines file.

    Cross-validation reads it to find the best epoch of every inner run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.is_global_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.unlink(missing_ok=True)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Runs after the module logged its epoch-level validation metrics.
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        row: dict[str, Any] = {"epoch": int(trainer.current_epoch), "step": int(trainer.global_step)}
        for key, value in trainer.callback_metrics.items():
            if key.startswith(("train/", "val/")) and not key.endswith("_step"):
                value = float(value)
                if not math.isnan(value):
                    row[key] = value
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


def read_epoch_metrics(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
