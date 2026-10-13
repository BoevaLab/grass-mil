"""Build a supervised module from config and load trained weights outside ``Trainer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig

from grass_mil.models.training.probe import attach_datamodule

__all__ = ["load_trained_module", "read_checkpoint_state"]


def read_checkpoint_state(ckpt_path: str | Path) -> Dict[str, torch.Tensor]:
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state, dict):
        raise ValueError(f"{ckpt_path} does not hold a state dict.")
    return state


def load_trained_module(
    cfg: DictConfig,
    *,
    ckpt_path: str | Path,
    datamodule: Any,
    device: Optional[torch.device | str] = None,
    strict: bool = True,
):
    """Instantiate ``cfg.model``, bind ``datamodule`` and load ``ckpt_path``.

    The datamodule must already be set up (``setup("fit")``) so the module can
    infer its input widths. Returns the module in eval mode on ``device``.
    """
    import copy

    from omegaconf import open_dict

    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        # The checkpoint carries the trained head; a probe pre-fit request
        # (linear_* regimes: task.head_init_path=auto) is a training-time step.
        if "task" in cfg and cfg.task.get("head_init_path") is not None:
            cfg.task.head_init_path = None
        if "init_from_ckpt" in cfg.model:
            cfg.model.init_from_ckpt = None
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False, enable_checkpointing=False)
    attach_datamodule(model, datamodule, trainer)
    model.setup("fit")
    state = read_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict:
        missing = [k for k in missing if not k.startswith("loss_fn.")]
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint {ckpt_path} does not match the model: missing {missing[:5]}, "
                f"unexpected {unexpected[:5]}. Compose the same model/regime as the run."
            )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
