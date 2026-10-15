"""Optimizer and learning-rate schedule construction."""

from __future__ import annotations

from typing import Any, Mapping

import hydra
import torch
from omegaconf import OmegaConf

__all__ = ["build_optimizer", "build_scheduler"]


def _plain(cfg: Any) -> dict:
    return dict(OmegaConf.to_container(cfg, resolve=True)) if OmegaConf.is_config(cfg) else dict(cfg)


def build_optimizer(cfg: Mapping[str, Any], groups: list[dict]) -> torch.optim.Optimizer:
    """Instantiate ``cfg`` (``_target_`` + kwargs) over parameter groups.

    Groups may carry their own ``lr`` / ``weight_decay``; the rest inherit the
    optimizer's.
    """
    kwargs = _plain(cfg)
    cls = hydra.utils.get_class(kwargs.pop("_target_"))
    return cls(groups, **kwargs)


def build_scheduler(
    cfg: Mapping[str, Any] | None,
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int | None,
    warmup_steps: int = 0,
):
    """A linear warmup over ``warmup_steps`` followed by ``cfg``.

    ``T_max: auto`` in ``cfg`` becomes the number of optimizer steps after the
    warmup, so a cosine schedule reaches its minimum exactly at the last step.
    """
    if cfg is None:
        return None
    kwargs = _plain(cfg)
    if str(kwargs.get("T_max", "")).lower() == "auto":
        if not total_steps:
            raise ValueError("scheduler.T_max=auto needs the total number of optimizer steps.")
        kwargs["T_max"] = max(1, int(total_steps) - int(warmup_steps))
    main = hydra.utils.instantiate(kwargs, optimizer=optimizer)
    if warmup_steps <= 0:
        return main
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / (warmup_steps + 1), end_factor=1.0, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, main], milestones=[warmup_steps])
