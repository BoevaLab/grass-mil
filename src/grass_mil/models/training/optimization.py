from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def instantiate_optimizer(
    cfg: Dict[str, Any] | DictConfig,
    params: Iterable[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    opt_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    # When *params* is a list of param-group dicts (used for per-component
    # learning rates), Hydra's instantiate would wrap them into OmegaConf
    # containers and lose tensor references.  Fall back to direct construction.
    if isinstance(params, list) and params and isinstance(params[0], dict):
        resolved = OmegaConf.to_container(opt_cfg, resolve=True)
        target = resolved.pop("_target_")
        resolved.pop("_recursive_", None)
        resolved.pop("_convert_", None)
        cls = hydra.utils.get_class(target)
        return cls(params, **resolved)
    return hydra.utils.instantiate(opt_cfg, params=params)


def instantiate_scheduler(
    cfg: Optional[Dict[str, Any] | DictConfig],
    optimizer: torch.optim.Optimizer,
):
    if cfg is None:
        return None
    sch_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    return hydra.utils.instantiate(sch_cfg, optimizer=optimizer)


def instantiate_scheduler_with_warmup(
    cfg: Optional[Dict[str, Any] | DictConfig],
    optimizer: torch.optim.Optimizer,
    warmup_cfg: Optional[Dict[str, Any] | DictConfig] = None,
):
    warm_cfg = (
        warmup_cfg if isinstance(warmup_cfg, DictConfig) else OmegaConf.create(warmup_cfg or {})
    )
    warmup_enabled = bool(warm_cfg.get("enabled", False))
    if not warmup_enabled:
        return instantiate_scheduler(cfg, optimizer)

    warmup_steps = int(warm_cfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        return instantiate_scheduler(cfg, optimizer)

    start_factor = float(warm_cfg.get("start_factor", 0.1))
    if start_factor <= 0:
        start_factor = 1e-4
    if start_factor > 1:
        start_factor = 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    base_scheduler = instantiate_scheduler(cfg, optimizer)
    if base_scheduler is None:
        return warmup_scheduler

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps],
    )
