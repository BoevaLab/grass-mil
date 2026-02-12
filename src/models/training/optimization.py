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
    return hydra.utils.instantiate(opt_cfg, params=params)


def instantiate_scheduler(
    cfg: Optional[Dict[str, Any] | DictConfig],
    optimizer: torch.optim.Optimizer,
):
    if cfg is None:
        return None
    sch_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    return hydra.utils.instantiate(sch_cfg, optimizer=optimizer)
