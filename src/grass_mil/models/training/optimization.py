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
    *,
    t_max_auto: Optional[int] = None,
):
    """Instantiate a scheduler; ``T_max: auto`` is replaced by ``t_max_auto`` steps."""
    if cfg is None:
        return None
    sch_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    resolved = OmegaConf.to_container(sch_cfg, resolve=True)
    if isinstance(resolved, dict) and str(resolved.get("T_max", "")).lower() == "auto":
        if t_max_auto is None or int(t_max_auto) < 1:
            raise ValueError(
                "scheduler.T_max='auto' needs the number of optimizer steps; the module "
                "computes it from max_epochs, the training set size and the hyperbatch."
            )
        resolved["T_max"] = int(t_max_auto)
        sch_cfg = OmegaConf.create(resolved)
    return hydra.utils.instantiate(sch_cfg, optimizer=optimizer)


def resolve_t_max_auto(*, num_train_units: int, max_epochs: int, hyperbatch_size: int) -> int:
    """Optimizer steps of a run: ``max_epochs * ceil(num_train_units / hyperbatch)``.

    The preprint anneals the cosine schedule over every optimizer step of the
    run (``T_max = epochs * n_train // hyperbatch``), so the learning rate never
    restarts.
    """
    hb = max(1, int(hyperbatch_size))
    steps_per_epoch = max(1, (int(num_train_units) + hb - 1) // hb)
    return max(1, int(max_epochs)) * steps_per_epoch


def build_param_groups(
    *,
    base_lr: float,
    base_weight_decay: float,
    encoder: Optional[torch.nn.Module],
    head: Optional[torch.nn.Module],
    attention: Optional[torch.nn.Module],
    encoder_lr_scale: float = 1.0,
    head_lr_scale: float = 1.0,
    attention_lr_scale: float = 1.0,
    head_weight_decay: Optional[float] = None,
    freeze_head: bool = False,
) -> list[dict]:
    """Parameter groups with per-component learning-rate multipliers.

    A scale of 0 removes the component from the optimizer (the caller also sets
    ``requires_grad=False`` and keeps it in eval mode). Groups are ordered
    encoder, head, attention; ``head_weight_decay`` overrides the optimizer's
    weight decay for the head group only (the preprint uses 1e-3 on a pre-fitted
    linear head).
    """
    seen: set[int] = set()

    def _params(module: Optional[torch.nn.Module]) -> list[torch.nn.Parameter]:
        out: list[torch.nn.Parameter] = []
        if module is None:
            return out
        for p in module.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
        return out

    groups: list[dict] = []
    if encoder is not None and float(encoder_lr_scale) > 0.0:
        enc = _params(encoder)
        if enc:
            groups.append(
                {"params": enc, "lr": float(base_lr) * float(encoder_lr_scale), "name": "encoder"}
            )
    if head is not None and not freeze_head and float(head_lr_scale) > 0.0:
        hp = _params(head)
        if hp:
            group = {"params": hp, "lr": float(base_lr) * float(head_lr_scale), "name": "head"}
            if head_weight_decay is not None:
                group["weight_decay"] = float(head_weight_decay)
            groups.append(group)
    if attention is not None and float(attention_lr_scale) > 0.0:
        ap = _params(attention)
        if ap:
            groups.append(
                {
                    "params": ap,
                    "lr": float(base_lr) * float(attention_lr_scale),
                    "name": "attention",
                }
            )
    if not groups:
        raise ValueError(
            "No trainable parameters: every component is frozen or has a zero learning-rate scale."
        )
    for g in groups:
        g.setdefault("weight_decay", float(base_weight_decay))
    return groups


def instantiate_scheduler_with_warmup(
    cfg: Optional[Dict[str, Any] | DictConfig],
    optimizer: torch.optim.Optimizer,
    warmup_cfg: Optional[Dict[str, Any] | DictConfig] = None,
    *,
    t_max_auto: Optional[int] = None,
):
    warm_cfg = (
        warmup_cfg if isinstance(warmup_cfg, DictConfig) else OmegaConf.create(warmup_cfg or {})
    )
    warmup_enabled = bool(warm_cfg.get("enabled", False))
    if not warmup_enabled:
        return instantiate_scheduler(cfg, optimizer, t_max_auto=t_max_auto)

    warmup_steps = int(warm_cfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        return instantiate_scheduler(cfg, optimizer, t_max_auto=t_max_auto)

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

    base_scheduler = instantiate_scheduler(cfg, optimizer, t_max_auto=t_max_auto)
    if base_scheduler is None:
        return warmup_scheduler

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps],
    )
