from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch

from src.models.components import (
    EncoderConfig,
    build_attention,
    build_encoder,
    build_graph_head,
    build_loss,
    build_ssl,
)


def infer_encoder_input_dim(module: L.LightningModule, fallback: int = 0) -> int:
    try:
        trainer = module.trainer
    except RuntimeError:
        return fallback
    if trainer is None or trainer.datamodule is None:
        return fallback
    datamodule = trainer.datamodule
    if not hasattr(datamodule, "dataset_train") or datamodule.dataset_train is None:
        # Datamodule may not have been set up yet; trigger setup so we can
        # inspect the first sample and infer input_dim.
        if hasattr(datamodule, "setup"):
            datamodule.setup("fit")
    if not hasattr(datamodule, "dataset_train") or datamodule.dataset_train is None:
        return fallback
    dataset = datamodule.dataset_train
    if len(dataset) == 0:
        return fallback
    item = dataset[0]
    x = getattr(item, "x", None)
    if x is None:
        return fallback
    return int(x.shape[1])


def validate_task_config(task_cfg: Dict[str, Any]) -> None:
    target_type = task_cfg.get("target_type", "binary")
    if target_type not in {"binary", "categorical", "regression", "survival"}:
        raise ValueError(
            f"Unsupported task.target_type='{target_type}'. "
            "Expected one of: binary, categorical, regression, survival."
        )
    instance_sampling = task_cfg.get("instance_sampling", "all")
    if instance_sampling not in {"all", "random"}:
        raise ValueError(
            "task.instance_sampling must be one of ['all', 'random'], "
            f"got '{instance_sampling}'."
        )


def _resolve_encoder_cfg(
    encoder_cfg: Dict[str, Any], inferred_input_dim: int
) -> Dict[str, Any]:
    cfg = dict(encoder_cfg)
    if cfg.get("input_dim", 0) in (None, 0):
        if inferred_input_dim <= 0:
            raise ValueError(
                "Could not infer encoder input_dim from datamodule; "
                "set model.encoder.input_dim explicitly."
            )
        cfg["input_dim"] = inferred_input_dim
    return cfg


def build_supervised_components(
    *,
    module: L.LightningModule,
    encoder_cfg: Dict[str, Any],
    graph_head_cfg: Dict[str, Any],
    attention_cfg: Optional[Dict[str, Any]],
    use_attention: bool,
    loss_cfg: Dict[str, Any],
    use_ssl: bool = False,
    ssl_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.nn.Module]:
    inferred_input_dim = infer_encoder_input_dim(module)
    resolved_encoder_cfg = _resolve_encoder_cfg(encoder_cfg, inferred_input_dim)
    encoder = build_encoder(EncoderConfig(**resolved_encoder_cfg))
    ssl_model = build_ssl(use_ssl, encoder, ssl_cfg) if use_ssl else None
    graph_head = build_graph_head(graph_head_cfg)
    attention = None
    if use_attention:
        if attention_cfg is None:
            raise ValueError("attention config is required when use_attention=True.")
        cfg = dict(attention_cfg)
        attention_type = cfg.pop("attention_type", "gated_projected")
        attention = build_attention(True, attention_type=attention_type, **cfg)
    loss_name = loss_cfg.pop("loss_type")
    loss_fn = build_loss(loss_name, **loss_cfg)
    return {
        "encoder": encoder,
        "ssl_model": ssl_model,
        "graph_head": graph_head,
        "attention": attention,
        "loss_fn": loss_fn,
    }
