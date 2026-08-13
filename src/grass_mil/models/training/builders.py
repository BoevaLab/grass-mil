from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import lightning as L
import torch

from grass_mil.contracts import validate_attention_width
from grass_mil.models.components.embeddings import resolve_categorical_embedding_config
from grass_mil.models.components import (
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


def infer_categorical_binding(
    module: L.LightningModule, label: str
) -> tuple[Optional[int], Optional[int]]:
    """Resolve ``(num_embeddings, column_index)`` for a categorical node label.

    The vocabulary size comes from the label map that precompute persists in
    ``processed_index.json``, and the column index from the first sample's
    ``categorical_slices``. Mirrors ``infer_encoder_input_dim``: best-effort,
    returning ``None`` when the datamodule cannot be inspected, so the caller
    can fall back to explicit config.
    """
    try:
        trainer = module.trainer
    except RuntimeError:
        return None, None
    if trainer is None or trainer.datamodule is None:
        return None, None
    datamodule = trainer.datamodule

    dataset = getattr(datamodule, "dataset_train", None)
    if dataset is None and hasattr(datamodule, "setup"):
        datamodule.setup("fit")
        dataset = getattr(datamodule, "dataset_train", None)
    if dataset is None or len(dataset) == 0:
        return None, None

    # `label_maps` lives on the graph dataset; unwrap any transform wrapper.
    inner = dataset
    while not hasattr(inner, "label_maps") and hasattr(inner, "dataset"):
        inner = inner.dataset

    num_embeddings: Optional[int] = None
    label_maps = getattr(inner, "label_maps", None)
    if isinstance(label_maps, dict) and label in label_maps:
        num_embeddings = len(label_maps[label])

    column_index: Optional[int] = None
    slices = getattr(dataset[0], "categorical_slices", None)
    if isinstance(slices, dict) and label in slices:
        column_index = int(slices[label])
    return num_embeddings, column_index


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
    if target_type == "survival":
        # Without these checks a mis-shaped survival config fails deep inside
        # the Cox loss with an opaque shape error.
        target_columns = task_cfg.get("target_columns")
        if target_columns is not None and len(target_columns) != 2:
            raise ValueError(
                "Survival tasks need exactly two target columns, ordered "
                f"[time, event]; got {list(target_columns)}."
            )
        if task_cfg.get("loss") not in (None, "survival_coxsgd"):
            raise ValueError(
                "task.target_type='survival' requires task.loss='survival_coxsgd', "
                f"got '{task_cfg.get('loss')}'."
            )


def resolve_encoder_cfg(
    encoder_cfg: Dict[str, Any],
    inferred_input_dim: int,
    *,
    module: Optional[L.LightningModule] = None,
) -> Dict[str, Any]:
    cfg = dict(encoder_cfg)
    categorical_cfg = cfg.get("categorical_embedding")
    if cfg.get("input_dim", 0) in (None, 0):
        if inferred_input_dim <= 0 and categorical_cfg is None:
            raise ValueError(
                "Could not infer encoder input_dim from datamodule; "
                "set model.encoder.input_dim explicitly."
            )
        # A cell-type-only cohort legitimately has zero continuous features; the
        # categorical embedding then carries the whole input representation.
        cfg["input_dim"] = max(int(inferred_input_dim), 0)

    if categorical_cfg is not None:
        label = str(dict(categorical_cfg).get("label", "cell_type"))
        num_embeddings, column_index = (
            infer_categorical_binding(module, label) if module is not None else (None, None)
        )
        resolved = resolve_categorical_embedding_config(
            categorical_cfg,
            cardinalities={label: num_embeddings} if num_embeddings is not None else None,
            slices={label: column_index} if column_index is not None else None,
        )
        cfg["categorical_embedding"] = asdict(resolved)
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
    resolved_encoder_cfg = resolve_encoder_cfg(encoder_cfg, inferred_input_dim, module=module)
    encoder = build_encoder(EncoderConfig(**resolved_encoder_cfg))
    ssl_model = build_ssl(use_ssl, encoder, ssl_cfg) if use_ssl else None
    graph_head = build_graph_head(graph_head_cfg)
    attention = None
    if use_attention:
        if attention_cfg is None:
            raise ValueError("attention config is required when use_attention=True.")
        cfg = dict(attention_cfg)
        attention_type = cfg.pop("attention_type", "gated_projected")
        # Fail at build time rather than part-way through the first batch.
        validate_attention_width(
            attention_width=int(cfg.get("n_classes", 1)),
            num_classes=int(graph_head_cfg.get("output_dim", 1)),
        )
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
