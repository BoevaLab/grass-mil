from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .attention import AttnNetGated, AttnNetGatedProjected
from .backbones import EncoderConfig, GNNEncoder
from .heads import GraphPredictionHead
from .losses import (
    CoxSGDLoss,
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    WeightedHuberLoss,
    WeightedMSELoss,
)
from .pooling import GraphPooling
from .ssl import BGRL, MLPPredictor


def build_encoder(config: Dict[str, Any] | EncoderConfig) -> GNNEncoder:
    if isinstance(config, EncoderConfig):
        return GNNEncoder(config)
    return GNNEncoder(EncoderConfig(**config))


def build_pooling(config: Dict[str, Any]) -> GraphPooling:
    return GraphPooling(**config)


def build_attention(
    use_attention: bool, attention_type: str = "gated", **kwargs: Any
) -> Optional[object]:
    if not use_attention:
        return None
    if attention_type == "gated":
        return AttnNetGated(**kwargs)
    if attention_type == "gated_projected":
        return AttnNetGatedProjected(**kwargs)
    raise ValueError(f"Unsupported attention type: {attention_type}")


def build_graph_head(config: Dict[str, Any]) -> GraphPredictionHead:
    return GraphPredictionHead(**config)


def build_ssl(use_ssl: bool, encoder, ssl_config: Optional[Dict[str, Any]] = None):
    if not use_ssl:
        return None
    cfg = ssl_config or {}
    method = cfg.get("method", "bgrl")
    if method != "bgrl":
        raise ValueError(f"Unsupported SSL method: {method}")
    predictor_cfg = cfg.get("predictor")
    if predictor_cfg is None:
        raise ValueError(
            "SSL predictor configuration is required when use_ssl=True "
            "(expected key: ssl.predictor)."
        )
    if "hidden_size" not in predictor_cfg:
        raise ValueError("ssl.predictor.hidden_size is required when use_ssl=True.")
    predictor_cfg = dict(predictor_cfg)
    predictor_cfg.pop("_target_", None)
    predictor_cfg.pop("_recursive_", None)
    predictor_cfg.pop("_convert_", None)
    predictor_cfg.setdefault("input_size", encoder.output_dim)
    predictor_cfg.setdefault("output_size", encoder.output_dim)
    predictor = MLPPredictor(**predictor_cfg)
    return BGRL(encoder=encoder, predictor=predictor)


def _as_weight_tensor(value):
    """Accept tensors, lists/tuples and OmegaConf sequences for loss weights."""
    if value is None or isinstance(value, torch.Tensor):
        return value
    try:
        from omegaconf import ListConfig, OmegaConf

        if isinstance(value, ListConfig):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, (list, tuple)):
        return torch.as_tensor([float(v) for v in value], dtype=torch.float32)
    if isinstance(value, (int, float)):
        return torch.as_tensor([float(value)], dtype=torch.float32)
    raise TypeError(f"Loss weight must be a tensor or a sequence of floats, got {type(value)!r}.")


def build_loss(loss_type: str, **kwargs):
    name = loss_type.lower()
    for key in ("class_weight", "pos_weight"):
        if key in kwargs:
            kwargs[key] = _as_weight_tensor(kwargs[key])
    if name == "categorical_ce":
        return WeightedCrossEntropyLoss(**kwargs)
    if name == "categorical_bce":
        return WeightedBCEWithLogitsLoss(**kwargs)
    if name == "regression_mse":
        return WeightedMSELoss(**kwargs)
    if name == "regression_huber":
        return WeightedHuberLoss(**kwargs)
    if name == "survival_coxsgd":
        return CoxSGDLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss_type}")
