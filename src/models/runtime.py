from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.models.components import (
    EncoderConfig,
    build_attention,
    build_encoder,
    build_graph_head,
    build_loss,
    build_ssl,
)

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import dropout_edge
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for training runtime") from exc


def _to_plain_dict(cfg: Dict[str, Any] | DictConfig | None) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def infer_encoder_input_dim(module: L.LightningModule, fallback: int = 0) -> int:
    trainer = module.trainer
    if trainer is None or trainer.datamodule is None:
        return fallback
    datamodule = trainer.datamodule
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
    if target_type not in {"binary", "regression", "survival"}:
        raise ValueError(
            f"Unsupported task.target_type='{target_type}'. "
            "Expected one of: binary, regression, survival."
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


def extract_bag_ids(
    batch: Data,
    *,
    bag_key: str,
    bag_fallback_key: str,
) -> List[str]:
    bag_values = getattr(batch, bag_key, None)
    if bag_values is None:
        bag_values = getattr(batch, bag_fallback_key, None)
    if bag_values is None:
        raise ValueError(
            f"Batch is missing both '{bag_key}' and '{bag_fallback_key}' for bagging."
        )
    if isinstance(bag_values, torch.Tensor):
        bag_values = bag_values.tolist()
    if isinstance(bag_values, (str, bytes)):
        bag_values = [bag_values]
    return [str(v) for v in bag_values]


def group_instance_indices_by_bag(bag_ids: Sequence[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, bag_id in enumerate(bag_ids):
        grouped.setdefault(bag_id, []).append(idx)
    return grouped


def maybe_sample_indices(
    indices: List[int],
    *,
    max_instances_per_bag: int,
    instance_sampling: str,
) -> List[int]:
    if max_instances_per_bag <= 0 or len(indices) <= max_instances_per_bag:
        return indices
    if instance_sampling == "all":
        return indices[:max_instances_per_bag]
    rand_perm = torch.randperm(len(indices))
    keep = rand_perm[:max_instances_per_bag].tolist()
    keep.sort()
    return [indices[i] for i in keep]


def aggregate_bag_logits_mean(
    logits: torch.Tensor,
    bag_groups: Dict[str, List[int]],
    *,
    max_instances_per_bag: int,
    instance_sampling: str,
) -> tuple[torch.Tensor, List[str], List[List[int]]]:
    bag_logits = []
    bag_ids = []
    bag_indices = []
    for bag_id, indices in bag_groups.items():
        selected = maybe_sample_indices(
            indices,
            max_instances_per_bag=max_instances_per_bag,
            instance_sampling=instance_sampling,
        )
        bag_ids.append(bag_id)
        bag_indices.append(selected)
        bag_logits.append(logits[selected].mean(dim=0, keepdim=True))
    return torch.cat(bag_logits, dim=0), bag_ids, bag_indices


def aggregate_bag_logits_attention(
    *,
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    attention: torch.nn.Module,
    bag_groups: Dict[str, List[int]],
    max_instances_per_bag: int,
    instance_sampling: str,
) -> tuple[torch.Tensor, List[str], Dict[str, torch.Tensor], List[List[int]]]:
    bag_logits = []
    bag_ids = []
    bag_attention = {}
    bag_indices = []
    for bag_id, indices in bag_groups.items():
        selected = maybe_sample_indices(
            indices,
            max_instances_per_bag=max_instances_per_bag,
            instance_sampling=instance_sampling,
        )
        sub_emb = embeddings[selected]
        sub_logits = logits[selected]
        attn_logits, _ = attention(sub_emb)
        attn_scores = torch.softmax(attn_logits.squeeze(-1), dim=0)
        weighted_logits = (sub_logits * attn_scores.unsqueeze(-1)).sum(
            dim=0, keepdim=True
        )
        bag_logits.append(weighted_logits)
        bag_ids.append(bag_id)
        bag_indices.append(selected)
        bag_attention[bag_id] = attn_scores
    return torch.cat(bag_logits, dim=0), bag_ids, bag_attention, bag_indices


def _find_label_index_by_name(label_names: Any, target_name: str) -> int:
    if isinstance(label_names, (list, tuple)):
        if target_name in label_names:
            return int(label_names.index(target_name))
    raise ValueError(
        f"Target '{target_name}' not found in graph_label_names={label_names}."
    )


def select_target_columns(
    graph_y: torch.Tensor,
    batch: Data,
    target_columns: Optional[Sequence[str]],
) -> torch.Tensor:
    if graph_y.ndim == 1:
        graph_y = graph_y.unsqueeze(-1)
    if not target_columns:
        return graph_y
    label_names = getattr(batch, "graph_label_names", None)
    if isinstance(label_names, list) and len(label_names) == graph_y.shape[0]:
        # Batched PyG list of per-graph name lists; use first graph as canonical.
        label_names = label_names[0]
    indices = [_find_label_index_by_name(label_names, name) for name in target_columns]
    return graph_y[:, indices]


def gather_bag_targets(
    *,
    batch: Data,
    bag_indices: List[List[int]],
    target_columns: Optional[Sequence[str]],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(batch, "graph_y"):
        raise ValueError("Batch is missing graph_y required for supervised training.")
    graph_y = select_target_columns(batch.graph_y.float(), batch, target_columns)
    graph_w = batch.graph_w.float() if hasattr(batch, "graph_w") else None
    bag_targets = torch.cat([graph_y[idxs[:1]] for idxs in bag_indices], dim=0)
    bag_weights = None
    if graph_w is not None:
        bag_weights = torch.cat([graph_w[idxs[:1]] for idxs in bag_indices], dim=0)
    return bag_targets, bag_weights


def compute_supervised_loss(
    *,
    loss_fn: torch.nn.Module,
    task_cfg: Dict[str, Any],
    bag_logits: torch.Tensor,
    bag_targets: torch.Tensor,
    bag_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    target_type = task_cfg["target_type"]
    if target_type == "binary":
        if bag_targets.ndim == 1:
            bag_targets = bag_targets.unsqueeze(-1)
        return loss_fn(bag_logits, bag_targets, bag_weights)
    if target_type == "regression":
        return loss_fn(bag_logits, bag_targets, bag_weights)
    # survival
    if bag_targets.ndim != 2 or bag_targets.shape[1] < 2:
        raise ValueError(
            "Survival task expects at least two target columns [time, event]."
        )
    return loss_fn(
        bag_logits.squeeze(-1),
        bag_targets[:, 0].float(),
        bag_targets[:, 1].float(),
    )


def build_mil_aux_targets(
    *,
    bag_targets: torch.Tensor,
    bag_indices: List[List[int]],
    bag_ids: Sequence[str],
    bag_attention: Dict[str, torch.Tensor],
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    aux_targets: List[torch.Tensor] = []
    aux_weights: List[torch.Tensor] = []
    for bag_pos, (bag_id, idxs) in enumerate(zip(bag_ids, bag_indices)):
        bag_target = bag_targets[bag_pos : bag_pos + 1]
        repeated_target = bag_target.repeat(len(idxs), 1)
        if target_mode == "graph_label":
            aux_targets.append(repeated_target)
            aux_weights.append(torch.ones_like(repeated_target))
            continue
        if target_mode == "attention_shaped_ti":
            attn = bag_attention[bag_id].reshape(-1, 1).to(repeated_target.device)
            target = 0.5 + (repeated_target - 0.5) * attn
            aux_targets.append(target)
            aux_weights.append(attn)
            continue
        raise ValueError(
            "node_aux.target_mode must be one of ['graph_label', 'attention_shaped_ti']"
        )
    return torch.cat(aux_targets, dim=0), torch.cat(aux_weights, dim=0)


def gather_instance_logits(
    patch_logits: torch.Tensor, bag_indices: List[List[int]]
) -> torch.Tensor:
    return torch.cat([patch_logits[idxs] for idxs in bag_indices], dim=0)


def compute_aux_node_loss(
    *,
    aux_logits: torch.Tensor,
    aux_targets: torch.Tensor,
    aux_weights: Optional[torch.Tensor],
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode not in {"bce", "weighted_bce"}:
        raise ValueError("node_aux.loss_mode must be one of ['bce', 'weighted_bce']")
    loss = F.binary_cross_entropy_with_logits(
        aux_logits, aux_targets.float(), reduction="none"
    )
    if loss_mode == "weighted_bce":
        if aux_weights is None:
            raise ValueError("weighted_bce requires non-null aux_weights.")
        loss = loss * aux_weights
    return loss.mean()


def compute_entropy_regularization(
    values: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    probs = values.float().clamp(min=eps, max=1.0)
    return -(probs * torch.log(probs + eps)).mean()


def compute_binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == target.float()).float().mean()


@dataclass
class CosineWarmup:
    base_value: float
    warmup_steps: int
    total_steps: int
    min_value: float = 0.0

    def value(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.base_value
        step = max(0, min(step, self.total_steps))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_value * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_value + (self.base_value - self.min_value) * cosine


def augment_graph(batch: Data, *, drop_edge_p: float, drop_feat_p: float) -> Data:
    aug = batch.clone()
    if drop_feat_p > 0 and hasattr(aug, "x") and aug.x.numel() > 0:
        feat_mask = torch.rand_like(aug.x) > drop_feat_p
        aug.x = aug.x * feat_mask.float()
    if drop_edge_p > 0 and hasattr(aug, "edge_index"):
        edge_index, edge_mask = dropout_edge(aug.edge_index, p=drop_edge_p)
        aug.edge_index = edge_index
        if hasattr(aug, "edge_attr") and aug.edge_attr is not None:
            aug.edge_attr = aug.edge_attr[edge_mask]
    return aug


def load_state_dict_with_optional_mapping(
    module: torch.nn.Module,
    *,
    init_from_ckpt: Optional[str],
    init_strict: bool,
    encoder_init_map: str,
) -> Dict[str, List[str]]:
    if not init_from_ckpt:
        return {"missing_keys": [], "unexpected_keys": []}
    ckpt_path = Path(init_from_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {init_from_ckpt}")
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = (
        raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must resolve to a state_dict dictionary.")
    remapped = remap_encoder_keys(state_dict, encoder_init_map=encoder_init_map)
    missing, unexpected = module.load_state_dict(remapped, strict=init_strict)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def remap_encoder_keys(
    state_dict: Dict[str, torch.Tensor], *, encoder_init_map: str
) -> Dict[str, torch.Tensor]:
    if encoder_init_map == "identity":
        return dict(state_dict)
    if encoder_init_map != "auto_bgrl_or_identity":
        raise ValueError(
            "encoder_init_map must be one of ['identity', 'auto_bgrl_or_identity']"
        )
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("online_encoder."):
            new_key = f"encoder.{key[len('online_encoder.') :]}"
        elif key.startswith("ssl_model.online_encoder."):
            new_key = f"encoder.{key[len('ssl_model.online_encoder.') :]}"
        elif key.startswith("model.online_encoder."):
            new_key = f"encoder.{key[len('model.online_encoder.') :]}"
        remapped[new_key] = value
    return remapped


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
