from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightning as L
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from grass_mil.batching import extract_subgraphs
from grass_mil.inference.aggregation import aggregate_group_logits
from grass_mil.inference.metrics import compute_task_metrics_from_tensors
from grass_mil.inference.schemas import BatchPredictionPayload

from .training import (
    aggregate_bag_logits_attention,
    aggregate_bag_logits_mean,
    build_finetune_augmentation,
    build_param_groups,
    build_supervised_components,
    collect_unit_labels,
    compute_binary_accuracy,
    compute_categorical_accuracy,
    compute_supervised_loss,
    extract_bag_ids,
    gather_bag_targets,
    group_instance_indices_by_bag,
    instantiate_optimizer,
    instantiate_scheduler_with_warmup,
    inverse_frequency_class_weights,
    load_state_dict_with_optional_mapping,
    resolve_t_max_auto,
    validate_task_config,
)


class SupervisedModule(L.LightningModule):
    """Unified supervised module supporting both mean-pooling and MIL-attention
    aggregation.

    Behaviour is controlled by ``task.aggregation``:

    * ``"mean"`` -- simple mean-pooling over patch logits (no attention, no
      auxiliary losses, no region accumulation).
    * ``"mil_attention"`` -- attention-weighted MIL aggregation with optional
      region accumulation, node auxiliary loss, and entropy regularisation.
    """

    def __init__(
        self,
        encoder: Dict[str, Any],
        graph_head: Dict[str, Any],
        loss: Dict[str, Any],
        optim: Optional[Dict[str, Any]] = None,
        attention: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        task: Optional[Dict[str, Any]] = None,
        flags: Optional[Dict[str, Any]] = None,
        ssl: Optional[Dict[str, Any]] = None,
        init_from_ckpt: Optional[str] = None,
        init_strict: bool = False,
        encoder_init_map: str = "auto_bgrl_or_identity",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._built = False
        self.encoder = None
        self.graph_head = None
        self.loss_fn = None
        self.attention = None
        self.ssl_model = None

        self.optim_cfg = optim
        self.scheduler_cfg = scheduler
        self.task_cfg = task or {}
        validate_task_config(self.task_cfg)
        self.flags_cfg = flags or {}
        self.ssl_cfg = ssl or {}
        self.init_from_ckpt = init_from_ckpt
        self.init_strict = init_strict
        self.encoder_init_map = encoder_init_map
        self.freeze_encoder = freeze_encoder

        # Derive the aggregation mode from the task config.
        aggregation = self.task_cfg.get("aggregation", "mean")
        self.use_attention = aggregation == "mil_attention"

        # MIL-specific config -- only meaningful when use_attention is True.
        if self.use_attention:
            self.region_accum_cfg = dict(self.task_cfg.get("region_accumulation", {}))
            self.region_accum_enabled = bool(self.region_accum_cfg.get("enabled", False))
            self.region_accum_hyperbatch_size = int(
                self.region_accum_cfg.get("hyperbatch_size", 8)
            )
            self.region_accum_flush_on_epoch_end = bool(
                self.region_accum_cfg.get("flush_on_epoch_end", True)
            )
            if self.region_accum_enabled and self.region_accum_hyperbatch_size < 1:
                raise ValueError(
                    "task.region_accumulation.hyperbatch_size must be >= 1 "
                    "when region accumulation is enabled."
                )
        else:
            self.region_accum_cfg = {}
            self.region_accum_enabled = False
            self.region_accum_hyperbatch_size = 0
            self.region_accum_flush_on_epoch_end = False

        # Regime knobs (all optional; see configs/regime/*.yaml for the presets).
        self.optimization_cfg = dict(self.task_cfg.get("optimization", {}) or {})
        self.class_weights_cfg = self.task_cfg.get("class_weights")
        self.finetune_aug_cfg = self.task_cfg.get("finetune_augmentation")
        self.head_init_path = self.task_cfg.get("head_init_path")
        self.freeze_head = bool(self.task_cfg.get("freeze_head", False))
        grad_clip = self.task_cfg.get("grad_clip_norm")
        self.grad_clip_norm = None if grad_clip in (None, "", "none") else float(grad_clip)
        self.scheduler_interval = str(self.task_cfg.get("scheduler_interval", "epoch"))
        if self.scheduler_interval not in {"epoch", "step"}:
            raise ValueError("task.scheduler_interval must be 'epoch' or 'step'.")
        self._finetune_augment = None
        self._encoder_frozen = False
        self._eval_stage = "val"
        self.test_predictions: Optional[Dict[str, Any]] = None
        self._skipped_optimizer_steps = 0

        self._region_loss_buffer: list[torch.Tensor] = []
        self._manual_optimizer_steps = 0
        # In inference, callers can disable instance-level payload emission to reduce
        # peak memory when aggregation only needs bag-level outputs.
        self._predict_emit_instance_payload = True
        self._predict_emit_instance_embeddings = False
        self._predict_emit_instance_composition = False
        self._predict_emit_instance_centroids = False
        self._predict_composition_label = "cell_type"
        self._predict_emit_embeddings_payload = False
        self._predict_emit_node_embeddings = False
        self._reset_val_epoch_buffers()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    @property
    def automatic_optimization(self) -> bool:  # noqa: D401
        return not self.region_accum_enabled

    def setup(self, stage: Optional[str] = None) -> None:
        if self._built:
            return

        loss_cfg = dict(self.hparams.loss)
        loss_cfg.setdefault("loss_type", self.task_cfg.get("loss", "categorical_bce"))

        # Build attention config only when MIL is active.
        if self.use_attention:
            attention_cfg = dict(self.hparams.attention)
            attention_cfg.pop("_target_", None)
        else:
            attention_cfg = None

        components = build_supervised_components(
            module=self,
            encoder_cfg=dict(self.hparams.encoder),
            graph_head_cfg=dict(self.hparams.graph_head),
            attention_cfg=attention_cfg,
            use_attention=self.use_attention,
            loss_cfg=loss_cfg,
            use_ssl=bool(self.flags_cfg.get("use_ssl", False)),
            ssl_cfg=dict(self.ssl_cfg),
        )

        self.encoder = components["encoder"]
        self.graph_head = components["graph_head"]
        self.loss_fn = components["loss_fn"]
        self.attention = components["attention"]
        self.ssl_model = components["ssl_model"]

        load_state_dict_with_optional_mapping(
            self,
            init_from_ckpt=self.init_from_ckpt,
            init_strict=self.init_strict,
            encoder_init_map=self.encoder_init_map,
        )
        encoder_scale = float(self.optimization_cfg.get("encoder_lr_scale", 1.0) or 0.0)
        if self.freeze_encoder or encoder_scale <= 0.0:
            # A zero learning-rate scale is the preprint's "frozen encoder": no
            # gradients, no optimizer group, and the encoder stays in eval mode
            # (dropout/batchnorm off) throughout training.
            for p in self.encoder.parameters():
                p.requires_grad = False
            self._encoder_frozen = True
            self.encoder.eval()

        if self.head_init_path:
            self._load_head_init(str(self.head_init_path))
        if self.freeze_head:
            for p in self.graph_head.parameters():
                p.requires_grad = False

        self._configure_class_weights(stage)
        self._configure_finetune_augmentation()
        self._built = True

    def _load_head_init(self, path: str) -> None:
        """Initialise the graph head from a probe-head file (``graph_head.net.*`` keys)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        head_state = {
            k.split("graph_head.", 1)[1]: v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith("graph_head.") and isinstance(v, torch.Tensor)
        }
        if not head_state:
            raise ValueError(f"No 'graph_head.*' tensors found in head_init_path={path!r}.")
        missing, unexpected = self.graph_head.load_state_dict(head_state, strict=False)
        if unexpected:
            raise ValueError(f"head_init_path carries keys the head does not have: {unexpected}")
        if missing:
            raise ValueError(
                f"head_init_path does not cover the whole head (missing {missing}); a probe "
                "pre-fit targets a linear head (model.graph_head.num_layers=1)."
            )

    def _configure_class_weights(self, stage: Optional[str]) -> None:
        cfg = self.class_weights_cfg
        if cfg is None or stage not in (None, "fit"):
            return
        if not hasattr(self.loss_fn, "class_weight"):
            return
        if isinstance(cfg, str):
            if cfg.lower() != "auto_train":
                raise ValueError("task.class_weights must be 'auto_train', a list, or null.")
            labels = self._training_unit_labels()
            if labels is None:
                return
            num_classes = int(self.hparams.graph_head.get("output_dim", 1))
            if num_classes <= 1:
                return
            weights = inverse_frequency_class_weights(labels, num_classes)
        else:
            weights = torch.as_tensor([float(v) for v in cfg], dtype=torch.float32)
        self.loss_fn.class_weight = weights
        self._class_weights = weights

    def _training_unit_labels(self) -> Optional[list[int]]:
        try:
            trainer = self.trainer
        except RuntimeError:
            return None
        if trainer is None or trainer.datamodule is None:
            return None
        dataset = getattr(trainer.datamodule, "dataset_train", None)
        if dataset is None:
            return None
        labels = collect_unit_labels(dataset, target_columns=self.task_cfg.get("target_columns"))
        return labels or None

    def _configure_finetune_augmentation(self) -> None:
        if not self.finetune_aug_cfg:
            return
        input_proj = getattr(self.encoder, "input_proj", None)
        self._finetune_augment = build_finetune_augmentation(
            self.finetune_aug_cfg,
            unassigned_index=getattr(input_proj, "unassigned_index", None),
            categorical_column=getattr(input_proj, "_column_index", None),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._encoder_frozen and self.encoder is not None:
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward_instances(self, batch):
        if self.training and self._finetune_augment is not None:
            batch = self._finetune_augment(batch)
        node_emb, graph_emb = self.encoder(
            batch.x,
            batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=getattr(batch, "batch", None),
            categorical_codes=getattr(batch, "categorical_codes", None),
            return_graph_embedding=True,
        )
        patch_logits = self.graph_head(graph_emb)
        return patch_logits, graph_emb, node_emb

    def _to_optional_str_list(
        self, values: Any, *, expected_length: int, field_name: str
    ) -> List[Optional[str]]:
        if values is None:
            return [None] * expected_length
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        if isinstance(values, (str, bytes)):
            raw_values = [values]
        else:
            try:
                raw_values = list(values)
            except TypeError:
                raw_values = [values]
        if len(raw_values) == expected_length:
            pass
        elif len(raw_values) == 1 and expected_length > 1:
            raw_values = raw_values * expected_length
        else:
            raise ValueError(
                f"Batch field '{field_name}' has {len(raw_values)} values, expected "
                f"{expected_length} for instance-level metadata alignment."
            )
        normalized: List[Optional[str]] = []
        for value in raw_values:
            if value is None:
                normalized.append(None)
                continue
            text = str(value).strip()
            if text.lower() in {"", "nan", "none", "null"}:
                normalized.append(None)
            else:
                normalized.append(str(value))
        return normalized

    @staticmethod
    def _resolve_categorical_column_index(batch: Any, label_name: str) -> Optional[int]:
        slices = getattr(batch, "categorical_slices", None)
        if isinstance(slices, dict):
            value = slices.get(label_name)
            return int(value) if value is not None else None
        if isinstance(slices, (list, tuple)):
            for item in slices:
                if isinstance(item, dict) and label_name in item:
                    value = item[label_name]
                    return int(value) if value is not None else None
        labels = getattr(batch, "categorical_labels", None)
        if isinstance(labels, (list, tuple)):
            for idx, value in enumerate(labels):
                if str(value) == str(label_name):
                    return idx
        return None

    @staticmethod
    def _compute_instance_composition(
        batch: Any,
        *,
        n_instances: int,
        label_name: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        categorical_codes = getattr(batch, "categorical_codes", None)
        if not isinstance(categorical_codes, torch.Tensor) or categorical_codes.numel() == 0:
            return None
        categorical_codes = categorical_codes.to(device=device).long()
        if categorical_codes.ndim != 2:
            return None
        col_idx = SupervisedModule._resolve_categorical_column_index(batch, label_name)
        if col_idx is None or col_idx < 0 or col_idx >= int(categorical_codes.shape[1]):
            return None

        node_labels = categorical_codes[:, int(col_idx)]
        if node_labels.numel() == 0:
            return None
        batch_index = getattr(batch, "batch", None)
        if isinstance(batch_index, torch.Tensor):
            batch_index = batch_index.to(device=device).long()
            if batch_index.numel() != node_labels.numel():
                return None
        elif n_instances == 1:
            batch_index = torch.zeros(node_labels.numel(), dtype=torch.long, device=device)
        else:
            return None

        n_classes = int(node_labels.max().item()) + 1
        if n_classes <= 0:
            return None
        composition = torch.zeros((n_instances, n_classes), dtype=torch.float32, device=device)
        for inst_idx in range(n_instances):
            mask = batch_index == inst_idx
            labels = node_labels[mask]
            if labels.numel() == 0:
                continue
            counts = torch.bincount(labels, minlength=n_classes).float()
            total = counts.sum()
            if total > 0:
                counts = counts / total
            composition[inst_idx, :] = counts
        return composition

    @staticmethod
    def _compute_instance_centroids(
        batch: Any,
        *,
        n_instances: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        positions = getattr(batch, "pos", None)
        if not isinstance(positions, torch.Tensor) or positions.ndim != 2:
            return None
        if positions.numel() == 0:
            return None
        positions = positions.to(device=device, dtype=torch.float32)
        if positions.shape[1] < 2:
            return None

        batch_index = getattr(batch, "batch", None)
        if isinstance(batch_index, torch.Tensor):
            batch_index = batch_index.to(device=device).long()
            if batch_index.numel() != positions.shape[0]:
                return None
        elif n_instances == 1:
            batch_index = torch.zeros(positions.shape[0], dtype=torch.long, device=device)
        else:
            return None

        centroids = torch.zeros((n_instances, 2), dtype=torch.float32, device=device)
        for inst_idx in range(n_instances):
            mask = batch_index == inst_idx
            if not torch.any(mask):
                continue
            centroids[inst_idx, :] = positions[mask][:, :2].mean(dim=0)
        return centroids

    def _attention_width(self) -> int:
        """Number of attention channels, i.e. 1 (shared) or one per class."""
        if self.attention is None:
            return 1
        head = getattr(self.attention, "attention_c", None)
        out_features = getattr(head, "out_features", None)
        return int(out_features) if out_features is not None else 1

    def _build_predict_group_metadata(
        self,
        batch,
        *,
        patch_logits: torch.Tensor,
        graph_emb: torch.Tensor,
        ordered_bag_ids: list[str],
        bag_indices: list[list[int]],
        include_instance_payload: bool = True,
    ) -> Dict[str, Any]:
        n_instances = int(patch_logits.shape[0])
        region_ids = self._to_optional_str_list(
            getattr(batch, "region_id", None),
            expected_length=n_instances,
            field_name="region_id",
        )
        sample_ids = self._to_optional_str_list(
            getattr(batch, "sample_id", None),
            expected_length=n_instances,
            field_name="sample_id",
        )
        row_region_ids: list[Optional[str]] = []
        row_sample_ids: list[Optional[str]] = []
        for indices in bag_indices:
            row_region_ids.append(region_ids[indices[0]] if indices else None)
            row_sample_ids.append(sample_ids[indices[0]] if indices else None)

        metadata: Dict[str, Any] = {
            "row_region_ids": row_region_ids,
            "row_sample_ids": row_sample_ids,
        }
        if not include_instance_payload:
            return metadata

        patch_ids = self._to_optional_str_list(
            getattr(batch, "patch_id", None),
            expected_length=n_instances,
            field_name="patch_id",
        )
        instance_logits_chunks: list[torch.Tensor] = []
        instance_patch_ids: list[str] = []
        instance_bag_ids: list[str] = []
        instance_region_ids: list[Optional[str]] = []
        instance_sample_ids: list[Optional[str]] = []
        instance_attention_logits_chunks: list[torch.Tensor] = []
        instance_embedding_chunks: list[torch.Tensor] = []
        instance_graph_chunks: list[Any] = []
        composition_by_instance = (
            self._compute_instance_composition(
                batch,
                n_instances=n_instances,
                label_name=str(getattr(self, "_predict_composition_label", "cell_type")),
                device=patch_logits.device,
            )
            if bool(getattr(self, "_predict_emit_instance_composition", False))
            else None
        )
        centroids_by_instance = (
            self._compute_instance_centroids(
                batch,
                n_instances=n_instances,
                device=patch_logits.device,
            )
            if bool(getattr(self, "_predict_emit_instance_centroids", False))
            else None
        )
        instance_composition_chunks: list[torch.Tensor] = []
        instance_centroid_chunks: list[torch.Tensor] = []
        compute_attention_logits = (
            include_instance_payload and self.use_attention and self.attention is not None
        )
        # ShaDow batches are built directly and delimited by `ptr`, so
        # `to_data_list()` raises on them. This previously sat behind a bare
        # `except` that turned that into a silent None, and the spatial-table
        # export then failed several steps later complaining about a missing
        # payload field instead of naming the cause.
        subgraph_data_list = extract_subgraphs(batch)

        for bag_id, indices in zip(ordered_bag_ids, bag_indices):
            if not indices:
                continue
            idx = torch.tensor(indices, dtype=torch.long, device=patch_logits.device)
            sub_logits = patch_logits.index_select(0, idx)
            instance_logits_chunks.append(sub_logits)
            instance_embedding_chunks.append(graph_emb.index_select(0, idx))
            instance_patch_ids.extend(
                [patch_ids[i] if patch_ids[i] is not None else str(bag_id) for i in indices]
            )
            instance_bag_ids.extend([str(bag_id)] * len(indices))
            instance_region_ids.extend([region_ids[i] for i in indices])
            instance_sample_ids.extend([sample_ids[i] for i in indices])
            if isinstance(subgraph_data_list, list):
                for i in indices:
                    if 0 <= int(i) < len(subgraph_data_list):
                        instance_graph_chunks.append(subgraph_data_list[int(i)])
            if composition_by_instance is not None:
                instance_composition_chunks.append(composition_by_instance.index_select(0, idx))
            if centroids_by_instance is not None:
                instance_centroid_chunks.append(centroids_by_instance.index_select(0, idx))
            if compute_attention_logits:
                sub_emb = graph_emb.index_select(0, idx)
                attn_logits, _ = self.attention(sub_emb)
                if attn_logits.dim() == 1:
                    attn_logits = attn_logits.unsqueeze(-1)
                instance_attention_logits_chunks.append(attn_logits)

        metadata["instance_logits"] = (
            torch.cat(instance_logits_chunks, dim=0)
            if instance_logits_chunks
            else torch.empty((0, patch_logits.shape[-1]), device=patch_logits.device)
        )
        metadata["instance_patch_ids"] = instance_patch_ids
        metadata["instance_bag_ids"] = instance_bag_ids
        metadata["instance_region_ids"] = instance_region_ids
        metadata["instance_sample_ids"] = instance_sample_ids
        metadata["instance_embeddings"] = (
            torch.cat(instance_embedding_chunks, dim=0)
            if instance_embedding_chunks
            else torch.empty((0, graph_emb.shape[-1]), device=graph_emb.device)
        )
        if composition_by_instance is not None:
            metadata["instance_composition"] = (
                torch.cat(instance_composition_chunks, dim=0)
                if instance_composition_chunks
                else torch.empty(
                    (0, composition_by_instance.shape[-1]),
                    device=composition_by_instance.device,
                )
            )
        if centroids_by_instance is not None:
            metadata["instance_centroids"] = (
                torch.cat(instance_centroid_chunks, dim=0)
                if instance_centroid_chunks
                else torch.empty((0, 2), device=patch_logits.device)
            )
        if compute_attention_logits:
            metadata["instance_attention_logits"] = (
                torch.cat(instance_attention_logits_chunks, dim=0)
                if instance_attention_logits_chunks
                else torch.empty((0, self._attention_width()), device=patch_logits.device)
            )
        if instance_graph_chunks:
            metadata["instance_graphs"] = instance_graph_chunks
        return metadata

    def _build_predict_embedding_payload(
        self,
        batch,
        *,
        graph_emb: torch.Tensor,
        node_emb: torch.Tensor,
        ordered_bag_ids: list[str],
        bag_indices: list[list[int]],
        include_node_embeddings: bool,
    ) -> Dict[str, Any]:
        bag_graph_embeddings: list[torch.Tensor] = []
        bag_counts: list[int] = []
        for indices in bag_indices:
            if not indices:
                continue
            idx = torch.tensor(indices, dtype=torch.long, device=graph_emb.device)
            bag_graph_embeddings.append(graph_emb.index_select(0, idx).mean(dim=0, keepdim=True))
            bag_counts.append(int(idx.numel()))

        payload: Dict[str, Any] = {
            "embedding_bag_ids": [str(v) for v in ordered_bag_ids],
            "graph_embeddings": (
                torch.cat(bag_graph_embeddings, dim=0)
                if bag_graph_embeddings
                else torch.empty((0, graph_emb.shape[-1]), device=graph_emb.device)
            ),
            "embedding_bag_counts": bag_counts,
        }
        if include_node_embeddings:
            bag_ids = extract_bag_ids(
                batch,
                bag_key=self.task_cfg.get("bag_key", "region_id"),
                bag_fallback_key=self.task_cfg.get("bag_fallback_key", "sample_id"),
            )
            batch_index = getattr(batch, "batch", None)
            if batch_index is None:
                batch_index = torch.zeros(
                    node_emb.shape[0], dtype=torch.long, device=node_emb.device
                )
            node_bag_ids = [
                str(bag_ids[int(graph_idx)]) for graph_idx in batch_index.detach().cpu()
            ]
            payload["node_embeddings"] = node_emb
            payload["node_bag_ids"] = node_bag_ids
        return payload

    def _forward_bags(self, batch, *, allow_missing_targets: bool = False):
        patch_logits, graph_emb, node_emb = self._forward_instances(batch)

        bag_ids = extract_bag_ids(
            batch,
            bag_key=self.task_cfg.get("bag_key", "region_id"),
            bag_fallback_key=self.task_cfg.get("bag_fallback_key", "sample_id"),
        )
        bag_groups = group_instance_indices_by_bag(bag_ids)

        if self.use_attention:
            bag_logits, ordered_bag_ids, bag_attention, bag_indices = (
                aggregate_bag_logits_attention(
                    logits=patch_logits,
                    embeddings=graph_emb,
                    attention=self.attention,
                    bag_groups=bag_groups,
                    max_instances_per_bag=int(self.task_cfg.get("max_instances_per_bag", 0)),
                    instance_sampling=self.task_cfg.get("instance_sampling", "all"),
                )
            )
        else:
            bag_logits, ordered_bag_ids, bag_indices = aggregate_bag_logits_mean(
                patch_logits,
                bag_groups,
                max_instances_per_bag=int(self.task_cfg.get("max_instances_per_bag", 0)),
                instance_sampling=self.task_cfg.get("instance_sampling", "all"),
            )
            bag_attention = None

        bag_targets = None
        bag_weights = None
        if hasattr(batch, "graph_y"):
            bag_targets, bag_weights = gather_bag_targets(
                batch=batch,
                bag_indices=bag_indices,
                target_columns=self.task_cfg.get("target_columns"),
            )
        elif not allow_missing_targets:
            raise ValueError("Batch is missing graph_y required for supervised loss computation.")

        return {
            "patch_logits": patch_logits,
            "graph_emb": graph_emb,
            "node_emb": node_emb,
            "bag_logits": bag_logits,
            "ordered_bag_ids": ordered_bag_ids,
            "bag_attention": bag_attention,
            "bag_indices": bag_indices,
            "bag_targets": bag_targets,
            "bag_weights": bag_weights,
        }

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_region_loss(
        self,
        *,
        bag_logits: torch.Tensor,
        bag_targets: torch.Tensor,
        bag_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return compute_supervised_loss(
            loss_fn=self.loss_fn,
            task_cfg=self.task_cfg,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            bag_weights=bag_weights,
        )

    def _compute_losses(
        self,
        *,
        patch_logits: torch.Tensor,
        bag_logits: torch.Tensor,
        bag_targets: torch.Tensor,
        bag_weights: Optional[torch.Tensor],
        ordered_bag_ids: list[str],
        bag_indices: list[list[int]],
        bag_attention: Optional[dict[str, torch.Tensor]],
        stage: str,
    ) -> torch.Tensor:
        """Region cross-entropy is the sole training objective."""
        return self._compute_region_loss(
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            bag_weights=bag_weights,
        )

    def _log_stage_metrics(
        self,
        *,
        stage: str,
        loss_value: torch.Tensor,
        bag_logits: torch.Tensor,
        bag_targets: torch.Tensor,
        region_loss: Optional[torch.Tensor] = None,
    ) -> None:
        if stage not in {"train", "val", "test"}:
            return
        self.log(
            f"{stage}/loss",
            loss_value,
            on_step=stage in {"train", "val"},
            on_epoch=True,
            prog_bar=True,
        )
        if self.use_attention and stage == "train" and region_loss is not None:
            self.log(
                "train/region_loss",
                region_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        if stage in {"train", "test"} and self.task_cfg["target_type"] == "binary":
            acc = compute_binary_accuracy(bag_logits, bag_targets)
            self.log(
                f"{stage}/acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=stage != "train",
            )
        elif stage in {"train", "test"} and self.task_cfg["target_type"] == "categorical":
            acc = compute_categorical_accuracy(bag_logits, bag_targets)
            self.log(
                f"{stage}/acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=stage != "train",
            )

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str):
        payload = self._forward_bags(batch)
        patch_logits = payload["patch_logits"]
        bag_logits = payload["bag_logits"]
        ordered_bag_ids = payload["ordered_bag_ids"]
        bag_attention = payload["bag_attention"]
        bag_indices = payload["bag_indices"]
        bag_targets = payload["bag_targets"]
        bag_weights = payload["bag_weights"]

        region_loss = self._compute_losses(
            patch_logits=patch_logits,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            bag_weights=bag_weights,
            ordered_bag_ids=ordered_bag_ids,
            bag_indices=bag_indices,
            bag_attention=bag_attention,
            stage=stage,
        )
        self._log_stage_metrics(
            stage=stage,
            loss_value=region_loss,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            region_loss=region_loss if self.use_attention else None,
        )

        result = {
            "loss": region_loss,
            "bag_ids": list(ordered_bag_ids),
            "bag_logits": bag_logits,
            "bag_targets": bag_targets,
            "patch_logits": patch_logits,
            "graph_emb": payload["graph_emb"],
            "ordered_bag_ids": list(ordered_bag_ids),
            "bag_indices": bag_indices,
            "bag_attention": bag_attention,
        }
        if self.use_attention:
            result["region_loss"] = region_loss
        return result

    # ------------------------------------------------------------------
    # Training / validation / test / predict steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if not self.region_accum_enabled:
            return self._shared_step(batch, "train")["loss"]

        # ---------- Region accumulation path (MIL only) ---------------
        payload = self._forward_bags(batch)
        bag_logits = payload["bag_logits"]
        bag_targets = payload["bag_targets"]
        bag_weights = payload["bag_weights"]
        ordered_bag_ids = payload["ordered_bag_ids"]
        bag_indices = payload["bag_indices"]
        bag_attention = payload["bag_attention"]
        patch_logits = payload["patch_logits"]

        region_losses: list[torch.Tensor] = []
        buffered_region_losses: list[torch.Tensor] = []
        for idx, bag_id in enumerate(ordered_bag_ids):
            region_w = bag_weights[idx : idx + 1] if bag_weights is not None else None
            region_loss = self._compute_losses(
                patch_logits=patch_logits,
                bag_logits=bag_logits[idx : idx + 1],
                bag_targets=bag_targets[idx : idx + 1],
                bag_weights=region_w,
                bag_indices=[bag_indices[idx]],
                ordered_bag_ids=[bag_id],
                bag_attention={bag_id: bag_attention[bag_id]},
                stage="train",
            )
            region_losses.append(region_loss.detach())
            buffered_region_losses.append(region_loss)

        if not buffered_region_losses:
            return torch.zeros((), device=self.device)

        # Buffer one scalar per forward pass to avoid chunked backward passes over
        # the same computation graph.
        step_region_loss = torch.stack(buffered_region_losses).mean()
        self._region_loss_buffer.append(step_region_loss)

        flush_losses = self._flush_region_buffer_if_needed(force=False)
        for flush_loss in flush_losses:
            self.log(
                "train/flush_loss",
                flush_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        self.log(
            "train/region_buffer_size",
            float(len(self._region_loss_buffer)),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        mean_region_loss_value = step_region_loss.detach()
        mean_region_loss = torch.stack(region_losses).mean()
        self._log_stage_metrics(
            stage="train",
            loss_value=mean_region_loss_value,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            region_loss=mean_region_loss,
        )

        return mean_region_loss_value

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def _eval_step(self, batch, stage: str):
        """Shared val/test step: buffer instance-level outputs for region-level
        aggregation at epoch end, so a region split over several sub-batches is
        scored exactly once with one attention softmax over all of its cells."""
        out = self._shared_step(batch, stage)
        metadata = self._build_predict_group_metadata(
            batch,
            patch_logits=out["patch_logits"],
            graph_emb=out["graph_emb"],
            ordered_bag_ids=out["ordered_bag_ids"],
            bag_indices=out["bag_indices"],
        )
        self._val_bag_ids.extend([str(v) for v in out["bag_ids"]])
        self._val_bag_logits_chunks.append(out["bag_logits"].detach().cpu())
        self._val_row_region_ids.extend(metadata["row_region_ids"])
        self._val_row_sample_ids.extend(metadata["row_sample_ids"])
        if out["bag_targets"] is not None:
            self._val_bag_targets_chunks.append(out["bag_targets"].detach().cpu())
        self._val_instance_logits_chunks.append(metadata["instance_logits"].detach().cpu())
        self._val_instance_patch_ids.extend([str(v) for v in metadata["instance_patch_ids"]])
        self._val_instance_region_ids.extend(metadata["instance_region_ids"])
        self._val_instance_sample_ids.extend(metadata["instance_sample_ids"])
        if "instance_attention_logits" in metadata:
            self._val_instance_attention_logits_chunks.append(
                metadata["instance_attention_logits"].detach().cpu()
            )
        return out

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def on_test_epoch_start(self) -> None:
        self._eval_stage = "test"
        self._reset_val_epoch_buffers()

    def on_test_epoch_end(self) -> None:
        self._finish_eval_epoch("test")
        self._eval_stage = "val"

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._forward_bags(batch, allow_missing_targets=True)
        emit_instance_payload = bool(getattr(self, "_predict_emit_instance_payload", True))
        emit_embeddings_payload = bool(getattr(self, "_predict_emit_embeddings_payload", False))
        emit_node_embeddings = bool(getattr(self, "_predict_emit_node_embeddings", False))
        metadata = self._build_predict_group_metadata(
            batch,
            patch_logits=out["patch_logits"],
            graph_emb=out.get("graph_emb", out["patch_logits"]),
            ordered_bag_ids=out["ordered_bag_ids"],
            bag_indices=out["bag_indices"],
            include_instance_payload=emit_instance_payload,
        )
        result: Dict[str, Any] = {
            "bag_ids": list(out["ordered_bag_ids"]),
            "bag_logits": out["bag_logits"].detach(),
            "row_region_ids": metadata["row_region_ids"],
            "row_sample_ids": metadata["row_sample_ids"],
        }
        if out["bag_targets"] is not None:
            result["bag_targets"] = out["bag_targets"].detach()
        if emit_instance_payload:
            result["instance_logits"] = metadata["instance_logits"].detach()
            result["instance_patch_ids"] = metadata["instance_patch_ids"]
            result["instance_bag_ids"] = metadata["instance_bag_ids"]
            result["instance_region_ids"] = metadata["instance_region_ids"]
            result["instance_sample_ids"] = metadata["instance_sample_ids"]
            if "instance_graphs" in metadata:
                result["instance_graphs"] = metadata["instance_graphs"]
            if bool(getattr(self, "_predict_emit_instance_embeddings", False)):
                result["instance_embeddings"] = metadata["instance_embeddings"].detach()
                if "instance_composition" in metadata:
                    result["instance_composition"] = metadata["instance_composition"].detach()
                if "instance_centroids" in metadata:
                    result["instance_centroids"] = metadata["instance_centroids"].detach()
        if self.use_attention:
            result["bag_attention"] = out["bag_attention"]
            if emit_instance_payload:
                result["instance_attention_logits"] = metadata[
                    "instance_attention_logits"
                ].detach()
        if emit_embeddings_payload:
            embedding_payload = self._build_predict_embedding_payload(
                batch,
                graph_emb=out.get("graph_emb", out["patch_logits"]),
                node_emb=out.get("node_emb", out.get("graph_emb", out["patch_logits"])),
                ordered_bag_ids=out["ordered_bag_ids"],
                bag_indices=out["bag_indices"],
                include_node_embeddings=emit_node_embeddings,
            )
            result["embedding_bag_ids"] = embedding_payload["embedding_bag_ids"]
            result["graph_embeddings"] = embedding_payload["graph_embeddings"].detach()
            result["embedding_bag_counts"] = embedding_payload["embedding_bag_counts"]
            if emit_node_embeddings:
                result["node_embeddings"] = embedding_payload["node_embeddings"].detach()
                result["node_bag_ids"] = embedding_payload["node_bag_ids"]
        return result

    # ------------------------------------------------------------------
    # Region accumulation helpers (MIL only)
    # ------------------------------------------------------------------

    def _flush_region_buffer_if_needed(self, *, force: bool) -> list[torch.Tensor]:
        if not self._region_loss_buffer:
            return []
        if not force and len(self._region_loss_buffer) < self.region_accum_hyperbatch_size:
            return []

        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        flush_losses: list[torch.Tensor] = []

        while self._region_loss_buffer:
            if force:
                chunk = self._region_loss_buffer
            else:
                if len(self._region_loss_buffer) < self.region_accum_hyperbatch_size:
                    break
                chunk = self._region_loss_buffer[: self.region_accum_hyperbatch_size]

            flush_loss = torch.stack(chunk, dim=0).mean()
            self.manual_backward(flush_loss)
            if self.grad_clip_norm is not None:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=float(self.grad_clip_norm),
                    gradient_clip_algorithm="norm",
                )
            scaler = self._amp_scaler()
            scale_before = float(scaler.get_scale()) if scaler is not None else None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Under mixed precision a step with inf/nan gradients is skipped and the
            # loss scale is reduced; the preprint only advances the schedule on
            # steps that actually updated the weights.
            stepped = scaler is None or float(scaler.get_scale()) >= scale_before
            if stepped:
                scheduler = self.lr_schedulers()
                if scheduler is not None:
                    if isinstance(scheduler, (list, tuple)):
                        for sch in scheduler:
                            sch.step()
                    else:
                        scheduler.step()
            else:
                self._skipped_optimizer_steps += 1
            self._manual_optimizer_steps += 1
            self.log(
                "train/manual_optimizer_steps",
                float(self._manual_optimizer_steps),
                on_step=not force,
                on_epoch=force,
                prog_bar=False,
            )
            flush_losses.append(flush_loss.detach())

            del self._region_loss_buffer[: len(chunk)]
            if force:
                break

        return flush_losses

    def _amp_scaler(self):
        """The GradScaler of the active mixed-precision plugin, if any."""
        try:
            plugin = self.trainer.precision_plugin
        except RuntimeError:
            return None
        return getattr(plugin, "scaler", None)

    def on_train_start(self) -> None:
        if self.region_accum_enabled:
            self._region_loss_buffer.clear()

    def on_validation_epoch_start(self) -> None:
        self._reset_val_epoch_buffers()

    def on_validation_epoch_end(self) -> None:
        self._finish_eval_epoch("val")

    def _finish_eval_epoch(self, stage: str) -> None:
        payload = self._build_validation_epoch_payload()
        if payload is None or payload.bag_targets is None:
            self._reset_val_epoch_buffers()
            return
        aggregated = aggregate_group_logits(
            payload,
            mode="attention_weighted" if self.use_attention else "mean",
            bag_scope="region",
            subsample_fraction=1.0,
            subsample_seed=None,
        )
        if aggregated.bag_targets is None:
            self._reset_val_epoch_buffers()
            return
        target_type = str(self.task_cfg.get("target_type", "binary"))
        metrics = compute_task_metrics_from_tensors(
            target_type=target_type,
            logits=aggregated.bag_logits,
            targets=aggregated.bag_targets,
            threshold=0.5,
        )
        region_metrics = self._region_level_metrics(
            target_type, aggregated.bag_logits, aggregated.bag_targets
        )

        def _log(name: str, value: float, prog_bar: bool = False) -> None:
            self.log(
                f"{stage}/{name}",
                torch.tensor(float(value), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
            )

        if target_type in {"binary", "categorical"}:
            _log("acc", metrics["accuracy"], prog_bar=True)
            for name, value in region_metrics.items():
                _log(name, value, prog_bar=name == "auroc")
        elif target_type == "regression":
            _log("mae", metrics["mae"], prog_bar=True)
            _log("rmse", metrics["rmse"], prog_bar=True)
            _log("r2", metrics["r2"])
        elif target_type == "survival":
            _log("c_index", metrics["c_index"], prog_bar=True)

        if stage == "test":
            self.test_predictions = {
                "bag_ids": list(aggregated.bag_ids),
                "logits": aggregated.bag_logits.detach().cpu(),
                "targets": aggregated.bag_targets.detach().cpu(),
                "region_ids": list(getattr(aggregated, "row_region_ids", None) or []),
                "sample_ids": list(getattr(aggregated, "row_sample_ids", None) or []),
                "target_type": target_type,
            }
        self._reset_val_epoch_buffers()

    @staticmethod
    def _region_level_metrics(
        target_type: str, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Unweighted region loss, AUROC and balanced accuracy on aggregated region logits.

        These are the quantities the preprint selects and early-stops on: the loss
        of the region-level prediction (not of training sub-batches) and the
        AUROC of the positive-class probability.
        """
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score

        out: Dict[str, float] = {}
        logits = logits.detach().float().cpu()
        y = targets.detach().cpu()
        if target_type == "categorical":
            y_int = y.long().view(-1)
            out["loss_region"] = float(torch.nn.functional.cross_entropy(logits, y_int))
            probs = torch.softmax(logits, dim=1).numpy()
            yi = y_int.numpy()
            n_classes = probs.shape[1]
            try:
                if n_classes == 2:
                    out["auroc"] = float(roc_auc_score(yi, probs[:, 1]))
                else:
                    out["auroc"] = float(
                        roc_auc_score(yi, probs, multi_class="ovr", average="macro")
                    )
            except ValueError:
                out["auroc"] = float("nan")
            pred = probs.argmax(axis=1)
            out["bal_acc"] = (
                float(balanced_accuracy_score(yi, pred))
                if len(np.unique(yi)) > 1
                else float("nan")
            )
            out["pred_pos_frac"] = float((pred == 1).mean()) if n_classes == 2 else float("nan")
        elif target_type == "binary":
            yb = y.float().view(-1)
            lg = logits.view(-1)
            out["loss_region"] = float(
                torch.nn.functional.binary_cross_entropy_with_logits(lg, yb)
            )
            prob = torch.sigmoid(lg).numpy()
            yi = yb.numpy().astype(int)
            try:
                out["auroc"] = float(roc_auc_score(yi, prob))
            except ValueError:
                out["auroc"] = float("nan")
            pred = (prob >= 0.5).astype(int)
            out["bal_acc"] = (
                float(balanced_accuracy_score(yi, pred))
                if len(np.unique(yi)) > 1
                else float("nan")
            )
            out["pred_pos_frac"] = float(pred.mean())
        return out

    def on_train_epoch_end(self) -> None:
        if not self.region_accum_enabled:
            return
        flush_losses = self._flush_region_buffer_if_needed(force=True)
        for flush_loss in flush_losses:
            self.log(
                "train/epoch_end_flush_loss",
                flush_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if not self._built:
            self.setup("fit")
        if self.optim_cfg is None:
            raise ValueError(
                "Missing optimizer config for training. Compose an optim group "
                "(for example '+optim=adamw') or set model.optim explicitly."
            )

        params = self._build_optimizer_params()
        optimizer = instantiate_optimizer(self.optim_cfg, params)
        scheduler = instantiate_scheduler_with_warmup(
            self.scheduler_cfg,
            optimizer,
            warmup_cfg=self.task_cfg.get("lr_warmup"),
            t_max_auto=self._resolve_t_max_auto(),
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": 1,
            },
        }

    def _build_optimizer_params(self):
        """Parameters or parameter groups for the optimizer.

        ``task.optimization`` accepts either the per-component multipliers
        ``encoder_lr_scale`` / ``head_lr_scale`` / ``attention_lr_scale`` (with
        ``head_weight_decay``), or the absolute pair ``backbone_lr`` +
        ``attention_lr``. Setting only one of the absolute pair is an error rather
        than a silent no-op.
        """
        opt = self.optimization_cfg
        scale_keys = {
            "encoder_lr_scale",
            "head_lr_scale",
            "attention_lr_scale",
            "head_weight_decay",
        }
        has_scales = any(opt.get(k) is not None for k in scale_keys)
        backbone_lr = opt.get("backbone_lr")
        attention_lr = opt.get("attention_lr")
        if (backbone_lr is None) != (attention_lr is None):
            raise ValueError(
                "task.optimization.backbone_lr and attention_lr must be set together "
                "(or use encoder_lr_scale / head_lr_scale / attention_lr_scale)."
            )
        if has_scales and backbone_lr is not None:
            raise ValueError(
                "task.optimization mixes absolute (backbone_lr/attention_lr) and scaled "
                "learning rates; use one scheme."
            )

        base_cfg = (
            OmegaConf.to_container(self.optim_cfg, resolve=True)
            if isinstance(self.optim_cfg, DictConfig)
            else dict(self.optim_cfg)
        )
        base_lr = float(base_cfg.get("lr", 1e-3))
        base_wd = float(base_cfg.get("weight_decay", 0.0) or 0.0)

        if has_scales or self._encoder_frozen or self.freeze_head:
            groups = build_param_groups(
                base_lr=base_lr,
                base_weight_decay=base_wd,
                encoder=self.encoder,
                head=self.graph_head,
                attention=self.attention if self.use_attention else None,
                encoder_lr_scale=0.0
                if self._encoder_frozen
                else float(opt.get("encoder_lr_scale", 1.0) or 0.0),
                head_lr_scale=float(opt.get("head_lr_scale", 1.0) or 0.0),
                attention_lr_scale=float(opt.get("attention_lr_scale", 1.0) or 0.0),
                head_weight_decay=opt.get("head_weight_decay"),
                freeze_head=self.freeze_head,
            )
            return groups

        params = [p for p in self.parameters() if p.requires_grad]
        if self.use_attention and backbone_lr is not None:
            # Absolute scheme: one backbone group (encoder + head) and one attention group.
            seen: set[int] = set()

            def _collect(module) -> list[torch.nn.Parameter]:
                out: list[torch.nn.Parameter] = []
                for p in module.parameters():
                    if p.requires_grad and id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
                return out

            backbone = _collect(self.encoder) + _collect(self.graph_head)
            attention = _collect(self.attention)
            groups: list[dict] = []
            if backbone:
                groups.append({"params": backbone, "lr": float(backbone_lr)})
            if attention:
                groups.append({"params": attention, "lr": float(attention_lr)})
            return groups or params
        return params

    def _resolve_t_max_auto(self) -> Optional[int]:
        """Optimizer steps for ``scheduler.T_max: auto``."""
        try:
            trainer = self.trainer
        except RuntimeError:
            return None
        if trainer is None:
            return None
        max_epochs = int(trainer.max_epochs or 1)
        if self.region_accum_enabled:
            n_units = None
            dm = getattr(trainer, "datamodule", None)
            ds = getattr(dm, "dataset_train", None) if dm is not None else None
            if ds is not None:
                try:
                    n_units = len(ds)
                except TypeError:
                    n_units = None
            if n_units is None:
                return None
            return resolve_t_max_auto(
                num_train_units=int(n_units),
                max_epochs=max_epochs,
                hyperbatch_size=self.region_accum_hyperbatch_size,
            )
        try:
            return int(trainer.estimated_stepping_batches)
        except Exception:
            return None

    def _reset_val_epoch_buffers(self) -> None:
        self._val_bag_ids: list[str] = []
        self._val_bag_logits_chunks: list[torch.Tensor] = []
        self._val_bag_targets_chunks: list[torch.Tensor] = []
        self._val_row_region_ids: list[Optional[str]] = []
        self._val_row_sample_ids: list[Optional[str]] = []
        self._val_instance_logits_chunks: list[torch.Tensor] = []
        self._val_instance_attention_logits_chunks: list[torch.Tensor] = []
        self._val_instance_patch_ids: list[str] = []
        self._val_instance_region_ids: list[Optional[str]] = []
        self._val_instance_sample_ids: list[Optional[str]] = []

    @staticmethod
    def _cat_or_none(chunks: list[torch.Tensor]) -> Optional[torch.Tensor]:
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def _gather_validation_objects(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        if not dist.is_available() or not dist.is_initialized():
            return [obj]
        gathered: list[dict[str, Any]] = [None] * dist.get_world_size()  # type: ignore[list-item]
        dist.all_gather_object(gathered, obj)
        return gathered

    def _build_validation_epoch_payload(self) -> Optional[BatchPredictionPayload]:
        local_payload = {
            "bag_ids": self._val_bag_ids,
            "bag_logits": self._cat_or_none(self._val_bag_logits_chunks),
            "bag_targets": self._cat_or_none(self._val_bag_targets_chunks),
            "row_region_ids": self._val_row_region_ids,
            "row_sample_ids": self._val_row_sample_ids,
            "instance_logits": self._cat_or_none(self._val_instance_logits_chunks),
            "instance_attention_logits": self._cat_or_none(
                self._val_instance_attention_logits_chunks
            ),
            "instance_patch_ids": self._val_instance_patch_ids,
            "instance_region_ids": self._val_instance_region_ids,
            "instance_sample_ids": self._val_instance_sample_ids,
        }
        gathered = self._gather_validation_objects(local_payload)

        bag_ids: list[str] = []
        bag_logits_chunks: list[torch.Tensor] = []
        bag_targets_chunks: list[torch.Tensor] = []
        row_region_ids: list[Optional[str]] = []
        row_sample_ids: list[Optional[str]] = []
        instance_logits_chunks: list[torch.Tensor] = []
        instance_attention_logits_chunks: list[torch.Tensor] = []
        instance_patch_ids: list[str] = []
        instance_region_ids: list[Optional[str]] = []
        instance_sample_ids: list[Optional[str]] = []

        for chunk in gathered:
            chunk_bag_logits = chunk.get("bag_logits")
            if chunk_bag_logits is not None and int(chunk_bag_logits.shape[0]) > 0:
                bag_logits_chunks.append(chunk_bag_logits)
                bag_ids.extend([str(v) for v in chunk["bag_ids"]])
                row_region_ids.extend(
                    [None if v is None else str(v) for v in chunk["row_region_ids"]]
                )
                row_sample_ids.extend(
                    [None if v is None else str(v) for v in chunk["row_sample_ids"]]
                )
            chunk_bag_targets = chunk.get("bag_targets")
            if chunk_bag_targets is not None and int(chunk_bag_targets.shape[0]) > 0:
                bag_targets_chunks.append(chunk_bag_targets)
            chunk_instance_logits = chunk.get("instance_logits")
            if chunk_instance_logits is not None and int(chunk_instance_logits.shape[0]) > 0:
                instance_logits_chunks.append(chunk_instance_logits)
                instance_patch_ids.extend([str(v) for v in chunk.get("instance_patch_ids", [])])
                instance_region_ids.extend(
                    [None if v is None else str(v) for v in chunk.get("instance_region_ids", [])]
                )
                instance_sample_ids.extend(
                    [None if v is None else str(v) for v in chunk.get("instance_sample_ids", [])]
                )
            chunk_instance_attn = chunk.get("instance_attention_logits")
            if chunk_instance_attn is not None and int(chunk_instance_attn.shape[0]) > 0:
                instance_attention_logits_chunks.append(chunk_instance_attn)

        if not bag_logits_chunks:
            return None
        bag_targets = self._cat_or_none(bag_targets_chunks)
        instance_logits = self._cat_or_none(instance_logits_chunks)
        instance_attention_logits = self._cat_or_none(instance_attention_logits_chunks)
        return BatchPredictionPayload(
            bag_ids=bag_ids,
            bag_logits=torch.cat(bag_logits_chunks, dim=0),
            bag_targets=bag_targets,
            bag_attention=None,
            row_region_ids=row_region_ids,
            row_sample_ids=row_sample_ids,
            instance_logits=instance_logits,
            instance_attention_logits=instance_attention_logits,
            instance_patch_ids=instance_patch_ids if instance_logits is not None else None,
            instance_region_ids=instance_region_ids if instance_logits is not None else None,
            instance_sample_ids=instance_sample_ids if instance_logits is not None else None,
        )
