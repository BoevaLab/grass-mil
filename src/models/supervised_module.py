from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightning as L
import torch
import torch.distributed as dist
from src.inference.aggregation import aggregate_group_logits
from src.inference.metrics import compute_task_metrics_from_tensors
from src.inference.schemas import BatchPredictionPayload

from .training import (
    aggregate_bag_logits_attention,
    aggregate_bag_logits_mean,
    build_mil_aux_targets,
    build_supervised_components,
    compute_aux_node_loss,
    compute_binary_accuracy,
    compute_categorical_accuracy,
    compute_entropy_regularization,
    compute_supervised_loss,
    extract_bag_ids,
    gather_bag_targets,
    gather_instance_logits,
    group_instance_indices_by_bag,
    instantiate_optimizer,
    instantiate_scheduler_with_warmup,
    load_state_dict_with_optional_mapping,
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
            self.region_accum_enabled = bool(
                self.region_accum_cfg.get("enabled", False)
            )
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
            self.node_aux_cfg = dict(self.task_cfg.get("node_aux", {}))
            self.entropy_reg_cfg = dict(self.task_cfg.get("entropy_reg", {}))
            self.loss_weights_cfg = dict(self.task_cfg.get("loss_weights", {}))
        else:
            self.region_accum_cfg = {}
            self.region_accum_enabled = False
            self.region_accum_hyperbatch_size = 0
            self.region_accum_flush_on_epoch_end = False
            self.node_aux_cfg = {}
            self.entropy_reg_cfg = {}
            self.loss_weights_cfg = {}

        self._region_total_loss_buffer: list[torch.Tensor] = []
        self._manual_optimizer_steps = 0
        # In inference, callers can disable instance-level payload emission to reduce
        # peak memory when aggregation only needs bag-level outputs.
        self._predict_emit_instance_payload = True
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
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self._built = True

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward_instances(self, batch):
        _, graph_emb = self.encoder(
            batch.x,
            batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=getattr(batch, "batch", None),
            return_graph_embedding=True,
        )
        patch_logits = self.graph_head(graph_emb)
        return patch_logits, graph_emb

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
        instance_region_ids: list[Optional[str]] = []
        instance_sample_ids: list[Optional[str]] = []
        instance_attention_logits_chunks: list[torch.Tensor] = []
        compute_attention_logits = (
            include_instance_payload and self.use_attention and self.attention is not None
        )

        for bag_id, indices in zip(ordered_bag_ids, bag_indices):
            if not indices:
                continue
            idx = torch.tensor(indices, dtype=torch.long, device=patch_logits.device)
            sub_logits = patch_logits.index_select(0, idx)
            instance_logits_chunks.append(sub_logits)
            instance_patch_ids.extend(
                [
                    patch_ids[i] if patch_ids[i] is not None else str(bag_id)
                    for i in indices
                ]
            )
            instance_region_ids.extend([region_ids[i] for i in indices])
            instance_sample_ids.extend([sample_ids[i] for i in indices])
            if compute_attention_logits:
                sub_emb = graph_emb.index_select(0, idx)
                attn_logits, _ = self.attention(sub_emb)
                instance_attention_logits_chunks.append(attn_logits.reshape(-1, 1))

        metadata["instance_logits"] = (
            torch.cat(instance_logits_chunks, dim=0)
            if instance_logits_chunks
            else torch.empty((0, patch_logits.shape[-1]), device=patch_logits.device)
        )
        metadata["instance_patch_ids"] = instance_patch_ids
        metadata["instance_region_ids"] = instance_region_ids
        metadata["instance_sample_ids"] = instance_sample_ids
        if compute_attention_logits:
            metadata["instance_attention_logits"] = (
                torch.cat(instance_attention_logits_chunks, dim=0)
                if instance_attention_logits_chunks
                else torch.empty((0, 1), device=patch_logits.device)
            )
        return metadata

    def collect_graph_embeddings(
        self, batch, *, return_node_embeddings: bool = False
    ) -> Dict[str, Any]:
        node_emb, graph_emb = self.encoder(
            batch.x,
            batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=getattr(batch, "batch", None),
            return_graph_embedding=True,
        )
        bag_ids = extract_bag_ids(
            batch,
            bag_key=self.task_cfg.get("bag_key", "region_id"),
            bag_fallback_key=self.task_cfg.get("bag_fallback_key", "sample_id"),
        )
        bag_groups = group_instance_indices_by_bag(bag_ids)
        ordered_bag_ids = sorted(bag_groups.keys())
        bag_graph_embeddings = []
        bag_counts: list[int] = []
        for bag_id in ordered_bag_ids:
            idx = torch.tensor(
                bag_groups[bag_id], dtype=torch.long, device=graph_emb.device
            )
            bag_graph_embeddings.append(
                graph_emb.index_select(0, idx).mean(dim=0, keepdim=True)
            )
            bag_counts.append(int(idx.numel()))

        payload: Dict[str, Any] = {
            "bag_ids": ordered_bag_ids,
            "graph_embeddings": torch.cat(bag_graph_embeddings, dim=0),
            "bag_counts": bag_counts,
        }
        if return_node_embeddings:
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
        patch_logits, graph_emb = self._forward_instances(batch)

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
                    max_instances_per_bag=int(
                        self.task_cfg.get("max_instances_per_bag", 0)
                    ),
                    instance_sampling=self.task_cfg.get("instance_sampling", "all"),
                )
            )
        else:
            bag_logits, ordered_bag_ids, bag_indices = aggregate_bag_logits_mean(
                patch_logits,
                bag_groups,
                max_instances_per_bag=int(
                    self.task_cfg.get("max_instances_per_bag", 0)
                ),
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
            raise ValueError(
                "Batch is missing graph_y required for supervised loss computation."
            )

        return {
            "patch_logits": patch_logits,
            "graph_emb": graph_emb,
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

    def _compute_optional_terms(
        self,
        *,
        patch_logits: torch.Tensor,
        bag_targets: torch.Tensor,
        bag_ids: list[str],
        bag_indices: list[list[int]],
        bag_attention: dict[str, torch.Tensor],
        stage: str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute node auxiliary loss and entropy regularisation.

        Only active during training when MIL attention is enabled and the
        relevant sub-configs are turned on.
        """
        if not self.use_attention:
            return None, None
        if stage != "train" or self.task_cfg.get("target_type") != "binary":
            return None, None

        node_aux_enabled = bool(self.node_aux_cfg.get("enabled", False))
        entropy_enabled = bool(self.entropy_reg_cfg.get("enabled", False))
        if not node_aux_enabled and not entropy_enabled:
            return None, None

        aux_logits = gather_instance_logits(patch_logits, bag_indices)
        aux_targets, aux_weights = build_mil_aux_targets(
            bag_targets=bag_targets,
            bag_indices=bag_indices,
            bag_ids=bag_ids,
            bag_attention=bag_attention,
            target_mode=str(
                self.node_aux_cfg.get("target_mode", "attention_shaped_ti")
            ),
        )

        node_aux_loss = None
        if node_aux_enabled:
            loss_mode = str(self.node_aux_cfg.get("loss_mode", "bce"))
            node_aux_loss = compute_aux_node_loss(
                aux_logits=aux_logits,
                aux_targets=aux_targets,
                aux_weights=aux_weights if loss_mode == "weighted_bce" else None,
                loss_mode=loss_mode,
            )

        entropy_reg = None
        if entropy_enabled:
            entropy_mode = str(
                self.entropy_reg_cfg.get("mode", "attention_shaped_target")
            )
            if entropy_mode == "attention":
                entropy_values = torch.cat(
                    [bag_attention[bag_id].reshape(-1, 1) for bag_id in bag_ids],
                    dim=0,
                )
            elif entropy_mode == "attention_shaped_target":
                entropy_values = aux_targets
            else:
                raise ValueError(
                    "entropy_reg.mode must be one of "
                    "['attention', 'attention_shaped_target']"
                )
            entropy_reg = compute_entropy_regularization(entropy_values)

        return node_aux_loss, entropy_reg

    def _compose_total_loss(
        self,
        region_loss: torch.Tensor,
        node_aux_loss: Optional[torch.Tensor],
        entropy_reg: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_attention:
            return region_loss

        region_w = float(self.loss_weights_cfg.get("region", 1.0))
        node_w = float(
            self.node_aux_cfg.get("weight", self.loss_weights_cfg.get("node_aux", 1.0))
        )
        entropy_w = float(
            self.entropy_reg_cfg.get(
                "weight", self.loss_weights_cfg.get("entropy", 0.0)
            )
        )
        total = region_w * region_loss
        if node_aux_loss is not None:
            total = total + node_w * node_aux_loss
        if entropy_reg is not None:
            total = total + entropy_w * entropy_reg
        return total

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
    ) -> tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        region_loss = self._compute_region_loss(
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            bag_weights=bag_weights,
        )
        node_aux_loss, entropy_reg = self._compute_optional_terms(
            patch_logits=patch_logits,
            bag_targets=bag_targets,
            bag_ids=ordered_bag_ids,
            bag_indices=bag_indices,
            bag_attention=bag_attention or {},
            stage=stage,
        )
        total_loss = self._compose_total_loss(region_loss, node_aux_loss, entropy_reg)
        return region_loss, node_aux_loss, entropy_reg, total_loss

    def _log_stage_metrics(
        self,
        *,
        stage: str,
        loss_value: torch.Tensor,
        bag_logits: torch.Tensor,
        bag_targets: torch.Tensor,
        region_loss: Optional[torch.Tensor] = None,
        node_aux_loss: Optional[torch.Tensor] = None,
        entropy_reg: Optional[torch.Tensor] = None,
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
            if node_aux_loss is not None:
                self.log(
                    "train/node_aux_loss",
                    node_aux_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
            if entropy_reg is not None:
                self.log(
                    "train/entropy_reg",
                    entropy_reg,
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
        elif (
            stage in {"train", "test"}
            and self.task_cfg["target_type"] == "categorical"
        ):
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

        region_loss, node_aux_loss, entropy_reg, total_loss = self._compute_losses(
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
            loss_value=total_loss if stage == "train" else region_loss,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            region_loss=region_loss if self.use_attention else None,
            node_aux_loss=node_aux_loss,
            entropy_reg=entropy_reg,
        )

        result = {
            "loss": total_loss if stage == "train" else region_loss,
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
            result.update(
                {
                    "region_loss": region_loss,
                    "node_aux_loss": node_aux_loss,
                    "entropy_reg": entropy_reg,
                }
            )
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
        total_losses: list[torch.Tensor] = []
        node_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for idx, bag_id in enumerate(ordered_bag_ids):
            region_w = bag_weights[idx : idx + 1] if bag_weights is not None else None
            region_loss, node_aux_loss, entropy_reg, total_loss = self._compute_losses(
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
            total_losses.append(total_loss)
            if node_aux_loss is not None:
                node_losses.append(node_aux_loss.detach())
            if entropy_reg is not None:
                entropies.append(entropy_reg.detach())

        if not total_losses:
            return torch.zeros((), device=self.device)

        # Buffer one scalar per forward pass to avoid chunked backward passes over
        # the same computation graph.
        step_total_loss = torch.stack(total_losses).mean()
        self._region_total_loss_buffer.append(step_total_loss)

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
            float(len(self._region_total_loss_buffer)),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        mean_total_loss = step_total_loss.detach()
        mean_region_loss = torch.stack(region_losses).mean()
        mean_node_aux = torch.stack(node_losses).mean() if node_losses else None
        mean_entropy = torch.stack(entropies).mean() if entropies else None
        self._log_stage_metrics(
            stage="train",
            loss_value=mean_total_loss,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            region_loss=mean_region_loss,
            node_aux_loss=mean_node_aux,
            entropy_reg=mean_entropy,
        )

        return mean_total_loss

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, "val")
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
        self._val_instance_patch_ids.extend(
            [str(v) for v in metadata["instance_patch_ids"]]
        )
        self._val_instance_region_ids.extend(metadata["instance_region_ids"])
        self._val_instance_sample_ids.extend(metadata["instance_sample_ids"])
        if "instance_attention_logits" in metadata:
            self._val_instance_attention_logits_chunks.append(
                metadata["instance_attention_logits"].detach().cpu()
            )
        return out

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._forward_bags(batch, allow_missing_targets=True)
        emit_instance_payload = bool(
            getattr(self, "_predict_emit_instance_payload", True)
        )
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
            result["instance_region_ids"] = metadata["instance_region_ids"]
            result["instance_sample_ids"] = metadata["instance_sample_ids"]
        if self.use_attention:
            result["bag_attention"] = out["bag_attention"]
            if emit_instance_payload:
                result["instance_attention_logits"] = metadata[
                    "instance_attention_logits"
                ].detach()
        return result

    # ------------------------------------------------------------------
    # Region accumulation helpers (MIL only)
    # ------------------------------------------------------------------

    def _flush_region_buffer_if_needed(self, *, force: bool) -> list[torch.Tensor]:
        if not self._region_total_loss_buffer:
            return []
        if (
            not force
            and len(self._region_total_loss_buffer) < self.region_accum_hyperbatch_size
        ):
            return []

        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        flush_losses: list[torch.Tensor] = []

        while self._region_total_loss_buffer:
            if force:
                chunk = self._region_total_loss_buffer
            else:
                if (
                    len(self._region_total_loss_buffer)
                    < self.region_accum_hyperbatch_size
                ):
                    break
                chunk = self._region_total_loss_buffer[
                    : self.region_accum_hyperbatch_size
                ]

            total_loss = torch.stack(chunk, dim=0).mean()
            self.manual_backward(total_loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler = self.lr_schedulers()
            if scheduler is not None:
                if isinstance(scheduler, (list, tuple)):
                    for sch in scheduler:
                        sch.step()
                else:
                    scheduler.step()
            self._manual_optimizer_steps += 1
            self.log(
                "train/manual_optimizer_steps",
                float(self._manual_optimizer_steps),
                on_step=not force,
                on_epoch=force,
                prog_bar=False,
            )
            flush_losses.append(total_loss.detach())

            del self._region_total_loss_buffer[: len(chunk)]
            if force:
                break

        return flush_losses

    def on_train_start(self) -> None:
        if self.region_accum_enabled:
            self._region_total_loss_buffer.clear()

    def on_validation_epoch_start(self) -> None:
        self._reset_val_epoch_buffers()

    def on_validation_epoch_end(self) -> None:
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
        if target_type in {"binary", "categorical"}:
            self.log(
                "val/acc",
                torch.tensor(float(metrics["accuracy"]), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif target_type == "regression":
            self.log(
                "val/mae",
                torch.tensor(float(metrics["mae"]), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/rmse",
                torch.tensor(float(metrics["rmse"]), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val/r2",
                torch.tensor(float(metrics["r2"]), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        elif target_type == "survival":
            self.log(
                "val/c_index",
                torch.tensor(float(metrics["c_index"]), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self._reset_val_epoch_buffers()

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

        optimization_cfg = dict(self.task_cfg.get("optimization", {}))
        backbone_lr = optimization_cfg.get("backbone_lr")
        attention_lr = optimization_cfg.get("attention_lr")

        params = [p for p in self.parameters() if p.requires_grad]

        if self.use_attention and backbone_lr is not None and attention_lr is not None:
            seen: set[int] = set()

            def _collect_params(module) -> list[torch.nn.Parameter]:
                out: list[torch.nn.Parameter] = []
                if module is None:
                    return out
                for param in module.parameters():
                    if not param.requires_grad:
                        continue
                    pid = id(param)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    out.append(param)
                return out

            backbone_params = _collect_params(self.encoder) + _collect_params(
                self.graph_head
            )
            attention_params = _collect_params(self.attention)
            param_groups: list[dict] = []
            if backbone_params:
                param_groups.append(
                    {"params": backbone_params, "lr": float(backbone_lr)}
                )
            if attention_params:
                param_groups.append(
                    {"params": attention_params, "lr": float(attention_lr)}
                )
            if param_groups:
                params = param_groups

        optimizer = instantiate_optimizer(self.optim_cfg, params)
        scheduler = instantiate_scheduler_with_warmup(
            self.scheduler_cfg,
            optimizer,
            warmup_cfg=self.task_cfg.get("lr_warmup"),
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

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
            if (
                chunk_instance_logits is not None
                and int(chunk_instance_logits.shape[0]) > 0
            ):
                instance_logits_chunks.append(chunk_instance_logits)
                instance_patch_ids.extend(
                    [str(v) for v in chunk.get("instance_patch_ids", [])]
                )
                instance_region_ids.extend(
                    [
                        None if v is None else str(v)
                        for v in chunk.get("instance_region_ids", [])
                    ]
                )
                instance_sample_ids.extend(
                    [
                        None if v is None else str(v)
                        for v in chunk.get("instance_sample_ids", [])
                    ]
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
