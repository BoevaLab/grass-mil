from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch

from .training import (
    aggregate_bag_logits_attention,
    build_mil_aux_targets,
    build_supervised_components,
    compute_aux_node_loss,
    compute_binary_accuracy,
    compute_entropy_regularization,
    compute_supervised_loss,
    extract_bag_ids,
    gather_bag_targets,
    gather_instance_logits,
    group_instance_indices_by_bag,
    instantiate_optimizer,
    instantiate_scheduler,
    load_state_dict_with_optional_mapping,
    validate_task_config,
)


class SupervisedMILModule(L.LightningModule):
    def __init__(
        self,
        encoder: Dict[str, Any],
        graph_head: Dict[str, Any],
        attention: Dict[str, Any],
        loss: Dict[str, Any],
        optim: Dict[str, Any],
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
                "task.region_accumulation.hyperbatch_size must be >= 1 when region accumulation is enabled."
            )
        self.node_aux_cfg = dict(self.task_cfg.get("node_aux", {}))
        self.entropy_reg_cfg = dict(self.task_cfg.get("entropy_reg", {}))
        self.loss_weights_cfg = dict(self.task_cfg.get("loss_weights", {}))

        self._region_total_loss_buffer: list[torch.Tensor] = []
        self._manual_optimizer_steps = 0

    @property
    def automatic_optimization(self) -> bool:
        return not self.region_accum_enabled

    def setup(self, stage: Optional[str] = None) -> None:
        if self._built:
            return
        loss_cfg = dict(self.hparams.loss)
        loss_cfg.setdefault("loss_type", self.task_cfg.get("loss", "categorical_bce"))
        attention_cfg = dict(self.hparams.attention)
        attention_cfg.pop("_target_", None)
        components = build_supervised_components(
            module=self,
            encoder_cfg=dict(self.hparams.encoder),
            graph_head_cfg=dict(self.hparams.graph_head),
            attention_cfg=attention_cfg,
            use_attention=True,
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

    def _forward_bags(self, batch):
        patch_logits, graph_emb = self._forward_instances(batch)
        bag_ids = extract_bag_ids(
            batch,
            bag_key=self.task_cfg.get("bag_key", "region_id"),
            bag_fallback_key=self.task_cfg.get("bag_fallback_key", "sample_id"),
        )
        bag_groups = group_instance_indices_by_bag(bag_ids)
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
        bag_targets, bag_weights = gather_bag_targets(
            batch=batch,
            bag_indices=bag_indices,
            target_columns=self.task_cfg.get("target_columns"),
        )
        return {
            "patch_logits": patch_logits,
            "bag_logits": bag_logits,
            "ordered_bag_ids": ordered_bag_ids,
            "bag_attention": bag_attention,
            "bag_indices": bag_indices,
            "bag_targets": bag_targets,
            "bag_weights": bag_weights,
        }

    def _shared_step(self, batch, stage: str):
        payload = self._forward_bags(batch)
        patch_logits = payload["patch_logits"]
        bag_logits = payload["bag_logits"]
        ordered_bag_ids = payload["ordered_bag_ids"]
        bag_attention = payload["bag_attention"]
        bag_indices = payload["bag_indices"]
        bag_targets = payload["bag_targets"]
        bag_weights = payload["bag_weights"]
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
            bag_attention=bag_attention,
            stage=stage,
        )
        total_loss = self._compose_total_loss(region_loss, node_aux_loss, entropy_reg)
        if stage in {"train", "val", "test"}:
            self.log(
                f"{stage}/loss",
                total_loss if stage == "train" else region_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=True,
            )
            if stage == "train":
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
        if self.task_cfg["target_type"] == "binary" and stage in {
            "train",
            "val",
            "test",
        }:
            acc = compute_binary_accuracy(bag_logits, bag_targets)
            self.log(
                f"{stage}/acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=stage != "train",
            )
        return {
            "loss": total_loss if stage == "train" else region_loss,
            "bag_ids": list(ordered_bag_ids),
            "bag_logits": bag_logits,
            "bag_targets": bag_targets,
            "bag_attention": bag_attention,
            "region_loss": region_loss,
            "node_aux_loss": node_aux_loss,
            "entropy_reg": entropy_reg,
        }

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
                    [bag_attention[bag_id].reshape(-1, 1) for bag_id in bag_ids], dim=0
                )
            elif entropy_mode == "attention_shaped_target":
                entropy_values = aux_targets
            else:
                raise ValueError(
                    "entropy_reg.mode must be one of ['attention', 'attention_shaped_target']"
                )
            entropy_reg = compute_entropy_regularization(entropy_values)

        return node_aux_loss, entropy_reg

    def _compose_total_loss(
        self,
        region_loss: torch.Tensor,
        node_aux_loss: Optional[torch.Tensor],
        entropy_reg: Optional[torch.Tensor],
    ) -> torch.Tensor:
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

    def training_step(self, batch, batch_idx):
        if not self.region_accum_enabled:
            return self._shared_step(batch, "train")["loss"]

        payload = self._forward_bags(batch)
        bag_logits = payload["bag_logits"]
        bag_targets = payload["bag_targets"]
        bag_weights = payload["bag_weights"]
        ordered_bag_ids = payload["ordered_bag_ids"]
        bag_indices = payload["bag_indices"]
        bag_attention = payload["bag_attention"]
        patch_logits = payload["patch_logits"]

        region_losses = []
        total_losses = []
        node_losses = []
        entropies = []
        for idx, bag_id in enumerate(ordered_bag_ids):
            region_w = bag_weights[idx : idx + 1] if bag_weights is not None else None
            region_loss = self._compute_region_loss(
                bag_logits=bag_logits[idx : idx + 1],
                bag_targets=bag_targets[idx : idx + 1],
                bag_weights=region_w,
            )
            node_aux_loss, entropy_reg = self._compute_optional_terms(
                patch_logits=patch_logits,
                bag_targets=bag_targets[idx : idx + 1],
                bag_ids=[bag_id],
                bag_indices=[bag_indices[idx]],
                bag_attention={bag_id: bag_attention[bag_id]},
                stage="train",
            )
            total_loss = self._compose_total_loss(
                region_loss, node_aux_loss, entropy_reg
            )
            self._region_total_loss_buffer.append(total_loss)
            region_losses.append(region_loss.detach())
            total_losses.append(total_loss.detach())
            if node_aux_loss is not None:
                node_losses.append(node_aux_loss.detach())
            if entropy_reg is not None:
                entropies.append(entropy_reg.detach())

        if not total_losses:
            return torch.zeros((), device=self.device)

        flush_loss = self._flush_region_buffer_if_needed(force=False)
        if flush_loss is not None:
            self.log(
                "train/flush_loss",
                flush_loss.detach(),
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
        self.log(
            "train/loss",
            torch.stack(total_losses).mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/region_loss",
            torch.stack(region_losses).mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        if node_losses:
            self.log(
                "train/node_aux_loss",
                torch.stack(node_losses).mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        if entropies:
            self.log(
                "train/entropy_reg",
                torch.stack(entropies).mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        if self.task_cfg["target_type"] == "binary":
            acc = compute_binary_accuracy(bag_logits, bag_targets)
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return torch.stack(total_losses).mean()

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._shared_step(batch, "predict")
        return {
            k: out[k].detach() if hasattr(out[k], "detach") else out[k]
            for k in ["bag_ids", "bag_logits", "bag_targets", "bag_attention"]
        }

    def _flush_region_buffer_if_needed(self, *, force: bool) -> Optional[torch.Tensor]:
        if not self._region_total_loss_buffer:
            return None
        if (
            not force
            and len(self._region_total_loss_buffer) < self.region_accum_hyperbatch_size
        ):
            return None

        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        total_loss = torch.stack(self._region_total_loss_buffer, dim=0).mean()
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
            on_epoch=False,
            prog_bar=False,
        )

        self._region_total_loss_buffer.clear()
        return total_loss.detach()

    def on_train_start(self) -> None:
        if self.region_accum_enabled:
            self._region_total_loss_buffer.clear()

    def on_train_epoch_end(self) -> None:
        if not self.region_accum_enabled:
            return
        if self.region_accum_flush_on_epoch_end:
            flush_loss = self._flush_region_buffer_if_needed(force=True)
            if flush_loss is not None:
                self.log(
                    "train/epoch_end_flush_loss",
                    flush_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

    def configure_optimizers(self):
        if not self._built:
            self.setup("fit")
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = instantiate_optimizer(self.optim_cfg, params)
        scheduler = instantiate_scheduler(self.scheduler_cfg, optimizer)
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
