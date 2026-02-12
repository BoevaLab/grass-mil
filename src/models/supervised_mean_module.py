from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L

from .runtime import (
    aggregate_bag_logits_mean,
    build_supervised_components,
    compute_binary_accuracy,
    compute_supervised_loss,
    extract_bag_ids,
    gather_bag_targets,
    group_instance_indices_by_bag,
    instantiate_optimizer,
    instantiate_scheduler,
    load_state_dict_with_optional_mapping,
    validate_task_config,
)


class SupervisedMeanModule(L.LightningModule):
    def __init__(
        self,
        encoder: Dict[str, Any],
        graph_head: Dict[str, Any],
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

    def setup(self, stage: Optional[str] = None) -> None:
        if self._built:
            return
        loss_cfg = dict(self.hparams.loss)
        loss_cfg.setdefault("loss_type", self.task_cfg.get("loss", "categorical_bce"))
        components = build_supervised_components(
            module=self,
            encoder_cfg=dict(self.hparams.encoder),
            graph_head_cfg=dict(self.hparams.graph_head),
            attention_cfg=None,
            use_attention=False,
            loss_cfg=loss_cfg,
            use_ssl=bool(self.flags_cfg.get("use_ssl", False)),
            ssl_cfg=dict(self.ssl_cfg),
        )
        self.encoder = components["encoder"]
        self.graph_head = components["graph_head"]
        self.loss_fn = components["loss_fn"]
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
        node_emb, graph_emb = self.encoder(
            batch.x,
            batch.edge_index,
            edge_attr=getattr(batch, "edge_attr", None),
            batch=getattr(batch, "batch", None),
            return_graph_embedding=True,
        )
        patch_logits = self.graph_head(graph_emb)
        return patch_logits, graph_emb

    def _shared_step(self, batch, stage: str):
        patch_logits, _ = self._forward_instances(batch)
        bag_ids = extract_bag_ids(
            batch,
            bag_key=self.task_cfg.get("bag_key", "region_id"),
            bag_fallback_key=self.task_cfg.get("bag_fallback_key", "sample_id"),
        )
        bag_groups = group_instance_indices_by_bag(bag_ids)
        bag_logits, ordered_bag_ids, bag_indices = aggregate_bag_logits_mean(
            patch_logits,
            bag_groups,
            max_instances_per_bag=int(self.task_cfg.get("max_instances_per_bag", 0)),
            instance_sampling=self.task_cfg.get("instance_sampling", "all"),
        )
        bag_targets, bag_weights = gather_bag_targets(
            batch=batch,
            bag_indices=bag_indices,
            target_columns=self.task_cfg.get("target_columns"),
        )
        loss = compute_supervised_loss(
            loss_fn=self.loss_fn,
            task_cfg=self.task_cfg,
            bag_logits=bag_logits,
            bag_targets=bag_targets,
            bag_weights=bag_weights,
        )
        if stage in {"train", "val", "test"}:
            self.log(
                f"{stage}/loss",
                loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=True,
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
            "loss": loss,
            "bag_ids": ordered_bag_ids,
            "bag_logits": bag_logits.detach(),
            "bag_targets": bag_targets.detach(),
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")["loss"]

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._shared_step(batch, "predict")
        return {k: out[k] for k in ["bag_ids", "bag_logits", "bag_targets"]}

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
