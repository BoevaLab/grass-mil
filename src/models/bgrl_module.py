from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F

from .components import EncoderConfig, build_encoder, build_ssl
from .training import (
    CosineWarmup,
    augment_graph,
    infer_encoder_input_dim,
    instantiate_optimizer,
    instantiate_scheduler_with_warmup,
    load_state_dict_with_optional_mapping,
)


class BGRLModule(L.LightningModule):
    def __init__(
        self,
        encoder: Dict[str, Any],
        ssl: Dict[str, Any],
        optim: Dict[str, Any],
        scheduler: Optional[Dict[str, Any]] = None,
        task: Optional[Dict[str, Any]] = None,
        init_from_ckpt: Optional[str] = None,
        init_strict: bool = False,
        encoder_init_map: str = "auto_bgrl_or_identity",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._built = False
        self.encoder = None
        self.ssl_model = None
        self.optim_cfg = optim
        self.scheduler_cfg = scheduler
        self.task_cfg = task or {}
        self.init_from_ckpt = init_from_ckpt
        self.init_strict = init_strict
        self.encoder_init_map = encoder_init_map

        self._drop_edge_p1 = float(self.task_cfg.get("drop_edge_p1", 0.0))
        self._drop_edge_p2 = float(self.task_cfg.get("drop_edge_p2", 0.0))
        self._drop_feat_p1 = float(self.task_cfg.get("drop_feat_p1", 0.0))
        self._drop_feat_p2 = float(self.task_cfg.get("drop_feat_p2", 0.0))
        self._momentum_base = float(self.task_cfg.get("momentum", 0.99))
        self._momentum_min = float(self.task_cfg.get("momentum_min", 0.99))
        self._warmup_steps = int(self.task_cfg.get("warmup_steps", 0))
        self._total_steps = int(self.task_cfg.get("total_steps", 0))

        self._momentum_scheduler: Optional[CosineWarmup] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._built:
            return
        encoder_cfg = dict(self.hparams.encoder)
        if encoder_cfg.get("input_dim", 0) in (None, 0):
            inferred = infer_encoder_input_dim(self)
            if inferred <= 0:
                raise ValueError(
                    "BGRLModule could not infer encoder.input_dim from datamodule."
                )
            encoder_cfg["input_dim"] = inferred
        self.encoder = build_encoder(EncoderConfig(**encoder_cfg))
        self.ssl_model = build_ssl(True, self.encoder, dict(self.hparams.ssl))
        if self.ssl_model is None:
            raise ValueError("BGRLModule requires ssl.method=bgrl config.")
        result = load_state_dict_with_optional_mapping(
            self,
            init_from_ckpt=self.init_from_ckpt,
            init_strict=self.init_strict,
            encoder_init_map=self.encoder_init_map,
        )
        if result["missing_keys"] or result["unexpected_keys"]:
            self.log(
                "init/state_dict_warnings",
                float(len(result["missing_keys"]) + len(result["unexpected_keys"])),
                prog_bar=False,
            )
        total_steps = self._total_steps
        if total_steps <= 0 and self.trainer is not None:
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
            # Give a warning if total_steps is not set
            self.log("init/total_steps_warning", float(total_steps), prog_bar=False)
        self._momentum_scheduler = CosineWarmup(
            base_value=self._momentum_base,
            warmup_steps=max(0, self._warmup_steps),
            total_steps=max(1, total_steps),
            min_value=self._momentum_min,
        )
        self._built = True

    def _encode_tuple(self, batch):
        return (
            batch.x,
            batch.edge_index,
            getattr(batch, "edge_attr", None),
            getattr(batch, "batch", None),
        )

    def _step(self, batch) -> torch.Tensor:
        view1 = augment_graph(
            batch,
            drop_edge_p=self._drop_edge_p1,
            drop_feat_p=self._drop_feat_p1,
        )
        view2 = augment_graph(
            batch,
            drop_edge_p=self._drop_edge_p2,
            drop_feat_p=self._drop_feat_p2,
        )
        q1, y2 = self.ssl_model(self._encode_tuple(view1), self._encode_tuple(view2))
        q2, y1 = self.ssl_model(self._encode_tuple(view2), self._encode_tuple(view1))
        loss = (
            2.0
            - F.cosine_similarity(q1, y2.detach(), dim=-1).mean()
            - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ssl_model is None or self._momentum_scheduler is None:
            return
        # Lightning increments global_step after optimizer.step(); map back to zero-based step.
        step = max(int(self.global_step) - 1, 0)
        momentum = self._momentum_scheduler.value(step)
        self.ssl_model.update_target_network(momentum)
        self.log(
            "train/momentum", momentum, on_step=True, on_epoch=False, prog_bar=False
        )

    def configure_optimizers(self):
        if not self._built:
            self.setup("fit")
        optimizer = instantiate_optimizer(
            self.optim_cfg, self.ssl_model.trainable_parameters()
        )
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
                "interval": "step",
                "frequency": 1,
            },
        }
