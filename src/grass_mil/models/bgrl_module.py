from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F

from .components import EncoderConfig, build_encoder, build_ssl
from .training import (
    CosineWarmup,
    build_augmentation,
    infer_encoder_input_dim,
    resolve_encoder_cfg,
    instantiate_optimizer,
    instantiate_scheduler_with_warmup,
    load_state_dict_with_optional_mapping,
)


def _molecular_feature_names(module) -> Optional[list]:
    """Feature names from the datamodule, for name-based column resolution."""
    try:
        trainer = module.trainer
    except RuntimeError:
        return None
    datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
    dataset = getattr(datamodule, "dataset_train", None)
    if dataset is None or len(dataset) == 0:
        return None
    names = getattr(dataset[0], "molecular_feature_names", None)
    return list(names) if names else None


class BGRLModule(L.LightningModule):
    def __init__(
        self,
        encoder: Dict[str, Any],
        ssl: Dict[str, Any],
        optim: Optional[Dict[str, Any]] = None,
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

        # Per-view uniform drop rates, used when no augmentation config is set.
        self._view_params = (
            {
                "mode": "uniform",
                "drop_edge_p": float(self.task_cfg.get("drop_edge_p1", 0.0)),
                "drop_feat_p": float(self.task_cfg.get("drop_feat_p1", 0.0)),
            },
            {
                "mode": "uniform",
                "drop_edge_p": float(self.task_cfg.get("drop_edge_p2", 0.0)),
                "drop_feat_p": float(self.task_cfg.get("drop_feat_p2", 0.0)),
            },
        )
        self._augmentation_seed = int(self.task_cfg.get("augmentation_seed", 0))
        self._view_generator: Optional[torch.Generator] = None
        self._momentum_base = float(self.task_cfg.get("momentum", 0.99))
        self._momentum_min = float(self.task_cfg.get("momentum_min", 0.99))
        self._warmup_steps = int(self.task_cfg.get("warmup_steps", 0))
        self._total_steps = int(self.task_cfg.get("total_steps", 0))

        self._momentum_scheduler: Optional[CosineWarmup] = None
        # When set, replaces the uniform drop_edge/drop_feat views.
        self._augmentation_cfg = self.task_cfg.get("augmentation")
        self._augmentation = None
        self._fallback_augmentations = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._built:
            return
        encoder_cfg = resolve_encoder_cfg(
            dict(self.hparams.encoder), infer_encoder_input_dim(self), module=self
        )
        self.encoder = build_encoder(EncoderConfig(**encoder_cfg))
        input_proj = getattr(self.encoder, "input_proj", None)
        categorical = getattr(input_proj, "categorical", None)
        build_kwargs = {
            "feature_names": _molecular_feature_names(self),
            "unassigned_index": getattr(input_proj, "unassigned_index", None),
            "categorical_column": (None if categorical is None else categorical.column_index or 0),
        }
        if self._augmentation_cfg:
            self._augmentation = build_augmentation(self._augmentation_cfg, **build_kwargs)
        else:
            self._fallback_augmentations = tuple(
                build_augmentation(params, **build_kwargs) for params in self._view_params
            )
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
            getattr(batch, "categorical_codes", None),
        )

    def _view_rng(self) -> torch.Generator:
        """Generator for the augmentation draws.

        Seeded once from ``task.augmentation_seed`` and then advanced, so the
        two views of a batch differ while the whole pretraining run stays
        reproducible. Without an explicit generator the augmentations would
        consume the global RNG and silently depend on unrelated call order.
        """
        if self._view_generator is None:
            self._view_generator = torch.Generator()
            self._view_generator.manual_seed(self._augmentation_seed)
        return self._view_generator

    def _make_views(self, batch):
        generator = self._view_rng()
        if self._augmentation is not None:
            return self._augmentation(batch, generator), self._augmentation(batch, generator)
        first, second = self._fallback_augmentations
        return first(batch, generator), second(batch, generator)

    def _step(self, batch) -> torch.Tensor:
        view1, view2 = self._make_views(batch)
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
        self.log("train/momentum", momentum, on_step=True, on_epoch=False, prog_bar=False)

    def configure_optimizers(self):
        if not self._built:
            self.setup("fit")
        if self.optim_cfg is None:
            raise ValueError(
                "Missing optimizer config for training. Compose an optim group "
                "(for example '+optim=adamw') or set model.optim explicitly."
            )
        optimizer = instantiate_optimizer(self.optim_cfg, self.ssl_model.trainable_parameters())
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
