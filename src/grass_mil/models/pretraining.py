"""Label-free BGRL pretraining of the ego-graph encoder.

Two augmented views of every ego-graph batch are encoded node by node by an
online encoder (plus a predictor) and by a target encoder that is an
exponential moving average of the online one; the loss is the symmetric
negative cosine similarity between the online predictions and the target node
embeddings. The EMA momentum follows a
cosine schedule from ``momentum.start`` to ``momentum.end`` over the run.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import OmegaConf
from torch import nn

from grass_mil.data.datamodule import InputSpec

from .augmentations import PretrainViews
from .checkpoints import load_encoder_weights
from .encoder import EncoderConfig, GINEEncoder
from .optimization import build_optimizer, build_scheduler

__all__ = ["PretrainConfig", "BGRLModule"]


@dataclass(frozen=True)
class PretrainConfig:
    """Attributes:
    total_steps: Optimizer steps of the run (also ``trainer.max_steps``).
    warmup_steps: Linear learning-rate warmup steps.
    momentum_start / momentum_end: EMA momentum of the target encoder.
    predictor_hidden_dim: Hidden width of the predictor MLP.
    augmentation: The two views.
    seed: Seed of the augmentation draws.
    """

    total_steps: int = 200_000
    warmup_steps: int = 1000
    momentum_start: float = 0.99
    momentum_end: float = 1.0
    predictor_hidden_dim: int = 512
    augmentation: PretrainViews = PretrainViews()
    seed: int = 0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> PretrainConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown pretrain keys: {sorted(unknown)}")
        cfg["augmentation"] = PretrainViews.from_dict(cfg.get("augmentation"))
        return cls(**cfg)

    def momentum(self, step: int) -> float:
        progress = min(max(step, 0), self.total_steps) / max(1, self.total_steps)
        return self.momentum_end - (self.momentum_end - self.momentum_start) * 0.5 * (1 + math.cos(math.pi * progress))


class Predictor(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.PReLU(1), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


class BGRLModule(L.LightningModule):
    def __init__(
        self,
        encoder: Mapping[str, Any],
        pretrain: Mapping[str, Any],
        optim: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
        init_from_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        hparams = {k: _plain(v) for k, v in locals().items() if k not in ("self", "__class__")}
        self.save_hyperparameters(hparams, logger=False)
        self.encoder_cfg = EncoderConfig.from_dict(hparams["encoder"])
        self.pretrain = PretrainConfig.from_dict(hparams["pretrain"])
        self.input_spec: InputSpec | None = None
        self.encoder: GINEEncoder | None = None
        self.target_encoder: GINEEncoder | None = None
        self.predictor: Predictor | None = None
        self._view = None
        self._generator = torch.Generator().manual_seed(int(self.pretrain.seed))
        self._last_ema_step = 0

    def build(self, spec: InputSpec, feature_names: list[str] | None = None) -> None:
        self.input_spec = spec
        self.encoder = GINEEncoder(
            self.encoder_cfg, feature_dim=spec.feature_dim, cell_types=spec.cell_types, edge_dim=spec.edge_dim
        )
        self.predictor = Predictor(self.encoder.output_dim, self.pretrain.predictor_hidden_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self._view = self.pretrain.augmentation.build(
            feature_names=feature_names or [], unassigned_index=self.encoder.unassigned_index
        )

    def setup(self, stage: str | None = None) -> None:
        if self.encoder is not None:
            return
        datamodule = self.trainer.datamodule
        self.build(datamodule.input_spec, feature_names=datamodule.index.feature_names)
        if self.hparams.init_from_ckpt:
            load_encoder_weights(self.encoder, self.hparams.init_from_ckpt)
            self.target_encoder.load_state_dict(self.encoder.state_dict())
        if self.trainer.max_steps not in (-1, None) and self.trainer.max_steps != self.pretrain.total_steps:
            rank_zero_warn(
                f"trainer.max_steps={self.trainer.max_steps} differs from pretrain.total_steps="
                f"{self.pretrain.total_steps}; the schedules follow pretrain.total_steps."
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["input_spec"] = self.input_spec.as_dict()

    def _loss(self, batch) -> torch.Tensor:
        view1, view2 = self._view(batch, self._generator), self._view(batch, self._generator)
        q1 = self.predictor(self.encoder.node_embeddings(view1))
        q2 = self.predictor(self.encoder.node_embeddings(view2))
        with torch.no_grad():
            y1 = self.target_encoder.node_embeddings(view1)
            y2 = self.target_encoder.node_embeddings(view2)
        return 2.0 - F.cosine_similarity(q1, y2, dim=-1).mean() - F.cosine_similarity(q2, y1, dim=-1).mean()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=int(batch.num_graphs))
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        self.log("val/loss", self._loss(batch), prog_bar=True, batch_size=int(batch.num_graphs))

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """EMA update of the target encoder after every optimizer step."""
        step = int(self.global_step)
        if step <= self._last_ema_step:
            return
        self._last_ema_step = step
        momentum = self.pretrain.momentum(step - 1)
        with torch.no_grad():
            for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
        self.log("train/momentum", momentum, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        optimizer = build_optimizer(self.hparams.optim, [{"params": params}])
        scheduler = build_scheduler(
            self.hparams.scheduler,
            optimizer,
            total_steps=self.pretrain.total_steps,
            warmup_steps=self.pretrain.warmup_steps,
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
