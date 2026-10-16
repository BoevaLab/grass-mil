"""GRASS-MIL supervised module: ego-graph instances pooled by gated attention.

Each training batch is one bag (all sampled ego-graphs of one unit). Bag
logits of ``finetune.hyperbatch_size`` consecutive bags are stacked and the
loss is computed over the stack before one optimizer step, so a step sees
several units (which the Cox objective needs) while a bag's attention softmax
always covers the whole unit. The last, smaller stack of an epoch is stepped on
its last batch, before validation. Validation and test units may be scored in
several chunks; their instances are pooled per unit at the end of the epoch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

import lightning as L
import torch
import torch.distributed as dist

from grass_mil.data.datamodule import InputSpec
from grass_mil.utils import checked, plain

from .augmentations import FinetuneAugmentation
from .checkpoints import load_encoder_weights
from .encoder import EncoderConfig, GINEEncoder
from .mil import (
    AttentionConfig,
    GatedAttention,
    HeadConfig,
    InstanceHead,
    attention_pool,
    pool_by_bag,
)
from .objectives import Objective, class_weights_from_labels, output_dim, region_metrics
from .optimization import build_optimizer, build_scheduler
from .probe import ProbeConfig, collect_embeddings, fit_linear_head

__all__ = ["FinetuneConfig", "SupervisedModule"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinetuneConfig:
    """The fine-tuning procedure (set by the ``regime`` presets).

    Attributes:
        lr_scales: Learning-rate multipliers of ``encoder``, ``head`` and
            ``attention`` on ``optim.lr``; ``0`` freezes the component (a
            frozen encoder also stays in eval mode).
        head_weight_decay: Weight decay of the head (``None``: the optimizer's).
        head_init: ``probe`` (fit a linear probe on the frozen embeddings of
            the training units first) or ``random``.
        hyperbatch_size: Bags per optimizer step.
        grad_clip_norm: Gradient-norm clip per step (``None``: off).
        warmup_steps: Linear learning-rate warmup steps.
        schedule_epochs: Epochs the learning-rate schedule spans (``None``:
            ``trainer.max_epochs``). Cross-validation refits keep the schedule
            of the inner runs and stop early, at the selected epoch.
        augmentation: Edge drop / node masking during training.
        probe: Probe settings for ``head_init: probe``.
    """

    lr_scales: Mapping[str, float]
    head_weight_decay: float | None = None
    head_init: str = "probe"
    hyperbatch_size: int = 12
    grad_clip_norm: float | None = 1.0
    warmup_steps: int = 0
    schedule_epochs: int | None = None
    augmentation: FinetuneAugmentation = FinetuneAugmentation()
    probe: ProbeConfig = ProbeConfig()

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> FinetuneConfig:
        cfg = checked(cls, cfg, "finetune")
        scales = dict(cfg.get("lr_scales") or {})
        if set(scales) != {"encoder", "head", "attention"} or any(
            v is None for v in scales.values()
        ):
            raise ValueError(
                "finetune.lr_scales needs numeric encoder, head and attention scales."
            )
        cfg["lr_scales"] = {k: float(v) for k, v in scales.items()}
        cfg["augmentation"] = FinetuneAugmentation.from_dict(cfg.get("augmentation"))
        cfg["probe"] = ProbeConfig.from_dict(cfg.get("probe"))
        out = cls(**cfg)
        if out.head_init not in {"probe", "random"}:
            raise ValueError("finetune.head_init must be 'probe' or 'random'.")
        if out.hyperbatch_size < 1:
            raise ValueError("finetune.hyperbatch_size must be >= 1.")
        return out


class SupervisedModule(L.LightningModule):
    def __init__(
        self,
        encoder: Mapping[str, Any],
        head: Mapping[str, Any],
        attention: Mapping[str, Any],
        task: Mapping[str, Any],
        finetune: Mapping[str, Any],
        optim: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
        init_from_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        hparams = {k: plain(v) for k, v in locals().items() if k not in ("self", "__class__")}
        self.save_hyperparameters(hparams, logger=False)
        self.automatic_optimization = False
        self.encoder_cfg = EncoderConfig.from_dict(hparams["encoder"])
        self.head_cfg = HeadConfig.from_dict(hparams["head"])
        self.attention_cfg = AttentionConfig.from_dict(hparams["attention"])
        self.task = dict(hparams["task"])
        self.finetune = FinetuneConfig.from_dict(hparams["finetune"])
        self.num_logits = output_dim(self.task)
        self.input_spec: InputSpec | None = None
        self.encoder: GINEEncoder | None = None
        self.head: InstanceHead | None = None
        self.attention: GatedAttention | None = None
        self.objective = Objective(self.task)
        self._augment = None
        self._bag_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._eval_buffer: dict[str, list] = {}
        self.test_predictions: dict[str, Any] | None = None
        # Sampling settings of the training data, set by from_checkpoint().
        self.trained_sampler: dict[str, Any] | None = None

    # ------------------------------------------------------------- building
    @property
    def built(self) -> bool:
        return self.encoder is not None

    def build(self, spec: InputSpec) -> None:
        """Create the layers for the given node/edge inputs."""
        self.input_spec = spec
        self.encoder = GINEEncoder(
            self.encoder_cfg,
            feature_dim=spec.feature_dim,
            cell_types=spec.cell_types,
            edge_dim=spec.edge_dim,
        )
        width = self.encoder.output_dim
        self.head = InstanceHead(width, self.num_logits, self.head_cfg)
        self.attention = GatedAttention(width, self.num_logits, self.attention_cfg)
        if self.finetune.augmentation.enabled:
            self._augment = self.finetune.augmentation.build(
                unassigned_index=self.encoder.unassigned_index
            )

    def setup(self, stage: str | None = None) -> None:
        if self.built:
            return
        datamodule = self.trainer.datamodule
        self.build(datamodule.input_spec)
        if self.hparams.init_from_ckpt:
            load_encoder_weights(self.encoder, self.hparams.init_from_ckpt)
        if stage == "fit":
            if self.task["target_type"] == "classification":
                self._check_classes(datamodule)
            if self.finetune.head_init == "probe":
                self._fit_probe_head(datamodule)
        self._apply_freezing()

    def _check_classes(self, datamodule) -> None:
        """Labels must be classes ``0..C-1``; sets the class weights."""
        train = None
        for dataset in (datamodule.dataset_train, datamodule.dataset_val, datamodule.dataset_test):
            labels = [dataset.labels.classes(dataset.label_id(e)) for e in dataset.entries]
            outside = sorted({c for c in labels if not 0 <= c < self.num_logits})
            if outside:
                raise ValueError(
                    f"Class labels {outside} are outside 0..{self.num_logits - 1} "
                    "(task.num_classes)."
                )
            train = labels if train is None else train
        setting = self.task.get("class_weights")
        if setting == "auto":
            self.objective.class_weight = class_weights_from_labels(train, self.num_logits)
        elif setting is not None:
            self.objective.class_weight = torch.tensor([float(w) for w in setting])

    def _fit_probe_head(self, datamodule) -> None:
        if not self.head.is_linear:
            raise ValueError(
                "finetune.head_init=probe needs a linear head (model.head.num_layers=1)."
            )
        if not self.hparams.init_from_ckpt:
            log.warning(
                "Fitting the probe head on a randomly initialised encoder (no model.init_from_ckpt)."
            )
        cfg = self.finetune.probe
        loader = datamodule.loader(
            datamodule.dataset_train, roots_per_unit=cfg.roots_per_unit, seed=cfg.seed
        )
        device = self.trainer.strategy.root_device
        self.encoder.to(device)
        embeddings, targets = collect_embeddings(self.encoder, loader, device)
        self.encoder.cpu()
        weight, bias = fit_linear_head(
            embeddings, targets, kind=self.task["target_type"], num_logits=self.num_logits, cfg=cfg
        )
        with torch.no_grad():
            self.head.net[0].weight.copy_(weight)
            self.head.net[0].bias.copy_(bias)
        norm = float(weight.norm())
        log.info(
            f"Probe head fitted on {len(embeddings)} ego-graphs of "
            f"{len(datamodule.dataset_train)} units (weight norm {norm:.3g})."
        )
        if norm > 20 and self.finetune.lr_scales["encoder"] > 0:
            log.warning(
                f"The probe head's weights are large (norm {norm:.3g}; a well-pretrained encoder "
                "on a cohort of hundreds of units gives a few units), so small changes of the "
                "encoder move the logits a lot and fine-tuning may diverge. On a small cohort "
                "use regime=attention_only (frozen encoder); otherwise pretrain longer or lower "
                "finetune.probe.C (e.g. 0.01)."
            )

    def _apply_freezing(self) -> None:
        scales = self.finetune.lr_scales
        for name in ("encoder", "head", "attention"):
            if scales[name] == 0:
                getattr(self, name).requires_grad_(False)
        if scales["encoder"] == 0:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.built and self.finetune.lr_scales["encoder"] == 0:
            self.encoder.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["input_spec"] = self.input_spec.as_dict()

    @classmethod
    def from_checkpoint(
        cls, path: str, map_location: str | torch.device = "cpu"
    ) -> SupervisedModule:
        """Rebuild a trained module from its checkpoint alone (eval mode)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        hparams = dict(ckpt["hyper_parameters"])
        hparams["init_from_ckpt"] = None
        hparams["finetune"] = {**hparams["finetune"], "head_init": "random"}
        module = cls(**hparams)
        module.build(InputSpec.from_dict(ckpt["input_spec"]))
        module.load_state_dict(ckpt["state_dict"])
        module.trained_sampler = ckpt["datamodule_hyper_parameters"]["sampler"]
        return module.eval()

    # -------------------------------------------------------------- forward
    def forward_instances(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embeddings, instance logits and attention scores of every ego-graph."""
        if self.training and self._augment is not None:
            batch = self._augment(batch)
        embeddings = self.encoder(batch)
        return embeddings, self.head(embeddings), self.attention(embeddings)

    def forward(self, batch) -> torch.Tensor:
        """Bag logits of a batch holding the ego-graphs of one unit."""
        _, logits, scores = self.forward_instances(batch)
        return attention_pool(logits, scores)[0]

    # ------------------------------------------------------------- training
    def training_step(self, batch, batch_idx: int) -> None:
        self._bag_buffer.append((self(batch).unsqueeze(0), batch.graph_y[:1].float()))
        if len(self._bag_buffer) >= self.finetune.hyperbatch_size or self.trainer.is_last_batch:
            self._optimizer_step()

    def on_train_epoch_end(self) -> None:
        # Only an epoch cut short (trainer.limit_train_batches) leaves bags here.
        if self._bag_buffer:
            self._optimizer_step()

    def _optimizer_step(self) -> None:
        logits = torch.cat([b[0] for b in self._bag_buffer])
        targets = torch.cat([b[1] for b in self._bag_buffer])
        self._bag_buffer.clear()
        loss = self.objective(logits, targets)
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        scale = scaler.get_scale() if scaler is not None else None
        optimizer.step()
        # Under fp16 a step with non-finite gradients is skipped (and the loss
        # scale lowered); the schedule only advances on steps that happened.
        if scaler is None or scaler.get_scale() >= scale:
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                scheduler.step()
        self.log(
            "train/loss", loss.detach(), on_epoch=True, prog_bar=True, batch_size=len(targets)
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        # Called after mixed-precision unscaling, so the norm is the true one.
        if self.finetune.grad_clip_norm:
            params = [p for p in self.parameters() if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, float(self.finetune.grad_clip_norm))

    def configure_optimizers(self):
        scales = self.finetune.lr_scales
        lr = float(self.hparams.optim["lr"])
        groups = []
        for name, module in (
            ("encoder", self.encoder),
            ("head", self.head),
            ("attention", self.attention),
        ):
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                group = {"params": params, "lr": lr * scales[name], "name": name}
                if name == "head" and self.finetune.head_weight_decay is not None:
                    group["weight_decay"] = float(self.finetune.head_weight_decay)
                groups.append(group)
        if not groups:
            raise ValueError("Nothing to train: every component is frozen.")
        optimizer = build_optimizer(self.hparams.optim, groups)
        epochs = self.finetune.schedule_epochs or int(self.trainer.max_epochs)
        n_units = len(self.trainer.datamodule.dataset_train)
        scheduler = build_scheduler(
            self.hparams.scheduler,
            optimizer,
            total_steps=epochs * math.ceil(n_units / self.finetune.hyperbatch_size),
            warmup_steps=self.finetune.warmup_steps,
        )
        # Stepped by _optimizer_step (manual optimisation).
        return optimizer if scheduler is None else ([optimizer], [scheduler])

    # ----------------------------------------------------------- evaluation
    def _eval_step(self, batch) -> None:
        _, logits, scores = self.forward_instances(batch)
        buf = self._eval_buffer
        buf.setdefault("unit_ids", []).extend(batch.unit_id)
        buf.setdefault("logits", []).append(logits.detach().float().cpu())
        buf.setdefault("scores", []).append(scores.detach().float().cpu())
        buf.setdefault("targets", {}).update(
            {u: y for u, y in zip(batch.unit_id, batch.graph_y.detach().float().cpu())}
        )

    def validation_step(self, batch, batch_idx: int) -> None:
        self._eval_step(batch)

    def test_step(self, batch, batch_idx: int) -> None:
        self._eval_step(batch)

    def on_validation_epoch_end(self) -> None:
        self._finish_eval("val")

    def on_test_epoch_end(self) -> None:
        self._finish_eval("test")

    def _finish_eval(self, stage: str) -> None:
        buf, self._eval_buffer = self._eval_buffer, {}
        if not buf:
            return
        if dist.is_available() and dist.is_initialized():
            gathered: list[dict] = [None] * dist.get_world_size()  # type: ignore[list-item]
            dist.all_gather_object(gathered, buf)
            buf = {
                "unit_ids": [u for g in gathered for u in g["unit_ids"]],
                "logits": [t for g in gathered for t in g["logits"]],
                "scores": [t for g in gathered for t in g["scores"]],
                "targets": {k: v for g in gathered for k, v in g["targets"].items()},
            }
        units, bag_logits = pool_by_bag(
            torch.cat(buf["logits"]), torch.cat(buf["scores"]), buf["unit_ids"]
        )
        targets = torch.stack([buf["targets"][u] for u in units])
        metrics = region_metrics(self.task["target_type"], bag_logits, targets)
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}", value, prog_bar=name in ("loss", "auroc", "c_index", "rmse")
            )
        if stage == "test":
            self.test_predictions = {"unit_ids": units, "logits": bag_logits, "targets": targets}
