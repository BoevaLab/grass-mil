"""``grass-mil-probe``: fit the linear probe head on a frozen encoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.models.training.probe import (  # noqa: E402
    attach_datamodule,
    collect_frozen_embeddings,
    fit_probe,
    probe_sampler_config,
    save_probe_head,
)
from grass_mil.utils import RankedLogger, extras  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


def _resolve_device(name: str) -> torch.device:
    name = str(name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_probe(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    pcfg = cfg.probe

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False, enable_checkpointing=False)
    attach_datamodule(model, datamodule, trainer)
    model.setup("fit")
    device = _resolve_device(pcfg.get("device", "auto"))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sampler_cfg = probe_sampler_config(
        datamodule.sampler_config,
        roots_per_unit=pcfg.get("roots_per_unit"),
        seed=int(pcfg.get("seed", 0)),
        batch_size=int(pcfg.get("batch_size", 256)),
    )
    target_columns = cfg.task.get("target_columns")
    target_columns = list(target_columns) if target_columns else None
    num_classes = int(cfg.model.graph_head.output_dim)

    log.info(
        f"Collecting frozen embeddings: {pcfg.get('roots_per_unit')} roots/region, "
        f"seed {pcfg.get('seed', 0)}, {len(datamodule.dataset_train)} training regions"
    )
    train_emb = collect_frozen_embeddings(
        model,
        datamodule.dataset_train,
        sampler_config=sampler_cfg,
        target_columns=target_columns,
        device=device,
        num_workers=int(cfg.data.get("num_workers", 0)),
    )
    eval_sets = {}
    for split in list(pcfg.get("eval_splits") or []):
        ds = getattr(datamodule, f"dataset_{split}", None)
        if ds is None or len(ds) == 0:
            log.warning(f"probe.eval_splits: split '{split}' is empty; skipped")
            continue
        eval_sets[str(split)] = collect_frozen_embeddings(
            model,
            ds,
            sampler_config=sampler_cfg,
            target_columns=target_columns,
            device=device,
            num_workers=int(cfg.data.get("num_workers", 0)),
        )

    class_weight = pcfg.get("class_weight")
    class_weight = None if class_weight in (None, "null", "none", "None") else str(class_weight)
    fit = fit_probe(
        train_emb,
        num_classes=num_classes,
        C=float(pcfg.get("C", 1.0)),
        class_weight=class_weight,
        max_iter=int(pcfg.get("max_iter", 3000)),
        eval_sets=eval_sets,
    )
    fit.settings.update(
        {
            "roots_per_unit": pcfg.get("roots_per_unit"),
            "seed": int(pcfg.get("seed", 0)),
            "sampler_runtime": dict(sampler_cfg.runtime),
            "init_from_ckpt": cfg.model.get("init_from_ckpt"),
            "n_train_regions": len(datamodule.dataset_train),
        }
    )
    out = save_probe_head(fit, str(pcfg.output_path))
    summary = {"output_path": str(out), "metrics": fit.metrics, "settings": fit.settings}
    summary_path = Path(str(out)).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=1, default=str))
    log.info(
        f"Probe head written to {out}: "
        + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in fit.metrics.items()
        )
    )
    return summary


@hydra.main(version_base="1.3", config_path="configs", config_name="probe.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    summary = run_probe(cfg)
    return summary["metrics"].get("val_region_auroc")


if __name__ == "__main__":
    main()
