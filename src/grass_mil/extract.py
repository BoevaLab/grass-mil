"""``grass-mil-extract``: per-cell outputs of trained models (see inference/cell_cache.py)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from grass_mil.data.splits import read_ids
from grass_mil.inference.cell_cache import extract_cells, scoring_sampler
from grass_mil.models import SupervisedModule
from grass_mil.runtime import resolve_device, start_run

__all__ = ["run_extract", "resolve_models", "main"]

log = logging.getLogger(__name__)


def resolve_models(ecfg: DictConfig) -> list[dict[str, Any]]:
    """``[{name, ckpt_path, held_out}]`` from a cross-validation summary and/or explicit models."""
    models = []
    if ecfg.get("nested_cv_summary"):
        summary_path = Path(str(ecfg.nested_cv_summary))
        summary = json.loads(summary_path.read_text())
        oof = pd.read_csv(summary_path.parent / "oof_predictions.csv")
        for fold in summary["folds"]:
            ckpt = Path(fold["run_dir"]) / "checkpoints" / "last.ckpt"
            held_out = oof.loc[oof["outer"] == fold["outer"], "unit_id"].astype(str).tolist()
            models.append(
                {"name": f"fold{fold['outer']}", "ckpt_path": str(ckpt), "held_out": held_out}
            )
    for name, ckpt in dict(ecfg.get("models") or {}).items():
        models.append({"name": str(name), "ckpt_path": str(ckpt), "held_out": []})
    if not models:
        raise ValueError("Set extract.nested_cv_summary or extract.models={name: checkpoint}.")
    return models


def run_extract(cfg: DictConfig) -> list[str]:
    start_run(cfg)
    ecfg = cfg.extract
    device = resolve_device(ecfg.device)
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.prepare_data()
    units = datamodule.all_units().select(read_ids(ecfg.get("units")))
    meta = datamodule.unit_table(units)
    unit_meta = {}
    for entry in units.entries:
        info = {k: str(v) for k, v in meta.loc[entry.unit_id].items() if k == "patient_id"}
        label_id = units.label_id(entry)
        if label_id is not None:
            info["targets"] = units.labels.target(label_id).tolist()
        unit_meta[entry.unit_id] = info
    written = []
    for spec in resolve_models(ecfg):
        log.info(f"[{spec['name']}] {spec['ckpt_path']}")
        model = SupervisedModule.from_checkpoint(spec["ckpt_path"])
        model.input_spec.check_matches(datamodule.input_spec, spec["ckpt_path"])
        model_dir = extract_cells(
            model,
            units,
            sampler=scoring_sampler(model.trained_sampler, int(ecfg.batch_size), every_cell=True),
            out_dir=str(ecfg.output_dir),
            model_name=spec["name"],
            ckpt_path=spec["ckpt_path"],
            held_out=spec["held_out"],
            unit_meta=unit_meta,
            device=device,
            resume=bool(ecfg.resume),
        )
        written.append(str(model_dir))
        log.info(f"[{spec['name']}] cells written to {model_dir}")
    return written


@hydra.main(version_base="1.3", config_path="configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> None:
    run_extract(cfg)


if __name__ == "__main__":
    main()
