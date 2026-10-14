"""``grass-mil-extract``: per-cell cache of trained models (see inference/cell_cache.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, open_dict

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.data.components.datasets import SpatialOmicsGraphDataset  # noqa: E402
from grass_mil.inference.cell_cache import extract_cell_cache  # noqa: E402
from grass_mil.inference.model_loading import load_trained_module  # noqa: E402
from grass_mil.models.training.probe import probe_sampler_config  # noqa: E402
from grass_mil.utils import RankedLogger, extras  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["run_extract", "resolve_models", "main"]


def resolve_models(cfg: DictConfig) -> List[Dict[str, Any]]:
    """[{name, ckpt_path, held_out}] from extract.models or a nested-CV summary."""
    models: List[Dict[str, Any]] = []
    summary_path = cfg.extract.get("nested_cv_summary")
    if summary_path:
        summary = json.loads(Path(str(summary_path)).read_text())
        oof = Path(str(summary_path)).parent / "oof_predictions.csv"
        held: Dict[int, List[str]] = {}
        if oof.exists():
            import pandas as pd

            df = pd.read_csv(oof)
            for outer, grp in df.groupby("outer"):
                held[int(outer)] = sorted(grp["region_id"].astype(str).unique())
        for fold in summary["folds"]:
            run_dir = Path(str(fold["run_dir"]))
            ckpt = run_dir / "checkpoints" / "last.ckpt"
            if not ckpt.exists():
                candidates = sorted((run_dir / "checkpoints").glob("*.ckpt"))
                if not candidates:
                    raise FileNotFoundError(f"No checkpoint under {run_dir / 'checkpoints'}")
                ckpt = candidates[-1]
            models.append(
                {
                    "name": f"fold{int(fold['outer'])}",
                    "ckpt_path": str(ckpt),
                    "held_out": held.get(int(fold["outer"]), []),
                }
            )
    held_cfg = dict(cfg.extract.get("held_out") or {})
    for name, path in dict(cfg.extract.get("models") or {}).items():
        held = held_cfg.get(str(name)) or []
        models.append(
            {"name": str(name), "ckpt_path": str(path), "held_out": [str(r) for r in held]}
        )
    if not models:
        raise ValueError("Set extract.nested_cv_summary or extract.models={name: ckpt}.")
    return models


def _regions_filter(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        p = Path(value)
        if p.suffix.lower() == ".json":
            return [str(v) for v in json.loads(p.read_text())]
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [str(v) for v in value]


def run_extract(cfg: DictConfig) -> Dict[str, Any]:
    ecfg = cfg.extract
    device = str(ecfg.get("device", "auto")).lower()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )

    # All regions of the processed cohort, regardless of persisted splits.
    with open_dict(cfg):
        cfg.data.split.explicit = {"train": None, "val": None, "test": None, "id_kind": "region"}
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    full = SpatialOmicsGraphDataset(Path(str(cfg.data.processed_dir)) / "processed_index.json")
    datamodule.dataset_train = full  # width inference for the model build
    sampler_cfg = probe_sampler_config(
        datamodule.sampler_config,
        roots_per_unit=None,
        seed=0,
        batch_size=int(ecfg.get("batch_size", 512)),
    )
    ct_names = None
    label_map = full.label_maps.get("cell_type") if full.label_maps else None
    if label_map:
        ct_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    out_dir = Path(str(ecfg.output_dir))
    written = []
    for spec in resolve_models(cfg):
        log.info(f"[{spec['name']}] loading {spec['ckpt_path']}")
        model = load_trained_module(
            cfg, ckpt_path=spec["ckpt_path"], datamodule=datamodule, device=device
        )
        model_dir = extract_cell_cache(
            model,
            full,
            out_dir=out_dir,
            model_name=spec["name"],
            sampler_config=sampler_cfg,
            ckpt_path=spec["ckpt_path"],
            held_out_regions=spec.get("held_out"),
            regions=_regions_filter(ecfg.get("regions")),
            cell_type_names=ct_names,
            autocast=bool(ecfg.get("autocast", True)),
            device=device,
            resume=bool(ecfg.get("resume", True)),
            unit_id=str(ecfg.get("unit_id", "auto")),
            log=log,
        )
        written.append(str(model_dir))
        log.info(f"[{spec['name']}] cache written to {model_dir}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"output_dir": str(out_dir), "models": written}


@hydra.main(version_base="1.3", config_path="configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    run_extract(cfg)


if __name__ == "__main__":
    main()
