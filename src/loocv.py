from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import pandas as pd
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train import train
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)


def _canonical_fold_id(fold_unit: str, sample_id: str, region_id: Optional[str]) -> str:
    if fold_unit == "sample":
        return str(sample_id)
    region = "__NONE__" if region_id is None else str(region_id)
    return f"{sample_id}::{region}"


def _discover_folds(cfg: DictConfig) -> List[Dict[str, Optional[str]]]:
    manifest_path = Path(str(cfg.data.raw_manifest_path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    sample_col = str(cfg.data.manifest.sample_id)
    if sample_col not in df.columns:
        raise ValueError(f"Manifest missing sample id column '{sample_col}'.")

    fold_unit = str(cfg.data.split.loocv.fold_unit)
    folds: List[Dict[str, Optional[str]]] = []
    if fold_unit == "sample":
        for sample_id in sorted(df[sample_col].astype(str).unique().tolist()):
            folds.append(
                {
                    "fold_id": str(sample_id),
                    "sample_id": str(sample_id),
                    "region_id": None,
                }
            )
        return folds

    if fold_unit != "region":
        raise ValueError(
            "data.split.loocv.fold_unit must be one of ['region', 'sample']."
        )

    region_col = str(cfg.data.manifest.region_id)
    if region_col not in df.columns:
        raise ValueError(
            "Region-based LOOCV requires a region column in manifest. "
            f"Missing '{region_col}'."
        )

    pairs = (
        df[[sample_col, region_col]]
        .drop_duplicates()
        .sort_values(by=[sample_col, region_col], na_position="first")
    )
    for _, row in pairs.iterrows():
        sample_id = str(row[sample_col])
        region_id = None if pd.isna(row[region_col]) else str(row[region_col])
        folds.append(
            {
                "fold_id": _canonical_fold_id("region", sample_id, region_id),
                "sample_id": sample_id,
                "region_id": region_id,
            }
        )
    return folds


def _resolve_selected_folds(
    cfg: DictConfig, discovered: List[Dict[str, Optional[str]]]
) -> List[str]:
    mode = str(cfg.loocv.mode)
    if mode not in {"single", "all"}:
        raise ValueError("loocv.mode must be one of ['single', 'all'].")

    canonical_ids = [str(item["fold_id"]) for item in discovered]
    if mode == "all":
        return canonical_ids

    requested = cfg.data.split.loocv.holdout_id
    fold_index = cfg.loocv.fold_index
    if requested is None and fold_index is None:
        raise ValueError(
            "Single LOOCV mode requires either data.split.loocv.holdout_id or loocv.fold_index."
        )
    if fold_index is not None:
        idx = int(fold_index)
        if idx < 0 or idx >= len(canonical_ids):
            raise ValueError(
                f"loocv.fold_index out of range: {idx}. Valid range: [0, {len(canonical_ids) - 1}]"
            )
        return [canonical_ids[idx]]

    requested = str(requested)
    if requested in canonical_ids:
        return [requested]

    if str(cfg.data.split.loocv.fold_unit) == "region":
        # Convenience support for plain region_id when unique.
        matches = [item for item in discovered if item["region_id"] == requested]
        if len(matches) == 1:
            return [str(matches[0]["fold_id"])]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous region alias '{requested}'. Use canonical fold id "
                f"'sample_id::region_id'. Matches: {[m['fold_id'] for m in matches]}"
            )

    raise ValueError(
        f"Requested holdout id '{requested}' not found. Available fold ids: {canonical_ids}"
    )


def _fold_slug(fold_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", fold_id)


def _to_float_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            try:
                converted[key] = float(value.item())
                continue
            except Exception:
                pass
        try:
            converted[key] = float(value)
        except Exception:
            continue
    return converted


def _aggregate_fold_metrics(
    rows: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    metric_names = sorted({name for row in rows for name in row})
    aggregated: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        vals = [row[name] for row in rows if name in row]
        if not vals:
            continue
        aggregated[name] = {
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "n": float(len(vals)),
        }
    return aggregated


def run_loocv(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    discovered = _discover_folds(cfg)
    fold_ids = _resolve_selected_folds(cfg, discovered)
    log.info(f"LOOCV fold count: {len(fold_ids)}")

    base_output_dir = Path(str(cfg.paths.output_dir))
    base_processed_dir = Path(str(cfg.data.processed_dir))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: List[Dict[str, Any]] = []
    fold_numeric_metrics: List[Dict[str, float]] = []

    for fold_id in fold_ids:
        fold_cfg = copy.deepcopy(cfg)
        fold_slug = _fold_slug(fold_id)
        with open_dict(fold_cfg):
            fold_cfg.data.split.loocv.enabled = True
            fold_cfg.data.split.loocv.holdout_id = fold_id
            if bool(cfg.loocv.force_precompute_per_fold):
                fold_cfg.data.force_precompute = True
            if bool(cfg.loocv.per_fold_processed_dir):
                fold_cfg.data.processed_dir = str(
                    base_processed_dir / f"fold_{fold_slug}"
                )
            fold_cfg.paths.output_dir = str(base_output_dir / f"fold_{fold_slug}")
            fold_cfg.task_name = f"{cfg.task_name}_fold_{fold_slug}"
            fold_cfg.tags = (
                list(cfg.tags) + [f"loocv:{fold_id}"]
                if cfg.get("tags")
                else [f"loocv:{fold_id}"]
            )

        log.info(f"Starting fold: {fold_id}")
        metric_dict, _ = train(fold_cfg)
        numeric = _to_float_metrics(metric_dict)
        fold_numeric_metrics.append(numeric)
        fold_results.append(
            {
                "fold_id": fold_id,
                "output_dir": str(fold_cfg.paths.output_dir),
                "processed_dir": str(fold_cfg.data.processed_dir),
                "metrics": numeric,
            }
        )

    summary = {
        "mode": str(cfg.loocv.mode),
        "fold_unit": str(cfg.data.split.loocv.fold_unit),
        "validation_strategy": str(cfg.data.split.loocv.validation_strategy),
        "folds": fold_results,
        "aggregate_metrics": _aggregate_fold_metrics(fold_numeric_metrics),
    }
    summary_path = base_output_dir / str(cfg.loocv.summary_filename)
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote LOOCV summary to {summary_path}")
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="loocv.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    run_loocv(cfg)


if __name__ == "__main__":
    main()
