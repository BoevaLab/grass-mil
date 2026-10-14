"""``grass-mil-niches``: niche discovery on one model's cell cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.inference.cell_cache import CellCache  # noqa: E402
from grass_mil.interpretability.niches.pipeline import run_niche_discovery, write_niche_outputs  # noqa: E402
from grass_mil.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["run_niches", "main", "resolve_regions", "load_region_labels"]


def resolve_regions(value, cache: CellCache):
    if value is None or (isinstance(value, str) and value.lower() == "held_out"):
        return cache.held_out or cache.regions
    if isinstance(value, str) and value.lower() == "all":
        return cache.regions
    if isinstance(value, str):
        p = Path(value)
        if p.suffix.lower() == ".json":
            return [str(v) for v in json.loads(p.read_text())]
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [str(v) for v in value]


def load_region_labels(cfg_labels) -> Optional[Dict[str, int]]:
    if not cfg_labels or not cfg_labels.get("label_file"):
        return None
    df = pd.read_csv(str(cfg_labels.label_file))
    df.columns = [str(c).strip() for c in df.columns]
    idc = str(cfg_labels.get("id_column", "REGION_ID"))
    lc = cfg_labels.get("label_column")
    if not lc:
        raise ValueError("labels.label_column is required with labels.label_file")
    df = df.dropna(subset=[str(lc)])
    return {str(r): int(float(v) >= 0.5) for r, v in zip(df[idc], df[str(lc)])}


def run_niches(cfg: DictConfig) -> Dict[str, Any]:
    cache = CellCache.open(str(cfg.cells_dir))
    regions = resolve_regions(cfg.get("regions"), cache)
    clustering = OmegaConf.to_container(cfg.clustering, resolve=True)
    reduction = OmegaConf.to_container(cfg.reduction, resolve=True)
    result = run_niche_discovery(
        cache,
        regions=regions,
        labels_of=load_region_labels(cfg.get("labels")),
        space=str(cfg.get("space", "gnn")),
        attention_ckpt=cfg.get("attention_ckpt") or cache.index.get("ckpt_path"),
        reduction=reduction,
        clustering=clustering,
        sample_size=cfg.get("sample_size"),
        l2_normalise=bool(cfg.get("l2_normalise", True)),
        assign_k=int(cfg.get("assign_k", 15)),
        seed=int(cfg.get("seed", 0)),
        positive_class=int(cfg.get("positive_class", 1)),
        present_threshold=float(cfg.get("present_threshold", 0.02)),
        n_boot=int(cfg.get("n_boot", 2000)),
    )
    title = (
        cfg.get("title")
        or f"Niche discovery: {cache.index.get('model_name', Path(str(cfg.cells_dir)).name)}"
    )
    paths = write_niche_outputs(
        result,
        str(cfg.output_dir),
        title=str(title),
        positive_name=str(cfg.get("positive_name", "positive")),
        figures=bool(cfg.get("figures", True)),
        umap=OmegaConf.to_container(cfg.umap, resolve=True)
        if cfg.get("umap") is not None
        else None,
        held_out_auroc=cfg.get("held_out_auroc"),
    )
    log.info(
        f"{result.n_clusters} clusters, noise {result.noise_fraction:.1%}; report: {paths['report']}"
    )
    return {
        "n_clusters": result.n_clusters,
        "noise_fraction": result.noise_fraction,
        "concordance": result.concordance,
        "paths": paths,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="niches.yaml")
def main(cfg: DictConfig) -> None:
    run_niches(cfg)


if __name__ == "__main__":
    main()
