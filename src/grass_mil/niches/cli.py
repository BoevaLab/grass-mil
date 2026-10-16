"""``grass-mil-niches``: niche analysis of one model's cell cache."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from grass_mil.inference.cell_cache import CellCache

from .analysis import analyse_niches, write_niche_outputs
from .cells import resolve_units
from .descriptors import DescriptorConfig
from .discovery import ClusteringConfig
from .survival import SurvivalConfig

__all__ = ["run_niches", "main"]

log = logging.getLogger(__name__)


def run_niches(cfg: DictConfig) -> Path:
    cache = CellCache.open(str(cfg.cells_dir))

    def plain(key: str):
        value = cfg.get(key)
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    result = analyse_niches(
        cache,
        resolve_units(plain("units"), cache),
        clustering=ClusteringConfig.from_dict(plain("clustering")),
        descriptors=DescriptorConfig.from_dict(plain("descriptors")),
        survival=SurvivalConfig.from_dict(plain("survival")),
        positive_class=int(cfg.positive_class),
        present_threshold=float(cfg.present_threshold),
        enrichment_threshold=float(cfg.enrichment_threshold),
        n_boot=int(cfg.n_boot),
    )
    out = write_niche_outputs(
        result,
        str(cfg.output_dir),
        title=str(cfg.title or f"Niches of {cache.name}"),
        positive_name=str(cfg.positive_name),
        render_units=int(cfg.render_units),
        figures_enabled=bool(cfg.figures),
    )
    log.info(
        f"{result.model.n_niches} niches in {result.cells.n} cells; report: {out / 'report.md'}"
    )
    return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="niches.yaml")
def main(cfg: DictConfig) -> None:
    run_niches(cfg)


if __name__ == "__main__":
    main()
