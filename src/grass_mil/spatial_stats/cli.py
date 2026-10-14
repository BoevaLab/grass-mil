"""``grass-mil-spatial-stats``: classical spatial statistics on a cell cache."""

from __future__ import annotations

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.inference.cell_cache import CellCache  # noqa: E402
from grass_mil.interpretability.metaclusters_cli import load_labels_and_patients  # noqa: E402
from grass_mil.interpretability.niches_cli import resolve_regions  # noqa: E402
from grass_mil.spatial_stats.run import run_spatial_stats  # noqa: E402
from grass_mil.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["run", "main"]


def run(cfg: DictConfig) -> Dict[str, Any]:
    cache = CellCache.open(str(cfg.cells_dir))
    labels_of, patient_of = load_labels_and_patients(cfg.labels)
    regions = resolve_regions(cfg.get("regions") or "all", cache)
    return run_spatial_stats(
        cache,
        labels_of=labels_of,
        patient_of=patient_of,
        out_dir=str(cfg.output_dir),
        regions=regions,
        group_map=OmegaConf.to_container(cfg.group_map, resolve=True)
        if cfg.get("group_map")
        else None,
        reference=list(cfg.reference) if cfg.get("reference") else None,
        partners=list(cfg.partners) if cfg.get("partners") else None,
        infiltrating=list(cfg.infiltrating) if cfg.get("infiltrating") else (),
        n_perms=int(cfg.nhood.n_perms),
        nhood_backend=str(cfg.nhood.backend),
        window_k=int(cfg.cn.window_k),
        n_cn=int(cfg.cn.n_cn),
        cn_sample_size=int(cfg.cn.sample_size),
        radii=[float(r) for r in cfg.crossk.radii],
        seed=int(cfg.seed),
        min_per_group=int(cfg.min_per_group),
        positive_name=str(cfg.positive_name),
        analyses=list(cfg.analyses),
        log=log,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="spatial_stats.yaml")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
