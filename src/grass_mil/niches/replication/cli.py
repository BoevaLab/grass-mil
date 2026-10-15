"""``grass-mil-metaclusters``: replication of niches across cross-validation models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from grass_mil.inference.cell_cache import CellCache

from ..discovery import ClusteringConfig
from .stages import (
    cluster_folds,
    held_out_shares,
    match_niches,
    share_tests,
    shared_basis,
    transfer_niches,
)

__all__ = ["run_metaclusters", "main", "open_caches"]

log = logging.getLogger(__name__)

STAGES = ("cluster", "match", "transfer", "basis")


def open_caches(cells_dir: str | Path) -> dict[str, CellCache]:
    """Every model cache under ``cells_dir`` that records held-out units."""
    caches = {}
    for sub in sorted(Path(cells_dir).iterdir()):
        if (sub / "index.json").exists():
            cache = CellCache.open(sub)
            if cache.held_out:
                caches[cache.name] = cache
    if len(caches) < 2:
        raise ValueError(f"{cells_dir} needs at least two model caches with held-out units.")
    return caches


def run_metaclusters(cfg: DictConfig) -> dict:
    stage = str(cfg.stage)
    if stage not in ("all", *STAGES):
        raise ValueError(f"stage must be 'all' or one of {STAGES}.")
    stages = STAGES if stage == "all" else (stage,)
    out = Path(str(cfg.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    caches = open_caches(str(cfg.cells_dir))
    clustering = ClusteringConfig.from_dict(OmegaConf.to_container(cfg.clustering, resolve=True))
    positive_class = int(cfg.positive_class)
    summary: dict = {"models": list(caches)}
    folds_path, records_path = out / "fold_niches.joblib", out / "matched_niches.csv"

    if "cluster" in stages:
        folds = cluster_folds(
            caches, clustering, positive_class=positive_class, n_boot=int(cfg.n_boot)
        )
        joblib.dump(folds, folds_path)
        folds.records.to_csv(out / "fold_niches.csv", index=False)
    if "match" in stages or "transfer" in stages:
        if not folds_path.exists():
            raise FileNotFoundError(f"{folds_path} is missing; run stage=cluster first.")
        folds = joblib.load(folds_path)
    if "match" in stages:
        start, stop, n = cfg.match.scan
        records, table, threshold, scan = match_niches(
            folds,
            threshold=cfg.match.threshold,
            scan=np.linspace(start, stop, int(n)),
            min_models=int(cfg.match.min_models),
        )
        records.to_csv(records_path, index=False)
        table.to_csv(out / "metaclusters.csv", index=False)
        if scan is not None:
            scan.to_csv(out / "threshold_scan.csv", index=False)
        units, patients = held_out_shares(folds, records)
        units.to_csv(out / "unit_metacluster_shares.csv")
        positive = {}
        for unit, pos in folds.positive.items():
            positive[folds.patients.get(unit, unit)] = (
                positive.get(folds.patients.get(unit, unit), False) or pos
            )
        share_tests(patients, positive).to_csv(out / "patient_metacluster_tests.csv", index=False)
        summary["match"] = {
            "threshold": threshold,
            "n_metaclusters": len(table),
            "n_replicated": int(table["replicated"].sum()),
        }
        _overview(
            table,
            records,
            folds.cell_types,
            out / "metaclusters.png",
            positive_name=str(cfg.positive_name),
        )
    if "transfer" in stages:
        if not records_path.exists():
            raise FileNotFoundError(f"{records_path} is missing; run stage=match first.")
        result = transfer_niches(
            caches, folds, pd.read_csv(records_path), positive_class=positive_class
        )
        tdir = out / "transfer"
        tdir.mkdir(exist_ok=True)
        for name, frame in result.items():
            frame.to_csv(tdir / f"{name}.csv")
        summary["transfer"] = {"mean_agreement": float(np.nanmean(result["agreement"].to_numpy()))}
    if "basis" in stages and cfg.basis.enabled:
        result = shared_basis(
            caches,
            space_model=cfg.basis.model,
            space=str(cfg.basis.space),
            k=int(cfg.basis.k),
            sample_size=int(cfg.basis.sample_size),
            seed=int(cfg.basis.seed),
            n_boot=int(cfg.n_boot),
            positive_class=positive_class,
        )
        bdir = out / "basis"
        bdir.mkdir(exist_ok=True)
        result["groups"].to_csv(bdir / "groups.csv", index=False)
        result["pairs"].to_csv(bdir / "attribution_pairs.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def _overview(
    table: pd.DataFrame, records: pd.DataFrame, cell_types, path: Path, *, positive_name: str
) -> None:
    """Composition per metacluster and every member's margin."""
    import matplotlib.pyplot as plt

    if table.empty:
        return
    frac = [f"frac_{c}" for c in cell_types]
    y = np.arange(len(table))[::-1]
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(14, 0.45 * len(table) + 2), gridspec_kw={"width_ratios": [2.5, 1.2]}
    )
    im = ax0.imshow(table[frac].to_numpy() * 100, aspect="auto", cmap="viridis", vmin=0)
    ax0.set_xticks(range(len(frac)), cell_types, rotation=90, fontsize=7)
    ax0.set_yticks(
        range(len(table)),
        [f"{m} ({n} models)" for m, n in zip(table["metacluster"], table["n_models"])],
        fontsize=7,
    )
    plt.colorbar(im, ax=ax0, fraction=0.03, label="% of cells")
    for row, meta in zip(y, table["metacluster"]):
        members = records[records["metacluster"] == int(meta[1:])]
        ax1.scatter(members["margin"], np.full(len(members), row), s=18, color="#1b9e77")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_yticks(y, [])
    ax1.set_xlabel(f"margin of each member niche (+ = {positive_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="metaclusters.yaml")
def main(cfg: DictConfig) -> None:
    run_metaclusters(cfg)


if __name__ == "__main__":
    main()
