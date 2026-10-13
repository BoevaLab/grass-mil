"""``grass-mil-metaclusters``: cross-fold niche replication (cluster / match / transfer / basis)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.inference.cell_cache import CellCache  # noqa: E402
from grass_mil.interpretability.metaclusters.basis import run_shared_basis, write_basis_outputs  # noqa: E402
from grass_mil.interpretability.metaclusters.match import match_metaclusters, write_match_outputs  # noqa: E402
from grass_mil.interpretability.metaclusters.records import (
    cluster_fold_models,
    load_fold_clustering,
)  # noqa: E402
from grass_mil.interpretability.metaclusters.transfer import run_transfer, write_transfer_outputs  # noqa: E402
from grass_mil.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["run_metaclusters", "main", "discover_caches", "load_labels_and_patients"]


def discover_caches(
    cells_dir: str | Path, models: Optional[List[str]] = None
) -> Dict[str, CellCache]:
    root = Path(cells_dir)
    if (root / "index.json").exists():
        c = CellCache.open(root)
        return {c.index.get("model_name", root.name): c}
    found: Dict[str, CellCache] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and (p / "index.json").exists()):
        c = CellCache.open(sub)
        name = c.index.get("model_name", sub.name)
        if models and name not in models:
            continue
        found[name] = c
    if not found:
        raise FileNotFoundError(f"No cell caches under {root}")
    return found


def load_labels_and_patients(cfg_labels):
    df = pd.read_csv(str(cfg_labels.label_file))
    df.columns = [str(c).strip() for c in df.columns]
    idc = str(cfg_labels.id_column)
    lc = str(cfg_labels.label_column)
    df = df.dropna(subset=[lc])
    labels_of = {str(r): int(float(v) >= 0.5) for r, v in zip(df[idc], df[lc])}
    pc = cfg_labels.get("patient_column")
    patient_of = {str(r): str(p) for r, p in zip(df[idc], df[pc])} if pc else None
    return labels_of, patient_of


def run_metaclusters(cfg: DictConfig) -> Dict[str, Any]:
    out = Path(str(cfg.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    caches = discover_caches(str(cfg.cells_dir), list(cfg.models) if cfg.get("models") else None)
    held = {k: v for k, v in caches.items() if v.held_out}
    if not held:
        raise ValueError(
            "No cache records held-out regions; extract from a nested-CV summary (or +extract.held_out)."
        )
    caches = held
    labels_of, patient_of = load_labels_and_patients(cfg.labels)
    stage = str(cfg.get("stage", "all")).lower()
    stages = ["cluster", "match", "transfer", "basis"] if stage == "all" else [stage]
    cluster_dir = out / "cluster"
    match_dir = out / "match"
    summary: Dict[str, Any] = {"models": list(caches)}
    names = None
    cohort = None
    fc = None
    if "cluster" in stages:
        fc = cluster_fold_models(
            caches,
            labels_of=labels_of,
            out_dir=cluster_dir,
            reduction=OmegaConf.to_container(cfg.reduction, resolve=True),
            clustering=OmegaConf.to_container(cfg.clustering, resolve=True),
            sample_size=cfg.get("sample_size"),
            l2_normalise=bool(cfg.get("l2_normalise", True)),
            assign_k=int(cfg.get("assign_k", 15)),
            seed=int(cfg.get("seed", 0)),
            n_boot=int(cfg.get("n_boot", 2000)),
            positive_class=int(cfg.get("positive_class", 1)),
            log=log,
        )
        summary["cluster"] = fc.settings["per_fold"]
    if "match" in stages or "transfer" in stages or "basis" in stages:
        if fc is None:
            fc = load_fold_clustering(cluster_dir)
    names = fc.cell_type_names
    cohort = fc.cohort_frequency
    matched = None
    if "match" in stages:
        m = cfg.match
        grid = np.linspace(float(m.scan.min), float(m.scan.max), int(m.scan.n))
        res = match_metaclusters(
            fc.records,
            cell_type_names=names,
            cohort_frequency=cohort,
            distance=str(m.distance),
            threshold=m.get("threshold"),
            scan_grid=grid,
            min_folds=int(m.min_folds),
            region_labels=fc.region_labels,
            region_fold=fc.region_fold,
            patient_of=patient_of,
            labels_of=labels_of,
        )
        paths = write_match_outputs(
            res,
            match_dir,
            cell_type_names=names,
            figures=bool(cfg.get("figures", True)),
            positive_name=str(cfg.positive_name),
        )
        matched = res.records
        summary["match"] = {"threshold": res.threshold, **res.settings, "report": paths["report"]}
        log.info(
            f"match: threshold {res.threshold:.3f}, {res.settings['n_metaclusters']} metaclusters, {res.settings['n_replicated']} replicated"
        )
    if "transfer" in stages:
        if matched is None:
            matched = pd.read_csv(match_dir / "fold_clusters_matched.csv")
        mt = pd.read_csv(match_dir / "metaclusters.csv")
        res_t = run_transfer(
            caches,
            cluster_dir=cluster_dir,
            matched_records=matched,
            labels_of=labels_of,
            patient_of=patient_of,
            positive_class=int(cfg.get("positive_class", 1)),
            min_cells=int(cfg.transfer.min_cells),
            present_threshold=float(cfg.transfer.present_threshold),
            log=log,
        )
        paths = write_transfer_outputs(
            res_t, out / "transfer", metaclusters=mt, positive_name=str(cfg.positive_name)
        )
        summary["transfer"] = {"refit_agreement": res_t.refit_agreement, "report": paths["report"]}
    if "basis" in stages and bool(cfg.basis.get("enabled", True)):
        b = cfg.basis
        basis_cache = None
        sm = b.get("space_model")
        if sm:
            basis_cache = caches.get(str(sm)) or CellCache.open(str(sm))
        res_b = run_shared_basis(
            caches,
            basis_cache=basis_cache,
            labels_of=labels_of,
            k=int(b.k),
            sample_size=int(b.sample_size),
            seed=int(b.seed),
            n_init=int(b.n_init),
            positive_class=int(cfg.get("positive_class", 1)),
            n_boot=int(cfg.get("n_boot", 2000)),
            log=log,
        )
        paths = write_basis_outputs(
            res_b,
            out / "basis",
            positive_name=str(cfg.positive_name),
            figures=bool(cfg.get("figures", True)),
        )
        summary["basis"] = {"k": int(b.k), "report": paths["report"]}
    (out / "metaclusters_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="metaclusters.yaml")
def main(cfg: DictConfig) -> None:
    run_metaclusters(cfg)


if __name__ == "__main__":
    main()
