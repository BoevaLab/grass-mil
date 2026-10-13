"""Stage 1: cluster every fold model's held-out cells in its own space."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache
from grass_mil.interpretability.niches.pipeline import NicheResult, run_niche_discovery

__all__ = ["FoldClustering", "cluster_fold_models", "load_fold_clustering", "fold_cluster_records"]


@dataclass
class FoldClustering:
    """Per-fold outputs needed by the later stages."""

    records: pd.DataFrame  # one row per fold-cluster (+ comp_* columns)
    region_labels: Dict[str, np.ndarray] = field(default_factory=dict)  # region -> cluster ids
    region_fold: Dict[str, str] = field(default_factory=dict)  # region -> fold model name
    cell_type_names: List[str] = field(default_factory=list)
    cohort_frequency: Optional[np.ndarray] = None
    settings: Dict[str, Any] = field(default_factory=dict)


def fold_cluster_records(result: NicheResult, fold: str, base_rate: float) -> pd.DataFrame:
    """Records of one fold's clusters from its niche table (noise excluded)."""
    T = result.table[result.table["cluster"] != "Noise"].copy()
    comp_cols = [c for c in T.columns if c.startswith("comp_")]
    rows = []
    for _, t in T.iterrows():
        rows.append(
            {
                "fold": fold,
                "cluster": int(t["cluster_id"]),
                "id": f"{fold}_C{int(t['cluster_id'])}",
                "share": float(t["share"]),
                "n_cells": int(t["n_cells"]),
                "cores": int(t["cores_present"]),
                "pos_cores": int(t["pos_cores_present"]),
                "neg_cores": int(t["neg_cores_present"]),
                "emp": float(t["emp_frac_pos"]),
                "emp_rel": float(t["emp_frac_pos"]) - float(base_rate),
                "margin": float(t["relattn"]),
                "margin_lo": float(t["relattn_lo"]),
                "margin_hi": float(t["relattn_hi"]),
                "lift": float(t["lift"]),
                "share_auroc": float(t["share_auroc"]),
                "top_types": t["top_types"],
                "top_enriched": t["top_enriched"],
                **{c: float(t[c]) for c in comp_cols},
            }
        )
    return pd.DataFrame(rows)


def cluster_fold_models(
    caches: Dict[str, CellCache],
    *,
    labels_of: Optional[Dict[str, int]] = None,
    out_dir: Optional[str | Path] = None,
    reduction: Optional[Dict[str, Any]] = None,
    clustering: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = 100_000,
    l2_normalise: bool = True,
    assign_k: int = 15,
    seed: int = 0,
    n_boot: int = 2000,
    positive_class: int = 1,
    regions_of: Optional[Dict[str, Sequence[str]]] = None,
    log=None,
) -> FoldClustering:
    """Cluster each fold model's held-out regions (or ``regions_of[fold]``) with one
    fixed setting; save ``labels_<fold>.npy``, ``cluster_model_<fold>.pkl`` and
    the fold-cluster records."""
    import joblib

    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    region_labels: Dict[str, np.ndarray] = {}
    region_fold: Dict[str, str] = {}
    names: List[str] = []
    all_ct: List[np.ndarray] = []
    per_fold_summary = {}
    for fold, cache in caches.items():
        regions = (
            list(regions_of[fold])
            if regions_of and fold in regions_of
            else (cache.held_out or cache.regions)
        )
        res = run_niche_discovery(
            cache,
            regions=regions,
            labels_of=labels_of,
            space="gnn",
            reduction=reduction,
            clustering=clustering,
            sample_size=sample_size,
            l2_normalise=l2_normalise,
            assign_k=assign_k,
            seed=seed,
            positive_class=positive_class,
            n_boot=n_boot,
        )
        base = float(res.y_cell.mean())
        frames.append(fold_cluster_records(res, fold, base))
        for r in res.cells.regions:
            m = res.cells.region_ids == r
            region_labels[r] = res.labels[m]
            region_fold[r] = fold
        names = res.cells.cell_type_names
        all_ct.append(res.cells.ct)
        per_fold_summary[fold] = {
            "regions": len(res.cells.regions),
            "cells": int(res.labels.size),
            "clusters": res.n_clusters,
            "noise": res.noise_fraction,
            "base_rate": base,
            **res.concordance,
        }
        if out is not None:
            np.save(out / f"labels_{fold}.npy", res.labels)
            joblib.dump(res.model, out / f"cluster_model_{fold}.pkl")
            (out / f"regions_{fold}.json").write_text(json.dumps(list(res.cells.regions)))
        if log is not None:
            log.info(
                f"[{fold}] {len(res.cells.regions)} regions, {res.labels.size} cells, {res.n_clusters} clusters, noise {res.noise_fraction:.2f}"
            )
    records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ct = np.concatenate(all_ct) if all_ct else np.zeros(0, dtype=int)
    cohort = np.bincount(ct, minlength=len(names)) / max(1, ct.size) if names else None
    fc = FoldClustering(
        records=records,
        region_labels=region_labels,
        region_fold=region_fold,
        cell_type_names=list(names),
        cohort_frequency=cohort,
        settings={
            "per_fold": per_fold_summary,
            "reduction": reduction,
            "clustering": clustering,
            "sample_size": sample_size,
            "l2_normalise": l2_normalise,
            "assign_k": assign_k,
            "seed": seed,
        },
    )
    if out is not None:
        records.to_csv(out / "fold_clusters.csv", index=False)
        joblib.dump(
            {
                "region_labels": region_labels,
                "region_fold": region_fold,
                "cell_type_names": list(names),
                "cohort_frequency": cohort,
                "settings": fc.settings,
            },
            out / "fold_clustering.pkl",
        )
    return fc


def load_fold_clustering(out_dir: str | Path) -> FoldClustering:
    import joblib

    out = Path(out_dir)
    payload = joblib.load(out / "fold_clustering.pkl")
    records = pd.read_csv(out / "fold_clusters.csv")
    return FoldClustering(
        records=records,
        cell_type_names=payload["cell_type_names"],
        cohort_frequency=payload["cohort_frequency"],
        settings=payload["settings"],
        region_labels=payload["region_labels"],
        region_fold=payload["region_fold"],
    )
