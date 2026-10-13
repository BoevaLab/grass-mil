"""Niche discovery on one model's cell cache.

Steps (preprint): take the cells of the chosen regions (held-out by default),
compute the relattn margin of every cell, cluster the chosen feature space
(encoder embedding or the attention head's 8-d space) with the user's
reduction + clustering settings (default cosine HDBSCAN: PCA-10 on a 100k
sample, L2, HDBSCAN mcs 300 / min_samples 1 / eom, 15-NN assignment), then
describe every cluster (composition, enrichment, empirical association,
margin with bootstrap CI over regions, share AUROC, co-localisation) and
draw the figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache
from grass_mil.interpretability.core.clustering import NicheClusterModel, fit_niche_clustering

from .colocalisation import cluster_colocalisation
from .margins import attention_space_from_checkpoint, relattn_margins
from .tables import niche_table, spearman_concordance

__all__ = ["NicheResult", "load_cells", "run_niche_discovery", "write_niche_outputs"]

DEFAULT_REDUCTION = {"method": "pca", "n_components": 10}
DEFAULT_CLUSTERING = {
    "method": "hdbscan",
    "params": {"min_cluster_size": 300, "min_samples": 1, "cluster_selection_method": "eom"},
}


@dataclass
class CellSet:
    regions: List[str]
    region_ids: np.ndarray  # (N,)
    E: np.ndarray  # (N, d) float32
    LG: np.ndarray
    AR: Optional[np.ndarray]
    ct: np.ndarray
    pos: np.ndarray
    edge_index: np.ndarray  # (2, M) global indices
    bias: np.ndarray
    cell_type_names: List[str]


def load_cells(cache: CellCache, regions: Sequence[str]) -> CellSet:
    """Concatenate the cached arrays of ``regions`` (in the given order)."""
    E, LG, AR, CT, POS, EDGES, RID = [], [], [], [], [], [], []
    off = 0
    for r in regions:
        d = cache.load(r)
        n = int(d["E"].shape[0])
        E.append(d["E"].astype(np.float32))
        LG.append(d["LG"])
        AR.append(d["AR"] if "AR" in d else None)
        CT.append(d["ct"].astype(int) if "ct" in d else np.zeros(n, dtype=int))
        POS.append(d["pos"])
        EDGES.append(d["edge_index"].astype(np.int64) + off)
        RID.append(np.full(n, str(r), dtype=object))
        off += n
    if not E:
        raise ValueError("No regions to load.")
    names = cache.cell_type_names
    ct = np.concatenate(CT)
    if names is None:
        names = [str(i) for i in range(int(ct.max()) + 1 if ct.size else 0)]
    return CellSet(
        regions=list(regions),
        region_ids=np.concatenate(RID),
        E=np.concatenate(E),
        LG=np.concatenate(LG),
        AR=np.concatenate(AR) if all(a is not None for a in AR) else None,
        ct=ct,
        pos=np.concatenate(POS),
        edge_index=np.concatenate(EDGES, axis=1),
        bias=cache.bias,
        cell_type_names=list(names),
    )


@dataclass
class NicheResult:
    labels: np.ndarray
    model: NicheClusterModel
    table: pd.DataFrame
    margin: np.ndarray
    attention: np.ndarray
    colocalisation: np.ndarray
    clusters: List[int]
    concordance: Dict[str, float]
    cells: CellSet
    features: np.ndarray
    settings: Dict[str, Any] = field(default_factory=dict)
    y_cell: Optional[np.ndarray] = None

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def noise_fraction(self) -> float:
        return float((self.labels == -1).mean()) if self.labels.size else float("nan")


def _region_labels(
    cache: CellCache, regions: Sequence[str], labels_of: Optional[Dict[str, int]]
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in regions:
        if labels_of is not None and r in labels_of:
            out[r] = int(labels_of[r])
        else:
            v = cache.label_of(r)
            if v is None:
                raise ValueError(f"No label for region {r}: pass labels_of or a labelled cache.")
            out[r] = int(round(float(v)))
    return out


def run_niche_discovery(
    cache: CellCache,
    *,
    regions: Optional[Sequence[str]] = None,
    labels_of: Optional[Dict[str, int]] = None,
    space: str = "gnn",
    attention_ckpt: Optional[str] = None,
    reduction: Optional[Dict[str, Any]] = None,
    clustering: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = 100_000,
    l2_normalise: bool = True,
    assign_k: int = 15,
    seed: int = 0,
    positive_class: int = 1,
    present_threshold: float = 0.02,
    n_boot: int = 2000,
) -> NicheResult:
    if regions is None:
        regions = cache.held_out or cache.regions
    regions = sorted(str(r) for r in regions)
    cells = load_cells(cache, regions)
    region_labels = _region_labels(cache, regions, labels_of)
    if cells.AR is None:
        raise ValueError(
            "The cache has no attention logits (AR); niche margins need an attention model."
        )
    rel = relattn_margins(
        cells.LG, cells.AR, cells.bias, cells.region_ids, positive_class=positive_class
    )
    space = str(space).lower()
    if space in {"gnn", "encoder", "embedding"}:
        X = cells.E
    elif space in {"attention", "attn"}:
        if not attention_ckpt:
            raise ValueError("space='attention' needs attention_ckpt (the model checkpoint).")
        X = attention_space_from_checkpoint(cells.E, attention_ckpt)
    else:
        raise ValueError("space must be 'gnn' or 'attention'")
    model, labels = fit_niche_clustering(
        X,
        reduction=reduction or DEFAULT_REDUCTION,
        clustering=clustering or DEFAULT_CLUSTERING,
        sample_size=sample_size,
        l2_normalise=l2_normalise,
        assign_k=assign_k,
        seed=seed,
    )
    table = niche_table(
        labels,
        cell_types=cells.ct,
        region_ids=cells.region_ids,
        region_labels=region_labels,
        margin=rel.margin,
        attention_pos=rel.attention[:, min(positive_class, rel.attention.shape[1] - 1)],
        cell_type_names=cells.cell_type_names,
        present_threshold=present_threshold,
        n_boot=n_boot,
        seed=seed,
    )
    CO, clusters = cluster_colocalisation(labels, cells.edge_index)
    y_cell = np.array([region_labels[r] for r in cells.region_ids])
    return NicheResult(
        labels=labels,
        model=model,
        table=table,
        margin=rel.margin,
        attention=rel.attention,
        colocalisation=CO,
        clusters=clusters,
        concordance=spearman_concordance(table),
        cells=cells,
        features=X,
        settings={
            "space": space,
            "regions": regions,
            "n_cells": int(labels.size),
            "positive_class": int(positive_class),
            **model.params,
        },
        y_cell=y_cell,
    )


def write_niche_outputs(
    result: NicheResult,
    out_dir: str | Path,
    *,
    title: str = "niche discovery",
    positive_name: str = "positive",
    figures: bool = True,
    umap: Optional[Dict[str, Any]] = None,
    held_out_auroc: Optional[float] = None,
) -> Dict[str, str]:
    """Write labels, cluster model, tables, report and figures; returns the paths."""
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    np.save(out / "labels.npy", result.labels)
    paths["labels"] = str(out / "labels.npy")
    joblib.dump(result.model, out / "cluster_model.pkl")
    paths["cluster_model"] = str(out / "cluster_model.pkl")
    cells = pd.DataFrame(
        {
            "region_id": result.cells.region_ids,
            "cell_index": np.concatenate(
                [np.arange((result.cells.region_ids == r).sum()) for r in result.cells.regions]
            ),
            "cluster": result.labels,
            "cell_type": [result.cells.cell_type_names[c] for c in result.cells.ct],
            "relattn": result.margin,
            "x": result.cells.pos[:, 0],
            "y": result.cells.pos[:, 1],
        }
    )
    cells.to_csv(out / "cells.csv", index=False)
    paths["cells"] = str(out / "cells.csv")
    result.table.to_csv(out / "clusters.csv", index=False)
    paths["clusters"] = str(out / "clusters.csv")
    np.save(out / "coloc_log2.npy", result.colocalisation)
    (out / "settings.json").write_text(json.dumps(result.settings, indent=1, default=str))

    T = result.table
    base = float(result.y_cell.mean()) if result.y_cell is not None else float("nan")
    n_pos = sum(
        1
        for r in result.cells.regions
        if result.y_cell is not None and result.y_cell[result.cells.region_ids == r][0] == 1
    )
    md = [
        f"# {title}",
        "",
        f"{len(result.cells.regions)} regions / {result.labels.size} cells; space = {result.settings['space']}; "
        f"clustering {json.dumps(result.model.params['clustering'])} on {json.dumps(result.model.params['reduction'])} "
        f"(sample {result.model.params['sample_size']}, L2 {result.model.params['l2_normalise']}, {result.model.params['assign_k']}-NN assignment).",
        f"Base rate (cells from {positive_name} regions) {base:.3f}; {positive_name} regions {n_pos}/{len(result.cells.regions)}"
        + (f"; held-out AUROC {held_out_auroc:.3f}" if held_out_auroc is not None else "")
        + f"; {result.n_clusters} clusters, noise {result.noise_fraction:.1%}.",
        "",
        f"| Cluster | Share | Regions (pos/neg, >{100 * 0.02:.0f}%) | Composition | Enriched (>1.5x) | Emp. frac. pos | Share AUROC | Margin relattn [95% CI] | Signed share | Sign consist. | Lift |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, t in T.sort_values("emp_frac_pos", ascending=False).iterrows():
        md.append(
            f"| {t.cluster} | {100 * t.share:.1f}% | {t.cores_present} ({t.pos_cores_present}/{t.neg_cores_present}) | {t.top_types} | {t.top_enriched} | "
            f"{t.emp_frac_pos:.2f} | {t.share_auroc:.2f} | {t.relattn:+.2f} [{t.relattn_lo:+.2f}, {t.relattn_hi:+.2f}] | {t.signed_share:+.2f} | {t.sign_consistency:.2f} | {t.lift:.2f} |"
        )
    conc = result.concordance
    if "spearman_emp_relattn" in conc:
        md.append(
            f"\nSpearman(emp_frac_pos, relattn) over {conc['n_clusters']} clusters: {conc['spearman_emp_relattn']:+.2f}"
            + (
                f"; Spearman(share AUROC, relattn): {conc['spearman_share_auroc_relattn']:+.2f}"
                if "spearman_share_auroc_relattn" in conc
                else ""
            )
        )
    CO, CL = result.colocalisation, result.clusters
    n = len(CL)
    if n >= 2:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        md.append(
            "\nCo-localisation log2(obs/exp) of cell-cell edges, top pairs: "
            + "; ".join(
                f"C{CL[i]}-C{CL[j]} {CO[i, j]:+.1f}"
                for i, j in sorted(pairs, key=lambda p: -CO[p])[:8]
            )
        )
        md.append(
            "Most avoided pairs: "
            + "; ".join(
                f"C{CL[i]}-C{CL[j]} {CO[i, j]:+.1f}"
                for i, j in sorted(pairs, key=lambda p: CO[p])[:5]
            )
        )
    (out / "report.md").write_text("\n".join(md) + "\n")
    paths["report"] = str(out / "report.md")

    if figures:
        try:
            from .figures import (
                clusters_overview,
                colocalisation_heatmap,
                embedding_triptych,
                umap_coordinates,
            )
        except Exception:  # pragma: no cover - matplotlib missing
            return paths
        samp = result.model.sample_idx
        P = result.model.reduce(result.features[samp])
        y_s = result.y_cell[samp] if result.y_cell is not None else np.zeros(samp.size)
        if P.shape[1] >= 2:
            ev = getattr(result.model.reducer, "explained_variance_ratio_", None)
            axis = (
                (f"PC1 ({100 * ev[0]:.1f}% var)", f"PC2 ({100 * ev[1]:.1f}% var)")
                if ev is not None
                else ("dim 1", "dim 2")
            )
            embedding_triptych(
                P[:, :2],
                result.labels[samp],
                result.margin[samp],
                y_s,
                out / "pca_triptych.png",
                title=f"{result.settings['space']} (PCA)",
                axis_labels=axis,
                positive_name=positive_name,
            )
            paths["pca_triptych"] = str(out / "pca_triptych.png")
        ucfg = dict(umap or {})
        if ucfg.get("enabled", True):
            U = umap_coordinates(
                P,
                n_neighbors=int(ucfg.get("n_neighbors", 30)),
                min_dist=float(ucfg.get("min_dist", 0.1)),
                metric=str(ucfg.get("metric", "cosine")),
                seed=int(ucfg.get("seed", 0)),
            )
            if U is not None:
                embedding_triptych(
                    U,
                    result.labels[samp],
                    result.margin[samp],
                    y_s,
                    out / "umap_triptych.png",
                    title=f"{result.settings['space']} (UMAP)",
                    axis_labels=("UMAP 1", "UMAP 2"),
                    positive_name=positive_name,
                )
                paths["umap_triptych"] = str(out / "umap_triptych.png")
        cohort = np.bincount(result.cells.ct, minlength=len(result.cells.cell_type_names)) / max(
            1, result.cells.ct.size
        )
        clusters_overview(
            T,
            out / "clusters_overview.png",
            cell_type_names=result.cells.cell_type_names,
            cohort_frequency=cohort,
            base_rate=base,
            positive_name=positive_name,
        )
        paths["clusters_overview"] = str(out / "clusters_overview.png")
        colocalisation_heatmap(CO, CL, out / "colocalisation.png")
        paths["colocalisation"] = str(out / "colocalisation.png")
    return paths
