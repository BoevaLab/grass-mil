"""Figures of the niche report (matplotlib; UMAP optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["embedding_triptych", "clusters_overview", "colocalisation_heatmap", "umap_coordinates"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def umap_coordinates(
    P: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 0,
) -> Optional[np.ndarray]:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    return umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
    ).fit_transform(P)


def embedding_triptych(
    coords: np.ndarray,
    labels: np.ndarray,
    margin: np.ndarray,
    y_cell: np.ndarray,
    path: Path,
    *,
    title: str,
    axis_labels: Sequence[str] = ("dim 1", "dim 2"),
    positive_name: str = "positive",
    margin_clip: float = 3.0,
) -> Path:
    """Clusters | attention-weighted margin | empirical association, side by side."""
    plt = _plt()
    clusters = sorted(set(labels.tolist()) - {-1})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    noise = labels < 0
    ax[0].scatter(coords[noise, 0], coords[noise, 1], s=1, c="#cccccc", lw=0, label="noise")
    for i, c in enumerate(clusters):
        mm = labels == c
        ax[0].scatter(coords[mm, 0], coords[mm, 1], s=1.5, color=cmap(i % 20), lw=0, label=f"C{c}")
    ax[0].legend(markerscale=8, fontsize=6, ncol=2, loc="best")
    ax[0].set_title(f"{title}: clusters ({len(clusters)}, noise {noise.mean():.0%})", fontsize=9)
    mv = np.clip(margin, -margin_clip, margin_clip)
    sc = ax[1].scatter(
        coords[:, 0],
        coords[:, 1],
        s=1.5,
        c=mv,
        cmap="RdBu_r",
        vmin=-margin_clip,
        vmax=margin_clip,
        lw=0,
    )
    plt.colorbar(sc, ax=ax[1], label=f"relattn margin (+ = {positive_name})")
    ax[1].set_title("attention-weighted margin", fontsize=9)
    hb = ax[2].hexbin(
        coords[:, 0], coords[:, 1], C=y_cell, gridsize=45, cmap="RdBu_r", vmin=0, vmax=1, mincnt=5
    )
    plt.colorbar(hb, ax=ax[2], label=f"fraction of cells from {positive_name} regions")
    ax[2].set_title("empirical association", fontsize=9)
    for a in ax:
        a.set_xlabel(axis_labels[0])
        a.set_ylabel(axis_labels[1])
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def clusters_overview(
    table: pd.DataFrame,
    path: Path,
    *,
    cell_type_names: Sequence[str],
    cohort_frequency: np.ndarray,
    base_rate: float,
    positive_name: str = "positive",
    max_types: int = 20,
) -> Path:
    """Composition heatmap | empirical association | margin with CI, per cluster."""
    plt = _plt()
    tt = table[table["cluster"] != "Noise"].sort_values("emp_frac_pos", ascending=False)
    if tt.empty:
        return path
    typ = np.argsort(-np.asarray(cohort_frequency))[:max_types]
    comp_cols = [f"comp_{cell_type_names[j]}" for j in typ]
    H = tt[comp_cols].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(
        1, 3, figsize=(16, 0.42 * len(tt) + 2.2), gridspec_kw=dict(width_ratios=[2.6, 1, 1.3])
    )
    im = ax[0].imshow(
        H, aspect="auto", cmap="viridis", vmin=0, vmax=min(100, H.max() if H.size else 100)
    )
    ax[0].set_xticks(range(len(typ)))
    ax[0].set_xticklabels([cell_type_names[j] for j in typ], rotation=90, fontsize=7)
    ax[0].set_yticks(range(len(tt)))
    ax[0].set_yticklabels(
        [
            f"{c} ({100 * s:.1f}%, {n} regions)"
            for c, s, n in zip(tt.cluster, tt.share, tt.cores_present)
        ],
        fontsize=7,
    )
    plt.colorbar(im, ax=ax[0], label="% of cluster cells", fraction=0.03)
    yy = np.arange(len(tt))[::-1]
    ax[1].barh(
        yy,
        tt.emp_frac_pos,
        color=["#b2182b" if v > base_rate else "#2166ac" for v in tt.emp_frac_pos],
        height=0.6,
    )
    ax[1].axvline(base_rate, color="k", ls=":")
    ax[1].set_yticks([])
    ax[1].set_xlabel(f"fraction of cells from {positive_name} regions")
    ax[1].set_xlim(0, 1)
    ax[2].errorbar(
        tt.relattn,
        yy,
        xerr=[tt.relattn - tt.relattn_lo, tt.relattn_hi - tt.relattn],
        fmt="o",
        ms=4,
        color="#1b9e77",
        capsize=2,
    )
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_yticks([])
    ax[2].set_xlabel(f"attention-weighted margin (+ = {positive_name})")
    for a in ax[1:]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def colocalisation_heatmap(CO: np.ndarray, clusters: Sequence[int], path: Path) -> Path:
    plt = _plt()
    n = len(clusters)
    if n == 0:
        return path
    fig, ax = plt.subplots(figsize=(0.45 * n + 2, 0.45 * n + 1.5))
    v = float(np.nanmax(np.abs(CO))) if np.isfinite(CO).any() else 1.0
    im = ax.imshow(CO, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in clusters], rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"C{c}" for c in clusters], fontsize=7)
    plt.colorbar(im, ax=ax, label="log2 obs/exp cell-cell edges")
    ax.set_title("cluster co-localisation", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
