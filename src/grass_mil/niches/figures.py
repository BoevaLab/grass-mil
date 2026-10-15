"""Figures of a niche analysis (matplotlib, written as PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "niche_colours",
    "niche_panel",
    "embedding_panel",
    "unit_composition",
    "unit_maps",
    "pair_heatmaps",
    "moran_heatmap",
    "colocalisation_heatmap",
    "survival_curves",
]

BACKGROUND_COLOUR = "#d9d9d9"


def niche_colours(names: Sequence[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    out, i = {}, 0
    for name in names:
        if name == "Background":
            out[name] = BACKGROUND_COLOUR
        else:
            out[name] = cmap(i % 20)
            i += 1
    return out


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def niche_panel(
    table: pd.DataFrame, cell_types: Sequence[str], path: Path, *, positive_name: str
) -> Path:
    """Composition | association | margin with 95% CI | share, one row per niche."""
    t = table[table["niche"] != "Background"].reset_index(drop=True)
    if t.empty:
        return path
    base = table.attrs.get("base_rate", np.nan)
    frac = [f"frac_{c}" for c in cell_types if f"frac_{c}" in t.columns]
    y = np.arange(len(t))
    widths = ([max(2.4, 0.3 * len(frac))] if frac else []) + [1.6, 1.6, 1.1]
    fig, axes = plt.subplots(
        1,
        len(widths),
        figsize=(1.6 * sum(widths) + 1.5, 0.45 * len(t) + 2.0),
        gridspec_kw={"width_ratios": widths, "wspace": 0.15},
        sharey=True,
    )
    axes = list(np.atleast_1d(axes))
    if frac:
        ax = axes.pop(0)
        im = ax.imshow(t[frac].to_numpy() * 100, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(frac)), [c[5:] for c in frac], rotation=90, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, location="top", shrink=0.8, pad=0.02)
        cbar.set_label("composition (% of niche cells)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax_assoc, ax_margin, ax_share = axes
    colours = ["#b2182b" if v > base else "#2166ac" for v in t["association"]]
    ax_assoc.barh(y, t["association"], color=colours, height=0.6)
    ax_assoc.axvline(base, color="k", ls=":", lw=1)
    ax_assoc.set_xlim(0, 1)
    ax_assoc.set_title(f"association\n(fraction from {positive_name} units)", fontsize=8)
    xerr = [t["margin"] - t["margin_lo"], t["margin_hi"] - t["margin"]]
    ax_margin.errorbar(t["margin"], y, xerr=xerr, fmt="o", color="#1b9e77", ms=4, capsize=2)
    ax_margin.axvline(0, color="k", lw=0.8)
    ax_margin.set_title(f"attention-weighted margin\n(95% CI; + = {positive_name})", fontsize=8)
    ax_share.barh(y, 100 * t["share"], color="#7f7f7f", height=0.6)
    ax_share.set_title("share\n(% of cells)", fontsize=8)
    axes_all = fig.axes
    axes_all[0].set_yticks(y, t["niche"], fontsize=8)
    if not frac:
        axes_all[0].invert_yaxis()  # Niche 1 on top (imshow already does this)
    for ax in (ax_assoc, ax_margin, ax_share):
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def embedding_panel(
    coords: np.ndarray,
    niche_names: np.ndarray,
    margin: np.ndarray,
    positive: np.ndarray,
    path: Path,
    *,
    positive_name: str,
    axis_labels=("PC 1", "PC 2"),
) -> Path:
    """The clustering space (first two PCs): niches | margin | unit class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    size = float(np.clip(20000.0 / max(len(coords), 1), 0.5, 20.0))
    colours = niche_colours(sorted(set(niche_names), key=lambda n: (n == "Background", n)))
    for name, colour in colours.items():
        m = niche_names == name
        axes[0].scatter(coords[m, 0], coords[m, 1], s=size, color=colour, lw=0, label=name)
    axes[0].legend(markerscale=6, fontsize=6, ncol=2)
    lim = float(np.nanpercentile(np.abs(margin), 99)) or 1.0
    sc = axes[1].scatter(
        coords[:, 0], coords[:, 1], s=size, c=margin, cmap="RdBu_r", vmin=-lim, vmax=lim, lw=0
    )
    plt.colorbar(sc, ax=axes[1], label=f"margin (+ = {positive_name})")
    hb = axes[2].hexbin(
        coords[:, 0],
        coords[:, 1],
        C=positive.astype(float),
        gridsize=40,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        mincnt=3,
    )
    plt.colorbar(hb, ax=axes[2], label=f"fraction from {positive_name} units")
    for ax, title in zip(axes, ("niches", "attention-weighted margin", "empirical association")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_xticks([])
        ax.set_yticks([])
    return _save(fig, path)


def unit_composition(
    shares: pd.DataFrame, positive: dict[str, bool], path: Path, *, positive_name: str
) -> Path:
    """Stacked niche shares of every unit, negative units first."""
    order = sorted(
        shares.index,
        key=lambda u: (
            positive.get(u, False),
            -shares.loc[u].drop("Background", errors="ignore").max(),
        ),
    )
    s = shares.loc[order]
    colours = niche_colours(list(s.columns))
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(s) + 3), 3.6))
    bottom = np.zeros(len(s))
    for name in s.columns:
        ax.bar(range(len(s)), s[name], bottom=bottom, color=colours[name], width=0.9, label=name)
        bottom += s[name].to_numpy()
    n_neg = sum(not positive.get(u, False) for u in order)
    ax.axvline(n_neg - 0.5, color="k", lw=1)
    ax.set_xticks([])
    ax.set_xlabel(f"units (left: other, right: {positive_name})")
    ax.set_ylabel("share of cells")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, path)


def unit_maps(
    pos: np.ndarray,
    cell_types: np.ndarray,
    type_names: Sequence[str],
    margin: np.ndarray,
    niche_names: np.ndarray,
    path: Path,
    *,
    title: str,
) -> Path:
    """One unit in the tissue: cell types | cell margins | niches."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap("tab20")
    size = float(np.clip(4000.0 / max(len(pos), 1), 1.0, 30.0))
    for j, name in enumerate(type_names):
        m = cell_types == j
        if m.any():
            axes[0].scatter(pos[m, 0], pos[m, 1], s=size, color=cmap(j % 20), lw=0, label=name)
    axes[0].legend(markerscale=3, fontsize=6, ncol=2)
    lim = float(np.nanpercentile(np.abs(margin), 99)) or 1.0
    sc = axes[1].scatter(
        pos[:, 0], pos[:, 1], s=size, c=margin, cmap="RdBu_r", vmin=-lim, vmax=lim, lw=0
    )
    plt.colorbar(sc, ax=axes[1], label="margin")
    colours = niche_colours(sorted(set(niche_names), key=lambda n: (n == "Background", n)))
    for name, colour in colours.items():
        m = niche_names == name
        axes[2].scatter(pos[m, 0], pos[m, 1], s=size, color=colour, lw=0, label=name)
    axes[2].legend(markerscale=3, fontsize=6)
    for ax, sub in zip(axes, ("cell types", "attention-weighted margin", "niches")):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}: {sub}", fontsize=9)
    return _save(fig, path)


def pair_heatmaps(
    table: pd.DataFrame, value: str, cell_types: Sequence[str], path: Path, *, label: str
) -> Path:
    """One symmetric cell-type x cell-type heatmap per niche."""
    niches = list(dict.fromkeys(table["niche"]))
    if not niches:
        return path
    k = len(cell_types)
    index = {n: i for i, n in enumerate(cell_types)}
    cols = min(4, len(niches))
    rows = int(np.ceil(len(niches) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols + 1, 3.0 * rows + 0.5), squeeze=False)
    lim = float(np.nanpercentile(np.abs(table[value]), 98)) if table[value].notna().any() else 1.0
    for ax, niche in zip(axes.flat, niches):
        m = np.full((k, k), np.nan)
        for _, r in table[table["niche"] == niche].iterrows():
            a, b = index[r["type_a"]], index[r["type_b"]]
            m[a, b] = m[b, a] = r[value]
        im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
        rows_of = table[table["niche"] == niche]
        suffix = f" (r* = {rows_of['r_star'].iloc[0]:.3g})" if "r_star" in table else ""
        ax.set_title(niche + suffix, fontsize=8)
        ax.set_xticks(range(k), cell_types, rotation=90, fontsize=5)
        ax.set_yticks(range(k), cell_types, fontsize=5)
    for ax in list(axes.flat)[len(niches) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label=label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def moran_heatmap(table: pd.DataFrame, path: Path) -> Path:
    """Moran's I per niche (rows) and cell type (columns); dots mark q < 0.05."""
    if table.empty:
        return path
    m = table.pivot(index="niche", columns="cell_type", values="morans_i")
    q = table.pivot(index="niche", columns="cell_type", values="q").reindex_like(m)
    fig, ax = plt.subplots(figsize=(0.35 * m.shape[1] + 3, 0.4 * m.shape[0] + 1.5))
    lim = float(np.nanmax(np.abs(m.to_numpy()))) or 1.0
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    yy, xx = np.nonzero((q.to_numpy() < 0.05))
    ax.scatter(xx, yy, s=6, color="k")
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Moran's I")
    return _save(fig, path)


def colocalisation_heatmap(matrix: np.ndarray, names: Sequence[str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(0.45 * len(names) + 2.5, 0.45 * len(names) + 1.8))
    lim = float(np.nanmax(np.abs(matrix))) or 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    plt.colorbar(im, ax=ax, label="log2 observed / expected contacts")
    return _save(fig, path)


def survival_curves(groups: pd.DataFrame, path: Path, *, score_label: str) -> Path:
    from .survival import kaplan_meier

    fig, ax = plt.subplots(figsize=(5, 4))
    for tertile, label, colour in (
        (0, "low", "#2166ac"),
        (1, "intermediate", "#7f7f7f"),
        (2, "high", "#b2182b"),
    ):
        g = groups[groups["tertile"] == tertile]
        if len(g):
            t, s = kaplan_meier(g["time"].to_numpy(float), g["event"].to_numpy(float))
            ax.step(t, s, where="post", color=colour, label=f"{label} (n={len(g)})")
    ax.set_xlabel("time")
    ax.set_ylabel("survival probability")
    ax.set_title(
        f"{score_label}; log-rank p = {groups.attrs.get('logrank_p', float('nan')):.3g}",
        fontsize=8,
    )
    ax.legend(fontsize=7)
    return _save(fig, path)
