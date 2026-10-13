"""Neighbourhood enrichment z-scores on a cell graph (label permutation null)."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["nhood_enrichment_zscores", "pair_counts"]


def pair_counts(labels: np.ndarray, edge_index: np.ndarray, n_groups: int) -> np.ndarray:
    """Symmetric (undirected) count of edges between group pairs."""
    la = labels[edge_index[0]]
    lb = labels[edge_index[1]]
    M = np.zeros((n_groups, n_groups), dtype=np.float64)
    np.add.at(M, (la, lb), 1.0)
    return M + M.T


def nhood_enrichment_zscores(
    labels: np.ndarray,
    edge_index: np.ndarray,
    *,
    n_groups: int,
    n_perms: int = 200,
    seed: int = 0,
    backend: str = "numpy",
    coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """z-score of observed vs permuted group-pair edge counts (squidpy's statistic).

    ``edge_index`` holds each undirected edge once or twice; counts are
    symmetrised either way. ``backend='squidpy'`` delegates to
    ``squidpy.gr.nhood_enrichment`` on the same graph (optional dependency).
    Groups absent from the region get NaN rows/columns.
    """
    labels = np.asarray(labels).astype(int)
    ei = np.asarray(edge_index).astype(int)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    # one direction only, so symmetrisation counts every edge exactly once per orientation
    a, b = ei
    keep = a < b
    if keep.sum() == 0 and ei.shape[1] > 0:
        keep = np.ones(ei.shape[1], dtype=bool)
    ei = np.stack([a[keep], b[keep]])
    if backend == "squidpy":
        return _squidpy_zscores(
            labels, ei, n_groups=n_groups, n_perms=n_perms, seed=seed, coords=coords
        )
    present = np.unique(labels)
    rng = np.random.default_rng(int(seed))
    observed = pair_counts(labels, ei, n_groups)
    perms = np.zeros((int(n_perms), n_groups, n_groups))
    for i in range(int(n_perms)):
        perms[i] = pair_counts(rng.permutation(labels), ei, n_groups)
    mean = perms.mean(axis=0)
    std = perms.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - mean) / std
    z[~np.isfinite(z)] = 0.0
    absent = np.setdiff1d(np.arange(n_groups), present)
    z[absent, :] = np.nan
    z[:, absent] = np.nan
    return z


def _squidpy_zscores(labels, ei, *, n_groups, n_perms, seed, coords):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    import squidpy as sq

    n = labels.size
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    adata = ad.AnnData(X=np.zeros((n, 1), np.float32))
    adata.obs["g"] = pd.Categorical(labels.astype(str))
    adata.obsm["spatial"] = (
        np.asarray(coords, dtype=float) if coords is not None else np.zeros((n, 2))
    )
    adata.obsp["spatial_connectivities"] = A
    adata.uns["spatial_neighbors"] = {
        "connectivities_key": "spatial_connectivities",
        "distances_key": "spatial_distances",
        "params": {"n_neighbors": 6, "radius": None, "transform": None},
    }
    sq.gr.nhood_enrichment(
        adata,
        cluster_key="g",
        n_perms=int(n_perms),
        seed=int(seed),
        show_progress_bar=False,
        n_jobs=1,
    )
    z = adata.uns["g_nhood_enrichment"]["zscore"]
    cats = [int(c) for c in adata.obs["g"].cat.categories]
    out = np.full((n_groups, n_groups), np.nan)
    for x, cx in enumerate(cats):
        for y, cy in enumerate(cats):
            out[cx, cy] = z[x, y]
    return out
