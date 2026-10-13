"""Cellular neighbourhoods: k-NN window composition clustered with k-means."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["window_composition", "cellular_neighbourhoods"]


def window_composition(
    coords: np.ndarray, labels: np.ndarray, *, n_groups: int, k: int = 10
) -> np.ndarray:
    """Fraction of each group among a cell and its ``k`` nearest neighbours."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels).astype(int)
    n = coords.shape[0]
    kk = min(int(k) + 1, n)
    _, nn = cKDTree(coords).query(coords, k=kk)
    nn = np.asarray(nn).reshape(n, -1)
    W = np.zeros((n, n_groups), dtype=np.float32)
    for j in range(nn.shape[1]):
        W[np.arange(n), labels[nn[:, j]]] += 1
    return W / W.sum(axis=1, keepdims=True)


def cellular_neighbourhoods(
    windows: Sequence[np.ndarray],
    *,
    n_cn: int = 10,
    sample_size: int = 200_000,
    seed: int = 0,
    n_init: int = 10,
    batch_size: int = 4096,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Fit ``MiniBatchKMeans(n_cn)`` on a sample of all windows; return per-region CN labels and centres."""
    from sklearn.cluster import MiniBatchKMeans

    W = np.concatenate(list(windows))
    rng = np.random.default_rng(int(seed))
    sub = rng.choice(len(W), min(int(sample_size), len(W)), replace=False)
    km = MiniBatchKMeans(
        int(n_cn), random_state=int(seed), n_init=int(n_init), batch_size=int(batch_size)
    ).fit(W[sub])
    out = []
    for w in windows:
        out.append(km.predict(np.asarray(w, dtype=np.float32)))
    return out, km.cluster_centers_
