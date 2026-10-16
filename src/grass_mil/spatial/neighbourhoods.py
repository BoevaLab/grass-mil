"""Cellular neighbourhoods: the cell-type composition of every cell's k-NN window, clustered."""

from __future__ import annotations

import numpy as np

__all__ = ["window_composition", "fit_neighbourhoods"]


def window_composition(
    coords: np.ndarray, labels: np.ndarray, n_labels: int, k: int
) -> np.ndarray:
    """Fraction of each label among a cell and its ``k`` nearest neighbours."""
    from scipy.spatial import cKDTree

    _, nn = cKDTree(coords).query(coords, k=min(k + 1, len(coords)))
    nn = np.asarray(nn).reshape(len(coords), -1)
    counts = np.zeros((len(coords), n_labels), dtype=np.float32)
    for column in nn.T:
        counts[np.arange(len(coords)), labels[column]] += 1
    return counts / counts.sum(1, keepdims=True)


def fit_neighbourhoods(windows: list[np.ndarray], n: int, sample_size: int, seed: int):
    """k-means of the windows (fitted on a sample): per-unit labels and the centres."""
    from sklearn.cluster import MiniBatchKMeans

    stacked = np.concatenate(windows)
    sample = np.random.default_rng(seed).choice(
        len(stacked), min(sample_size, len(stacked)), replace=False
    )
    kmeans = MiniBatchKMeans(n, random_state=seed, n_init=10).fit(stacked[sample])
    return [kmeans.predict(w) for w in windows], kmeans.cluster_centers_
