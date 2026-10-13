"""Niche-niche co-localisation on the cell graph: log2(observed / expected) edge counts."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["cluster_colocalisation"]


def cluster_colocalisation(
    labels: np.ndarray, edge_index: np.ndarray
) -> Tuple[np.ndarray, List[int]]:
    """``log2((observed + 1) / (E + 1))`` with ``E_ab = d_a d_b / sum(d)`` from the
    symmetrised cluster-cluster edge count matrix ``observed`` (noise cells ignored)."""
    labels = np.asarray(labels).astype(int)
    ei = np.asarray(edge_index).astype(int)
    la, lb = labels[ei[0]], labels[ei[1]]
    keep = (la >= 0) & (lb >= 0)
    la, lb = la[keep], lb[keep]
    clusters = sorted(set(labels.tolist()) - {-1})
    idx = {c: i for i, c in enumerate(clusters)}
    n = len(clusters)
    observed = np.zeros((n, n))
    if la.size:
        ia = np.vectorize(idx.get)(la)
        ib = np.vectorize(idx.get)(lb)
        np.add.at(observed, (ia, ib), 1)
    observed = observed + observed.T
    deg = observed.sum(axis=1)
    total = deg.sum()
    expected = np.outer(deg, deg) / total if total > 0 else np.zeros_like(observed)
    return np.log2((observed + 1.0) / (expected + 1.0)), clusters
