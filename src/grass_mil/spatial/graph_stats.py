"""Spatial statistics of labelled cells on a cell graph or in the plane.

* :func:`enrichment_zscores`: neighbourhood enrichment, the z-score of the
  observed count of edges between two labels against label permutations over
  the fixed graph (the statistic of ``squidpy.gr.nhood_enrichment``).
* :func:`morans_i`: Moran's I of label indicators over the binary adjacency.
* :func:`cross_l`: Ripley's centred cross-L, ``L_ab(r) - r``.
* :func:`contact_counts`: symmetric label-pair edge counts.
* :func:`colocalisation`: ``log2`` observed/expected edges between labels.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "undirected_edges",
    "contact_counts",
    "enrichment_zscores",
    "morans_i",
    "cross_l",
    "colocalisation",
]


def undirected_edges(edge_index: np.ndarray) -> np.ndarray:
    """Each undirected edge once (``src < dst``)."""
    a, b = np.asarray(edge_index)
    keep = a < b
    return np.stack([a[keep], b[keep]])


def contact_counts(labels: np.ndarray, edges: np.ndarray, n_labels: int) -> np.ndarray:
    """Symmetric ``[n_labels, n_labels]`` edge-end counts of undirected edges.

    Entry ``(a, b)`` counts the edges between labels ``a`` and ``b``; the
    diagonal counts same-label edges twice (once per end), so a row sums to the
    number of edge ends carrying that label.
    """
    counts = np.zeros((n_labels, n_labels))
    np.add.at(counts, (labels[edges[0]], labels[edges[1]]), 1.0)
    return counts + counts.T


def enrichment_zscores(
    labels: np.ndarray, edges: np.ndarray, n_labels: int, *, n_perms: int, rng: np.random.Generator
) -> np.ndarray:
    """Neighbourhood enrichment z-scores; labels absent from the graph get NaN."""
    labels = np.asarray(labels, dtype=int)
    observed = contact_counts(labels, edges, n_labels)
    null = np.stack(
        [contact_counts(rng.permutation(labels), edges, n_labels) for _ in range(n_perms)]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (observed - null.mean(0)) / null.std(0)
    z[~np.isfinite(z)] = 0.0
    absent = np.setdiff1d(np.arange(n_labels), labels)
    z[absent, :] = np.nan
    z[:, absent] = np.nan
    return z


def morans_i(indicators: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Moran's I of every column of ``indicators`` ``[n, k]`` over the undirected ``edges``.

    ``I = (n / S0) * z' W z / z' z`` with binary symmetric ``W`` and the
    centred column ``z``; NaN for constant columns or a graph without edges.
    """
    from scipy.sparse import coo_matrix

    x = np.asarray(indicators, dtype=float)
    n = x.shape[0]
    if n < 2 or edges.shape[1] == 0:
        return np.full(x.shape[1], np.nan)
    rows = np.concatenate([edges[0], edges[1]])
    cols = np.concatenate([edges[1], edges[0]])
    w = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    z = x - x.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (n / w.sum()) * (z * (w @ z)).sum(0) / (z * z).sum(0)


def cross_l(
    coords: np.ndarray, in_a: np.ndarray, in_b: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """Centred cross-L ``L_ab(r) - r`` of two point sets on their bounding box.

    ``K_ab(r) = |A| / (n_a n_b) * #{(i in a, j in b): d_ij <= r, i != j}`` with
    ``n_a (n_a - 1)`` in the denominator when ``a`` and ``b`` are the same set.
    """
    from scipy.spatial import cKDTree

    a, b = coords[in_a], coords[in_b]
    span = coords.max(0) - coords.min(0)
    area = max(float(np.prod(span)), 1e-12)
    same = bool(np.array_equal(in_a, in_b))
    counts = cKDTree(a).count_neighbors(cKDTree(b), radii).astype(float)
    if same:
        counts -= len(a)
        pairs = len(a) * (len(a) - 1)
    else:
        pairs = len(a) * len(b)
    k = area * counts / max(pairs, 1)
    return np.sqrt(np.maximum(k, 0.0) / np.pi) - radii


def colocalisation(labels: np.ndarray, edges: np.ndarray, n_labels: int) -> np.ndarray:
    """``log2((observed + 1) / (expected + 1))`` edges between labels, with the
    expectation ``d_a d_b / sum(d)`` from the label degrees (negative labels ignored)."""
    keep = (labels[edges[0]] >= 0) & (labels[edges[1]] >= 0)
    observed = contact_counts(np.maximum(labels, 0), edges[:, keep], n_labels)
    degree = observed.sum(1)
    expected = np.outer(degree, degree) / max(degree.sum(), 1.0)
    return np.log2((observed + 1.0) / (expected + 1.0))
