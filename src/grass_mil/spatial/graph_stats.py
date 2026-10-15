"""Spatial statistics of labelled cells on a cell graph or in the plane.

* :func:`enrichment`: neighbourhood enrichment of label pairs against label
  permutations over the fixed graph: the z-score of
  ``squidpy.gr.nhood_enrichment`` and a size-free ``log2`` ratio.
* :func:`morans_i`: Moran's I of label indicators over the binary adjacency,
  with its exact mean and variance under randomisation.
* :func:`cross_l`: Ripley's cross-L of label pairs relative to random labelling.
* :func:`contact_counts`: symmetric label-pair edge counts.
* :func:`colocalisation`: ``log2`` observed/expected edges between labels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["contact_counts", "enrichment", "morans_i", "cross_l", "colocalisation"]


def contact_counts(labels: np.ndarray, edges: np.ndarray, n_labels: int) -> np.ndarray:
    """Symmetric ``[n_labels, n_labels]`` edge-end counts of undirected edges.

    Entry ``(a, b)`` counts the edges between labels ``a`` and ``b``; the
    diagonal counts same-label edges twice (once per end), so a row sums to the
    number of edge ends carrying that label.
    """
    flat = labels[edges[0]] * n_labels + labels[edges[1]]
    counts = np.bincount(flat, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    return (counts + counts.T).astype(float)


def enrichment(
    labels: np.ndarray, edges: np.ndarray, n_labels: int, *, n_perms: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbourhood enrichment of every label pair: ``(z, log2 ratio)``.

    ``z`` compares the observed edge count with label permutations over the
    fixed graph; it grows with the number of edges, so it is comparable only
    between graphs of similar size. ``log2((observed + 1) / (null mean + 1))``
    is the size-free effect. Both are NaN for labels absent from the graph, and
    ``z`` where the permutations cannot change the count.
    """
    labels = np.asarray(labels, dtype=int)
    observed = contact_counts(labels, edges, n_labels)
    null = np.stack(
        [contact_counts(rng.permutation(labels), edges, n_labels) for _ in range(n_perms)]
    )
    mean, sd = null.mean(0), null.std(0)
    z = np.full_like(observed, np.nan)
    np.divide(observed - mean, sd, out=z, where=sd > 0)
    ratio = np.log2((observed + 1.0) / (mean + 1.0))
    absent = ~np.isin(np.arange(n_labels), labels)
    for values in (z, ratio):
        values[absent, :] = values[:, absent] = np.nan
    return z, ratio


def morans_i(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Moran's I of every column of ``values`` ``[n, k]`` over undirected ``edges`` (each once).

    ``I = (n / S0) z'Wz / z'z`` with binary symmetric ``W`` and centred columns
    ``z``. Returns ``(I, E[I], Var[I])``, the moments under randomisation
    (Cliff and Ord), which equal the mean and variance over label permutations,
    so ``(I - E) / sqrt(Var)`` is the permutation z-score without permutations.
    NaN for constant columns, fewer than four cells or no edges.
    """
    x = np.asarray(values, dtype=float)
    n, k = x.shape
    if n < 4 or edges.shape[1] == 0:
        return np.full(k, np.nan), float("nan"), np.full(k, np.nan)
    degree = np.bincount(edges.ravel(), minlength=n).astype(float)
    s0 = 2.0 * edges.shape[1]
    s1, s2 = 2.0 * s0, 4.0 * float((degree**2).sum())
    z = x - x.mean(0)
    m2 = (z**2).sum(0)
    expected = -1.0 / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        i = (n / s0) * 2.0 * (z[edges[0]] * z[edges[1]]).sum(0) / m2
        kurtosis = n * (z**4).sum(0) / m2**2
    a = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0**2)
    b = kurtosis * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0**2)
    var = (a - b) / ((n - 1) * (n - 2) * (n - 3) * s0**2) - expected**2
    return i, expected, np.where(var > 0, var, np.nan)


def cross_l(
    coords: np.ndarray, labels: np.ndarray, pairs: Sequence[tuple[int, int]], radii: np.ndarray
) -> np.ndarray:
    """Cross-L of label pairs relative to random labelling, ``[len(pairs), len(radii)]``.

    ``K_ab(r) = |W| / (n_a n_b) * #{(i in a, j in b), i != j: d_ij <= r}`` on
    the bounding box ``W`` of ``coords``, ``L = sqrt(K / pi)``, and the value
    is ``L_ab(r) - L(r)``, where ``L`` is the same function of all points.
    Under random labelling of the fixed points ``E[K_ab] = K``, so the window,
    edge effects and the clustering of the points themselves cancel: a
    positive value means ``b`` lies closer to ``a`` than cells in general do.
    NaN for a pair with a label of fewer than two points.
    """
    from scipy.spatial import cKDTree

    radii = np.asarray(radii, dtype=float)
    out = np.full((len(pairs), radii.size), np.nan)
    n = len(coords)
    if n < 2:
        return out
    area = max(float(np.prod(np.ptp(coords, axis=0))), 1e-12)

    def l_function(counts: np.ndarray, n_pairs: int) -> np.ndarray:
        return np.sqrt(area * counts / n_pairs / np.pi)

    tree = cKDTree(coords)
    reference = l_function(tree.count_neighbors(tree, radii) - n, n * (n - 1))
    points = {c: coords[labels == c] for pair in pairs for c in pair}
    trees = {c: cKDTree(p) for c, p in points.items() if len(p) >= 2}
    for row, (a, b) in enumerate(pairs):
        if a not in trees or b not in trees:
            continue
        n_a, n_b = len(points[a]), len(points[b])
        counts = trees[a].count_neighbors(trees[b], radii).astype(float)
        if a == b:
            counts, n_pairs = counts - n_a, n_a * (n_a - 1)
        else:
            n_pairs = n_a * n_b
        out[row] = l_function(counts, n_pairs) - reference
    return out


def colocalisation(labels: np.ndarray, edges: np.ndarray, n_labels: int) -> np.ndarray:
    """``log2((observed + 1) / (expected + 1))`` edges between labels, with the
    expectation ``d_a d_b / sum(d)`` from the label degrees (negative labels ignored)."""
    keep = (labels[edges[0]] >= 0) & (labels[edges[1]] >= 0)
    observed = contact_counts(np.maximum(labels, 0), edges[:, keep], n_labels)
    degree = observed.sum(1)
    expected = np.outer(degree, degree) / max(degree.sum(), 1.0)
    return np.log2((observed + 1.0) / (expected + 1.0))
