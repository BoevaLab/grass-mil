"""Ripley cross-K between a reference group and partner groups, normalised by pi r^2."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["cross_k"]


def cross_k(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    group_names: Sequence[str],
    reference: Sequence[str],
    partners: Sequence[str],
    radii: Sequence[float] = (20.0, 50.0, 100.0),
    min_cells: int = 5,
) -> Dict[str, float]:
    """``K(r) = area * pairs_within_r / (n_ref * n_partner) / (pi r^2)``; 1 = CSR.

    Self pairs are removed when the partner group is the reference itself.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels).astype(int)
    gl = np.asarray(list(group_names), dtype=object)[labels]
    ref = np.isin(gl, list(reference))
    out: Dict[str, float] = {}
    if ref.sum() < int(min_cells):
        return out
    area = float(np.prod(coords.max(axis=0) - coords.min(axis=0)))
    tree_ref = cKDTree(coords[ref])
    for x in partners:
        mk = ref if x == "reference" else (gl == x)
        if mk.sum() < int(min_cells):
            continue
        tree_x = cKDTree(coords[mk])
        self_pairs = int(mk.sum()) if x == "reference" else 0
        for r in radii:
            cnt = tree_ref.count_neighbors(tree_x, float(r)) - self_pairs
            out[f"K_{x}_r{int(r) if float(r).is_integer() else r}"] = float(
                area * cnt / (ref.sum() * mk.sum()) / (np.pi * float(r) ** 2)
            )
    return out
