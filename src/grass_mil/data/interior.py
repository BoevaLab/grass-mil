"""Restricting ego-graph roots to the tissue interior.

Ego-graphs rooted at the edge of a section are truncated by the boundary rather
than by biology, so they are not comparable to interior ones. "Interior"
excludes the ``n_hops`` graph-hop rim around the convex hull of the cells; holes
and low-cellularity gaps inside the tissue are not treated as boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["InteriorConfig", "InteriorCache", "interior_mask", "interior_indices"]


@dataclass(frozen=True)
class InteriorConfig:
    """Interior root restriction.

    Attributes:
        enabled: Restrict roots to the interior.
        n_hops: Width of the excluded rim in graph hops (``None``: the sampler
            depth, so the rim matches the ego-graph radius).
        hull_tolerance: Cells within this multiple of the median
            nearest-neighbour distance of the hull edge start the rim.
        keep_largest_component: Drop interior islands cut off from the main body.
        cache_dir: Directory caching the masks (``None``: no cache).
    """

    enabled: bool = False
    n_hops: int | None = None
    hull_tolerance: float = 1.0
    keep_largest_component: bool = True
    cache_dir: str | None = None

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> InteriorConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown interior_roots keys: {sorted(unknown)}")
        return cls(**cfg)


def _hull_distance(coords: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Distance from every point to the convex-hull polygon's edges."""
    starts = coords[hull]
    ends = coords[np.roll(hull, -1)]
    out = np.full(coords.shape[0], np.inf)
    for start, end in zip(starts, ends):
        segment = end - start
        t = np.clip(((coords - start) @ segment) / (segment @ segment + 1e-12), 0.0, 1.0)
        out = np.minimum(out, np.linalg.norm(coords - (start + np.outer(t, segment)), axis=1))
    return out


def _adjacency(edge_index: np.ndarray, n: int):
    from scipy.sparse import coo_matrix

    rows = np.concatenate([edge_index[0], edge_index[1]])
    cols = np.concatenate([edge_index[1], edge_index[0]])
    return coo_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(n, n)).tocsr()


def interior_mask(
    coords: np.ndarray,
    edge_index: np.ndarray,
    n_hops: int,
    *,
    hull_tolerance: float = 1.0,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Boolean per-cell mask, ``True`` for interior cells.

    Degenerate inputs (fewer than 4 cells, collinear cells, ``n_hops <= 0``)
    mark every cell interior.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull, QhullError, cKDTree

    n = int(coords.shape[0])
    if n_hops <= 0 or n < 4:
        return np.ones(n, dtype=bool)
    try:
        hull = ConvexHull(coords).vertices
    except QhullError:
        return np.ones(n, dtype=bool)

    spacing = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    rim = _hull_distance(coords, hull) <= hull_tolerance * max(spacing, 1e-12)
    adjacency = _adjacency(edge_index, n)
    reached = rim.copy()
    frontier = rim.copy()
    for _ in range(int(n_hops) - 1):
        frontier = adjacency.dot(frontier.astype(np.int8)).astype(bool) & ~reached
        if not frontier.any():
            break
        reached |= frontier
    interior = ~reached

    if keep_largest_component and interior.any():
        idx = np.flatnonzero(interior)
        n_comp, labels = connected_components(adjacency[idx][:, idx], directed=False)
        if n_comp > 1:
            interior = np.zeros(n, dtype=bool)
            interior[idx[labels == np.argmax(np.bincount(labels))]] = True
    return interior


def interior_indices(data, cfg: InteriorConfig, *, n_hops: int) -> np.ndarray:
    """Interior node indices of a graph unit (every node when the mask is empty)."""
    coords = data.pos.detach().cpu().numpy().astype(float)[:, :2]
    mask = interior_mask(
        coords,
        data.edge_index.detach().cpu().numpy(),
        int(cfg.n_hops) if cfg.n_hops is not None else int(n_hops),
        hull_tolerance=cfg.hull_tolerance,
        keep_largest_component=cfg.keep_largest_component,
    )
    idx = np.flatnonzero(mask)
    return idx if idx.size else np.arange(coords.shape[0])


class InteriorCache:
    """On-disk cache of interior indices, keyed by unit and every mask setting."""

    def __init__(self, cache_dir: str | None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _path(self, unit_id: str, cfg: InteriorConfig, n_hops: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{unit_id}__h{n_hops}_t{cfg.hull_tolerance}_c{int(cfg.keep_largest_component)}"
        return self.cache_dir / ("".join(c if c.isalnum() or c in "._-" else "_" for c in key) + ".npy")

    def get(self, data, cfg: InteriorConfig, *, n_hops: int) -> np.ndarray:
        path = self._path(str(data.unit_id), cfg, n_hops)
        if path is not None and path.exists():
            return np.load(path)
        idx = interior_indices(data, cfg, n_hops=n_hops)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, so concurrent workers never read a partial file.
            tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, idx)
            os.replace(tmp, path)
        return idx
