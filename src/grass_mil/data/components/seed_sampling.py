"""Interior-only seed sampling for ego-graph extraction.

Ego-graphs rooted on cells at the edge of a tissue section are truncated: their
neighbourhood is cut off by the section boundary rather than by biology, so
they are not comparable to interior ego-graphs. Restricting seeds to the
interior removes that artefact.

"Interior" here excludes only the ``n_hops`` graph-hop rim around the **convex
hull** of the cell coordinates -- the outer tissue perimeter. Internal holes and
low-cellularity gaps are deliberately *not* treated as boundary.

Ported from ``boundary_sampling.convex_hull_interior_mask`` in the NSCLC tuning
study, keeping the mask definition identical so that models trained against the
legacy definition remain comparable. The graph traversal is expressed with
sparse matrix operations instead of a Python BFS; the result is the same.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "InteriorSeedConfig",
    "InteriorSeedIndexCache",
    "convex_hull_interior_mask",
    "distance_to_hull_polygon",
    "compute_interior_seed_indices",
]


@dataclass(frozen=True)
class InteriorSeedConfig:
    """Configuration for interior seed restriction.

    Attributes:
        enabled: Restrict seeds to the interior mask.
        n_hops: Width of the excluded rim, in graph hops. ``None`` uses the
            sampler depth, so the rim matches the ego-graph radius.
        seed_tolerance_multiplier: Cells within this multiple of the median
            nearest-neighbour distance of the hull polygon start the rim.
        keep_largest_component: Discard interior islands disconnected from the
            main interior body.
        cache_dir: Directory for the on-disk index cache. ``None`` disables it.
    """

    enabled: bool = False
    n_hops: Optional[int] = None
    seed_tolerance_multiplier: float = 1.0
    keep_largest_component: bool = True
    cache_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "InteriorSeedConfig":
        cfg = dict(cfg or {})
        allowed = {
            "enabled",
            "n_hops",
            "seed_tolerance_multiplier",
            "keep_largest_component",
            "cache_dir",
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown interior seed keys: {sorted(unknown)}. "
                f"Expected a subset of {sorted(allowed)}."
            )
        n_hops = cfg.get("n_hops")
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            n_hops=None if n_hops is None else int(n_hops),
            seed_tolerance_multiplier=float(cfg.get("seed_tolerance_multiplier", 1.0)),
            keep_largest_component=bool(cfg.get("keep_largest_component", True)),
            cache_dir=cfg.get("cache_dir"),
        )


def distance_to_hull_polygon(coords: np.ndarray, hull_vertices: np.ndarray) -> np.ndarray:
    """Distance from every point to the convex-hull polygon's edge segments."""
    coords = np.asarray(coords, dtype=float)
    hull_vertices = np.asarray(hull_vertices)
    starts = coords[hull_vertices]
    ends = coords[np.roll(hull_vertices, -1)]

    distances = np.full(coords.shape[0], np.inf)
    for start, end in zip(starts, ends):
        segment = end - start
        denom = float(segment @ segment) + 1e-12
        t = np.clip(((coords - start) @ segment) / denom, 0.0, 1.0)
        projection = start + np.outer(t, segment)
        distances = np.minimum(distances, np.linalg.norm(coords - projection, axis=1))
    return distances


def _adjacency(edge_index: np.ndarray, num_nodes: int):
    from scipy.sparse import coo_matrix

    edge_index = np.asarray(edge_index)
    if edge_index.size == 0:
        return coo_matrix((num_nodes, num_nodes), dtype=bool).tocsr()
    rows = np.concatenate([edge_index[0], edge_index[1]]).astype(np.int64)
    cols = np.concatenate([edge_index[1], edge_index[0]]).astype(np.int64)
    data = np.ones(rows.shape[0], dtype=bool)
    return coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def convex_hull_interior_mask(
    coords: np.ndarray,
    edge_index: np.ndarray,
    n_hops: int,
    *,
    seed_tolerance_multiplier: float = 1.0,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Boolean per-cell mask; ``True`` marks interior cells.

    Degenerate inputs (fewer than 4 points, collinear point sets, ``n_hops <=
    0``) fall back to marking every cell interior, so callers always get a
    usable seed pool.
    """
    coords = np.asarray(coords, dtype=float)
    num_nodes = int(coords.shape[0])
    if n_hops <= 0 or num_nodes < 4:
        return np.ones(num_nodes, dtype=bool)

    try:
        from scipy.spatial import ConvexHull, cKDTree
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return np.ones(num_nodes, dtype=bool)

    try:
        hull = np.asarray(ConvexHull(coords).vertices)
    except Exception:
        # Degenerate or collinear point set: no meaningful hull.
        return np.ones(num_nodes, dtype=bool)

    nearest = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    edge_distance = distance_to_hull_polygon(coords, hull)
    tolerance = seed_tolerance_multiplier * max(nearest, 1e-6)

    rim = edge_distance <= tolerance
    if not rim.any():
        rim = np.zeros(num_nodes, dtype=bool)
        rim[np.unique(hull)] = True

    adjacency = _adjacency(edge_index, num_nodes)
    # Grey out everything within (n_hops - 1) further hops of the hull rim,
    # i.e. graph distance 0..n_hops-1 from a rim cell.
    reached = rim.copy()
    frontier = rim.copy()
    for _ in range(max(int(n_hops) - 1, 0)):
        if not frontier.any():
            break
        neighbours = adjacency.dot(frontier.astype(np.int8)).astype(bool)
        frontier = neighbours & ~reached
        reached |= frontier

    interior = ~reached
    if keep_largest_component and interior.any():
        interior = _largest_component(adjacency, interior)
    return interior


def _largest_component(adjacency, interior: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of the interior subgraph."""
    from scipy.sparse.csgraph import connected_components

    indices = np.where(interior)[0]
    if indices.size == 0:
        return interior
    sub = adjacency[indices][:, indices]
    n_components, labels = connected_components(sub, directed=False)
    if n_components <= 1:
        return interior
    counts = np.bincount(labels)
    keep = labels == int(np.argmax(counts))
    mask = np.zeros_like(interior)
    mask[indices[keep]] = True
    return mask


def compute_interior_seed_indices(
    data,
    cfg: InteriorSeedConfig,
    *,
    n_hops: int,
) -> np.ndarray:
    """Interior node indices for one graph unit.

    Falls back to every node when coordinates are unavailable or the mask is
    empty, so seeding never silently yields an empty pool.
    """
    pos = getattr(data, "pos", None)
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if pos is None:
        return np.arange(num_nodes, dtype=np.int64)

    coords = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
    coords = np.asarray(coords, dtype=float)[:, :2]
    edge_index = getattr(data, "edge_index", None)
    edge_index = (
        edge_index.detach().cpu().numpy()
        if hasattr(edge_index, "detach")
        else np.asarray(edge_index if edge_index is not None else np.empty((2, 0)))
    )

    hops = int(cfg.n_hops) if cfg.n_hops is not None else int(n_hops)
    mask = convex_hull_interior_mask(
        coords,
        edge_index,
        hops,
        seed_tolerance_multiplier=cfg.seed_tolerance_multiplier,
        keep_largest_component=cfg.keep_largest_component,
    )
    indices = np.where(mask)[0].astype(np.int64)
    if indices.size == 0:
        return np.arange(coords.shape[0], dtype=np.int64)
    return indices


class InteriorSeedIndexCache:
    """On-disk cache of interior indices, keyed by graph identity and rim width.

    The hull mask is deterministic given (coordinates, edges, n_hops), and
    recomputing it per epoch is wasteful for large regions.
    """

    def __init__(self, cache_dir: Optional[str]) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in key)
        return self.cache_dir / f"{safe}.npy"

    def load(self, key: str) -> Optional[np.ndarray]:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:
            # A truncated or corrupt cache entry must not be fatal; recompute.
            return None

    def store(self, key: str, indices: np.ndarray) -> None:
        path = self._path(key)
        if path is None:
            return
        # Write to a process-unique temporary file and rename, so concurrent
        # dataloader workers cannot observe a half-written array.
        tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
        try:
            np.save(tmp, indices)
            os.replace(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def cache_key_for_unit(data, *, n_hops: int) -> str:
    """Stable identity for a graph unit's interior mask."""
    parts = [
        str(getattr(data, "sample_id", "") or ""),
        str(getattr(data, "region_id", "") or ""),
        str(getattr(data, "patch_id", "") or ""),
        f"hops{int(n_hops)}",
    ]
    return "__".join(part for part in parts if part)
