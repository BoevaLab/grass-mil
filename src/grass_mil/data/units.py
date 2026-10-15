"""Cutting a region into graph units: the whole region, square tiles or polygons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from grass_mil.utils import checked

__all__ = ["TilingConfig", "grid_tiles", "polygon_members", "inside_any"]


@dataclass(frozen=True)
class TilingConfig:
    """Square tiles of ``size`` with a step of ``stride`` (coordinate units)."""

    size: float = 200.0
    stride: float = 200.0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> TilingConfig:
        cfg = checked(cls, cfg, "data.tiling")
        out = cls(**{k: float(v) for k, v in cfg.items()})
        if out.size <= 0 or out.stride <= 0:
            raise ValueError("data.tiling.size and stride must be positive.")
        return out


def grid_tiles(coords: np.ndarray, cfg: TilingConfig) -> list[np.ndarray]:
    """Cell indices of every tile of a regular grid anchored at the minimum corner."""
    if coords.size == 0:
        return []
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    tiles = []
    for x0 in np.arange(lo[0], hi[0] + cfg.stride, cfg.stride):
        for y0 in np.arange(lo[1], hi[1] + cfg.stride, cfg.stride):
            inside = (
                (coords[:, 0] >= x0)
                & (coords[:, 0] < x0 + cfg.size)
                & (coords[:, 1] >= y0)
                & (coords[:, 1] < y0 + cfg.size)
            )
            if inside.any():
                tiles.append(np.flatnonzero(inside))
    return tiles


def polygon_members(coords: np.ndarray, polygons: Sequence[Any]) -> list[np.ndarray]:
    """Cell indices inside each polygon."""
    import shapely

    return [np.flatnonzero(shapely.contains_xy(p, coords[:, 0], coords[:, 1])) for p in polygons]


def inside_any(coords: np.ndarray, polygons: Sequence[Any]) -> np.ndarray:
    """Indices of the cells inside the union of ``polygons``."""
    import shapely

    union = shapely.union_all(list(polygons))
    return np.flatnonzero(shapely.contains_xy(union, coords[:, 0], coords[:, 1]))
