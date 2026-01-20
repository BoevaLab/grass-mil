from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class TileConfig:
    tile_size_um: float
    stride_um: float
    min_cells: int


def build_patches_from_polygons(coords: np.ndarray, polygons: Sequence[object]) -> List[np.ndarray]:
    from shapely import vectorized
    from shapely.geometry import Point
    from shapely.prepared import prep

    patches: List[np.ndarray] = []
    xs = coords[:, 0]
    ys = coords[:, 1]

    for poly in polygons:
        try:
            mask = vectorized.contains(poly, xs, ys)
        except Exception:
            prepared = prep(poly)
            mask = np.array([prepared.contains(Point(x, y)) for x, y in coords])
        indices = np.where(mask)[0]
        patches.append(indices)
    return patches


def filter_coords_by_union(coords: np.ndarray, polygons: Sequence[object]) -> np.ndarray:
    from shapely.ops import unary_union
    from shapely import vectorized
    from shapely.geometry import Point
    from shapely.prepared import prep

    union = unary_union(list(polygons))
    xs = coords[:, 0]
    ys = coords[:, 1]
    try:
        mask = vectorized.contains(union, xs, ys)
    except Exception:
        prepared = prep(union)
        mask = np.array([prepared.contains(Point(x, y)) for x, y in coords])
    return np.where(mask)[0]


def build_grid_tiles(coords: np.ndarray, tile_config: TileConfig) -> List[np.ndarray]:
    if coords.size == 0:
        return []

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    tile_size = tile_config.tile_size_um
    stride = tile_config.stride_um

    patches: List[np.ndarray] = []
    x_starts = np.arange(x_min, x_max + 1e-6, stride)
    y_starts = np.arange(y_min, y_max + 1e-6, stride)

    for x0 in x_starts:
        for y0 in y_starts:
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            mask = (
                (coords[:, 0] >= x0)
                & (coords[:, 0] < x1)
                & (coords[:, 1] >= y0)
                & (coords[:, 1] < y1)
            )
            idx = np.where(mask)[0]
            if idx.size >= tile_config.min_cells:
                patches.append(idx)
    return patches
