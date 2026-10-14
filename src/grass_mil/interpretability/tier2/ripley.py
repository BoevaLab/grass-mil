"""Ripley's K/L and the bivariate cross-K.

For a pattern of intensity :math:`\\lambda` on a region of area :math:`|A|`,

.. math:: K(r) = \\frac{1}{\\lambda^2 |A|} \\sum_i \\sum_{j \\neq i}
                 \\mathbf{1}[d_{ij} \\le r], \\qquad L(r) = \\sqrt{K(r)/\\pi}

reported in the centred form :math:`L(r) - r`, which is zero under complete
spatial randomness, positive under clustering and negative under regularity.
The bivariate form replaces the pair count with cross-type pairs.

Ported from ``_cross_L_for_region`` / ``_aggregate_cross_L``, keeping the
self-pair correction (a point is not its own neighbour, so the count loses
``n_a`` and the denominator is ``n_a(n_a - 1)``) and the region-equal
aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["RipleyResult", "resolve_radii", "ripley_cross_l", "aggregate_ripley"]

PairKey = Tuple[str, str]


@dataclass(frozen=True)
class RipleyResult:
    """Centred cross-L curves per label pair.

    Attributes:
        radii: Radii the curves are evaluated at.
        curves: ``(label_a, label_b) -> L(r) - r``.
        pair_counts: ``(label_a, label_b) -> (n_a, n_b)``.
        n_groups: Number of groups contributing to each pair.
    """

    radii: np.ndarray
    curves: Dict[PairKey, np.ndarray]
    pair_counts: Dict[PairKey, Tuple[int, int]] = field(default_factory=dict)
    n_groups: Dict[PairKey, int] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Curves as a radius-indexed frame, one column per pair."""
        return pd.DataFrame(
            {f"{a}__{b}": curve for (a, b), curve in self.curves.items()},
            index=pd.Index(self.radii, name="radius"),
        )


def resolve_radii(
    coords: np.ndarray,
    *,
    n_radii: int = 50,
    max_fraction: float = 0.25,
) -> np.ndarray:
    """Radius grid up to ``max_fraction`` of the bounding-box diagonal."""
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return np.linspace(0.0, 1.0, n_radii)
    spans = coords.max(axis=0) - coords.min(axis=0)
    diagonal = float(np.sqrt(float(spans @ spans)))
    limit = max(diagonal * float(max_fraction), 1e-9)
    return np.linspace(limit / n_radii, limit, n_radii)


def ripley_cross_l(
    coords: np.ndarray,
    labels: Sequence[str],
    *,
    pairs: Iterable[PairKey],
    radii: np.ndarray,
) -> RipleyResult:
    """Centred cross-L for each label pair within one group."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=float)
    labels = np.asarray([str(v) for v in labels])
    radii = np.asarray(radii, dtype=float)

    spans = coords.max(axis=0) - coords.min(axis=0) if coords.size else np.zeros(2)
    area = max(float(np.prod(spans)), 1e-10)

    index_map: Dict[str, np.ndarray] = {}
    tree_map = {}
    for label in set(labels.tolist()):
        idx = np.where(labels == label)[0]
        if idx.size >= 2:
            index_map[label] = idx
            tree_map[label] = cKDTree(coords[idx])

    curves: Dict[PairKey, np.ndarray] = {}
    counts: Dict[PairKey, Tuple[int, int]] = {}
    for a, b in pairs:
        a, b = str(a), str(b)
        if a not in tree_map or b not in tree_map:
            curves[(a, b)] = np.full(radii.shape, np.nan)
            counts[(a, b)] = (0, 0)
            continue
        n_a, n_b = int(index_map[a].size), int(index_map[b].size)
        pair_counts = tree_map[a].count_neighbors(tree_map[b], radii).astype(float)
        if a == b:
            # A point is not its own neighbour.
            pair_counts = pair_counts - n_a
            denom = n_a * (n_a - 1)
        else:
            denom = n_a * n_b
        k = (area / max(denom, 1)) * pair_counts
        curves[(a, b)] = np.sqrt(np.maximum(k, 0.0) / np.pi) - radii
        counts[(a, b)] = (n_a, n_b)
    return RipleyResult(radii=radii, curves=curves, pair_counts=counts)


def aggregate_ripley(
    node_table: pd.DataFrame,
    *,
    label_column: str,
    group_column: Optional[str] = None,
    x_column: str = "center_x",
    y_column: str = "center_y",
    pairs: Optional[Iterable[PairKey]] = None,
    n_radii: int = 50,
    max_fraction: float = 0.25,
    min_count: int = 5,
    radius_source: str = "median",
    radii: Optional[np.ndarray] = None,
) -> RipleyResult:
    """Region-equal mean of the centred cross-L curves.

    Args:
        radius_source: How to fix the shared radius grid across groups --
            ``median`` (of per-group extents), ``max``, or ``first``. The
            legacy implementation used the first group encountered, which made
            the grid depend on iteration order; ``median`` is the default here.
        min_count: Groups with fewer than this many points of a label
            contribute NaN for that pair rather than a noisy curve.
    """
    for column in (label_column, x_column, y_column):
        if column not in node_table.columns:
            raise ValueError(f"node_table is missing required column {column!r}.")

    if group_column is None:
        groups: List[Tuple[object, pd.DataFrame]] = [("__all__", node_table)]
    else:
        if group_column not in node_table.columns:
            raise ValueError(f"node_table is missing group column {group_column!r}.")
        groups = list(node_table.groupby(group_column))

    if pairs is None:
        labels = sorted({str(v) for v in node_table[label_column]})
        pairs = [(a, b) for a in labels for b in labels]
    pairs = [(str(a), str(b)) for a, b in pairs]

    if radii is None:
        radii = _shared_radii(
            groups,
            x_column=x_column,
            y_column=y_column,
            n_radii=n_radii,
            max_fraction=max_fraction,
            radius_source=radius_source,
        )
    radii = np.asarray(radii, dtype=float)

    stacked: Dict[PairKey, List[np.ndarray]] = {pair: [] for pair in pairs}
    totals: Dict[PairKey, Tuple[int, int]] = {pair: (0, 0) for pair in pairs}
    for _, frame in groups:
        coords = frame[[x_column, y_column]].to_numpy(dtype=float)
        if coords.shape[0] < 2:
            continue
        result = ripley_cross_l(coords, frame[label_column].tolist(), pairs=pairs, radii=radii)
        for pair, curve in result.curves.items():
            n_a, n_b = result.pair_counts[pair]
            if n_a < min_count or n_b < min_count or not np.isfinite(curve).any():
                continue
            stacked[pair].append(curve)
            totals[pair] = (totals[pair][0] + n_a, totals[pair][1] + n_b)

    curves: Dict[PairKey, np.ndarray] = {}
    n_groups: Dict[PairKey, int] = {}
    for pair, values in stacked.items():
        n_groups[pair] = len(values)
        if not values:
            curves[pair] = np.full(radii.shape, np.nan)
            continue
        # Region-equal: every group counts once regardless of its size.
        curves[pair] = np.nanmean(np.vstack(values), axis=0)
    return RipleyResult(radii=radii, curves=curves, pair_counts=totals, n_groups=n_groups)


def _shared_radii(
    groups: Sequence[Tuple[object, pd.DataFrame]],
    *,
    x_column: str,
    y_column: str,
    n_radii: int,
    max_fraction: float,
    radius_source: str,
) -> np.ndarray:
    extents: List[float] = []
    for _, frame in groups:
        coords = frame[[x_column, y_column]].to_numpy(dtype=float)
        if coords.shape[0] < 2:
            continue
        spans = coords.max(axis=0) - coords.min(axis=0)
        extents.append(float(np.sqrt(float(spans @ spans))))
    if not extents:
        return np.linspace(0.0, 1.0, n_radii)

    source = str(radius_source).strip().lower()
    if source == "median":
        diagonal = float(np.median(extents))
    elif source == "max":
        diagonal = float(np.max(extents))
    elif source == "first":
        diagonal = float(extents[0])
    else:
        raise ValueError(
            f"Unsupported radius_source '{radius_source}'. "
            "Expected one of: 'median', 'max', 'first'."
        )
    limit = max(diagonal * float(max_fraction), 1e-9)
    return np.linspace(limit / n_radii, limit, n_radii)
