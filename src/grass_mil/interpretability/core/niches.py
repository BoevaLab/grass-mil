"""Niche identity: naming, the background label, and composition ordering.

A *niche* is a recurring local tissue neighbourhood — a group of ego-graphs with
similar cellular composition. The clustering algorithm produces integer labels;
this module turns them into something a biologist reads.

Two conventions:

- Label ``-1`` is **Background**, not "noise". Density-based clustering assigns
  it to neighbourhoods that do not fit any dense group. Those are usually
  transitional or sparsely populated tissue, which is a meaningful category
  rather than a failure, so it is kept and named.
- Niches are ordered by a dendrogram over their cellular composition, so
  compositionally similar niches sit next to each other in every table and
  heatmap. Background is pinned last: it is a catch-all, not a niche, and
  letting it participate in the ordering would distort the tree.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "BACKGROUND_NICHE_ID",
    "BACKGROUND_NICHE_NAME",
    "is_background",
    "niche_display_name",
    "niche_display_names",
    "order_niches_by_composition",
    "relabel_niches_by_composition",
]

BACKGROUND_NICHE_ID = -1
BACKGROUND_NICHE_NAME = "Background"


def is_background(niche_id: object) -> bool:
    """True for the catch-all label produced by density-based clustering."""
    try:
        return int(niche_id) == BACKGROUND_NICHE_ID
    except (TypeError, ValueError):
        return str(niche_id) == BACKGROUND_NICHE_NAME


def niche_display_name(niche_id: object) -> str:
    """Human-readable name for one niche.

    The numeric id is retained in the name (``Niche 3``) so a reader can join a
    figure back to the ``niche_label`` column without a lookup table.
    """
    if is_background(niche_id):
        return BACKGROUND_NICHE_NAME
    return f"Niche {niche_id}"


def niche_display_names(niche_ids: Iterable[object]) -> List[str]:
    return [niche_display_name(value) for value in niche_ids]


def order_niches_by_composition(
    composition: pd.DataFrame,
    *,
    method: str = "average",
    metric: str = "correlation",
    background_last: bool = True,
) -> List:
    """Order niches so compositionally similar ones are adjacent.

    Uses hierarchical linkage over the per-niche composition profiles and
    returns the dendrogram leaf order. Correlation distance is the default
    because composition vectors are proportions: two niches with the same
    cellular makeup at different densities should read as similar.

    Args:
        composition: Niche x feature composition, indexed by niche id.
        background_last: Pin Background to the end instead of letting it join
            the tree.

    Returns:
        Niche ids in dendrogram order. Falls back to the existing order when
        there are too few niches to build a tree.
    """
    if composition is None or composition.empty:
        return []

    index = list(composition.index)
    if background_last:
        ordered_ids = [i for i in index if not is_background(i)]
        background = [i for i in index if is_background(i)]
    else:
        ordered_ids, background = list(index), []

    if len(ordered_ids) < 3:
        # Fewer than three leaves: linkage adds nothing over the given order.
        return ordered_ids + background

    values = composition.loc[ordered_ids].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.all():
        values = np.where(finite, values, 0.0)

    # scipy is a hard dependency; import outside the guard so a genuinely
    # missing install raises instead of silently degrading to input order.
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    distances = pdist(values, metric=metric)
    if not np.isfinite(distances).all():
        # Correlation distance is undefined for a constant profile.
        distances = pdist(values, metric="euclidean")
    try:
        order = leaves_list(linkage(distances, method=method))
    except (ValueError, RuntimeError):
        return ordered_ids + background

    return [ordered_ids[int(position)] for position in order] + background


def relabel_niches_by_composition(
    table: pd.DataFrame,
    labels: np.ndarray,
    *,
    composition_prefix: str = "comp_",
    method: str = "average",
    metric: str = "correlation",
    start_at: int = 1,
) -> tuple[np.ndarray, dict]:
    """Renumber niches so that the id *is* the dendrogram position.

    Called once, immediately after clustering. Niche 1 is first in dendrogram
    order, Niche 2 second, and so on, so every downstream table, figure and
    export can simply sort by id and get composition order for free -- no
    ordering state has to be threaded through the pipeline. Background keeps its
    own label and takes no position in the tree.

    Returns:
        ``(relabelled, mapping)`` where ``mapping`` is old id -> new id.
    """
    labels = np.asarray(labels)
    composition_columns = [c for c in table.columns if str(c).startswith(composition_prefix)]
    if not composition_columns:
        # Nothing to order by; leave the labels untouched rather than guessing.
        return labels, {value: value for value in np.unique(labels)}

    composition = (
        table.loc[:, composition_columns]
        .astype(float)
        .groupby(pd.Series(labels, name="niche_label"))
        .mean()
        .sort_index()
    )
    order = order_niches_by_composition(composition, method=method, metric=metric)

    mapping = {}
    next_id = int(start_at)
    for niche_id in order:
        if is_background(niche_id):
            mapping[niche_id] = BACKGROUND_NICHE_ID
            continue
        mapping[niche_id] = next_id
        next_id += 1

    relabelled = np.array([mapping.get(value, value) for value in labels])
    return relabelled, mapping
