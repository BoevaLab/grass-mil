"""Cell-level tier-2 analyses, run within each niche.

The instance-level analyses treat one ego-graph as one node, so they measure
relationships between neighbourhood summaries. These measure the cells: every
node of every sampled ego-graph is a row, carrying its own type, and each cell
inherits the niche of the subgraph it came from. Within a niche you then get
cell-type x cell-type statistics.

That is the unit the legacy NSCLC reports use: `run_per_cluster_nhood_enrichment`
pools the member ego-graphs and runs cell-type enrichment per cluster,
`run_moran_per_cluster` computes Moran's I per cell type over the same pooled
graph, and `run_filtration_curves` counts both endpoints of every edge by type.

The numerics are the existing tier-2 primitives -- these functions only assemble
the right inputs, because "cell-level" is a statement about the table, not about
the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from grass_mil.interpretability.tier2.autocorrelation import MoranResult, run_morans_i
from grass_mil.interpretability.tier2.neighborhood import (
    run_neighborhood_enrichment,
)

BACKGROUND_NICHE_ID = -1


@dataclass(frozen=True)
class PerNicheEnrichmentResult:
    """Cell-type x cell-type enrichment, one matrix per niche."""

    enrichment: Dict[Any, pd.DataFrame]
    pvalues: Dict[Any, pd.DataFrame]
    n_cells: Dict[Any, int]
    n_edges: Dict[Any, int]


def attach_niche_labels_to_cells(
    cell_table: pd.DataFrame,
    niche_labels: Sequence[Any],
    instance_ids: Sequence[Any],
    *,
    instance_id_column: str = "instance_id",
    niche_column: str = "niche_label",
) -> pd.DataFrame:
    """Give every cell the niche of the ego-graph it was sampled into.

    Niches are assigned to instances, and a cell belongs to whichever ego-graph
    it appeared in, so the label is joined through `instance_id`. This mirrors
    the legacy `cluster_labels[batch.batch]` step.
    """
    if instance_id_column not in cell_table.columns:
        raise ValueError(f"Cell table is missing {instance_id_column!r}.")
    if len(niche_labels) != len(instance_ids):
        raise ValueError(
            "niche_labels and instance_ids must align: "
            f"{len(niche_labels)} vs {len(instance_ids)}."
        )
    mapping = pd.DataFrame(
        {instance_id_column: list(instance_ids), niche_column: list(niche_labels)}
    ).drop_duplicates(subset=[instance_id_column])
    merged = cell_table.merge(mapping, on=instance_id_column, how="left")
    return merged


def add_cell_type_indicators(
    cell_table: pd.DataFrame,
    *,
    cell_type_column: str = "cell_type",
    prefix: str = "ct_",
    categories: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, List[str]]:
    """One-hot the cell type, so each type can be treated as a signal.

    Moran's I asks whether a value is spatially clustered; for a categorical
    type that question is asked per type, over its indicator.
    """
    if cell_type_column not in cell_table.columns:
        raise ValueError(f"Cell table is missing {cell_type_column!r}.")
    values = cell_table[cell_type_column].astype(str)
    cats = list(categories) if categories is not None else sorted(values.unique())
    out = cell_table.copy()
    columns: List[str] = []
    for cat in cats:
        col = f"{prefix}{cat}"
        out[col] = (values == cat).astype(float)
        columns.append(col)
    return out, columns


def _subset_edges(edges: pd.DataFrame, ids: set, *, source: str, target: str) -> pd.DataFrame:
    return edges[edges[source].isin(ids) & edges[target].isin(ids)]


def per_niche_cell_type_enrichment(
    cell_table: pd.DataFrame,
    cell_edge_table: pd.DataFrame,
    *,
    cell_type_column: str = "cell_type",
    niche_column: str = "niche_label",
    cell_id_column: str = "cell_id",
    source_column: str = "source_id",
    target_column: str = "target_id",
    n_perms: int = 0,
    random_state: int = 42,
    undirected: bool = False,
    enrichment_mode: str = "zscore",
    min_cells: int = 10,
    skip_background: bool = False,
) -> PerNicheEnrichmentResult:
    """Cell-type neighbourhood enrichment computed separately within each niche.

    Only edges whose both endpoints lie in the niche are used, so each matrix
    describes the internal wiring of that niche rather than its border.
    """
    for col in (cell_type_column, niche_column, cell_id_column):
        if col not in cell_table.columns:
            raise ValueError(f"Cell table is missing {col!r}.")

    # A shared category list keeps the matrices aligned across niches, which is
    # what makes them comparable (and differenceable) afterwards.
    categories = sorted(cell_table[cell_type_column].astype(str).unique())

    enrichment: Dict[Any, pd.DataFrame] = {}
    pvalues: Dict[Any, pd.DataFrame] = {}
    n_cells: Dict[Any, int] = {}
    n_edges: Dict[Any, int] = {}

    for niche, group in cell_table.groupby(niche_column, dropna=True):
        if skip_background and int(niche) == BACKGROUND_NICHE_ID:
            continue
        if len(group) < int(min_cells):
            continue
        ids = set(group[cell_id_column])
        sub_edges = _subset_edges(cell_edge_table, ids, source=source_column, target=target_column)
        if sub_edges.empty:
            continue
        result = run_neighborhood_enrichment(
            group,
            sub_edges,
            label_column=cell_type_column,
            id_column=cell_id_column,
            n_perms=int(n_perms),
            random_state=int(random_state),
            undirected=bool(undirected),
            categories=categories,
            warn_analytical=False,
            enrichment_mode=str(enrichment_mode),
        )
        enrichment[niche] = result.enrichment
        pvalues[niche] = result.pvalues
        n_cells[niche] = int(len(group))
        n_edges[niche] = int(len(sub_edges))

    return PerNicheEnrichmentResult(
        enrichment=enrichment, pvalues=pvalues, n_cells=n_cells, n_edges=n_edges
    )


def differential_cell_type_enrichment(
    result: PerNicheEnrichmentResult,
    *,
    reference: Any = BACKGROUND_NICHE_ID,
) -> Dict[Any, pd.DataFrame]:
    """Each niche's enrichment minus a reference niche's.

    The legacy reports difference every cluster against the noise cluster, which
    separates structure specific to a niche from structure present everywhere.
    """
    if reference not in result.enrichment:
        return {}
    ref = result.enrichment[reference]
    return {
        niche: (matrix - ref) for niche, matrix in result.enrichment.items() if niche != reference
    }


def per_niche_cell_type_moran(
    cell_table: pd.DataFrame,
    cell_edge_table: pd.DataFrame,
    *,
    cell_type_column: str = "cell_type",
    niche_column: str = "niche_label",
    cell_id_column: str = "cell_id",
    n_perms: int = 100,
    random_state: int = 0,
    min_nodes: int = 3,
) -> MoranResult:
    """Moran's I of each cell-type indicator, per niche.

    Yields a niche x cell-type table: whether cells of a given type are
    spatially clustered within that niche's graph.
    """
    table, indicator_columns = add_cell_type_indicators(
        cell_table, cell_type_column=cell_type_column
    )
    return run_morans_i(
        table,
        cell_edge_table,
        feature_columns=indicator_columns,
        group_column=niche_column,
        id_column=cell_id_column,
        n_perms=int(n_perms),
        random_state=int(random_state),
        min_nodes=int(min_nodes),
    )


def niche_label_moran(
    instance_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    niche_column: str = "niche_label",
    id_column: str = "instance_id",
    n_perms: int = 100,
    random_state: int = 0,
    group_column: Optional[str] = None,
) -> MoranResult:
    """Moran's I of the niche assignment itself, over the instance graph.

    The global counterpart to the per-niche cell-type analyses: it asks whether
    niches form contiguous territories rather than being interleaved. The
    legacy equivalent is `run_cluster_self_moran`.
    """
    table = instance_table.copy()
    if group_column is None:
        group_column = "_all"
        table[group_column] = "all"
    table, indicator_columns = add_cell_type_indicators(
        table, cell_type_column=niche_column, prefix="niche_"
    )
    return run_morans_i(
        table,
        spatial_table,
        feature_columns=indicator_columns,
        group_column=group_column,
        id_column=id_column,
        n_perms=int(n_perms),
        random_state=int(random_state),
    )


def cell_filtration_curves(
    cell_table: pd.DataFrame,
    cell_edge_table: pd.DataFrame,
    *,
    thresholds: np.ndarray,
    cell_type_column: str = "cell_type",
    niche_column: str = "niche_label",
    cell_id_column: str = "cell_id",
    distance_column: str = "distance",
    scale_within_niche: bool = True,
):
    """Filtration curves over cell-cell edges, counting both endpoints.

    The instance-level version walks edges between ego-graphs and has one label
    per node, so its curves describe neighbourhoods. This walks the edges
    *inside* the ego-graphs, where each endpoint is a cell with its own type --
    the legacy `run_filtration_curves` behaviour.
    """
    from grass_mil.interpretability.tier2.filtration import compute_filtration_curves

    return compute_filtration_curves(
        cell_table,
        cell_edge_table,
        thresholds=np.asarray(thresholds, dtype=float),
        niche_column=niche_column,
        cell_type_column=cell_type_column,
        id_column=cell_id_column,
        distance_column=distance_column,
        scale_within_cluster=bool(scale_within_niche),
    )


def per_niche_cell_type_ripley(
    cell_table: pd.DataFrame,
    *,
    cell_type_column: str = "cell_type",
    niche_column: str = "niche_label",
    x_column: str = "x",
    y_column: str = "y",
    n_radii: int = 50,
    max_fraction: float = 0.25,
    min_count: int = 5,
    radius_source: str = "median",
):
    """Ripley cross-L between cell types, with each niche a region.

    Uses the cells' own coordinates rather than ego-graph centroids, so the
    curves describe how cell types are arranged relative to one another.
    """
    from grass_mil.interpretability.tier2.ripley import aggregate_ripley

    return aggregate_ripley(
        cell_table,
        label_column=cell_type_column,
        group_column=niche_column,
        x_column=x_column,
        y_column=y_column,
        n_radii=int(n_radii),
        max_fraction=float(max_fraction),
        min_count=int(min_count),
        radius_source=str(radius_source),
    )
