from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grass_mil.contracts import (
    NicheSummary,
    PluginResult,
    ReportSection,
)
from grass_mil.interpretability.core.agreement import compute_niche_agreement
from grass_mil.interpretability.core.attribution import niche_attribution_summary
from grass_mil.interpretability.core.biomarkers import niche_composition_summary
from grass_mil.interpretability.plugins.base import InterpretabilityPlugin, PluginContext
from grass_mil.interpretability.plugins.registry import PluginRegistry, create_plugin_registry
from grass_mil.interpretability.tier2.cell_level import (
    attach_niche_labels_to_cells,
    differential_cell_type_enrichment,
    cell_filtration_curves,
    niche_label_moran,
    per_niche_cell_type_enrichment,
    per_niche_cell_type_moran,
    per_niche_cell_type_ripley,
)
from grass_mil.interpretability.tier2.autocorrelation import (
    diff_morans_i_vs_reference,
    run_morans_i,
)
from grass_mil.interpretability.tier2.filtration import compute_filtration_curves
from grass_mil.interpretability.tier2.neighborhood import (
    run_diff_neighborhood_enrichment,
    run_neighborhood_enrichment,
)
from grass_mil.interpretability.tier2.ripley import aggregate_ripley
from grass_mil.interpretability.tier2.tissue_graph import prepare_tissue_graph_view


def _resolve_filtration_thresholds(params: Dict[str, Any]) -> np.ndarray:
    explicit = params.get("thresholds")
    if explicit is not None:
        return np.asarray(explicit, dtype=float)

    # 500 bins from 0 to 55 micrometers.
    start = float(params.get("threshold_start", 0.0))
    stop = float(params.get("threshold_stop", 55.0))
    count = int(params.get("threshold_count", 500))
    if count < 2:
        raise ValueError("filtration threshold_count must be >= 2.")
    if stop < start:
        raise ValueError("filtration threshold_stop must be >= threshold_start.")
    return np.linspace(start, stop, count, dtype=float)


def _context_cluster_summary(
    context: PluginContext,
    *,
    labels: np.ndarray,
) -> NicheSummary | None:
    summary = context.state.get("niche_summary")
    if not isinstance(summary, NicheSummary):
        return None
    if summary.niche_labels.shape != labels.shape:
        return None
    if not np.array_equal(summary.niche_labels, labels):
        return None
    return summary


@dataclass
class NicheProfilesPlugin(InterpretabilityPlugin):
    name: str = "niche_profiles"

    def required_inputs(self) -> List[str]:
        return ["niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        labels = np.asarray(context.state["niche_labels"])
        summary = _context_cluster_summary(context, labels=labels)
        if summary is None:
            summary = niche_composition_summary(
                dataset.instance_table,
                labels,
                composition_prefix=str(params.get("composition_prefix", "comp_")),
                variance_estimator=str(params.get("variance_estimator", "unbiased")),
            )
        payload = {
            "composition": summary.composition,
            "enrichment": summary.enrichment,
            "counts": summary.niche_counts,
        }
        sections = [
            ReportSection(
                title="Niche Profiles",
                description="Niche-level composition and enrichment summary.",
                tables={
                    "composition": summary.composition,
                    "enrichment": summary.enrichment,
                },
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class NeighborhoodEnrichmentPlugin(InterpretabilityPlugin):
    name: str = "neighborhood_enrichment"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for neighborhood enrichment plugin.")
        table = dataset.instance_table.copy()
        table["niche_label"] = np.asarray(context.state["niche_labels"])
        result = run_neighborhood_enrichment(
            table,
            dataset.spatial_table,
            label_column=str(params.get("label_column", "niche_label")),
            id_column=str(params.get("id_column", dataset.id_column)),
            n_perms=int(params.get("n_perms", 0)),
            random_state=int(params.get("random_state", 42)),
            undirected=bool(params.get("undirected", False)),
            enrichment_mode=str(params.get("enrichment_mode", "zscore")),
        )
        payload = {
            "enrichment": result.enrichment,
            "observed": result.observed,
            "expected": result.expected,
            "pvalues": result.pvalues,
        }
        section_tables = {
            "enrichment": result.enrichment,
            "observed": result.observed,
            "expected": result.expected,
        }
        if result.pvalues is not None:
            section_tables["pvalues"] = result.pvalues
        sections = [
            ReportSection(
                title="Neighborhood Enrichment",
                description="Niche adjacency enrichment relative to expected connectivity.",
                tables=section_tables,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class DiffNeighborhoodEnrichmentPlugin(InterpretabilityPlugin):
    name: str = "diff_neighborhood_enrichment"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for differential neighborhood enrichment.")
        table = dataset.instance_table.copy()
        table["niche_label"] = np.asarray(context.state["niche_labels"])
        condition_column = str(
            params.get("condition_column", dataset.condition_column or "condition")
        )
        pairwise = run_diff_neighborhood_enrichment(
            table,
            dataset.spatial_table,
            label_column=str(params.get("label_column", "niche_label")),
            condition_column=condition_column,
            permutation_group_column=str(params.get("permutation_group_column", "sample_id")),
            id_column=str(params.get("id_column", dataset.id_column)),
            n_perms=int(params.get("n_perms", 0)),
            random_state=int(params.get("random_state", 42)),
            undirected=bool(params.get("undirected", False)),
            enrichment_mode=str(params.get("enrichment_mode", "zscore")),
        )
        payload = {
            "enrichment_by_pair": {pair: out.enrichment for pair, out in pairwise.items()},
            "observed_by_pair": {pair: out.observed for pair, out in pairwise.items()},
            "expected_by_pair": {pair: out.expected for pair, out in pairwise.items()},
            "pvalues_by_pair": {
                pair: out.pvalues for pair, out in pairwise.items() if out.pvalues is not None
            },
        }
        sections: List[ReportSection] = []
        for pair, out in pairwise.items():
            section_tables = {
                "enrichment": out.enrichment,
                "observed": out.observed,
                "expected": out.expected,
            }
            if out.pvalues is not None:
                section_tables["pvalues"] = out.pvalues
            sections.append(
                ReportSection(
                    title=f"Differential Neighborhood Enrichment ({pair})",
                    description=(
                        "Pairwise condition differential neighborhood enrichment "
                        "(left condition minus right condition)."
                    ),
                    tables=section_tables,
                )
            )
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class FiltrationCurvesPlugin(InterpretabilityPlugin):
    name: str = "filtration_curves"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for filtration curves plugin.")
        table = dataset.instance_table.copy()
        table["niche_label"] = np.asarray(context.state["niche_labels"])
        thresholds = _resolve_filtration_thresholds(dict(params))
        result = compute_filtration_curves(
            table,
            dataset.spatial_table,
            thresholds=thresholds,
            niche_column=str(params.get("niche_column", "niche_label")),
            cell_type_column=str(params.get("cell_type_column", "cell_type")),
            id_column=str(params.get("id_column", dataset.id_column)),
            distance_column=str(params.get("distance_column", "distance")),
            scale_within_cluster=bool(params.get("scale_within_cluster", True)),
        )
        payload: Dict[str, Any] = {"thresholds": result.thresholds, "curves": result.curves}
        sections = [
            ReportSection(
                title="Filtration Curves",
                description="Distance-threshold accumulation by niche and cell type.",
                metadata={"thresholds": result.thresholds.tolist()},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class TissueGraphPlugin(InterpretabilityPlugin):
    name: str = "tissue_graph"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for tissue graph plugin.")
        sample_value = params.get("sample_value")
        if sample_value is None or str(sample_value).strip() == "":
            raise ValueError("tissue_graph plugin requires a non-empty sample_value parameter.")

        table = dataset.instance_table.copy()
        table["niche_label"] = np.asarray(context.state["niche_labels"])
        sample_column = str(params.get("sample_column", "sample_id"))
        id_column = str(params.get("id_column", dataset.id_column))
        x_column = str(params.get("x_column", "center_x"))
        y_column = str(params.get("y_column", "center_y"))
        label_column = str(params.get("label_column", "niche_label"))

        view = prepare_tissue_graph_view(
            table,
            dataset.spatial_table,
            sample_column=sample_column,
            sample_value=sample_value,
            id_column=id_column,
            x_column=x_column,
            y_column=y_column,
            label_column=label_column,
            coerce_ids_to_str=True,
            include_edge_distances=True,
        )
        meta = dict(view.metadata)
        niche_counts = meta.get("niche_counts", {})
        section_summary = pd.DataFrame(
            [
                {
                    "sample_column": meta.get("sample_column"),
                    "sample_value": meta.get("sample_value"),
                    "node_count": meta.get("node_count"),
                    "edge_count": meta.get("edge_count"),
                    "niche_count": len(niche_counts) if isinstance(niche_counts, dict) else 0,
                }
            ]
        )
        style = {
            "show_edges": bool(params.get("show_edges", True)),
            "node_size": float(params.get("node_size", 5.0)),
            "edge_width": float(params.get("edge_width", 0.5)),
            "edge_opacity": float(params.get("edge_opacity", 0.25)),
            "colorscale": str(params.get("colorscale", "Viridis")),
            "reverse_y": bool(params.get("reverse_y", True)),
            "title_prefix": str(params.get("title_prefix", "Tissue Graph")),
        }
        payload = {
            "tissue_graph_view": view,
            "tissue_graph_meta": meta,
            "tissue_graph_style": style,
        }
        sections = [
            ReportSection(
                title=f"Tissue Graph ({sample_column}={sample_value})",
                description="Single tissue/sample graph with niche-colored cells.",
                tables={"summary": section_summary},
                metadata=meta,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class PerNicheCellTypeEnrichmentPlugin(InterpretabilityPlugin):
    """Compute cell-type neighborhood enrichment within each niche separately.

    For each niche, subsets the node table and spatial table to nodes belonging
    to that niche, then runs neighborhood enrichment with label_column=cell_type.
    This answers: "within niche C, which cell types are spatially co-located?"
    """

    name: str = "per_niche_cell_type_enrichment"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for per-niche cell-type enrichment.")
        cell_type_column = str(
            params.get("cell_type_column", dataset.cell_type_column or "cell_type")
        )
        niche_column = str(params.get("niche_column", "niche_label"))
        id_column = str(params.get("id_column", dataset.id_column))

        table = dataset.instance_table.copy()
        table[niche_column] = np.asarray(context.state["niche_labels"])

        unique_niches = sorted(set(table[niche_column]))
        skip_background = bool(params.get("skip_background", True))

        enrichment_by_niche: Dict[str, pd.DataFrame] = {}
        sections: List[ReportSection] = []

        for niche_id in unique_niches:
            if skip_background and niche_id == -1:
                continue
            niche_mask = table[niche_column] == niche_id
            niche_table = table[niche_mask].copy()
            if len(niche_table) < 2:
                continue

            niche_node_ids = set(niche_table[id_column].astype(str))
            spatial = dataset.spatial_table.copy()
            niche_spatial = spatial[
                spatial["source_id"].astype(str).isin(niche_node_ids)
                & spatial["target_id"].astype(str).isin(niche_node_ids)
            ]
            if niche_spatial.empty:
                continue

            try:
                result = run_neighborhood_enrichment(
                    niche_table,
                    niche_spatial,
                    label_column=cell_type_column,
                    id_column=id_column,
                    n_perms=int(params.get("n_perms", 0)),
                    random_state=int(params.get("random_state", 42)),
                    undirected=bool(params.get("undirected", False)),
                    enrichment_mode=str(params.get("enrichment_mode", "obs-exp")),
                )
                enrichment_by_niche[str(niche_id)] = result.enrichment
                sections.append(
                    ReportSection(
                        title=f"Cell-Type Enrichment — Niche {niche_id}",
                        description=(
                            f"Cell-type neighborhood enrichment within niche {niche_id}."
                        ),
                        tables={"enrichment": result.enrichment},
                    )
                )
            except (ValueError, KeyError):
                continue

        payload: Dict[str, Any] = {"enrichment_by_niche": enrichment_by_niche}
        return PluginResult(name=self.name, payload=payload, sections=sections)


def _resolve_columns(table: pd.DataFrame, prefix: str, explicit) -> List[str]:
    if explicit:
        return [str(c) for c in explicit]
    return [c for c in table.columns if str(c).startswith(prefix)]


def _resolve_logit_bias(params: Dict[str, Any]) -> Optional[List[float]]:
    """Head output bias, read from a checkpoint unless given explicitly.

    Subtracting it makes the margin reflect the instance-driven part of the
    decision rather than the head's prior, so it should normally come straight
    from the trained model rather than being transcribed by hand.
    """
    explicit = params.get("logit_bias")
    if explicit is not None:
        return [float(v) for v in explicit]

    checkpoint = params.get("logit_bias_checkpoint")
    if not checkpoint:
        return None

    from grass_mil.models.training.checkpoint_init import read_graph_head_bias

    bias = read_graph_head_bias(
        str(checkpoint), head_prefix=str(params.get("head_prefix", "graph_head"))
    )
    if bias is None:
        raise ValueError(
            f"No graph head bias found in {checkpoint!r}. Check head_prefix, or set "
            "logit_bias explicitly if the checkpoint holds only an encoder."
        )
    return bias


@dataclass
class MarginAttributionPlugin(InterpretabilityPlugin):
    """Exact additive attribution of bag decisions to instance niches.

    Requires per-class attention columns, which only exist when the head emits
    one attention channel per class.
    """

    name: str = "margin_attribution"

    def required_inputs(self) -> List[str]:
        return ["niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        table = dataset.instance_table
        labels = np.asarray(context.state["niche_labels"])

        attention_columns = _resolve_columns(
            table,
            str(params.get("attention_prefix", "attention_c")),
            params.get("attention_columns"),
        )
        if not attention_columns:
            attention_columns = [str(params.get("attention_column", "attention"))]
        logit_columns = _resolve_columns(
            table, str(params.get("logit_prefix", "logit_")), params.get("logit_columns")
        )
        if not logit_columns:
            raise ValueError(
                "Margin attribution needs instance logit columns (default prefix 'logit_'). "
                "Enable the interpretability export in the predict step."
            )

        result = niche_attribution_summary(
            table,
            labels,
            attention_columns=attention_columns,
            logit_columns=logit_columns,
            bag_id_column=str(params.get("bag_id_column", dataset.bag_id_column)),
            logit_bias=_resolve_logit_bias(params),
            n_bootstrap=int(params.get("n_bootstrap", 200)),
            random_state=int(params.get("random_state", 0)),
            margin_eps=float(params.get("margin_eps", 1e-6)),
            focus_classes=params.get("focus_classes"),
        )
        payload = {
            "per_niche": result.per_niche,
            "identity_residual": result.identity_residual,
        }
        sections = [
            ReportSection(
                title="Margin Attribution",
                description=(
                    "Exact additive contributions M[i,c] = A[i,c] * l[i,c], summarised per "
                    "niche with percentile bootstrap intervals over regions."
                ),
                tables={"per_niche": result.per_niche},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class MoransIPlugin(InterpretabilityPlugin):
    """Spatial autocorrelation per niche, with a permutation null."""

    name: str = "morans_i"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Moran's I requires a spatial table.")
        table = dataset.instance_table.copy()
        niche_column = str(params.get("niche_column", "niche_label"))
        table[niche_column] = np.asarray(context.state["niche_labels"])

        feature_columns = _resolve_columns(
            table,
            str(params.get("feature_prefix", "comp_")),
            params.get("feature_columns"),
        )
        if not feature_columns:
            raise ValueError("Moran's I found no feature columns to evaluate.")

        result = run_morans_i(
            table,
            dataset.spatial_table,
            feature_columns=feature_columns,
            group_column=niche_column,
            id_column=str(params.get("id_column", dataset.id_column)),
            n_perms=int(params.get("n_perms", 100)),
            random_state=int(params.get("random_state", 0)),
            min_nodes=int(params.get("min_nodes", 3)),
        )
        payload: Dict[str, Any] = {
            "statistic": result.statistic,
            "pvalue": result.pvalue,
            "qvalue": result.qvalue,
            "n_nodes": result.n_nodes,
        }
        tables = {"morans_i": result.statistic, "qvalue": result.qvalue}

        reference = params.get("reference_group", -1)
        if reference in result.statistic.index:
            differential = diff_morans_i_vs_reference(result, reference_group=reference)
            payload["differential_z"] = differential
            tables["differential_z"] = differential

        sections = [
            ReportSection(
                title="Spatial Autocorrelation (Moran's I)",
                description=(
                    "Global Moran's I per niche over the instance graph, with a "
                    "permutation null and BH-FDR adjustment."
                ),
                tables=tables,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class RipleyPlugin(InterpretabilityPlugin):
    """Centred cross-L curves over instance centroids."""

    name: str = "ripley"

    def required_inputs(self) -> List[str]:
        return ["niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        table = dataset.instance_table.copy()
        label_column = str(params.get("label_column", "niche_label"))
        if label_column == "niche_label":
            table[label_column] = np.asarray(context.state["niche_labels"])

        pairs = params.get("pairs")
        if pairs is not None:
            pairs = [(str(a), str(b)) for a, b in pairs]

        result = aggregate_ripley(
            table,
            label_column=label_column,
            group_column=params.get("group_column"),
            x_column=str(params.get("x_column", "center_x")),
            y_column=str(params.get("y_column", "center_y")),
            pairs=pairs,
            n_radii=int(params.get("n_radii", 50)),
            max_fraction=float(params.get("max_fraction", 0.25)),
            min_count=int(params.get("min_count", 5)),
            radius_source=str(params.get("radius_source", "median")),
        )
        curves = result.to_frame()
        payload = {
            "radii": result.radii,
            "curves": curves,
            "n_groups": result.n_groups,
        }
        sections = [
            ReportSection(
                title="Ripley Cross-L",
                description=(
                    "Centred cross-L, L(r) - r: zero under complete spatial randomness, "
                    "positive under clustering, negative under regularity."
                ),
                tables={"cross_l": curves},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class NicheAgreementPlugin(InterpretabilityPlugin):
    """Agreement between the active partition and previously stored ones.

    Extra labelings are read from columns of the instance table, so a user
    joins a prior run's niche_labels.csv rather than re-running clustering.
    """

    name: str = "niche_agreement"

    def required_inputs(self) -> List[str]:
        return ["niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        table = dataset.instance_table
        labelings: Dict[str, Any] = {
            str(params.get("active_name", "active")): np.asarray(context.state["niche_labels"])
        }
        for column in params.get("label_columns", []) or []:
            if column not in table.columns:
                raise ValueError(
                    f"niche_agreement label column {column!r} is not in the instance table."
                )
            labelings[str(column)] = table[column].to_numpy()

        if len(labelings) < 2:
            raise ValueError(
                "niche_agreement needs at least one additional labeling; set "
                "params.niche_agreement.label_columns."
            )

        result = compute_niche_agreement(
            labelings,
            metrics=tuple(params.get("metrics", ("ari", "ami", "nmi"))),
            include_jaccard=bool(params.get("include_jaccard", True)),
        )
        payload = {
            "pairwise_metrics": result.pairwise_metrics,
            "jaccard": result.jaccard,
        }
        sections = [
            ReportSection(
                title="Cross-Space Niche Agreement",
                description=(
                    "Chance-corrected agreement between partitions of the same instances."
                ),
                tables={"pairwise_metrics": result.pairwise_metrics.reset_index()},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


def register_builtin_plugins(registry: PluginRegistry) -> None:
    registry.register(NicheProfilesPlugin())
    registry.register(NeighborhoodEnrichmentPlugin())
    registry.register(DiffNeighborhoodEnrichmentPlugin())
    registry.register(FiltrationCurvesPlugin())
    registry.register(TissueGraphPlugin())
    registry.register(PerNicheCellTypeEnrichmentPlugin())
    registry.register(MarginAttributionPlugin())
    registry.register(MoransIPlugin())
    registry.register(RipleyPlugin())
    registry.register(NicheAgreementPlugin())
    registry.register(CellTypeEnrichmentPerNichePlugin())
    registry.register(CellTypeMoranPerNichePlugin())
    registry.register(NicheLabelMoranPlugin())
    registry.register(CellFiltrationCurvesPlugin())
    registry.register(CellTypeRipleyPerNichePlugin())


def create_builtin_registry() -> PluginRegistry:
    registry = create_plugin_registry()
    register_builtin_plugins(registry)
    return registry


def _cells_with_niches(dataset, context: PluginContext, niche_column: str):
    """Cell table with each cell carrying its ego-graph's niche."""
    if dataset.cell_table is None or dataset.cell_edge_table is None:
        raise ValueError(
            "Cell-level analyses require cell_table and cell_edge_table. Export them "
            "with `grass-mil-predict interpretability.cells.enabled=true`."
        )
    return attach_niche_labels_to_cells(
        dataset.cell_table,
        list(np.asarray(context.state["niche_labels"])),
        list(dataset.instance_table[dataset.id_column]),
        instance_id_column=dataset.id_column,
        niche_column=niche_column,
    )


@dataclass
class CellTypeEnrichmentPerNichePlugin(InterpretabilityPlugin):
    """Cell-type x cell-type neighbourhood enrichment, within each niche.

    The instance-level `neighborhood_enrichment` treats a whole ego-graph as one
    node, so it measures how neighbourhood *labels* co-occur. This measures how
    the *cells* inside a niche are wired to each other.
    """

    name: str = "cell_type_enrichment_per_niche"

    def required_inputs(self) -> List[str]:
        return ["cell_table", "cell_edge_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        niche_column = str(params.get("niche_column", "niche_label"))
        cells = _cells_with_niches(dataset, context, niche_column)
        result = per_niche_cell_type_enrichment(
            cells,
            dataset.cell_edge_table,
            cell_type_column=str(params.get("cell_type_column", "cell_type")),
            niche_column=niche_column,
            cell_id_column=str(params.get("cell_id_column", dataset.cell_id_column)),
            n_perms=int(params.get("n_perms", 0)),
            random_state=int(params.get("random_state", 42)),
            undirected=bool(params.get("undirected", False)),
            enrichment_mode=str(params.get("enrichment_mode", "zscore")),
            min_cells=int(params.get("min_cells", 10)),
            skip_background=bool(params.get("skip_background", False)),
        )
        tables = {
            (f"niche_{k}" if k != -1 else "Background"): v for k, v in result.enrichment.items()
        }
        payload: Dict[str, Any] = {
            "enrichment": result.enrichment,
            "pvalues": result.pvalues,
            "n_cells": result.n_cells,
            "n_edges": result.n_edges,
        }
        if bool(params.get("differential_vs_background", True)):
            differential = differential_cell_type_enrichment(
                result, reference=int(params.get("reference_niche", -1))
            )
            payload["differential"] = differential
            for k, v in differential.items():
                tables[f"niche_{k}_vs_Background"] = v

        sections = [
            ReportSection(
                title="Cell-type neighbourhood enrichment, per niche",
                description=(
                    "Cell-type x cell-type enrichment computed over the cells inside "
                    "each niche, using only edges whose both endpoints lie in that "
                    "niche. Unlike the instance-level analysis, this measures how "
                    "cells are wired to one another rather than how neighbourhood "
                    "labels co-occur."
                ),
                tables=tables,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class CellTypeMoranPerNichePlugin(InterpretabilityPlugin):
    """Moran's I of each cell type, within each niche."""

    name: str = "cell_type_moran_per_niche"

    def required_inputs(self) -> List[str]:
        return ["cell_table", "cell_edge_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        niche_column = str(params.get("niche_column", "niche_label"))
        cells = _cells_with_niches(dataset, context, niche_column)
        result = per_niche_cell_type_moran(
            cells,
            dataset.cell_edge_table,
            cell_type_column=str(params.get("cell_type_column", "cell_type")),
            niche_column=niche_column,
            cell_id_column=str(params.get("cell_id_column", dataset.cell_id_column)),
            n_perms=int(params.get("n_perms", 100)),
            random_state=int(params.get("random_state", 0)),
            min_nodes=int(params.get("min_nodes", 3)),
        )
        sections = [
            ReportSection(
                title="Cell-type spatial autocorrelation, per niche",
                description=(
                    "Moran's I of each cell-type indicator over the cell graph of "
                    "each niche: whether cells of that type cluster together "
                    "spatially inside the niche. Rows are niches, columns cell types."
                ),
                tables={"morans_i": result.statistic, "qvalue": result.qvalue},
            )
        ]
        return PluginResult(
            name=self.name,
            payload={
                "statistic": result.statistic,
                "pvalue": result.pvalue,
                "qvalue": result.qvalue,
                "n_nodes": result.n_nodes,
            },
            sections=sections,
        )


@dataclass
class NicheLabelMoranPlugin(InterpretabilityPlugin):
    """Moran's I of the niche assignment itself, over the instance graph."""

    name: str = "niche_label_moran"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Niche-label Moran's I requires a spatial table.")
        niche_column = str(params.get("niche_column", "niche_label"))
        table = dataset.instance_table.copy()
        table[niche_column] = np.asarray(context.state["niche_labels"])
        result = niche_label_moran(
            table,
            dataset.spatial_table,
            niche_column=niche_column,
            id_column=str(params.get("id_column", dataset.id_column)),
            n_perms=int(params.get("n_perms", 100)),
            random_state=int(params.get("random_state", 0)),
        )
        sections = [
            ReportSection(
                title="Niche spatial autocorrelation (global)",
                description=(
                    "Moran's I of each niche indicator across the whole instance "
                    "graph: whether niches form contiguous territories rather than "
                    "interleaving. The global counterpart to the per-niche "
                    "cell-type analyses."
                ),
                tables={"morans_i": result.statistic, "qvalue": result.qvalue},
            )
        ]
        return PluginResult(
            name=self.name,
            payload={
                "statistic": result.statistic,
                "pvalue": result.pvalue,
                "qvalue": result.qvalue,
            },
            sections=sections,
        )


@dataclass
class CellFiltrationCurvesPlugin(InterpretabilityPlugin):
    """Filtration curves over cell-cell edges, counting both endpoints."""

    name: str = "cell_filtration_curves"

    def required_inputs(self) -> List[str]:
        return ["cell_table", "cell_edge_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        niche_column = str(params.get("niche_column", "niche_label"))
        cells = _cells_with_niches(dataset, context, niche_column)
        thresholds = np.linspace(
            float(params.get("threshold_start", 0.0)),
            float(params.get("threshold_stop", 55.0)),
            int(params.get("threshold_count", 500)),
        )
        result = cell_filtration_curves(
            cells,
            dataset.cell_edge_table,
            thresholds=thresholds,
            cell_type_column=str(params.get("cell_type_column", "cell_type")),
            niche_column=niche_column,
            cell_id_column=str(params.get("cell_id_column", dataset.cell_id_column)),
            scale_within_niche=bool(params.get("scale_within_niche", True)),
        )
        sections = [
            ReportSection(
                title="Cell-type filtration curves, per niche",
                description=(
                    "Cells of each type reached as the edge-distance threshold "
                    "grows, computed over the edges inside the ego-graphs. Both "
                    "endpoints of every edge are counted, so the curves describe "
                    "cells rather than neighbourhoods. Distances are micrometres."
                ),
                metadata={"thresholds": result.thresholds.tolist()},
            )
        ]
        return PluginResult(
            name=self.name,
            payload={"curves": result.curves, "thresholds": result.thresholds},
            sections=sections,
        )


@dataclass
class CellTypeRipleyPerNichePlugin(InterpretabilityPlugin):
    """Ripley cross-L between cell types, with each niche as a region."""

    name: str = "cell_type_ripley_per_niche"

    def required_inputs(self) -> List[str]:
        return ["cell_table", "niche_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        niche_column = str(params.get("niche_column", "niche_label"))
        cells = _cells_with_niches(dataset, context, niche_column)
        result = per_niche_cell_type_ripley(
            cells,
            cell_type_column=str(params.get("cell_type_column", "cell_type")),
            niche_column=niche_column,
            x_column=str(params.get("x_column", "x")),
            y_column=str(params.get("y_column", "y")),
            n_radii=int(params.get("n_radii", 50)),
            max_fraction=float(params.get("max_fraction", 0.25)),
            min_count=int(params.get("min_count", 5)),
            radius_source=str(params.get("radius_source", "median")),
        )
        sections = [
            ReportSection(
                title="Cell-type Ripley cross-L, per niche",
                description=(
                    "Centred cross-L curves between cell types, using the cells' "
                    "own coordinates and treating each niche as a region. Positive "
                    "values indicate co-aggregation at that radius, negative "
                    "values mutual avoidance."
                ),
                tables={"ripley": result.to_frame()} if hasattr(result, "to_frame") else {},
            )
        ]
        return PluginResult(
            name=self.name,
            payload={
                "curves": result.curves,
                "radii": result.radii,
                "pair_counts": result.pair_counts,
            },
            sections=sections,
        )
