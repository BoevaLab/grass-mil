from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grass_mil.interpretability.contracts import (
    ClusterSummary,
    PluginResult,
    ReportSection,
)
from grass_mil.interpretability.core.agreement import compute_cluster_agreement
from grass_mil.interpretability.core.attribution import cluster_attribution_summary
from grass_mil.interpretability.core.biomarkers import (
    cluster_attention_summary,
    cluster_biomarker_summary,
)
from grass_mil.interpretability.plugins.base import InterpretabilityPlugin, PluginContext
from grass_mil.interpretability.plugins.registry import PluginRegistry, create_plugin_registry
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
) -> ClusterSummary | None:
    summary = context.state.get("cluster_summary")
    if not isinstance(summary, ClusterSummary):
        return None
    if summary.cluster_labels.shape != labels.shape:
        return None
    if not np.array_equal(summary.cluster_labels, labels):
        return None
    return summary


@dataclass
class ClusterProfilesPlugin(InterpretabilityPlugin):
    name: str = "cluster_profiles"

    def required_inputs(self) -> List[str]:
        return ["cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        labels = np.asarray(context.state["cluster_labels"])
        summary = _context_cluster_summary(context, labels=labels)
        if summary is None:
            summary = cluster_biomarker_summary(
                dataset.instance_table,
                labels,
                composition_prefix=str(params.get("composition_prefix", "comp_")),
                variance_estimator=str(params.get("variance_estimator", "unbiased")),
            )
        payload = {
            "composition": summary.composition,
            "enrichment": summary.enrichment,
            "counts": summary.cluster_counts,
        }
        sections = [
            ReportSection(
                title="Cluster Profiles",
                description="Cluster-level composition and enrichment summary.",
                tables={
                    "composition": summary.composition,
                    "enrichment": summary.enrichment,
                },
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class AttentionAttributionPlugin(InterpretabilityPlugin):
    name: str = "attention_attribution"

    def required_inputs(self) -> List[str]:
        return ["cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        labels = np.asarray(context.state["cluster_labels"])
        attn_summary = cluster_attention_summary(
            dataset.instance_table,
            labels,
            attention_column=str(params.get("attention_column", "attention")),
            score_column=str(params.get("score_column", "score")),
            bag_id_column=str(params.get("bag_id_column", dataset.bag_id_column)),
            composition_prefix=str(params.get("composition_prefix", "comp_")),
            variance_estimator=str(params.get("variance_estimator", "unbiased")),
        )
        base_summary = _context_cluster_summary(context, labels=labels)
        if base_summary is not None:
            summary = ClusterSummary(
                cluster_labels=base_summary.cluster_labels,
                composition=base_summary.composition,
                enrichment=base_summary.enrichment,
                cluster_counts=base_summary.cluster_counts,
                weighted_scores=attn_summary.weighted_scores,
                mean_scores=attn_summary.mean_scores,
                attention_present=attn_summary.attention_present,
                attention_lift_present=attn_summary.attention_lift_present,
            )
        else:
            summary = attn_summary
        payload = {
            "weighted_scores": summary.weighted_scores,
            "mean_scores": summary.mean_scores,
            "attention_present": summary.attention_present,
            "attention_lift_present": summary.attention_lift_present,
        }
        section_tables = {}
        for key, value in payload.items():
            if value is not None:
                section_tables[key] = value.to_frame(name=key)
        sections = [
            ReportSection(
                title="Attention Attribution",
                description="Attention-weighted and abundance-corrected cluster scores.",
                tables=section_tables,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class NeighborhoodEnrichmentPlugin(InterpretabilityPlugin):
    name: str = "neighborhood_enrichment"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for neighborhood enrichment plugin.")
        table = dataset.instance_table.copy()
        table["cluster_label"] = np.asarray(context.state["cluster_labels"])
        result = run_neighborhood_enrichment(
            table,
            dataset.spatial_table,
            label_column=str(params.get("label_column", "cluster_label")),
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
                description="Cluster adjacency enrichment relative to expected connectivity.",
                tables=section_tables,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class DiffNeighborhoodEnrichmentPlugin(InterpretabilityPlugin):
    name: str = "diff_neighborhood_enrichment"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for differential neighborhood enrichment.")
        table = dataset.instance_table.copy()
        table["cluster_label"] = np.asarray(context.state["cluster_labels"])
        condition_column = str(
            params.get("condition_column", dataset.condition_column or "condition")
        )
        pairwise = run_diff_neighborhood_enrichment(
            table,
            dataset.spatial_table,
            label_column=str(params.get("label_column", "cluster_label")),
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
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for filtration curves plugin.")
        table = dataset.instance_table.copy()
        table["cluster_label"] = np.asarray(context.state["cluster_labels"])
        thresholds = _resolve_filtration_thresholds(dict(params))
        result = compute_filtration_curves(
            table,
            dataset.spatial_table,
            thresholds=thresholds,
            cluster_column=str(params.get("cluster_column", "cluster_label")),
            cell_type_column=str(params.get("cell_type_column", "cell_type")),
            id_column=str(params.get("id_column", dataset.id_column)),
            distance_column=str(params.get("distance_column", "distance")),
            scale_within_cluster=bool(params.get("scale_within_cluster", True)),
        )
        payload: Dict[str, Any] = {"thresholds": result.thresholds, "curves": result.curves}
        sections = [
            ReportSection(
                title="Filtration Curves",
                description="Distance-threshold accumulation by cluster and cell type.",
                metadata={"thresholds": result.thresholds.tolist()},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class TissueGraphPlugin(InterpretabilityPlugin):
    name: str = "tissue_graph"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for tissue graph plugin.")
        sample_value = params.get("sample_value")
        if sample_value is None or str(sample_value).strip() == "":
            raise ValueError("tissue_graph plugin requires a non-empty sample_value parameter.")

        table = dataset.instance_table.copy()
        table["cluster_label"] = np.asarray(context.state["cluster_labels"])
        sample_column = str(params.get("sample_column", "sample_id"))
        id_column = str(params.get("id_column", dataset.id_column))
        x_column = str(params.get("x_column", "center_x"))
        y_column = str(params.get("y_column", "center_y"))
        label_column = str(params.get("label_column", "cluster_label"))

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
        cluster_counts = meta.get("cluster_counts", {})
        section_summary = pd.DataFrame(
            [
                {
                    "sample_column": meta.get("sample_column"),
                    "sample_value": meta.get("sample_value"),
                    "node_count": meta.get("node_count"),
                    "edge_count": meta.get("edge_count"),
                    "cluster_count": len(cluster_counts)
                    if isinstance(cluster_counts, dict)
                    else 0,
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
                description="Single tissue/sample graph with cluster-colored cells.",
                tables={"summary": section_summary},
                metadata=meta,
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class PerClusterCellTypeEnrichmentPlugin(InterpretabilityPlugin):
    """Compute cell-type neighborhood enrichment within each cluster separately.

    For each cluster, subsets the node table and spatial table to nodes belonging
    to that cluster, then runs neighborhood enrichment with label_column=cell_type.
    This answers: "within cluster C, which cell types are spatially co-located?"
    """

    name: str = "per_cluster_cell_type_enrichment"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for per-cluster cell-type enrichment.")
        cell_type_column = str(
            params.get("cell_type_column", dataset.cell_type_column or "cell_type")
        )
        cluster_column = str(params.get("cluster_column", "cluster_label"))
        id_column = str(params.get("id_column", dataset.id_column))

        table = dataset.instance_table.copy()
        table[cluster_column] = np.asarray(context.state["cluster_labels"])

        unique_clusters = sorted(set(table[cluster_column]))
        skip_noise = bool(params.get("skip_noise", True))

        enrichment_by_cluster: Dict[str, pd.DataFrame] = {}
        sections: List[ReportSection] = []

        for cluster_id in unique_clusters:
            if skip_noise and cluster_id == -1:
                continue
            cluster_mask = table[cluster_column] == cluster_id
            cluster_table = table[cluster_mask].copy()
            if len(cluster_table) < 2:
                continue

            cluster_node_ids = set(cluster_table[id_column].astype(str))
            spatial = dataset.spatial_table.copy()
            cluster_spatial = spatial[
                spatial["source_id"].astype(str).isin(cluster_node_ids)
                & spatial["target_id"].astype(str).isin(cluster_node_ids)
            ]
            if cluster_spatial.empty:
                continue

            try:
                result = run_neighborhood_enrichment(
                    cluster_table,
                    cluster_spatial,
                    label_column=cell_type_column,
                    id_column=id_column,
                    n_perms=int(params.get("n_perms", 0)),
                    random_state=int(params.get("random_state", 42)),
                    undirected=bool(params.get("undirected", False)),
                    enrichment_mode=str(params.get("enrichment_mode", "obs-exp")),
                )
                enrichment_by_cluster[str(cluster_id)] = result.enrichment
                sections.append(
                    ReportSection(
                        title=f"Cell-Type Enrichment — Cluster {cluster_id}",
                        description=(
                            f"Cell-type neighborhood enrichment within cluster {cluster_id}."
                        ),
                        tables={"enrichment": result.enrichment},
                    )
                )
            except (ValueError, KeyError):
                continue

        payload: Dict[str, Any] = {"enrichment_by_cluster": enrichment_by_cluster}
        return PluginResult(name=self.name, payload=payload, sections=sections)


def _resolve_columns(table: pd.DataFrame, prefix: str, explicit) -> List[str]:
    if explicit:
        return [str(c) for c in explicit]
    return [c for c in table.columns if str(c).startswith(prefix)]


@dataclass
class MarginAttributionPlugin(InterpretabilityPlugin):
    """Exact additive attribution of bag decisions to instance clusters.

    Requires per-class attention columns, which only exist when the head emits
    one attention channel per class.
    """

    name: str = "margin_attribution"

    def required_inputs(self) -> List[str]:
        return ["cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        table = dataset.instance_table
        labels = np.asarray(context.state["cluster_labels"])

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

        result = cluster_attribution_summary(
            table,
            labels,
            attention_columns=attention_columns,
            logit_columns=logit_columns,
            bag_id_column=str(params.get("bag_id_column", dataset.bag_id_column)),
            logit_bias=params.get("logit_bias"),
            n_bootstrap=int(params.get("n_bootstrap", 200)),
            random_state=int(params.get("random_state", 0)),
            margin_eps=float(params.get("margin_eps", 1e-6)),
        )
        payload = {
            "per_cluster": result.per_cluster,
            "identity_residual": result.identity_residual,
        }
        sections = [
            ReportSection(
                title="Margin Attribution",
                description=(
                    "Exact additive contributions M[i,c] = A[i,c] * l[i,c], summarised per "
                    "cluster with percentile bootstrap intervals over regions."
                ),
                tables={"per_cluster": result.per_cluster},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


@dataclass
class MoransIPlugin(InterpretabilityPlugin):
    """Spatial autocorrelation per cluster, with a permutation null."""

    name: str = "morans_i"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Moran's I requires a spatial table.")
        table = dataset.instance_table.copy()
        cluster_column = str(params.get("cluster_column", "cluster_label"))
        table[cluster_column] = np.asarray(context.state["cluster_labels"])

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
            group_column=cluster_column,
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
                    "Global Moran's I per cluster over the instance graph, with a "
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
        return ["cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        table = dataset.instance_table.copy()
        label_column = str(params.get("label_column", "cluster_label"))
        if label_column == "cluster_label":
            table[label_column] = np.asarray(context.state["cluster_labels"])

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
class ClusterAgreementPlugin(InterpretabilityPlugin):
    """Agreement between the active partition and previously stored ones.

    Extra labelings are read from columns of the instance table, so a user
    joins a prior run's cluster_labels.csv rather than re-running clustering.
    """

    name: str = "cluster_agreement"

    def required_inputs(self) -> List[str]:
        return ["cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        table = dataset.instance_table
        labelings: Dict[str, Any] = {
            str(params.get("active_name", "active")): np.asarray(context.state["cluster_labels"])
        }
        for column in params.get("label_columns", []) or []:
            if column not in table.columns:
                raise ValueError(
                    f"cluster_agreement label column {column!r} is not in the instance table."
                )
            labelings[str(column)] = table[column].to_numpy()

        if len(labelings) < 2:
            raise ValueError(
                "cluster_agreement needs at least one additional labeling; set "
                "params.cluster_agreement.label_columns."
            )

        result = compute_cluster_agreement(
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
                title="Cross-Space Cluster Agreement",
                description=(
                    "Chance-corrected agreement between partitions of the same instances."
                ),
                tables={"pairwise_metrics": result.pairwise_metrics.reset_index()},
            )
        ]
        return PluginResult(name=self.name, payload=payload, sections=sections)


def register_builtin_plugins(registry: PluginRegistry) -> None:
    registry.register(ClusterProfilesPlugin())
    registry.register(AttentionAttributionPlugin())
    registry.register(NeighborhoodEnrichmentPlugin())
    registry.register(DiffNeighborhoodEnrichmentPlugin())
    registry.register(FiltrationCurvesPlugin())
    registry.register(TissueGraphPlugin())
    registry.register(PerClusterCellTypeEnrichmentPlugin())
    registry.register(MarginAttributionPlugin())
    registry.register(MoransIPlugin())
    registry.register(RipleyPlugin())
    registry.register(ClusterAgreementPlugin())


def create_builtin_registry() -> PluginRegistry:
    registry = create_plugin_registry()
    register_builtin_plugins(registry)
    return registry
