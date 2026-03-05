from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.interpretability.contracts import (
    PluginResult,
    ReportSection,
)
from src.interpretability.core.biomarkers import (
    cluster_attention_summary,
    cluster_biomarker_summary,
)
from src.interpretability.plugins.base import InterpretabilityPlugin, PluginContext
from src.interpretability.plugins.registry import PluginRegistry, create_plugin_registry
from src.interpretability.tier2.filtration import compute_filtration_curves
from src.interpretability.tier2.neighborhood import (
    run_diff_neighborhood_enrichment,
    run_neighborhood_enrichment,
)


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


@dataclass
class ClusterProfilesPlugin(InterpretabilityPlugin):
    name: str = "cluster_profiles"

    def required_inputs(self) -> List[str]:
        return ["cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        labels = np.asarray(context.state["cluster_labels"])
        summary = cluster_biomarker_summary(
            dataset.instance_table,
            labels,
            cell_type_column=params.get("cell_type_column", dataset.cell_type_column),
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
        summary = cluster_attention_summary(
            dataset.instance_table,
            labels,
            attention_column=str(params.get("attention_column", "attention")),
            score_column=str(params.get("score_column", "score")),
            bag_id_column=str(params.get("bag_id_column", dataset.bag_id_column)),
            cell_type_column=params.get("cell_type_column", dataset.cell_type_column),
            composition_prefix=str(params.get("composition_prefix", "comp_")),
            variance_estimator=str(params.get("variance_estimator", "unbiased")),
        )
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


def register_builtin_plugins(registry: PluginRegistry) -> None:
    registry.register(ClusterProfilesPlugin())
    registry.register(AttentionAttributionPlugin())
    registry.register(NeighborhoodEnrichmentPlugin())
    registry.register(DiffNeighborhoodEnrichmentPlugin())
    registry.register(FiltrationCurvesPlugin())


def create_builtin_registry() -> PluginRegistry:
    registry = create_plugin_registry()
    register_builtin_plugins(registry)
    return registry
