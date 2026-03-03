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
from src.interpretability.plugins.registry import register_plugin
from src.interpretability.tier2.filtration import compute_filtration_curves
from src.interpretability.tier2.neighborhood import run_neighborhood_enrichment


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
class FiltrationCurvesPlugin(InterpretabilityPlugin):
    name: str = "filtration_curves"

    def required_inputs(self) -> List[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params: Any) -> PluginResult:
        if dataset.spatial_table is None:
            raise ValueError("Spatial table is required for filtration curves plugin.")
        table = dataset.instance_table.copy()
        table["cluster_label"] = np.asarray(context.state["cluster_labels"])
        thresholds = np.asarray(params.get("thresholds", np.linspace(0.0, 1.0, 20)))
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


def register_builtin_plugins() -> None:
    register_plugin(ClusterProfilesPlugin())
    register_plugin(AttentionAttributionPlugin())
    register_plugin(NeighborhoodEnrichmentPlugin())
    register_plugin(FiltrationCurvesPlugin())
