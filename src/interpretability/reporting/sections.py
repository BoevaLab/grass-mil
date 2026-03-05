from __future__ import annotations

from typing import List

from src.interpretability.contracts import ReportBundle, ReportSection


def build_report_sections(bundle: ReportBundle) -> List[ReportSection]:
    sections: List[ReportSection] = []

    sections.append(
        ReportSection(
            title="Executive Summary",
            description="Interpretability analysis output bundle summary.",
            metadata=bundle.metadata,
        )
    )
    if bundle.cluster_summary is not None:
        sections.append(
            ReportSection(
                title="Cluster Summary",
                description="Cluster counts and enrichment tables.",
                tables={
                    "cluster_counts": bundle.cluster_summary.cluster_counts.to_frame(name="count"),
                    "enrichment": bundle.cluster_summary.enrichment,
                },
            )
        )

    for plugin_result in bundle.plugin_results.values():
        sections.extend(plugin_result.sections)
    return sections
