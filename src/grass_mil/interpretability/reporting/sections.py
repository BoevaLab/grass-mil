from __future__ import annotations

from typing import List

from grass_mil.interpretability.contracts import ReportBundle, ReportSection


def build_report_sections(bundle: ReportBundle) -> List[ReportSection]:
    sections: List[ReportSection] = []

    sections.append(
        ReportSection(
            title="Executive Summary",
            description="Interpretability analysis output bundle summary.",
            metadata=bundle.metadata,
        )
    )
    if bundle.niche_summary is not None:
        sections.append(
            ReportSection(
                title="Niche Summary",
                description="Niche counts and enrichment tables.",
                tables={
                    "niche_counts": bundle.niche_summary.niche_counts.to_frame(name="count"),
                    "enrichment": bundle.niche_summary.enrichment,
                },
            )
        )

    for plugin_result in bundle.plugin_results.values():
        sections.extend(plugin_result.sections)
    return sections
