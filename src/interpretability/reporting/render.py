from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.interpretability.contracts import ReportBundle
from src.interpretability.reporting.html import render_html_report
from src.interpretability.reporting.pdf import render_pdf_via_playwright
from src.interpretability.reporting.plotly_builders import bundle_figures
from src.interpretability.reporting.sections import build_report_sections
from src.interpretability.reporting.snapshot import export_plotly_snapshots


def render_interpretability_report(
    bundle: ReportBundle,
    *,
    output_dir: Path,
    html_enabled: bool = True,
    pdf_enabled: bool = True,
    snapshot_dpi_scale: float = 2.0,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_pairs = bundle_figures(bundle)
    section_list = build_report_sections(bundle)
    figure_map = {name: fig for name, fig in figure_pairs}

    html_path = None
    if html_enabled:
        html_path = render_html_report(section_list, figure_map, output_dir / "report.html")

    snapshot_dir = output_dir / "artifacts" / "snapshots"
    snapshot_paths = export_plotly_snapshots(
        figure_pairs,
        snapshot_dir,
        scale=snapshot_dpi_scale,
    )

    pdf_path = None
    if pdf_enabled:
        if html_path is None:
            html_path = render_html_report(section_list, figure_map, output_dir / "report.html")
        pdf_path = render_pdf_via_playwright(
            html_path=html_path,
            pdf_path=output_dir / "report.pdf",
        )

    return {
        "html_path": str(html_path) if html_path else None,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "snapshot_paths": {k: str(v) for k, v in snapshot_paths.items()},
    }
