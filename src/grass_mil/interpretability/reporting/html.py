from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from grass_mil.contracts import ReportSection


def _render_table(title: str, table: pd.DataFrame) -> str:
    return f"<h4>{html.escape(title)}</h4>" + table.to_html(
        classes="table table-striped", border=0, index=True
    )


def render_html_report(
    sections: List[ReportSection],
    figures: Dict[str, Any],
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_blocks: List[str] = []
    for name, figure in figures.items():
        figure_blocks.append(
            f"<section><h2>{html.escape(name)}</h2>{figure.to_html(full_html=False, include_plotlyjs='cdn')}</section>"
        )

    section_blocks: List[str] = []
    for section in sections:
        meta_html = ""
        if section.metadata:
            meta_df = pd.DataFrame([section.metadata])
            meta_html = _render_table("Metadata", meta_df)
        tables_html = "".join(_render_table(k, v) for k, v in section.tables.items())
        section_blocks.append(
            "<section>"
            f"<h2>{html.escape(section.title)}</h2>"
            f"<p>{html.escape(section.description)}</p>"
            f"{meta_html}{tables_html}"
            "</section>"
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Interpretability Report</title>
  <style>
    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 2rem; }}
    h1, h2, h3, h4 {{ color: #1f2937; }}
    section {{ margin-bottom: 2rem; page-break-inside: avoid; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; }}
    @media print {{
      section {{ page-break-inside: avoid; }}
      .page-break {{ page-break-after: always; }}
    }}
  </style>
</head>
<body>
  <h1>Interpretability Report</h1>
  {''.join(section_blocks)}
  <div class="page-break"></div>
  <h1>Interactive Plots</h1>
  {''.join(figure_blocks)}
</body>
</html>"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
