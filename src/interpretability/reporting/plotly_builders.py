from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd

from src.interpretability.contracts import ReportBundle
from src.interpretability.tier2.tissue_graph import TissueGraphView, build_tissue_graph_figure

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotly is required for interpretability plotting. "
            "Install it with: pip install plotly"
        ) from exc
    return go


def _safe_slug(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    slug = "".join(out).strip("_")
    return slug or "unknown"


def build_reduction_scatter(bundle: ReportBundle) -> "go.Figure":
    go = _require_plotly()
    if bundle.reduction is None:
        raise ValueError("Reduction result is missing.")
    emb = np.asarray(bundle.reduction.embedding)
    if emb.shape[1] < 2:
        raise ValueError("Reduction embedding must have at least 2 dimensions for scatter.")
    labels = (
        bundle.clustering.labels
        if bundle.clustering is not None
        else np.zeros((emb.shape[0],), dtype=int)
    )
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=emb[:, 0],
                y=emb[:, 1],
                mode="markers",
                marker={"size": 6, "color": labels, "colorscale": "Viridis"},
                name="instances",
            )
        ]
    )
    fig.update_layout(
        title="Reduced Embedding Scatter",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        template="plotly_white",
    )
    return fig


def build_cluster_enrichment_heatmap(bundle: ReportBundle) -> "go.Figure":
    go = _require_plotly()
    if bundle.cluster_summary is None:
        raise ValueError("Cluster summary is missing.")
    enr = bundle.cluster_summary.enrichment
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=enr.to_numpy(),
                x=[str(c) for c in enr.columns],
                y=[str(i) for i in enr.index],
                colorscale="RdBu",
                zmid=0.0,
            )
        ]
    )
    fig.update_layout(
        title="Cluster Enrichment Heatmap",
        xaxis_title="Features",
        yaxis_title="Cluster",
        template="plotly_white",
    )
    return fig


def build_plugin_figures(bundle: ReportBundle) -> Dict[str, "go.Figure"]:
    go = _require_plotly()
    figures: Dict[str, go.Figure] = {}
    for name, result in bundle.plugin_results.items():
        if "curves" in result.payload and "thresholds" in result.payload:
            thresholds = np.asarray(result.payload["thresholds"])
            curves = result.payload["curves"]
            fig = go.Figure()
            for cluster, ct_map in curves.items():
                for cell_type, values in ct_map.items():
                    fig.add_trace(
                        go.Scatter(
                            x=thresholds,
                            y=np.asarray(values),
                            mode="lines",
                            name=f"{cluster}:{cell_type}",
                        )
                    )
            fig.update_layout(
                title=f"{name} Curves",
                xaxis_title="Threshold",
                yaxis_title="Value",
                template="plotly_white",
            )
            figures[name] = fig
        elif "enrichment" in result.payload and isinstance(
            result.payload["enrichment"], pd.DataFrame
        ):
            enr = result.payload["enrichment"]
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=enr.to_numpy(),
                        x=[str(c) for c in enr.columns],
                        y=[str(i) for i in enr.index],
                        colorscale="RdBu",
                        zmid=0.0,
                    )
                ]
            )
            fig.update_layout(title=f"{name} Enrichment", template="plotly_white")
            figures[name] = fig
        elif "enrichment_by_pair" in result.payload and isinstance(
            result.payload["enrichment_by_pair"], dict
        ):
            for pair, enr in result.payload["enrichment_by_pair"].items():
                if not isinstance(enr, pd.DataFrame):
                    continue
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=enr.to_numpy(),
                            x=[str(c) for c in enr.columns],
                            y=[str(i) for i in enr.index],
                            colorscale="RdBu",
                            zmid=0.0,
                        )
                    ]
                )
                fig.update_layout(
                    title=f"{name} Enrichment ({pair})",
                    template="plotly_white",
                )
                figures[f"{name}_{pair}"] = fig
        elif "tissue_graph_view" in result.payload and isinstance(
            result.payload["tissue_graph_view"], TissueGraphView
        ):
            view = result.payload["tissue_graph_view"]
            style = result.payload.get("tissue_graph_style", {})
            fig = build_tissue_graph_figure(
                view,
                show_edges=bool(style.get("show_edges", True)),
                node_size=float(style.get("node_size", 5.0)),
                edge_width=float(style.get("edge_width", 0.5)),
                edge_opacity=float(style.get("edge_opacity", 0.25)),
                colorscale=str(style.get("colorscale", "Viridis")),
                reverse_y=bool(style.get("reverse_y", True)),
                title_prefix=str(style.get("title_prefix", "Tissue Graph")),
            )
            meta = result.payload.get("tissue_graph_meta", {})
            sample_column = _safe_slug(meta.get("sample_column", "sample"))
            sample_value = _safe_slug(meta.get("sample_value", "unknown"))
            figures[f"tissue_graph_{sample_column}_{sample_value}"] = fig
    return figures


def bundle_figures(bundle: ReportBundle) -> List[tuple[str, "go.Figure"]]:
    figures: List[tuple[str, go.Figure]] = []
    if bundle.reduction is not None:
        figures.append(("reduction_scatter", build_reduction_scatter(bundle)))
    if bundle.cluster_summary is not None:
        figures.append(("cluster_enrichment", build_cluster_enrichment_heatmap(bundle)))
    for name, fig in build_plugin_figures(bundle).items():
        figures.append((name, fig))
    return figures
