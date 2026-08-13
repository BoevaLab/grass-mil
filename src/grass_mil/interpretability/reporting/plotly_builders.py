from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd

from grass_mil.contracts import ReportBundle
from grass_mil.interpretability.core.niches import (
    BACKGROUND_NICHE_ID,
    niche_display_name,
    niche_display_names,
)
from grass_mil.interpretability.tier2.tissue_graph import (
    TissueGraphView,
    build_tissue_graph_figure,
)

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


def build_niche_enrichment_heatmap(bundle: ReportBundle) -> "go.Figure":
    go = _require_plotly()
    if bundle.niche_summary is None:
        raise ValueError("Niche summary is missing.")
    enr = bundle.niche_summary.enrichment
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=enr.to_numpy(),
                x=[str(c) for c in enr.columns],
                # Rows are already in dendrogram order from the niche summary.
                y=niche_display_names(enr.index),
                colorscale="RdBu",
                zmid=0.0,
            )
        ]
    )
    fig.update_layout(
        title="Niche Enrichment Heatmap",
        xaxis_title="Features",
        yaxis_title="Niche",
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
            for niche, ct_map in curves.items():
                for cell_type, values in ct_map.items():
                    fig.add_trace(
                        go.Scatter(
                            x=thresholds,
                            y=np.asarray(values),
                            mode="lines",
                            name=f"{niche}:{cell_type}",
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
        elif "enrichment_by_niche" in result.payload and isinstance(
            result.payload["enrichment_by_niche"], dict
        ):
            for niche_id, enr in result.payload["enrichment_by_niche"].items():
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
                    title=f"{name} — {niche_display_name(niche_id)}",
                    template="plotly_white",
                )
                figures[f"{name}_niche_{_safe_slug(niche_id)}"] = fig
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


def _resolve_color_values(series: "pd.Series") -> tuple[np.ndarray, Dict[str, object]]:
    """Map an instance-table column onto numeric marker colors.

    Numeric columns are used as-is. Non-numeric columns (``condition`` is
    routinely a string label) are factorized into integer codes, with the
    original labels retained as colorbar ticks, so categorical attributes stay
    plottable instead of raising.

    Returns:
        (values, colorbar_extra) where ``colorbar_extra`` is merged into the
        marker colorbar spec.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.notna().any()):
        return numeric.to_numpy(dtype=float), {}

    codes, uniques = pd.factorize(series, sort=True)
    values = codes.astype(float)
    values[codes < 0] = np.nan
    colorbar_extra: Dict[str, object] = {
        "tickmode": "array",
        "tickvals": list(range(len(uniques))),
        "ticktext": [str(value) for value in uniques],
    }
    return values, colorbar_extra


def build_multi_attribute_scatters(
    bundle: ReportBundle,
    extra_columns: Dict[str, str] | None = None,
) -> List[tuple[str, "go.Figure"]]:
    """Build scatter plots from reduction embeddings colored by multiple attributes.

    Creates scatters for both PCA (from niche_feature_reduction) and the main
    reduction (e.g. UMAP), each colored by niche labels and any extra columns
    found in the instance_table.

    Args:
        bundle: ReportBundle with reduction and optionally niche_feature_reduction.
        extra_columns: mapping of {column_name: colorscale}. Defaults to common
            attributes: condition, score, attention.

    Returns:
        List of (name, figure) tuples.
    """
    go = _require_plotly()
    figures: List[tuple[str, go.Figure]] = []
    table = bundle.dataset.instance_table

    if extra_columns is None:
        extra_columns = {}
        for col, cscale in [
            ("condition", "Viridis"),
            ("score", "Magma"),
            ("attention", "Viridis"),
        ]:
            if col in table.columns:
                extra_columns[col] = cscale

    labels = (
        bundle.clustering.labels
        if bundle.clustering is not None
        else np.zeros((len(table),), dtype=int)
    )

    # True for real niches; Background (-1) is excluded from the filtered views.
    background_mask = labels != BACKGROUND_NICHE_ID if BACKGROUND_NICHE_ID in labels else None

    reductions: List[tuple[str, np.ndarray, str, str]] = []
    if bundle.niche_feature_reduction is not None:
        emb = np.asarray(bundle.niche_feature_reduction.embedding)
        if emb.shape[1] >= 2:
            fitted = bundle.niche_feature_reduction.fitted_object
            if fitted is not None and hasattr(fitted, "explained_variance_ratio_"):
                ev = fitted.explained_variance_ratio_
                x_label = f"PC1 ({ev[0] * 100:.1f}% var)"
                y_label = f"PC2 ({ev[1] * 100:.1f}% var)"
            else:
                x_label, y_label = "PC1", "PC2"
            reductions.append(("pca", emb, x_label, y_label))
    if bundle.reduction is not None:
        emb = np.asarray(bundle.reduction.embedding)
        if emb.shape[1] >= 2:
            method = bundle.reduction.method.upper()
            reductions.append(
                (
                    bundle.reduction.method,
                    emb,
                    f"{method} 1",
                    f"{method} 2",
                )
            )

    for red_name, emb, x_label, y_label in reductions:
        # Niche scatter (all points)
        fig = go.Figure(
            data=[
                go.Scattergl(
                    x=emb[:, 0],
                    y=emb[:, 1],
                    mode="markers",
                    marker={
                        "size": 3,
                        "color": labels,
                        "colorscale": "Viridis",
                        "showscale": True,
                    },
                )
            ]
        )
        fig.update_layout(
            title=f"{red_name.upper()} — Niche Labels",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
        )
        figures.append((f"{red_name}_niche_all", fig))

        # Niche scatter (excluding Background)
        if background_mask is not None:
            fig = go.Figure(
                data=[
                    go.Scattergl(
                        x=emb[background_mask, 0],
                        y=emb[background_mask, 1],
                        mode="markers",
                        marker={
                            "size": 3,
                            "color": labels[background_mask],
                            "colorscale": "Viridis",
                            "showscale": True,
                        },
                    )
                ]
            )
            fig.update_layout(
                title=f"{red_name.upper()} — Niche Labels (excluding Background)",
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="plotly_white",
            )
            figures.append((f"{red_name}_niche_filtered", fig))

        # Extra column scatters
        for col, cscale in extra_columns.items():
            values, colorbar_extra = _resolve_color_values(table[col])
            colorbar = {"title": col, **colorbar_extra}
            fig = go.Figure(
                data=[
                    go.Scattergl(
                        x=emb[:, 0],
                        y=emb[:, 1],
                        mode="markers",
                        marker={
                            "size": 3,
                            "color": values,
                            "colorscale": cscale,
                            "showscale": True,
                            "colorbar": colorbar,
                        },
                    )
                ]
            )
            fig.update_layout(
                title=f"{red_name.upper()} — {col}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="plotly_white",
            )
            figures.append((f"{red_name}_{_safe_slug(col)}", fig))

            if background_mask is not None:
                fig = go.Figure(
                    data=[
                        go.Scattergl(
                            x=emb[background_mask, 0],
                            y=emb[background_mask, 1],
                            mode="markers",
                            marker={
                                "size": 3,
                                "color": values[background_mask],
                                "colorscale": cscale,
                                "showscale": True,
                                "colorbar": colorbar,
                            },
                        )
                    ]
                )
                fig.update_layout(
                    title=f"{red_name.upper()} — {col} (excluding Background)",
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    template="plotly_white",
                )
                figures.append((f"{red_name}_{_safe_slug(col)}_filtered", fig))

    return figures


def build_composite_niche_heatmap(bundle: ReportBundle) -> "go.Figure":
    """Composition enrichment beside niche abundance.

    Attention-only panels were removed: interpretation is expressed through
    weighted logit margins (the ``margin_attribution`` plugin), not through
    attention weights on their own, which say what the model looked at but not
    which way it pushed the decision.
    """
    go = _require_plotly()
    from plotly.subplots import make_subplots

    summary = bundle.niche_summary
    if summary is None:
        raise ValueError("Niche summary is missing.")

    enr = summary.enrichment
    # Niche ids already carry dendrogram order; Background is named.
    niche_ids = niche_display_names(enr.index)
    feature_names = [str(c) for c in enr.columns]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[13, 1],
        horizontal_spacing=0.02,
    )
    fig.add_trace(
        go.Heatmap(
            z=enr.to_numpy(),
            x=feature_names,
            y=niche_ids,
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "Enrichment", "x": 0.86, "len": 0.9},
        ),
        row=1,
        col=1,
    )

    counts = summary.niche_counts.reindex(enr.index).to_numpy().astype(float)
    log_counts = np.log(np.clip(counts, 1, None)).reshape(-1, 1)
    fig.add_trace(
        go.Heatmap(
            z=log_counts,
            x=["Log Abund."],
            y=niche_ids,
            colorscale="Viridis",
            colorbar={"title": "Log N", "x": 1.02, "len": 0.9},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Niche Composition and Abundance",
        template="plotly_white",
        height=max(400, 50 * len(niche_ids) + 200),
    )
    return fig


def bundle_figures(bundle: ReportBundle) -> List[tuple[str, "go.Figure"]]:
    figures: List[tuple[str, go.Figure]] = []
    if bundle.reduction is not None:
        figures.append(("reduction_scatter", build_reduction_scatter(bundle)))
    if bundle.niche_summary is not None:
        figures.append(("niche_enrichment", build_niche_enrichment_heatmap(bundle)))

    # Multi-attribute scatters (PCA + UMAP colored by various attributes)
    if bundle.reduction is not None or bundle.niche_feature_reduction is not None:
        figures.extend(build_multi_attribute_scatters(bundle))

    # Composite niche heatmap (enrichment + predictions + attention + abundance)
    if bundle.niche_summary is not None:
        try:
            figures.append(("composite_niche_summary", build_composite_niche_heatmap(bundle)))
        except (ValueError, KeyError):
            pass

    for name, fig in build_plugin_figures(bundle).items():
        figures.append((name, fig))
    return figures
