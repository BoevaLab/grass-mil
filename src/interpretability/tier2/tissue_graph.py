from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class TissueGraphView:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


def _require_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [col for col in required if str(col) not in frame.columns]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{frame_name} is missing required columns: {missing_text}.")


def prepare_tissue_graph_view(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    sample_column: str,
    sample_value: Any,
    id_column: str,
    x_column: str = "center_x",
    y_column: str = "center_y",
    label_column: str = "cluster_label",
    coerce_ids_to_str: bool = True,
    include_edge_distances: bool = True,
) -> TissueGraphView:
    if sample_value is None or str(sample_value).strip() == "":
        raise ValueError("sample_value is required for tissue graph selection.")

    _require_columns(
        node_table,
        [str(sample_column), str(id_column), str(x_column), str(y_column), str(label_column)],
        "node_table",
    )
    _require_columns(spatial_table, ["source_id", "target_id"], "spatial_table")

    selected = node_table[node_table[str(sample_column)].astype(str) == str(sample_value)].copy()
    if selected.empty:
        raise ValueError(
            "No rows found for tissue graph selection: " f"{sample_column}={sample_value!r}."
        )

    node_cols = [
        str(id_column),
        str(sample_column),
        str(x_column),
        str(y_column),
        str(label_column),
    ]
    nodes = selected[node_cols].copy()
    if coerce_ids_to_str:
        nodes[str(id_column)] = nodes[str(id_column)].astype(str)

    edges = spatial_table[["source_id", "target_id"]].copy()
    if include_edge_distances and "distance" in spatial_table.columns:
        edges["distance"] = spatial_table["distance"]
    if coerce_ids_to_str:
        edges["source_id"] = edges["source_id"].astype(str)
        edges["target_id"] = edges["target_id"].astype(str)

    node_ids = set(nodes[str(id_column)].astype(str).tolist())
    edges = edges[
        edges["source_id"].astype(str).isin(node_ids)
        & edges["target_id"].astype(str).isin(node_ids)
    ].copy()

    coords = nodes.set_index(str(id_column))[[str(x_column), str(y_column)]]
    edges = (
        edges.join(
            coords.rename(columns={str(x_column): "x0", str(y_column): "y0"}), on="source_id"
        )
        .join(coords.rename(columns={str(x_column): "x1", str(y_column): "y1"}), on="target_id")
        .dropna(subset=["x0", "y0", "x1", "y1"])
    )

    edge_cols = ["source_id", "target_id", "x0", "y0", "x1", "y1"]
    if "distance" in edges.columns:
        edge_cols.append("distance")
    edges = edges[edge_cols].reset_index(drop=True)

    cluster_counts = (
        nodes[str(label_column)].astype(str).value_counts().sort_index().astype(int).to_dict()
    )
    metadata: Dict[str, Any] = {
        "sample_column": str(sample_column),
        "sample_value": str(sample_value),
        "id_column": str(id_column),
        "x_column": str(x_column),
        "y_column": str(y_column),
        "label_column": str(label_column),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "cluster_counts": cluster_counts,
    }
    return TissueGraphView(nodes=nodes.reset_index(drop=True), edges=edges, metadata=metadata)


def build_tissue_graph_figure(
    view: TissueGraphView,
    *,
    show_edges: bool = True,
    node_size: float = 5.0,
    edge_width: float = 0.5,
    edge_opacity: float = 0.25,
    colorscale: str = "Viridis",
    reverse_y: bool = True,
    title_prefix: str = "Tissue Graph",
) -> go.Figure:
    nodes = view.nodes.copy()
    metadata = dict(view.metadata)

    label_column = str(metadata.get("label_column", "cluster_label"))
    id_column = str(metadata.get("id_column", "instance_id"))
    x_column = str(metadata.get("x_column", "center_x"))
    y_column = str(metadata.get("y_column", "center_y"))
    sample_column = str(metadata.get("sample_column", "sample_id"))
    sample_value = str(metadata.get("sample_value", "unknown"))

    _require_columns(nodes, [id_column, x_column, y_column, label_column], "view.nodes")

    label_series = nodes[label_column].astype(str)
    label_codes = pd.Categorical(label_series).codes

    fig = go.Figure()
    if show_edges and not view.edges.empty:
        edges = view.edges
        _require_columns(edges, ["x0", "y0", "x1", "y1"], "view.edges")
        line_x = np.empty(len(edges) * 3)
        line_y = np.empty(len(edges) * 3)
        line_x[0::3] = edges["x0"].to_numpy(dtype=float)
        line_x[1::3] = edges["x1"].to_numpy(dtype=float)
        line_x[2::3] = np.nan
        line_y[0::3] = edges["y0"].to_numpy(dtype=float)
        line_y[1::3] = edges["y1"].to_numpy(dtype=float)
        line_y[2::3] = np.nan
        fig.add_trace(
            go.Scattergl(
                x=line_x,
                y=line_y,
                mode="lines",
                line={
                    "width": float(edge_width),
                    "color": f"rgba(120,120,120,{float(edge_opacity)})",
                },
                hoverinfo="skip",
                name="edges",
            )
        )

    fig.add_trace(
        go.Scattergl(
            x=nodes[x_column].to_numpy(dtype=float),
            y=nodes[y_column].to_numpy(dtype=float),
            mode="markers",
            marker={
                "size": float(node_size),
                "color": label_codes,
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {"title": "cluster"},
            },
            customdata=np.stack(
                [
                    nodes[id_column].astype(str).to_numpy(),
                    label_series.to_numpy(),
                ],
                axis=1,
            ),
            hovertemplate=(
                f"{id_column}: %{{customdata[0]}}<br>"
                f"{label_column}: %{{customdata[1]}}"
                "<extra></extra>"
            ),
            name="cells",
        )
    )

    fig.update_layout(
        title=f"{title_prefix} ({sample_column}={sample_value})",
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white",
    )
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    return fig
