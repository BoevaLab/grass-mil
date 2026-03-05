from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.interpretability.tier2.tissue_graph import (
    build_tissue_graph_figure,
    prepare_tissue_graph_view,
)


def _sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1", "n2", "n3", "n4"],
            "sample_id": ["sx", "sx", "sx", "sy", "sy"],
            "center_x": [0.0, 1.0, 2.0, 10.0, 11.0],
            "center_y": [0.0, 1.0, 0.5, 10.0, 11.0],
            "cluster_label": [0, 1, 1, 0, 2],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0", "n1", "n2", "n3"],
            "target_id": ["n1", "n2", "n3", "n4"],
            "distance": [1.0, 1.2, 2.0, 0.9],
        }
    )
    return node_table, spatial_table


def test_prepare_tissue_graph_view_happy_path_filters_to_selected_sample() -> None:
    node_table, spatial_table = _sample_tables()
    view = prepare_tissue_graph_view(
        node_table,
        spatial_table,
        sample_column="sample_id",
        sample_value="sx",
        id_column="instance_id",
        x_column="center_x",
        y_column="center_y",
        label_column="cluster_label",
    )

    assert len(view.nodes) == 3
    assert set(view.nodes["instance_id"].tolist()) == {"n0", "n1", "n2"}
    assert len(view.edges) == 2
    assert set(view.edges["source_id"].tolist()) == {"n0", "n1"}
    assert view.metadata["sample_column"] == "sample_id"
    assert view.metadata["sample_value"] == "sx"
    assert view.metadata["node_count"] == 3
    assert view.metadata["edge_count"] == 2
    assert view.metadata["cluster_counts"] == {"0": 1, "1": 2}


def test_prepare_tissue_graph_view_fails_for_missing_coords() -> None:
    node_table, spatial_table = _sample_tables()
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_tissue_graph_view(
            node_table.drop(columns=["center_x"]),
            spatial_table,
            sample_column="sample_id",
            sample_value="sx",
            id_column="instance_id",
            x_column="center_x",
            y_column="center_y",
            label_column="cluster_label",
        )


def test_prepare_tissue_graph_view_fails_for_missing_or_empty_sample_selector() -> None:
    node_table, spatial_table = _sample_tables()
    with pytest.raises(ValueError, match="sample_value is required"):
        prepare_tissue_graph_view(
            node_table,
            spatial_table,
            sample_column="sample_id",
            sample_value=None,
            id_column="instance_id",
        )
    with pytest.raises(ValueError, match="No rows found"):
        prepare_tissue_graph_view(
            node_table,
            spatial_table,
            sample_column="sample_id",
            sample_value="missing",
            id_column="instance_id",
        )


def test_build_tissue_graph_figure_returns_expected_traces_and_title() -> None:
    node_table, spatial_table = _sample_tables()
    view = prepare_tissue_graph_view(
        node_table,
        spatial_table,
        sample_column="sample_id",
        sample_value="sx",
        id_column="instance_id",
    )
    with_edges = build_tissue_graph_figure(view, show_edges=True, reverse_y=True)
    assert isinstance(with_edges, go.Figure)
    assert len(with_edges.data) == 2
    assert "sample_id=sx" in str(with_edges.layout.title.text)

    without_edges = build_tissue_graph_figure(view, show_edges=False)
    assert isinstance(without_edges, go.Figure)
    assert len(without_edges.data) == 1
