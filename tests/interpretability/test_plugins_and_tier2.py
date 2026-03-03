from __future__ import annotations

import numpy as np
import pandas as pd

from src.interpretability.contracts import InterpretabilityDataset
from src.interpretability.plugins.base import PluginContext
from src.interpretability.plugins.builtin import register_builtin_plugins
from src.interpretability.plugins.registry import get_plugin, list_plugins
from src.interpretability.tier2.filtration import compute_filtration_curves
from src.interpretability.tier2.neighborhood import run_neighborhood_enrichment


def _sample_dataset() -> InterpretabilityDataset:
    node_table = pd.DataFrame(
        {
            "instance_id": [f"n{i}" for i in range(6)],
            "bag_id": ["b0", "b0", "b0", "b1", "b1", "b1"],
            "cell_type": ["A", "A", "B", "A", "B", "B"],
            "score": [0.3, 0.4, 0.9, 0.1, 0.8, 0.7],
            "attention": [0.2, 0.5, 0.3, 0.1, 0.2, 0.7],
            "inst_emb_0": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            "inst_emb_1": [0.0, 0.2, 0.1, 1.0, 1.2, 1.1],
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": ["n0", "n1", "n2", "n3", "n4", "n5"],
            "target_id": ["n1", "n2", "n0", "n4", "n5", "n3"],
            "distance": [0.1, 0.3, 0.2, 0.8, 0.7, 0.9],
            "weight": [1, 1, 1, 1, 1, 1],
        }
    )
    return InterpretabilityDataset(
        instance_table=node_table,
        spatial_table=edges,
        id_column="instance_id",
        bag_id_column="bag_id",
        cell_type_column="cell_type",
    )


def test_plugin_registry_and_builtin_execution() -> None:
    register_builtin_plugins()
    assert "cluster_profiles" in list_plugins()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    plugin = get_plugin("cluster_profiles")
    out = plugin.run(ds, PluginContext(state={"cluster_labels": labels}))
    assert out.name == "cluster_profiles"
    assert "composition" in out.payload


def test_tier2_neighborhood_and_filtration() -> None:
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    frame = ds.instance_table.copy()
    frame["cluster_label"] = labels
    enr = run_neighborhood_enrichment(
        frame,
        ds.spatial_table,  # type: ignore[arg-type]
        label_column="cluster_label",
        id_column="instance_id",
        n_perms=4,
        random_state=2,
    )
    assert enr.enrichment.shape[0] > 0
    curves = compute_filtration_curves(
        frame,
        ds.spatial_table,  # type: ignore[arg-type]
        thresholds=np.array([0.2, 0.5, 1.0]),
        cluster_column="cluster_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
    )
    assert len(curves.curves) > 0


def test_filtration_subgraph_centric_counts_both_edge_endpoints() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1", "n2"],
            "cluster_label": [0, 0, 1],
            "cell_type": ["A", "B", "A"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0", "n0", "n1"],
            "target_id": ["n1", "n2", "n2"],
            "distance": [0.10, 0.20, 0.20],
        }
    )
    curves = compute_filtration_curves(
        node_table,
        spatial_table,
        thresholds=np.array([0.15, 0.25]),
        cluster_column="cluster_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
        scale_within_cluster=False,
    )
    # At 0.15 only n0->n1 contributes in cluster 0: A=1 (n0), B=1 (n1).
    # At 0.25:
    # - subgraph n0 contributes unique nodes {n0,n1,n2}: A=2, B=1
    # - subgraph n1 contributes unique nodes {n1,n2}: A=1, B=1
    # Total for cluster 0: A=3, B=2.
    assert np.allclose(curves.curves["0"]["A"], np.array([1.0, 3.0]))
    assert np.allclose(curves.curves["0"]["B"], np.array([1.0, 2.0]))
