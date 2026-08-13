from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from grass_mil.interpretability.contracts import NicheSummary, InterpretabilityDataset
from grass_mil.interpretability.plugins.base import PluginContext
from grass_mil.interpretability.plugins.builtin import create_builtin_registry
from grass_mil.interpretability.plugins.registry import create_plugin_registry
from grass_mil.interpretability.tier2.filtration import compute_filtration_curves
from grass_mil.interpretability.tier2.neighborhood import (
    NeighborhoodEnrichmentResult,
    run_diff_neighborhood_enrichment,
    run_neighborhood_enrichment,
)


def _sample_dataset() -> InterpretabilityDataset:
    node_table = pd.DataFrame(
        {
            "instance_id": [f"n{i}" for i in range(6)],
            "bag_id": ["b0", "b0", "b0", "b1", "b1", "b1"],
            "cell_type": ["A", "A", "B", "A", "B", "B"],
            "comp_A": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            "comp_B": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "score": [0.3, 0.4, 0.9, 0.1, 0.8, 0.7],
            "attention": [0.2, 0.5, 0.3, 0.1, 0.2, 0.7],
            "sample_id": ["sx", "sx", "sx", "sy", "sy", "sy"],
            "condition": ["X", "X", "X", "Y", "Y", "Y"],
            "center_x": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            "center_y": [0.0, 0.2, 0.1, 1.0, 1.2, 1.1],
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
    registry = create_builtin_registry()
    assert "niche_profiles" in registry.list()
    assert "tissue_graph" in registry.list()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    plugin = registry.get("niche_profiles")
    out = plugin.run(ds, PluginContext(state={"niche_labels": labels}))
    assert out.name == "niche_profiles"
    assert "composition" in out.payload


def test_builtin_tissue_graph_plugin_execution() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    plugin = registry.get("tissue_graph")
    out = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels}),
        sample_column="sample_id",
        sample_value="sx",
        id_column="instance_id",
    )
    assert out.name == "tissue_graph"
    assert "tissue_graph_view" in out.payload
    assert "tissue_graph_meta" in out.payload
    view = out.payload["tissue_graph_view"]
    assert view.metadata["sample_value"] == "sx"
    assert len(out.sections) == 1


def test_builtin_tissue_graph_plugin_requires_sample_value() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    plugin = registry.get("tissue_graph")
    with pytest.raises(ValueError, match="requires a non-empty sample_value"):
        plugin.run(
            ds,
            PluginContext(state={"niche_labels": labels}),
            sample_column="sample_id",
            id_column="instance_id",
        )


def test_builtin_tissue_graph_plugin_fails_when_coords_missing() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    missing_coords_ds = InterpretabilityDataset(
        instance_table=ds.instance_table.drop(columns=["center_x", "center_y"]),
        spatial_table=ds.spatial_table,
        id_column=ds.id_column,
        bag_id_column=ds.bag_id_column,
        cell_type_column=ds.cell_type_column,
    )
    labels = np.array([0, 0, 1, 1, 1, 0])
    plugin = registry.get("tissue_graph")
    with pytest.raises(ValueError, match="missing required columns"):
        plugin.run(
            missing_coords_ds,
            PluginContext(state={"niche_labels": labels}),
            sample_column="sample_id",
            sample_value="sx",
            id_column="instance_id",
        )


def test_builtin_diff_neighborhood_plugin_execution() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    plugin = registry.get("diff_neighborhood_enrichment")
    out = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels}),
        condition_column="condition",
        permutation_group_column="sample_id",
        n_perms=0,
        undirected=True,
    )
    assert out.name == "diff_neighborhood_enrichment"
    assert "enrichment_by_pair" in out.payload
    assert "X_Y" in out.payload["enrichment_by_pair"]
    assert any(
        section.title.startswith("Differential Neighborhood Enrichment")
        for section in out.sections
    )


def test_cluster_profiles_reuses_context_cluster_summary(monkeypatch) -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    summary = NicheSummary(
        niche_labels=labels.copy(),
        composition=pd.DataFrame({"comp_A": [0.1, 0.2]}, index=[0, 1]),
        enrichment=pd.DataFrame({"comp_A": [1.0, -1.0]}, index=[0, 1]),
        niche_counts=pd.Series({0: 2, 1: 4}),
    )

    def _should_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("niche_composition_summary should not be recomputed")

    monkeypatch.setattr(
        "grass_mil.interpretability.plugins.builtin.niche_composition_summary", _should_not_run
    )
    plugin = registry.get("niche_profiles")
    out = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels, "niche_summary": summary}),
    )
    assert out.payload["composition"].equals(summary.composition)
    assert out.payload["enrichment"].equals(summary.enrichment)


def test_attention_attribution_reuses_context_base_summary(monkeypatch) -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    base_summary = NicheSummary(
        niche_labels=labels.copy(),
        composition=pd.DataFrame({"comp_A": [0.6, 0.4]}, index=[0, 1]),
        enrichment=pd.DataFrame({"comp_A": [0.2, -0.2]}, index=[0, 1]),
        niche_counts=pd.Series({0: 3, 1: 3}),
    )
    attn_summary = NicheSummary(
        niche_labels=labels.copy(),
        composition=pd.DataFrame({"comp_A": [0.9, 0.1]}, index=[0, 1]),
        enrichment=pd.DataFrame({"comp_A": [1.2, -1.2]}, index=[0, 1]),
        niche_counts=pd.Series({0: 2, 1: 4}),
        weighted_scores=pd.Series({0: 0.3, 1: 0.7}),
        mean_scores=pd.Series({0: 0.2, 1: 0.8}),
        attention_present=pd.Series({0: 0.4, 1: 0.6}),
        attention_lift_present=pd.Series({0: 1.1, 1: 0.9}),
    )

    monkeypatch.setattr(
        "grass_mil.interpretability.plugins.builtin.niche_attention_summary",
        lambda *args, **kwargs: attn_summary,  # type: ignore[no-untyped-def]
    )
    plugin = registry.get("attention_attribution")
    out = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels, "niche_summary": base_summary}),
    )
    assert out.payload["weighted_scores"].equals(attn_summary.weighted_scores)
    assert out.payload["mean_scores"].equals(attn_summary.mean_scores)


def test_plugin_registry_instances_are_isolated() -> None:
    class _DummyPlugin:
        name = "dummy"

        def required_inputs(self) -> list[str]:
            return []

        def run(self, dataset, context: PluginContext, **params):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    left = create_plugin_registry()
    right = create_plugin_registry()
    left.register(_DummyPlugin())

    assert "dummy" in left.list()
    assert "dummy" not in right.list()


def test_plugin_registry_rejects_duplicate_names_by_default() -> None:
    class _PluginA:
        name = "duplicate"

        def required_inputs(self) -> list[str]:
            return []

        def run(self, dataset, context: PluginContext, **params):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    class _PluginB:
        name = "duplicate"

        def required_inputs(self) -> list[str]:
            return []

        def run(self, dataset, context: PluginContext, **params):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    registry = create_plugin_registry()
    registry.register(_PluginA())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_PluginB())


def test_plugin_registry_allows_explicit_replace_for_duplicate_names() -> None:
    class _PluginA:
        name = "duplicate"

        def required_inputs(self) -> list[str]:
            return []

        def run(self, dataset, context: PluginContext, **params):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    class _PluginB:
        name = "duplicate"

        def required_inputs(self) -> list[str]:
            return []

        def run(self, dataset, context: PluginContext, **params):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    registry = create_plugin_registry()
    first = _PluginA()
    second = _PluginB()
    registry.register(first)
    registry.register(second, replace=True)
    assert registry.get("duplicate") is second


def test_tier2_neighborhood_and_filtration() -> None:
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 1, 0])
    frame = ds.instance_table.copy()
    frame["niche_label"] = labels
    enr = run_neighborhood_enrichment(
        frame,
        ds.spatial_table,  # type: ignore[arg-type]
        label_column="niche_label",
        id_column="instance_id",
        n_perms=4,
        random_state=2,
    )
    assert enr.enrichment.shape[0] > 0
    curves = compute_filtration_curves(
        frame,
        ds.spatial_table,  # type: ignore[arg-type]
        thresholds=np.array([0.2, 0.5, 1.0]),
        niche_column="niche_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
    )
    assert len(curves.curves) > 0


def test_filtration_subgraph_centric_counts_both_edge_endpoints() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1", "n2"],
            "niche_label": [0, 0, 1],
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
        niche_column="niche_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
        scale_within_cluster=False,
    )
    # At 0.15 only n0->n1 contributes in niche 0: A=1 (n0), B=1 (n1).
    # At 0.25:
    # - subgraph n0 contributes unique nodes {n0,n1,n2}: A=2, B=1
    # - subgraph n1 contributes unique nodes {n1,n2}: A=1, B=1
    # Total for niche 0: A=3, B=2.
    assert np.allclose(curves.curves["0"]["A"], np.array([1.0, 3.0]))
    assert np.allclose(curves.curves["0"]["B"], np.array([1.0, 2.0]))


def test_neighborhood_undirected_option_adds_reverse_direction() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1"],
            "niche_label": ["A", "B"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0"],
            "target_id": ["n1"],
            "weight": [1.0],
        }
    )
    directed = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
    )
    undirected = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=True,
        warn_analytical=False,
    )
    assert directed.observed.loc["A", "B"] == 1.0
    assert directed.observed.loc["B", "A"] == 0.0
    assert undirected.observed.loc["A", "B"] == 1.0
    assert undirected.observed.loc["B", "A"] == 1.0


def test_neighborhood_undirected_deduplicates_already_bidirectional_edges() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1"],
            "niche_label": ["A", "B"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0", "n1"],
            "target_id": ["n1", "n0"],
        }
    )
    out = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=True,
        warn_analytical=False,
    )
    assert out.observed.loc["A", "B"] == 1.0
    assert out.observed.loc["B", "A"] == 1.0


def test_neighborhood_ignores_weight_semantics() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1", "n2"],
            "niche_label": ["A", "B", "B"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0", "n0"],
            "target_id": ["n1", "n2"],
            "weight": [100.0, 0.001],
        }
    )
    out = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
    )
    # Squidpy-style neighborhood enrichment is count-based, not weighted.
    assert out.observed.loc["A", "B"] == 2.0


def test_neighborhood_enrichment_mode_is_tunable() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1"],
            "niche_label": ["A", "B"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0"],
            "target_id": ["n1"],
        }
    )
    zscore = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
        enrichment_mode="zscore",
    )
    difference = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
        enrichment_mode="obs-exp",
    )
    log2fc = run_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
        enrichment_mode="log2fc",
    )
    assert zscore.enrichment.loc["A", "B"] == pytest.approx(1.7320508075688774)
    assert difference.enrichment.loc["A", "B"] == pytest.approx(0.75)
    assert log2fc.enrichment.loc["A", "B"] == pytest.approx(2.0)


def test_neighborhood_enrichment_mode_rejects_unknown_values() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1"],
            "niche_label": ["A", "B"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0"],
            "target_id": ["n1"],
        }
    )
    with pytest.raises(ValueError, match="Unsupported enrichment_mode"):
        run_neighborhood_enrichment(
            node_table,
            spatial_table,
            label_column="niche_label",
            id_column="instance_id",
            n_perms=0,
            warn_analytical=False,
            enrichment_mode="unknown_mode",
        )


def test_diff_neighborhood_returns_pairwise_differences_with_pvalues() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["x0", "x1", "y0", "y1"],
            "niche_label": ["A", "B", "A", "B"],
            "sample_id": ["sx", "sx", "sy", "sy"],
            "condition": ["X", "X", "Y", "Y"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["x0", "x1", "y1", "y0"],
            "target_id": ["x1", "x0", "y0", "y1"],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out = run_diff_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        condition_column="condition",
        id_column="instance_id",
        n_perms=8,
        random_state=3,
        undirected=False,
        warn_analytical=False,
    )
    assert "X_Y" in out
    diff = out["X_Y"]
    assert diff.enrichment.shape == (2, 2)
    assert diff.pvalues is not None
    assert diff.pvalues.shape == (2, 2)


def test_diff_neighborhood_pvalues_are_centered_on_permutation_mean(monkeypatch) -> None:
    calls = {"i": 0}
    # First two calls are baseline left/right. Remaining calls are (left,right) per permutation.
    per_call_scores = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]

    def _fake_run_neighborhood(*args, **kwargs):  # type: ignore[no-untyped-def]
        idx = calls["i"]
        calls["i"] += 1
        score = per_call_scores[idx]
        frame = pd.DataFrame([[score]], index=["A"], columns=["A"])
        zeros = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
        return NeighborhoodEnrichmentResult(
            enrichment=frame,
            observed=zeros,
            expected=zeros,
            pvalues=None,
        )

    monkeypatch.setattr(
        "grass_mil.interpretability.tier2.neighborhood.run_neighborhood_enrichment",
        _fake_run_neighborhood,
    )
    node_table = pd.DataFrame(
        {
            "instance_id": ["x0", "x1", "y0", "y1"],
            "niche_label": ["A", "A", "A", "A"],
            "sample_id": ["sx0", "sx1", "sy0", "sy1"],
            "condition": ["X", "X", "Y", "Y"],
        }
    )
    spatial_table = pd.DataFrame({"source_id": ["x0"], "target_id": ["x1"]})
    out = run_diff_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        condition_column="condition",
        permutation_group_column="sample_id",
        n_perms=4,
        warn_analytical=False,
    )
    p = float(out["X_Y"].pvalues.loc["A", "A"])  # type: ignore[union-attr]
    assert p == pytest.approx(1.0)
    assert p != pytest.approx(0.8)


def test_diff_neighborhood_respects_enrichment_mode() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["x0", "x1", "y0", "y1"],
            "niche_label": ["A", "B", "A", "B"],
            "condition": ["X", "X", "Y", "Y"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["x0", "y1"],
            "target_id": ["x1", "y0"],
        }
    )
    zscore_out = run_diff_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        condition_column="condition",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
        enrichment_mode="zscore",
    )
    diff_out = run_diff_neighborhood_enrichment(
        node_table,
        spatial_table,
        label_column="niche_label",
        condition_column="condition",
        id_column="instance_id",
        n_perms=0,
        undirected=False,
        warn_analytical=False,
        enrichment_mode="obs-exp",
    )
    assert zscore_out["X_Y"].enrichment.loc["A", "B"] == pytest.approx(2.309401076758503)
    assert diff_out["X_Y"].enrichment.loc["A", "B"] == pytest.approx(1.0)


def test_diff_neighborhood_requires_permutation_group_column_for_permutation_mode() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["x0", "x1", "y0", "y1"],
            "niche_label": ["A", "B", "A", "B"],
            "condition": ["X", "X", "Y", "Y"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["x0", "x1", "y1", "y0"],
            "target_id": ["x1", "x0", "y0", "y1"],
        }
    )
    with pytest.raises(ValueError, match="Missing permutation group column"):
        run_diff_neighborhood_enrichment(
            node_table,
            spatial_table,
            label_column="niche_label",
            condition_column="condition",
            n_perms=4,
            warn_analytical=False,
        )


def test_diff_neighborhood_rejects_mixed_condition_groups() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["x0", "x1", "y0", "y1"],
            "niche_label": ["A", "B", "A", "B"],
            "sample_id": ["s0", "s0", "s1", "s1"],
            "condition": ["X", "Y", "X", "Y"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["x0", "x1", "y1", "y0"],
            "target_id": ["x1", "x0", "y0", "y1"],
        }
    )
    with pytest.raises(ValueError, match="must belong to exactly one condition"):
        run_diff_neighborhood_enrichment(
            node_table,
            spatial_table,
            label_column="niche_label",
            condition_column="condition",
            permutation_group_column="sample_id",
            n_perms=4,
            warn_analytical=False,
        )


def test_neighborhood_warns_in_analytical_mode() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["n0", "n1"],
            "niche_label": ["A", "B"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["n0"],
            "target_id": ["n1"],
        }
    )
    with pytest.warns(UserWarning, match="analytical expected/std"):
        run_neighborhood_enrichment(
            node_table,
            spatial_table,
            label_column="niche_label",
            id_column="instance_id",
            n_perms=0,
        )


def test_diff_neighborhood_warns_about_analytical_baseline() -> None:
    node_table = pd.DataFrame(
        {
            "instance_id": ["x0", "x1", "y0", "y1"],
            "niche_label": ["A", "B", "A", "B"],
            "condition": ["X", "X", "Y", "Y"],
        }
    )
    spatial_table = pd.DataFrame(
        {
            "source_id": ["x0", "x1", "y1", "y0"],
            "target_id": ["x1", "x0", "y0", "y1"],
        }
    )
    with pytest.warns(UserWarning, match="analytical neighborhood model"):
        run_diff_neighborhood_enrichment(
            node_table,
            spatial_table,
            label_column="niche_label",
            condition_column="condition",
            id_column="instance_id",
            n_perms=0,
        )


def _legacy_filtration_curves(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    thresholds: np.ndarray,
    niche_column: str,
    cell_type_column: str,
    id_column: str,
    distance_column: str,
) -> dict[str, dict[str, np.ndarray]]:
    node_to_cell_type = node_table.set_index(id_column)[cell_type_column].astype(str).to_dict()
    node_to_cluster = node_table.set_index(id_column)[niche_column].astype(str).to_dict()
    edges = spatial_table.copy()
    edges["source_id"] = edges["source_id"].astype(str)
    edges["target_id"] = edges["target_id"].astype(str)
    edges = edges[edges["source_id"].isin(node_to_cluster.keys())].copy()
    edges["source_cluster"] = edges["source_id"].map(node_to_cluster)

    niches = sorted(node_table[niche_column].astype(str).unique())
    cell_types = sorted(node_table[cell_type_column].astype(str).unique())
    curves: dict[str, dict[str, np.ndarray]] = {
        c: {ct: np.zeros_like(thresholds, dtype=float) for ct in cell_types} for c in niches
    }
    for niche in niches:
        c_edges = edges[edges["source_cluster"] == str(niche)]
        by_subgraph = {
            str(sub_id): frame for sub_id, frame in c_edges.groupby("source_id", sort=False)
        }
        for i, thr in enumerate(thresholds):
            threshold_counts: dict[str, float] = {ct: 0.0 for ct in cell_types}
            for subgraph_edges in by_subgraph.values():
                selected = subgraph_edges[subgraph_edges[distance_column] <= float(thr)]
                if selected.empty:
                    continue
                unique_node_ids = set(selected["source_id"].tolist()) | set(
                    selected["target_id"].tolist()
                )
                for node_id in unique_node_ids:
                    ct = node_to_cell_type.get(str(node_id))
                    if ct is not None:
                        threshold_counts[ct] += 1.0
            for ct in cell_types:
                curves[niche][ct][i] = float(threshold_counts.get(ct, 0.0))
    return curves


def test_filtration_vectorized_matches_legacy_counts() -> None:
    rng = np.random.default_rng(19)
    n_nodes = 32
    node_ids = [f"n{i}" for i in range(n_nodes)]
    node_table = pd.DataFrame(
        {
            "instance_id": node_ids,
            "niche_label": rng.choice(["0", "1", "2"], size=n_nodes),
            "cell_type": rng.choice(["A", "B", "C"], size=n_nodes),
        }
    )
    edge_count = 120
    spatial_table = pd.DataFrame(
        {
            "source_id": rng.choice(node_ids, size=edge_count),
            "target_id": rng.choice(node_ids, size=edge_count),
            "distance": rng.uniform(0.0, 1.0, size=edge_count),
        }
    )
    thresholds = np.array([0.6, 0.2, 0.9, 0.1, 0.4], dtype=float)

    legacy = _legacy_filtration_curves(
        node_table,
        spatial_table,
        thresholds=thresholds,
        niche_column="niche_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
    )
    out = compute_filtration_curves(
        node_table,
        spatial_table,
        thresholds=thresholds,
        niche_column="niche_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
        scale_within_cluster=False,
    )
    for niche, ct_map in legacy.items():
        for cell_type, values in ct_map.items():
            assert np.allclose(out.curves[niche][cell_type], values)


def test_filtration_runtime_sanity() -> None:
    rng = np.random.default_rng(11)
    n_nodes = 220
    node_ids = [f"n{i}" for i in range(n_nodes)]
    node_table = pd.DataFrame(
        {
            "instance_id": node_ids,
            "niche_label": rng.choice(["0", "1", "2", "3"], size=n_nodes),
            "cell_type": rng.choice(["A", "B", "C", "D"], size=n_nodes),
        }
    )
    edge_count = 2400
    spatial_table = pd.DataFrame(
        {
            "source_id": rng.choice(node_ids, size=edge_count),
            "target_id": rng.choice(node_ids, size=edge_count),
            "distance": rng.uniform(0.0, 55.0, size=edge_count),
        }
    )
    thresholds = np.linspace(0.0, 55.0, 300)
    t0 = time.perf_counter()
    out = compute_filtration_curves(
        node_table,
        spatial_table,
        thresholds=thresholds,
        niche_column="niche_label",
        cell_type_column="cell_type",
        id_column="instance_id",
        distance_column="distance",
    )
    elapsed = time.perf_counter() - t0
    assert len(out.curves) > 0
    assert elapsed < 8.0


def test_builtin_per_niche_cell_type_enrichment_plugin_execution() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 0, 1, 1, 1])
    plugin = registry.get("per_niche_cell_type_enrichment")
    out = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels}),
        n_perms=0,
        enrichment_mode="obs-exp",
    )
    assert out.name == "per_niche_cell_type_enrichment"
    enrichment = out.payload["enrichment_by_niche"]
    # Both niches are internally connected in the sample spatial table.
    assert set(enrichment) == {"0", "1"}
    for frame in enrichment.values():
        # Enrichment is a square cell-type x cell-type matrix.
        assert list(frame.index) == list(frame.columns)
    assert all(section.title.startswith("Cell-Type Enrichment") for section in out.sections)


def test_per_niche_cell_type_enrichment_skips_noise_cluster() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([-1, -1, -1, 1, 1, 1])
    plugin = registry.get("per_niche_cell_type_enrichment")

    skipped = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels}),
        skip_background=True,
        n_perms=0,
    )
    assert set(skipped.payload["enrichment_by_niche"]) == {"1"}

    retained = plugin.run(
        ds,
        PluginContext(state={"niche_labels": labels}),
        skip_background=False,
        n_perms=0,
    )
    assert set(retained.payload["enrichment_by_niche"]) == {"-1", "1"}


def _attribution_dataset() -> InterpretabilityDataset:
    """Sample dataset carrying per-class attention and instance logits."""
    ds = _sample_dataset()
    table = ds.instance_table.copy()
    # Two attention channels, each normalised within its bag.
    table["attention_c0"] = [2 / 3, 1 / 6, 1 / 6, 0.2, 0.3, 0.5]
    table["attention_c1"] = [0.2, 0.5, 0.3, 1 / 6, 1 / 6, 2 / 3]
    table["logit_0"] = [0.5, -0.2, 0.9, 0.1, -0.4, 0.7]
    table["logit_1"] = [-0.3, 0.8, 0.2, 0.6, 0.1, -0.5]
    return InterpretabilityDataset(
        instance_table=table,
        spatial_table=ds.spatial_table,
        id_column=ds.id_column,
        bag_id_column=ds.bag_id_column,
        cell_type_column=ds.cell_type_column,
    )


def test_margin_attribution_plugin_reports_per_niche_intervals() -> None:
    registry = create_builtin_registry()
    ds = _attribution_dataset()
    labels = np.array([0, 0, 1, 1, 0, 1])

    out = registry.get("margin_attribution").run(
        ds, PluginContext(state={"niche_labels": labels}), n_bootstrap=25
    )
    assert out.name == "margin_attribution"
    summary = out.payload["per_niche"]
    # (niche_label, class_index); a binary head summarises the positive class.
    assert set(summary.index) == {(0, 1), (1, 1)}
    for column in ("margin_signed_share", "attention_lift", "prevalence"):
        assert column in summary.columns
        assert f"{column}_lo" in summary.columns


def test_margin_attribution_requires_instance_logits() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()  # has no logit_* columns
    with pytest.raises(ValueError, match="instance logit columns"):
        registry.get("margin_attribution").run(
            ds, PluginContext(state={"niche_labels": np.zeros(6, dtype=int)})
        )


def test_morans_i_plugin_runs_over_the_instance_graph() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 0, 1, 1, 1])

    out = registry.get("morans_i").run(
        ds, PluginContext(state={"niche_labels": labels}), n_perms=10, min_nodes=2
    )
    assert out.name == "morans_i"
    assert set(out.payload["statistic"].index) == {0, 1}
    assert "comp_A" in out.payload["statistic"].columns
    assert (out.payload["qvalue"].to_numpy(dtype=float) >= 0).any()


def test_ripley_plugin_produces_centred_curves() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    labels = np.array([0, 0, 1, 1, 0, 1])

    out = registry.get("ripley").run(
        ds, PluginContext(state={"niche_labels": labels}), n_radii=6, min_count=2
    )
    assert out.name == "ripley"
    curves = out.payload["curves"]
    assert len(out.payload["radii"]) == 6
    assert curves.shape[0] == 6


def test_cluster_agreement_plugin_compares_stored_labelings() -> None:
    registry = create_builtin_registry()
    ds = _sample_dataset()
    table = ds.instance_table.copy()
    table["prior_labels"] = [0, 0, 0, 1, 1, 1]
    ds = InterpretabilityDataset(
        instance_table=table,
        spatial_table=ds.spatial_table,
        id_column=ds.id_column,
        bag_id_column=ds.bag_id_column,
        cell_type_column=ds.cell_type_column,
    )

    out = registry.get("niche_agreement").run(
        ds,
        PluginContext(state={"niche_labels": np.array([0, 0, 0, 1, 1, 1])}),
        label_columns=["prior_labels"],
    )
    metrics = out.payload["pairwise_metrics"]
    # Identical partitions agree perfectly.
    assert metrics["ari"].iloc[0] == pytest.approx(1.0)


def test_cluster_agreement_plugin_requires_a_second_labeling() -> None:
    registry = create_builtin_registry()
    with pytest.raises(ValueError, match="at least one additional labeling"):
        registry.get("niche_agreement").run(
            _sample_dataset(), PluginContext(state={"niche_labels": np.zeros(6, dtype=int)})
        )
