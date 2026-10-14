"""Niche naming, the Background label, and composition ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.interpretability.core.biomarkers import niche_composition_summary
from grass_mil.interpretability.core.niches import (
    BACKGROUND_NICHE_ID,
    BACKGROUND_NICHE_NAME,
    is_background,
    niche_display_name,
    niche_display_names,
    order_niches_by_composition,
    relabel_niches_by_composition,
)


def test_background_replaces_the_noise_label() -> None:
    """Density-based clustering's -1 is a named category, not "noise"."""
    assert niche_display_name(BACKGROUND_NICHE_ID) == BACKGROUND_NICHE_NAME
    assert is_background(-1)
    assert is_background(BACKGROUND_NICHE_NAME)
    assert not is_background(0)


def test_niche_names_keep_the_numeric_id_for_joinability() -> None:
    assert niche_display_name(3) == "Niche 3"
    assert niche_display_names([0, 2, -1]) == ["Niche 0", "Niche 2", "Background"]


def test_niches_are_ordered_so_similar_compositions_are_adjacent() -> None:
    # Two tumour-rich niches, two stroma-rich ones, interleaved by id.
    composition = pd.DataFrame(
        {
            "comp_tumour": [0.9, 0.1, 0.85, 0.15],
            "comp_stroma": [0.1, 0.9, 0.15, 0.85],
        },
        index=[0, 1, 2, 3],
    )
    order = order_niches_by_composition(composition)

    assert sorted(order) == [0, 1, 2, 3]
    positions = {niche: position for position, niche in enumerate(order)}
    # The tumour-rich pair (0, 2) must be adjacent, as must the stroma pair.
    assert abs(positions[0] - positions[2]) == 1
    assert abs(positions[1] - positions[3]) == 1


def test_background_is_pinned_last_and_excluded_from_the_tree() -> None:
    """Background is a catch-all; letting it join the tree distorts it."""
    composition = pd.DataFrame(
        {
            "comp_a": [0.9, 0.85, 0.1, 0.5],
            "comp_b": [0.1, 0.15, 0.9, 0.5],
        },
        index=[0, 1, 2, BACKGROUND_NICHE_ID],
    )
    order = order_niches_by_composition(composition)
    assert order[-1] == BACKGROUND_NICHE_ID
    assert set(order[:-1]) == {0, 1, 2}


def test_ordering_degrades_gracefully_on_tiny_or_empty_inputs() -> None:
    assert order_niches_by_composition(pd.DataFrame()) == []

    two = pd.DataFrame({"comp_a": [0.1, 0.9]}, index=[0, 1])
    # Fewer than three leaves: linkage adds nothing, order is preserved.
    assert order_niches_by_composition(two) == [0, 1]

    constant = pd.DataFrame({"comp_a": [0.5] * 4, "comp_b": [0.5] * 4}, index=[0, 1, 2, 3])
    # Correlation distance is undefined here; must still return a full order.
    assert sorted(order_niches_by_composition(constant)) == [0, 1, 2, 3]


def test_relabelling_makes_the_niche_id_the_dendrogram_position() -> None:
    """Niche 1 is first in dendrogram order, Niche 2 second, and so on."""
    rows = []
    labels = []
    # Ids 0 and 2 are tumour-rich, 1 and 3 stroma-rich, deliberately interleaved.
    profiles = {0: (0.9, 0.1), 1: (0.1, 0.9), 2: (0.85, 0.15), 3: (0.15, 0.85)}
    for niche, (tumour, stroma) in profiles.items():
        for _ in range(5):
            rows.append({"comp_tumour": tumour, "comp_stroma": stroma})
            labels.append(niche)
    table = pd.DataFrame(rows)

    relabelled, mapping = relabel_niches_by_composition(table, np.array(labels))

    # Ids are 1-based and contiguous.
    assert sorted(set(relabelled)) == [1, 2, 3, 4]
    # The two tumour-rich niches now hold adjacent ids, as do the stroma pair.
    assert abs(mapping[0] - mapping[2]) == 1
    assert abs(mapping[1] - mapping[3]) == 1


def test_relabelling_keeps_background_out_of_the_numbering() -> None:
    rows = []
    labels = []
    for niche, (a, b) in {0: (0.9, 0.1), 1: (0.1, 0.9), 2: (0.5, 0.5), -1: (0.4, 0.6)}.items():
        for _ in range(4):
            rows.append({"comp_a": a, "comp_b": b})
            labels.append(niche)

    relabelled, mapping = relabel_niches_by_composition(pd.DataFrame(rows), np.array(labels))

    assert mapping[-1] == BACKGROUND_NICHE_ID
    assert sorted(set(relabelled)) == [-1, 1, 2, 3]


def test_relabelling_is_a_no_op_without_composition_columns() -> None:
    table = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    labels = np.array([2, 0, 1])
    relabelled, mapping = relabel_niches_by_composition(table, labels)
    np.testing.assert_array_equal(relabelled, labels)
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_composition_summary_follows_niche_id_order() -> None:
    """Ordering is carried by the ids, not re-derived per summary."""
    rows = []
    labels = []
    for niche, (tumour, stroma) in {1: (0.9, 0.1), 2: (0.1, 0.9), -1: (0.5, 0.5)}.items():
        for _ in range(6):
            rows.append({"comp_tumour": tumour, "comp_stroma": stroma})
            labels.append(niche)

    summary = niche_composition_summary(pd.DataFrame(rows), np.array(labels))

    order = list(summary.composition.index)
    assert order == sorted(order)
    assert list(summary.enrichment.index) == order
    assert list(summary.niche_counts.index) == order


def test_report_heatmap_labels_niches_by_name() -> None:
    pytest.importorskip("plotly.graph_objects")
    from grass_mil.contracts import NicheSummary, ReportBundle
    from grass_mil.interpretability.core.data import InterpretabilityDataset
    from grass_mil.interpretability.reporting.plotly_builders import (
        build_niche_enrichment_heatmap,
    )

    enrichment = pd.DataFrame(
        {"comp_a": [1.0, -1.0], "comp_b": [-1.0, 1.0]}, index=[0, BACKGROUND_NICHE_ID]
    )
    summary = NicheSummary(
        niche_labels=np.array([0, -1]),
        composition=enrichment.copy(),
        enrichment=enrichment,
        niche_counts=pd.Series({0: 1, -1: 1}),
    )
    bundle = ReportBundle(
        dataset=InterpretabilityDataset(
            instance_table=pd.DataFrame({"instance_id": ["a", "b"]}),
            spatial_table=None,
            id_column="instance_id",
            bag_id_column="bag_id",
            cell_type_column="cell_type",
        ),
        reduction=None,
        clustering=None,
        niche_summary=summary,
        plugin_results={},
        niche_feature_reduction=None,
        artifacts_dir=None,
    )

    fig = build_niche_enrichment_heatmap(bundle)
    assert list(fig.data[0].y) == ["Niche 0", "Background"]
    assert "Niche" in fig.layout.title.text


def test_pipeline_exports_niches_already_in_dendrogram_order(tmp_path) -> None:
    """The relabelling happens once, in the pipeline, and reaches the exports."""
    import pandas as pd

    from grass_mil.contracts import InterpretabilityDataset
    from grass_mil.interpretability.pipeline import run_interpretability_pipeline

    rng = np.random.default_rng(0)
    rows = []
    # Four well-separated composition profiles; two tumour-rich, two stroma-rich.
    for tumour, stroma in [(0.95, 0.05), (0.05, 0.95), (0.85, 0.15), (0.15, 0.85)]:
        for _ in range(12):
            jitter = rng.normal(scale=0.005)
            rows.append(
                {
                    "comp_tumour": tumour + jitter,
                    "comp_stroma": stroma - jitter,
                    "inst_emb_0": tumour + jitter,
                    "inst_emb_1": stroma - jitter,
                }
            )
    table = pd.DataFrame(rows)
    table.insert(0, "instance_id", [f"i{i}" for i in range(len(table))])
    table["bag_id"] = "b0"

    dataset = InterpretabilityDataset(
        instance_table=table,
        spatial_table=None,
        id_column="instance_id",
        bag_id_column="bag_id",
    )
    bundle = run_interpretability_pipeline(
        dataset,
        {
            "reduction": {"enabled": True, "method": "pca", "params": {"n_components": 2}},
            "clustering": {
                "enabled": True,
                "method": "agglomerative",
                "params": {"n_clusters": 4, "linkage": "ward"},
            },
            "plugins": {"enabled": []},
        },
        artifacts_dir=tmp_path / "artifacts",
    )

    labels = bundle.clustering.labels
    # Ids are 1-based after relabelling, and the summary is already ordered.
    assert sorted(set(labels)) == [1, 2, 3, 4]
    assert list(bundle.niche_summary.composition.index) == [1, 2, 3, 4]

    # Compositionally similar niches carry adjacent ids. Leaf order groups the
    # pairs; it does not sort by magnitude within a pair, so adjacency -- not
    # monotonicity -- is the property to assert.
    tumour = bundle.niche_summary.composition["comp_tumour"].to_numpy()
    tumour_rich = sorted(np.argsort(tumour)[-2:])
    stroma_rich = sorted(np.argsort(tumour)[:2])
    assert tumour_rich[1] - tumour_rich[0] == 1
    assert stroma_rich[1] - stroma_rich[0] == 1

    exported = pd.read_csv(tmp_path / "artifacts" / "niche_labels.csv")
    assert sorted(exported["niche_label"].unique()) == [1, 2, 3, 4]
