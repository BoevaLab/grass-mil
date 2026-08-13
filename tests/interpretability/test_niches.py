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
    sort_by_niche_order,
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


def test_sort_by_niche_order_keeps_unlisted_rows() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
    ordered = sort_by_niche_order(frame, [2, 0])
    assert list(ordered.index) == [2, 0, 1]


def test_composition_summary_returns_niches_in_dendrogram_order() -> None:
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    # Niches 0 and 2 are tumour-rich; 1 and 3 stroma-rich; -1 is Background.
    profiles = {0: (0.9, 0.1), 1: (0.1, 0.9), 2: (0.85, 0.15), 3: (0.15, 0.85), -1: (0.5, 0.5)}
    for niche, (tumour, stroma) in profiles.items():
        for _ in range(8):
            jitter = rng.normal(scale=0.01)
            rows.append({"comp_tumour": tumour + jitter, "comp_stroma": stroma - jitter})
            labels.append(niche)

    summary = niche_composition_summary(pd.DataFrame(rows), np.array(labels))

    order = list(summary.composition.index)
    assert order[-1] == BACKGROUND_NICHE_ID
    positions = {niche: position for position, niche in enumerate(order)}
    assert abs(positions[0] - positions[2]) == 1
    assert abs(positions[1] - positions[3]) == 1

    # Enrichment and counts follow the same order, so tables line up.
    assert list(summary.enrichment.index) == order
    assert list(summary.niche_counts.index) == order


def test_composition_ordering_can_be_disabled() -> None:
    rows = [{"comp_a": 0.9, "comp_b": 0.1}] * 4 + [{"comp_a": 0.1, "comp_b": 0.9}] * 4
    labels = np.array([2, 2, 0, 0, 1, 1, 3, 3])
    summary = niche_composition_summary(pd.DataFrame(rows), labels, order_by_composition=False)
    assert list(summary.composition.index) == [0, 1, 2, 3]


def test_report_heatmap_labels_niches_by_name() -> None:
    pytest.importorskip("plotly.graph_objects")
    from grass_mil.interpretability.contracts import NicheSummary, ReportBundle
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
