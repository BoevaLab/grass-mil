"""Cell-level tier-2 analyses.

The instance-level analyses treat a whole ego-graph as one node, so they measure
how neighbourhood labels relate. These measure the cells inside those graphs,
which is the unit the legacy NSCLC reports use.
"""

import pandas as pd
import pytest

from grass_mil.interpretability.tier2.cell_level import (
    add_cell_type_indicators,
    attach_niche_labels_to_cells,
    differential_cell_type_enrichment,
    per_niche_cell_type_enrichment,
    per_niche_cell_type_moran,
)


def _two_niche_cells():
    """Two niches, each an 8-cell chain.

    Niche 1 is segregated (AAAA|BBBB), niche 2 alternates (ABABABAB), so the
    two should give opposite-signed same-type enrichment and Moran's I.
    """
    rows, edges = [], []
    for niche, types in ((1, list("AAAABBBB")), (2, list("ABABABAB"))):
        ids = [f"n{niche}_{i}" for i in range(8)]
        for i, (cid, ct) in enumerate(zip(ids, types)):
            rows.append(
                {
                    "cell_id": cid,
                    "instance_id": f"inst{niche}",
                    "cell_type": ct,
                    "x": float(i),
                    "y": 0.0,
                }
            )
        for i in range(7):
            edges.append({"source_id": ids[i], "target_id": ids[i + 1], "distance": 1.0})
            edges.append({"source_id": ids[i + 1], "target_id": ids[i], "distance": 1.0})
    return pd.DataFrame(rows), pd.DataFrame(edges)


def test_cells_inherit_the_niche_of_their_ego_graph():
    cells, _ = _two_niche_cells()
    out = attach_niche_labels_to_cells(cells, [1, 2], ["inst1", "inst2"])
    assert out.loc[out["instance_id"] == "inst1", "niche_label"].eq(1).all()
    assert out.loc[out["instance_id"] == "inst2", "niche_label"].eq(2).all()
    # Every cell is labelled: this join is what makes per-niche analysis possible.
    assert out["niche_label"].notna().all()


def test_attach_rejects_misaligned_labels():
    cells, _ = _two_niche_cells()
    with pytest.raises(ValueError):
        attach_niche_labels_to_cells(cells, [1], ["inst1", "inst2"])


def test_one_hot_indicators_cover_every_type():
    cells, _ = _two_niche_cells()
    out, cols = add_cell_type_indicators(cells)
    assert cols == ["ct_A", "ct_B"]
    assert out[cols].sum(axis=1).eq(1.0).all()


def test_enrichment_separates_segregated_from_alternating_niches():
    """The statistic must reflect the cells, not a per-niche summary label.

    Both niches have the same composition (4 A, 4 B); they differ only in how
    the cells are arranged. An analysis that saw one label per ego-graph could
    not tell them apart.
    """
    cells, edges = _two_niche_cells()
    cells = attach_niche_labels_to_cells(cells, [1, 2], ["inst1", "inst2"])
    result = per_niche_cell_type_enrichment(cells, edges, n_perms=0, min_cells=4)

    assert set(result.enrichment) == {1, 2}
    segregated = result.enrichment[1]
    alternating = result.enrichment[2]
    # Segregated: same-type contacts dominate. Alternating: cross-type do.
    assert segregated.loc["A", "A"] > segregated.loc["A", "B"]
    assert alternating.loc["A", "B"] > alternating.loc["A", "A"]


def test_differential_subtracts_the_reference_niche():
    cells, edges = _two_niche_cells()
    cells = attach_niche_labels_to_cells(cells, [1, 2], ["inst1", "inst2"])
    result = per_niche_cell_type_enrichment(cells, edges, n_perms=0, min_cells=4)
    diff = differential_cell_type_enrichment(result, reference=1)
    assert set(diff) == {2}
    expected = result.enrichment[2] - result.enrichment[1]
    pd.testing.assert_frame_equal(diff[2], expected)


def test_moran_is_higher_where_a_type_is_spatially_clustered():
    cells, edges = _two_niche_cells()
    cells = attach_niche_labels_to_cells(cells, [1, 2], ["inst1", "inst2"])
    result = per_niche_cell_type_moran(cells, edges, n_perms=0, min_nodes=3)
    stat = result.statistic
    assert stat.shape[1] == 2  # one column per cell type
    # Segregated niche clusters each type; the alternating one anti-correlates.
    assert stat.loc[1, "ct_A"] > stat.loc[2, "ct_A"]
    assert stat.loc[2, "ct_A"] < 0
