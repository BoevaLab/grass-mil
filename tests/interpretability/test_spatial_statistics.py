"""Contracts for Moran's I, Ripley's cross-L, and cross-space agreement.

Each statistic is checked against a case with a known answer rather than
against a golden value, so the tests state what the statistic means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.interpretability.core.agreement import (
    niche_jaccard_matrix,
    compute_niche_agreement,
)
from grass_mil.interpretability.tier2.autocorrelation import (
    diff_morans_i_vs_reference,
    morans_i,
    run_morans_i,
)
from grass_mil.interpretability.tier2.ripley import (
    aggregate_ripley,
    resolve_radii,
    ripley_cross_l,
)


def _chain_adjacency(n: int):
    from scipy.sparse import coo_matrix

    rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    return coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()


def test_morans_i_is_high_for_a_smooth_field_and_low_when_shuffled() -> None:
    n = 60
    adjacency = _chain_adjacency(n)

    smooth = np.arange(n, dtype=float)
    clustered = morans_i(smooth, adjacency)
    assert clustered > 0.9, "a monotone field along a chain is strongly autocorrelated"

    rng = np.random.default_rng(0)
    shuffled = np.mean([morans_i(rng.permutation(smooth), adjacency) for _ in range(50)])
    assert abs(shuffled) < 0.2, "permuting the field destroys autocorrelation"


def test_morans_i_is_negative_for_an_alternating_field() -> None:
    n = 60
    values = np.array([i % 2 for i in range(n)], dtype=float)
    assert morans_i(values, _chain_adjacency(n)) < -0.9


def test_morans_i_is_undefined_for_a_constant_feature() -> None:
    assert np.isnan(morans_i(np.ones(10), _chain_adjacency(10)))


def _autocorrelation_tables(n_per_group: int = 30):
    rows = []
    edges = []
    for group in ("a", "b"):
        for i in range(n_per_group):
            node = f"{group}_{i}"
            # Group a carries a smooth gradient; group b alternates.
            value = float(i) if group == "a" else float(i % 2)
            rows.append({"instance_id": node, "niche_label": group, "feature": value})
            if i + 1 < n_per_group:
                edges.append({"source_id": node, "target_id": f"{group}_{i + 1}"})
    return pd.DataFrame(rows), pd.DataFrame(edges)


def test_run_morans_i_separates_smooth_from_alternating_groups() -> None:
    nodes, edges = _autocorrelation_tables()
    result = run_morans_i(
        nodes,
        edges,
        feature_columns=["feature"],
        group_column="niche_label",
        n_perms=50,
        random_state=0,
    )
    assert result.statistic.loc["a", "feature"] > 0.8
    assert result.statistic.loc["b", "feature"] < -0.8
    # Both are far from the permutation null, so both are significant.
    assert (result.pvalue["feature"] < 0.1).all()
    assert (result.qvalue["feature"] >= result.pvalue["feature"]).all()
    assert result.n_nodes.loc["a"] == 30


def test_run_morans_i_skips_groups_below_the_size_threshold() -> None:
    nodes = pd.DataFrame(
        {
            "instance_id": ["n0", "n1", "n2"],
            "niche_label": ["small", "small", "other"],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    edges = pd.DataFrame({"source_id": ["n0"], "target_id": ["n1"]})
    result = run_morans_i(
        nodes, edges, feature_columns=["feature"], group_column="niche_label", n_perms=5
    )
    assert np.isnan(result.statistic.loc["small", "feature"])


def test_diff_morans_i_reports_a_z_statistic_against_the_reference() -> None:
    nodes, edges = _autocorrelation_tables()
    nodes["niche_label"] = nodes["niche_label"].map({"a": 0, "b": -1})
    result = run_morans_i(
        nodes,
        edges,
        feature_columns=["feature"],
        group_column="niche_label",
        n_perms=50,
        random_state=0,
    )
    diff = diff_morans_i_vs_reference(result, reference_group=-1)
    assert list(diff.index) == [0]
    assert diff.loc[0, "feature"] > 0

    with pytest.raises(ValueError, match="not present"):
        diff_morans_i_vs_reference(result, reference_group=99)


def _poisson_points(n: int, seed: int, size: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, size, size=(n, 2))


def test_ripley_l_is_near_zero_for_a_poisson_pattern() -> None:
    coords = _poisson_points(600, seed=0)
    labels = ["x"] * len(coords)
    radii = resolve_radii(coords, n_radii=20, max_fraction=0.15)

    result = ripley_cross_l(coords, labels, pairs=[("x", "x")], radii=radii)
    curve = result.curves[("x", "x")]
    # Complete spatial randomness gives L(r) - r == 0 up to sampling noise.
    assert np.nanmax(np.abs(curve)) < 4.0


def test_ripley_l_is_positive_for_a_clustered_pattern() -> None:
    rng = np.random.default_rng(1)
    centres = rng.uniform(10.0, 90.0, size=(8, 2))
    coords = np.vstack([centre + rng.normal(scale=1.5, size=(60, 2)) for centre in centres])
    labels = ["x"] * len(coords)
    radii = resolve_radii(coords, n_radii=20, max_fraction=0.15)

    curve = ripley_cross_l(coords, labels, pairs=[("x", "x")], radii=radii).curves[("x", "x")]
    assert np.nanmax(curve) > 5.0, "tight clumps must show strong positive clustering"


def test_ripley_self_pair_correction_matches_a_hand_computed_case() -> None:
    # Four points on a unit square; every pairwise distance <= sqrt(2).
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    radii = np.array([2.0])
    result = ripley_cross_l(coords, ["x"] * 4, pairs=[("x", "x")], radii=radii)

    # With r beyond every distance: counts = 16, minus 4 self pairs = 12,
    # denominator 4*3 = 12, area = 1 -> K = 1, L = sqrt(1/pi) - 2.
    expected = np.sqrt(1.0 / np.pi) - 2.0
    assert result.curves[("x", "x")][0] == pytest.approx(expected)


def test_aggregate_ripley_is_region_equal_and_honours_min_count() -> None:
    frames = []
    for group in range(3):
        coords = _poisson_points(80, seed=group)
        frames.append(
            pd.DataFrame(
                {
                    "center_x": coords[:, 0],
                    "center_y": coords[:, 1],
                    "cell_type": ["x"] * 78 + ["rare"] * 2,
                    "region": f"r{group}",
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)

    result = aggregate_ripley(
        table,
        label_column="cell_type",
        group_column="region",
        pairs=[("x", "x"), ("rare", "rare")],
        n_radii=10,
        min_count=5,
    )
    assert result.n_groups[("x", "x")] == 3
    # `rare` never reaches min_count, so it contributes no curve at all.
    assert result.n_groups[("rare", "rare")] == 0
    assert np.isnan(result.curves[("rare", "rare")]).all()


def test_aggregate_ripley_radius_grid_does_not_depend_on_group_order() -> None:
    """The legacy grid came from whichever region was seen first."""
    frames = []
    for group, scale in enumerate((10.0, 100.0, 50.0)):
        coords = _poisson_points(40, seed=group, size=scale)
        frames.append(
            pd.DataFrame(
                {
                    "center_x": coords[:, 0],
                    "center_y": coords[:, 1],
                    "cell_type": ["x"] * len(coords),
                    "region": f"r{group}",
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    reversed_table = pd.concat(frames[::-1], ignore_index=True)

    forward = aggregate_ripley(table, label_column="cell_type", group_column="region", n_radii=8)
    backward = aggregate_ripley(
        reversed_table, label_column="cell_type", group_column="region", n_radii=8
    )
    np.testing.assert_allclose(forward.radii, backward.radii)

    with pytest.raises(ValueError, match="Unsupported radius_source"):
        aggregate_ripley(
            table, label_column="cell_type", group_column="region", radius_source="nonsense"
        )


def test_cluster_agreement_is_one_for_identical_and_near_zero_for_independent() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=300)
    independent = rng.integers(0, 4, size=300)

    result = compute_niche_agreement(
        {"encoder": labels, "same": labels.copy(), "random": independent}
    )
    metrics = result.pairwise_metrics
    assert metrics.loc[("encoder", "same"), "ari"] == pytest.approx(1.0)
    assert abs(metrics.loc[("encoder", "random"), "ari"]) < 0.1
    # A relabelled partition IS the same partition: these indices are
    # chance-corrected and invariant to how the niches are numbered.
    relabelled = compute_niche_agreement({"a": labels, "b": (labels + 1) % 4})
    assert relabelled.pairwise_metrics.loc[("a", "b"), "ari"] == pytest.approx(1.0)


def test_cluster_agreement_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="at least two labelings"):
        compute_niche_agreement({"only": [0, 1, 2]})
    with pytest.raises(ValueError, match="same instances"):
        compute_niche_agreement({"a": [0, 1, 2], "b": [0, 1]})
    with pytest.raises(ValueError, match="Unsupported agreement metric"):
        compute_niche_agreement({"a": [0, 1], "b": [1, 0]}, metrics=["nonsense"])


def test_cluster_jaccard_matrix_reports_overlap_fractions() -> None:
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 1, 1])
    matrix = niche_jaccard_matrix(a, b)

    # Niche 0 of `a` = {0,1}; niche 1 of `b` = {1,2,3}; overlap {1}, union 4.
    assert matrix.loc[0, 1] == pytest.approx(0.25)
    # Niche 1 of `a` = {2,3} is fully inside niche 1 of `b`.
    assert matrix.loc[1, 1] == pytest.approx(2 / 3)

    with pytest.raises(ValueError, match="must align"):
        niche_jaccard_matrix(np.array([0, 1]), np.array([0, 1, 2]))
