"""Spatial statistics, cross-fold replication and the model-free command."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.niches.replication.cli import run_metaclusters
from grass_mil.niches.replication.stages import constrained_linkage, hellinger
from grass_mil.spatial.cli import run_spatial_stats
from grass_mil.spatial.graph_stats import contact_counts, cross_l, enrichment, morans_i
from grass_mil.spatial.stats import bh, bootstrap_mean, group_tests
from tests.helpers.cohort import compose_cfg


def _chain(n: int):
    return np.stack([np.arange(n - 1), np.arange(1, n)])


def test_contact_counts_count_edge_ends():
    labels = np.array([0, 0, 1, 1])
    counts = contact_counts(labels, _chain(4), 2)
    assert counts.tolist() == [[2, 1], [1, 2]]
    assert counts.sum() == 2 * 3


def test_enrichment_is_positive_for_segregated_labels():
    labels = np.repeat([0, 1], 50)
    z, ratio = enrichment(labels, _chain(100), 2, n_perms=200, rng=np.random.default_rng(0))
    assert z[0, 0] > 3 and z[0, 1] < -3
    assert ratio[0, 0] > 0 > ratio[0, 1]


def test_morans_i_moments_match_label_permutations():
    rng = np.random.default_rng(0)
    clustered = np.repeat([1.0, 0.0], 50)[:, None]
    alternating = (np.arange(100) % 2).astype(float)[:, None]
    assert morans_i(clustered, _chain(100))[0][0] > 0.9
    assert morans_i(alternating, _chain(100))[0][0] < -0.9
    values = np.eye(3)[rng.integers(0, 3, 60)]
    edges = np.stack([rng.integers(0, 60, 150), rng.integers(0, 60, 150)])
    edges = np.unique(np.sort(edges[:, edges[0] != edges[1]], axis=0), axis=1)
    _, expected, var = morans_i(values, edges)
    null = np.stack([morans_i(values[rng.permutation(60)], edges)[0] for _ in range(4000)])
    assert np.allclose(null.mean(0), expected, atol=0.01)
    assert np.allclose(null.var(0), var, rtol=0.1)


def test_cross_l_is_zero_under_random_labelling_of_clustered_points():
    rng = np.random.default_rng(0)
    radii = np.array([5.0, 20.0, 60.0])
    blobs = np.concatenate([rng.normal(30, 8, (150, 2)), rng.normal(170, 8, (150, 2))])
    curves = [cross_l(blobs, rng.integers(0, 2, 300), [(0, 0), (0, 1)], radii) for _ in range(50)]
    # Neither the clustering of the points nor the window shows up as an effect.
    assert np.all(np.abs(np.mean(curves, axis=0)) < 0.5)
    segregated = cross_l(blobs, np.repeat([0, 1], 150), [(0, 0), (0, 1)], radii)
    assert segregated[0, -1] > 10 and segregated[1, -1] < -10


def test_bh_bootstrap_and_group_tests():
    assert np.allclose(bh(np.array([0.01, 0.04, np.nan])), [0.02, 0.04, np.nan], equal_nan=True)
    mean, lo, hi = bootstrap_mean([1.0, 2.0, 3.0, np.nan], n_boot=500)
    assert mean == 2.0 and lo <= 2.0 <= hi
    # Resampling patients: three units of one patient move together.
    values = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    _, lo_units, hi_units = bootstrap_mean(values, n_boot=2000)
    _, lo_patients, hi_patients = bootstrap_mean(values, clusters=list("aaabcd"), n_boot=2000)
    assert hi_patients - lo_patients > hi_units - lo_units
    assert np.isnan(bootstrap_mean(values, clusters=["a"] * 6)[1])  # one patient: no interval
    frame = pd.DataFrame({"x": [3.0, 4.0, 5.0, 0.0, 1.0, 2.0]}, index=list("abcdef"))
    table = group_tests(frame, pd.Series([True] * 3 + [False] * 3, index=list("abcdef")))
    assert table.loc[0, "auroc"] == 1.0 and table.loc[0, "p"] < 0.2


def test_linkage_never_merges_two_niches_of_one_model():
    comp = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
    groups = constrained_linkage(hellinger(comp), ["m1", "m1", "m2"], threshold=0.5)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_metaclusters_command(cells_dir, tmp_path):
    cfg = compose_cfg(
        "metaclusters",
        [
            f"cells_dir={cells_dir}",
            "+clustering.min_cluster_size=20",
            "match.min_models=2",
            "basis.k=3",
            "n_boot=50",
        ],
        tmp_path,
    )
    summary = run_metaclusters(cfg)
    assert summary["models"] == ["fold0", "fold1", "fold2"]
    table = pd.read_csv(tmp_path / "metaclusters.csv")
    assert table["metacluster"].str.startswith("M").all()
    assert (tmp_path / "transfer" / "agreement.csv").exists()
    assert len(pd.read_csv(tmp_path / "basis" / "groups.csv")) == 3


def test_metaclusters_rejects_unknown_stages(cells_dir, tmp_path):
    cfg = compose_cfg("metaclusters", [f"cells_dir={cells_dir}", "stage=everything"], tmp_path)
    with pytest.raises(ValueError, match="stage"):
        run_metaclusters(cfg)


def test_spatial_stats_command(cells_dir, tmp_path):
    cfg = compose_cfg(
        "spatial_stats",
        [
            f"cells_dir={cells_dir / 'fold0'}",
            "reference=[D]",
            "n_perms=20",
            "neighbourhoods.n=3",
            "min_patients=2",
        ],
        tmp_path,
    )
    run_spatial_stats(cfg)
    tests = pd.read_csv(tmp_path / "cross_l_tests.csv").set_index("metric")
    # The planted niche: D cells cluster in positive regions only.
    assert tests.loc["D->D@20", "mean_pos"] > tests.loc["D->D@20", "mean_neg"]
    for name in ("enrichment", "neighbourhoods", "contacts"):
        assert (tmp_path / f"{name}_tests.csv").exists()
