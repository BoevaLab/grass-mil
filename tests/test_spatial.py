"""Spatial statistics, cross-fold replication and the model-free command."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.niches.replication.cli import run_metaclusters
from grass_mil.niches.replication.stages import constrained_linkage, hellinger
from grass_mil.spatial.cli import run_spatial_stats
from grass_mil.spatial.graph_stats import contact_counts, cross_l, enrichment_zscores, morans_i
from grass_mil.spatial.stats import bh, bootstrap_mean
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
    z = enrichment_zscores(labels, _chain(100), 2, n_perms=200, rng=np.random.default_rng(0))
    assert z[0, 0] > 3 and z[0, 1] < -3


def test_morans_i_of_clustered_and_alternating_indicators():
    clustered = np.repeat([1.0, 0.0], 50)[:, None]
    alternating = (np.arange(100) % 2).astype(float)[:, None]
    assert morans_i(clustered, _chain(100))[0] > 0.9
    assert morans_i(alternating, _chain(100))[0] < -0.9


def test_cross_l_is_near_zero_under_complete_spatial_randomness():
    coords = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    radii = np.array([5.0, 10.0])
    labels = np.arange(2000) % 2
    l_aa = cross_l(coords, labels == 0, labels == 0, radii)
    l_ab = cross_l(coords, labels == 0, labels == 1, radii)
    # Edge effects bias L slightly downwards; the self pairs must not bias it upwards.
    assert np.all(np.abs(l_aa) < 0.6) and np.all(np.abs(l_ab) < 0.6)


def test_bh_and_bootstrap():
    assert np.allclose(bh(np.array([0.01, 0.04, np.nan])), [0.02, 0.04, np.nan], equal_nan=True)
    mean, lo, hi = bootstrap_mean([1.0, 2.0, 3.0, np.nan], n_boot=500)
    assert mean == 2.0 and lo <= 2.0 <= hi


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
