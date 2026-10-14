"""Phase 8: classical spatial statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort

pytest.importorskip("scipy")


def _delaunay_edges(coords):
    from scipy.spatial import Delaunay

    tri = Delaunay(coords)
    e = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = int(s[i]), int(s[(i + 1) % 3])
            e.add((min(a, b), max(a, b)))
    return np.array(sorted(e)).T


def test_nhood_zscores_random_vs_segregated():
    from grass_mil.spatial_stats import nhood_enrichment_zscores

    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, (400, 2))
    ei = _delaunay_edges(coords)
    random_labels = rng.integers(0, 3, 400)
    z = nhood_enrichment_zscores(random_labels, ei, n_groups=3, n_perms=100, seed=0)
    assert z.shape == (3, 3) and np.allclose(z, z.T) and np.abs(z).max() < 3.5
    segregated = (coords[:, 0] > 50).astype(int)
    z2 = nhood_enrichment_zscores(segregated, ei, n_groups=3, n_perms=100, seed=0)
    assert z2[0, 0] > 3 and z2[1, 1] > 3 and z2[0, 1] < -3
    assert np.isnan(z2[2]).all()  # absent group


def test_window_composition_and_cellular_neighbourhoods():
    from grass_mil.spatial_stats import cellular_neighbourhoods, window_composition

    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 100, (300, 2))
    labels = (coords[:, 0] > 50).astype(int)
    W = window_composition(coords, labels, n_groups=2, k=10)
    assert W.shape == (300, 2) and np.allclose(W.sum(1), 1.0)
    assert W[labels == 0, 0].mean() > 0.8
    cn, centres = cellular_neighbourhoods([W, W[:50]], n_cn=2, sample_size=100, seed=0, n_init=2)
    assert len(cn) == 2 and cn[0].shape == (300,) and centres.shape == (2, 2)


def test_crossk_csr_vs_clustered_and_contacts():
    from grass_mil.spatial_stats import contact_fractions, cross_k

    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 1000, (2000, 2))
    labels = rng.integers(0, 2, 2000)
    k = cross_k(
        coords, labels, group_names=["T", "X"], reference=["T"], partners=["X"], radii=[50, 100]
    )
    assert abs(k["K_X_r50"] - 1.0) < 0.3 and abs(k["K_X_r100"] - 1.0) < 0.2
    # every X cell placed next to a T cell: strong attraction at short range
    coords2 = coords.copy()
    x_idx = np.where(labels == 1)[0]
    t_idx = rng.choice(np.where(labels == 0)[0], x_idx.size)
    coords2[x_idx] = coords[t_idx] + rng.normal(0, 3, (x_idx.size, 2))
    k2 = cross_k(
        coords2, labels, group_names=["T", "X"], reference=["T"], partners=["X"], radii=[20]
    )
    assert k2["K_X_r20"] > 1.5
    ei = _delaunay_edges(coords2)
    c = contact_fractions(
        labels, ei, group_names=["T", "X"], reference=["T"], partners=["X"], infiltrating=["X"]
    )
    assert 0 <= c["contact_X"] <= 1 and 0 <= c["homotypic"] <= 1 and "infiltration" in c


def test_mann_whitney_table_and_bh():
    from grass_mil.spatial_stats import bh_adjust, mann_whitney_table, patient_mean

    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"a": np.r_[rng.normal(1, 1, 20), rng.normal(0, 1, 20)], "b": rng.normal(size=40)},
        index=[f"r{i}" for i in range(40)],
    )
    y = pd.Series(np.r_[np.ones(20), np.zeros(20)].astype(int), index=df.index)
    T = mann_whitney_table(df, y)
    assert list(T.columns) == [
        "metric",
        "n_pos",
        "n_neg",
        "mean_pos",
        "mean_neg",
        "diff",
        "p",
        "q",
    ]
    assert T.set_index("metric").loc["a", "p"] < 0.05
    q = bh_adjust([0.01, 0.02, 0.5, np.nan])
    assert np.isnan(q[3]) and q[0] <= q[2]
    pm = patient_mean(df, {f"r{i}": f"p{i // 2}" for i in range(40)})
    assert len(pm) == 20


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    pytest.importorskip("torch_geometric")
    import hydra

    from grass_mil.extract import run_extract
    from grass_mil.train import train

    root = tmp_path_factory.mktemp("cohort8")
    c = make_synthetic_cohort(root, n_patients=12, regions_per_patient=2, n_cells=80)
    hydra.utils.instantiate(
        compose_cfg("train", cohort_overrides(c, root / "pre")).data
    ).prepare_data()
    train_out = root / "train"
    train(compose_cfg("train", cohort_overrides(c, train_out) + ["trainer.max_epochs=1"]))
    ecfg = compose_cfg(
        "extract",
        cohort_overrides(c, root / "extract")
        + [
            f"+extract.models.final={train_out / 'checkpoints' / 'last.ckpt'}",
            "extract.device=cpu",
        ],
    )
    return c, Path(run_extract(ecfg)["models"][0])


def test_spatial_stats_cli_finds_planted_niche(cache_dir, tmp_path):
    from grass_mil.spatial_stats.cli import run

    c, model_dir = cache_dir
    cfg = compose_cfg(
        "spatial_stats",
        [
            f"cells_dir={model_dir}",
            f"output_dir={tmp_path / 'spatial'}",
            f"labels.label_file={c.label_path}",
            "labels.label_column=STAS",
            "labels.patient_column=PATIENT",
            "reference=[C]",
            "partners=[D,A]",
            "infiltrating=[D]",
            "nhood.n_perms=50",
            "cn.n_cn=3",
            "cn.sample_size=500",
            "min_per_group=3",
            "crossk.radii=[30,60]",
        ],
    )
    run(cfg)
    out = tmp_path / "spatial"
    for name in (
        "nhood_tests.csv",
        "cn_tests.csv",
        "contact_tests.csv",
        "crossk_tests.csv",
        "report_spatial.md",
    ):
        assert (out / name).exists(), name
    nh = pd.read_csv(out / "nhood_tests.csv").set_index("metric")
    # the planted niche: D cells cluster together (and near C) only in positive regions
    assert nh.loc["D|D", "diff"] > 0 and nh.loc["D|D", "p"] < 0.05
    ck = pd.read_csv(out / "crossk_tests.csv").set_index("metric")
    assert ck.loc["K_D_r30", "mean_pos"] > ck.loc["K_D_r30", "mean_neg"]
