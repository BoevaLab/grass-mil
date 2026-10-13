"""Phase 6: niche discovery (cosine HDBSCAN, relattn margins, tables, co-localisation, pipeline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort

pytest.importorskip("sklearn")


def _blobs(n_per=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, d)) * 6
    X = np.concatenate([c + rng.normal(size=(n_per, d)) for c in centres]).astype(np.float32)
    y = np.repeat(np.arange(3), n_per)
    return X, y


def test_cosine_hdbscan_recovers_blobs_and_assigns_unsampled_rows():
    from sklearn.metrics import adjusted_rand_score

    from grass_mil.interpretability.core.clustering import cosine_hdbscan, fit_niche_clustering

    X, y = _blobs()
    model, labels = cosine_hdbscan(X, n_components=5, min_cluster_size=20, sample_size=150, seed=0)
    assert model.sample_idx.size == 150 and model.n_clusters == 3
    assigned = labels >= 0
    assert assigned.mean() > 0.95
    assert adjusted_rand_score(y[assigned], labels[assigned]) > 0.95
    unsampled = np.setdiff1d(np.arange(len(X)), model.sample_idx)
    assert adjusted_rand_score(y[unsampled], labels[unsampled]) > 0.95
    assert np.array_equal(model.predict(X[model.sample_idx]), model.labels_sample)
    # any registry backend: k-means with the same sample/assign machinery
    km, lab_km = fit_niche_clustering(
        X,
        reduction={"method": "pca", "n_components": 5},
        clustering={"method": "kmeans", "n_clusters": 3},
        sample_size=None,
    )
    assert adjusted_rand_score(y, lab_km) > 0.95 and km.clustering_method == "kmeans"
    assert km.params["clustering"]["n_clusters"] == 3


def test_relattn_formula_and_region_sum_identity():
    from grass_mil.interpretability.niches.margins import relattn_margins

    rng = np.random.default_rng(0)
    rid = np.array(["a"] * 30 + ["b"] * 50)
    LG = rng.normal(size=(80, 2))
    AR = rng.normal(size=(80, 2))
    bias = np.array([0.3, -0.2])
    res = relattn_margins(LG, AR, bias, rid)
    for r, n in (("a", 30), ("b", 50)):
        m = rid == r
        a = np.exp(AR[m] - AR[m].max(0))
        a /= a.sum(0, keepdims=True)
        direct = (a[:, 1] * (LG[m, 1] - bias[1]) - a[:, 0] * (LG[m, 0] - bias[0])) * n
        assert np.allclose(res.margin[m], direct, atol=1e-5)
        assert np.allclose(res.attention[m].sum(0), 1.0, atol=1e-5)
        L = (a * LG[m]).sum(0)  # bag logits = sum A l
        assert res.margin[m].sum() / n == pytest.approx(
            (L[1] - bias[1]) - (L[0] - bias[0]), abs=1e-5
        )


def test_niche_table_and_colocalisation():
    from grass_mil.interpretability.niches.colocalisation import cluster_colocalisation
    from grass_mil.interpretability.niches.tables import niche_table, spearman_concordance

    rng = np.random.default_rng(0)
    n = 600
    rid = np.repeat([f"r{i}" for i in range(6)], 100)
    region_labels = {f"r{i}": i % 2 for i in range(6)}
    labels = rng.integers(0, 3, n)
    labels[:20] = -1
    ct = rng.integers(0, 4, n)
    margin = rng.normal(size=n) + (labels == 2) * 1.0
    att = rng.uniform(size=n)
    T = niche_table(
        labels,
        cell_types=ct,
        region_ids=rid,
        region_labels=region_labels,
        margin=margin,
        attention_pos=att,
        cell_type_names=list("ABCD"),
        n_boot=50,
    )
    assert set(T.cluster) == {"C0", "C1", "C2", "Noise"}
    for col in (
        "share",
        "cores_present",
        "emp_frac_pos",
        "share_auroc",
        "relattn",
        "relattn_lo",
        "relattn_hi",
        "signed_share",
        "sign_consistency",
        "lift",
        "comp_A",
    ):
        assert col in T.columns
    assert T.share.sum() == pytest.approx(1.0)
    assert (
        T.set_index("cluster").loc["C2", "relattn"] > T.set_index("cluster").loc["C0", "relattn"]
    )
    conc = spearman_concordance(T)
    assert conc["n_clusters"] == 3 and "spearman_emp_relattn" in conc

    edges = np.stack([rng.integers(0, n, 3000), rng.integers(0, n, 3000)])
    CO, clusters = cluster_colocalisation(labels, edges)
    assert clusters == [0, 1, 2] and np.allclose(CO, CO.T)
    assert np.abs(CO).max() < 0.5  # random edges: observed ~ expected
    # segregated graph: edges only within cluster -> positive diagonal, negative off-diagonal
    same = np.array(
        [
            (i, j)
            for i in range(n)
            for j in rng.integers(0, n, 5)
            if labels[i] == labels[j] and labels[i] >= 0
        ][:3000]
    ).T
    CO2, _ = cluster_colocalisation(labels, same)
    assert np.all(np.diag(CO2) > 0) and (CO2[~np.eye(3, dtype=bool)] < 0).all()


def test_identity_residual_from_bag_table():
    from grass_mil.interpretability.core.attribution import niche_attribution_summary
    from grass_mil.interpretability.plugins.builtin import _bag_logits_from_table

    rng = np.random.default_rng(0)
    rows = []
    bag_rows = []
    for b in range(3):
        n = 10
        l = rng.normal(size=(n, 2))
        a_logit = rng.normal(size=(n, 2))
        a = np.exp(a_logit - a_logit.max(0))
        a /= a.sum(0)
        L = (a * l).sum(0)
        bag_rows.append({"bag_id": f"b{b}", "logit_0": L[0], "logit_1": L[1]})
        for i in range(n):
            rows.append(
                {
                    "bag_id": f"b{b}",
                    "attention_c0": a[i, 0],
                    "attention_c1": a[i, 1],
                    "logit_0": l[i, 0],
                    "logit_1": l[i, 1],
                }
            )
    table = pd.DataFrame(rows)
    bag_logits = _bag_logits_from_table(
        pd.DataFrame(bag_rows), bag_id_column="bag_id", logit_prefix="logit_"
    )
    assert set(bag_logits) == {"b0", "b1", "b2"}
    res = niche_attribution_summary(
        table,
        np.zeros(len(table), dtype=int),
        attention_columns=["attention_c0", "attention_c1"],
        logit_columns=["logit_0", "logit_1"],
        bag_logits=bag_logits,
        n_bootstrap=10,
    )
    assert res.identity_residual < 1e-6
    assert np.isnan(
        niche_attribution_summary(
            table,
            np.zeros(len(table), dtype=int),
            attention_columns=["attention_c0", "attention_c1"],
            logit_columns=["logit_0", "logit_1"],
            n_bootstrap=10,
        ).identity_residual
    )


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    pytest.importorskip("torch_geometric")
    import hydra

    from grass_mil.extract import run_extract
    from grass_mil.train import train

    root = tmp_path_factory.mktemp("cohort6")
    c = make_synthetic_cohort(root, n_patients=4, regions_per_patient=2, n_cells=80)
    cfg = compose_cfg("train", cohort_overrides(c, root / "pre"))
    hydra.utils.instantiate(cfg.data).prepare_data()
    train_out = root / "train"
    train(compose_cfg("train", cohort_overrides(c, train_out) + ["trainer.max_epochs=1"]))
    ckpt = train_out / "checkpoints" / "last.ckpt"
    ecfg = compose_cfg(
        "extract",
        cohort_overrides(c, root / "extract")
        + [f"+extract.models.final={ckpt}", "extract.device=cpu"],
    )
    result = run_extract(ecfg)
    return Path(result["models"][0]), ckpt


def test_niche_discovery_pipeline_default_and_kmeans(cache_dir, tmp_path):
    from grass_mil.inference.cell_cache import CellCache
    from grass_mil.interpretability.niches.pipeline import run_niche_discovery, write_niche_outputs

    model_dir, ckpt = cache_dir
    cache = CellCache.open(model_dir)
    res = run_niche_discovery(
        cache,
        regions="all" and cache.regions,
        clustering={
            "method": "hdbscan",
            "params": {
                "min_cluster_size": 15,
                "min_samples": 1,
                "cluster_selection_method": "eom",
            },
        },
        sample_size=None,
        n_boot=50,
    )
    assert res.labels.size == sum(cache.n_cells(r) for r in cache.regions)
    assert res.n_clusters >= 1 and res.table.share.sum() == pytest.approx(1.0)
    assert res.colocalisation.shape == (res.n_clusters, res.n_clusters)
    paths = write_niche_outputs(res, tmp_path / "default", title="test", umap={"enabled": False})
    for key in (
        "labels",
        "cluster_model",
        "cells",
        "clusters",
        "report",
        "pca_triptych",
        "clusters_overview",
        "colocalisation",
    ):
        assert key in paths and Path(paths[key]).exists(), key
    cells = pd.read_csv(paths["cells"])
    assert len(cells) == res.labels.size and set(cells.columns) >= {
        "region_id",
        "cluster",
        "cell_type",
        "relattn",
        "x",
        "y",
    }

    km = run_niche_discovery(
        cache,
        regions=cache.regions,
        clustering={"method": "kmeans", "n_clusters": 4},
        reduction={"method": "pca", "n_components": 5},
        sample_size=None,
        n_boot=20,
    )
    assert km.n_clusters == 4 and km.noise_fraction == 0.0

    att = run_niche_discovery(
        cache,
        regions=cache.regions,
        space="attention",
        attention_ckpt=str(ckpt),
        clustering={"method": "kmeans", "n_clusters": 3},
        reduction={"method": "none"},
        sample_size=None,
        n_boot=20,
    )
    assert att.features.shape[1] == 4  # attention hidden_dim of the synthetic model


def test_niches_cli_runs(cache_dir, tmp_path):
    from grass_mil.interpretability.niches_cli import run_niches

    model_dir, _ = cache_dir
    cfg = compose_cfg(
        "niches",
        [
            f"cells_dir={model_dir}",
            f"output_dir={tmp_path / 'niches'}",
            "regions=all",
            "clustering.params.min_cluster_size=15",
            "sample_size=null",
            "n_boot=20",
            "umap.enabled=false",
        ],
    )
    out = run_niches(cfg)
    assert out["n_clusters"] >= 1 and Path(out["paths"]["report"]).exists()
