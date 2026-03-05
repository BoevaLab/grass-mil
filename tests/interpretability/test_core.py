from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.interpretability.core.biomarkers import (
    cluster_attention_summary,
    cluster_biomarker_summary,
)
from src.interpretability.core.clustering import run_clustering
from src.interpretability.core.reduction import run_reduction


def test_reduction_and_clustering_deterministic() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 6))
    red1 = run_reduction("pca", x, n_components=2, random_state=42)
    red2 = run_reduction("pca", x, n_components=2, random_state=42)
    assert np.allclose(red1.embedding, red2.embedding)

    cl = run_clustering("agglomerative", red1.embedding, n_clusters=3)
    assert cl.labels.shape[0] == 50
    assert sorted(set(cl.labels.tolist())) == [0, 1, 2]


def test_cluster_biomarker_and_attention_summary() -> None:
    table = pd.DataFrame(
        {
            "instance_id": [f"i{i}" for i in range(6)],
            "bag_id": ["b0", "b0", "b0", "b1", "b1", "b1"],
            "cell_type": ["A", "A", "B", "A", "B", "B"],
            "comp_A": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            "comp_B": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "attention": [0.7, 0.2, 0.1, 0.1, 0.3, 0.6],
            "score": [0.9, 0.8, 0.4, 0.2, 0.3, 0.7],
        }
    )
    labels = np.array([0, 0, 1, 1, 1, 0])
    bio = cluster_biomarker_summary(table, labels, cell_type_column="cell_type")
    assert "comp_A" in bio.composition.columns
    assert bio.cluster_counts.sum() == len(table)

    att = cluster_attention_summary(
        table,
        labels,
        attention_column="attention",
        score_column="score",
        bag_id_column="bag_id",
        cell_type_column="cell_type",
    )
    assert att.weighted_scores is not None
    assert att.attention_lift_present is not None


def test_cluster_biomarker_requires_composition_columns() -> None:
    table = pd.DataFrame(
        {
            "instance_id": ["i0", "i1"],
            "cell_type": ["A", "B"],
        }
    )
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="No composition columns found"):
        cluster_biomarker_summary(table, labels, cell_type_column="cell_type")


def test_cluster_biomarker_variance_estimator_is_tunable() -> None:
    table = pd.DataFrame(
        {
            "instance_id": ["i0", "i1", "i2"],
            "comp_A": [0.0, 1.0, 2.0],
            "comp_B": [2.0, 1.0, 0.0],
        }
    )
    labels = np.array([0, 1, 1])

    unbiased = cluster_biomarker_summary(table, labels, variance_estimator="unbiased")
    biased = cluster_biomarker_summary(table, labels, variance_estimator="biased")

    assert unbiased.enrichment.loc[0, "comp_A"] == pytest.approx(-1.0)
    assert biased.enrichment.loc[0, "comp_A"] == pytest.approx(-1.224744871391589)
    assert abs(unbiased.enrichment.loc[0, "comp_A"]) < abs(biased.enrichment.loc[0, "comp_A"])


def test_cluster_biomarker_variance_estimator_rejects_unknown_values() -> None:
    table = pd.DataFrame(
        {
            "instance_id": ["i0", "i1"],
            "comp_A": [1.0, 0.0],
            "comp_B": [0.0, 1.0],
        }
    )
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="Unsupported variance_estimator"):
        cluster_biomarker_summary(table, labels, variance_estimator="invalid_mode")


def test_hdbscan_clustering_forwards_selection_and_density_knobs(monkeypatch) -> None:
    class _FakeHDBSCAN:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def fit_predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0],), dtype=int)

    import sklearn.cluster as sklearn_cluster

    monkeypatch.setattr(sklearn_cluster, "HDBSCAN", _FakeHDBSCAN, raising=False)
    x = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=float)
    out = run_clustering(
        "hdbscan",
        x,
        min_cluster_size=42,
        min_samples=11,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.15,
        alpha=1.2,
        metric="manhattan",
        allow_single_cluster=True,
        n_jobs=3,
    )
    assert out.labels.shape[0] == x.shape[0]
    assert out.method == "hdbscan"
    assert out.fitted_object.kwargs["min_cluster_size"] == 42
    assert out.fitted_object.kwargs["min_samples"] == 11
    assert out.fitted_object.kwargs["cluster_selection_method"] == "leaf"
    assert out.fitted_object.kwargs["cluster_selection_epsilon"] == pytest.approx(0.15)
    assert out.fitted_object.kwargs["alpha"] == pytest.approx(1.2)
    assert out.fitted_object.kwargs["metric"] == "manhattan"
    assert out.fitted_object.kwargs["allow_single_cluster"] is True
    assert out.fitted_object.kwargs["n_jobs"] == 3
