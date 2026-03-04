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
