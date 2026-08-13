from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from grass_mil.contracts import (
    ClusteringResult,
    InterpretabilityDataset,
    ReportBundle,
)
from grass_mil.interpretability.core.reduction import run_reduction
from grass_mil.interpretability.core.transfer import (
    apply_niche_transfer,
    fit_niche_transfer_from_report_bundle,
    load_niche_transfer_bundle,
    save_niche_transfer_bundle,
)


def _sample_instance_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": [f"i{i}" for i in range(10)],
            "bag_id": ["b0"] * 5 + ["b1"] * 5,
            "inst_emb_0": [0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4],
            "inst_emb_1": [0.0, 0.15, 0.25, 0.35, 0.45, 1.0, 1.15, 1.25, 1.35, 1.45],
        }
    )


def _build_report_bundle(
    *,
    dataset: InterpretabilityDataset,
    labels: np.ndarray,
    artifacts_dir: Path,
    cluster_on_pca: bool,
    pca_components: int = 2,
) -> ReportBundle:
    emb = dataset.instance_table[["inst_emb_0", "inst_emb_1"]].to_numpy(dtype=float)
    pca_result = run_reduction("pca", emb, n_components=pca_components, random_state=11)
    clustering = ClusteringResult(
        method="hdbscan",
        labels=np.asarray(labels),
        params={},
        fitted_object=None,
    )
    return ReportBundle(
        dataset=dataset,
        reduction=pca_result,
        niche_feature_reduction=pca_result if cluster_on_pca else None,
        clustering=clustering,
        niche_summary=None,
        plugin_results={},
        artifacts_dir=artifacts_dir,
        metadata={"cluster_on_pca": cluster_on_pca},
    )


def test_niche_transfer_fit_and_apply_matches_direct_pca_knn_reference(tmp_path: Path) -> None:
    table = _sample_instance_table()
    dataset = InterpretabilityDataset(instance_table=table, id_column="instance_id")
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1], dtype=int)
    report_bundle = _build_report_bundle(
        dataset=dataset,
        labels=labels,
        artifacts_dir=tmp_path,
        cluster_on_pca=True,
        pca_components=2,
    )

    bundle = fit_niche_transfer_from_report_bundle(dataset, report_bundle, n_neighbors=3)
    query = table[["instance_id", "inst_emb_1", "inst_emb_0"]].sample(frac=1.0, random_state=3)
    out = apply_niche_transfer(bundle, query, id_column="instance_id")

    query_matrix = query[["inst_emb_0", "inst_emb_1"]].to_numpy(dtype=float)
    ref_knn = KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="minkowski")
    ref_knn.fit(report_bundle.niche_feature_reduction.embedding, labels)  # type: ignore[union-attr]
    ref_features = report_bundle.niche_feature_reduction.fitted_object.transform(  # type: ignore[union-attr]
        query_matrix
    )
    ref_labels = ref_knn.predict(ref_features)
    ref_conf = np.max(ref_knn.predict_proba(ref_features), axis=1)

    assert out["transferred_cluster_label"].tolist() == ref_labels.tolist()
    assert np.allclose(out["transfer_confidence"].to_numpy(dtype=float), ref_conf)


def test_niche_transfer_preserves_noise_label_minus_one(tmp_path: Path) -> None:
    table = _sample_instance_table()
    dataset = InterpretabilityDataset(instance_table=table, id_column="instance_id")
    labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1, 2, 2], dtype=int)
    report_bundle = _build_report_bundle(
        dataset=dataset,
        labels=labels,
        artifacts_dir=tmp_path,
        cluster_on_pca=False,
    )
    bundle = fit_niche_transfer_from_report_bundle(dataset, report_bundle, n_neighbors=1)
    out = apply_niche_transfer(bundle, table[["instance_id", "inst_emb_0", "inst_emb_1"]])

    predicted = out["transferred_cluster_label"].to_numpy(dtype=int)
    assert -1 in set(predicted.tolist())
    assert np.array_equal(predicted, labels)


def test_niche_transfer_bundle_serialization_roundtrip(tmp_path: Path) -> None:
    table = _sample_instance_table()
    dataset = InterpretabilityDataset(instance_table=table, id_column="instance_id")
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1], dtype=int)
    report_bundle = _build_report_bundle(
        dataset=dataset,
        labels=labels,
        artifacts_dir=tmp_path,
        cluster_on_pca=True,
        pca_components=2,
    )
    bundle = fit_niche_transfer_from_report_bundle(dataset, report_bundle, n_neighbors=3)

    bundle_path = tmp_path / "cluster_transfer_bundle.joblib"
    save_niche_transfer_bundle(bundle, bundle_path)
    restored = load_niche_transfer_bundle(bundle_path)

    query = table[["instance_id", "inst_emb_0", "inst_emb_1"]].copy()
    left = apply_niche_transfer(bundle, query)
    right = apply_niche_transfer(restored, query)
    assert left.equals(right)


def test_niche_transfer_query_column_validation_and_reordered_columns(tmp_path: Path) -> None:
    table = _sample_instance_table()
    dataset = InterpretabilityDataset(instance_table=table, id_column="instance_id")
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1], dtype=int)
    report_bundle = _build_report_bundle(
        dataset=dataset,
        labels=labels,
        artifacts_dir=tmp_path,
        cluster_on_pca=False,
    )
    bundle = fit_niche_transfer_from_report_bundle(dataset, report_bundle, n_neighbors=3)

    ordered = table[["instance_id", "inst_emb_0", "inst_emb_1"]]
    reordered = table[["instance_id", "inst_emb_1", "inst_emb_0"]]
    out_ordered = apply_niche_transfer(bundle, ordered)
    out_reordered = apply_niche_transfer(bundle, reordered)
    assert out_ordered.equals(out_reordered)

    with pytest.raises(ValueError, match="missing embedding columns"):
        apply_niche_transfer(bundle, table[["instance_id", "inst_emb_0"]])


def test_niche_transfer_fit_fails_fast_for_missing_inputs(tmp_path: Path) -> None:
    table = _sample_instance_table()
    dataset = InterpretabilityDataset(instance_table=table, id_column="instance_id")
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1], dtype=int)
    valid = _build_report_bundle(
        dataset=dataset,
        labels=labels,
        artifacts_dir=tmp_path,
        cluster_on_pca=True,
        pca_components=2,
    )

    missing_clustering = ReportBundle(
        dataset=dataset,
        reduction=valid.reduction,
        niche_feature_reduction=valid.niche_feature_reduction,
        clustering=None,
        niche_summary=None,
        plugin_results={},
        artifacts_dir=tmp_path,
        metadata={"cluster_on_pca": True},
    )
    with pytest.raises(ValueError, match="missing clustering result"):
        fit_niche_transfer_from_report_bundle(dataset, missing_clustering, n_neighbors=3)

    missing_cluster_pca = ReportBundle(
        dataset=dataset,
        reduction=valid.reduction,
        niche_feature_reduction=None,
        clustering=valid.clustering,
        niche_summary=None,
        plugin_results={},
        artifacts_dir=tmp_path,
        metadata={"cluster_on_pca": True},
    )
    with pytest.raises(ValueError, match="missing niche_feature_reduction"):
        fit_niche_transfer_from_report_bundle(dataset, missing_cluster_pca, n_neighbors=3)

    with pytest.raises(ValueError, match="n_neighbors cannot exceed"):
        fit_niche_transfer_from_report_bundle(dataset, valid, n_neighbors=len(table) + 1)


def test_niche_transfer_apply_fails_for_empty_input(tmp_path: Path) -> None:
    table = _sample_instance_table()
    dataset = InterpretabilityDataset(instance_table=table, id_column="instance_id")
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1], dtype=int)
    report_bundle = _build_report_bundle(
        dataset=dataset,
        labels=labels,
        artifacts_dir=tmp_path,
        cluster_on_pca=False,
    )
    bundle = fit_niche_transfer_from_report_bundle(dataset, report_bundle, n_neighbors=3)
    empty = pd.DataFrame(columns=["instance_id", "inst_emb_0", "inst_emb_1"])
    with pytest.raises(ValueError, match="empty instance table"):
        apply_niche_transfer(bundle, empty)
