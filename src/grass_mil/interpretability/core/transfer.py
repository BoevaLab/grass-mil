from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from grass_mil.interpretability.contracts import (
    NicheTransferBundle,
    InterpretabilityDataset,
    ReportBundle,
)
from grass_mil.interpretability.core.data import extract_embedding_set

_TRANSFER_BUNDLE_SCHEMA_VERSION = 1


def fit_niche_transfer_from_report_bundle(
    dataset: InterpretabilityDataset,
    report_bundle: ReportBundle,
    *,
    embedding_prefixes: Iterable[str] = ("inst_emb_", "emb_", "graph_emb_"),
    n_neighbors: int = 15,
    weights: str = "uniform",
    metric: str = "minkowski",
) -> NicheTransferBundle:
    if report_bundle.clustering is None:
        raise ValueError("Cannot fit niche transfer: report bundle is missing clustering result.")

    emb_set = extract_embedding_set(dataset, embedding_prefixes=tuple(embedding_prefixes))
    labels = np.asarray(report_bundle.clustering.labels)
    if labels.ndim != 1:
        raise ValueError("Cannot fit niche transfer: clustering labels must be 1D.")
    if emb_set.matrix.shape[0] != labels.shape[0]:
        raise ValueError(
            "Cannot fit niche transfer: embedding rows and clustering labels length mismatch."
        )
    if emb_set.matrix.shape[0] == 0:
        raise ValueError("Cannot fit niche transfer on an empty instance table.")

    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("Cannot fit niche transfer: n_neighbors must be >= 1.")
    if n_neighbors > int(labels.shape[0]):
        raise ValueError(
            "Cannot fit niche transfer: n_neighbors cannot exceed number of training rows."
        )

    cluster_on_pca = bool(report_bundle.metadata.get("cluster_on_pca", False))
    pca_model: Any = None
    train_features = np.asarray(emb_set.matrix)
    if cluster_on_pca:
        if report_bundle.niche_feature_reduction is None:
            raise ValueError(
                "Cannot fit niche transfer: cluster_on_pca=true but "
                "report bundle is missing niche_feature_reduction."
            )
        pca_result = report_bundle.niche_feature_reduction
        pca_model = pca_result.fitted_object
        if pca_model is None or not hasattr(pca_model, "transform"):
            raise ValueError(
                "Cannot fit niche transfer: niche_feature_reduction fitted object "
                "is missing a transform(...) method."
            )
        train_features = np.asarray(pca_result.embedding)
        if train_features.ndim != 2:
            raise ValueError(
                "Cannot fit niche transfer: niche_feature_reduction embedding must be 2D."
            )
        if train_features.shape[0] != labels.shape[0]:
            raise ValueError(
                "Cannot fit niche transfer: niche_feature_reduction rows and clustering "
                "labels length mismatch."
            )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    knn.fit(train_features, labels)
    label_values = np.asarray(knn.classes_)
    metadata = {
        "schema_version": _TRANSFER_BUNDLE_SCHEMA_VERSION,
        "rows": int(labels.shape[0]),
        "n_neighbors": int(n_neighbors),
        "weights": str(weights),
        "metric": str(metric),
        "cluster_on_pca": bool(cluster_on_pca),
        "cluster_method": report_bundle.clustering.method,
    }
    return NicheTransferBundle(
        embedding_columns=list(emb_set.feature_columns),
        id_column=str(dataset.id_column),
        cluster_on_pca=cluster_on_pca,
        pca_model=pca_model,
        knn_model=knn,
        label_values=label_values,
        metadata=metadata,
    )


def apply_niche_transfer(
    bundle: NicheTransferBundle,
    instance_table: pd.DataFrame,
    *,
    id_column: str | None = None,
    output_label_column: str = "transferred_cluster_label",
    output_confidence_column: str = "transfer_confidence",
) -> pd.DataFrame:
    if len(instance_table) == 0:
        raise ValueError("Cannot apply niche transfer to an empty instance table.")

    resolved_id_column = str(id_column or bundle.id_column)
    if resolved_id_column not in instance_table.columns:
        raise ValueError(f"Cannot apply niche transfer: missing id column {resolved_id_column!r}.")

    missing_embeddings = [
        col for col in bundle.embedding_columns if str(col) not in instance_table.columns
    ]
    if missing_embeddings:
        preview = ", ".join(sorted(missing_embeddings)[:8])
        raise ValueError(
            "Cannot apply niche transfer: instance table is missing embedding columns: "
            f"{preview}."
        )

    if bundle.knn_model is None or not hasattr(bundle.knn_model, "predict"):
        raise ValueError(
            "Cannot apply niche transfer: transfer bundle is missing a fitted kNN model."
        )

    query_matrix = instance_table[bundle.embedding_columns].to_numpy(dtype=float)
    query_features = np.asarray(query_matrix)
    if bundle.pca_model is not None:
        if not hasattr(bundle.pca_model, "transform"):
            raise ValueError(
                "Cannot apply niche transfer: transfer bundle PCA model is missing "
                "transform(...)."
            )
        query_features = np.asarray(bundle.pca_model.transform(query_features))

    predicted = np.asarray(bundle.knn_model.predict(query_features))
    confidence = np.full((predicted.shape[0],), np.nan, dtype=float)
    if hasattr(bundle.knn_model, "predict_proba"):
        proba = np.asarray(bundle.knn_model.predict_proba(query_features))
        if proba.ndim == 2 and proba.shape[0] == predicted.shape[0] and proba.shape[1] > 0:
            confidence = np.max(proba, axis=1).astype(float, copy=False)

    return pd.DataFrame(
        {
            resolved_id_column: instance_table[resolved_id_column].tolist(),
            output_label_column: predicted.tolist(),
            output_confidence_column: confidence.tolist(),
        }
    )


def save_niche_transfer_bundle(bundle: NicheTransferBundle, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _TRANSFER_BUNDLE_SCHEMA_VERSION,
        "embedding_columns": list(bundle.embedding_columns),
        "id_column": str(bundle.id_column),
        "cluster_on_pca": bool(bundle.cluster_on_pca),
        "pca_model": bundle.pca_model,
        "knn_model": bundle.knn_model,
        "label_values": np.asarray(bundle.label_values),
        "metadata": dict(bundle.metadata),
    }
    joblib.dump(payload, output_path)


def load_niche_transfer_bundle(path: str | Path) -> NicheTransferBundle:
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Invalid transfer bundle format: expected dictionary payload.")

    schema_version = payload.get("schema_version")
    try:
        schema_version_int = int(schema_version)
    except Exception as exc:
        raise ValueError(
            "Invalid transfer bundle format: missing or invalid schema_version."
        ) from exc
    if schema_version_int != _TRANSFER_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported transfer bundle schema version: "
            f"{schema_version!r}. Expected {_TRANSFER_BUNDLE_SCHEMA_VERSION}."
        )

    required_keys = (
        "embedding_columns",
        "id_column",
        "cluster_on_pca",
        "knn_model",
        "label_values",
        "metadata",
    )
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"Invalid transfer bundle format: missing keys: {missing_text}.")

    knn_model = payload.get("knn_model")
    if knn_model is None or not hasattr(knn_model, "predict"):
        raise ValueError("Invalid transfer bundle format: missing or invalid knn_model.")

    return NicheTransferBundle(
        embedding_columns=[str(col) for col in payload["embedding_columns"]],
        id_column=str(payload["id_column"]),
        cluster_on_pca=bool(payload["cluster_on_pca"]),
        pca_model=payload.get("pca_model"),
        knn_model=knn_model,
        label_values=np.asarray(payload["label_values"]),
        metadata=dict(payload["metadata"]),
    )
