from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.cluster import (
    OPTICS,
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    SpectralClustering,
)

from grass_mil.interpretability.contracts import ClusteringResult


def run_clustering(
    method: str,
    X: np.ndarray,
    *,
    n_clusters: Optional[int] = None,
    random_state: Optional[int] = 42,
    **params: Any,
) -> ClusteringResult:
    """Run a clustering backend with estimator kwargs forwarded from `params`.

    Notes:
    - `n_clusters` and `random_state` are accepted as common convenience args.
    - Any additional `params` are passed directly to the selected estimator constructor.
    """
    method_key = str(method).strip().lower()
    if method_key == "kmeans":
        if n_clusters is None:
            raise ValueError("kmeans requires `n_clusters`.")
        model = KMeans(n_clusters=n_clusters, random_state=random_state, **params)
        labels = model.fit_predict(X)
    elif method_key == "agglomerative":
        if n_clusters is None:
            raise ValueError("agglomerative requires `n_clusters`.")
        model = AgglomerativeClustering(n_clusters=n_clusters, **params)
        labels = model.fit_predict(X)
    elif method_key == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HDBSCAN requested but sklearn HDBSCAN is unavailable in this environment."
            ) from exc
        model = HDBSCAN(**params)
        labels = model.fit_predict(X)
    elif method_key == "spectral":
        if n_clusters is None:
            raise ValueError("spectral requires `n_clusters`.")
        model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, **params)
        labels = model.fit_predict(X)
    elif method_key == "optics":
        model = OPTICS(**params)
        labels = model.fit_predict(X)
    elif method_key == "dbscan":
        model = DBSCAN(**params)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unsupported clustering method: {method!r}")

    all_params: Dict[str, Any] = {"n_clusters": n_clusters, "random_state": random_state}
    all_params.update(params)
    return ClusteringResult(
        method=method_key,
        labels=np.asarray(labels),
        params=all_params,
        fitted_object=model,
    )
