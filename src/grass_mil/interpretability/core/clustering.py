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

from grass_mil.contracts import ClusteringResult


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


# ----------------------------------------------------------------------------
# Sample -> reduce -> (L2) -> cluster -> assign: the GRASS-MIL niche recipe
# ----------------------------------------------------------------------------


class NicheClusterModel:
    """A fitted "reduce on a sample, cluster, assign the rest" model.

    ``predict(X)`` reduces new embeddings with the fitted reducer and assigns
    them with the k-NN classifier fitted on the clustered sample (HDBSCAN has
    no native predict; k-NN on the reduced coordinates is what the preprint
    uses). Backends with their own ``predict`` (k-means) use it directly.
    """

    def __init__(
        self,
        *,
        reducer,
        clusterer,
        assigner,
        sample_idx: np.ndarray,
        labels_sample: np.ndarray,
        params: Dict[str, Any],
        l2_normalise: bool,
        reducer_method: str,
        clustering_method: str,
    ) -> None:
        self.reducer = reducer
        self.clusterer = clusterer
        self.assigner = assigner
        self.sample_idx = np.asarray(sample_idx)
        self.labels_sample = np.asarray(labels_sample)
        self.params = dict(params)
        self.l2_normalise = bool(l2_normalise)
        self.reducer_method = reducer_method
        self.clustering_method = clustering_method

    def reduce(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return X if self.reducer is None else self.reducer.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        P = self.reduce(X)
        if self.assigner is not None:
            return np.asarray(self.assigner.predict(P))
        from sklearn.preprocessing import normalize

        Pc = normalize(P, norm="l2") if self.l2_normalise else P
        return np.asarray(self.clusterer.predict(Pc))

    @property
    def n_clusters(self) -> int:
        return int(len(set(self.labels_sample.tolist()) - {-1}))


def fit_niche_clustering(
    X: np.ndarray,
    *,
    reduction: Optional[Dict[str, Any]] = None,
    clustering: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = 100_000,
    l2_normalise: bool = True,
    assign_k: int = 15,
    seed: int = 0,
) -> tuple[NicheClusterModel, np.ndarray]:
    """Fit the niche clustering and label every row of ``X``.

    Default (preprint): PCA to 10 components fitted on a random sample of at
    most ``sample_size`` rows, L2 normalisation (so euclidean HDBSCAN acts on
    the cosine geometry), ``HDBSCAN(min_cluster_size=300, min_samples=1,
    cluster_selection_method="eom")`` on the sample, and 15-nearest-neighbour
    assignment of the remaining rows on the (un-normalised) PCs.

    ``reduction`` = ``{"method": "pca"|"none", "n_components": int, ...}``;
    ``clustering`` = ``{"method": <run_clustering backend>, "params": {...},
    "n_clusters": int}``. Any backend of :func:`run_clustering` is accepted.
    Returns ``(model, labels_all)`` with ``-1`` for noise.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    from grass_mil.interpretability.core.reduction import run_reduction

    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    reduction = dict(reduction or {"method": "pca", "n_components": 10})
    clustering = dict(
        clustering
        or {
            "method": "hdbscan",
            "params": {
                "min_cluster_size": 300,
                "min_samples": 1,
                "cluster_selection_method": "eom",
            },
        }
    )
    rng = np.random.default_rng(int(seed))
    if sample_size is None or int(sample_size) >= n:
        sample_idx = np.arange(n)
    else:
        sample_idx = np.sort(rng.choice(n, int(sample_size), replace=False))

    red_method = str(reduction.get("method", "pca")).strip().lower()
    reducer = None
    if red_method not in {"none", "identity"}:
        red_params = {
            k: v
            for k, v in reduction.items()
            if k not in {"method", "n_components", "random_state"}
        }
        n_components = int(reduction.get("n_components", 10))
        n_components = max(1, min(n_components, int(X.shape[1]), int(sample_idx.size)))
        res = run_reduction(
            red_method,
            X[sample_idx],
            n_components=n_components,
            random_state=int(reduction.get("random_state", 42)),
            **red_params,
        )
        reducer = res.fitted_object
    P_sample = X[sample_idx] if reducer is None else reducer.transform(X[sample_idx])
    P_cluster = normalize(P_sample, norm="l2") if l2_normalise else P_sample

    cl_method = str(clustering.get("method", "hdbscan")).strip().lower()
    cl_params = dict(clustering.get("params", {}) or {})
    cl_params.pop("metric", None) if cl_method == "hdbscan" and l2_normalise else None
    if cl_method == "hdbscan":
        cl_params.setdefault("min_cluster_size", 300)
        cl_params.setdefault("min_samples", 1)
        cl_params.setdefault("cluster_selection_method", "eom")
        cl_params["min_cluster_size"] = int(
            min(int(cl_params["min_cluster_size"]), max(2, sample_idx.size))
        )
    result = run_clustering(
        cl_method,
        P_cluster,
        n_clusters=clustering.get("n_clusters"),
        random_state=int(clustering.get("random_state", 42)),
        **cl_params,
    )
    labels_sample = np.asarray(result.labels).astype(int)

    assigner = None
    if int(assign_k) > 0 or not hasattr(result.fitted_object, "predict"):
        k = max(1, min(int(assign_k) if int(assign_k) > 0 else 15, int(sample_idx.size)))
        assigner = KNeighborsClassifier(k).fit(P_sample, labels_sample)
    model = NicheClusterModel(
        reducer=reducer,
        clusterer=result.fitted_object,
        assigner=assigner,
        sample_idx=sample_idx,
        labels_sample=labels_sample,
        params={
            "reduction": {
                "method": red_method,
                **(
                    {"n_components": int(reducer.n_components_)}
                    if reducer is not None and hasattr(reducer, "n_components_")
                    else {}
                ),
            },
            "clustering": {
                "method": cl_method,
                **cl_params,
                **(
                    {"n_clusters": clustering.get("n_clusters")}
                    if clustering.get("n_clusters")
                    else {}
                ),
            },
            "sample_size": int(sample_idx.size),
            "l2_normalise": bool(l2_normalise),
            "assign_k": int(assign_k),
            "seed": int(seed),
        },
        l2_normalise=l2_normalise,
        reducer_method=red_method,
        clustering_method=cl_method,
    )
    labels_all = model.predict(X)
    labels_all[sample_idx] = labels_sample
    return model, labels_all


def cosine_hdbscan(
    X: np.ndarray,
    *,
    n_components: int = 10,
    min_cluster_size: int = 300,
    min_samples: Optional[int] = 1,
    cluster_selection_method: str = "eom",
    sample_size: Optional[int] = 100_000,
    assign_k: int = 15,
    seed: int = 0,
) -> tuple[NicheClusterModel, np.ndarray]:
    """The preprint's niche clustering with its defaults (see :func:`fit_niche_clustering`)."""
    return fit_niche_clustering(
        X,
        reduction={"method": "pca", "n_components": int(n_components)},
        clustering={
            "method": "hdbscan",
            "params": {
                "min_cluster_size": int(min_cluster_size),
                "min_samples": min_samples,
                "cluster_selection_method": str(cluster_selection_method),
            },
        },
        sample_size=sample_size,
        l2_normalise=True,
        assign_k=assign_k,
        seed=seed,
    )
