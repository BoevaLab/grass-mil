from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.interpretability.contracts import ReductionResult


def run_reduction(
    method: str,
    X: np.ndarray,
    *,
    n_components: int = 2,
    random_state: Optional[int] = 42,
    **params: Any,
) -> ReductionResult:
    method_key = str(method).strip().lower()
    if method_key == "pca":
        model = PCA(n_components=n_components, random_state=random_state, **params)
        embedding = model.fit_transform(X)
    elif method_key == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "UMAP reduction requested but package `umap-learn` is unavailable."
            ) from exc
        model = umap.UMAP(n_components=n_components, random_state=random_state, **params)
        embedding = model.fit_transform(X)
    elif method_key == "tsne":
        model = TSNE(n_components=n_components, random_state=random_state, **params)
        embedding = model.fit_transform(X)
    else:
        raise ValueError(f"Unsupported reduction method: {method!r}")

    all_params: Dict[str, Any] = {"n_components": int(n_components), "random_state": random_state}
    all_params.update(params)
    return ReductionResult(
        method=method_key,
        embedding=np.asarray(embedding),
        params=all_params,
        fitted_object=model,
    )
