"""Optional dimensionality reduction of the continuous node features.

``identity`` keeps the configured features as they are. ``pca`` projects them
onto principal components fitted on:

* ``all``: every cell of the cohort (label-free, so it never leaks a split);
* ``train``: the cells of the training units of the configured split. Only
  valid for a single random split: cross-validation re-partitions the same
  graphs, so a train-only fit would leak into other folds.

One basis serves every sample, so a component means the same in all of them.

The fitted state is stored in ``processed_index.json`` and can be applied to a
new cohort with ``state_from`` (the index of the cohort it was fitted on), so
both are projected onto the same components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from grass_mil.utils import checked

__all__ = ["ReducerConfig", "Pca", "load_reducer_state"]


@dataclass(frozen=True)
class ReducerConfig:
    name: str = "identity"  # identity | pca
    n_components: int = 32
    fit_on: str = "all"  # all | train
    state_from: str | None = None

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> ReducerConfig:
        cfg = checked(cls, cfg, "data.feature_reducer")
        out = cls(**cfg)
        if out.name not in {"identity", "pca"}:
            raise ValueError("data.feature_reducer.name must be 'identity' or 'pca'.")
        if out.fit_on not in {"all", "train"}:
            raise ValueError("data.feature_reducer.fit_on must be 'all' or 'train'.")
        return out


@dataclass
class Pca:
    mean: np.ndarray  # (f,)
    components: np.ndarray  # (k, f)

    @classmethod
    def fit(cls, features: np.ndarray, n_components: int) -> Pca:
        """Leading eigenvectors of the feature covariance (memory independent of the cell count)."""
        x = np.asarray(features, dtype=np.float64)
        mean = x.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]))
        return cls(mean=mean, components=vectors[:, ::-1][:, :n_components].T.copy())

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) @ self.components.T).astype(np.float32)

    def state(self) -> dict[str, list]:
        return {"mean": self.mean.tolist(), "components": self.components.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Pca:
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float64),
            components=np.asarray(state["components"], dtype=np.float64),
        )


def load_reducer_state(index_path: str | Path) -> Pca:
    """The shared PCA basis stored in another cohort's ``processed_index.json``."""
    index = json.loads(Path(index_path).read_text())
    state = (index.get("feature_reducer") or {}).get("state")
    if not state or "mean" not in state:
        raise ValueError(f"{index_path} holds no shared PCA basis.")
    return Pca.from_state(state)
