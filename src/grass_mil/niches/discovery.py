"""Clustering cells into niches.

The clustering is fitted on a random sample of at most ``sample_size`` cells:
an optional PCA, L2 normalisation (``cosine``, so euclidean distances follow
the cosine geometry), then HDBSCAN (or k-means). The remaining cells are
assigned by a k-nearest-neighbour vote on the same coordinates. Cells HDBSCAN
leaves unassigned form the ``Background``.

Niches are numbered by their attention-weighted margin (the mean over units of
the niche's mean cell margin), ``Niche 1`` being the most positive; ``0`` is
the Background.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import numpy as np

__all__ = ["ClusteringConfig", "NicheModel", "fit_niches", "niche_name", "BACKGROUND"]

BACKGROUND = 0


def niche_name(niche_id: int) -> str:
    return "Background" if int(niche_id) == BACKGROUND else f"Niche {int(niche_id)}"


@dataclass(frozen=True)
class ClusteringConfig:
    """Attributes:
    space: ``attention`` (the attention head's pre-tanh space) or ``embedding``.
    pca_components: PCA before clustering (``None``: none).
    cosine: L2-normalise before clustering.
    method: ``hdbscan`` or ``kmeans``.
    min_cluster_size, min_samples, selection: HDBSCAN settings.
    n_clusters: k-means clusters.
    sample_size: Cells used to fit (``None``: all).
    assign_k: Neighbours voting for the cells outside the sample.
    seed: Sampling and k-means seed.
    """

    space: str = "attention"
    pca_components: int | None = None
    cosine: bool = True
    method: str = "hdbscan"
    min_cluster_size: int = 300
    min_samples: int = 1
    selection: str = "eom"
    n_clusters: int = 10
    sample_size: int | None = 100_000
    assign_k: int = 15
    seed: int = 0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> ClusteringConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown clustering keys: {sorted(unknown)}")
        out = cls(**cfg)
        if out.space not in ("attention", "embedding"):
            raise ValueError("clustering.space must be 'attention' or 'embedding'.")
        if out.method not in ("hdbscan", "kmeans"):
            raise ValueError("clustering.method must be 'hdbscan' or 'kmeans'.")
        return out


class NicheModel:
    """A fitted niche clustering that labels any cells of the same space."""

    def __init__(
        self, cfg: ClusteringConfig, pca, assigner, mapping: dict[int, int], sample: np.ndarray
    ) -> None:
        self.cfg = cfg
        self.pca = pca
        self.assigner = assigner
        self.mapping = mapping  # raw cluster id -> niche id
        self.sample = sample

    @property
    def n_niches(self) -> int:
        return len([v for v in self.mapping.values() if v != BACKGROUND])

    def coordinates(self, x: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import normalize

        z = np.asarray(x, dtype=np.float32)
        if self.pca is not None:
            z = self.pca.transform(z)
        return normalize(z) if self.cfg.cosine else z

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Niche ids of the cells (``0`` = Background)."""
        raw = self.assigner.predict(self.coordinates(x))
        return np.vectorize(self.mapping.get)(raw).astype(int)

    def settings(self) -> dict[str, Any]:
        return {**asdict(self.cfg), "n_niches": self.n_niches, "n_fitted": int(self.sample.size)}


def fit_niches(
    x: np.ndarray, margin: np.ndarray, unit_of_cell: np.ndarray, cfg: ClusteringConfig
) -> tuple[NicheModel, np.ndarray]:
    """Fit the clustering on ``x`` and return the model and every cell's niche id.

    Niches are ordered by their margin: the mean over units of the niche's mean
    cell ``margin`` in the unit (the statistic the niche table reports).
    """
    import pandas as pd

    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    rng = np.random.default_rng(cfg.seed)
    sample = (
        np.arange(n)
        if cfg.sample_size is None or cfg.sample_size >= n
        else np.sort(rng.choice(n, cfg.sample_size, replace=False))
    )
    pca = None
    z = x[sample]
    if cfg.pca_components:
        pca = PCA(
            n_components=min(cfg.pca_components, x.shape[1], sample.size), random_state=cfg.seed
        ).fit(z)
        z = pca.transform(z)
    if cfg.cosine:
        z = normalize(z)
    if cfg.method == "hdbscan":
        raw = HDBSCAN(
            min_cluster_size=int(min(cfg.min_cluster_size, max(2, sample.size))),
            min_samples=cfg.min_samples,
            cluster_selection_method=cfg.selection,
        ).fit_predict(z)
    else:
        raw = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.seed, n_init=10).fit_predict(z)
    assigner = KNeighborsClassifier(n_neighbors=max(1, min(cfg.assign_k, sample.size))).fit(z, raw)
    model = NicheModel(cfg, pca, assigner, {}, sample)
    labels = assigner.predict(model.coordinates(x))
    labels[sample] = raw
    # Number the clusters by margin, most positive first; -1 is Background.
    per_unit = pd.Series(margin).groupby([labels, unit_of_cell]).mean()
    score = per_unit.groupby(level=0).mean()
    order = sorted((c for c in score.index if c >= 0), key=lambda c: -float(score[c]))
    model.mapping = {-1: BACKGROUND, **{int(c): rank + 1 for rank, c in enumerate(order)}}
    return model, np.vectorize(model.mapping.get)(labels).astype(int)
