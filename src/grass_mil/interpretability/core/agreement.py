"""Agreement between partitions of the same instances.

The suite clusters several representation spaces -- the encoder embedding, the
attention head space, and a model-independent cell-type composition space.
Structure that appears in a learned space but not in the composition baseline is
attributable to the model rather than to raw abundance, so comparing partitions
is how that claim is checked.

All indices are chance-corrected or normalised, and computed over the same
instances in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["ClusterAgreementResult", "cluster_jaccard_matrix", "compute_cluster_agreement"]

_METRICS = ("ari", "ami", "nmi")


@dataclass(frozen=True)
class ClusterAgreementResult:
    """Pairwise agreement between labelings.

    Attributes:
        pairwise_metrics: One row per space pair, columns ``ari``/``ami``/``nmi``.
        jaccard: ``(space_a, space_b) -> cluster x cluster`` Jaccard overlap.
    """

    pairwise_metrics: pd.DataFrame
    jaccard: Dict[Tuple[str, str], pd.DataFrame]


def cluster_jaccard_matrix(labels_a: Sequence, labels_b: Sequence) -> pd.DataFrame:
    """Jaccard overlap between every pair of clusters from two labelings."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"Labelings must align; got {a.shape} and {b.shape}.")

    clusters_a = sorted(set(a.tolist()))
    clusters_b = sorted(set(b.tolist()))
    matrix = np.zeros((len(clusters_a), len(clusters_b)), dtype=float)
    for i, ca in enumerate(clusters_a):
        mask_a = a == ca
        for j, cb in enumerate(clusters_b):
            mask_b = b == cb
            union = int(np.sum(mask_a | mask_b))
            matrix[i, j] = float(np.sum(mask_a & mask_b) / union) if union else np.nan
    return pd.DataFrame(matrix, index=clusters_a, columns=clusters_b)


def compute_cluster_agreement(
    labelings: Mapping[str, Sequence],
    *,
    metrics: Sequence[str] = _METRICS,
    include_jaccard: bool = True,
) -> ClusterAgreementResult:
    """Chance-corrected agreement between every pair of labelings."""
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    unknown = set(metrics) - set(_METRICS)
    if unknown:
        raise ValueError(
            f"Unsupported agreement metric(s): {sorted(unknown)}. Expected a subset of {_METRICS}."
        )
    if len(labelings) < 2:
        raise ValueError("Cluster agreement needs at least two labelings to compare.")

    arrays = {name: np.asarray(values) for name, values in labelings.items()}
    sizes = {name: values.shape[0] for name, values in arrays.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"All labelings must cover the same instances; got sizes {sizes}.")

    functions = {
        "ari": adjusted_rand_score,
        "ami": adjusted_mutual_info_score,
        "nmi": normalized_mutual_info_score,
    }

    rows = []
    jaccard: Dict[Tuple[str, str], pd.DataFrame] = {}
    for name_a, name_b in combinations(sorted(arrays), 2):
        a, b = arrays[name_a], arrays[name_b]
        row: Dict[str, object] = {"space_a": name_a, "space_b": name_b}
        for metric in metrics:
            row[metric] = float(functions[metric](a, b))
        rows.append(row)
        if include_jaccard:
            jaccard[(name_a, name_b)] = cluster_jaccard_matrix(a, b)

    frame = pd.DataFrame(rows).set_index(["space_a", "space_b"])
    return ClusterAgreementResult(pairwise_metrics=frame, jaccard=jaccard)
