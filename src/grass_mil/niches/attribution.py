"""Per-cell attention-weighted margins.

Within a unit, the attention of class ``c`` is a softmax over the unit's cells,
so for any reference logit ``r_c`` the bag logit is exactly
``L_c = r_c + sum_i A[i, c] (l[i, c] - r_c)``. The reference is the model's
average cell (its mean instance logit over the cohort, recorded by
``grass-mil-extract``), so a cell's contribution is positive when the cell
pushes its unit's logit above what an average cell would. (The head bias is no
neutral reference: the embeddings are not centred, so ``l - b`` carries a
model-specific offset that would set the sign of every contribution.)

The margin of a cell is its contribution to the positive class minus its
contribution to the other classes (their mean beyond two classes), multiplied
by the unit's cell count ``N_u`` so that units of different sizes are
comparable (a uniformly attended cell gets weight 1). For regression and
survival models it is the contribution to the first output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cell_margins", "unit_attention"]


def unit_attention(scores: np.ndarray, unit_of_cell: np.ndarray) -> np.ndarray:
    """Attention weights: a softmax over each unit's cells, per class."""
    s = pd.DataFrame(np.asarray(scores, dtype=np.float64))
    e = np.exp(s - s.groupby(unit_of_cell).transform("max"))
    return (e / e.groupby(unit_of_cell).transform("sum")).to_numpy()


def cell_margins(
    logits: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray,
    unit_of_cell: np.ndarray,
    *,
    positive_class: int = 1,
) -> np.ndarray:
    """Size-scaled, attention-weighted margin of every cell toward the positive output."""
    logits = np.asarray(logits, dtype=np.float64)
    weights = unit_attention(np.asarray(scores, dtype=np.float64), unit_of_cell)
    contrib = weights * (logits - np.asarray(reference, dtype=np.float64)[None, :])
    n_classes = contrib.shape[1]
    if n_classes == 1:
        margin = contrib[:, 0]
    else:
        rest = (contrib.sum(axis=1) - contrib[:, positive_class]) / (n_classes - 1)
        margin = contrib[:, positive_class] - rest
    _, inverse, counts = np.unique(unit_of_cell, return_inverse=True, return_counts=True)
    return (margin * counts[inverse]).astype(np.float32)
