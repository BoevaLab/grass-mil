"""Per-cell attention-weighted margins.

Within a unit, the attention of class ``c`` is a softmax over the unit's cells,
so a cell's contribution ``A[i, c] (l[i, c] - b_c)`` sums over the unit to its
bag logit minus the head bias ``b_c``. The margin of a cell is its
contribution to the positive class minus its contribution to the other classes
(their mean beyond two classes), multiplied by the unit's cell count ``N_u`` so
that units of different sizes are comparable (a uniformly attended cell gets
weight 1). For regression and survival models it is the contribution to the
first output.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cell_margins", "unit_attention"]


def unit_attention(scores: np.ndarray, unit_of_cell: np.ndarray) -> np.ndarray:
    """Attention weights: a softmax over each unit's cells, per class."""
    weights = np.zeros_like(scores, dtype=np.float64)
    for unit in np.unique(unit_of_cell):
        m = unit_of_cell == unit
        s = scores[m] - scores[m].max(axis=0, keepdims=True)
        e = np.exp(s)
        weights[m] = e / e.sum(axis=0, keepdims=True)
    return weights


def cell_margins(
    logits: np.ndarray,
    scores: np.ndarray,
    bias: np.ndarray,
    unit_of_cell: np.ndarray,
    *,
    positive_class: int = 1,
) -> np.ndarray:
    """Size-scaled, attention-weighted margin of every cell toward the positive output."""
    logits = np.asarray(logits, dtype=np.float64)
    weights = unit_attention(np.asarray(scores, dtype=np.float64), unit_of_cell)
    contrib = weights * (logits - np.asarray(bias, dtype=np.float64)[None, :])
    n_classes = contrib.shape[1]
    if n_classes == 1:
        margin = contrib[:, 0]
    else:
        rest = (contrib.sum(axis=1) - contrib[:, positive_class]) / (n_classes - 1)
        margin = contrib[:, positive_class] - rest
    _, inverse, counts = np.unique(unit_of_cell, return_inverse=True, return_counts=True)
    return (margin * counts[inverse]).astype(np.float32)
