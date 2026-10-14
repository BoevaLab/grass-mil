"""Class weights for the region cross-entropy, computed from the TRAINING split only.

The preprint weights class ``c`` by inverse frequency on the training regions,
``w_c = n / (k * count_c)``, normalised to mean 1, so the weights change the
balance between classes but not the overall loss scale. Computing them on the
training split alone keeps the test fold's label prior out of the objective.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import torch

__all__ = ["inverse_frequency_class_weights", "collect_unit_labels"]


def inverse_frequency_class_weights(
    labels: Sequence[int] | torch.Tensor,
    num_classes: int,
    *,
    normalize_mean: bool = True,
) -> torch.Tensor:
    """``w_c = n / (k * count_c)`` over ``labels`` (absent classes get weight 1)."""
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    y = torch.as_tensor(list(labels) if not isinstance(labels, torch.Tensor) else labels)
    y = y.long().view(-1)
    if y.numel() == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = torch.ones(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = float(y.numel()) / (float(int(present.sum())) * counts[present])
    if normalize_mean:
        weights = weights / weights.mean()
    return weights


def collect_unit_labels(
    dataset: Iterable[Any],
    *,
    target_columns: Optional[Sequence[str]] = None,
) -> list[int]:
    """Read the integer class label of every unit of a graph dataset.

    Units carry ``graph_y`` (shape ``[1, T]`` or ``[T]``); the first target column
    is used unless ``target_columns`` names one and the unit exposes
    ``graph_label_names``.
    """
    labels: list[int] = []
    for data in dataset:
        y = getattr(data, "graph_y", None)
        if y is None:
            continue
        y = torch.as_tensor(y).view(-1)
        col = 0
        names = getattr(data, "graph_label_names", None)
        if target_columns and isinstance(names, (list, tuple)):
            flat = list(names[0]) if names and isinstance(names[0], (list, tuple)) else list(names)
            if target_columns[0] in flat:
                col = int(flat.index(target_columns[0]))
        value = float(y[col])
        if value != value:  # NaN: unlabelled unit
            continue
        labels.append(int(round(value)))
    return labels
