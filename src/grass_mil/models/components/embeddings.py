"""Input embeddings for node features.

The published encoder consumes cell identity through a learned embedding table
rather than as a one-hot column of ``x``:

.. math:: h_v^{(0)} = E[t_v] + W_x x_v

with :math:`t_v` the categorical cell-type code and :math:`x_v` the continuous
node features (cell size, where available). The two terms are **summed**, not
concatenated, so the embedding dimension is the hidden dimension.

This matters for the cohorts whose only node input is cell type: there ``x`` has
zero columns and the embedding is the entire input representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

__all__ = ["CategoricalEmbeddingConfig", "NodeInputEmbedding"]


@dataclass(frozen=True)
class CategoricalEmbeddingConfig:
    """Selects one categorical node label to embed.

    Attributes:
        label: Key into ``data.categorical_slices`` naming the column of
            ``data.categorical_codes`` to embed (typically ``cell_type``).
        num_embeddings: Vocabulary size. ``None`` defers to the cardinality
            recorded by precompute.
        column_index: Column of ``categorical_codes``. ``None`` defers to
            ``categorical_slices[label]``.
        reserve_unassigned: Reserve one extra row so augmentations can mark a
            dropped node as "unassigned" without resizing the table. The
            reserved code is the last row.
    """

    label: str = "cell_type"
    num_embeddings: Optional[int] = None
    column_index: Optional[int] = None
    reserve_unassigned: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Mapping[str, Any]]) -> Optional["CategoricalEmbeddingConfig"]:
        if cfg is None:
            return None
        allowed = {"label", "num_embeddings", "column_index", "reserve_unassigned"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown categorical embedding keys: {sorted(unknown)}. "
                f"Expected a subset of {sorted(allowed)}."
            )
        return cls(**dict(cfg))


class NodeInputEmbedding(nn.Module):
    """Project node inputs to the hidden dimension.

    With no categorical config this reproduces the plain input projection: an
    ``Identity`` when the dimensions already agree, otherwise a ``Linear``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        categorical: Optional[CategoricalEmbeddingConfig] = None,
    ) -> None:
        super().__init__()
        if input_dim < 0:
            raise ValueError(f"input_dim must be >= 0, got {input_dim}.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.categorical = categorical

        if self.input_dim == 0:
            # Cell-type-only cohorts: precompute emits an empty feature matrix
            # and the embedding carries the whole representation.
            if categorical is None:
                raise ValueError(
                    "input_dim=0 requires a categorical embedding; the encoder would "
                    "otherwise receive no input features at all."
                )
            self.linear: nn.Module = nn.Identity()
            self._uses_linear = False
        else:
            self.linear = (
                nn.Identity()
                if self.input_dim == self.hidden_dim
                else nn.Linear(self.input_dim, self.hidden_dim)
            )
            self._uses_linear = True

        self.embedding: Optional[nn.Embedding] = None
        self._column_index: Optional[int] = None
        self._unassigned_index: Optional[int] = None
        if categorical is not None:
            if categorical.num_embeddings is None:
                raise ValueError(
                    "categorical.num_embeddings is unset. Resolve it from the datamodule "
                    "metadata (categorical_cardinalities) or set it explicitly in config."
                )
            if categorical.num_embeddings < 1:
                raise ValueError(
                    f"categorical.num_embeddings must be >= 1, got {categorical.num_embeddings}."
                )
            num_rows = int(categorical.num_embeddings) + int(categorical.reserve_unassigned)
            self.embedding = nn.Embedding(num_rows, self.hidden_dim)
            self._column_index = (
                0 if categorical.column_index is None else int(categorical.column_index)
            )
            if self._column_index < 0:
                raise ValueError(
                    f"categorical.column_index must be >= 0, got {self._column_index}."
                )
            if categorical.reserve_unassigned:
                self._unassigned_index = num_rows - 1

    @property
    def unassigned_index(self) -> Optional[int]:
        """Reserved code marking a dropped node, or ``None`` when not reserved."""
        return self._unassigned_index

    def extra_repr(self) -> str:
        label = self.categorical.label if self.categorical is not None else None
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, categorical={label}"

    def forward(
        self,
        x: Optional[torch.Tensor],
        categorical_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embedding is None:
            if x is None:
                raise ValueError("NodeInputEmbedding requires `x` when no categorical embedding.")
            return self.linear(x)

        if categorical_codes is None:
            raise ValueError(
                f"A categorical embedding on '{self.categorical.label}' is configured but the "
                "batch carries no `categorical_codes`. Enable the categorical feature in the "
                "datamodule, or unset model.encoder.categorical_embedding."
            )
        if categorical_codes.dim() != 2:
            raise ValueError(
                "categorical_codes must have shape [num_nodes, num_labels]; got "
                f"{tuple(categorical_codes.shape)}."
            )
        if self._column_index >= categorical_codes.size(1):
            raise ValueError(
                f"categorical column_index={self._column_index} is out of range for "
                f"categorical_codes with {categorical_codes.size(1)} column(s)."
            )

        codes = categorical_codes[:, self._column_index].long()
        num_rows = int(self.embedding.num_embeddings)
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= num_rows):
            # Fail loudly rather than clamping: an out-of-range code means the
            # vocabulary and the data disagree, and silently folding it into a
            # valid row would train on a wrong cell type.
            raise ValueError(
                f"Categorical code out of range for '{self.categorical.label}': got "
                f"[{int(codes.min())}, {int(codes.max())}] for an embedding with "
                f"{num_rows} rows."
            )
        embedded = self.embedding(codes)

        if not self._uses_linear or x is None or x.size(1) == 0:
            return embedded
        return self.linear(x) + embedded


def resolve_categorical_embedding_config(
    cfg: Optional[Mapping[str, Any]],
    *,
    cardinalities: Optional[Dict[str, int]] = None,
    slices: Optional[Mapping[str, int]] = None,
) -> Optional[CategoricalEmbeddingConfig]:
    """Fill in vocabulary size and column index from datamodule metadata.

    Explicit config values always win; ``None`` triggers inference, mirroring
    the ``encoder.input_dim`` contract.
    """
    resolved = CategoricalEmbeddingConfig.from_dict(cfg)
    if resolved is None:
        return None

    num_embeddings = resolved.num_embeddings
    if num_embeddings is None and cardinalities:
        num_embeddings = cardinalities.get(resolved.label)
    column_index = resolved.column_index
    if column_index is None and slices:
        column_index = slices.get(resolved.label)

    if num_embeddings is None:
        raise ValueError(
            f"Could not resolve the vocabulary size for categorical label "
            f"'{resolved.label}'. Set model.encoder.categorical_embedding.num_embeddings "
            "explicitly, or include the label in data.categorical_features.include_labels."
        )
    return CategoricalEmbeddingConfig(
        label=resolved.label,
        num_embeddings=int(num_embeddings),
        column_index=None if column_index is None else int(column_index),
        reserve_unassigned=resolved.reserve_unassigned,
    )
