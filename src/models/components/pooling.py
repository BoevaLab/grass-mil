from __future__ import annotations

from typing import Optional

from torch import nn

try:
    from torch_geometric.nn import (
        GlobalAttention,
        Set2Set,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for src.models.components.pooling") from exc


class GraphPooling(nn.Module):
    """Thin pooling wrapper with explicit output_dim contract."""

    def __init__(self, name: Optional[str], input_dim: int, set2set_steps: int = 3):
        super().__init__()
        self.name = (name or "mean").lower()
        self.input_dim = input_dim
        if self.name == "sum":
            self.pool = global_add_pool
            self.output_dim = input_dim
        elif self.name == "mean":
            self.pool = global_mean_pool
            self.output_dim = input_dim
        elif self.name == "max":
            self.pool = global_max_pool
            self.output_dim = input_dim
        elif self.name == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Linear(input_dim, 1))
            self.output_dim = input_dim
        elif self.name == "set2set":
            self.pool = Set2Set(input_dim, processing_steps=set2set_steps)
            self.output_dim = input_dim * 2
        else:
            raise ValueError(f"Unsupported graph pooling: {name}")

    def forward(self, x, batch):
        return self.pool(x, batch)
