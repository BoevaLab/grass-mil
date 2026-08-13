from __future__ import annotations

from typing import Optional

from torch import nn

try:
    from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for grass_mil.models.components.pooling"
    ) from exc

_SUPPORTED = ("sum", "mean", "max")


class GraphPooling(nn.Module):
    """Pool node embeddings into one vector per graph.

    Only order-invariant reductions are offered. Learned pooling
    (``GlobalAttention``, ``Set2Set``) was removed: instance-level selection is
    the job of the MIL attention head, which pools *across* ego-graphs, and
    duplicating it inside each ego-graph adds parameters without a
    corresponding claim. ``GlobalAttention`` is also deprecated upstream.
    """

    def __init__(self, name: Optional[str], input_dim: int, **_: object):
        super().__init__()
        self.name = (name or "mean").lower()
        self.input_dim = input_dim
        self.output_dim = input_dim
        if self.name == "sum":
            self.pool = global_add_pool
        elif self.name == "mean":
            self.pool = global_mean_pool
        elif self.name == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(
                f"Unsupported graph pooling '{name}'. Expected one of {list(_SUPPORTED)}."
            )

    def forward(self, x, batch):
        return self.pool(x, batch)
