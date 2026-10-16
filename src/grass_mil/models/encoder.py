"""The ego-graph encoder: GINE message passing conditioned on edge length.

Node inputs are a learned cell-type embedding summed with a linear projection
of the continuous node features,

.. math:: h_v^{(0)} = E[t_v] + W x_v,

followed by ``num_layers`` GINE layers (norm, activation and dropout between
layers) and a pooling of the node states into one vector per ego-graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool

from grass_mil.utils import checked

__all__ = ["EncoderConfig", "GINEEncoder"]

_POOL = {"max": global_max_pool, "mean": global_mean_pool, "sum": global_add_pool}
_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}
_NORM = {"layernorm": nn.LayerNorm, "batchnorm": nn.BatchNorm1d}


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder architecture (input widths come from the data, not the config).

    Attributes:
        hidden_dim: Width of every GINE layer.
        out_dim: Width of the ego-graph embedding (a final linear map when it
            differs from the jumping-knowledge output).
        num_layers: GINE layers (match ``data.sampler.depth``).
        dropout: Dropout after every layer.
        norm: ``layernorm`` | ``batchnorm`` | ``none``.
        jk: Jumping knowledge over layer outputs: ``last`` | ``concat`` |
            ``max`` | ``sum``.
        act: ``relu`` | ``gelu`` | ``leaky_relu``.
        pooling: Ego-graph readout: ``max`` | ``mean`` | ``sum``.
        mlp_ratio: Hidden width of each GINE MLP, as a multiple of ``hidden_dim``.
        train_eps: Learn the GIN epsilon.
        edge_feature: Column of ``edge_attr`` the convolutions see (``None``:
            every column).
        input_init: ``xavier`` or ``default`` initialisation of the input
            embedding and projection.
        unassigned_cell_type: Cell type used to mark a masked node in
            augmentations; ``None`` reserves an extra embedding row for it.
    """

    hidden_dim: int = 256
    out_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.0
    norm: str = "layernorm"
    jk: str = "last"
    act: str = "relu"
    pooling: str = "max"
    mlp_ratio: int = 2
    train_eps: bool = True
    edge_feature: int | None = 0
    input_init: str = "xavier"
    unassigned_cell_type: str | None = None

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> EncoderConfig:
        cfg = checked(cls, cfg, "model.encoder")
        out = cls(**cfg)
        checks = {
            "norm": (out.norm, {"layernorm", "batchnorm", "none"}),
            "jk": (out.jk, {"last", "concat", "max", "sum"}),
            "act": (out.act, set(_ACT)),
            "pooling": (out.pooling, set(_POOL)),
            "input_init": (out.input_init, {"xavier", "default"}),
        }
        for key, (value, allowed) in checks.items():
            if value not in allowed:
                raise ValueError(f"model.encoder.{key} must be one of {sorted(allowed)}")
        return out


class NodeInput(nn.Module):
    """``E[cell_type] + W x``; either term is absent when the data lacks it."""

    def __init__(self, feature_dim: int, cell_types: tuple[str, ...], cfg: EncoderConfig) -> None:
        super().__init__()
        if feature_dim == 0 and not cell_types:
            raise ValueError("The encoder needs cell types or continuous node features.")
        self.linear = nn.Linear(feature_dim, cfg.hidden_dim) if feature_dim else None
        self.embedding = None
        self.unassigned_index = None
        if cell_types:
            if cfg.unassigned_cell_type is None:
                rows, self.unassigned_index = len(cell_types) + 1, len(cell_types)
            elif cfg.unassigned_cell_type in cell_types:
                rows, self.unassigned_index = (
                    len(cell_types),
                    cell_types.index(cfg.unassigned_cell_type),
                )
            else:
                raise ValueError(
                    f"model.encoder.unassigned_cell_type '{cfg.unassigned_cell_type}' is not a "
                    f"cell type of the data ({list(cell_types)})."
                )
            self.embedding = nn.Embedding(rows, cfg.hidden_dim)
        if cfg.input_init == "xavier":
            if self.embedding is not None:
                nn.init.xavier_uniform_(self.embedding.weight)
            if self.linear is not None:
                nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, cell_type: torch.Tensor | None) -> torch.Tensor:
        h = self.linear(x) if self.linear is not None else 0.0
        if self.embedding is not None:
            if cell_type is None:
                raise ValueError("The encoder embeds cell types but the batch has none.")
            h = h + self.embedding(cell_type)
        return h


class GINEEncoder(nn.Module):
    def __init__(
        self, cfg: EncoderConfig, *, feature_dim: int, cell_types: tuple[str, ...], edge_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.edge_feature is not None and not 0 <= cfg.edge_feature < edge_dim:
            raise ValueError(
                f"model.encoder.edge_feature={cfg.edge_feature} but edges have {edge_dim} feature(s)."
            )
        conv_edge_dim = 1 if cfg.edge_feature is not None else edge_dim
        h, k = cfg.hidden_dim, cfg.mlp_ratio * cfg.hidden_dim
        self.input_proj = NodeInput(feature_dim, cell_types, cfg)
        self.layers = nn.ModuleList(
            GINEConv(
                nn.Sequential(nn.Linear(h, k), nn.ReLU(), nn.Linear(k, h)),
                edge_dim=conv_edge_dim,
                train_eps=cfg.train_eps,
            )
            for _ in range(cfg.num_layers)
        )
        self.norms = nn.ModuleList(
            _NORM[cfg.norm](h) if cfg.norm != "none" else nn.Identity()
            for _ in range(cfg.num_layers)
        )
        self.act = _ACT[cfg.act]()
        self.dropout = nn.Dropout(cfg.dropout)
        jk_dim = h * (cfg.num_layers + 1) if cfg.jk == "concat" else h
        self.output_proj = (
            nn.Identity() if jk_dim == cfg.out_dim else nn.Linear(jk_dim, cfg.out_dim)
        )
        self.output_dim = cfg.out_dim

    @property
    def unassigned_index(self) -> int | None:
        return self.input_proj.unassigned_index

    def node_embeddings(self, batch) -> torch.Tensor:
        """One embedding per node of ``batch``."""
        edge_attr = batch.edge_attr
        if self.cfg.edge_feature is not None:
            edge_attr = edge_attr[:, self.cfg.edge_feature : self.cfg.edge_feature + 1]
        h = self.input_proj(batch.x, getattr(batch, "cell_type", None))
        states = [h]
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            h = norm(conv(h, batch.edge_index, edge_attr.float()))
            if i < len(self.layers) - 1:
                h = self.act(h)
            h = self.dropout(h)
            states.append(h)
        if self.cfg.jk == "last":
            nodes = states[-1]
        elif self.cfg.jk == "concat":
            nodes = torch.cat(states, dim=-1)
        else:
            stacked = torch.stack(states)
            nodes = stacked.max(0).values if self.cfg.jk == "max" else stacked.sum(0)
        return self.output_proj(nodes)

    def forward(self, batch) -> torch.Tensor:
        """One embedding per ego-graph of ``batch`` (the pooled node embeddings)."""
        nodes = self.node_embeddings(batch)
        index = getattr(batch, "batch", None)
        if index is None:
            index = torch.zeros(nodes.size(0), dtype=torch.long, device=nodes.device)
        return _POOL[self.cfg.pooling](nodes, index)
