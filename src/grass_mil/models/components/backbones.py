from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from .embeddings import CategoricalEmbeddingConfig, NodeInputEmbedding
from .pooling import GraphPooling

try:
    from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, SAGEConv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for grass_mil.models.components.backbones"
    ) from exc


NormType = Literal["batchnorm", "layernorm", "none"]
JKType = Literal["last", "concat", "max", "sum"]
ConvType = Literal["gin", "gcn", "gat", "graphsage", "gine"]


@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    out_dim: Optional[int] = None
    num_layers: int = 3
    dropout: float = 0.0
    conv_type: ConvType = "gin"
    norm: NormType = "batchnorm"
    jk: JKType = "last"
    act: Literal["relu", "gelu", "leaky_relu"] = "relu"
    pooling: Optional[str] = None
    set2set_steps: int = 3
    gat_heads: int = 4
    use_edge_attr: bool = False
    edge_weight_index: int = 0
    edge_attr_dim: Optional[int] = None
    # Selects a single scalar column of `edge_attr` (the edge length) for
    # edge-aware message passing. When set, the conv sees only that column and
    # `edge_attr_dim` is 1. Leave unset to pass the whole edge_attr vector.
    edge_feature_index: Optional[int] = None
    categorical_embedding: Optional[dict] = None


def _get_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _get_norm(name: NormType, dim: int) -> nn.Module:
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {name}")


class GNNEncoder(nn.Module):
    """PyG-first encoder with thin adapters for consistent edge handling."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        if cfg.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.cfg = cfg
        self._validate_edge_attr_config()
        self.act = _get_activation(cfg.act)
        self.dropout = nn.Dropout(cfg.dropout)

        self.input_proj = NodeInputEmbedding(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            categorical=CategoricalEmbeddingConfig.from_dict(cfg.categorical_embedding),
        )

        self.layers = nn.ModuleList([self._build_conv_layer() for _ in range(cfg.num_layers)])
        self.norms = nn.ModuleList(
            [_get_norm(cfg.norm, cfg.hidden_dim) for _ in range(cfg.num_layers)]
        )

        combined_dim = self._combined_node_dim()
        final_dim = cfg.out_dim if cfg.out_dim is not None else combined_dim
        self.output_proj = (
            nn.Identity() if final_dim == combined_dim else nn.Linear(combined_dim, final_dim)
        )
        self.output_dim = final_dim
        self.graph_pool = (
            GraphPooling(
                name=cfg.pooling,
                input_dim=final_dim,
                set2set_steps=cfg.set2set_steps,
            )
            if cfg.pooling is not None
            else None
        )

    def _build_conv_layer(self) -> nn.Module:
        ctype = self.cfg.conv_type
        hdim = self.cfg.hidden_dim
        if ctype == "gin":
            mlp = nn.Sequential(
                nn.Linear(hdim, hdim * 2),
                nn.ReLU(),
                nn.Linear(hdim * 2, hdim),
            )
            return GINConv(mlp)
        if ctype == "gcn":
            return GCNConv(hdim, hdim, add_self_loops=True, normalize=True)
        if ctype == "gat":
            if hdim % self.cfg.gat_heads != 0:
                raise ValueError("hidden_dim must be divisible by gat_heads")
            out_per_head = hdim // self.cfg.gat_heads
            return GATConv(
                in_channels=hdim,
                out_channels=out_per_head,
                heads=self.cfg.gat_heads,
                concat=True,
                edge_dim=None,
                add_self_loops=True,
            )
        if ctype == "graphsage":
            return SAGEConv(hdim, hdim, aggr="mean")
        if ctype == "gine":
            if self.cfg.edge_attr_dim is None:
                raise ValueError("conv_type='gine' requires edge_attr_dim in EncoderConfig.")
            mlp = nn.Sequential(
                nn.Linear(hdim, hdim * 2),
                nn.ReLU(),
                nn.Linear(hdim * 2, hdim),
            )
            return GINEConv(mlp, edge_dim=self.cfg.edge_attr_dim)
        raise ValueError(f"Unsupported conv_type: {ctype}")

    def _supports_edge_attr(self) -> bool:
        return self.cfg.conv_type in {"gcn", "gine"}

    def _validate_edge_attr_config(self) -> None:
        if self.cfg.conv_type == "gine" and self.cfg.edge_attr_dim is None:
            raise ValueError("conv_type='gine' requires edge_attr_dim to be set in EncoderConfig.")
        if self.cfg.edge_feature_index is not None:
            if self.cfg.edge_feature_index < 0:
                raise ValueError(
                    f"edge_feature_index must be >= 0, got {self.cfg.edge_feature_index}."
                )
            if self.cfg.conv_type != "gine":
                raise ValueError(
                    "edge_feature_index selects a scalar edge feature for GINE message "
                    f"passing, but conv_type='{self.cfg.conv_type}'. Use conv_type='gine', "
                    "or 'gcn' with edge_weight_index for scalar edge weights."
                )
            if self.cfg.edge_attr_dim != 1:
                raise ValueError(
                    "edge_feature_index selects exactly one edge column, so edge_attr_dim "
                    f"must be 1; got {self.cfg.edge_attr_dim}."
                )
        if self.cfg.use_edge_attr and not self._supports_edge_attr():
            warnings.warn(
                f"use_edge_attr=True is ignored for conv_type='{self.cfg.conv_type}'. "
                "Choose 'gcn' (scalar edge_weight) or 'gine' (edge_attr) for edge-aware message passing.",
                stacklevel=2,
            )

    def _combined_node_dim(self) -> int:
        if self.cfg.jk == "concat":
            return self.cfg.hidden_dim * (self.cfg.num_layers + 1)
        return self.cfg.hidden_dim

    def _apply_conv(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.cfg.conv_type == "gcn" and self.cfg.use_edge_attr:
            if edge_attr is None:
                raise ValueError(
                    "use_edge_attr=True with conv_type='gcn' requires edge_attr in forward inputs."
                )
            if edge_attr.dim() != 2:
                raise ValueError("edge_attr must have shape [num_edges, num_edge_features].")
            if self.cfg.edge_weight_index >= edge_attr.size(1):
                raise ValueError("edge_weight_index is out of range for provided edge_attr.")
            edge_weight = edge_attr[:, self.cfg.edge_weight_index]
            return layer(x, edge_index, edge_weight=edge_weight)
        if self.cfg.conv_type == "gine":
            if edge_attr is None:
                raise ValueError("conv_type='gine' requires edge_attr in forward inputs.")
            return layer(x, edge_index, self._select_edge_features(edge_attr))
        return layer(x, edge_index)

    def _select_edge_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Narrow ``edge_attr`` to the configured scalar column.

        The production encoder conditions message passing on the edge *length*
        alone, so only that column reaches the convolution. Without
        ``edge_feature_index`` the whole edge-attribute vector is passed
        through unchanged.
        """
        index = self.cfg.edge_feature_index
        if index is None:
            return edge_attr
        if edge_attr.dim() != 2:
            raise ValueError("edge_attr must have shape [num_edges, num_edge_features].")
        if index < 0 or index >= edge_attr.size(1):
            raise ValueError(
                f"edge_feature_index={index} is out of range for edge_attr with "
                f"{edge_attr.size(1)} column(s)."
            )
        return edge_attr[:, index : index + 1].float()

    def _apply_jk(self, states: list[torch.Tensor]) -> torch.Tensor:
        mode = self.cfg.jk
        if mode == "last":
            return states[-1]
        if mode == "concat":
            return torch.cat(states, dim=-1)
        if mode == "max":
            return torch.stack(states, dim=0).max(dim=0).values
        if mode == "sum":
            return torch.stack(states, dim=0).sum(dim=0)
        raise ValueError(f"Unsupported JK mode: {mode}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        categorical_codes: Optional[torch.Tensor] = None,
        return_graph_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x, categorical_codes)
        states = [h]
        for idx, layer in enumerate(self.layers):
            h = self._apply_conv(layer, h, edge_index, edge_attr)
            h = self.norms[idx](h)
            if idx != len(self.layers) - 1:
                h = self.act(h)
            h = self.dropout(h)
            states.append(h)

        node_emb = self.output_proj(self._apply_jk(states))
        if not return_graph_embedding:
            return node_emb
        if self.graph_pool is None:
            raise ValueError("return_graph_embedding=True requires cfg.pooling to be set")
        if batch is None:
            batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        graph_emb = self.graph_pool(node_emb, batch)
        return node_emb, graph_emb
