from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from .pooling import GraphPooling

try:
    from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for src.models.components.backbones"
    ) from exc


NormType = Literal["batchnorm", "layernorm", "none"]
JKType = Literal["last", "concat", "max", "sum"]
ConvType = Literal["gin", "gcn", "gat", "graphsage"]


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
        self.act = _get_activation(cfg.act)
        self.dropout = nn.Dropout(cfg.dropout)

        self.input_proj = (
            nn.Identity()
            if cfg.input_dim == cfg.hidden_dim
            else nn.Linear(cfg.input_dim, cfg.hidden_dim)
        )

        self.layers = nn.ModuleList(
            [self._build_conv_layer() for _ in range(cfg.num_layers)]
        )
        self.norms = nn.ModuleList(
            [_get_norm(cfg.norm, cfg.hidden_dim) for _ in range(cfg.num_layers)]
        )

        combined_dim = self._combined_node_dim()
        final_dim = cfg.out_dim if cfg.out_dim is not None else combined_dim
        self.output_proj = (
            nn.Identity()
            if final_dim == combined_dim
            else nn.Linear(combined_dim, final_dim)
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
        raise ValueError(f"Unsupported conv_type: {ctype}")

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
        if (
            self.cfg.conv_type == "gcn"
            and edge_attr is not None
            and self.cfg.use_edge_attr
        ):
            edge_weight = edge_attr[:, self.cfg.edge_weight_index]
            return layer(x, edge_index, edge_weight=edge_weight)
        return layer(x, edge_index)

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
        return_graph_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
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
            raise ValueError(
                "return_graph_embedding=True requires cfg.pooling to be set"
            )
        if batch is None:
            batch = torch.zeros(
                node_emb.size(0), dtype=torch.long, device=node_emb.device
            )
        graph_emb = self.graph_pool(node_emb, batch)
        return node_emb, graph_emb
