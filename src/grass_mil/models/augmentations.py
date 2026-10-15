"""Stochastic graph augmentations (pretraining views and fine-tuning regularisation).

All functions take an explicit ``torch.Generator`` (so seeded runs are
reproducible) and return a new batch instead of modifying their input. Edges
are always dropped in both directions together, so the graph stays undirected.
A masked node keeps its place in the graph, its features are zeroed and its
cell type is set to the encoder's "unassigned" code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Sequence

import torch

__all__ = [
    "PretrainViews",
    "FinetuneAugmentation",
    "degree_importance",
    "drop_importance",
    "drop_edges",
    "mask_nodes",
    "feature_noise",
]


def _rand(shape, generator, device) -> torch.Tensor:
    # A CPU generator cannot drive a CUDA draw: draw on CPU and move.
    return torch.rand(shape, generator=generator).to(device)


def _undirected(edge_index: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Restrict ``keep`` to edges whose reverse edge is kept as well."""
    if edge_index.numel() == 0:
        return keep
    n = int(edge_index.max()) + 1
    src, dst = edge_index
    kept = src[keep] * n + dst[keep]
    return keep & torch.isin(dst * n + src, kept)


def _keep_edges(batch, keep: torch.Tensor):
    batch.edge_index = batch.edge_index[:, keep]
    if getattr(batch, "edge_attr", None) is not None:
        batch.edge_attr = batch.edge_attr[keep]
    return batch


def mask_nodes(batch, masked: torch.Tensor, unassigned_index: int | None):
    """Zero the features of ``masked`` nodes and mark their cell type unassigned."""
    if batch.x.numel():
        batch.x = batch.x.masked_fill(masked.view(-1, 1), 0.0)
    if unassigned_index is not None and getattr(batch, "cell_type", None) is not None:
        batch.cell_type = batch.cell_type.masked_fill(masked, int(unassigned_index))
    return batch


def degree_importance(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-node importance in ``[0, 1]``: log degree normalised as ``(v - mean) / (max - mean)``,
    clamped at zero (every below-average node gets 0)."""
    from torch_geometric.utils import degree

    v = torch.log1p(degree(edge_index[0], num_nodes))
    return torch.clamp((v - v.mean()) / (v.max() - v.mean() + 1e-8), min=0.0)


def drop_importance(batch, *, mu: float, p_lambda: float, unassigned_index: int | None, generator):
    """Mask low-degree nodes and drop low-importance edges.

    The drop probability of a node (or edge) with importance ``s`` is
    ``min((1 - s) * mu, p_lambda)``; an edge's importance is the mean of its
    end points', renormalised the same way.
    """
    if not (0.0 < mu <= 1.0 and 0.0 < p_lambda <= 1.0):
        raise ValueError("mu and p_lambda must be in (0, 1].")
    out = batch.clone()
    imp = degree_importance(out.edge_index, int(out.num_nodes))
    p_node = torch.clamp((1.0 - imp) * mu, max=p_lambda)
    out = mask_nodes(
        out, _rand(p_node.shape, generator, p_node.device) <= p_node, unassigned_index
    )
    if out.edge_index.numel():
        e = (imp[out.edge_index[0]] + imp[out.edge_index[1]]) / 2.0
        e = torch.clamp((e - e.mean()) / (e.max() - e.mean() + 1e-8), min=0.0)
        p_edge = torch.clamp((1.0 - e) * mu, max=p_lambda)
        keep = _rand(p_edge.shape, generator, p_edge.device) > p_edge
        out = _keep_edges(out, _undirected(out.edge_index, keep))
    return out


def drop_edges(batch, *, p: float, generator):
    """Drop each undirected edge with probability ``p``."""
    if p <= 0.0 or batch.edge_index.numel() == 0:
        return batch
    out = batch.clone()
    keep = _rand((out.edge_index.size(1),), generator, out.edge_index.device) >= p
    return _keep_edges(out, _undirected(out.edge_index, keep))


def feature_noise(batch, *, std: float, columns: Sequence[int], generator):
    """Add Gaussian noise of ``std`` to the given feature columns."""
    if std <= 0.0 or not columns or batch.x.numel() == 0:
        return batch
    out = batch.clone()
    x = out.x.clone()
    for c in columns:
        x[:, c] += torch.randn(x.size(0), generator=generator).to(x.device) * std
    out.x = x
    return out


def _columns(columns: Sequence[Any] | None, feature_names: Sequence[str]) -> list[int]:
    out = []
    for c in columns or []:
        if isinstance(c, int):
            out.append(c)
        elif str(c) in feature_names:
            out.append(list(feature_names).index(str(c)))
        else:
            raise ValueError(f"Feature '{c}' not among the node features {list(feature_names)}.")
    return out


@dataclass(frozen=True)
class PretrainViews:
    """Two stochastic views per batch: importance-weighted node/edge drop plus
    Gaussian noise on selected continuous features."""

    mu: float = 0.25
    p_lambda: float = 0.45
    noise_std: float = 0.2
    noise_features: tuple = ()

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> PretrainViews:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown pretrain.augmentation keys: {sorted(unknown)}")
        cfg["noise_features"] = tuple(cfg.get("noise_features") or ())
        return cls(**cfg)

    def build(self, *, feature_names: Sequence[str], unassigned_index: int | None) -> Callable:
        columns = _columns(self.noise_features, feature_names)

        def view(batch, generator):
            out = drop_importance(
                batch,
                mu=self.mu,
                p_lambda=self.p_lambda,
                unassigned_index=unassigned_index,
                generator=generator,
            )
            return feature_noise(out, std=self.noise_std, columns=columns, generator=generator)

        return view


@dataclass(frozen=True)
class FinetuneAugmentation:
    """Uniform edge drop and node masking during fine-tuning (both off by default)."""

    drop_edge_p: float = 0.0
    mask_node_p: float = 0.0
    seed: int = 0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> FinetuneAugmentation:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown finetune.augmentation keys: {sorted(unknown)}")
        return cls(**cfg)

    @property
    def enabled(self) -> bool:
        return self.drop_edge_p > 0 or self.mask_node_p > 0

    def build(self, *, unassigned_index: int | None) -> Callable:
        generator = torch.Generator().manual_seed(int(self.seed))

        def augment(batch):
            out = drop_edges(batch, p=self.drop_edge_p, generator=generator)
            if self.mask_node_p > 0:
                out = out.clone() if out is batch else out
                masked = _rand((int(out.num_nodes),), generator, out.x.device) < self.mask_node_p
                out = mask_nodes(out, masked, unassigned_index)
            return out

        return augment
