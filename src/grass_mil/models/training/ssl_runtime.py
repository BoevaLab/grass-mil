from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import dropout_edge
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for ssl runtime helpers") from exc


@dataclass
class CosineWarmup:
    base_value: float
    warmup_steps: int
    total_steps: int
    min_value: float = 0.0

    def value(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.base_value
        step = max(0, min(step, self.total_steps))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_value * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_value + (self.base_value - self.min_value) * cosine


def augment_graph(
    batch: Data,
    *,
    drop_edge_p: float,
    drop_feat_p: float,
    force_undirected: bool = True,
) -> Data:
    """Elementwise feature masking plus uniform edge dropping.

    ``force_undirected`` defaults to True. The legacy transform left it False
    here while BGRL pretraining used True, so finetuning silently augmented an
    asymmetric graph and message passing became direction-dependent.
    """
    aug = batch.clone()
    if drop_feat_p > 0 and hasattr(aug, "x") and aug.x.numel() > 0:
        feat_mask = torch.rand_like(aug.x) > drop_feat_p
        aug.x = aug.x * feat_mask.float()
    if drop_edge_p > 0 and hasattr(aug, "edge_index"):
        edge_index, edge_mask = dropout_edge(
            aug.edge_index, p=drop_edge_p, force_undirected=force_undirected
        )
        aug.edge_index = edge_index
        if hasattr(aug, "edge_attr") and aug.edge_attr is not None:
            kept = aug.edge_attr[edge_mask]
            if force_undirected:
                # dropout_edge re-symmetrises after masking, so attributes are
                # duplicated to match the restored reverse edges.
                kept = torch.cat([kept, kept], dim=0)[: edge_index.size(1)]
            aug.edge_attr = kept
    return aug
