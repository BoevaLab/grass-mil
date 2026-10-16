"""Euclidean radius cap for k-hop ego-graphs.

Hop depth alone does not bound the physical extent of an ego-graph in sparse
tissue. With the cap enabled, nodes farther than

.. math:: r_k = \\mathrm{radius\\_per\\_hop} \\cdot k + \\mathrm{radius\\_offset}

from the root are dropped (``k`` is the sampler depth), in coordinate units
after ``coord_scale_um``. Applied to every sampled ego-graph after sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from grass_mil.utils import checked

__all__ = ["EgoRadiusConfig", "EgoRadiusCutoff", "radius_for_depth"]


@dataclass(frozen=True)
class EgoRadiusConfig:
    enabled: bool = False
    radius_per_hop: float = 75.0
    radius_offset: float = 55.0
    keep_root_component: bool = True

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> EgoRadiusConfig:
        cfg = checked(cls, cfg, "ego_radius")
        return cls(**cfg)


def radius_for_depth(depth: int, cfg: EgoRadiusConfig) -> float:
    return float(cfg.radius_per_hop) * int(depth) + float(cfg.radius_offset)


class EgoRadiusCutoff:
    """Drop the nodes of an ego-graph farther than ``radius`` from its root.

    The root is always kept. With ``keep_root_component`` the nodes left
    disconnected from the root by the cut are dropped too.
    """

    def __init__(self, *, radius: float, keep_root_component: bool = True) -> None:
        self.radius = float(radius)
        self.keep_root_component = bool(keep_root_component)

    def __repr__(self) -> str:
        return f"EgoRadiusCutoff(radius={self.radius}, keep_root_component={self.keep_root_component})"

    def __call__(self, data, root: int):
        """Return the cut ego-graph and the new index of its root."""
        from torch_geometric.utils import subgraph

        n = int(data.num_nodes)
        dist = torch.linalg.norm(data.pos[:, :2] - data.pos[root, :2], dim=1)
        keep = dist <= self.radius
        keep[root] = True
        if self.keep_root_component:
            keep = self._root_component(data.edge_index, keep, root, n)
        if bool(keep.all()):
            return data, root
        idx = torch.nonzero(keep).view(-1)
        edge_index, edge_attr = subgraph(
            idx, data.edge_index, getattr(data, "edge_attr", None), relabel_nodes=True, num_nodes=n
        )
        out = data.clone()
        for key, value in data:
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == n:
                out[key] = value[idx]
        out.edge_index = edge_index
        if edge_attr is not None:
            out.edge_attr = edge_attr
        out.num_nodes = int(idx.numel())
        return out, int((idx < root).sum())

    @staticmethod
    def _root_component(edge_index: torch.Tensor, keep: torch.Tensor, root: int, n: int):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        keep_np = keep.cpu().numpy()
        src, dst = edge_index.cpu().numpy()
        both = keep_np[src] & keep_np[dst]
        adj = coo_matrix(
            (np.ones(int(both.sum()), dtype=bool), (src[both], dst[both])), shape=(n, n)
        )
        _, labels = connected_components(adj, directed=False)
        return torch.from_numpy(keep_np & (labels == labels[root]))
