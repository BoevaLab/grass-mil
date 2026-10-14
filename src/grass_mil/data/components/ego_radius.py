"""Euclidean radius cutoff for k-hop ego-graphs.

A k-hop ego-graph can reach far across sparse tissue, so hop depth alone does
not bound its physical extent. The published sampler additionally caps the
ego-graph at a radius that grows linearly with depth:

.. math:: r_k = \\mathrm{radius\\_per\\_hop} \\cdot k + \\mathrm{radius\\_offset}

Distances are in the datamodule's coordinate units (micrometres once
``coord_scale_um`` has been applied), *not* pixels: the published 75/55 are
pixel values at the NSCLC imaging scale and must be converted per cohort.

Applied as a post-hoc transform on sampled subgraphs, so no sampler internals
are involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

__all__ = ["EgoRadiusConfig", "EgoRadiusCutoff", "resolve_ego_radius", "apply_ego_radius_cutoff"]


@dataclass(frozen=True)
class EgoRadiusConfig:
    enabled: bool = False
    radius_per_hop: float = 75.0
    radius_offset: float = 55.0
    keep_root_component: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Mapping[str, Any]]) -> "EgoRadiusConfig":
        cfg = dict(cfg or {})
        allowed = {"enabled", "radius_per_hop", "radius_offset", "keep_root_component"}
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(
                f"Unknown ego radius keys: {sorted(unknown)}. "
                f"Expected a subset of {sorted(allowed)}."
            )
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            radius_per_hop=float(cfg.get("radius_per_hop", 75.0)),
            radius_offset=float(cfg.get("radius_offset", 55.0)),
            keep_root_component=bool(cfg.get("keep_root_component", True)),
        )


def resolve_ego_radius(depth: int, *, radius_per_hop: float, radius_offset: float) -> float:
    """``r_k = radius_per_hop * k + radius_offset``."""
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}.")
    return float(radius_per_hop) * float(depth) + float(radius_offset)


def _root_index(data) -> Optional[int]:
    root = getattr(data, "root_n_id", None)
    if root is None:
        return None
    if isinstance(root, torch.Tensor):
        if root.numel() == 0:
            return None
        return int(root.reshape(-1)[0])
    return int(root)


def apply_ego_radius_cutoff(data, *, radius: float, keep_root_component: bool = True):
    """Drop nodes farther than ``radius`` from the ego-graph root.

    The root is always retained. When ``keep_root_component`` is set, nodes left
    disconnected from the root by the cut are dropped too, so the ego-graph
    stays a single connected neighbourhood rather than the root plus islands.
    """
    from torch_geometric.utils import subgraph

    pos = getattr(data, "pos", None)
    if pos is None or radius <= 0:
        return data
    root = _root_index(data)
    if root is None:
        return data

    coords = pos[:, :2].float()
    num_nodes = int(coords.size(0))
    if num_nodes == 0 or root >= num_nodes:
        return data

    distances = torch.linalg.norm(coords - coords[root].unsqueeze(0), dim=1)
    keep = distances <= float(radius)
    keep[root] = True

    if keep_root_component and bool(keep.any()):
        keep = _root_connected_mask(data, keep, root=root, num_nodes=num_nodes)

    if bool(keep.all()):
        return data

    keep_idx = torch.where(keep)[0]
    edge_attr = getattr(data, "edge_attr", None)
    new_edge_index, new_edge_attr = subgraph(
        subset=keep_idx,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    out = data.clone()
    out.edge_index = new_edge_index
    if edge_attr is not None:
        out.edge_attr = new_edge_attr
    for key, value in data:
        if key in {"edge_index", "edge_attr"}:
            continue
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == num_nodes:
            setattr(out, key, value[keep_idx])
    out.num_nodes = int(keep_idx.numel())

    # Root position shifts when earlier nodes are dropped.
    remap = torch.full((num_nodes,), -1, dtype=torch.long)
    remap[keep_idx] = torch.arange(keep_idx.numel(), dtype=torch.long)
    out.root_n_id = remap[root].reshape(1)
    return out


def _root_connected_mask(data, keep: torch.Tensor, *, root: int, num_nodes: int) -> torch.Tensor:
    """Restrict ``keep`` to the connected component containing the root."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keep_np = keep.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    if edge_index.size == 0:
        mask = np.zeros(num_nodes, dtype=bool)
        mask[root] = True
        return torch.from_numpy(mask)

    src, dst = edge_index[0], edge_index[1]
    both_kept = keep_np[src] & keep_np[dst]
    src, dst = src[both_kept], dst[both_kept]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adjacency = coo_matrix(
        (np.ones(rows.shape[0], dtype=bool), (rows, cols)),
        shape=(num_nodes, num_nodes),
    ).tocsr()

    _, labels = connected_components(adjacency, directed=False)
    mask = keep_np & (labels == labels[root])
    mask[root] = True
    return torch.from_numpy(mask)


class EgoRadiusCutoff:
    """Transform form of :func:`apply_ego_radius_cutoff`.

    Constructed with an explicit radius so the transform itself carries no
    dependency on sampler internals.
    """

    def __init__(self, *, radius: float, keep_root_component: bool = True) -> None:
        self.radius = float(radius)
        self.keep_root_component = bool(keep_root_component)

    def __call__(self, data):
        return apply_ego_radius_cutoff(
            data,
            radius=self.radius,
            keep_root_component=self.keep_root_component,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(radius={self.radius}, "
            f"keep_root_component={self.keep_root_component})"
        )
