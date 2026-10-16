"""Cell graphs: Delaunay triangulation over cell centroids, with edge features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from grass_mil.utils import checked

__all__ = ["GraphConfig", "delaunay_edges", "edge_features", "log_distance_feature"]

EDGE_FEATURES = ("log_distance", "distance", "neighbor")


@dataclass(frozen=True)
class GraphConfig:
    """Edge features of the cell graph.

    Attributes:
        edge_features: Any of ``log_distance`` (bounded log edge length),
            ``distance`` (raw length) and ``neighbor`` (1 when the length is at
            most ``neighbor_cutoff``), in that column order.
        neighbor_cutoff: Contact distance for ``neighbor``, in coordinate units
            (after ``coord_scale_um``).
        log_distance_bounds: ``(lower, upper)`` of the log length mapped to
            ``[0, 1]``: ``clip((ln d - lower) / (upper - lower), 0, 1)``. The
            default ``(2, 5)`` maps lengths of about 7 to 0 and 150 to 1.
    """

    edge_features: tuple[str, ...] = ("log_distance", "neighbor")
    neighbor_cutoff: float = 20.0
    log_distance_bounds: tuple[float, float] = (2.0, 5.0)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> GraphConfig:
        cfg = checked(cls, cfg, "data.graph")
        out = cls(
            edge_features=tuple(cfg.get("edge_features", cls.edge_features)),
            neighbor_cutoff=float(cfg.get("neighbor_cutoff", cls.neighbor_cutoff)),
            log_distance_bounds=tuple(cfg.get("log_distance_bounds", cls.log_distance_bounds)),
        )
        bad = [f for f in out.edge_features if f not in EDGE_FEATURES]
        if bad:
            raise ValueError(f"Unknown edge feature(s) {bad}; expected {list(EDGE_FEATURES)}.")
        lower, upper = out.log_distance_bounds
        if upper <= lower:
            raise ValueError("data.graph.log_distance_bounds must be increasing.")
        return out


def delaunay_edges(coords: np.ndarray) -> torch.Tensor:
    """Undirected Delaunay edges as a ``[2, E]`` index with both directions."""
    n = int(coords.shape[0])
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    if n < 3:
        pairs = np.array([[0, 1]])
    else:
        from scipy.spatial import Delaunay

        simplices = Delaunay(coords).simplices
        pairs = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    both = np.concatenate([pairs, pairs[:, ::-1]])
    return torch.from_numpy(np.ascontiguousarray(both.T)).long()


def log_distance_feature(distances: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Bounded log edge length ``clip((ln d - lower) / (upper - lower), 0, 1)``."""
    log_d = np.log(np.maximum(np.asarray(distances, dtype=np.float64), 1e-12))
    return np.clip((log_d - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def edge_features(coords: np.ndarray, edge_index: torch.Tensor, cfg: GraphConfig) -> torch.Tensor:
    """``[E, len(cfg.edge_features)]`` edge attributes in the configured order."""
    src, dst = edge_index.numpy()
    lengths = np.linalg.norm(coords[src] - coords[dst], axis=1)
    columns = {
        "log_distance": lambda: log_distance_feature(lengths, *cfg.log_distance_bounds),
        "distance": lambda: lengths.astype(np.float32),
        "neighbor": lambda: (lengths <= cfg.neighbor_cutoff).astype(np.float32),
    }
    stacked = np.stack([columns[name]() for name in cfg.edge_features], axis=1)
    return torch.from_numpy(stacked.reshape(len(lengths), len(cfg.edge_features)))
