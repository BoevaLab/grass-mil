from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


@dataclass
class GraphBuildResult:
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    edge_attr_names: Optional[List[str]] = None


class GraphBuilder:
    def build(self, coords: np.ndarray) -> GraphBuildResult:
        raise NotImplementedError


@dataclass
class GraphBuilderConfig:
    name: str
    kwargs: Dict[str, object]

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GraphBuilderConfig":
        return cls(
            name=data["name"],
            kwargs=data.get("kwargs", {}),
        )


_GRAPH_BUILDERS: Dict[str, Callable[..., GraphBuilder]] = {}


def register_graph_builder(name: str):
    def decorator(cls: Callable[..., GraphBuilder]):
        _GRAPH_BUILDERS[name] = cls
        return cls

    return decorator


def get_graph_builder(config: GraphBuilderConfig) -> GraphBuilder:
    if config.name in _GRAPH_BUILDERS:
        return _GRAPH_BUILDERS[config.name](**(config.kwargs or {}))
    if "." in config.name:
        module_name, class_name = config.name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**(config.kwargs or {}))
    raise ValueError(
        f"Unknown graph builder '{config.name}'. Available: {list(_GRAPH_BUILDERS)}"
    )


@register_graph_builder("delaunay")
class DelaunayGraphBuilder(GraphBuilder):
    """Delaunay triangulation over cell centroids.

    Edge features (``edge_features``):

    - ``distance``: raw Euclidean distance in coordinate units.
    - ``log_distance``: ``clip((ln(d + eps) - log_lower) / (log_upper - log_lower), 0, 1)``,
      the bounded log-distance used as the GINE edge input in the GRASS-MIL preprint
      (defaults ``log_lower=2.0``, ``log_upper=5.0``; with coordinates in pixels this maps
      ~7 px -> 0 and ~148 px -> 1).
    - ``neighbor``: 1 if ``distance <= neighbor_cutoff_um`` else 0.
    """

    def __init__(
        self,
        edge_features: Optional[List[str]] = None,
        neighbor_cutoff_um: Optional[float] = None,
        log_lower: float = 2.0,
        log_upper: float = 5.0,
        log_eps: float = 1e-5,
    ) -> None:
        self.edge_features = edge_features or []
        self.neighbor_cutoff_um = neighbor_cutoff_um
        if log_upper <= log_lower:
            raise ValueError("log_upper must be greater than log_lower")
        self.log_lower = float(log_lower)
        self.log_upper = float(log_upper)
        self.log_eps = float(log_eps)

    def build(self, coords: np.ndarray) -> GraphBuildResult:
        if coords.shape[0] < 2:
            return GraphBuildResult(edge_index=torch.empty((2, 0), dtype=torch.long))
        if coords.shape[0] < 3:
            edges = {(0, 1), (1, 0)} if coords.shape[0] == 2 else set()
            edge_index = _edges_to_tensor(edges)
            edge_attr, edge_attr_names = _build_edge_attr(
                edge_index,
                coords,
                self.edge_features,
                self.neighbor_cutoff_um,
                log_lower=self.log_lower,
                log_upper=self.log_upper,
                log_eps=self.log_eps,
            )
            return GraphBuildResult(
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_attr_names=edge_attr_names,
            )

        from scipy.spatial import Delaunay

        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a = int(simplex[i])
                    b = int(simplex[j])
                    edges.add((a, b))
                    edges.add((b, a))
        edge_index = _edges_to_tensor(edges)
        edge_attr, edge_attr_names = _build_edge_attr(
            edge_index,
            coords,
            self.edge_features,
            self.neighbor_cutoff_um,
            log_lower=self.log_lower,
            log_upper=self.log_upper,
            log_eps=self.log_eps,
        )
        return GraphBuildResult(
            edge_index=edge_index, edge_attr=edge_attr, edge_attr_names=edge_attr_names
        )


def _edges_to_tensor(edges: set[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index


def log_distance_feature(
    distances: np.ndarray, log_lower: float = 2.0, log_upper: float = 5.0, eps: float = 1e-5
) -> np.ndarray:
    """Bounded log-distance edge feature: clip((ln(d + eps) - lower) / (upper - lower), 0, 1)."""
    log_d = np.log(np.asarray(distances, dtype=np.float64) + eps)
    return np.clip((log_d - log_lower) / (log_upper - log_lower), 0.0, 1.0).astype(np.float32)


def _build_edge_attr(
    edge_index: torch.Tensor,
    coords: np.ndarray,
    edge_features: List[str],
    neighbor_cutoff_um: Optional[float],
    log_lower: float = 2.0,
    log_upper: float = 5.0,
    log_eps: float = 1e-5,
) -> tuple[Optional[torch.Tensor], Optional[List[str]]]:
    if not edge_features:
        return None, None
    if edge_index.numel() == 0:
        return torch.empty((0, len(edge_features)), dtype=torch.float), list(
            edge_features
        )

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    deltas = coords[src] - coords[dst]
    distances = np.linalg.norm(deltas, axis=1)

    columns: List[np.ndarray] = []
    for feat in edge_features:
        if feat == "distance":
            columns.append(distances.astype(np.float32))
        elif feat == "log_distance":
            columns.append(log_distance_feature(distances, log_lower, log_upper, log_eps))
        elif feat == "neighbor":
            if neighbor_cutoff_um is None:
                raise ValueError(
                    "neighbor_cutoff_um must be set to compute neighbor edge feature"
                )
            columns.append((distances <= neighbor_cutoff_um).astype(np.float32))
        else:
            raise ValueError(f"Unsupported edge feature '{feat}'")

    edge_attr = np.stack(columns, axis=1)
    return torch.from_numpy(edge_attr), list(edge_features)
