"""Graph augmentations for self-supervised pretraining.

BGRL needs two stochastic views of each ego-graph. The production views combine
a degree-importance-weighted node/edge drop with additive Gaussian noise on the
continuous cell-size feature.

Ported from ``graph_augmentations_phenotype`` in the spatial-augmentations
tree. Two deviations from that code are deliberate and marked below; one
deviation from the manuscript is deliberate too -- the manuscript describes a
min-max degree normalisation, but the final run code uses mean-max, and the
code is authoritative.

Every function is pure: it takes an explicit ``torch.Generator`` and returns a
new graph rather than mutating its input.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch

__all__ = [
    "drop_importance",
    "drop_edges",
    "drop_features",
    "feature_noise",
    "build_augmentation",
    "degree_importance",
]


def _clone(data):
    return data.clone()


def degree_importance(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-node importance from log degree, normalised to ``[0, 1]``.

    Uses the legacy **mean-max** normalisation ``(v - mean) / (max - mean)``
    clamped at zero, which maps every below-average-degree node to 0. The
    manuscript describes min-max; the final run code does this, and the code is
    the reference.
    """
    from torch_geometric.utils import degree

    deg = degree(edge_index[0], num_nodes).float()
    log_deg = torch.log1p(deg)
    importance = (log_deg - log_deg.mean()) / (log_deg.max() - log_deg.mean() + 1e-8)
    return torch.clamp(importance, min=0.0)


def _drop_probability(importance: torch.Tensor, *, mu: float, p_lambda: float) -> torch.Tensor:
    """``min((1 - importance) * mu, p_lambda)``.

    Note the legacy name for this quantity was ``keep_prob``, but it is used as
    ``rand > p`` to decide retention, so it is the *drop* probability. The
    behaviour is preserved; only the name is corrected.
    """
    ceiling = torch.full_like(importance, float(p_lambda))
    return torch.minimum((1.0 - importance) * float(mu), ceiling)


def _mutual_edge_mask(edge_index: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Restrict ``keep`` to edges whose reverse also survives.

    Dropping one direction of an undirected edge leaves an asymmetric graph and
    makes message passing direction-dependent.
    """
    if edge_index.numel() == 0:
        return keep
    num_nodes = int(edge_index.max()) + 1
    src, dst = edge_index[0], edge_index[1]
    kept_keys = (src[keep] * num_nodes + dst[keep]).to(torch.int64)
    reverse_keys = (dst * num_nodes + src).to(torch.int64)
    return keep & torch.isin(reverse_keys, kept_keys)


def drop_importance(
    data,
    *,
    mu: float,
    p_lambda: float,
    categorical_column: Optional[int] = 0,
    unassigned_index: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
):
    """Drop low-degree nodes and edges, keeping the graph undirected.

    Dropped nodes keep their position in the graph but have their features
    zeroed and their cell type set to the reserved "unassigned" code, so the
    encoder sees a masked node rather than a spurious cell type.
    """
    if not 0.0 < mu <= 1.0:
        raise ValueError(f"mu must be within (0, 1], got {mu}.")
    if not 0.0 < p_lambda <= 1.0:
        raise ValueError(f"p_lambda must be within (0, 1], got {p_lambda}.")

    out = _clone(data)
    edge_index = out.edge_index
    num_nodes = int(getattr(out, "num_nodes", out.x.size(0)))
    importance = degree_importance(edge_index, num_nodes)

    node_drop_p = _drop_probability(importance, mu=mu, p_lambda=p_lambda)
    draws = torch.rand(node_drop_p.shape, generator=generator, device=node_drop_p.device)
    dropped = ~(draws > node_drop_p)

    if out.x is not None and out.x.numel():
        # Clone before the in-place write: `x` may be a view of the source
        # graph, in which case masking here would corrupt the original.
        x = out.x.clone()
        x[dropped] = 0
        out.x = x

    codes = getattr(out, "categorical_codes", None)
    if codes is not None and unassigned_index is not None and categorical_column is not None:
        codes = codes.clone()
        codes[dropped, int(categorical_column)] = int(unassigned_index)
        out.categorical_codes = codes

    if edge_index.numel():
        edge_importance = (importance[edge_index[0]] + importance[edge_index[1]]) / 2.0
        edge_importance = (edge_importance - edge_importance.mean()) / (
            edge_importance.max() - edge_importance.mean() + 1e-8
        )
        edge_importance = torch.clamp(edge_importance, min=0.0)
        edge_drop_p = _drop_probability(edge_importance, mu=mu, p_lambda=p_lambda)
        edge_draws = torch.rand(edge_drop_p.shape, generator=generator, device=edge_drop_p.device)
        keep = edge_draws > edge_drop_p
        keep = _mutual_edge_mask(edge_index, keep)

        out.edge_index = edge_index[:, keep]
        edge_attr = getattr(out, "edge_attr", None)
        if edge_attr is not None:
            out.edge_attr = edge_attr[keep]
    return out


def drop_edges(
    data,
    *,
    p: float,
    force_undirected: bool = True,
    generator: Optional[torch.Generator] = None,
):
    """Uniformly drop edges.

    ``force_undirected`` keeps both directions of an edge together: the legacy
    finetune transform left it off while pretraining had it on, so the two
    regimes augmented differently and message passing became
    direction-dependent.

    The draw is taken here rather than through ``torch_geometric.utils
    .dropout_edge``, which offers no generator argument and would silently
    consume the global RNG, making seeded runs irreproducible.
    """
    if p <= 0.0:
        return data
    if not 0.0 <= p < 1.0:
        raise ValueError(f"drop probability must be within [0, 1), got {p}.")

    out = _clone(data)
    edge_index = out.edge_index
    if edge_index.numel() == 0:
        return out

    draws = torch.rand((edge_index.size(1),), generator=generator, device=edge_index.device)
    keep = draws >= float(p)
    if force_undirected:
        keep = _mutual_edge_mask(edge_index, keep)

    out.edge_index = edge_index[:, keep]
    edge_attr = getattr(out, "edge_attr", None)
    if edge_attr is not None:
        out.edge_attr = edge_attr[keep]
    return out


def drop_features(data, *, p: float, generator: Optional[torch.Generator] = None):
    """Zero whole feature columns with probability ``p``."""
    if p <= 0.0 or data.x is None or data.x.numel() == 0:
        return data
    out = _clone(data)
    x = out.x.clone()
    draws = torch.rand((x.size(1),), generator=generator, device=x.device)
    x[:, draws < float(p)] = 0
    out.x = x
    return out


def feature_noise(
    data,
    *,
    std: float,
    feature_indices: Sequence[int],
    generator: Optional[torch.Generator] = None,
):
    """Add Gaussian noise to selected continuous feature columns."""
    if std <= 0.0 or data.x is None or data.x.numel() == 0:
        return data
    indices = [int(i) for i in feature_indices]
    if not indices:
        return data
    out = _clone(data)
    x = out.x.clone()
    for index in indices:
        if index < 0 or index >= x.size(1):
            raise ValueError(
                f"feature noise column {index} is out of range for x with "
                f"{x.size(1)} column(s)."
            )
        noise = torch.randn(x[:, index].shape, generator=generator, device=x.device)
        x[:, index] = x[:, index] + noise * float(std)
    out.x = x
    return out


def resolve_feature_indices(
    columns: Optional[Sequence[Any]],
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> list[int]:
    """Resolve feature columns by name, falling back to integer indices.

    Resolving by name matters: the legacy default noised a hardcoded column
    index, which pointed at cell size in one cohort and at an unrelated feature
    in another, feeding pure noise to the encoder.
    """
    if not columns:
        return []
    names = list(feature_names or [])
    resolved: list[int] = []
    for column in columns:
        if isinstance(column, int):
            resolved.append(int(column))
            continue
        name = str(column)
        if name not in names:
            raise ValueError(
                f"Feature column '{name}' not found in molecular feature names {names}. "
                "Set the column by integer index if the datamodule does not record names."
            )
        resolved.append(names.index(name))
    return resolved


def build_augmentation(
    cfg: Optional[Mapping[str, Any]],
    *,
    feature_names: Optional[Sequence[str]] = None,
    unassigned_index: Optional[int] = None,
    categorical_column: Optional[int] = 0,
) -> Callable[[Any, Optional[torch.Generator]], Any]:
    """Build a view-generating transform from config.

    Returns a callable ``(data, generator) -> data``.
    """
    cfg = dict(cfg or {})
    mode = str(cfg.get("mode", "uniform")).strip().lower()

    if mode == "importance":
        mu = float(cfg.get("mu", 0.25))
        p_lambda = float(cfg.get("p_lambda", 0.45))
        std = float(cfg.get("feature_noise_std", 0.0))
        indices = resolve_feature_indices(
            cfg.get("feature_noise_columns"), feature_names=feature_names
        )

        def _importance(data, generator: Optional[torch.Generator] = None):
            out = drop_importance(
                data,
                mu=mu,
                p_lambda=p_lambda,
                categorical_column=categorical_column,
                unassigned_index=unassigned_index,
                generator=generator,
            )
            if std > 0.0 and indices:
                out = feature_noise(out, std=std, feature_indices=indices, generator=generator)
            return out

        return _importance

    if mode == "uniform":
        drop_edge_p = float(cfg.get("drop_edge_p", 0.0))
        drop_feat_p = float(cfg.get("drop_feat_p", 0.0))

        def _uniform(data, generator: Optional[torch.Generator] = None):
            out = drop_features(data, p=drop_feat_p, generator=generator)
            return drop_edges(out, p=drop_edge_p, force_undirected=True, generator=generator)

        return _uniform

    raise ValueError(
        f"Unsupported augmentation mode '{mode}'. Expected one of: 'importance', 'uniform'."
    )
