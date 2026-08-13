"""Contracts for the BGRL view generators."""

from __future__ import annotations

import pytest
import torch

from grass_mil.models.training.augmentations import (
    build_augmentation,
    degree_importance,
    drop_edges,
    drop_importance,
    feature_noise,
    resolve_feature_indices,
)


def _star_graph(n_leaves: int = 8):
    """A hub with many neighbours and leaves with exactly one."""
    from torch_geometric.data import Data

    leaves = torch.arange(1, n_leaves + 1, dtype=torch.long)
    hub = torch.zeros_like(leaves)
    edge_index = torch.cat([torch.stack([hub, leaves]), torch.stack([leaves, hub])], dim=1)
    n_nodes = n_leaves + 1
    data = Data(
        x=torch.ones(n_nodes, 3),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 2),
    )
    data.num_nodes = n_nodes
    data.categorical_codes = torch.zeros((n_nodes, 1), dtype=torch.long)
    return data


def _generator(seed: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def test_degree_importance_ranks_hubs_above_leaves() -> None:
    data = _star_graph(n_leaves=8)
    importance = degree_importance(data.edge_index, int(data.num_nodes))
    assert importance[0] > importance[1:].max()
    # Mean-max normalisation clamps every below-average node to zero.
    assert float(importance[1:].max()) == pytest.approx(0.0)
    assert float(importance.max()) == pytest.approx(1.0)


def test_drop_importance_preserves_high_degree_nodes() -> None:
    """Low-degree nodes are dropped far more often than the hub."""
    data = _star_graph(n_leaves=12)
    generator = _generator(0)

    hub_dropped = 0
    leaf_dropped = 0
    trials = 40
    for _ in range(trials):
        out = drop_importance(data, mu=1.0, p_lambda=1.0, unassigned_index=7, generator=generator)
        dropped = out.categorical_codes.view(-1) == 7
        hub_dropped += int(dropped[0])
        leaf_dropped += int(dropped[1:].sum())

    assert hub_dropped == 0, "the hub has maximal importance and must never drop"
    assert leaf_dropped > 0


def test_drop_importance_marks_dropped_nodes_unassigned_and_zeroes_features() -> None:
    data = _star_graph(n_leaves=10)
    out = drop_importance(data, mu=1.0, p_lambda=1.0, unassigned_index=99, generator=_generator(1))
    dropped = out.categorical_codes.view(-1) == 99
    assert bool(dropped.any())
    assert torch.all(out.x[dropped] == 0)
    assert torch.all(out.x[~dropped] == 1)


def test_drop_importance_does_not_mutate_the_source_graph() -> None:
    """Guards the view-vs-clone class of bug.

    The legacy implementation wrote through a view, so a fixup applied to the
    augmented graph silently reverted on the original.
    """
    data = _star_graph(n_leaves=10)
    x_before = data.x.clone()
    codes_before = data.categorical_codes.clone()
    edges_before = data.edge_index.clone()

    drop_importance(data, mu=1.0, p_lambda=1.0, unassigned_index=5, generator=_generator(2))

    torch.testing.assert_close(data.x, x_before)
    torch.testing.assert_close(data.categorical_codes, codes_before)
    torch.testing.assert_close(data.edge_index, edges_before)


def test_drop_importance_keeps_the_graph_undirected() -> None:
    """Dropping one direction would make message passing direction-dependent."""
    from torch_geometric.utils import is_undirected

    data = _star_graph(n_leaves=16)
    for seed in range(5):
        out = drop_importance(
            data, mu=1.0, p_lambda=1.0, unassigned_index=3, generator=_generator(seed)
        )
        assert is_undirected(out.edge_index), f"asymmetric edges at seed {seed}"
        assert out.edge_attr.shape[0] == out.edge_index.shape[1]


def test_drop_edges_forces_undirected_by_default() -> None:
    from torch_geometric.utils import is_undirected

    data = _star_graph(n_leaves=16)
    out = drop_edges(data, p=0.5, generator=_generator(0))
    assert is_undirected(out.edge_index)
    assert out.edge_attr.shape[0] == out.edge_index.shape[1]


def test_augmentations_are_deterministic_under_a_fixed_generator() -> None:
    data = _star_graph(n_leaves=12)
    first = drop_importance(
        data, mu=0.5, p_lambda=0.5, unassigned_index=4, generator=_generator(7)
    )
    second = drop_importance(
        data, mu=0.5, p_lambda=0.5, unassigned_index=4, generator=_generator(7)
    )
    torch.testing.assert_close(first.x, second.x)
    torch.testing.assert_close(first.edge_index, second.edge_index)


def test_feature_noise_touches_only_the_selected_columns() -> None:
    data = _star_graph(n_leaves=4)
    out = feature_noise(data, std=1.0, feature_indices=[1], generator=_generator(0))

    torch.testing.assert_close(out.x[:, 0], data.x[:, 0])
    torch.testing.assert_close(out.x[:, 2], data.x[:, 2])
    assert not torch.allclose(out.x[:, 1], data.x[:, 1])


def test_feature_noise_rejects_out_of_range_columns() -> None:
    data = _star_graph(n_leaves=4)
    with pytest.raises(ValueError, match="out of range"):
        feature_noise(data, std=1.0, feature_indices=[9], generator=_generator(0))


def test_feature_columns_resolve_by_name() -> None:
    names = ["cell_type", "SIZE", "other"]
    assert resolve_feature_indices(["SIZE"], feature_names=names) == [1]
    # Integer indices remain a usable fallback.
    assert resolve_feature_indices([2], feature_names=names) == [2]
    with pytest.raises(ValueError, match="not found"):
        resolve_feature_indices(["MISSING"], feature_names=names)


def test_build_augmentation_dispatches_on_mode() -> None:
    data = _star_graph(n_leaves=8)

    importance = build_augmentation(
        {"mode": "importance", "mu": 0.5, "p_lambda": 0.5},
        unassigned_index=6,
    )
    assert importance(data, _generator(0)) is not data

    uniform = build_augmentation({"mode": "uniform", "drop_edge_p": 0.5})
    assert uniform(data, _generator(0)) is not data

    with pytest.raises(ValueError, match="Unsupported augmentation mode"):
        build_augmentation({"mode": "nonsense"})


@pytest.mark.parametrize("mu,p_lambda", [(0.0, 0.5), (1.5, 0.5), (0.5, 0.0), (0.5, 1.5)])
def test_drop_importance_validates_its_hyperparameters(mu: float, p_lambda: float) -> None:
    with pytest.raises(ValueError):
        drop_importance(_star_graph(), mu=mu, p_lambda=p_lambda)
