"""Contracts for interior seed restriction and the ego-radius cutoff."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from grass_mil.data.components.ego_radius import (
    EgoRadiusConfig,
    apply_ego_radius_cutoff,
    resolve_ego_radius,
)
from grass_mil.data.components.seed_sampling import (
    InteriorSeedConfig,
    InteriorSeedIndexCache,
    compute_interior_seed_indices,
    convex_hull_interior_mask,
)


def _grid(side: int = 9, spacing: float = 1.0):
    """A regular square grid with 4-neighbour connectivity."""
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float) * spacing

    edges = []
    for i in range(side):
        for j in range(side):
            node = i * side + j
            if i + 1 < side:
                edges.append((node, (i + 1) * side + j))
            if j + 1 < side:
                edges.append((node, i * side + (j + 1)))
    edge_index = np.array(edges, dtype=np.int64).T
    return coords, edge_index


def test_interior_mask_excludes_the_hull_rim() -> None:
    side = 9
    coords, edge_index = _grid(side)

    # A tolerance below the grid spacing makes only the cells lying exactly on
    # the hull polygon seeds, so one hop removes exactly the outer ring.
    mask = convex_hull_interior_mask(
        coords, edge_index, n_hops=1, seed_tolerance_multiplier=0.5
    )
    assert mask.sum() == (side - 2) ** 2

    grid_mask = mask.reshape(side, side)
    assert not grid_mask[0].any() and not grid_mask[-1].any()
    assert not grid_mask[:, 0].any() and not grid_mask[:, -1].any()
    assert grid_mask[1:-1, 1:-1].all()


def test_seed_tolerance_widens_the_rim_by_proximity_to_the_hull() -> None:
    """Seeds are cells *near* the hull, not only those exactly on it.

    The tolerance is a multiple of the median nearest-neighbour distance, so at
    the default multiplier a unit-spaced grid seeds its two outermost rings.
    """
    side = 9
    coords, edge_index = _grid(side)

    tight = convex_hull_interior_mask(
        coords, edge_index, n_hops=1, seed_tolerance_multiplier=0.5
    )
    default = convex_hull_interior_mask(coords, edge_index, n_hops=1)

    assert tight.sum() == (side - 2) ** 2
    assert default.sum() == (side - 4) ** 2
    # The wider tolerance removes a strict superset of cells.
    assert not (default & ~tight).any()


def test_interior_shrinks_monotonically_with_hop_count() -> None:
    coords, edge_index = _grid(11)
    sizes = [
        int(convex_hull_interior_mask(coords, edge_index, n_hops=hops).sum())
        for hops in (1, 2, 3)
    ]
    assert sizes[0] > sizes[1] > sizes[2]


def test_interior_mask_falls_back_to_all_nodes_on_degenerate_input() -> None:
    # Collinear points have no area hull, and tiny graphs have no meaningful rim.
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    assert convex_hull_interior_mask(collinear, edge_index, n_hops=2).all()

    tiny = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert convex_hull_interior_mask(tiny, np.empty((2, 0), dtype=np.int64), n_hops=2).all()

    # A non-positive rim width disables the restriction entirely.
    coords, edges = _grid(6)
    assert convex_hull_interior_mask(coords, edges, n_hops=0).all()


def test_interior_mask_can_drop_disconnected_islands() -> None:
    """An interior island cut off from the main body is folded into the rim."""
    coords, edge_index = _grid(9)
    # Sever the centre cell from its neighbours: it survives the rim test but is
    # no longer connected to the interior body.
    centre = 4 * 9 + 4
    keep = ~((edge_index[0] == centre) | (edge_index[1] == centre))
    severed = edge_index[:, keep]

    with_islands = convex_hull_interior_mask(
        coords, severed, n_hops=1, keep_largest_component=False
    )
    without_islands = convex_hull_interior_mask(
        coords, severed, n_hops=1, keep_largest_component=True
    )
    assert with_islands[centre]
    assert not without_islands[centre]
    assert without_islands.sum() == with_islands.sum() - 1


def test_compute_interior_seed_indices_falls_back_without_coordinates() -> None:
    from torch_geometric.data import Data

    data = Data(x=torch.randn(5, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
    data.num_nodes = 5
    indices = compute_interior_seed_indices(
        data, InteriorSeedConfig(enabled=True), n_hops=2
    )
    assert indices.tolist() == list(range(5))


def test_interior_seed_cache_round_trip_and_corruption_recovery(tmp_path) -> None:
    cache = InteriorSeedIndexCache(str(tmp_path))
    indices = np.array([1, 2, 3], dtype=np.int64)
    cache.store("region_a__hops2", indices)
    np.testing.assert_array_equal(cache.load("region_a__hops2"), indices)

    assert cache.load("missing") is None

    # A truncated cache entry must trigger recomputation, not an exception.
    corrupt = tmp_path / "region_a__hops2.npy"
    corrupt.write_bytes(b"not a numpy file")
    assert cache.load("region_a__hops2") is None


def test_disabled_cache_is_a_no_op() -> None:
    cache = InteriorSeedIndexCache(None)
    cache.store("anything", np.array([1]))
    assert cache.load("anything") is None


def test_resolve_ego_radius_is_linear_in_depth() -> None:
    # The published NSCLC setting: r_k = 75k + 55, so r_4 = 355.
    assert resolve_ego_radius(4, radius_per_hop=75.0, radius_offset=55.0) == 355.0
    assert resolve_ego_radius(0, radius_per_hop=75.0, radius_offset=55.0) == 55.0
    with pytest.raises(ValueError):
        resolve_ego_radius(-1, radius_per_hop=75.0, radius_offset=55.0)


def _line_graph(n: int = 6, spacing: float = 10.0):
    from torch_geometric.data import Data

    pos = torch.arange(n, dtype=torch.float32).unsqueeze(-1) * spacing
    pos = torch.cat([pos, torch.zeros_like(pos)], dim=1)
    src = torch.arange(n - 1, dtype=torch.long)
    edge_index = torch.stack([src, src + 1], dim=0)
    data = Data(
        x=torch.randn(n, 3),
        edge_index=edge_index,
        edge_attr=torch.rand(edge_index.size(1), 2),
    )
    data.pos = pos
    data.num_nodes = n
    data.root_n_id = torch.tensor([0], dtype=torch.long)
    data.categorical_codes = torch.arange(n, dtype=torch.long).view(-1, 1)
    return data


def test_ego_radius_cutoff_drops_distant_nodes_and_keeps_the_root() -> None:
    data = _line_graph(n=6, spacing=10.0)
    out = apply_ego_radius_cutoff(data, radius=25.0)

    # Nodes at 0, 10, 20 are within the radius; 30, 40, 50 are not.
    assert int(out.num_nodes) == 3
    assert int(out.root_n_id.item()) == 0
    torch.testing.assert_close(out.pos[:, 0], torch.tensor([0.0, 10.0, 20.0]))
    # Node-level tensors are sliced consistently with x.
    assert out.categorical_codes.view(-1).tolist() == [0, 1, 2]
    assert out.edge_index.max() < out.num_nodes


def test_ego_radius_cutoff_is_a_no_op_when_everything_fits() -> None:
    data = _line_graph(n=4, spacing=1.0)
    out = apply_ego_radius_cutoff(data, radius=1000.0)
    assert int(out.num_nodes) == 4
    assert out is data


def test_ego_radius_cutoff_can_drop_nodes_disconnected_from_the_root() -> None:
    """Within the radius but unreachable: an island, not a neighbourhood."""
    from torch_geometric.data import Data

    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    # Node 2 is close to the root but only connected to node 1... which we omit.
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data = Data(x=torch.randn(3, 2), edge_index=edge_index)
    data.pos = pos
    data.num_nodes = 3
    data.root_n_id = torch.tensor([0], dtype=torch.long)

    kept = apply_ego_radius_cutoff(data, radius=10.0, keep_root_component=False)
    assert int(kept.num_nodes) == 3

    pruned = apply_ego_radius_cutoff(data, radius=10.0, keep_root_component=True)
    assert int(pruned.num_nodes) == 2


def test_ego_radius_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown ego radius keys"):
        EgoRadiusConfig.from_dict({"enabled": True, "radius": 10.0})


def test_interior_seed_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown interior seed keys"):
        InteriorSeedConfig.from_dict({"enabled": True, "hops": 2})
