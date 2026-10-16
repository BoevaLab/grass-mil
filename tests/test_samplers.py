"""Ego-graph sampling."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from grass_mil.data.datamodule import GraphUnitDataset
from grass_mil.data.precompute import UnitEntry
from grass_mil.data.samplers import EgoGraphLoader, SamplerConfig


class _Units(GraphUnitDataset):
    """In-memory units with the dataset interface the loader needs."""

    def __init__(self, graphs):
        super().__init__(
            [UnitEntry(g.unit_id, g.region_id, g.sample_id, "", int(g.num_nodes)) for g in graphs]
        )
        self.graphs = graphs

    def __getitem__(self, idx):
        return self.graphs[idx]


def _grid_unit(name: str, side: int = 8, y: float = 1.0) -> Data:
    xs, ys = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([xs.flatten(), ys.flatten()], 1).float() * 10
    edges = []
    for i in range(side * side):
        r, c = divmod(i, side)
        if c + 1 < side:
            edges += [(i, i + 1), (i + 1, i)]
        if r + 1 < side:
            edges += [(i, i + side), (i + side, i)]
    edge_index = torch.tensor(edges).t()
    n = side * side
    return Data(
        x=torch.arange(n, dtype=torch.float).view(-1, 1),
        pos=pos,
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 1),
        cell_type=torch.arange(n) % 3,
        num_nodes=n,
        unit_id=name,
        region_id=name,
        sample_id=name,
        graph_y=torch.tensor([[y]]),
    )


def _loader(units, mode, **cfg):
    return EgoGraphLoader(
        _Units(units),
        SamplerConfig.from_dict({"depth": 2, "batch_size": 5, **cfg}),
        mode=mode,
        roots_per_unit=cfg.get("roots_per_unit"),
    )


def test_bags_hold_one_unit_each_with_its_target():
    units = [_grid_unit("a", y=0.0), _grid_unit("b", y=1.0)]
    batches = list(_loader(units, "bags", roots_per_unit=7))
    assert len(batches) == 2
    for batch in batches:
        assert batch.num_graphs == 7
        assert len(set(batch.unit_id)) == 1
        assert batch.graph_y.shape == (7, 1)


def test_ego_graphs_are_the_k_hop_neighbourhoods_of_their_roots():
    unit = _grid_unit("a")
    for batch in _loader([unit], "chunks"):
        for i in range(batch.num_graphs):
            root = int(batch.root_cell[i])
            nodes, _, _, _ = k_hop_subgraph(root, 2, unit.edge_index, num_nodes=unit.num_nodes)
            sub = batch.x[batch.batch == i].view(-1).long()
            assert sorted(sub.tolist()) == sorted(nodes.tolist())
            assert int(batch.x[batch.root_n_id[i]]) == root


def test_chunks_score_every_cell_once_and_len_matches():
    loader = _loader([_grid_unit("a"), _grid_unit("b")], "chunks")
    batches = list(loader)
    assert len(batches) == len(loader)
    for name in "ab":
        roots = torch.cat([b.root_cell for b in batches if b.unit_id[0] == name])
        assert sorted(roots.tolist()) == list(range(64))


def test_fixed_seed_draws_the_same_roots_every_pass():
    units = [_grid_unit("a")]
    loader = EgoGraphLoader(
        _Units(units),
        SamplerConfig.from_dict({"depth": 1}),
        mode="chunks",
        roots_per_unit=10,
        seed=3,
    )
    first = torch.cat([b.root_cell for b in loader])
    second = torch.cat([b.root_cell for b in loader])
    assert torch.equal(first, second) and first.numel() == 10


def test_root_weighting_draws_distinct_roots():
    loader = EgoGraphLoader(
        _Units([_grid_unit("a")]),
        SamplerConfig.from_dict({"depth": 1, "root_weighting": "inverse"}),
        mode="bags",
        roots_per_unit=20,
    )
    roots = next(iter(loader)).root_cell
    assert roots.unique().numel() == 20


def test_merged_mode_mixes_units():
    loader = _loader(
        [_grid_unit("a"), _grid_unit("b")], "merged", roots_per_unit=None, merge_units=True
    )
    batches = list(loader)
    assert len(batches) == len(loader) == -(-128 // 5)
    assert sum(b.num_graphs for b in batches) == 128


def test_interior_roots_exclude_the_rim():
    loader = _loader(
        [_grid_unit("a", side=10)], "chunks", interior_roots={"enabled": True, "n_hops": 1}
    )
    roots = torch.cat([b.root_cell for b in loader])
    rows, cols = roots // 10, roots % 10
    assert roots.numel() < 100
    assert bool(((rows > 0) & (rows < 9) & (cols > 0) & (cols < 9)).all())


def test_ego_radius_keeps_the_root_and_bounds_the_extent():
    loader = _loader(
        [_grid_unit("a")],
        "chunks",
        ego_radius={"enabled": True, "radius_per_hop": 5.0, "radius_offset": 0.0},
    )
    for batch in loader:
        for i in range(batch.num_graphs):
            nodes = batch.batch == i
            root = batch.root_n_id[i]
            assert bool(nodes[root])
            dist = torch.linalg.norm(batch.pos[nodes] - batch.pos[root], dim=1)
            assert float(dist.max()) <= 10.0 + 1e-6


def test_unknown_sampler_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        SamplerConfig.from_dict({"subsample_count": 3})
