import pytest
import torch


def _toy_dataset(n_graphs: int = 3, n_nodes: int = 20):
    from torch_geometric.data import Data

    graphs = []
    for _ in range(n_graphs):
        edge_index = torch.randint(0, n_nodes, (2, 64), dtype=torch.long)
        x = torch.randn(n_nodes, 8)
        edge_attr = torch.rand(edge_index.size(1), 2)
        graphs.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)
        )
    return graphs


def _toy_dataset_with_graph_attrs(n_graphs: int = 2, n_nodes: int = 20):
    from torch_geometric.data import Data

    graphs = []
    for i in range(n_graphs):
        edge_index = torch.randint(0, n_nodes, (2, 64), dtype=torch.long)
        x = torch.randn(n_nodes, 8)
        edge_attr = torch.rand(edge_index.size(1), 2)
        graph_y = torch.tensor([[float(i % 2)]], dtype=torch.float32)
        graph_w = torch.ones_like(graph_y)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_y=graph_y,
            graph_w=graph_w,
            num_nodes=n_nodes,
        )
        data.sample_id = f"sample_{i}"
        data.region_id = f"region_{i}"
        data.patch_id = f"patch_{i}"
        graphs.append(data)
    return graphs


def _attach_cell_type_metadata(data, labels):
    data.categorical_codes = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    data.categorical_slices = {"cell_type": 0}
    return data


def test_sequence_assignment_preserves_schema_name_lists_without_nesting():
    from grass_mil.data.components.samplers import _select_sequence_value_for_subgraph

    # Typical Batch collation of list metadata from 2 subgraphs.
    value = [["edge_type", "distance"], ["edge_type", "distance"]]
    out = _select_sequence_value_for_subgraph(
        key="edge_attr_names",
        value=value,
        subgraph_index=0,
        num_subgraphs=2,
    )
    assert out == ["edge_type", "distance"]


def test_sequence_assignment_does_not_slice_flat_schema_name_lists_by_length():
    from grass_mil.data.components.samplers import _select_sequence_value_for_subgraph

    # Coincidental length match with num_subgraphs should not imply per-subgraph slicing.
    value = ["edge_type", "distance"]
    out = _select_sequence_value_for_subgraph(
        key="edge_attr_names",
        value=value,
        subgraph_index=0,
        num_subgraphs=2,
    )
    assert out == ["edge_type", "distance"]


def test_sequence_assignment_still_slices_per_subgraph_non_schema_lists():
    from grass_mil.data.components.samplers import _select_sequence_value_for_subgraph

    value = ["sample_0", "sample_1"]
    out = _select_sequence_value_for_subgraph(
        key="sample_id",
        value=value,
        subgraph_index=1,
        num_subgraphs=2,
    )
    assert out == ["sample_1"]


def test_identity_sampler_strategy():
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    dataset = _toy_dataset()
    strategy = get_sampler_strategy({"name": "identity", "kwargs": {}})
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert hasattr(batch, "x")
    assert hasattr(batch, "edge_index")


def test_shadow_native_requires_no_post_transform():
    pytest.importorskip("torch_geometric")
    try:
        import torch_sparse  # noqa: F401
    except Exception:
        pytest.skip("torch_sparse is not available")

    from grass_mil.data.components.samplers import get_sampler_strategy

    data = _toy_dataset(n_graphs=1)[0]
    strategy = get_sampler_strategy({"name": "shadow_native", "kwargs": {}})
    with pytest.raises(ValueError):
        strategy.build_unit_loader(
            data=data,
            depth=2,
            num_neighbors=8,
            batch_size=4,
            transform=lambda x: x,
        )


def test_shadow_custom_unit_loader():
    pytest.importorskip("torch_geometric")
    try:
        import torch_sparse  # noqa: F401
    except Exception:
        pytest.skip("torch_sparse is not available")

    from grass_mil.data.components.samplers import get_sampler_strategy

    data = _toy_dataset(n_graphs=1)[0]
    strategy = get_sampler_strategy({"name": "shadow_custom", "kwargs": {}})
    loader = strategy.build_unit_loader(
        data=data,
        depth=2,
        num_neighbors=8,
        batch_size=4,
        transform=lambda d: d,
    )
    batch = next(iter(loader))
    assert hasattr(batch, "x")
    assert hasattr(batch, "edge_index")


def test_shadow_custom_without_torch_sparse_rejects_transform(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components import samplers as samplers_mod
    from grass_mil.data.components.samplers import ShadowCustomStrategy

    data = _toy_dataset(n_graphs=1)[0]
    monkeypatch.setattr(samplers_mod, "WITH_TORCH_SPARSE", False)
    strategy = ShadowCustomStrategy(runtime={"enabled": False})
    with pytest.raises(ValueError, match="requires torch-sparse"):
        strategy.build_unit_loader(
            data=data,
            depth=2,
            num_neighbors=8,
            batch_size=4,
            transform=lambda d: d,
        )


def test_shadow_custom_runtime_dataset_loader_uses_unit_loader(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    dataset = _toy_dataset_with_graph_attrs(n_graphs=2)
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "replace": False,
                "shuffle_subgraphs": True,
            },
        }
    )

    call_counter = {"n": 0}

    def _fake_unit_loader(data, **kwargs):
        call_counter["n"] += 1
        return [data]

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)

    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    batches = list(loader)
    assert call_counter["n"] == len(dataset)
    assert len(batches) == len(dataset)


def test_shadow_runtime_normalizes_graph_level_metadata(monkeypatch):
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch

    from grass_mil.data.components.samplers import get_sampler_strategy

    dataset = _toy_dataset_with_graph_attrs(n_graphs=1)
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {"enabled": True, "depth": 2, "num_neighbors": 8},
        }
    )

    def _fake_unit_loader(data, **kwargs):
        b = Batch.from_data_list([data, data])
        # Simulate graph-level attrs carried as singletons before normalization.
        b.sample_id = data.sample_id
        b.region_id = data.region_id
        b.patch_id = data.patch_id
        b.graph_y = data.graph_y
        b.graph_w = data.graph_w
        return [b]

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)

    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    batch = next(iter(loader))
    assert isinstance(batch.sample_id, list) and len(batch.sample_id) == 2
    assert isinstance(batch.region_id, list) and len(batch.region_id) == 2
    assert batch.graph_y.size(0) == 2
    assert batch.graph_w.size(0) == 2


def test_runtime_loader_len_matches_yielded_batches(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    dataset = _toy_dataset_with_graph_attrs(n_graphs=2, n_nodes=6)
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "proportional_root_sampling": False,
            },
        }
    )

    def _fake_unit_loader(data, **kwargs):
        roots = kwargs.get("node_idx")
        if roots is None:
            n_roots = int(data.num_nodes)
        else:
            n_roots = int(roots.numel())
        bs = int(kwargs["batch_size"])
        n_batches = (n_roots + bs - 1) // bs
        return [data] * n_batches

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    yielded = sum(1 for _ in loader)
    assert len(loader) == yielded


def test_runtime_shadow_config_proportional_defaults():
    from grass_mil.data.components.samplers import RuntimeShadowConfig

    cfg = RuntimeShadowConfig.from_dict({})
    assert cfg.proportional_root_sampling is True
    assert cfg.property_name == "cell_type"
    assert cfg.weight_mode == "inverse"
    assert cfg.min_weight > 0


def test_shadow_runtime_passes_weighted_node_idx(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    unit = _toy_dataset_with_graph_attrs(n_graphs=1, n_nodes=6)[0]
    unit = _attach_cell_type_metadata(unit, [0, 0, 0, 0, 1, 1])
    dataset = [unit]
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "proportional_root_sampling": True,
                "property_name": "cell_type",
                "weight_mode": "inverse",
            },
        }
    )

    observed = {"node_idx": None}

    def _fake_unit_loader(data, **kwargs):
        observed["node_idx"] = kwargs.get("node_idx")
        return [data]

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    _ = list(loader)
    node_idx = observed["node_idx"]
    assert isinstance(node_idx, torch.Tensor)
    assert node_idx.dtype == torch.long
    assert node_idx.numel() == unit.num_nodes


def test_shadow_runtime_falls_back_to_none_when_property_missing(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    unit = _toy_dataset_with_graph_attrs(n_graphs=1, n_nodes=6)[0]
    dataset = [unit]
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "proportional_root_sampling": True,
                "property_name": "cell_type",
                "weight_mode": "inverse",
                "node_idx": None,
            },
        }
    )

    observed = {"node_idx": "unset", "unit_num_nodes": int(unit.num_nodes)}

    def _fake_unit_loader(data, **kwargs):
        observed["node_idx"] = kwargs.get("node_idx")
        return [data]

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    _ = list(loader)
    node_idx = observed["node_idx"]
    assert isinstance(node_idx, torch.Tensor)
    assert node_idx.dtype == torch.long
    assert node_idx.numel() == observed["unit_num_nodes"]
    assert torch.equal(
        node_idx, torch.arange(observed["unit_num_nodes"], dtype=torch.long)
    )


def test_shadow_runtime_can_disable_proportional_sampling(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    unit = _toy_dataset_with_graph_attrs(n_graphs=1, n_nodes=6)[0]
    unit = _attach_cell_type_metadata(unit, [0, 0, 0, 0, 1, 1])
    dataset = [unit]
    explicit_node_idx = torch.tensor([0, 1], dtype=torch.long)
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "proportional_root_sampling": False,
                "property_name": "cell_type",
                "weight_mode": "inverse",
                "node_idx": explicit_node_idx,
            },
        }
    )

    observed = {"node_idx": None}

    def _fake_unit_loader(data, **kwargs):
        observed["node_idx"] = kwargs.get("node_idx")
        return [data]

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    _ = list(loader)
    assert torch.equal(observed["node_idx"], explicit_node_idx)


@pytest.mark.parametrize("weight_mode", ["inverse", "sqrt_inverse", "proportional"])
def test_shadow_runtime_accepts_weight_modes(weight_mode):
    from grass_mil.data.components.samplers import RuntimeShadowConfig

    cfg = RuntimeShadowConfig.from_dict({"weight_mode": weight_mode})
    assert cfg.weight_mode == weight_mode


def test_runtime_shadow_config_accepts_subsample_defaults():
    from grass_mil.data.components.samplers import RuntimeShadowConfig

    cfg = RuntimeShadowConfig.from_dict({})
    assert cfg.subsample_fraction == 1.0
    assert cfg.subsample_seed is None


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_runtime_shadow_config_rejects_invalid_subsample_fraction(fraction):
    from grass_mil.data.components.samplers import RuntimeShadowConfig

    with pytest.raises(ValueError, match="subsample_fraction"):
        RuntimeShadowConfig.from_dict({"subsample_fraction": fraction})


def test_shadow_runtime_subsamples_roots_with_seed(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    unit = _toy_dataset_with_graph_attrs(n_graphs=1, n_nodes=10)[0]
    dataset = [unit]
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "proportional_root_sampling": False,
                "subsample_fraction": 0.4,
                "subsample_seed": 17,
            },
        }
    )

    observed = {"node_idx": None}

    def _fake_unit_loader(data, **kwargs):
        observed["node_idx"] = kwargs.get("node_idx")
        return [data]

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    _ = list(loader)
    first = observed["node_idx"]
    assert isinstance(first, torch.Tensor)
    assert first.dtype == torch.long
    assert int(first.numel()) == 4

    observed["node_idx"] = None
    _ = list(loader)
    second = observed["node_idx"]
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, second)


def test_shadow_runtime_subsampled_len_matches_yielded_batches(monkeypatch):
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.samplers import get_sampler_strategy

    dataset = _toy_dataset_with_graph_attrs(n_graphs=2, n_nodes=10)
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 3,
                "proportional_root_sampling": False,
                "subsample_fraction": 0.5,
                "subsample_seed": 11,
            },
        }
    )

    def _fake_unit_loader(data, **kwargs):
        roots = kwargs.get("node_idx")
        n_roots = int(roots.numel()) if roots is not None else int(data.num_nodes)
        bs = int(kwargs["batch_size"])
        n_batches = (n_roots + bs - 1) // bs
        return [data] * n_batches

    monkeypatch.setattr(strategy, "build_unit_loader", _fake_unit_loader)
    loader = strategy.build_dataset_loader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    yielded = sum(1 for _ in loader)
    assert len(loader) == yielded


def test_shadow_custom_per_subgraph_transform_preserves_node_level_tensors():
    """Node-level tensors must survive per-subgraph transform application.

    ``categorical_codes`` carries the cell-type codes the encoder embeds and
    ``pos`` carries coordinates; neither is recoverable from ``x``. They were
    previously dropped as "already represented by x", which silently removed
    cell type from every sampled subgraph.
    """
    pytest.importorskip("torch_geometric")
    try:
        import torch_sparse  # noqa: F401
    except Exception:
        pytest.skip("torch_sparse is not available")

    from grass_mil.data.components.samplers import get_sampler_strategy

    n_nodes = 20
    data = _toy_dataset(n_graphs=1, n_nodes=n_nodes)[0]
    _attach_cell_type_metadata(data, list(range(n_nodes)))
    data.pos = torch.arange(n_nodes, dtype=torch.float32).view(-1, 1).repeat(1, 2)

    strategy = get_sampler_strategy({"name": "shadow_custom", "kwargs": {}})
    loader = strategy.build_unit_loader(
        data=data,
        depth=2,
        num_neighbors=8,
        batch_size=4,
        transform=lambda d: d,
    )
    batch = next(iter(loader))

    assert hasattr(batch, "categorical_codes"), "cell-type codes were dropped by sampling"
    assert hasattr(batch, "pos"), "coordinates were dropped by sampling"
    assert batch.categorical_codes.size(0) == batch.x.size(0)
    assert batch.pos.size(0) == batch.x.size(0)

    # The codes must be the ones belonging to the sampled nodes, not a stale
    # or truncated slice: node i of the source graph carries code i.
    expected = data.categorical_codes[batch.n_id].view(-1)
    assert torch.equal(batch.categorical_codes.view(-1), expected)


def test_node_level_codes_batch_along_the_node_dimension():
    """Guard the attribute name against PyTorch Geometric's index heuristic.

    ``Data.__cat_dim__`` returns -1 for any key containing ``index``, so an
    attribute named ``categorical_index`` is concatenated along the feature
    dimension when graphs are batched -- silently corrupting per-node codes (or
    raising, when the graphs differ in size). The codes must batch along dim 0
    like any other node attribute.
    """
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch, Data

    graphs = []
    for n_nodes in (4, 7):
        data = Data(x=torch.randn(n_nodes, 3), edge_index=torch.empty(2, 0, dtype=torch.long))
        data.num_nodes = n_nodes
        _attach_cell_type_metadata(data, list(range(n_nodes)))
        graphs.append(data)

    batch = Batch.from_data_list(graphs)
    assert batch.categorical_codes.shape == (11, 1)
    assert torch.equal(
        batch.categorical_codes.view(-1),
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]),
    )
