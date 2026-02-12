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


def test_identity_sampler_strategy():
    pytest.importorskip("torch_geometric")
    from src.data.components.samplers import get_sampler_strategy

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

    from src.data.components.samplers import get_sampler_strategy

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

    from src.data.components.samplers import get_sampler_strategy

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


def test_shadow_custom_runtime_dataset_loader_uses_unit_loader(monkeypatch):
    pytest.importorskip("torch_geometric")
    from src.data.components.samplers import get_sampler_strategy

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

    from src.data.components.samplers import get_sampler_strategy

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
