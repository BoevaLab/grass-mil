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
