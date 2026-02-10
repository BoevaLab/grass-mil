import pytest
import torch


def _toy_graph(num_nodes: int = 12):
    from torch_geometric.data import Data

    row = torch.randint(0, num_nodes, (36,), dtype=torch.long)
    col = torch.randint(0, num_nodes, (36,), dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    x = torch.randn(num_nodes, 16)
    edge_attr = torch.rand(edge_index.size(1), 2)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


@pytest.mark.parametrize("conv_type", ["gin", "gcn", "gat", "graphsage"])
def test_encoder_forward_node_and_graph(conv_type):
    pytest.importorskip("torch_geometric")
    from src.models.components.backbones import EncoderConfig, GNNEncoder

    data = _toy_graph()
    cfg = EncoderConfig(
        input_dim=16,
        hidden_dim=32,
        out_dim=32,
        num_layers=3,
        conv_type=conv_type,
        pooling="mean",
        use_edge_attr=True if conv_type == "gcn" else False,
    )
    encoder = GNNEncoder(cfg)
    node_emb, graph_emb = encoder(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        batch=data.batch,
        return_graph_embedding=True,
    )
    assert node_emb.shape == (data.num_nodes, 32)
    assert graph_emb.shape == (1, encoder.graph_pool.output_dim)


@pytest.mark.parametrize("jk", ["last", "concat", "max", "sum"])
def test_encoder_jk_modes(jk):
    pytest.importorskip("torch_geometric")
    from src.models.components.backbones import EncoderConfig, GNNEncoder

    data = _toy_graph()
    cfg = EncoderConfig(
        input_dim=16,
        hidden_dim=24,
        out_dim=None,
        num_layers=2,
        conv_type="gin",
        jk=jk,
        pooling=None,
    )
    encoder = GNNEncoder(cfg)
    node_emb = encoder(data.x, data.edge_index, data.edge_attr)
    expected_dim = 24 * 3 if jk == "concat" else 24
    assert node_emb.shape == (data.num_nodes, expected_dim)
