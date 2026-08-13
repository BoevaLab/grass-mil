"""Contracts for the encoder's input embeddings.

The published encoder sums a learned cell-type embedding with a projection of
the continuous node features, and conditions message passing on the scalar edge
length alone.
"""

from __future__ import annotations

import pytest
import torch

from grass_mil.models.components.embeddings import (
    CategoricalEmbeddingConfig,
    NodeInputEmbedding,
    resolve_categorical_embedding_config,
)


def _categorical(**overrides):
    params = {"label": "cell_type", "num_embeddings": 5, "column_index": 0}
    params.update(overrides)
    return CategoricalEmbeddingConfig(**params)


def test_node_embedding_sums_rather_than_concatenates() -> None:
    layer = NodeInputEmbedding(input_dim=3, hidden_dim=8, categorical=_categorical())
    x = torch.randn(6, 3)
    codes = torch.randint(0, 5, (6, 1))

    out = layer(x, codes)
    assert out.shape == (6, 8), "embedding must be summed into the hidden dim, not concatenated"

    expected = layer.linear(x) + layer.embedding(codes[:, 0])
    torch.testing.assert_close(out, expected)


def test_node_embedding_supports_cell_type_only_cohorts() -> None:
    """`x` legitimately has zero columns when no continuous features exist."""
    layer = NodeInputEmbedding(input_dim=0, hidden_dim=8, categorical=_categorical())
    codes = torch.randint(0, 5, (4, 1))

    out = layer(torch.zeros(4, 0), codes)
    assert out.shape == (4, 8)
    torch.testing.assert_close(out, layer.embedding(codes[:, 0]))


def test_node_embedding_without_categorical_matches_plain_projection() -> None:
    """Existing configs must be unaffected by the new embedding path."""
    identity = NodeInputEmbedding(input_dim=8, hidden_dim=8, categorical=None)
    x = torch.randn(5, 8)
    torch.testing.assert_close(identity(x), x)

    projected = NodeInputEmbedding(input_dim=4, hidden_dim=8, categorical=None)
    torch.testing.assert_close(projected(x[:, :4]), projected.linear(x[:, :4]))


def test_node_embedding_reserves_an_unassigned_row() -> None:
    reserved = NodeInputEmbedding(
        input_dim=2, hidden_dim=4, categorical=_categorical(reserve_unassigned=True)
    )
    assert reserved.embedding.num_embeddings == 6
    assert reserved.unassigned_index == 5

    plain = NodeInputEmbedding(
        input_dim=2, hidden_dim=4, categorical=_categorical(reserve_unassigned=False)
    )
    assert plain.embedding.num_embeddings == 5
    assert plain.unassigned_index is None


def test_node_embedding_rejects_out_of_range_codes() -> None:
    """Clamping would silently train on the wrong cell type."""
    layer = NodeInputEmbedding(
        input_dim=2, hidden_dim=4, categorical=_categorical(reserve_unassigned=False)
    )
    codes = torch.tensor([[0], [99]])
    with pytest.raises(ValueError, match="out of range"):
        layer(torch.randn(2, 2), codes)


def test_node_embedding_requires_codes_when_configured() -> None:
    layer = NodeInputEmbedding(input_dim=2, hidden_dim=4, categorical=_categorical())
    with pytest.raises(ValueError, match="no `categorical_codes`"):
        layer(torch.randn(2, 2), None)


def test_zero_input_dim_without_categorical_is_rejected() -> None:
    with pytest.raises(ValueError, match="no input features"):
        NodeInputEmbedding(input_dim=0, hidden_dim=4, categorical=None)


def test_resolve_categorical_config_prefers_explicit_values() -> None:
    resolved = resolve_categorical_embedding_config(
        {"label": "cell_type", "num_embeddings": 3},
        cardinalities={"cell_type": 17},
        slices={"cell_type": 2},
    )
    assert resolved.num_embeddings == 3, "explicit config must win over inference"
    assert resolved.column_index == 2, "unset fields are inferred"


def test_resolve_categorical_config_reports_unresolvable_label() -> None:
    with pytest.raises(ValueError, match="vocabulary size"):
        resolve_categorical_embedding_config({"label": "missing"}, cardinalities={}, slices={})


def _encoder(**overrides):
    from grass_mil.models.components.backbones import EncoderConfig, GNNEncoder

    params = {
        "input_dim": 2,
        "hidden_dim": 8,
        "out_dim": 8,
        "num_layers": 2,
        "conv_type": "gine",
        "norm": "layernorm",
        "pooling": "max",
        "use_edge_attr": True,
        "edge_attr_dim": 1,
        "edge_feature_index": 0,
    }
    params.update(overrides)
    return GNNEncoder(EncoderConfig(**params))


def _toy_graph(num_edge_features: int = 2):
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    x = torch.randn(4, 2)
    edge_attr = torch.rand(edge_index.size(1), num_edge_features)
    codes = torch.randint(0, 5, (4, 1))
    return x, edge_index, edge_attr, codes


def test_gine_reads_only_the_selected_edge_column() -> None:
    """The conv is conditioned on edge length; other columns must not leak in.

    This is the fidelity assertion for the edge path: perturbing any
    non-selected column must leave the output bit-identical.
    """
    torch.manual_seed(0)
    encoder = _encoder().eval()
    x, edge_index, edge_attr, _ = _toy_graph(num_edge_features=3)

    with torch.no_grad():
        baseline = encoder(x, edge_index, edge_attr=edge_attr)

        perturbed = edge_attr.clone()
        perturbed[:, 1:] += 17.0
        after = encoder(x, edge_index, edge_attr=perturbed)
    assert torch.equal(baseline, after)

    with torch.no_grad():
        length_changed = edge_attr.clone()
        length_changed[:, 0] += 17.0
        moved = encoder(x, edge_index, edge_attr=length_changed)
    assert not torch.allclose(baseline, moved), "the selected length column must matter"


def test_encoder_embeds_cell_type_from_categorical_codes() -> None:
    torch.manual_seed(0)
    encoder = _encoder(
        categorical_embedding={"label": "cell_type", "num_embeddings": 5, "column_index": 0}
    ).eval()
    x, edge_index, edge_attr, codes = _toy_graph()

    with torch.no_grad():
        baseline = encoder(x, edge_index, edge_attr=edge_attr, categorical_codes=codes)
        shuffled = encoder(x, edge_index, edge_attr=edge_attr, categorical_codes=(codes + 1) % 5)
    assert baseline.shape == (4, 8)
    assert not torch.allclose(baseline, shuffled), "cell type must influence the encoding"


def test_encoder_rejects_edge_feature_index_for_non_gine_convs() -> None:
    with pytest.raises(ValueError, match="conv_type='gin'"):
        _encoder(conv_type="gin", edge_attr_dim=None, use_edge_attr=False)


def test_encoder_requires_unit_edge_dim_when_selecting_one_column() -> None:
    with pytest.raises(ValueError, match="edge_attr_dim"):
        _encoder(edge_attr_dim=2)
