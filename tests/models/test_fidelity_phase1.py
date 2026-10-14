"""Phase 1 fidelity: graph edge features, encoder epsilon/init, attention init and space."""

import numpy as np
import torch
from torch import nn

from grass_mil.data.components.graph_builders import DelaunayGraphBuilder, log_distance_feature
from grass_mil.models.components.attention import AttnNetGatedProjected
from grass_mil.models.components.backbones import EncoderConfig, GNNEncoder
from grass_mil.models.components.embeddings import (
    CategoricalEmbeddingConfig,
    NodeInputEmbedding,
    resolve_categorical_embedding_config,
)
from grass_mil.models.components.factory import build_loss


def test_log_distance_matches_reference_formula():
    d = np.array([0.0, 7.389, 20.0, 148.41, 1000.0])
    got = log_distance_feature(d, log_lower=2.0, log_upper=5.0)
    expected = np.clip((np.log(d + 1e-5) - 2.0) / 3.0, 0.0, 1.0)
    assert np.allclose(got, expected, atol=1e-6)
    assert got[0] == 0.0 and got[-1] == 1.0


def test_delaunay_builder_emits_log_distance_and_stores_edges_twice():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 500, size=(60, 2))
    builder = DelaunayGraphBuilder(
        edge_features=["log_distance", "neighbor", "distance"], neighbor_cutoff_um=55.0
    )
    res = builder.build(coords)
    assert res.edge_attr_names == ["log_distance", "neighbor", "distance"]
    ei = res.edge_index.numpy()
    pairs = set(map(tuple, ei.T.tolist()))
    assert all((b, a) in pairs for a, b in pairs)
    dist = res.edge_attr[:, 2].numpy()
    assert np.allclose(res.edge_attr[:, 0].numpy(), log_distance_feature(dist), atol=1e-6)
    assert np.array_equal(res.edge_attr[:, 1].numpy(), (dist <= 55.0).astype(np.float32))


def _gine_cfg(**kw):
    base = dict(
        input_dim=3,
        hidden_dim=16,
        out_dim=16,
        num_layers=2,
        conv_type="gine",
        norm="layernorm",
        pooling="max",
        edge_attr_dim=1,
        edge_feature_index=0,
        use_edge_attr=True,
    )
    base.update(kw)
    return EncoderConfig(**base)


def test_gine_eps_is_trainable_by_default_and_fixed_when_disabled():
    enc = GNNEncoder(_gine_cfg())
    assert all(isinstance(layer.eps, nn.Parameter) for layer in enc.layers)
    enc_fixed = GNNEncoder(_gine_cfg(train_eps=False))
    assert all(not isinstance(layer.eps, nn.Parameter) for layer in enc_fixed.layers)


def test_input_embedding_xavier_init_changes_scale_and_default_keeps_torch():
    torch.manual_seed(0)
    cat = CategoricalEmbeddingConfig(label="cell_type", num_embeddings=10, column_index=0)
    xav = NodeInputEmbedding(input_dim=4, hidden_dim=64, categorical=cat, init="xavier")
    torch.manual_seed(0)
    dflt = NodeInputEmbedding(input_dim=4, hidden_dim=64, categorical=cat, init="default")
    # nn.Embedding default is N(0, 1); xavier bound for (11, 64) is sqrt(6 / 75) ~ 0.28.
    assert xav.embedding.weight.abs().max() < 0.5
    assert dflt.embedding.weight.abs().max() > 1.0


def test_unassigned_label_resolves_to_existing_category():
    cfg = {"label": "cell_type", "reserve_unassigned": False, "unassigned_label": "other"}
    resolved = resolve_categorical_embedding_config(
        cfg,
        cardinalities={"cell_type": 3},
        slices={"cell_type": 0},
        label_maps={"cell_type": {"tumor": 0, "other": 1, "immune": 2}},
    )
    assert resolved.unassigned_index == 1
    emb = NodeInputEmbedding(input_dim=0, hidden_dim=8, categorical=resolved)
    assert emb.unassigned_index == 1
    assert emb.embedding.num_embeddings == 3


def test_attention_kaiming_init_and_attention_space():
    torch.manual_seed(0)
    att = AttnNetGatedProjected(
        input_dim=32, projection_dim=8, hidden_dim=4, dropout=True, n_classes=2
    )
    att.eval()
    x = torch.randn(20, 32)
    space = att.attention_space(x)
    assert space.shape == (20, 4)
    manual = att.attention_a[0](att.projection[1](att.projection[0](x)))
    assert torch.allclose(space, manual)
    logits, passthrough = att(x)
    assert logits.shape == (20, 2) and passthrough is x
    # kaiming_uniform (a=0): bound = sqrt(2) * sqrt(3 / fan_in) = sqrt(6 / 32) ~ 0.433
    assert att.projection[0].weight.abs().max() <= np.sqrt(6.0 / 32) + 1e-6
    assert att.projection[0].weight.abs().max() > 1.0 / np.sqrt(32)  # wider than torch default


def test_loss_weights_accept_lists():
    loss = build_loss("categorical_ce", class_weight=[1.0, 3.0])
    assert isinstance(loss.class_weight, torch.Tensor)
    logits = torch.randn(5, 2)
    target = torch.tensor([0, 1, 1, 0, 1])
    assert torch.isfinite(loss(logits, target))
