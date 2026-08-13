from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from grass_mil.inference.interpretability_export import build_instance_table, build_spatial_table
from grass_mil.inference.schemas import BatchPredictionPayload


def test_build_instance_table_adds_uniform_attention_and_unique_ids() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5], [1.2]]),
        instance_patch_ids=["p0", "p0", "p1"],
        instance_bag_ids=["b0", "b0", "b1"],
        instance_region_ids=["r0", "r0", "r1"],
        instance_sample_ids=["s0", "s0", "s1"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )
    frame = build_instance_table(payload, require_composition=True)
    assert list(frame["instance_id"]) == ["p0", "p0::1", "p1"]
    assert "attention" in frame.columns
    # Bag b0 has two instances with uniform fallback attention.
    assert frame.loc[frame["bag_id"] == "b0", "attention"].tolist() == [0.5, 0.5]
    assert frame.loc[frame["bag_id"] == "b1", "attention"].tolist() == [1.0]
    expected_scores = [1.0 / (1.0 + math.exp(-v)) for v in [0.2, 0.5, 1.2]]
    assert frame["score"].tolist() == pytest.approx(expected_scores)
    assert "inst_emb_0" in frame.columns
    assert "comp_0" in frame.columns


def test_build_instance_table_supports_identity_score_mode() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5], [1.2]]),
        instance_patch_ids=["p0", "p1", "p2"],
        instance_bag_ids=["b0", "b0", "b1"],
        instance_region_ids=["r0", "r0", "r1"],
        instance_sample_ids=["s0", "s0", "s1"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )
    frame = build_instance_table(
        payload,
        require_composition=True,
        score_mode="identity",
        score_logit_index=0,
    )
    assert frame["score"].tolist() == pytest.approx([0.2, 0.5, 1.2])


def test_build_instance_table_supports_softmax_score_mode() -> None:
    logits = torch.tensor([[2.0, 1.0], [0.5, 3.0], [1.5, 1.5]])
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0, 0.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=logits,
        instance_patch_ids=["p0", "p1", "p2"],
        instance_bag_ids=["b0", "b0", "b1"],
        instance_region_ids=["r0", "r0", "r1"],
        instance_sample_ids=["s0", "s0", "s1"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )
    frame = build_instance_table(
        payload,
        require_composition=True,
        score_mode="softmax",
        score_logit_index=1,
    )
    expected_scores = torch.softmax(logits, dim=1)[:, 1].tolist()
    assert frame["score"].tolist() == pytest.approx(expected_scores)


def test_build_instance_table_rejects_unknown_score_mode() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5]]),
        instance_patch_ids=["p0", "p1"],
        instance_bag_ids=["b0", "b0"],
        instance_region_ids=["r0", "r0"],
        instance_sample_ids=["s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    )
    with pytest.raises(ValueError, match="Unsupported score_mode"):
        build_instance_table(payload, require_composition=True, score_mode="unknown")


def test_build_instance_table_rejects_out_of_bounds_score_index() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5]]),
        instance_patch_ids=["p0", "p1"],
        instance_bag_ids=["b0", "b0"],
        instance_region_ids=["r0", "r0"],
        instance_sample_ids=["s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    )
    with pytest.raises(ValueError, match="score_logit_index is out of bounds"):
        build_instance_table(payload, require_composition=True, score_logit_index=3)


def test_build_spatial_table_uses_payload_connectivity_edges() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5], [1.2]]),
        instance_patch_ids=["p0", "p1", "p2"],
        instance_bag_ids=["b0", "b0", "b0"],
        instance_region_ids=["r0", "r0", "r0"],
        instance_sample_ids=["s0", "s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        instance_graphs=[
            SimpleNamespace(
                root_n_id=torch.tensor([0], dtype=torch.long),
                n_id=torch.tensor([100, 101], dtype=torch.long),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([[1.0]], dtype=torch.float32),
                edge_attr_names=["distance"],
            ),
            SimpleNamespace(
                root_n_id=torch.tensor([0], dtype=torch.long),
                n_id=torch.tensor([101, 100, 102], dtype=torch.long),
                edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
                edge_attr=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                edge_attr_names=["distance"],
            ),
            SimpleNamespace(
                root_n_id=torch.tensor([0], dtype=torch.long),
                n_id=torch.tensor([102, 101], dtype=torch.long),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([[2.0]], dtype=torch.float32),
                edge_attr_names=["distance"],
            ),
        ],
    )
    instance_frame = build_instance_table(payload, require_composition=True)
    spatial = build_spatial_table(instance_frame, payload, undirected=True)
    assert set(spatial.columns) == {"source_id", "target_id", "distance"}
    # Undirected expansion guarantees reverse edges.
    assert ("p0", "p1") in set(zip(spatial["source_id"], spatial["target_id"]))
    assert ("p1", "p0") in set(zip(spatial["source_id"], spatial["target_id"]))


def test_build_spatial_table_requires_payload_connectivity() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5]]),
        instance_patch_ids=["p0", "p1"],
        instance_bag_ids=["b0", "b0"],
        instance_region_ids=["r0", "r0"],
        instance_sample_ids=["s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    )
    instance_frame = build_instance_table(payload, require_composition=True)
    with pytest.raises(ValueError, match="missing instance_graphs"):
        build_spatial_table(instance_frame, payload)


def test_build_instance_table_exports_per_class_attention_columns() -> None:
    """Per-class attention is exported, not rejected.

    The attribution identity needs one attention column per class, plus the raw
    pre-softmax logits (which, unlike the within-bag normalised scores, are
    comparable across bags).
    """
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0, 0.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2, 0.7], [0.5, 0.1]]),
        instance_attention_logits=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_patch_ids=["p0", "p1"],
        instance_bag_ids=["b0", "b0"],
        instance_region_ids=["r0", "r0"],
        instance_sample_ids=["s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    )
    table = build_instance_table(payload, require_composition=True)

    for column in ("attention", "attention_c0", "attention_c1"):
        assert column in table.columns
    assert "attention_logit_c0" in table.columns

    # Each class column is a distribution over the bag's instances.
    assert table["attention_c0"].sum() == pytest.approx(1.0)
    assert table["attention_c1"].sum() == pytest.approx(1.0)
    # The canonical `attention` column defaults to the last class.
    assert table["attention"].tolist() == pytest.approx(table["attention_c1"].tolist())
    # Raw logits are passed through unnormalised.
    assert table["attention_logit_c0"].tolist() == pytest.approx([0.1, 0.3])


def test_build_instance_table_attention_class_index_is_selectable() -> None:
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0, 0.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2, 0.7], [0.5, 0.1]]),
        instance_attention_logits=torch.tensor([[0.1, 5.0], [0.3, 0.4]]),
        instance_patch_ids=["p0", "p1"],
        instance_bag_ids=["b0", "b0"],
        instance_region_ids=["r0", "r0"],
        instance_sample_ids=["s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_composition=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    )
    table = build_instance_table(payload, require_composition=True, attention_class_index=0)
    assert table["attention"].tolist() == pytest.approx(table["attention_c0"].tolist())

    with pytest.raises(ValueError, match="out of range"):
        build_instance_table(payload, require_composition=True, attention_class_index=5)


def _instance_graph(codes, n_id, root_global):
    """Minimal stand-in for a sampled ego-graph."""
    from torch_geometric.data import Data

    g = Data(x=torch.zeros(len(n_id), 1))
    g.categorical_codes = torch.tensor(codes).view(-1, 1)
    g.n_id = torch.tensor(n_id)
    g.root_n_id = torch.tensor([root_global])
    g.root_n_id_is_global = torch.tensor([True])
    g.categorical_slices = {"cell_type": 0}
    return g


def test_build_instance_table_emits_the_root_cell_type() -> None:
    """One label per instance: the type of the cell it is rooted at.

    There is exactly one instance per cell -- an instance is that cell's k-hop
    ego-graph -- so the instance table is a cell table. `filtration_curves` and
    `per_niche_cell_type_enrichment` both need this column and raise without it.

    It must be the root cell's own type. Summarising the neighbourhood instead
    (its dominant type, say) would quietly turn cell-level statistics into
    statistics over neighbourhood labels. The neighbourhood composition is
    exported separately as `comp_*`.
    """
    # Instance 0 is rooted at global node 7, whose code is 1 ("Tcell"), even
    # though its neighbourhood is mostly code 0 ("Epithelial").
    graphs = [
        _instance_graph(codes=[0, 0, 1], n_id=[5, 6, 7], root_global=7),
        _instance_graph(codes=[0, 1, 1], n_id=[8, 9, 10], root_global=8),
    ]
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2], [0.5]]),
        instance_patch_ids=["p0", "p1"],
        instance_bag_ids=["b0", "b0"],
        instance_region_ids=["r0", "r0"],
        instance_sample_ids=["s0", "s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        instance_composition=torch.tensor([[0.67, 0.33], [0.33, 0.67]]),
        instance_graphs=graphs,
    )
    frame = build_instance_table(
        payload,
        require_composition=True,
        composition_column_names=["comp_Epithelial", "comp_Tcell"],
    )
    assert "cell_type" in frame.columns
    # Root types, not the neighbourhood majority: instance 0 is majority
    # Epithelial but rooted at a Tcell, and instance 1 the other way round.
    assert frame["cell_type"].tolist() == ["Tcell", "Epithelial"]


def test_instance_cell_type_column_can_be_disabled() -> None:
    graphs = [_instance_graph(codes=[0, 1], n_id=[3, 4], root_global=3)]
    payload = BatchPredictionPayload(
        bag_ids=["b0"],
        bag_logits=torch.tensor([[1.0]]),
        bag_targets=None,
        bag_attention=None,
        instance_logits=torch.tensor([[0.2]]),
        instance_patch_ids=["p0"],
        instance_bag_ids=["b0"],
        instance_region_ids=["r0"],
        instance_sample_ids=["s0"],
        instance_embeddings=torch.tensor([[0.1, 0.2]]),
        instance_composition=torch.tensor([[0.7, 0.3]]),
        instance_graphs=graphs,
    )
    frame = build_instance_table(payload, require_composition=True, cell_type_column=None)
    assert "cell_type" not in frame.columns
