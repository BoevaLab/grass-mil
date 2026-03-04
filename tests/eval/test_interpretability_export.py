from __future__ import annotations

import math

import pytest
import torch

from src.inference.interpretability_export import build_instance_table, build_spatial_table
from src.inference.schemas import BatchPredictionPayload


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


def test_build_spatial_table_constructs_knn_edges_within_bag() -> None:
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
        instance_centroids=torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
    )
    instance_frame = build_instance_table(payload, require_composition=True)
    spatial = build_spatial_table(instance_frame, n_neighbors=1, undirected=True)
    assert set(spatial.columns) == {"source_id", "target_id", "distance"}
    # Undirected expansion guarantees reverse edges.
    assert ("p0", "p1") in set(zip(spatial["source_id"], spatial["target_id"]))
    assert ("p1", "p0") in set(zip(spatial["source_id"], spatial["target_id"]))
