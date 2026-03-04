from __future__ import annotations

import pytest
import torch

from src.inference.collectors import collect_inference_payload


class _FakeTrainer:
    def __init__(self, outputs):
        self._outputs = outputs

    def predict(self, model=None, datamodule=None, ckpt_path=None):
        return self._outputs


def test_collect_inference_payload_normalizes_predictions() -> None:
    outputs = [
        {
            "bag_ids": ["region_1", "region_2"],
            "bag_logits": torch.tensor([[1.0], [0.0]]),
            "bag_targets": torch.tensor([[1.0], [0.0]]),
            "row_region_ids": ["region_1", "region_2"],
            "row_sample_ids": ["sample_a", "sample_b"],
        },
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[2.0]]),
            "bag_targets": torch.tensor([[1.0]]),
            "bag_attention": {"region_1": torch.tensor([0.2, 0.8])},
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "instance_logits": torch.tensor([[1.5], [2.5]]),
            "instance_attention_logits": torch.tensor([[0.1], [0.2]]),
            "instance_patch_ids": ["region_1", "region_1"],
            "instance_region_ids": ["region_1", "region_1"],
            "instance_sample_ids": ["sample_a", "sample_a"],
        },
    ]
    trainer = _FakeTrainer(outputs)
    collected = collect_inference_payload(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
    )
    payload = collected.prediction_payload
    assert payload.bag_logits.shape[0] == 3
    assert payload.bag_targets is not None
    assert payload.bag_attention is not None
    assert payload.bag_ids[0] == "region_1"
    assert len(payload.bag_attention) == 3
    assert payload.bag_attention[0] is None
    assert payload.bag_attention[1] is None
    assert torch.allclose(payload.bag_attention[2], torch.tensor([0.2, 0.8]))
    assert payload.row_region_ids == ["region_1", "region_2", "region_1"]
    assert payload.row_sample_ids == ["sample_a", "sample_b", "sample_a"]
    assert payload.instance_logits is not None
    assert payload.instance_logits.shape == (2, 1)
    assert payload.instance_patch_ids == ["region_1", "region_1"]
    assert collected.embedding_payload is None


def test_collect_inference_payload_can_skip_instance_payload_collection() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            # Intentionally malformed instance payload to ensure collector ignores it.
            "instance_logits": torch.tensor([[1.0], [2.0]]),
            "instance_patch_ids": ["patch_1"],
            "instance_region_ids": ["region_1"],
            "instance_sample_ids": ["sample_a"],
        }
    ]
    trainer = _FakeTrainer(outputs)
    collected = collect_inference_payload(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
        include_instance_payload=False,
    )
    payload = collected.prediction_payload
    assert payload.bag_logits.shape == (1, 1)
    assert payload.instance_logits is None
    assert payload.instance_patch_ids is None
    assert payload.instance_region_ids is None
    assert payload.instance_sample_ids is None


def test_collect_inference_payload_aggregates_duplicate_embedding_bag_ids_with_counts() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "embedding_bag_ids": ["region_1", "region_2"],
            "graph_embeddings": torch.tensor([[1.0, 1.0], [10.0, 10.0]]),
            "embedding_bag_counts": [2, 1],
            "node_embeddings": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "node_bag_ids": ["region_1", "region_2"],
        },
        {
            "bag_ids": ["region_2"],
            "bag_logits": torch.tensor([[2.0]]),
            "row_region_ids": ["region_2"],
            "row_sample_ids": ["sample_b"],
            "embedding_bag_ids": ["region_1"],
            "graph_embeddings": torch.tensor([[3.0, 3.0]]),
            "embedding_bag_counts": [1],
            "node_embeddings": torch.tensor([[0.5, 0.6]]),
            "node_bag_ids": ["region_1"],
        },
    ]
    trainer = _FakeTrainer(outputs)
    collected = collect_inference_payload(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
        include_embeddings=True,
        include_node_embeddings=True,
    )
    payload = collected.embedding_payload
    assert payload is not None
    assert payload.bag_ids == ["region_1", "region_2"]
    # region_1 weighted mean: (1.0*2 + 3.0*1) / 3
    assert torch.allclose(payload.graph_embeddings[0], torch.tensor([5.0 / 3.0, 5.0 / 3.0]))
    assert torch.allclose(payload.graph_embeddings[1], torch.tensor([10.0, 10.0]))
    assert payload.node_embeddings is not None
    assert payload.node_embeddings.shape == (3, 2)
    assert payload.node_bag_ids == ["region_1", "region_2", "region_1"]


def test_collect_inference_payload_raises_when_embedding_fields_missing() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "embedding_bag_ids": ["region_1"],
            # Missing graph_embeddings and embedding_bag_counts.
        }
    ]
    trainer = _FakeTrainer(outputs)
    with pytest.raises(ValueError, match="missing required embedding fields"):
        collect_inference_payload(
            trainer=trainer,  # type: ignore[arg-type]
            model=None,  # type: ignore[arg-type]
            datamodule=None,  # type: ignore[arg-type]
            ckpt_path=None,
            include_embeddings=True,
        )


def test_collect_inference_payload_raises_when_node_ids_missing() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "embedding_bag_ids": ["region_1"],
            "graph_embeddings": torch.tensor([[1.0, 1.0]]),
            "embedding_bag_counts": [1],
            "node_embeddings": torch.tensor([[0.1, 0.2]]),
        }
    ]
    trainer = _FakeTrainer(outputs)
    with pytest.raises(ValueError, match="must include both node_embeddings and node_bag_ids"):
        collect_inference_payload(
            trainer=trainer,  # type: ignore[arg-type]
            model=None,  # type: ignore[arg-type]
            datamodule=None,  # type: ignore[arg-type]
            ckpt_path=None,
            include_embeddings=True,
            include_node_embeddings=True,
        )


def test_collect_inference_payload_collects_instance_export_fields() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "instance_logits": torch.tensor([[1.5], [2.5]]),
            "instance_attention_logits": torch.tensor([[0.1], [0.2]]),
            "instance_patch_ids": ["patch_0", "patch_1"],
            "instance_bag_ids": ["sample_a::region_1", "sample_a::region_1"],
            "instance_region_ids": ["region_1", "region_1"],
            "instance_sample_ids": ["sample_a", "sample_a"],
            "instance_embeddings": torch.tensor([[0.11, 0.12], [0.21, 0.22]]),
            "instance_composition": torch.tensor([[1.0, 0.0], [0.2, 0.8]]),
            "instance_centroids": torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
        }
    ]
    trainer = _FakeTrainer(outputs)
    collected = collect_inference_payload(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
        include_instance_embeddings=True,
    )
    payload = collected.prediction_payload
    assert payload.instance_bag_ids == ["sample_a::region_1", "sample_a::region_1"]
    assert payload.instance_embeddings is not None
    assert payload.instance_embeddings.shape == (2, 2)
    assert payload.instance_composition is not None
    assert payload.instance_composition.shape == (2, 2)
    assert payload.instance_centroids is not None
    assert payload.instance_centroids.shape == (2, 2)


def test_collect_inference_payload_requires_instance_embeddings_when_enabled() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "instance_logits": torch.tensor([[1.5]]),
            "instance_patch_ids": ["patch_0"],
            "instance_bag_ids": ["sample_a::region_1"],
            "instance_region_ids": ["region_1"],
            "instance_sample_ids": ["sample_a"],
        }
    ]
    trainer = _FakeTrainer(outputs)
    with pytest.raises(KeyError, match="instance_embeddings"):
        collect_inference_payload(
            trainer=trainer,  # type: ignore[arg-type]
            model=None,  # type: ignore[arg-type]
            datamodule=None,  # type: ignore[arg-type]
            ckpt_path=None,
            include_instance_embeddings=True,
        )
