from __future__ import annotations

import pytest
import torch

from src.inference.collectors import collect_embeddings, collect_predictions


class _FakeTrainer:
    def __init__(self, outputs):
        self._outputs = outputs

    def predict(self, model=None, datamodule=None, ckpt_path=None):
        return self._outputs


def test_collect_predictions_normalizes_payload() -> None:
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
    payload = collect_predictions(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
    )
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


def test_collect_predictions_preserves_attention_for_duplicate_bags() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "bag_attention": {"region_1": torch.tensor([0.1, 0.9])},
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "instance_logits": torch.tensor([[1.0]]),
            "instance_attention_logits": torch.tensor([[0.1]]),
            "instance_patch_ids": ["region_1"],
            "instance_region_ids": ["region_1"],
            "instance_sample_ids": ["sample_a"],
        },
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[2.0]]),
            "bag_attention": {"region_1": torch.tensor([0.8, 0.2])},
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
            "instance_logits": torch.tensor([[2.0]]),
            "instance_attention_logits": torch.tensor([[0.2]]),
            "instance_patch_ids": ["region_1"],
            "instance_region_ids": ["region_1"],
            "instance_sample_ids": ["sample_a"],
        },
    ]
    trainer = _FakeTrainer(outputs)
    payload = collect_predictions(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
    )
    assert payload.bag_attention is not None
    assert len(payload.bag_attention) == 2
    assert torch.allclose(payload.bag_attention[0], torch.tensor([0.1, 0.9]))
    assert torch.allclose(payload.bag_attention[1], torch.tensor([0.8, 0.2]))
    assert payload.instance_logits is not None
    assert payload.instance_logits.shape == (2, 1)


def test_collect_predictions_accepts_instance_logits_without_attention_logits() -> None:
    outputs = [
        {
            "bag_ids": ["region_1", "region_2"],
            "bag_logits": torch.tensor([[1.0], [4.0]]),
            "row_region_ids": ["region_1", "region_2"],
            "row_sample_ids": ["sample_a", "sample_b"],
            "instance_logits": torch.tensor([[1.0], [2.0], [4.0]]),
            "instance_patch_ids": ["patch_1", "patch_1", "patch_2"],
            "instance_region_ids": ["region_1", "region_1", "region_2"],
            "instance_sample_ids": ["sample_a", "sample_a", "sample_b"],
        }
    ]
    trainer = _FakeTrainer(outputs)
    payload = collect_predictions(
        trainer=trainer,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        datamodule=None,  # type: ignore[arg-type]
        ckpt_path=None,
    )
    assert payload.instance_logits is not None
    assert payload.instance_logits.shape == (3, 1)
    assert payload.instance_attention_logits is None
    assert payload.instance_patch_ids == ["patch_1", "patch_1", "patch_2"]
    assert payload.instance_region_ids == ["region_1", "region_1", "region_2"]
    assert payload.instance_sample_ids == ["sample_a", "sample_a", "sample_b"]


def test_collect_predictions_raises_for_inconsistent_target_presence() -> None:
    outputs = [
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[1.0]]),
            "bag_targets": torch.tensor([[1.0]]),
            "row_region_ids": ["region_1"],
            "row_sample_ids": ["sample_a"],
        },
        {
            "bag_ids": ["region_2"],
            "bag_logits": torch.tensor([[2.0]]),
            "row_region_ids": ["region_2"],
            "row_sample_ids": ["sample_b"],
        },
    ]
    trainer = _FakeTrainer(outputs)
    with pytest.raises(ValueError, match="either all chunks must include bag_targets"):
        collect_predictions(
            trainer=trainer,  # type: ignore[arg-type]
            model=None,  # type: ignore[arg-type]
            datamodule=None,  # type: ignore[arg-type]
            ckpt_path=None,
        )


def test_collect_embeddings_aggregates_duplicate_bag_ids_with_counts() -> None:
    class _Batch:
        def __init__(self, idx: int):
            self.idx = idx

        def to(self, device):
            return self

    class _FakeDataModule:
        def setup(self, stage=None):
            return None

        def predict_dataloader(self):
            return [_Batch(0), _Batch(1)]

    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def eval(self):
            return self

        def collect_graph_embeddings(self, batch, return_node_embeddings=False):
            if batch.idx == 0:
                payload = {
                    "bag_ids": ["region_1", "region_2"],
                    "graph_embeddings": torch.tensor([[1.0, 1.0], [10.0, 10.0]]),
                    "bag_counts": [2, 1],
                }
                if return_node_embeddings:
                    payload["node_embeddings"] = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
                    payload["node_bag_ids"] = ["region_1", "region_2"]
                return payload
            payload = {
                "bag_ids": ["region_1"],
                "graph_embeddings": torch.tensor([[3.0, 3.0]]),
                "bag_counts": [1],
            }
            if return_node_embeddings:
                payload["node_embeddings"] = torch.tensor([[0.5, 0.6]])
                payload["node_bag_ids"] = ["region_1"]
            return payload

    payload = collect_embeddings(
        model=_FakeModel(),  # type: ignore[arg-type]
        datamodule=_FakeDataModule(),  # type: ignore[arg-type]
        include_node_embeddings=True,
    )
    assert payload.bag_ids == ["region_1", "region_2"]
    # region_1 weighted mean: (1.0*2 + 3.0*1) / 3
    assert torch.allclose(payload.graph_embeddings[0], torch.tensor([5.0 / 3.0, 5.0 / 3.0]))
    assert torch.allclose(payload.graph_embeddings[1], torch.tensor([10.0, 10.0]))
    assert payload.node_embeddings is not None
    assert payload.node_embeddings.shape == (3, 2)
    assert payload.node_bag_ids == ["region_1", "region_2", "region_1"]


def test_collect_embeddings_raises_when_node_ids_missing() -> None:
    class _Batch:
        def to(self, device):
            return self

    class _FakeDataModule:
        def setup(self, stage=None):
            return None

        def predict_dataloader(self):
            return [_Batch()]

    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def eval(self):
            return self

        def collect_graph_embeddings(self, batch, return_node_embeddings=False):
            return {
                "bag_ids": ["region_1"],
                "graph_embeddings": torch.tensor([[1.0, 1.0]]),
                "bag_counts": [1],
                "node_embeddings": torch.tensor([[0.1, 0.2]]),
            }

    with pytest.raises(ValueError, match="node_embeddings without node_bag_ids"):
        collect_embeddings(
            model=_FakeModel(),  # type: ignore[arg-type]
            datamodule=_FakeDataModule(),  # type: ignore[arg-type]
            include_node_embeddings=True,
        )
