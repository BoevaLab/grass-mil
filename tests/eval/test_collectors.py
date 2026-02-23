from __future__ import annotations

import torch

from src.inference.collectors import collect_predictions


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
        },
        {
            "bag_ids": ["region_1"],
            "bag_logits": torch.tensor([[2.0]]),
            "bag_attention": {"region_1": torch.tensor([0.2, 0.8])},
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
