"""Lightweight run-artefact callbacks.

``EpochMetricsWriter`` appends one JSON line per epoch with every logged
``val/*`` and ``train/*`` metric to ``metrics_per_epoch.jsonl`` in the run's
output directory. The nested cross-validation driver reads it to pick the best
epoch (``argmax val/auroc``) of each inner fold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import torch

__all__ = ["EpochMetricsWriter", "write_test_predictions"]


def _to_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


class EpochMetricsWriter(L.Callback):
    def __init__(self, output_dir: str, filename: str = "metrics_per_epoch.jsonl") -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename = filename
        self._path: Optional[Path] = None

    @property
    def path(self) -> Path:
        if self._path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.output_dir / self.filename
        return self._path

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.is_global_zero and self.path.exists():
            self.path.unlink()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        row: Dict[str, Any] = {
            "epoch": int(trainer.current_epoch),
            "step": int(trainer.global_step),
        }
        for key, value in trainer.callback_metrics.items():
            if key.startswith(("val/", "train/")) and not key.endswith("_step"):
                v = _to_float(value)
                if v is not None:
                    row[key] = v
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")


def write_test_predictions(pl_module: L.LightningModule, output_dir: str) -> Optional[Path]:
    """Write region-level test predictions (``test_predictions.csv``) if present."""
    preds = getattr(pl_module, "test_predictions", None)
    if not preds:
        return None
    import pandas as pd

    logits = preds["logits"]
    targets = preds["targets"]
    rows = []
    n = int(logits.shape[0])
    region_ids = preds.get("region_ids") or [None] * n
    sample_ids = preds.get("sample_ids") or [None] * n
    for i in range(n):
        row: Dict[str, Any] = {
            "bag_id": preds["bag_ids"][i],
            "region_id": region_ids[i] if i < len(region_ids) else None,
            "sample_id": sample_ids[i] if i < len(sample_ids) else None,
        }
        lg = logits[i].view(-1)
        for c in range(int(lg.numel())):
            row[f"logit_{c}"] = float(lg[c])
        if preds.get("target_type") == "categorical":
            probs = torch.softmax(lg.float(), dim=0)
            for c in range(int(lg.numel())):
                row[f"prob_class_{c}"] = float(probs[c])
            row["label"] = int(targets[i].view(-1)[0])
        elif preds.get("target_type") == "binary":
            row["prob_class_1"] = float(torch.sigmoid(lg.float())[0])
            row["label"] = int(targets[i].view(-1)[0])
        else:
            tv = targets[i].view(-1)
            for c in range(int(tv.numel())):
                row[f"target_{c}"] = float(tv[c])
        rows.append(row)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
