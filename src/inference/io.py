from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

from src.inference.schemas import BatchPredictionPayload, EmbeddingPayload


def ensure_output_dir(base_output_dir: Path, output_subdir: str) -> Path:
    output_dir = base_output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _tensor_columns(prefix: str, values: torch.Tensor) -> Dict[str, Any]:
    values = values.detach().cpu()
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    data: Dict[str, Any] = {}
    for col in range(values.shape[1]):
        data[f"{prefix}_{col}"] = values[:, col].tolist()
    return data


def predictions_to_dataframe(
    payload: BatchPredictionPayload, *, include_targets: bool, include_attention: bool
) -> pd.DataFrame:
    data: Dict[str, Any] = {"bag_id": payload.bag_ids}
    data.update(_tensor_columns("logit", payload.bag_logits))

    if include_targets and payload.bag_targets is not None:
        data.update(_tensor_columns("target", payload.bag_targets))

    if include_attention and payload.bag_attention is not None:
        data["attention"] = [
            json.dumps(
                payload.bag_attention.get(bag_id, torch.empty(0)).detach().cpu().tolist()
            )
            for bag_id in payload.bag_ids
        ]

    frame = pd.DataFrame(data)
    return frame.sort_values(by=["bag_id"]).reset_index(drop=True)


def embeddings_to_dataframe(payload: EmbeddingPayload) -> pd.DataFrame:
    data: Dict[str, Any] = {"bag_id": payload.bag_ids}
    data.update(_tensor_columns("graph_emb", payload.graph_embeddings))
    frame = pd.DataFrame(data)
    return frame.sort_values(by=["bag_id"]).reset_index(drop=True)


def node_embeddings_to_dataframe(payload: EmbeddingPayload) -> pd.DataFrame:
    if payload.node_embeddings is None or payload.node_bag_ids is None:
        return pd.DataFrame(columns=["bag_id"])
    data: Dict[str, Any] = {"bag_id": payload.node_bag_ids}
    data.update(_tensor_columns("node_emb", payload.node_embeddings))
    frame = pd.DataFrame(data)
    return frame.sort_values(by=["bag_id"]).reset_index(drop=True)


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_summary_payload(
    *,
    prediction_path: Optional[Path],
    metrics_path: Optional[Path],
    embeddings_path: Optional[Path],
    row_count: int,
) -> Dict[str, Any]:
    return {
        "prediction_path": str(prediction_path) if prediction_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
        "row_count": int(row_count),
    }
