from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.inference.schemas import BatchPredictionPayload


def _tensor_to_2d_cpu(values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if values is None:
        return None
    values = values.detach().cpu()
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    return values


def _normalize_optional(
    values: Optional[Sequence[Optional[str]]], size: int
) -> List[Optional[str]]:
    if values is None:
        return [None] * size
    out = [None if v is None else str(v) for v in values]
    if len(out) != size:
        raise ValueError(f"Expected {size} values, got {len(out)}.")
    return out


def _infer_bag_ids(payload: BatchPredictionPayload, size: int) -> List[str]:
    if payload.instance_bag_ids is not None:
        bag_ids = [str(v) for v in payload.instance_bag_ids]
        if len(bag_ids) != size:
            raise ValueError(
                "Mismatch between instance logits rows and instance_bag_ids: "
                f"{size} vs {len(bag_ids)}."
            )
        return bag_ids

    region_ids = _normalize_optional(payload.instance_region_ids, size)
    sample_ids = _normalize_optional(payload.instance_sample_ids, size)
    patch_ids = [str(v) for v in (payload.instance_patch_ids or [])]
    bag_ids: List[str] = []
    for idx in range(size):
        region = region_ids[idx]
        sample = sample_ids[idx]
        if sample and region:
            bag_ids.append(f"{sample}::{region}")
        elif region:
            bag_ids.append(region)
        elif sample:
            bag_ids.append(sample)
        elif idx < len(patch_ids):
            bag_ids.append(patch_ids[idx])
        else:
            bag_ids.append(f"bag_{idx}")
    return bag_ids


def _make_unique_ids(values: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = defaultdict(int)
    out: List[str] = []
    for value in values:
        base = str(value) if str(value).strip() else "instance"
        counts[base] += 1
        if counts[base] == 1:
            out.append(base)
        else:
            out.append(f"{base}::{counts[base] - 1}")
    return out


def _softmax_by_group(logits: torch.Tensor, groups: Sequence[str]) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("instance_attention_logits must be a 2D tensor with >=1 column.")
    if logits.shape[0] != len(groups):
        raise ValueError(
            "Mismatch between attention logits rows and group ids: "
            f"{logits.shape[0]} vs {len(groups)}."
        )
    attention = np.zeros((len(groups),), dtype=np.float64)
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped_indices[str(group)].append(idx)
    for indices in grouped_indices.values():
        chunk = logits[indices, 0]
        scores = torch.softmax(chunk, dim=0).cpu().numpy().astype(np.float64)
        for local, global_idx in enumerate(indices):
            attention[global_idx] = float(scores[local])
    return attention


def _uniform_attention(groups: Sequence[str]) -> np.ndarray:
    attention = np.zeros((len(groups),), dtype=np.float64)
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped_indices[str(group)].append(idx)
    for indices in grouped_indices.values():
        if not indices:
            continue
        value = 1.0 / float(len(indices))
        for global_idx in indices:
            attention[global_idx] = value
    return attention


def resolve_composition_column_names(
    datamodule: Any,
    *,
    composition_label: str,
    composition_prefix: str,
    width: int,
) -> List[str]:
    label_map = None
    for attr in ("dataset_train", "dataset_val", "dataset_test"):
        dataset = getattr(datamodule, attr, None)
        maps = getattr(dataset, "label_maps", None) if dataset is not None else None
        if (
            isinstance(maps, dict)
            and composition_label in maps
            and isinstance(maps[composition_label], dict)
        ):
            label_map = maps[composition_label]
            break
    if not isinstance(label_map, dict):
        return [f"{composition_prefix}{idx}" for idx in range(width)]

    inv = {int(v): str(k) for k, v in label_map.items()}
    names: List[str] = []
    for idx in range(width):
        label = inv.get(idx, str(idx))
        names.append(f"{composition_prefix}{label}")
    return names


def build_instance_table(
    payload: BatchPredictionPayload,
    *,
    composition_prefix: str = "comp_",
    embedding_prefix: str = "inst_emb",
    composition_column_names: Optional[Sequence[str]] = None,
    require_composition: bool = True,
    id_column: str = "instance_id",
    bag_id_column: str = "bag_id",
    score_column: str = "score",
    attention_column: str = "attention",
) -> pd.DataFrame:
    logits = _tensor_to_2d_cpu(payload.instance_logits)
    if logits is None:
        raise ValueError("Interpretability export requires instance_logits in predict payload.")

    embeddings = _tensor_to_2d_cpu(payload.instance_embeddings)
    if embeddings is None:
        raise ValueError(
            "Interpretability export requires instance_embeddings in predict payload."
        )
    if embeddings.shape[0] != logits.shape[0]:
        raise ValueError(
            "Mismatch between instance_logits and instance_embeddings rows: "
            f"{logits.shape[0]} vs {embeddings.shape[0]}."
        )

    size = int(logits.shape[0])
    patch_ids = (
        [str(v) for v in payload.instance_patch_ids]
        if payload.instance_patch_ids is not None
        else [f"instance_{idx}" for idx in range(size)]
    )
    if len(patch_ids) != size:
        raise ValueError(
            "Mismatch between instance logits rows and instance_patch_ids: "
            f"{size} vs {len(patch_ids)}."
        )
    bag_ids = _infer_bag_ids(payload, size)
    region_ids = _normalize_optional(payload.instance_region_ids, size)
    sample_ids = _normalize_optional(payload.instance_sample_ids, size)
    instance_ids = _make_unique_ids(patch_ids)

    data: Dict[str, List[object]] = {
        id_column: instance_ids,
        bag_id_column: bag_ids,
        "patch_id": patch_ids,
        "region_id": region_ids,
        "sample_id": sample_ids,
        score_column: logits[:, 0].tolist(),
    }
    for col in range(int(logits.shape[1])):
        data[f"logit_{col}"] = logits[:, col].tolist()

    attention_logits = _tensor_to_2d_cpu(payload.instance_attention_logits)
    if attention_logits is not None:
        data[attention_column] = _softmax_by_group(attention_logits, bag_ids).tolist()
    else:
        data[attention_column] = _uniform_attention(bag_ids).tolist()

    for col in range(int(embeddings.shape[1])):
        data[f"{embedding_prefix}_{col}"] = embeddings[:, col].tolist()

    composition = _tensor_to_2d_cpu(payload.instance_composition)
    if composition is not None:
        if composition.shape[0] != size:
            raise ValueError(
                "Mismatch between instance_logits and instance_composition rows: "
                f"{size} vs {composition.shape[0]}."
            )
        if composition_column_names is not None and len(composition_column_names) == int(
            composition.shape[1]
        ):
            comp_cols = list(composition_column_names)
        else:
            comp_cols = [f"{composition_prefix}{idx}" for idx in range(int(composition.shape[1]))]
        for col_name, values in zip(comp_cols, composition.T):
            data[col_name] = values.tolist()
    elif require_composition:
        raise ValueError(
            "Composition columns are required for interpretability export but "
            "instance_composition was not provided by predict payload."
        )

    centroids = _tensor_to_2d_cpu(payload.instance_centroids)
    if centroids is not None:
        if centroids.shape[0] != size:
            raise ValueError(
                "Mismatch between instance_logits and instance_centroids rows: "
                f"{size} vs {centroids.shape[0]}."
            )
        if centroids.shape[1] >= 1:
            data["center_x"] = centroids[:, 0].tolist()
        if centroids.shape[1] >= 2:
            data["center_y"] = centroids[:, 1].tolist()

    return pd.DataFrame(data)


def build_spatial_table(
    instance_table: pd.DataFrame,
    *,
    id_column: str = "instance_id",
    bag_id_column: str = "bag_id",
    x_column: str = "center_x",
    y_column: str = "center_y",
    n_neighbors: int = 8,
    undirected: bool = True,
) -> pd.DataFrame:
    required = {id_column, bag_id_column, x_column, y_column}
    missing = required - set(instance_table.columns)
    if missing:
        raise ValueError(
            f"Cannot build spatial table, missing required columns: {sorted(missing)}."
        )
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be >= 1.")

    edge_rows: List[tuple[str, str, float]] = []
    for _, group in instance_table.groupby(bag_id_column, dropna=False):
        local = group[[id_column, x_column, y_column]].copy()
        local[id_column] = local[id_column].astype(str)
        local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column])
        if len(local) < 2:
            continue
        ids = local[id_column].tolist()
        coords = local[[x_column, y_column]].to_numpy(dtype=np.float64)
        k = min(int(n_neighbors), len(ids) - 1)
        dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        for idx in range(len(ids)):
            nbr_idx = np.argpartition(dist[idx], kth=k - 1)[:k]
            nbr_idx = nbr_idx[np.argsort(dist[idx, nbr_idx])]
            for nbr in nbr_idx.tolist():
                edge_rows.append((ids[idx], ids[nbr], float(dist[idx, nbr])))

    if undirected:
        symmetric_rows = [(dst, src, d) for src, dst, d in edge_rows]
        edge_rows = edge_rows + symmetric_rows

    if not edge_rows:
        return pd.DataFrame(columns=["source_id", "target_id", "distance"])

    edge_df = pd.DataFrame(edge_rows, columns=["source_id", "target_id", "distance"])
    edge_df = (
        edge_df.sort_values(["source_id", "target_id", "distance"])
        .drop_duplicates(subset=["source_id", "target_id"], keep="first")
        .reset_index(drop=True)
    )
    return edge_df
