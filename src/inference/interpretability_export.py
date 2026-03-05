from __future__ import annotations

import warnings
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
        # Notebook parity: interpretability score is sigmoid-transformed first logit.
        score_column: torch.sigmoid(logits[:, 0]).tolist(),
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
    payload: BatchPredictionPayload,
    *,
    id_column: str = "instance_id",
    undirected: bool = True,
) -> pd.DataFrame:
    if id_column not in instance_table.columns:
        raise ValueError(f"Cannot build spatial table, missing required column '{id_column}'.")

    instance_graphs = payload.instance_graphs
    if instance_graphs is None:
        raise ValueError(
            "Cannot build spatial table: predict payload is missing "
            "instance_graphs. Spatial edges must come from original subgraph connectivity."
        )
    if len(instance_graphs) != len(instance_table):
        raise ValueError(
            "Mismatch between instance table rows and payload.instance_graphs: "
            f"{len(instance_table)} vs {len(instance_graphs)}."
        )

    instance_ids = instance_table[id_column].astype(str).tolist()
    bag_ids = (
        [str(v) for v in instance_table["bag_id"].tolist()]
        if "bag_id" in instance_table
        else ["" for _ in instance_ids]
    )
    roots: List[int] = []
    for graph in instance_graphs:
        roots.append(_resolve_instance_root_node_id(graph))

    root_to_instance: Dict[tuple[str, int], int] = {}
    for idx, (bag_id, root_id) in enumerate(zip(bag_ids, roots)):
        root_to_instance[(bag_id, int(root_id))] = int(idx)

    edge_rows: List[tuple[str, str, float]] = []
    for src_idx, graph in enumerate(instance_graphs):
        src_bag = bag_ids[src_idx]
        src_root = roots[src_idx]
        edge_index = getattr(graph, "edge_index", None)
        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            continue
        n_id = getattr(graph, "n_id", None)
        if not isinstance(n_id, torch.Tensor):
            continue
        n_id = n_id.detach().cpu().long().view(-1)
        if n_id.numel() == 0:
            continue
        src = edge_index[0].detach().cpu().long().view(-1)
        dst = edge_index[1].detach().cpu().long().view(-1)
        if src.numel() != dst.numel():
            continue

        distance_values = _resolve_graph_edge_distances(graph, int(src.numel()))
        for edge_pos, (local_src, local_dst) in enumerate(zip(src.tolist(), dst.tolist())):
            if local_src < 0 or local_src >= int(n_id.numel()):
                continue
            if int(n_id[local_src].item()) != int(src_root):
                continue
            if local_dst < 0 or local_dst >= int(n_id.numel()):
                continue
            dst_root_global = int(n_id[local_dst].item())
            dst_idx = root_to_instance.get((src_bag, dst_root_global))
            if dst_idx is None:
                continue
            distance = (
                float(distance_values[edge_pos].item())
                if distance_values is not None and edge_pos < int(distance_values.numel())
                else float("nan")
            )
            edge_rows.append((instance_ids[src_idx], instance_ids[dst_idx], distance))

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


def _resolve_instance_root_node_id(graph: object) -> int:
    root_n_id = getattr(graph, "root_n_id", None)
    if not isinstance(root_n_id, torch.Tensor) or root_n_id.numel() == 0:
        raise ValueError(
            "Each instance graph must include non-empty root_n_id tensor for spatial export."
        )
    root_local = int(root_n_id.detach().cpu().view(-1)[0].item())
    root_is_global = getattr(graph, "root_n_id_is_global", None)
    if isinstance(root_is_global, torch.Tensor) and root_is_global.numel() > 0:
        if bool(root_is_global.detach().cpu().view(-1)[0].item()):
            return root_local
    elif isinstance(root_is_global, bool) and root_is_global:
        return root_local
    n_id = getattr(graph, "n_id", None)
    if isinstance(n_id, torch.Tensor):
        n_id = n_id.detach().cpu().long().view(-1)
        if 0 <= root_local < int(n_id.numel()):
            return int(n_id[root_local].item())
    return root_local


def _resolve_graph_edge_distances(graph: object, n_edges: int) -> Optional[torch.Tensor]:
    edge_attr = getattr(graph, "edge_attr", None)
    if not isinstance(edge_attr, torch.Tensor):
        return None
    edge_attr = edge_attr.detach().cpu()
    if int(edge_attr.shape[0]) != int(n_edges):
        return None
    if edge_attr.ndim == 1:
        return edge_attr.float().view(-1)
    if edge_attr.ndim != 2 or int(edge_attr.shape[1]) == 0:
        return None
    dist_col = 1
    names = getattr(graph, "edge_attr_names", None)
    if isinstance(names, (list, tuple)):
        for idx, name in enumerate(names):
            if str(name) == "distance":
                dist_col = int(idx)
                break
        warnings.warn("No distance column found in edge_attr, using column 1.")
    if dist_col >= int(edge_attr.shape[1]):
        return None
    return edge_attr[:, dist_col].float().view(-1)
