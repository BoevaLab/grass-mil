from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional

import torch
from lightning import LightningDataModule, LightningModule, Trainer

from src.inference.schemas import (
    BatchPredictionPayload,
    CollectedInferencePayload,
    EmbeddingPayload,
)


def _as_str_list(values: Iterable[Any]) -> List[str]:
    return [str(v) for v in values]


def _ensure_2d(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(-1)
    return values


def _stack_optional_tensors(values: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not values:
        return None
    return torch.cat([_ensure_2d(v) for v in values], dim=0)


def _as_positive_count_tensor(values: Any, *, expected_length: int) -> torch.Tensor:
    if values is None:
        raise ValueError(
            "Missing required 'embedding_bag_counts' while collecting embeddings."
        )
    if isinstance(values, torch.Tensor):
        counts = values.detach().cpu().reshape(-1).float()
    else:
        counts = torch.tensor(list(values), dtype=torch.float32).reshape(-1)
    if counts.numel() != expected_length:
        raise ValueError(
            "Mismatch between embedding_bag_ids and embedding_bag_counts rows while "
            "collecting embeddings: "
            f"{expected_length} ids vs {counts.numel()} counts."
        )
    if not torch.all(counts > 0):
        raise ValueError("embedding_bag_counts must be strictly positive for aggregation.")
    return counts


def _align_chunk_attention_to_rows(
    chunk_bag_ids: List[str], chunk_attention: Optional[Dict[str, torch.Tensor]]
) -> List[Optional[torch.Tensor]]:
    if not chunk_attention:
        return [None] * len(chunk_bag_ids)
    return [
        chunk_attention[bag_id].detach().cpu() if bag_id in chunk_attention else None
        for bag_id in chunk_bag_ids
    ]


def collect_inference_payload(
    *,
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    ckpt_path: Optional[str],
    include_instance_payload: bool = True,
    include_embeddings: bool = False,
    include_node_embeddings: bool = False,
) -> CollectedInferencePayload:
    """Collect predictions and optional embeddings from a single predict pass."""
    outputs = deque(
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    )

    bag_ids: List[str] = []
    logits_chunks: List[torch.Tensor] = []
    target_chunks: List[torch.Tensor] = []
    attention_rows: List[Optional[torch.Tensor]] = []
    row_region_ids: List[Optional[str]] = []
    row_sample_ids: List[Optional[str]] = []
    instance_logits_chunks: List[torch.Tensor] = []
    instance_attention_logits_chunks: List[torch.Tensor] = []
    instance_patch_ids: List[str] = []
    instance_region_ids: List[Optional[str]] = []
    instance_sample_ids: List[Optional[str]] = []
    expect_targets: Optional[bool] = None

    bag_emb_sums: Dict[str, torch.Tensor] = {}
    bag_emb_weights: Dict[str, float] = {}
    node_embs: List[torch.Tensor] = []
    node_bag_ids: List[str] = []

    while outputs:
        chunk = outputs.popleft()
        if not isinstance(chunk, dict):
            raise TypeError("Expected `predict_step` output to be a dict.")
        if "bag_ids" not in chunk or "bag_logits" not in chunk:
            raise KeyError("`predict_step` must include bag_ids and bag_logits.")

        chunk_bag_ids = _as_str_list(chunk["bag_ids"])
        chunk_logits = _ensure_2d(chunk["bag_logits"]).detach().cpu()
        if len(chunk_bag_ids) != chunk_logits.shape[0]:
            raise ValueError(
                "Mismatch within predict chunk between bag_ids and bag_logits rows: "
                f"{len(chunk_bag_ids)} ids vs {chunk_logits.shape[0]} logits."
            )

        bag_ids.extend(chunk_bag_ids)
        logits_chunks.append(chunk_logits)

        has_targets = "bag_targets" in chunk and chunk["bag_targets"] is not None
        if expect_targets is None:
            expect_targets = has_targets
        elif has_targets != expect_targets:
            raise ValueError(
                "Inconsistent predict_step target payloads across chunks: either all "
                "chunks must include bag_targets or none."
            )
        if has_targets:
            chunk_targets = _ensure_2d(chunk["bag_targets"]).detach().cpu()
            if chunk_targets.shape[0] != len(chunk_bag_ids):
                raise ValueError(
                    "Mismatch within predict chunk between bag_ids and bag_targets rows: "
                    f"{len(chunk_bag_ids)} ids vs {chunk_targets.shape[0]} targets."
                )
            target_chunks.append(chunk_targets)

        chunk_row_region_ids = chunk.get("row_region_ids")
        if chunk_row_region_ids is None:
            row_region_ids.extend([None] * len(chunk_bag_ids))
        else:
            if len(chunk_row_region_ids) != len(chunk_bag_ids):
                raise ValueError(
                    "Mismatch between bag_ids and row_region_ids within predict chunk: "
                    f"{len(chunk_bag_ids)} ids vs {len(chunk_row_region_ids)} row ids."
                )
            row_region_ids.extend(
                [None if v is None else str(v) for v in chunk_row_region_ids]
            )

        chunk_row_sample_ids = chunk.get("row_sample_ids")
        if chunk_row_sample_ids is None:
            row_sample_ids.extend([None] * len(chunk_bag_ids))
        else:
            if len(chunk_row_sample_ids) != len(chunk_bag_ids):
                raise ValueError(
                    "Mismatch between bag_ids and row_sample_ids within predict chunk: "
                    f"{len(chunk_bag_ids)} ids vs {len(chunk_row_sample_ids)} row ids."
                )
            row_sample_ids.extend(
                [None if v is None else str(v) for v in chunk_row_sample_ids]
            )

        attention_rows.extend(
            _align_chunk_attention_to_rows(chunk_bag_ids, chunk.get("bag_attention"))
        )

        if (
            include_instance_payload
            and "instance_logits" in chunk
            and chunk["instance_logits"] is not None
        ):
            for field in ("instance_patch_ids", "instance_region_ids", "instance_sample_ids"):
                if field not in chunk:
                    raise KeyError(
                        f"predict_step provided instance_logits without required field '{field}'."
                    )
            chunk_instance_logits = _ensure_2d(chunk["instance_logits"]).detach().cpu()
            if len(chunk["instance_patch_ids"]) != chunk_instance_logits.shape[0]:
                raise ValueError(
                    "Mismatch between instance_logits and instance_patch_ids rows: "
                    f"{chunk_instance_logits.shape[0]} vs {len(chunk['instance_patch_ids'])}."
                )
            if len(chunk["instance_region_ids"]) != chunk_instance_logits.shape[0]:
                raise ValueError(
                    "Mismatch between instance_logits and instance_region_ids rows: "
                    f"{chunk_instance_logits.shape[0]} vs {len(chunk['instance_region_ids'])}."
                )
            if len(chunk["instance_sample_ids"]) != chunk_instance_logits.shape[0]:
                raise ValueError(
                    "Mismatch between instance_logits and instance_sample_ids rows: "
                    f"{chunk_instance_logits.shape[0]} vs {len(chunk['instance_sample_ids'])}."
                )
            instance_logits_chunks.append(chunk_instance_logits)
            chunk_instance_attn = chunk.get("instance_attention_logits")
            if chunk_instance_attn is not None:
                chunk_instance_attn = _ensure_2d(chunk_instance_attn).detach().cpu()
                if chunk_instance_logits.shape[0] != chunk_instance_attn.shape[0]:
                    raise ValueError(
                        "Mismatch between instance_logits and instance_attention_logits rows: "
                        f"{chunk_instance_logits.shape[0]} vs {chunk_instance_attn.shape[0]}."
                    )
                instance_attention_logits_chunks.append(chunk_instance_attn)
            instance_patch_ids.extend([str(v) for v in chunk["instance_patch_ids"]])
            instance_region_ids.extend(
                [None if v is None else str(v) for v in chunk["instance_region_ids"]]
            )
            instance_sample_ids.extend(
                [None if v is None else str(v) for v in chunk["instance_sample_ids"]]
            )

        if include_embeddings:
            missing_embed_fields = [
                field
                for field in ("embedding_bag_ids", "graph_embeddings", "embedding_bag_counts")
                if field not in chunk or chunk[field] is None
            ]
            if missing_embed_fields:
                raise ValueError(
                    "predict_step is missing required embedding fields while embeddings "
                    f"collection is enabled: {missing_embed_fields}."
                )
            chunk_embedding_bag_ids = _as_str_list(chunk["embedding_bag_ids"])
            chunk_graph_embeddings = _ensure_2d(chunk["graph_embeddings"]).detach().cpu()
            if len(chunk_embedding_bag_ids) != chunk_graph_embeddings.shape[0]:
                raise ValueError(
                    "Mismatch between embedding_bag_ids and graph_embeddings rows while "
                    "collecting embeddings: "
                    f"{len(chunk_embedding_bag_ids)} ids vs "
                    f"{chunk_graph_embeddings.shape[0]} rows."
                )
            chunk_counts = _as_positive_count_tensor(
                chunk["embedding_bag_counts"], expected_length=len(chunk_embedding_bag_ids)
            )
            for idx, emb_bag_id in enumerate(chunk_embedding_bag_ids):
                weight = float(chunk_counts[idx].item())
                row = chunk_graph_embeddings[idx]
                if emb_bag_id not in bag_emb_sums:
                    bag_emb_sums[emb_bag_id] = row * weight
                    bag_emb_weights[emb_bag_id] = weight
                else:
                    bag_emb_sums[emb_bag_id] = bag_emb_sums[emb_bag_id] + (row * weight)
                    bag_emb_weights[emb_bag_id] += weight

            chunk_node_embeddings = chunk.get("node_embeddings")
            chunk_node_bag_ids = chunk.get("node_bag_ids")
            if include_node_embeddings:
                if chunk_node_embeddings is None or chunk_node_bag_ids is None:
                    raise ValueError(
                        "predict_step must include both node_embeddings and node_bag_ids "
                        "when node embedding extraction is enabled."
                    )
            if chunk_node_embeddings is not None or chunk_node_bag_ids is not None:
                if chunk_node_embeddings is None or chunk_node_bag_ids is None:
                    raise ValueError(
                        "predict_step returned partial node embedding payload; both "
                        "node_embeddings and node_bag_ids are required together."
                    )
                node_tensor = _ensure_2d(chunk_node_embeddings).detach().cpu()
                node_id_list = _as_str_list(chunk_node_bag_ids)
                if len(node_id_list) != node_tensor.shape[0]:
                    raise ValueError(
                        "Mismatch between node_embeddings rows and node_bag_ids while "
                        "collecting embeddings: "
                        f"{node_tensor.shape[0]} rows vs {len(node_id_list)} ids."
                    )
                node_embs.append(node_tensor)
                node_bag_ids.extend(node_id_list)

    bag_logits = _stack_optional_tensors(logits_chunks)
    if bag_logits is None:
        raise ValueError("No logits were collected from predict outputs.")
    bag_targets = _stack_optional_tensors(target_chunks) if expect_targets else None
    bag_attention = (
        attention_rows if any(attn is not None for attn in attention_rows) else None
    )
    instance_logits = _stack_optional_tensors(instance_logits_chunks)
    instance_attention_logits = _stack_optional_tensors(instance_attention_logits_chunks)

    if len(bag_ids) != bag_logits.shape[0]:
        raise ValueError(
            "Mismatch between collected bag_ids and logits rows: "
            f"{len(bag_ids)} ids vs {bag_logits.shape[0]} logits."
        )
    if bag_attention is not None and len(bag_attention) != len(bag_ids):
        raise ValueError(
            "Mismatch between collected bag_ids and attention rows: "
            f"{len(bag_ids)} ids vs {len(bag_attention)} attention rows."
        )
    if len(row_region_ids) != len(bag_ids):
        raise ValueError(
            "Mismatch between collected bag_ids and row_region_ids: "
            f"{len(bag_ids)} ids vs {len(row_region_ids)} row ids."
        )
    if len(row_sample_ids) != len(bag_ids):
        raise ValueError(
            "Mismatch between collected bag_ids and row_sample_ids: "
            f"{len(bag_ids)} ids vs {len(row_sample_ids)} row ids."
        )
    if bag_targets is not None and bag_targets.shape[0] != len(bag_ids):
        raise ValueError(
            "Mismatch between collected bag_ids and bag_targets rows: "
            f"{len(bag_ids)} ids vs {bag_targets.shape[0]} targets."
        )
    if instance_logits is not None:
        if len(instance_patch_ids) != instance_logits.shape[0]:
            raise ValueError(
                "Mismatch between collected instance_logits and instance_patch_ids: "
                f"{instance_logits.shape[0]} vs {len(instance_patch_ids)}."
            )
        if len(instance_region_ids) != instance_logits.shape[0]:
            raise ValueError(
                "Mismatch between collected instance_logits and instance_region_ids: "
                f"{instance_logits.shape[0]} vs {len(instance_region_ids)}."
            )
        if len(instance_sample_ids) != instance_logits.shape[0]:
            raise ValueError(
                "Mismatch between collected instance_logits and instance_sample_ids: "
                f"{instance_logits.shape[0]} vs {len(instance_sample_ids)}."
            )

    prediction_payload = BatchPredictionPayload(
        bag_ids=bag_ids,
        bag_logits=bag_logits,
        bag_targets=bag_targets,
        bag_attention=bag_attention,
        row_region_ids=row_region_ids,
        row_sample_ids=row_sample_ids,
        instance_logits=instance_logits,
        instance_attention_logits=instance_attention_logits,
        instance_patch_ids=instance_patch_ids if instance_logits is not None else None,
        instance_region_ids=instance_region_ids if instance_logits is not None else None,
        instance_sample_ids=instance_sample_ids if instance_logits is not None else None,
    )

    embedding_payload: Optional[EmbeddingPayload] = None
    if include_embeddings:
        if not bag_emb_sums:
            raise ValueError("No embeddings were collected from predict outputs.")
        ordered_bag_ids = sorted(bag_emb_sums.keys())
        graph_embs = torch.stack(
            [
                bag_emb_sums[bag_id] / bag_emb_weights[bag_id]
                for bag_id in ordered_bag_ids
            ],
            dim=0,
        )
        embedding_payload = EmbeddingPayload(
            bag_ids=ordered_bag_ids,
            graph_embeddings=graph_embs,
            node_embeddings=torch.cat(node_embs, dim=0) if node_embs else None,
            node_bag_ids=node_bag_ids if node_bag_ids else None,
        )

    return CollectedInferencePayload(
        prediction_payload=prediction_payload,
        embedding_payload=embedding_payload,
    )
