from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from lightning import LightningDataModule, LightningModule, Trainer

from src.inference.schemas import BatchPredictionPayload, EmbeddingPayload


def _as_str_list(values: Iterable[Any]) -> List[str]:
    return [str(v) for v in values]


def _ensure_2d(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(-1)
    return values


def _stack_optional_tensors(values: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not values:
        return None
    return torch.cat([_ensure_2d(v.detach().cpu()) for v in values], dim=0)


def _align_chunk_attention_to_rows(
    chunk_bag_ids: List[str], chunk_attention: Optional[Dict[str, torch.Tensor]]
) -> List[Optional[torch.Tensor]]:
    if not chunk_attention:
        return [None] * len(chunk_bag_ids)
    return [
        chunk_attention[bag_id].detach().cpu() if bag_id in chunk_attention else None
        for bag_id in chunk_bag_ids
    ]


def collect_predictions(
    *,
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    ckpt_path: Optional[str],
) -> BatchPredictionPayload:
    """Collect and normalize `predict_step` payloads across all batches."""
    outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

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

    for chunk in outputs:
        if not isinstance(chunk, dict):
            raise TypeError("Expected `predict_step` output to be a dict.")
        if "bag_ids" not in chunk or "bag_logits" not in chunk:
            raise KeyError("`predict_step` must include bag_ids and bag_logits.")

        chunk_bag_ids = _as_str_list(chunk["bag_ids"])
        chunk_logits = _ensure_2d(chunk["bag_logits"])
        if len(chunk_bag_ids) != chunk_logits.shape[0]:
            raise ValueError(
                "Mismatch within predict chunk between bag_ids and bag_logits rows: "
                f"{len(chunk_bag_ids)} ids vs {chunk_logits.shape[0]} logits."
            )

        bag_ids.extend(chunk_bag_ids)
        logits_chunks.append(chunk_logits)

        if "bag_targets" in chunk and chunk["bag_targets"] is not None:
            target_chunks.append(chunk["bag_targets"])
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
        if "instance_logits" in chunk and chunk["instance_logits"] is not None:
            for field in ("instance_patch_ids", "instance_region_ids", "instance_sample_ids"):
                if field not in chunk:
                    raise KeyError(
                        f"predict_step provided instance_logits without required field '{field}'."
                    )
            chunk_instance_logits = _ensure_2d(chunk["instance_logits"])
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
                chunk_instance_attn = _ensure_2d(chunk_instance_attn)
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

    bag_logits = _stack_optional_tensors(logits_chunks)
    if bag_logits is None:
        raise ValueError("No logits were collected from predict outputs.")
    bag_targets = _stack_optional_tensors(target_chunks)
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

    return BatchPredictionPayload(
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


def collect_embeddings(
    *,
    model: LightningModule,
    datamodule: LightningDataModule,
    include_node_embeddings: bool = False,
) -> EmbeddingPayload:
    """Collect graph-level embeddings from a model helper."""
    if not hasattr(model, "collect_graph_embeddings"):
        raise AttributeError(
            "Model does not implement collect_graph_embeddings(batch). "
            "Expected SupervisedModule-compatible implementation."
        )

    datamodule.setup(stage="predict")
    loader = None
    if hasattr(datamodule, "predict_dataloader"):
        loader = datamodule.predict_dataloader()
    if loader is None:
        if hasattr(datamodule, "test_dataloader"):
            loader = datamodule.test_dataloader()
        else:
            raise ValueError(
                "Datamodule has neither predict_dataloader nor test_dataloader."
            )

    bag_ids: List[str] = []
    graph_embs: List[torch.Tensor] = []
    node_embs: List[torch.Tensor] = []
    node_bag_ids: List[str] = []

    model.eval()
    device = model.device
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, "to"):
                batch = batch.to(device)
            payload = model.collect_graph_embeddings(
                batch, return_node_embeddings=include_node_embeddings
            )
            bag_ids.extend(_as_str_list(payload["bag_ids"]))
            graph_embs.append(payload["graph_embeddings"].detach().cpu())
            node_values = payload.get("node_embeddings")
            if node_values is not None:
                node_embs.append(node_values.detach().cpu())
                node_bag_ids.extend(_as_str_list(payload.get("node_bag_ids", [])))

    if not graph_embs:
        raise ValueError("No embeddings were collected from dataloader.")

    return EmbeddingPayload(
        bag_ids=bag_ids,
        graph_embeddings=torch.cat(graph_embs, dim=0),
        node_embeddings=torch.cat(node_embs, dim=0) if node_embs else None,
        node_bag_ids=node_bag_ids if node_bag_ids else None,
    )
