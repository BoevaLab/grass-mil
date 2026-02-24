from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.inference.aggregation import aggregate_group_logits  # noqa: E402
from src.inference.collectors import (  # noqa: E402
    collect_embeddings,
    collect_predictions,
)
from src.inference.io import (  # noqa: E402
    build_summary_payload,
    embeddings_to_dataframe,
    ensure_output_dir,
    node_embeddings_to_dataframe,
    predictions_to_dataframe,
    write_dataframe,
    write_json,
)
from src.inference.metrics import compute_task_metrics  # noqa: E402
from src.inference.schemas import BatchPredictionPayload  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _as_batch_payload(payload: Any) -> BatchPredictionPayload:
    return BatchPredictionPayload(
        bag_ids=list(payload.bag_ids),
        bag_logits=payload.bag_logits,
        bag_targets=payload.bag_targets,
        bag_attention=getattr(payload, "bag_attention", None),
        row_region_ids=getattr(payload, "row_region_ids", None),
        row_sample_ids=getattr(payload, "row_sample_ids", None),
        instance_logits=getattr(payload, "instance_logits", None),
        instance_attention_logits=getattr(payload, "instance_attention_logits", None),
        instance_patch_ids=getattr(payload, "instance_patch_ids", None),
        instance_region_ids=getattr(payload, "instance_region_ids", None),
        instance_sample_ids=getattr(payload, "instance_sample_ids", None),
    )


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    base_output_dir = Path(str(cfg.paths.output_dir))
    output_dir = ensure_output_dir(
        base_output_dir=base_output_dir,
        output_subdir=str(cfg.predict.output_subdir),
    )

    pred_payload = collect_predictions(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.ckpt_path,
    )

    if bool(cfg.aggregation.enabled):
        aggregated = aggregate_group_logits(
            pred_payload,
            mode=str(cfg.aggregation.mode),
            bag_scope=str(cfg.aggregation.get("bag_scope", "patch")),
            subsample_fraction=float(cfg.aggregation.get("subsample_fraction", 1.0)),
            subsample_seed=cfg.aggregation.get("subsample_seed"),
        )
        pred_payload = _as_batch_payload(aggregated)

    prediction_path = None
    if bool(cfg.predict.save_predictions):
        pred_frame = predictions_to_dataframe(
            pred_payload,
            include_targets=bool(cfg.predict.include_targets),
            include_attention=bool(cfg.predict.include_attention),
        )
        prediction_path = output_dir / str(cfg.output.predictions_filename)
        write_dataframe(pred_frame, prediction_path)
    else:
        pred_frame = predictions_to_dataframe(
            pred_payload,
            include_targets=False,
            include_attention=False,
        )

    metrics_path = None
    metrics_payload: Dict[str, Any] = {}
    if bool(cfg.predict.save_metrics) and bool(cfg.metrics.enabled):
        metrics_payload = compute_task_metrics(
            payload=pred_payload,
            target_type=str(cfg.task.target_type),
            metrics_cfg=cfg.metrics,
        )
        metrics_path = output_dir / str(cfg.output.metrics_filename)
        write_json(metrics_payload, metrics_path)

    embeddings_path = None
    if bool(cfg.embeddings.enabled) and bool(cfg.embeddings.save):
        emb_payload = collect_embeddings(
            model=model,
            datamodule=datamodule,
            include_node_embeddings=bool(cfg.embeddings.extract_node),
        )
        emb_frame = embeddings_to_dataframe(emb_payload)
        embeddings_path = output_dir / str(cfg.embeddings.filename)
        write_dataframe(emb_frame, embeddings_path)

        if bool(cfg.embeddings.extract_node):
            node_frame = node_embeddings_to_dataframe(emb_payload)
            node_path = output_dir / str(cfg.embeddings.node_filename)
            write_dataframe(node_frame, node_path)

    summary_path = output_dir / str(cfg.output.summary_filename)
    summary_payload = build_summary_payload(
        prediction_path=prediction_path,
        metrics_path=metrics_path,
        embeddings_path=embeddings_path,
        row_count=len(pred_frame),
    )
    summary_payload["metrics"] = metrics_payload
    write_json(summary_payload, summary_path)
    return summary_payload, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="inference/predict.yaml",
)
def main(cfg: DictConfig) -> None:
    extras(cfg)
    predict(cfg)


if __name__ == "__main__":
    main()
