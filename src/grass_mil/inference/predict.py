from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.inference.aggregation import aggregate_group_logits  # noqa: E402
from grass_mil.inference.collectors import collect_inference_payload  # noqa: E402
from grass_mil.inference.interpretability_export import (  # noqa: E402
    build_instance_table,
    build_spatial_table,
    resolve_composition_column_names,
)
from grass_mil.inference.io import (  # noqa: E402
    build_summary_payload,
    embeddings_to_dataframe,
    ensure_output_dir,
    node_embeddings_to_dataframe,
    predictions_to_dataframe,
    write_dataframe,
    write_json,
)
from grass_mil.inference.metrics import compute_task_metrics  # noqa: E402
from grass_mil.inference.schemas import (  # noqa: E402
    BatchPredictionPayload,
    PreforwardSubsamplingDecision,
)
from grass_mil.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _supports_preforward_subsampling(mode: str) -> bool:
    return mode in {"mean", "max", "attention_weighted"}


def _needs_instance_payload(cfg: DictConfig) -> bool:
    if not bool(cfg.aggregation.enabled):
        return False
    mode = str(cfg.aggregation.mode)
    return mode in {"mean", "max", "attention_weighted"}


def _plan_preforward_subsampling(
    datamodule: LightningDataModule, cfg: DictConfig
) -> PreforwardSubsamplingDecision:
    aggregation_enabled = bool(cfg.aggregation.enabled)
    mode = str(cfg.aggregation.mode)
    subsample_fraction = float(cfg.aggregation.get("subsample_fraction", 1.0))
    subsample_seed_raw = cfg.aggregation.get("subsample_seed")
    subsample_seed = None if subsample_seed_raw is None else int(subsample_seed_raw)
    requested = aggregation_enabled and subsample_fraction < 1.0

    if not aggregation_enabled:
        return PreforwardSubsamplingDecision(
            requested=False,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="aggregation_disabled",
        )
    if not requested:
        return PreforwardSubsamplingDecision(
            requested=False,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="not_requested",
        )
    if not _supports_preforward_subsampling(mode):
        return PreforwardSubsamplingDecision(
            requested=True,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="unsupported_mode",
        )

    sampler_cfg = getattr(datamodule, "sampler_config", None)
    if sampler_cfg is None:
        return PreforwardSubsamplingDecision(
            requested=True,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="missing_sampler_config",
        )
    strategy_name_raw = getattr(sampler_cfg, "name", None)
    strategy_name = None if strategy_name_raw is None else str(strategy_name_raw).strip().lower()
    if not strategy_name:
        return PreforwardSubsamplingDecision(
            requested=True,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="missing_sampler_config",
        )
    if strategy_name == "identity":
        return PreforwardSubsamplingDecision(
            requested=True,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="identity_sampler",
            strategy_name=strategy_name,
            runtime_enabled=None,
        )

    runtime = getattr(sampler_cfg, "runtime", None)
    if runtime is None:
        runtime = {}
    runtime = dict(runtime)
    runtime_enabled = bool(runtime.get("enabled", False))
    if not runtime_enabled:
        return PreforwardSubsamplingDecision(
            requested=True,
            effective=False,
            fraction=subsample_fraction,
            seed=subsample_seed,
            reason="runtime_disabled",
            strategy_name=strategy_name,
            runtime_enabled=False,
        )

    runtime["subsample_fraction"] = subsample_fraction
    runtime["subsample_seed"] = subsample_seed
    sampler_cfg.runtime = runtime

    # Force a strategy rebuild in case setup() was invoked previously.
    if hasattr(datamodule, "sampler_strategy"):
        setattr(datamodule, "sampler_strategy", None)
    if hasattr(datamodule, "val_sampler_strategy"):
        setattr(datamodule, "val_sampler_strategy", None)
    return PreforwardSubsamplingDecision(
        requested=True,
        effective=True,
        fraction=subsample_fraction,
        seed=subsample_seed,
        reason="applied",
        strategy_name=strategy_name,
        runtime_enabled=True,
    )


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
        instance_bag_ids=getattr(payload, "instance_bag_ids", None),
        instance_region_ids=getattr(payload, "instance_region_ids", None),
        instance_sample_ids=getattr(payload, "instance_sample_ids", None),
        instance_embeddings=getattr(payload, "instance_embeddings", None),
        instance_composition=getattr(payload, "instance_composition", None),
        instance_centroids=getattr(payload, "instance_centroids", None),
        instance_graphs=getattr(payload, "instance_graphs", None),
    )


def _preforward_decision_to_dict(
    decision: PreforwardSubsamplingDecision,
) -> Dict[str, Any]:
    return {
        "requested": bool(decision.requested),
        "effective": bool(decision.effective),
        "fraction": float(decision.fraction),
        "seed": decision.seed,
        "reason": str(decision.reason),
        "strategy_name": decision.strategy_name,
        "runtime_enabled": decision.runtime_enabled,
    }


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None or not str(ckpt_path).strip():
        raise ValueError("Missing required `ckpt_path` for inference.")

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
    preforward_subsampling = _plan_preforward_subsampling(datamodule, cfg)
    if preforward_subsampling.requested and not preforward_subsampling.effective:
        raise ValueError(
            "Preforward subsampling was requested via aggregation.subsample_fraction < 1.0 "
            "but cannot be applied effectively. "
            "Use data.sampler.name in {'shadow_native','shadow_custom'} with "
            "data.sampler.runtime.enabled=true, or set aggregation.subsample_fraction=1.0. "
            f"reason={preforward_subsampling.reason!r}, "
            f"strategy={preforward_subsampling.strategy_name!r}, "
            f"runtime_enabled={preforward_subsampling.runtime_enabled!r}."
        )
    aggregation_metadata: Dict[str, Any] | None = None
    interpret_cfg = cfg.get("interpretability")
    export_interpretability = bool(interpret_cfg and interpret_cfg.get("enabled", False))
    include_instance_payload = _needs_instance_payload(cfg) or export_interpretability
    include_instance_embeddings = export_interpretability
    include_embeddings_payload = bool(cfg.embeddings.enabled) and bool(cfg.embeddings.save)
    include_node_embeddings = include_embeddings_payload and bool(cfg.embeddings.extract_node)
    previous_emit_setting = getattr(model, "_predict_emit_instance_payload", None)
    had_emit_setting = hasattr(model, "_predict_emit_instance_payload")
    previous_emit_instance_embedding_setting = getattr(
        model, "_predict_emit_instance_embeddings", None
    )
    had_emit_instance_embedding_setting = hasattr(model, "_predict_emit_instance_embeddings")
    previous_emit_instance_comp_setting = getattr(
        model, "_predict_emit_instance_composition", None
    )
    had_emit_instance_comp_setting = hasattr(model, "_predict_emit_instance_composition")
    previous_emit_instance_centroid_setting = getattr(
        model, "_predict_emit_instance_centroids", None
    )
    had_emit_instance_centroid_setting = hasattr(model, "_predict_emit_instance_centroids")
    previous_predict_comp_label = getattr(model, "_predict_composition_label", None)
    had_predict_comp_label = hasattr(model, "_predict_composition_label")
    previous_emit_embeddings_setting = getattr(model, "_predict_emit_embeddings_payload", None)
    had_emit_embeddings_setting = hasattr(model, "_predict_emit_embeddings_payload")
    previous_emit_node_embeddings_setting = getattr(model, "_predict_emit_node_embeddings", None)
    had_emit_node_embeddings_setting = hasattr(model, "_predict_emit_node_embeddings")
    setattr(model, "_predict_emit_instance_payload", include_instance_payload)
    setattr(model, "_predict_emit_instance_embeddings", include_instance_embeddings)
    setattr(model, "_predict_emit_instance_composition", include_instance_embeddings)
    setattr(model, "_predict_emit_instance_centroids", include_instance_embeddings)
    setattr(
        model,
        "_predict_composition_label",
        str((interpret_cfg or {}).get("composition_label", "cell_type")),
    )
    setattr(model, "_predict_emit_embeddings_payload", include_embeddings_payload)
    setattr(model, "_predict_emit_node_embeddings", include_node_embeddings)

    try:
        collected = collect_inference_payload(
            trainer=trainer,
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
            include_instance_payload=include_instance_payload,
            include_instance_embeddings=include_instance_embeddings,
            include_embeddings=include_embeddings_payload,
            include_node_embeddings=include_node_embeddings,
        )
    finally:
        if had_emit_setting:
            setattr(model, "_predict_emit_instance_payload", previous_emit_setting)
        else:
            delattr(model, "_predict_emit_instance_payload")
        if had_emit_instance_embedding_setting:
            setattr(
                model,
                "_predict_emit_instance_embeddings",
                previous_emit_instance_embedding_setting,
            )
        else:
            delattr(model, "_predict_emit_instance_embeddings")
        if had_emit_instance_comp_setting:
            setattr(
                model, "_predict_emit_instance_composition", previous_emit_instance_comp_setting
            )
        else:
            delattr(model, "_predict_emit_instance_composition")
        if had_emit_instance_centroid_setting:
            setattr(
                model, "_predict_emit_instance_centroids", previous_emit_instance_centroid_setting
            )
        else:
            delattr(model, "_predict_emit_instance_centroids")
        if had_predict_comp_label:
            setattr(model, "_predict_composition_label", previous_predict_comp_label)
        else:
            delattr(model, "_predict_composition_label")
        if had_emit_embeddings_setting:
            setattr(
                model,
                "_predict_emit_embeddings_payload",
                previous_emit_embeddings_setting,
            )
        else:
            delattr(model, "_predict_emit_embeddings_payload")
        if had_emit_node_embeddings_setting:
            setattr(
                model,
                "_predict_emit_node_embeddings",
                previous_emit_node_embeddings_setting,
            )
        else:
            delattr(model, "_predict_emit_node_embeddings")

    raw_pred_payload = collected.prediction_payload
    pred_payload = raw_pred_payload

    if bool(cfg.aggregation.enabled):
        aggregation_subsample_fraction = float(cfg.aggregation.get("subsample_fraction", 1.0))
        aggregation_subsample_seed = cfg.aggregation.get("subsample_seed")
        if preforward_subsampling.effective:
            aggregation_subsample_fraction = 1.0
            aggregation_subsample_seed = None
        aggregated = aggregate_group_logits(
            pred_payload,
            mode=str(cfg.aggregation.mode),
            bag_scope=str(cfg.aggregation.get("bag_scope", "patch")),
            subsample_fraction=aggregation_subsample_fraction,
            subsample_seed=aggregation_subsample_seed,
        )
        aggregation_metadata = dict(getattr(aggregated, "metadata", {}))
        aggregation_metadata["preforward_subsampling_applied"] = bool(
            preforward_subsampling.effective
        )
        aggregation_metadata["preforward_subsampling"] = _preforward_decision_to_dict(
            preforward_subsampling
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
    if include_embeddings_payload:
        emb_payload = collected.embedding_payload
        if emb_payload is None:
            raise ValueError("Embeddings were requested but no embedding payload was collected.")
        emb_frame = embeddings_to_dataframe(emb_payload)
        embeddings_path = output_dir / str(cfg.embeddings.filename)
        write_dataframe(emb_frame, embeddings_path)

        if include_node_embeddings:
            node_frame = node_embeddings_to_dataframe(emb_payload)
            node_path = output_dir / str(cfg.embeddings.node_filename)
            write_dataframe(node_frame, node_path)

    interpretability_payload: Dict[str, Any] | None = None
    if export_interpretability:
        composition_prefix = str((interpret_cfg or {}).get("composition_prefix", "comp_"))
        composition_label = str((interpret_cfg or {}).get("composition_label", "cell_type"))
        composition_names = None
        if raw_pred_payload.instance_composition is not None:
            composition_names = resolve_composition_column_names(
                datamodule,
                composition_label=composition_label,
                composition_prefix=composition_prefix,
                width=int(raw_pred_payload.instance_composition.shape[1]),
            )
        instance_frame = build_instance_table(
            raw_pred_payload,
            composition_prefix=composition_prefix,
            embedding_prefix=str((interpret_cfg or {}).get("embedding_prefix", "inst_emb")),
            composition_column_names=composition_names,
            require_composition=bool((interpret_cfg or {}).get("require_composition", True)),
            id_column=str((interpret_cfg or {}).get("id_column", "instance_id")),
            bag_id_column=str((interpret_cfg or {}).get("bag_id_column", "bag_id")),
            score_column=str((interpret_cfg or {}).get("score_column", "score")),
            score_mode=str((interpret_cfg or {}).get("score_mode", "sigmoid")),
            score_logit_index=int((interpret_cfg or {}).get("score_logit_index", 0)),
            attention_column=str((interpret_cfg or {}).get("attention_column", "attention")),
            attention_class_index=(interpret_cfg or {}).get("attention_class_index"),
        )
        instance_path = output_dir / str(
            (interpret_cfg or {}).get("instance_filename", "instance_table.csv")
        )
        write_dataframe(instance_frame, instance_path)

        spatial_path = None
        spatial_cfg = (interpret_cfg or {}).get("spatial", {})
        if bool(spatial_cfg.get("enabled", False)):
            spatial_frame = build_spatial_table(
                instance_frame,
                raw_pred_payload,
                id_column=str((interpret_cfg or {}).get("id_column", "instance_id")),
                undirected=bool(spatial_cfg.get("undirected", True)),
            )
            spatial_path = output_dir / str(
                (interpret_cfg or {}).get("spatial_filename", "spatial_table.csv")
            )
            write_dataframe(spatial_frame, spatial_path)

        interpretability_payload = {
            "instance_table_path": str(instance_path),
            "spatial_table_path": str(spatial_path) if spatial_path is not None else None,
            "rows": int(len(instance_frame)),
        }

    summary_path = output_dir / str(cfg.output.summary_filename)
    summary_payload = build_summary_payload(
        prediction_path=prediction_path,
        metrics_path=metrics_path,
        embeddings_path=embeddings_path,
        row_count=len(pred_frame),
    )
    summary_payload["metrics"] = metrics_payload
    if aggregation_metadata is not None:
        summary_payload["aggregation"] = aggregation_metadata
    if interpretability_payload is not None:
        summary_payload["interpretability"] = interpretability_payload
    write_json(summary_payload, summary_path)
    return summary_payload, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="inference/predict.yaml",
)
def main(cfg: DictConfig) -> None:
    extras(cfg)
    predict(cfg)


if __name__ == "__main__":
    main()
