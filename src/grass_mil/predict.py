"""``grass-mil-predict``: unit (and patient) predictions of a trained model on a cohort."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from grass_mil.data.samplers import EgoGraphLoader
from grass_mil.data.splits import read_ids
from grass_mil.inference.cell_cache import scoring_sampler
from grass_mil.inference.predictions import frame_metrics, patient_frame, prediction_frame
from grass_mil.models import SupervisedModule, pool_by_bag
from grass_mil.runtime import resolve_device, start_run

__all__ = ["run_predict", "main"]

log = logging.getLogger(__name__)


@torch.no_grad()
def run_predict(cfg: DictConfig) -> dict:
    start_run(cfg)
    pcfg = cfg.predict
    if pcfg.patient_aggregation not in ("mean", "merged_bag"):
        raise ValueError("predict.patient_aggregation must be 'mean' or 'merged_bag'.")
    out_dir = Path(str(cfg.paths.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(pcfg.device)
    model = SupervisedModule.from_checkpoint(str(cfg.ckpt_path)).to(device)
    kind = model.task["target_type"]
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.prepare_data()
    model.input_spec.check_matches(datamodule.input_spec, str(cfg.ckpt_path))
    units = datamodule.all_units().select(read_ids(pcfg.get("units")))

    # Every root training could draw from, as when the test split is scored.
    sampler = scoring_sampler(model.trained_sampler, int(pcfg.batch_size), every_cell=False)
    unit_of, logits, scores, targets = [], [], [], {}
    for batch in EgoGraphLoader(units, sampler, mode="chunks", roots_per_unit=None):
        _, instance_logits, attention_scores = model.forward_instances(batch.to(device))
        unit_of += list(batch.unit_id)
        logits.append(instance_logits.float().cpu())
        scores.append(attention_scores.float().cpu())
        if getattr(batch, "graph_y", None) is not None:
            targets[batch.unit_id[0]] = batch.graph_y[0].float().cpu()
    logits, scores = torch.cat(logits), torch.cat(scores)
    unit_ids, bag_logits = pool_by_bag(logits, scores, unit_of)
    labelled = len(targets) == len(unit_ids)
    frame = prediction_frame(
        unit_ids,
        bag_logits,
        target_type=kind,
        targets=torch.stack([targets[u] for u in unit_ids]) if labelled else None,
        meta=datamodule.unit_table(units),
    )
    frame.to_csv(out_dir / "predictions.csv", index=False)
    summary: dict = {"n_units": len(frame), "ckpt_path": str(cfg.ckpt_path)}
    if labelled:
        summary["unit_metrics"] = frame_metrics(frame, kind)

    if "patient_id" in frame.columns:
        try:
            patients = patient_frame(frame, kind)
        except ValueError as err:
            log.warning(f"No patient predictions: {err}")
            patients = None
        if patients is not None and pcfg.patient_aggregation == "merged_bag":
            # One attention bag over every cell of the patient's units.
            patient_of = pd.Series(unit_of).map(frame.set_index("unit_id")["patient_id"])
            known = patient_of.notna().to_numpy()
            ids, pooled = pool_by_bag(logits[known], scores[known], patient_of[known].tolist())
            merged = prediction_frame(ids, pooled, target_type=kind).set_index("unit_id")
            patients = patients.set_index("patient_id")
            patients[merged.columns] = merged
            patients = patients.reset_index()
        if patients is not None:
            patients.to_csv(out_dir / "patient_predictions.csv", index=False)
            if labelled:
                summary["patient_metrics"] = frame_metrics(patients, kind)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    log.info(f"Predictions for {len(frame)} units written to {out_dir}")
    return summary


@hydra.main(version_base="1.3", config_path="configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    run_predict(cfg)


if __name__ == "__main__":
    main()
