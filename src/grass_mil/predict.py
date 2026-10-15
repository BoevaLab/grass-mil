"""``grass-mil-predict``: unit (and patient) predictions of a trained model on a cohort."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from grass_mil.data.datamodule import GraphUnitDataset
from grass_mil.data.samplers import EgoGraphLoader
from grass_mil.data.splits import read_ids
from grass_mil.inference.cell_cache import sampler_of_checkpoint
from grass_mil.inference.predictions import frame_metrics, patient_frame, prediction_frame
from grass_mil.models import SupervisedModule, attention_pool
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
    ckpt = torch.load(str(cfg.ckpt_path), map_location="cpu", weights_only=False)
    model = SupervisedModule.from_checkpoint(str(cfg.ckpt_path)).to(device)
    kind = model.task["target_type"]
    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.prepare_data()
    if tuple(datamodule.input_spec.cell_types) != tuple(model.input_spec.cell_types):
        raise ValueError("The cohort's cell types differ from the ones the model was trained on.")
    units = datamodule.all_units()
    wanted = read_ids(pcfg.get("units"))
    if wanted is not None:
        wanted = set(wanted)
        units = GraphUnitDataset(
            [e for e in units.entries if {e.unit_id, e.region_id, e.sample_id} & wanted],
            units.labels,
        )
    sampler = sampler_of_checkpoint(ckpt, int(pcfg.batch_size))
    meta = datamodule.unit_table(units)

    unit_ids, bag_logits, targets, instances = [], [], [], {}
    loader = EgoGraphLoader(units, sampler, mode="chunks", roots_per_unit=None)
    current: dict[str, list] = {}
    for batch in loader:
        unit = batch.unit_id[0]
        _, logits, scores = model.forward_instances(batch.to(device))
        current.setdefault(unit, [[], [], getattr(batch, "graph_y", None)])
        current[unit][0].append(logits.float().cpu())
        current[unit][1].append(scores.float().cpu())
    for unit, (logits, scores, y) in current.items():
        logits, scores = torch.cat(logits), torch.cat(scores)
        unit_ids.append(unit)
        bag_logits.append(attention_pool(logits, scores)[0])
        targets.append(None if y is None else y[0].cpu())
        instances[unit] = (logits, scores)
    labelled = all(t is not None for t in targets)
    frame = prediction_frame(
        unit_ids,
        torch.stack(bag_logits),
        target_type=kind,
        targets=torch.stack(targets) if labelled else None,
        meta=meta,
    )
    frame.to_csv(out_dir / "predictions.csv", index=False)
    summary: dict = {"n_units": len(frame), "ckpt_path": str(cfg.ckpt_path)}
    if labelled:
        summary["unit_metrics"] = frame_metrics(frame, kind)

    if "patient_id" in frame.columns:
        if pcfg.patient_aggregation == "mean":
            patients = patient_frame(frame, kind)
        else:
            rows = []
            for patient, group in frame.groupby("patient_id"):
                logits = torch.cat([instances[u][0] for u in group["unit_id"]])
                scores = torch.cat([instances[u][1] for u in group["unit_id"]])
                rows.append((patient, attention_pool(logits, scores)[0], group.index[0]))
            patients = prediction_frame(
                [r[0] for r in rows],
                torch.stack([r[1] for r in rows]),
                target_type=kind,
                targets=torch.stack([targets[i] for _, _, i in rows]) if labelled else None,
            ).rename(columns={"unit_id": "patient_id"})
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
