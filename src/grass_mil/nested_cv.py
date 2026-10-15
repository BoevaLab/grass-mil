"""``grass-mil-nested-cv``: nested, patient-grouped cross-validation.

For every outer fold the remaining folds form the development set, which is
split into group-aware inner folds. Every inner run reports the epoch with the
best validation metric (``task.selection``); the model is refitted on the whole
development set with the same learning-rate schedule, stopped at the reduced
number of epochs, without validation, and scored once on the held-out fold.
The summary reports the metric of every outer fold, its mean and standard
deviation, and the pooled out-of-fold metric at unit and patient level.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Sequence

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict

from grass_mil.data.datamodule import GraphUnitDataset
from grass_mil.data.splits import inner_folds, load_folds
from grass_mil.inference.predictions import frame_metrics, patient_frame
from grass_mil.runtime import read_epoch_metrics, start_run
from grass_mil.train import train

__all__ = ["run_nested_cv", "best_epoch", "main"]

log = logging.getLogger(__name__)


def best_epoch(rows: Sequence[dict[str, Any]], metric: str, mode: str) -> tuple[int, str]:
    """Number of epochs (1-based) reaching the best ``metric``, earliest on ties.

    Falls back to the lowest ``val/loss`` when ``metric`` is never finite (for
    example AUROC on a single-class validation fold), and to the last epoch
    without any validation metric. Returns ``(epochs, metric_used)``.
    """
    sign = 1.0 if mode == "max" else -1.0
    for name, s in ((metric, sign), ("val/loss", -1.0)):
        usable = [r for r in rows if r.get(name) is not None and np.isfinite(r[name])]
        if usable:
            best = max(usable, key=lambda r: (s * float(r[name]), -int(r["epoch"])))
            return int(best["epoch"]) + 1, name
    return (max(int(r["epoch"]) for r in rows) + 1 if rows else 1), "last"


def _reduce(epochs: Sequence[int], how: str, minimum: int) -> int:
    reducers = {"median": statistics.median, "mean": statistics.mean, "max": max}
    if how not in reducers:
        raise ValueError("nested_cv.epoch_reduce must be median, mean or max.")
    return max(int(minimum), int(round(reducers[how](epochs))))


def _run(
    cfg: DictConfig, *, train_ids, val_ids, test_ids, out: Path, max_epochs: int | None = None
) -> dict[str, Any]:
    """One training run on explicit unit lists (resumable through ``done.json``)."""
    done = out / "done.json"
    if done.exists() and cfg.nested_cv.resume:
        return json.loads(done.read_text())
    run_cfg = cfg.copy()
    with open_dict(run_cfg):
        run_cfg.data.split.train = list(train_ids)
        run_cfg.data.split.val = list(val_ids)
        run_cfg.data.split.test = list(test_ids)
        run_cfg.paths.output_dir = str(out)
        run_cfg.trainer.default_root_dir = str(out)
        run_cfg.test = bool(test_ids)
        if max_epochs is not None:
            # Stop at the selected epoch of the schedule the inner runs followed.
            run_cfg.finetune.schedule_epochs = cfg.finetune.schedule_epochs or int(
                cfg.trainer.max_epochs
            )
            run_cfg.trainer.max_epochs = run_cfg.trainer.min_epochs = int(max_epochs)
    out.mkdir(parents=True, exist_ok=True)
    metrics, _ = train(run_cfg)
    epochs, used = best_epoch(
        read_epoch_metrics(out / "metrics_per_epoch.jsonl"),
        cfg.task.selection.metric,
        cfg.task.selection.mode,
    )
    result = {
        "metrics": metrics,
        "best_epoch": epochs,
        "selection_metric_used": used,
        "output_dir": str(out),
    }
    done.write_text(json.dumps(result, indent=1))
    return result


def _check_outer_folds(
    folds: list[list[str]], groups: dict[str, str], classes: dict[str, int] | None
) -> None:
    """Every fold holds labelled units, and every unit and patient sits in one fold only."""
    if any(not f for f in folds):
        raise ValueError("An outer fold holds no labelled unit.")
    for i, fold in enumerate(folds if classes is not None else []):
        present = sorted({classes[u] for u in fold})
        if len(present) < 2:
            log.warning(
                f"Outer fold {i} holds only class {present[0]}: its AUROC is undefined and the "
                "fold is left out of the mean. Stratify the folds by class."
            )
    fold_of_unit: dict[str, int] = {}
    fold_of_group: dict[str, int] = {}
    for i, fold in enumerate(folds):
        for unit in fold:
            for key, seen in ((unit, fold_of_unit), (groups[unit], fold_of_group)):
                if seen.setdefault(key, i) != i:
                    raise ValueError(
                        f"'{key}' is in outer folds {seen[key]} and {i}: the folds must keep "
                        "every unit and every patient (or region) in one fold."
                    )


def run_nested_cv(cfg: DictConfig) -> dict[str, Any]:
    start_run(cfg)
    ncfg = cfg.nested_cv
    kind = str(cfg.task.target_type)
    out_dir = Path(str(cfg.paths.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    datamodule.prepare_data()
    units = datamodule.all_units()
    if datamodule.read_labels() is None:
        raise ValueError("Cross-validation needs labels (data.labels.file).")
    labelled = GraphUnitDataset(
        [e for e in units.entries if units.label_id(e) is not None], units.labels
    )
    groups = {e.unit_id: datamodule.group_of(e) for e in labelled.entries}
    classes = (
        {e.unit_id: units.labels.classes(units.label_id(e)) for e in labelled.entries}
        if kind == "classification"
        else None
    )
    folds = [
        sorted(e.unit_id for e in labelled.select(ids).entries)
        for ids in load_folds(str(ncfg.folds_file))
    ]
    _check_outer_folds(folds, groups, classes)
    selected = list(ncfg.outer_folds) if ncfg.outer_folds is not None else range(len(folds))
    # The reported metric of the task, whichever validation metric selects the epochs.
    metric = {"classification": "auroc", "regression": "rmse", "survival": "c_index"}[kind]

    results, oof = [], []
    for i in selected:
        fold_dir = out_dir / f"outer_{i}"
        result_path = fold_dir / "result.json"
        if result_path.exists() and ncfg.resume:
            results.append(json.loads(result_path.read_text()))
            oof.append(pd.read_csv(fold_dir / "oof_predictions.csv"))
            continue
        test = folds[i]
        dev = sorted(u for j, f in enumerate(folds) if j != i for u in f)
        inner = inner_folds(
            dev, groups=groups, classes=classes, k=int(ncfg.inner_k), seed=int(ncfg.inner_seed)
        )
        epochs = []
        for j, (tr, va) in enumerate(inner):
            res = _run(cfg, train_ids=tr, val_ids=va, test_ids=[], out=fold_dir / f"inner_{j}")
            if res["selection_metric_used"] != cfg.task.selection.metric:
                log.warning(
                    f"[outer {i} inner {j}] epoch chosen by {res['selection_metric_used']}"
                )
            epochs.append(int(res["best_epoch"]))
        n_epochs = _reduce(epochs, str(ncfg.epoch_reduce), int(ncfg.min_refit_epochs))
        refit = _run(
            cfg,
            train_ids=dev,
            val_ids=[],
            test_ids=test,
            out=fold_dir / "refit",
            max_epochs=n_epochs,
        )
        preds = pd.read_csv(fold_dir / "refit" / "test_predictions.csv")
        preds.insert(0, "outer", i)
        preds.to_csv(fold_dir / "oof_predictions.csv", index=False)
        fold = {
            "outer": i,
            "epochs": n_epochs,
            "inner_best_epochs": epochs,
            "n_dev": len(dev),
            "n_test": len(test),
            "metrics": frame_metrics(preds, kind),
            "run_dir": refit["output_dir"],
        }
        result_path.write_text(json.dumps(fold, indent=1))
        results.append(fold)
        oof.append(preds)
        log.info(
            f"[outer {i}] {n_epochs} epochs, held-out {metric} {fold['metrics'].get(metric, float('nan')):.4f}"
        )

    oof_frame = pd.concat(oof, ignore_index=True)
    oof_frame.to_csv(out_dir / "oof_predictions.csv", index=False)
    values = [r["metrics"].get(metric, float("nan")) for r in results]
    finite = [v for v in values if np.isfinite(v)]
    summary: dict[str, Any] = {
        "regime": cfg.get("regime_name"),
        "target_type": kind,
        "metric": metric,
        "fold_values": values,
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "pooled": frame_metrics(oof_frame, kind),
        "complete": len(results) == len(folds),
        "folds": results,
    }
    if "patient_id" in oof_frame.columns:
        try:
            patients = patient_frame(oof_frame, kind)
        except ValueError as err:
            log.warning(f"No patient-level metrics: {err}")
        else:
            patients.to_csv(out_dir / "oof_patient_predictions.csv", index=False)
            summary["patient"] = frame_metrics(patients, kind)
    (out_dir / "nested_cv_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log.info(
        f"{metric}: {summary['mean']:.4f} ± {summary['sd']:.4f} over {len(finite)} outer folds; "
        f"summary: {out_dir / 'nested_cv_summary.json'}"
    )
    return summary


@hydra.main(version_base="1.3", config_path="configs", config_name="nested_cv.yaml")
def main(cfg: DictConfig) -> None:
    run_nested_cv(cfg)


if __name__ == "__main__":
    main()
