"""``grass-mil-nested-cv``: nested, patient-grouped cross-validation (preprint protocol).

Per outer fold ``i`` (``folds_file``): development = all other folds, split into
``inner_k`` stratified patient-grouped inner folds; every inner run trains the
selected regime and reports its best epoch; the refit trains on the whole
development set for the median best epoch and is scored once on fold ``i`` with
every cell of every region. One out-of-fold prediction per region and a mean
per patient are written alongside the per-fold summary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import hydra
import lightning as L
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.data.components.splits import (  # noqa: E402
    LabelTable,
    load_fold_json,
    load_labels,
    make_inner_folds,
)
from grass_mil.train import train  # noqa: E402
from grass_mil.utils import RankedLogger, extras  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["run_nested_cv", "best_epoch_from_jsonl", "main"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def best_epoch_from_jsonl(
    path: str | Path,
    *,
    metric: str,
    mode: str = "max",
    fallback_metric: Optional[str] = "val/loss_region",
) -> Optional[tuple[int, str]]:
    """1-based number of epochs reaching the best ``metric`` (earliest on ties).

    Returns ``(epochs, metric_used)``. When ``metric`` was never logged (e.g. the
    validation fold holds a single class, so AUROC is undefined) the
    ``fallback_metric`` is minimised instead, and failing that the last epoch is
    returned with ``"last"``.
    """
    path = Path(path)
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    usable = [r for r in rows if r.get(metric) is not None]
    if usable:
        sign = 1.0 if mode == "max" else -1.0
        best = max(usable, key=lambda r: (sign * float(r[metric]), -int(r["epoch"])))
        return int(best["epoch"]) + 1, metric
    if fallback_metric:
        usable = [r for r in rows if r.get(fallback_metric) is not None]
        if usable:
            best = min(usable, key=lambda r: (float(r[fallback_metric]), int(r["epoch"])))
            return int(best["epoch"]) + 1, fallback_metric
    return int(max(int(r["epoch"]) for r in rows)) + 1, "last"


def _reduce_epochs(epochs: Sequence[int], how: str, minimum: int) -> int:
    if not epochs:
        raise ValueError("No inner-fold epochs to reduce.")
    how = str(how).lower()
    if how == "median":
        value = statistics.median(epochs)
    elif how == "mean":
        value = statistics.mean(epochs)
    elif how == "max":
        value = max(epochs)
    else:
        raise ValueError("nested_cv.epoch_reduce must be median, mean or max.")
    return max(int(minimum), int(value))


def _resolve_label_column(cfg: DictConfig) -> str:
    explicit = cfg.nested_cv.get("label_column")
    if explicit:
        return str(explicit)
    targets = cfg.task.get("target_columns")
    if targets:
        return str(list(targets)[0])
    tasks = cfg.data.graph_labels.get("tasks")
    if tasks:
        return str(list(tasks)[0])
    raise ValueError(
        "Set nested_cv.label_column (or task.target_columns / data.graph_labels.tasks) "
        "so the driver knows which label column stratifies the inner folds."
    )


def _probe_mode(cfg: DictConfig) -> str:
    """'always' | 'never' | 'auto' -> resolved to whether a probe is needed."""
    enabled = cfg.nested_cv.probe.get("enabled", "auto")
    if isinstance(enabled, bool):
        return "always" if enabled else "never"
    enabled = str(enabled).lower()
    if enabled not in {"auto", "true", "false"}:
        raise ValueError("nested_cv.probe.enabled must be auto, true or false.")
    if enabled == "auto":
        return "auto"
    return "always" if enabled == "true" else "never"


def _needs_probe(cfg: DictConfig) -> bool:
    mode = _probe_mode(cfg)
    if mode == "always":
        return True
    if mode == "never":
        return False
    return str(cfg.task.get("head_init_path") or "").strip().lower() == "auto"


def _split_hash(train_ids: Sequence[str], cfg: DictConfig) -> str:
    h = hashlib.sha1()
    h.update("|".join(sorted(str(r) for r in train_ids)).encode())
    h.update(str(cfg.model.get("init_from_ckpt")).encode())
    p = cfg.nested_cv.probe
    h.update(f"{p.roots_per_unit}|{p.seed}|{p.C}|{p.class_weight}".encode())
    h.update(f"{cfg.data.sampler.runtime.depth}|{cfg.data.sampler.runtime.num_neighbors}".encode())
    return h.hexdigest()[:10]


def _with_split(cfg: DictConfig, *, train_ids, val_ids, test_ids, output_dir: Path) -> DictConfig:
    fold_cfg = copy.deepcopy(cfg)
    with open_dict(fold_cfg):
        fold_cfg.data.split.explicit = {
            "train": [str(r) for r in train_ids],
            "val": [str(r) for r in val_ids],
            "test": [str(r) for r in test_ids],
            "id_kind": "region",
        }
        fold_cfg.paths.output_dir = str(output_dir)
        fold_cfg.trainer.default_root_dir = str(output_dir)
    return fold_cfg


def _fit_probe_for(cfg: DictConfig, *, train_ids, val_ids, tag: str) -> Path:
    from grass_mil.probe import run_probe

    cache_dir = Path(str(cfg.nested_cv.probe.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"probe_{tag}_{_split_hash(train_ids, cfg)}.pt"
    if out.exists() and bool(cfg.nested_cv.get("resume", True)):
        log.info(f"[{tag}] probe cached: {out}")
        return out
    probe_cfg = _with_split(
        cfg, train_ids=train_ids, val_ids=val_ids, test_ids=val_ids, output_dir=cache_dir / tag
    )
    with open_dict(probe_cfg):
        probe_cfg.task.head_init_path = None
        probe_cfg.probe = {
            "roots_per_unit": cfg.nested_cv.probe.roots_per_unit,
            "seed": cfg.nested_cv.probe.seed,
            "batch_size": cfg.nested_cv.probe.batch_size,
            "C": cfg.nested_cv.probe.C,
            "class_weight": cfg.nested_cv.probe.class_weight,
            "max_iter": cfg.nested_cv.probe.max_iter,
            "eval_splits": ["val"],
            "output_path": str(out),
            "device": cfg.nested_cv.probe.device,
        }
    run_probe(probe_cfg)
    return out


def _train_run(
    cfg: DictConfig,
    *,
    train_ids,
    val_ids,
    test_ids,
    output_dir: Path,
    head_init: Optional[Path],
    max_epochs: Optional[int] = None,
    refit: bool = False,
    test: bool = False,
) -> Dict[str, Any]:
    done_path = output_dir / "done.json"
    if done_path.exists() and bool(cfg.nested_cv.get("resume", True)):
        log.info(f"resume: {output_dir} already finished")
        return json.loads(done_path.read_text())
    fold_cfg = _with_split(
        cfg, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, output_dir=output_dir
    )
    with open_dict(fold_cfg):
        fold_cfg.task.head_init_path = str(head_init) if head_init is not None else None
        fold_cfg.train = True
        fold_cfg.test = bool(test)
        fold_cfg.task_name = f"{cfg.task_name}_{output_dir.name}"
        if max_epochs is not None:
            fold_cfg.trainer.max_epochs = int(max_epochs)
            fold_cfg.trainer.min_epochs = int(max_epochs)
        if refit and fold_cfg.get("callbacks"):
            # Fixed-length refit: no early stopping, no metric-based selection;
            # the last epoch's weights are the model.
            cbs = fold_cfg.callbacks
            if "early_stopping" in cbs:
                del cbs["early_stopping"]
            if "model_checkpoint" in cbs and cbs.model_checkpoint is not None:
                cbs.model_checkpoint.monitor = None
                cbs.model_checkpoint.save_top_k = 0
                cbs.model_checkpoint.save_last = True
                cbs.model_checkpoint.filename = "last_epoch"
    metrics, _objects = train(fold_cfg)
    numeric = {}
    for k, v in metrics.items():
        try:
            numeric[k] = float(v.item() if hasattr(v, "item") else v)
        except Exception:
            continue
    result = {
        "output_dir": str(output_dir),
        "metrics": numeric,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "max_epochs": int(fold_cfg.trainer.max_epochs),
        "head_init_path": str(head_init) if head_init is not None else None,
    }
    best = best_epoch_from_jsonl(
        output_dir / "metrics_per_epoch.jsonl",
        metric=str(cfg.nested_cv.selection_metric),
        mode=str(cfg.nested_cv.selection_mode),
    )
    result["best_epoch"] = best[0] if best else None
    result["selection_metric_used"] = best[1] if best else None
    done_path.write_text(json.dumps(result, indent=1))
    return result


def _read_test_predictions(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Refit run wrote no test predictions: {path}")
    return pd.read_csv(path)


def _auroc(y: Sequence[int], score: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def run_nested_cv(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    ncfg = cfg.nested_cv
    base_out = Path(str(cfg.paths.output_dir))
    base_out.mkdir(parents=True, exist_ok=True)

    label_column = _resolve_label_column(cfg)
    patient_column = ncfg.get("patient_column")
    table: LabelTable = load_labels(
        str(ncfg.label_file),
        id_column=str(ncfg.id_column),
        label_column=label_column,
        patient_column=str(patient_column) if patient_column else None,
    )
    folds = load_fold_json(str(ncfg.folds_file))
    labelled = set(table.labels.index)
    dropped = sum(len([r for r in f if r not in labelled]) for f in folds)
    if dropped:
        log.warning(f"{dropped} region(s) in the folds have no '{label_column}' label; dropped")
    folds = [[r for r in f if r in labelled] for f in folds]
    if any(len(f) == 0 for f in folds):
        raise ValueError("An outer fold is empty after dropping unlabelled regions.")

    # Precompute once (the inner runs re-partition the same processed graphs).
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    needs_probe = _needs_probe(cfg)
    if needs_probe and int(cfg.model.graph_head.get("num_layers", 2)) != 1:
        raise ValueError(
            "A probe pre-fit targets a linear instance head; set model.graph_head.num_layers=1 "
            "(the linear_* regimes do) or nested_cv.probe.enabled=false."
        )
    if not needs_probe and str(cfg.task.get("head_init_path") or "").lower() == "auto":
        raise ValueError(
            "task.head_init_path='auto' needs a probe but nested_cv.probe.enabled=false; "
            "pass an explicit task.head_init_path or enable the probe."
        )
    fixed_head = None
    if not needs_probe and cfg.task.get("head_init_path"):
        fixed_head = Path(str(cfg.task.head_init_path))
        log.warning(f"Using one fixed head init for every fold: {fixed_head}")

    selected = ncfg.get("outer_folds")
    outer_indices = list(range(len(folds))) if not selected else [int(i) for i in selected]
    inner_k = int(ncfg.inner_k)
    regime = str(cfg.get("regime_name", "custom"))
    log.info(
        f"Nested CV [{regime}]: {len(folds)} outer folds ({len(outer_indices)} selected), "
        f"inner k={inner_k} seed={int(ncfg.inner_seed)}, probe={'yes' if needs_probe else 'no'}"
    )

    fold_results: List[Dict[str, Any]] = []
    oof_frames: List[pd.DataFrame] = []
    for i in outer_indices:
        outer_dir = base_out / f"outer_{i}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        result_path = outer_dir / "result.json"
        if result_path.exists() and bool(ncfg.get("resume", True)):
            res = json.loads(result_path.read_text())
            log.info(f"[outer {i}] resume: AUROC {res['test_auroc']:.4f} (E={res['E']})")
            fold_results.append(res)
            oof_frames.append(pd.read_csv(outer_dir / "oof_predictions.csv"))
            continue

        test_ids = list(folds[i])
        dev_ids = sorted(r for j, f in enumerate(folds) if j != i for r in f)
        inner = make_inner_folds(dev_ids, table, k=inner_k, seed=int(ncfg.inner_seed))

        inner_epochs: List[int] = []
        inner_summaries: List[Dict[str, Any]] = []
        for j, (tr, va) in enumerate(inner):
            tag = f"o{i}_i{j}"
            head = fixed_head
            if needs_probe:
                head = _fit_probe_for(cfg, train_ids=tr, val_ids=va, tag=tag)
            res = _train_run(
                cfg,
                train_ids=tr,
                val_ids=va,
                test_ids=test_ids,
                output_dir=outer_dir / f"inner_{j}",
                head_init=head,
                test=False,
            )
            if res.get("best_epoch") is None:
                raise RuntimeError(
                    f"[{tag}] no '{ncfg.selection_metric}' in metrics_per_epoch.jsonl; "
                    "is validation enabled with region-level metrics?"
                )
            used = res.get("selection_metric_used")
            if used != str(ncfg.selection_metric):
                log.warning(
                    f"[{tag}] '{ncfg.selection_metric}' was never logged (single-class "
                    f"validation fold?); epoch chosen by '{used}'"
                )
            inner_epochs.append(int(res["best_epoch"]))
            inner_summaries.append(
                {
                    "inner": j,
                    "best_epoch": res["best_epoch"],
                    "selection_metric_used": used,
                    "n_train": len(tr),
                    "n_val": len(va),
                }
            )
            log.info(f"[{tag}] best epoch {res['best_epoch']} ({used})")

        E = _reduce_epochs(inner_epochs, str(ncfg.epoch_reduce), int(ncfg.min_refit_epochs))
        tag = f"o{i}_refit"
        head = fixed_head
        if needs_probe:
            head = _fit_probe_for(cfg, train_ids=dev_ids, val_ids=test_ids, tag=tag)
        res = _train_run(
            cfg,
            train_ids=dev_ids,
            val_ids=test_ids,
            test_ids=test_ids,
            output_dir=outer_dir / "refit",
            head_init=head,
            max_epochs=E,
            refit=True,
            test=True,
        )
        preds = _read_test_predictions(outer_dir / "refit")
        preds.insert(0, "outer", i)
        preds["patient_id"] = [table.group_of(r) for r in preds["region_id"].astype(str)]
        preds.to_csv(outer_dir / "oof_predictions.csv", index=False)
        score_col = "prob_class_1" if "prob_class_1" in preds.columns else None
        test_auroc = res["metrics"].get("test/auroc")
        if test_auroc is None and score_col is not None:
            test_auroc = _auroc(preds["label"], preds[score_col])
        fold_res = {
            "outer": i,
            "E": E,
            "inner_epochs": inner_epochs,
            "inner": inner_summaries,
            "test_auroc": float(test_auroc) if test_auroc is not None else float("nan"),
            "test_bal_acc": res["metrics"].get("test/bal_acc"),
            "n_test": len(test_ids),
            "n_test_evaluated": int(preds["region_id"].nunique()),
            "n_dev": len(dev_ids),
            "run_dir": str(outer_dir / "refit"),
            "head_init_path": res.get("head_init_path"),
        }
        if fold_res["n_test_evaluated"] != len(test_ids):
            log.warning(
                f"[outer {i}] evaluated {fold_res['n_test_evaluated']} of {len(test_ids)} regions"
            )
        result_path.write_text(json.dumps(fold_res, indent=1))
        fold_results.append(fold_res)
        oof_frames.append(preds)
        log.info(f"[outer {i}] E={E}  held-out AUROC {fold_res['test_auroc']:.4f}")

    # ------------------------------------------------------------------ summary
    oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    oof.to_csv(base_out / "oof_predictions.csv", index=False)
    aucs = [r["test_auroc"] for r in fold_results if r["test_auroc"] == r["test_auroc"]]
    summary: Dict[str, Any] = {
        "regime": regime,
        "task": label_column,
        "n_outer": len(fold_results),
        "n_outer_total": len(folds),
        "complete": len(fold_results) == len(folds),
        "inner_k": inner_k,
        "inner_seed": int(ncfg.inner_seed),
        "epoch_reduce": str(ncfg.epoch_reduce),
        "selection_metric": str(ncfg.selection_metric),
        "max_epochs": int(cfg.trainer.max_epochs),
        "probe": needs_probe,
        "encoder_ckpt": cfg.model.get("init_from_ckpt"),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "auc_folds": [round(a, 4) for a in aucs],
        "folds": fold_results,
    }
    if not oof.empty and "prob_class_1" in oof.columns:
        summary["pooled_region_auroc"] = _auroc(oof["label"], oof["prob_class_1"])
        agg = str(ncfg.get("patient_aggregation", "mean")).lower()
        if agg != "mean":
            raise ValueError(
                "nested_cv.patient_aggregation supports 'mean' here; use grass-mil-predict for merged bags."
            )
        pat = (
            oof.groupby("patient_id")
            .agg(
                prob_class_1=("prob_class_1", "mean"),
                label=("label", "max"),
                n_regions=("region_id", "nunique"),
                outer=("outer", "first"),
            )
            .reset_index()
        )
        pat.to_csv(base_out / "oof_patient_predictions.csv", index=False)
        summary["patient_auroc"] = _auroc(pat["label"], pat["prob_class_1"])
        summary["n_patients"] = int(len(pat))
    (base_out / str(ncfg.summary_filename)).write_text(json.dumps(summary, indent=1, default=str))
    log.info(
        f"== [{regime}] AUROC {summary['auc_mean']:.4f} +/- {summary['auc_sd']:.4f} over "
        f"{len(aucs)} outer folds; patient-level {summary.get('patient_auroc', float('nan')):.4f}"
    )
    return summary


@hydra.main(version_base="1.3", config_path="configs", config_name="nested_cv.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    summary = run_nested_cv(cfg)
    return summary.get("auc_mean")


if __name__ == "__main__":
    main()
