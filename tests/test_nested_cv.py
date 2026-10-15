"""Nested cross-validation."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from grass_mil.nested_cv import best_epoch, run_nested_cv
from tests.helpers.cohort import cohort_overrides, compose_cfg


def test_best_epoch_prefers_the_metric_then_the_loss_then_the_last_epoch():
    rows = [
        {"epoch": 0, "val/auroc": 0.6, "val/loss": 0.7},
        {"epoch": 1, "val/auroc": 0.8, "val/loss": 0.8},
        {"epoch": 2, "val/auroc": 0.8, "val/loss": 0.5},
    ]
    assert best_epoch(rows, "val/auroc", "max") == (2, "val/auroc")
    nan = [
        {"epoch": 0, "val/auroc": float("nan"), "val/loss": 0.9},
        {"epoch": 1, "val/auroc": float("nan"), "val/loss": 0.4},
    ]
    assert best_epoch(nan, "val/auroc", "max") == (2, "val/loss")
    assert best_epoch([{"epoch": 0}, {"epoch": 3}], "val/auroc", "max") == (4, "last")


def test_nested_cv_summary_and_out_of_fold_predictions(cohort, nested_cv):
    summary = json.loads((nested_cv / "nested_cv_summary.json").read_text())
    assert summary["regime"] == "finetune" and summary["metric"] == "auroc"
    assert len(summary["fold_values"]) == 3 and summary["complete"]
    oof = pd.read_csv(nested_cv / "oof_predictions.csv")
    assert sorted(oof["unit_id"]) == sorted(cohort.regions)
    patients = pd.read_csv(nested_cv / "oof_patient_predictions.csv")
    assert len(patients) == len(set(cohort.patient_of.values()))
    assert "auroc" in summary["patient"]


def test_refit_trains_without_validation(nested_cv):
    refit = nested_cv / "outer_0" / "refit"
    assert not (refit / "metrics_per_epoch.jsonl").exists()
    assert (refit / "checkpoints" / "last.ckpt").exists()
    done = json.loads((refit / "done.json").read_text())
    assert not any(k.startswith("val/") for k in done["metrics"])


def test_refit_follows_the_schedule_of_the_inner_runs(nested_cv):
    ckpt = nested_cv / "outer_0" / "refit" / "checkpoints" / "last.ckpt"
    finetune = torch.load(ckpt, weights_only=True)["hyper_parameters"]["finetune"]
    assert finetune["schedule_epochs"] == 2  # the inner runs' trainer.max_epochs


def test_outer_folds_must_keep_every_patient_in_one_fold(cohort, tmp_path):
    folds = json.loads(cohort.folds.read_text())
    moved = folds[0].pop()  # one region of a patient whose other region stays in fold 0
    folds[1].append(moved)
    (tmp_path / "folds.json").write_text(json.dumps(folds))
    cfg = compose_cfg(
        "nested_cv",
        cohort_overrides(cohort) + [f"nested_cv.folds_file={tmp_path / 'folds.json'}"],
        tmp_path / "out",
    )
    with pytest.raises(ValueError, match="outer folds 0 and 1"):
        run_nested_cv(cfg)


@pytest.mark.slow
def test_nested_cv_for_survival_selects_on_the_concordance(cohort, pretrained, tmp_path):
    cfg = compose_cfg(
        "nested_cv",
        cohort_overrides(cohort)
        + [
            "task=survival",
            "task.target_columns=[time,event]",
            f"nested_cv.folds_file={cohort.folds}",
            "nested_cv.inner_k=2",
            "nested_cv.outer_folds=[0]",
            "trainer.max_epochs=1",
            f"model.init_from_ckpt={pretrained}",
        ],
        tmp_path,
    )
    summary = run_nested_cv(cfg)
    assert summary["metric"] == "c_index" and summary["target_type"] == "survival"
    assert "risk" in pd.read_csv(tmp_path / "oof_predictions.csv").columns
