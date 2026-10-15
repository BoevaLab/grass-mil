"""Prediction with a checkpoint and per-cell extraction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from grass_mil.inference.cell_cache import CellCache
from grass_mil.inference.predictions import patient_frame
from grass_mil.predict import run_predict
from tests.helpers.cohort import cohort_overrides, compose_cfg


@pytest.mark.parametrize("aggregation", ["mean", "merged_bag"])
def test_predict_needs_only_the_checkpoint_and_reports_auroc(
    cohort, nested_cv, tmp_path, aggregation
):
    ckpt = nested_cv / "outer_0" / "refit" / "checkpoints" / "last.ckpt"
    cfg = compose_cfg(
        "predict",
        cohort_overrides(cohort)
        + [f"ckpt_path={ckpt}", f"predict.patient_aggregation={aggregation}"],
        tmp_path,
    )
    summary = run_predict(cfg)
    assert np.isfinite(summary["unit_metrics"]["auroc"])
    assert "auroc" in summary["patient_metrics"]
    preds = pd.read_csv(tmp_path / "predictions.csv")
    assert sorted(preds["unit_id"]) == sorted(cohort.regions)
    assert json.loads((tmp_path / "metrics.json").read_text())["n_units"] == len(cohort.regions)
    assert len(pd.read_csv(tmp_path / "patient_predictions.csv")) == 6


def test_predict_refuses_a_cohort_preprocessed_differently(cohort, nested_cv, tmp_path):
    ckpt = nested_cv / "outer_0" / "refit" / "checkpoints" / "last.ckpt"
    overrides = cohort_overrides(cohort, processed="processed_cutoff30")
    cfg = compose_cfg(
        "predict", overrides + [f"ckpt_path={ckpt}", "data.graph.neighbor_cutoff=30"], tmp_path
    )
    with pytest.raises(ValueError, match="preprocessing"):
        run_predict(cfg)


def test_patient_scores_need_one_target_per_patient():
    units = pd.DataFrame(
        {"unit_id": ["a", "b"], "patient_id": ["p", "p"], "prob_0": [0.2, 0.6], "label": [0, 1]}
    )
    with pytest.raises(ValueError, match="different targets"):
        patient_frame(units, "classification")
    units["label"] = 1
    assert patient_frame(units, "classification")["prob_0"].tolist() == [pytest.approx(0.4)]


def test_extract_writes_every_cell_of_every_unit(cohort, cells_dir):
    cache = CellCache.open(cells_dir / "fold0")
    assert sorted(cache.units) == sorted(cohort.regions)
    assert cache.held_out and set(cache.held_out) < set(cache.units)
    data = cache.load(cache.units[0])
    n = data["pos"].shape[0]
    for key in (
        "embedding",
        "attention_space",
        "instance_logits",
        "attention_logits",
        "cell_type",
        "cell_ids",
    ):
        assert data[key].shape[0] == n
    assert data["attention_space"].shape[1] == 8
    info = cache.unit_info(cache.units[0])
    assert info["patient_id"] == cohort.patient_of[cache.units[0]]
    assert info["targets"] == [float(cohort.label_of[cache.units[0]])]
