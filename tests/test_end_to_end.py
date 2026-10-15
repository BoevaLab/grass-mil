"""The documented workflow, through the installed commands, on synthetic data."""

from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd
import pytest

from tests.helpers.cohort import cohort_overrides

pytestmark = pytest.mark.slow


def _run(command: str, *args: str, cwd) -> None:
    exe = shutil.which(command)
    if exe is None:
        pytest.skip("the package is not installed")
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}
    result = subprocess.run([exe, *args], cwd=cwd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-3000:]


def test_workflow(cohort, tmp_path):
    data = cohort_overrides(cohort, processed="processed_e2e", finetune=False)
    small = ["finetune.hyperbatch_size=3", "finetune.probe.roots_per_unit=16"]
    pre, cv, cells = tmp_path / "pretrain", tmp_path / "cv", tmp_path / "cells"
    _run(
        "grass-mil-train",
        *data,
        "regime=pretrain",
        "pretrain.total_steps=3",
        "pretrain.warmup_steps=1",
        f"hydra.run.dir={pre}",
        cwd=tmp_path,
    )
    ckpt = pre / "checkpoints" / "last.ckpt"
    _run(
        "grass-mil-nested-cv",
        *data,
        *small,
        f"model.init_from_ckpt={ckpt}",
        f"nested_cv.folds_file={cohort.folds}",
        "nested_cv.inner_k=2",
        "trainer.max_epochs=1",
        f"hydra.run.dir={cv}",
        cwd=tmp_path,
    )
    _run(
        "grass-mil-extract",
        *data,
        f"extract.nested_cv_summary={cv / 'nested_cv_summary.json'}",
        f"extract.output_dir={cells}",
        f"hydra.run.dir={tmp_path / 'x'}",
        cwd=tmp_path,
    )
    _run(
        "grass-mil-niches",
        f"cells_dir={cells / 'fold1'}",
        "clustering.min_cluster_size=20",
        "descriptors.min_cells=10",
        "descriptors.n_perms=5",
        "n_boot=50",
        f"hydra.run.dir={tmp_path / 'niches'}",
        cwd=tmp_path,
    )
    _run(
        "grass-mil-predict",
        *data,
        f"ckpt_path={cv / 'outer_1' / 'refit' / 'checkpoints' / 'last.ckpt'}",
        f"hydra.run.dir={tmp_path / 'predict'}",
        cwd=tmp_path,
    )
    assert len(pd.read_csv(tmp_path / "niches" / "niches.csv")) >= 1
    assert len(pd.read_csv(tmp_path / "predict" / "predictions.csv")) == len(cohort.regions)
