"""Shared fixtures: one synthetic cohort and, built once per session, a pretrained
encoder, a cross-validation and the per-cell outputs of its models."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.cohort import Cohort, cohort_overrides, compose_cfg, make_cohort


@pytest.fixture(scope="session")
def cohort(tmp_path_factory) -> Cohort:
    return make_cohort(tmp_path_factory.mktemp("cohort"))


@pytest.fixture(scope="session")
def pretrained(cohort: Cohort, tmp_path_factory) -> Path:
    """``last.ckpt`` of a few BGRL steps."""
    from grass_mil.train import train

    out = tmp_path_factory.mktemp("pretrain")
    cfg = compose_cfg(
        "train",
        cohort_overrides(cohort, finetune=False)
        + [
            "regime=pretrain",
            "pretrain.total_steps=4",
            "pretrain.warmup_steps=1",
            "pretrain.augmentation.noise_features=[area]",
        ],
        out,
    )
    train(cfg)
    return out / "checkpoints" / "last.ckpt"


@pytest.fixture(scope="session")
def nested_cv(cohort: Cohort, pretrained: Path, tmp_path_factory) -> Path:
    """Output directory of a three-fold nested cross-validation."""
    from grass_mil.nested_cv import run_nested_cv

    out = tmp_path_factory.mktemp("nested_cv")
    cfg = compose_cfg(
        "nested_cv",
        cohort_overrides(cohort)
        + [
            f"model.init_from_ckpt={pretrained}",
            f"nested_cv.folds_file={cohort.folds}",
            "nested_cv.inner_k=2",
        ],
        out,
    )
    run_nested_cv(cfg)
    return out


@pytest.fixture(scope="session")
def cells_dir(cohort: Cohort, nested_cv: Path, tmp_path_factory) -> Path:
    """Per-cell outputs of every fold model (``<dir>/fold<i>``)."""
    from grass_mil.extract import run_extract

    out = tmp_path_factory.mktemp("extract")
    cfg = compose_cfg(
        "extract",
        cohort_overrides(cohort)
        + [f"extract.nested_cv_summary={nested_cv / 'nested_cv_summary.json'}"],
        out,
    )
    run_extract(cfg)
    return out / "cells"
