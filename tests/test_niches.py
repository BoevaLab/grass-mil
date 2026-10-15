"""Niche discovery, attribution, descriptors, survival and the niches command."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grass_mil.niches import cell_margins, fit_niches
from grass_mil.niches.cli import run_niches
from grass_mil.niches.discovery import ClusteringConfig
from grass_mil.niches.survival import cox_univariate, logrank
from tests.helpers.cohort import compose_cfg


def test_margins_sum_to_the_bag_margin():
    rng = np.random.default_rng(0)
    logits, scores = rng.normal(size=(30, 2)), rng.normal(size=(30, 2))
    bias = np.array([0.2, -0.1])
    units = np.array(["a"] * 10 + ["b"] * 20)
    margin = cell_margins(logits, scores, bias, units)
    for unit, n in (("a", 10), ("b", 20)):
        m = units == unit
        w = np.exp(scores[m] - scores[m].max(0))
        w /= w.sum(0)
        bag = (w * logits[m]).sum(0) - bias
        assert margin[m].sum() / n == pytest.approx(bag[1] - bag[0], rel=1e-5)


def test_niches_are_numbered_by_margin_with_background_zero():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(c, 0.05, size=(100, 2)) for c in ((0, 1), (1, 0), (-1, 0))])
    margin = np.repeat([0.0, 2.0, -2.0], 100)
    model, niches = fit_niches(x, margin, ClusteringConfig(min_cluster_size=20, sample_size=200))
    assert model.n_niches == 3
    assert np.mean(niches[100:200] == 1) > 0.9 and np.mean(niches[200:] == 3) > 0.9
    assert np.array_equal(model.predict(x[:5]), niches[:5])


def test_cox_and_logrank_detect_a_strong_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    time = rng.exponential(1.0 / np.exp(0.8 * x))
    event = np.ones(300)
    hr, lo, hi, p = cox_univariate(x, time, event)
    assert lo < np.exp(0.8) < hi and p < 1e-6
    groups = (x > 0).astype(int)
    assert logrank(time, event, groups) < 1e-4
    assert logrank(time, event, rng.integers(0, 2, 300)) > 1e-3


def test_niches_command_writes_the_primary_results(cohort, cells_dir, tmp_path):
    (tmp_path / "outcomes.csv").write_text(
        "patient_id,time,event\n" + "\n".join(f"P{p:02d},{10 + p},{p % 2}" for p in range(6))
    )
    cfg = compose_cfg(
        "niches",
        [
            f"cells_dir={cells_dir / 'fold0'}",
            "units=all",
            "clustering.min_cluster_size=20",
            "descriptors.min_cells=10",
            "descriptors.n_perms=10",
            "descriptors.min_contacts=5",
            "n_boot=100",
            f"survival.file={tmp_path / 'outcomes.csv'}",
            "survival.score=[Niche 1]",
        ],
        tmp_path / "out",
    )
    out = run_niches(cfg)
    table = pd.read_csv(out / "niches.csv")
    assert {"niche", "share", "association", "margin", "margin_lo", "margin_hi", "frac_D"} <= set(
        table.columns
    )
    assert table["share"].sum() == pytest.approx(1.0)
    for name in ("moran", "enrichment", "cross_l", "contacts"):
        assert (out / "descriptors" / f"{name}.csv").exists()
    cells = pd.read_csv(out / "cells.csv")
    assert len(cells) == sum(120 for _ in cohort.regions)
    for figure in ("niche_panel", "embedding", "unit_composition", "survival"):
        assert (out / "figures" / f"{figure}.png").exists()
    assert len(list((out / "figures" / "units").glob("*.png"))) == 4
    assert "| Niche |" in (out / "report.md").read_text()
    assert len(pd.read_csv(out / "survival_cox.csv")) >= 1
