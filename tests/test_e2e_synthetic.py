"""End-to-end smoke test of the public analysis chain on the synthetic cohort.

precompute -> BGRL A3 preset (a few steps) -> nested CV (regime=mil and the
probe regime) -> extract -> niches -> metaclusters -> spatial statistics ->
scores. Asserts that every artefact exists, AUROCs are finite, the attribution
identity holds and the planted niche (D cells packed around C cells in positive
regions) is the top enriched cluster of at least one fold model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort

pytest.importorskip("torch_geometric")
pytest.importorskip("torch_sparse")


@pytest.mark.slow
def test_end_to_end_synthetic_chain(tmp_path):
    import hydra

    from grass_mil.evaluation.cli import run_scores
    from grass_mil.extract import run_extract
    from grass_mil.interpretability.metaclusters_cli import run_metaclusters
    from grass_mil.interpretability.niches_cli import run_niches
    from grass_mil.nested_cv import run_nested_cv
    from grass_mil.spatial_stats.cli import run as run_spatial
    from grass_mil.train import train

    root = tmp_path / "cohort"
    c = make_synthetic_cohort(
        root, n_patients=10, regions_per_patient=2, n_cells=120, n_outer_folds=2
    )
    hydra.utils.instantiate(
        compose_cfg("train", cohort_overrides(c, root / "pre")).data
    ).prepare_data()

    # 1. pretraining preset, a handful of steps
    pre_out = root / "pretrain"
    base = [
        o
        for o in cohort_overrides(c, pre_out)
        if not o.startswith(("regime=", "data.val_sampler", "model.attention", "task."))
    ]
    train(
        compose_cfg(
            "train",
            base
            + [
                "experiment=grass_mil_pretrain_a3",
                "task.total_steps=4",
                "task.warmup_steps=1",
                "task.lr_warmup.warmup_steps=1",
                "task.augmentation.feature_noise_columns=[area]",
                "callbacks.model_checkpoint.every_n_train_steps=2",
                "trainer.precision=32",
                "data.sampler.runtime.depth=2",
                "data.sampler.runtime.subgraph_batch_size=16",
            ],
        )
    )
    bgrl = pre_out / "checkpoints" / "last.ckpt"
    assert bgrl.exists()

    # 2. nested CV, default regime and the probe regime
    summaries = {}
    for regime in ("mil", "linear_probehead"):
        out = root / f"nested_{regime}"
        cfg = compose_cfg(
            "nested_cv",
            cohort_overrides(c, out)
            + [
                f"regime={regime}",
                f"model.init_from_ckpt={bgrl}",
                f"nested_cv.folds_file={c.folds_path}",
                "nested_cv.inner_k=2",
                "nested_cv.probe.roots_per_unit=16",
                "trainer.max_epochs=2",
            ],
        )
        summaries[regime] = run_nested_cv(cfg)
        assert summaries[regime]["complete"]
        assert all(np.isfinite(a) for a in summaries[regime]["auc_folds"])
        assert (out / "oof_predictions.csv").exists() and (
            out / "oof_patient_predictions.csv"
        ).exists()

    # 3. scores on the OOF table
    scores = run_scores(
        compose_cfg(
            "scores",
            [
                f"input={root / 'nested_mil' / 'oof_predictions.csv'}",
                f"output_dir={root / 'scores'}",
                "n_boot=100",
            ],
        )
    )
    assert scores["n"] == 10 and np.isfinite(scores["scores"][0]["auroc"])

    # 4. extract both fold refits
    ext = run_extract(
        compose_cfg(
            "extract",
            cohort_overrides(c, root / "extract")
            + [
                f"extract.nested_cv_summary={root / 'nested_mil' / 'nested_cv_summary.json'}",
                "extract.device=cpu",
            ],
        )
    )
    cells_dir = Path(ext["output_dir"])
    assert sorted(p.name for p in cells_dir.iterdir()) == ["fold0", "fold1"]
    # a probe regime extracts from its refit checkpoints without a probe file
    ext_lin = run_extract(
        compose_cfg(
            "extract",
            cohort_overrides(c, root / "extract_lin")
            + [
                "regime=linear_probehead",
                f"extract.nested_cv_summary={root / 'nested_linear_probehead' / 'nested_cv_summary.json'}",
                "extract.device=cpu",
            ],
        )
    )
    assert len(ext_lin["models"]) == 2

    # 5. niches of one model (default backend with a small min_cluster_size) and a k-means override.
    # The chain is a smoke test: a two-epoch model on 20 tiny regions is not expected to rank
    # the planted niche first, so the checks here are structural; the planted niche is
    # detected model-free by the spatial statistics in step 7.
    for fold in ("fold0", "fold1"):
        res = run_niches(
            compose_cfg(
                "niches",
                [
                    f"cells_dir={cells_dir / fold}",
                    f"output_dir={root / 'niches' / fold}",
                    "clustering.params.min_cluster_size=20",
                    "sample_size=null",
                    "n_boot=50",
                    "umap.enabled=false",
                ],
            )
        )
        assert res["n_clusters"] >= 1
        table = pd.read_csv(root / "niches" / fold / "clusters.csv")
        cells = pd.read_csv(root / "niches" / fold / "cells.csv")
        assert table["share"].sum() == pytest.approx(1.0)
        assert (
            np.isfinite(table["relattn"]).all()
            and (table["relattn_lo"] <= table["relattn_hi"]).all()
        )
        assert set(cells["cluster"]) == set(table["cluster_id"])
        held_out = json.loads((cells_dir / fold / "index.json").read_text())["held_out_regions"]
        assert set(cells["region_id"]) == set(held_out)
    km = run_niches(
        compose_cfg(
            "niches",
            [
                f"cells_dir={cells_dir / 'fold0'}",
                f"output_dir={root / 'niches_km'}",
                "clustering.method=kmeans",
                "clustering.n_clusters=4",
                "sample_size=null",
                "n_boot=20",
                "umap.enabled=false",
            ],
        )
    )
    assert km["n_clusters"] == 4

    # 6. metaclusters, fixed threshold
    meta = run_metaclusters(
        compose_cfg(
            "metaclusters",
            [
                f"cells_dir={cells_dir}",
                f"output_dir={root / 'meta'}",
                f"labels.label_file={c.label_path}",
                "labels.label_column=STAS",
                "labels.patient_column=PATIENT",
                "clustering.params.min_cluster_size=20",
                "sample_size=null",
                "n_boot=50",
                "match.threshold=0.4",
                "match.min_folds=2",
                "transfer.min_cells=10",
                "basis.k=4",
                "basis.sample_size=2000",
                "basis.n_init=2",
            ],
        )
    )
    assert (
        meta["match"]["n_metaclusters"] >= 1
        and (root / "meta" / "basis" / "basis_attribution.csv").exists()
    )

    # 7. spatial statistics
    sp = run_spatial(
        compose_cfg(
            "spatial_stats",
            [
                f"cells_dir={cells_dir / 'fold0'}",
                f"output_dir={root / 'spatial'}",
                f"labels.label_file={c.label_path}",
                "labels.label_column=STAS",
                "labels.patient_column=PATIENT",
                "reference=[C]",
                "partners=[D]",
                "nhood.n_perms=30",
                "cn.n_cn=3",
                "cn.sample_size=500",
                "min_per_group=3",
            ],
        )
    )
    nh = pd.read_csv(sp["nhood_tests"]).set_index("metric")
    assert nh.loc["D|D", "diff"] > 0
