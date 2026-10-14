"""Phase 7: cross-fold metaclusters, transfer and shared basis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort

pytest.importorskip("sklearn")


def test_hellinger_range_and_constrained_linkage_never_merges_same_fold():
    from grass_mil.interpretability.metaclusters.match import (
        constrained_average_linkage,
        hellinger_matrix,
        scan_threshold,
    )

    comp = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    D = hellinger_matrix(comp)
    assert D.shape == (5, 5) and np.allclose(D, D.T) and D.min() >= 0 and D.max() <= 1 + 1e-9
    assert D[0, 3] == 0 and D[0, 1] == pytest.approx(1.0)
    fold = np.array(["f0", "f1", "f0", "f1", "f2"])
    groups = constrained_average_linkage(D, fold, 0.9)
    for g in groups:
        assert len(set(fold[g])) == len(g)  # cannot-link respected
    assert sorted(sorted(g) for g in groups)[0] == [0, 3]  # identical compositions merge
    thr, best, scan = scan_threshold(D, fold, grid=[0.05, 0.5, 0.9], min_folds=2)
    assert scan.n_replicated.max() == sum(len(g) >= 2 for g in best)
    # ties -> fewest metaclusters
    tied = scan[scan.n_replicated == scan.n_replicated.max()]
    assert len(best) == tied.n_metaclusters.min()


def test_patient_shares_and_tests():
    from grass_mil.interpretability.metaclusters.match import patient_shares, patient_tests

    region_labels = {
        "r0": np.array([0, 0, 1, -1]),
        "r1": np.array([1, 1, 1, 1]),
        "r2": np.array([0, 0, 0, 0]),
    }
    region_fold = {"r0": "f0", "r1": "f0", "r2": "f1"}
    id_to_meta = {"f0_C0": 1, "f0_C1": 2, "f1_C0": 1}
    reg, pat = patient_shares(
        region_labels, region_fold, id_to_meta, {"r0": "P", "r1": "P", "r2": "Q"}, [1, 2]
    )
    assert (
        reg.loc["r0", "M1"] == 0.5 and reg.loc["r0", "M2"] == 0.25 and reg.loc["r2", "M1"] == 1.0
    )
    assert (reg.sum(axis=1) <= 1 + 1e-9).all() and pat.loc["P", "M2"] == pytest.approx(
        (0.25 + 1.0) / 2
    )
    shares = pd.DataFrame(
        {"M1": np.r_[np.ones(5), np.zeros(5)], "M2": np.random.default_rng(0).uniform(size=10)},
        index=[f"p{i}" for i in range(10)],
    )
    y = pd.Series(np.r_[np.ones(5), np.zeros(5)].astype(int), index=shares.index)
    T = patient_tests(shares, y)
    assert set(T.columns) >= {"meta", "share_pos", "share_neg", "auroc", "p", "q"}
    assert T.set_index("meta").loc[1, "auroc"] == 1.0


@pytest.fixture(scope="module")
def two_fold_caches(tmp_path_factory):
    pytest.importorskip("torch_geometric")
    import hydra

    from grass_mil.extract import run_extract
    from grass_mil.train import train

    root = tmp_path_factory.mktemp("cohort7")
    c = make_synthetic_cohort(
        root, n_patients=6, regions_per_patient=2, n_cells=80, n_outer_folds=2
    )
    hydra.utils.instantiate(
        compose_cfg("train", cohort_overrides(c, root / "pre")).data
    ).prepare_data()
    ckpts = {}
    for i in range(2):
        test_ids = c.folds[i]
        train_ids = [r for j, f in enumerate(c.folds) if j != i for r in f]
        out = root / f"train{i}"
        train(
            compose_cfg(
                "train",
                cohort_overrides(c, out)
                + [
                    "trainer.max_epochs=1",
                    f"data.split.explicit.train=[{','.join(train_ids)}]",
                    f"data.split.explicit.val=[{','.join(test_ids)}]",
                    f"data.split.explicit.test=[{','.join(test_ids)}]",
                ],
            )
        )
        ckpts[f"fold{i}"] = (out / "checkpoints" / "last.ckpt", test_ids)
    overrides = cohort_overrides(c, root / "extract") + ["extract.device=cpu"]
    for name, (ckpt, held) in ckpts.items():
        overrides += [
            f"+extract.models.{name}={ckpt}",
            f"+extract.held_out.{name}=[{','.join(held)}]",
        ]
    result = run_extract(compose_cfg("extract", overrides))
    return c, Path(result["output_dir"])


def test_metaclusters_cli_all_stages(two_fold_caches, tmp_path):
    from grass_mil.interpretability.metaclusters_cli import run_metaclusters

    c, cells_dir = two_fold_caches
    cfg = compose_cfg(
        "metaclusters",
        [
            f"cells_dir={cells_dir}",
            f"output_dir={tmp_path / 'meta'}",
            f"labels.label_file={c.label_path}",
            "labels.id_column=REGION_ID",
            "labels.label_column=STAS",
            "labels.patient_column=PATIENT",
            "clustering.method=kmeans",
            "clustering.n_clusters=3",
            "sample_size=null",
            "n_boot=20",
            "match.min_folds=2",
            "transfer.min_cells=5",
            "basis.k=4",
            "basis.sample_size=1000",
            "basis.n_init=2",
        ],
    )
    summary = run_metaclusters(cfg)
    assert set(summary["cluster"]) == {"fold0", "fold1"}
    out = tmp_path / "meta"
    fc = pd.read_csv(out / "cluster" / "fold_clusters.csv")
    assert set(fc.fold) == {"fold0", "fold1"} and len(fc) == 6
    mt = pd.read_csv(out / "match" / "metaclusters.csv")
    assert mt.n_folds.max() <= 2 and summary["match"]["n_metaclusters"] == len(mt)
    matched = pd.read_csv(out / "match" / "fold_clusters_matched.csv")
    for _, g in matched.groupby("meta"):
        assert g.fold.nunique() == len(g)  # cannot-link
    shares = pd.read_csv(out / "match" / "metacluster_patient_shares.csv", index_col=0)
    assert len(shares) == 6 and (shares.sum(axis=1) <= 1 + 1e-6).all()
    assert (out / "match" / "metacluster_patient_tests.csv").exists()
    ML = np.load(out / "transfer" / "transfer_metalabels.npy")
    assert ML.shape[0] == 2 and ML.shape[1] == sum(80 for _ in c.region_ids)
    ari = pd.read_csv(out / "transfer" / "transfer_model_ari.csv", index_col=0)
    assert ari.loc["fold0", "fold0"] == pytest.approx(1.0)
    assert all(v > 0.99 for v in summary["transfer"]["refit_agreement"].values())
    basis = pd.read_csv(out / "basis" / "basis_attribution.csv")
    assert len(basis) == 4 and {"relattn_heldout", "relattn_train", "relattn_all"} <= set(
        basis.columns
    )
    assert (out / "basis" / "report_basis.md").exists() and (
        out / "metaclusters_summary.json"
    ).exists()

    # fixed threshold + stage re-run reuses stage-1 outputs
    cfg2 = compose_cfg(
        "metaclusters",
        [
            f"cells_dir={cells_dir}",
            f"output_dir={tmp_path / 'meta'}",
            f"labels.label_file={c.label_path}",
            "labels.label_column=STAS",
            "labels.patient_column=PATIENT",
            "stage=match",
            "match.threshold=0.4",
            "match.min_folds=2",
        ],
    )
    s2 = run_metaclusters(cfg2)
    assert s2["match"]["threshold"] == 0.4
