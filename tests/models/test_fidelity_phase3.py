"""Phase 3: explicit splits, probe pre-fit, nested cross-validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort

pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------


def test_explicit_split_from_lists_and_files(tmp_path):
    from grass_mil.data.components.splits import ExplicitSplit

    (tmp_path / "tr.json").write_text(json.dumps(["a", "b"]))
    (tmp_path / "va.txt").write_text("c\n\nd\n")
    split = ExplicitSplit.from_dict(
        {"train": str(tmp_path / "tr.json"), "val": str(tmp_path / "va.txt"), "test": ["e"]}
    )
    assert split.train == ["a", "b"] and split.val == ["c", "d"] and split.test == ["e"]
    assert split.lookup() == {"a": "train", "b": "train", "c": "val", "d": "val", "e": "test"}
    assert ExplicitSplit.from_dict({"train": None, "val": None, "test": None}) is None
    with pytest.raises(ValueError):
        ExplicitSplit.from_dict({"train": ["a"], "val": ["a"], "test": ["b"]})
    # val == test is the refit convention and is tolerated
    ExplicitSplit.from_dict({"train": ["a"], "val": ["b"], "test": ["b"]})


def test_fold_json_roundtrip_and_duplicates(tmp_path):
    from grass_mil.data.components.splits import load_fold_json, write_fold_json

    write_fold_json([["r1", "r2"], ["r3"]], tmp_path / "f.json")
    assert load_fold_json(tmp_path / "f.json") == [["r1", "r2"], ["r3"]]
    (tmp_path / "bad.json").write_text(json.dumps([["r1"], ["r1"]]))
    with pytest.raises(ValueError):
        load_fold_json(tmp_path / "bad.json")


def test_inner_folds_are_patient_safe_and_deterministic(tmp_path):
    from grass_mil.data.components.splits import load_labels, make_inner_folds

    rows = []
    for p in range(9):
        for r in range(2):
            rows.append({"REGION_ID": f"P{p}_{r}", "STAS": p % 2, "PATIENT": f"P{p}"})
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    table = load_labels(
        tmp_path / "labels.csv",
        id_column="REGION_ID",
        label_column="STAS",
        patient_column="PATIENT",
    )
    ids = [r["REGION_ID"] for r in rows]
    folds_a = make_inner_folds(ids, table, k=3, seed=0)
    folds_b = make_inner_folds(list(reversed(ids)), table, k=3, seed=0)
    assert folds_a == folds_b  # order of the id set does not matter
    for tr, va in folds_a:
        assert not (set(tr) & set(va))
        assert not ({table.group_of(r) for r in tr} & {table.group_of(r) for r in va})
        assert len({table.label_of(r) for r in tr}) == 2


# ----------------------------------------------------------------------------
# probe folding
# ----------------------------------------------------------------------------


def test_probe_folding_reproduces_decision_function():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from grass_mil.models.training.probe import fold_probe_into_linear_head

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12)) * rng.uniform(0.5, 3.0, size=12) + rng.normal(size=12)
    y = (X[:, 0] - X[:, 3] > 0).astype(int)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    W, B = fold_probe_into_linear_head(
        clf.coef_, clf.intercept_, sc.mean_, sc.scale_, num_classes=2
    )
    assert torch.all(W[0] == 0) and B[0] == 0
    logits = torch.from_numpy(X).float() @ W.T + B
    ref = clf.decision_function(sc.transform(X))
    assert np.allclose((logits[:, 1] - logits[:, 0]).numpy(), ref, atol=1e-4)


# ----------------------------------------------------------------------------
# cohort fixture: precompute once per module
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort")
    c = make_synthetic_cohort(root, n_patients=8, regions_per_patient=2, n_cells=120)
    cfg = compose_cfg("train", cohort_overrides(c, root / "out_precompute"))
    import hydra

    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    return c


def test_precompute_records_patient_and_label(cohort):
    index = json.loads((cohort.processed_dir / "processed_index.json").read_text())
    entries = index["entries"]
    assert len(entries) == len(cohort.region_ids)
    for e in entries:
        assert e["patient_id"] == cohort.patient_of[e["region_id"]]
        assert int(round(e["graph_y"][0])) == cohort.label_of[e["region_id"]]
    data = torch.load(entries[0]["path"], weights_only=False)
    assert data.patient_id == cohort.patient_of[entries[0]["region_id"]]
    assert list(data.graph_label_names) == ["STAS"]


def test_explicit_split_repartitions_without_reprocessing(cohort, tmp_path):
    import hydra

    ids = cohort.region_ids
    cfg = compose_cfg(
        "train",
        cohort_overrides(cohort, tmp_path)
        + [
            f"data.split.explicit.train=[{','.join(ids[:10])}]",
            f"data.split.explicit.val=[{','.join(ids[10:13])}]",
            f"data.split.explicit.test=[{','.join(ids[13:15])}]",
        ],
    )
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    assert sorted(e.region_id for e in dm.dataset_train.entries) == sorted(ids[:10])
    assert sorted(e.region_id for e in dm.dataset_val.entries) == sorted(ids[10:13])
    assert sorted(e.region_id for e in dm.dataset_test.entries) == sorted(ids[13:15])
    cfg2 = compose_cfg(
        "train",
        cohort_overrides(cohort, tmp_path) + ["data.split.explicit.train=[nope]"],
    )
    dm2 = hydra.utils.instantiate(cfg2.data)
    with pytest.raises(ValueError, match="no processed graph"):
        dm2.setup("fit")


def test_probe_cli_writes_folded_head(cohort, tmp_path):
    from grass_mil.probe import run_probe

    ids = cohort.region_ids
    assert len(ids) == 16
    cfg = compose_cfg(
        "probe",
        cohort_overrides(cohort, tmp_path)
        + [
            "regime=linear_probehead",
            "task.head_init_path=null",
            f"data.split.explicit.train=[{','.join(ids[:10])}]",
            f"data.split.explicit.val=[{','.join(ids[10:])}]",
            f"data.split.explicit.test=[{','.join(ids[10:])}]",
            "probe.roots_per_unit=16",
            f"probe.output_path={tmp_path}/probe_head.pt",
        ],
    )
    summary = run_probe(cfg)
    head = torch.load(tmp_path / "probe_head.pt", weights_only=False)
    W = head["graph_head.net.0.weight"]
    assert W.shape == (2, 16) and torch.all(W[0] == 0)
    assert "val_region_auroc" in summary["metrics"]
    assert summary["settings"]["roots_per_unit"] == 16
    assert summary["settings"]["n_train_regions"] == 10
    assert (tmp_path / "probe_head.json").exists()


# ----------------------------------------------------------------------------
# nested CV
# ----------------------------------------------------------------------------


def test_best_epoch_from_jsonl(tmp_path):
    from grass_mil.nested_cv import best_epoch_from_jsonl

    p = tmp_path / "m.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"epoch": 0, "val/auroc": 0.5},
                {"epoch": 1, "val/auroc": 0.7},
                {"epoch": 2, "val/auroc": 0.7},
                {"epoch": 3, "val/auroc": 0.6},
            ]
        )
    )
    assert best_epoch_from_jsonl(p, metric="val/auroc") == (
        2,
        "val/auroc",
    )  # earliest tie, 1-based
    assert best_epoch_from_jsonl(p, metric="val/loss", mode="min") == (4, "last")
    assert best_epoch_from_jsonl(p, metric="val/x", fallback_metric="val/auroc") == (
        1,
        "val/auroc",
    )


@pytest.mark.parametrize("regime", ["mil", "linear_probehead"])
def test_nested_cv_smoke(cohort, tmp_path, regime):
    from grass_mil.nested_cv import run_nested_cv

    out = tmp_path / regime
    cfg = compose_cfg(
        "nested_cv",
        cohort_overrides(cohort, out)
        + [
            f"regime={regime}",
            f"nested_cv.folds_file={cohort.folds_path}",
            "nested_cv.inner_k=2",
            "nested_cv.outer_folds=[0,1]",
            "nested_cv.probe.roots_per_unit=16",
            "trainer.max_epochs=2",
        ],
    )
    summary = run_nested_cv(cfg)
    assert summary["n_outer"] == 2 and not summary["complete"]
    assert len(summary["auc_folds"]) == 2
    assert all(np.isfinite(a) for a in summary["auc_folds"])
    assert summary["probe"] == (regime == "linear_probehead")
    oof = pd.read_csv(out / "oof_predictions.csv")
    expected = sorted(cohort.folds[0] + cohort.folds[1])
    assert sorted(oof["region_id"]) == expected  # one OOF prediction per held-out region
    assert set(oof["patient_id"]) == {cohort.patient_of[r] for r in expected}
    pat = pd.read_csv(out / "oof_patient_predictions.csv")
    assert len(pat) == len({cohort.patient_of[r] for r in expected})
    for i in (0, 1):
        res = json.loads((out / f"outer_{i}" / "result.json").read_text())
        assert 1 <= res["E"] <= 2 and len(res["inner_epochs"]) == 2
        assert (out / f"outer_{i}" / "refit" / "test_predictions.csv").exists()
        if regime == "linear_probehead":
            assert res["head_init_path"] and Path(res["head_init_path"]).exists()
    # resume: a second call reuses every finished run
    summary2 = run_nested_cv(cfg)
    assert summary2["auc_folds"] == summary["auc_folds"]
