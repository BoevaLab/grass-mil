"""Phase 5: cell cache extraction, patient aggregation, evaluation statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort

pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------


def test_auroc_matches_sklearn_and_delong_identity_and_symmetry():
    from sklearn.metrics import roc_auc_score

    from grass_mil.evaluation import auroc, delong_test

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    s1 = y + rng.normal(0, 0.5, 80)
    s2 = rng.normal(size=80)
    assert auroc(y, s1) == pytest.approx(roc_auc_score(y, s1))
    same = delong_test(y, s1, s1)
    assert same.delta == 0.0 and same.p_value == pytest.approx(1.0)
    d12 = delong_test(y, s1, s2)
    d21 = delong_test(y, s2, s1)
    assert d12.delta == pytest.approx(-d21.delta) and d12.p_value == pytest.approx(d21.p_value)
    assert d12.auroc_1 == pytest.approx(roc_auc_score(y, s1))
    assert d12.p_value < 0.05  # informative vs random score on 80 units


def test_bootstrap_ci_brackets_point_and_group_bootstrap_runs():
    from grass_mil.evaluation import bootstrap_auroc_ci

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    s = y + rng.normal(0, 1, 60)
    ci = bootstrap_auroc_ci(y, s, n_boot=300, seed=0)
    assert ci.lower <= ci.auroc <= ci.upper and ci.n_valid > 0
    groups = np.repeat(np.arange(30), 2)
    ci_g = bootstrap_auroc_ci(y, s, n_boot=300, seed=0, groups=groups)
    assert ci_g.lower <= ci_g.auroc <= ci_g.upper


def test_within_fold_standardisation_and_per_fold_auroc():
    from grass_mil.evaluation import per_fold_auroc, rank_within_fold, z_within_fold

    folds = np.array([0] * 5 + [1] * 5)
    s = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
    z = z_within_fold(s, folds)
    assert z[:5].mean() == pytest.approx(0) and z[5:].std() == pytest.approx(1)
    assert np.allclose(z[:5], z[5:])  # scale-free
    r = rank_within_fold(s, folds)
    assert np.allclose(r[:5], (np.arange(5) + 0.5) / 5)
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    pf = per_fold_auroc(y, s, folds)
    assert pf["n_folds"] == 2 and 0 <= pf["mean"] <= 1 and "fold_0" in pf


def test_operating_points_from_confusion_and_derived_quantities():
    from grass_mil.evaluation import (
        confusion_counts,
        flag_top_within_fold,
        likelihood_ratios,
        metrics_from_confusion,
        ppv_npv_at_prevalence,
        threshold_points,
    )

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    c = confusion_counts(pred, y)
    assert (c.tp, c.fp, c.fn, c.tn) == (3, 2, 1, 4)
    m = metrics_from_confusion(c)
    assert m["sensitivity"] == pytest.approx(0.75) and m["specificity"] == pytest.approx(4 / 6)
    assert m["ppv"] == pytest.approx(0.6) and m["npv"] == pytest.approx(0.8)
    lr = likelihood_ratios(0.75, 4 / 6)
    assert lr["lr_plus"] == pytest.approx(0.75 / (1 - 4 / 6))
    r = ppv_npv_at_prevalence(0.75, 4 / 6, 0.4)
    assert r["ppv"] == pytest.approx(0.75 * 0.4 / (0.75 * 0.4 + (1 / 3) * 0.6))
    folds = np.array([0] * 5 + [1] * 5)
    score = np.arange(10, dtype=float)
    flagged = flag_top_within_fold(score, folds, 0.4)
    assert flagged.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]  # top 40 % of EACH fold
    t = threshold_points(score, y)
    assert set(t) == {"youden", "rule_out", "rule_in"}


# ----------------------------------------------------------------------------
# patient aggregation
# ----------------------------------------------------------------------------


def _payload():
    from grass_mil.inference.schemas import BatchPredictionPayload

    torch.manual_seed(0)
    regions = ["P1_a", "P1_b", "P2_a"]
    counts = [3, 2, 4]
    inst_logits = torch.randn(sum(counts), 2)
    inst_attn = torch.randn(sum(counts), 2)
    inst_region = [r for r, n in zip(regions, counts) for _ in range(n)]
    bag_logits = torch.stack(
        [
            (
                inst_logits[[i for i, rr in enumerate(inst_region) if rr == r]]
                * torch.softmax(
                    inst_attn[[i for i, rr in enumerate(inst_region) if rr == r]], dim=0
                )
            ).sum(0)
            for r in regions
        ]
    )
    return (
        BatchPredictionPayload(
            bag_ids=regions,
            bag_logits=bag_logits,
            bag_targets=torch.tensor([[1.0], [1.0], [0.0]]),
            bag_attention=None,
            row_region_ids=regions,
            row_sample_ids=regions,
            instance_logits=inst_logits,
            instance_attention_logits=inst_attn,
            instance_patch_ids=[f"p{i}" for i in range(sum(counts))],
            instance_bag_ids=inst_region,
            instance_region_ids=inst_region,
            instance_sample_ids=inst_region,
        ),
        inst_logits,
        inst_attn,
        inst_region,
    )


def test_patient_mean_probability_and_merged_bag_aggregation():
    from grass_mil.inference.aggregation import aggregate_group_logits

    payload, inst_logits, inst_attn, inst_region = _payload()
    gmap = {"P1_a": "P1", "P1_b": "P1", "P2_a": "P2"}
    mean_p = aggregate_group_logits(
        payload, mode="mean_probability", bag_scope="patient", group_map=gmap
    )
    assert mean_p.bag_ids == ["P1", "P2"]
    probs = torch.softmax(mean_p.bag_logits, dim=1)
    expected = torch.softmax(payload.bag_logits[:2], dim=1).mean(0)
    assert torch.allclose(probs[0], expected, atol=1e-6)
    assert mean_p.bag_targets[0].item() == 1.0 and mean_p.bag_targets[1].item() == 0.0

    merged = aggregate_group_logits(
        payload, mode="attention_weighted", bag_scope="patient", group_map=gmap
    )
    idx = [i for i, r in enumerate(inst_region) if gmap[r] == "P1"]
    direct = (inst_logits[idx] * torch.softmax(inst_attn[idx], dim=0)).sum(0)
    assert torch.allclose(merged.bag_logits[0], direct, atol=1e-6)
    with pytest.raises(ValueError, match="patient"):
        aggregate_group_logits(payload, mode="mean_probability", bag_scope="patient")


# ----------------------------------------------------------------------------
# extraction + scores on the synthetic cohort
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort5")
    c = make_synthetic_cohort(root, n_patients=4, regions_per_patient=2, n_cells=80)
    import hydra

    cfg = compose_cfg("train", cohort_overrides(c, root / "out_precompute"))
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    return c


def test_extract_cell_cache_covers_every_cell(cohort, tmp_path):
    from grass_mil.extract import run_extract
    from grass_mil.inference.cell_cache import CellCache
    from grass_mil.train import train

    train_out = tmp_path / "train"
    cfg = compose_cfg("train", cohort_overrides(cohort, train_out) + ["trainer.max_epochs=1"])
    train(cfg)
    ckpt = train_out / "checkpoints" / "last.ckpt"
    assert ckpt.exists()

    ext_out = tmp_path / "extract"
    ecfg = compose_cfg(
        "extract",
        cohort_overrides(cohort, ext_out)
        + [f"+extract.models.final={ckpt}", "extract.batch_size=32", "extract.device=cpu"],
    )
    result = run_extract(ecfg)
    cache = CellCache.open(Path(result["models"][0]))
    assert sorted(cache.regions) == sorted(cohort.region_ids)
    assert len(cache.bias) == 2 and set(cache.cell_type_names) == {"A", "B", "C", "D"}
    rid = cache.regions[0]
    arrays = cache.load(rid)
    n = cache.n_cells(rid)
    assert arrays["E"].shape == (n, 16) and arrays["E"].dtype == np.float16
    assert arrays["LG"].shape == (n, 2) and arrays["AR"].shape == (n, 2)
    assert arrays["ct"].shape == (n,) and arrays["pos"].shape == (n, 2)
    assert arrays["edge_index"].shape[0] == 2 and arrays["edge_index"].max() < n
    assert np.isfinite(arrays["E"].astype(np.float32)).all()
    assert cache.label_of(rid) == float(cohort.label_of[rid])
    # row order = cell order of the region table (first cell's type matches the CSV)
    csv = pd.read_csv(cohort.manifest_path).set_index("region_id").loc[rid, "input_path"]
    types = pd.read_csv(csv)["cell_type"].tolist()
    assert [cache.cell_type_names[c] for c in arrays["ct"]] == types


def test_scores_cli_on_synthetic_oof_table(tmp_path):
    from grass_mil.evaluation.cli import run_scores

    rng = np.random.default_rng(0)
    rows = []
    for outer in range(3):
        for p in range(20):
            pid = f"o{outer}_P{p}"
            label = int(p % 2 == 0)
            for core in range(2):
                rows.append(
                    {
                        "outer": outer,
                        "region_id": f"{pid}_c{core}",
                        "patient_id": pid,
                        "label": label,
                        "prob_class_1": 0.3 + 0.3 * label + rng.normal(0, 0.15) + 0.1 * outer,
                        "other": rng.uniform(),
                        "subtype": "A" if p < 10 else "B",
                    }
                )
    table = tmp_path / "oof.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    cfg = compose_cfg(
        "scores",
        [
            f"input={table}",
            f"output_dir={tmp_path / 'scores'}",
            "scores=[prob_class_1,other]",
            "strata=[subtype]",
            "n_boot=200",
        ],
    )
    summary = run_scores(cfg)
    assert summary["n"] == 60  # patient level
    first = summary["scores"][0]
    assert first["auroc"] > 0.8 and first["lower"] <= first["auroc"] <= first["upper"]
    assert first["perfold_n_folds"] == 3
    assert summary["delong"][0]["p_value"] < 0.05
    out = tmp_path / "scores"
    for name in (
        "scores_summary.csv",
        "delong.csv",
        "operating_points.csv",
        "strata.csv",
        "ppv_npv_by_prevalence.csv",
        "report_scores.md",
    ):
        assert (out / name).exists(), name
    ops = pd.read_csv(out / "operating_points.csv")
    top30 = ops[(ops.score == "prob_class_1") & (ops.rule.str.startswith("top 30"))].iloc[0]
    assert abs(top30["flagged"] - 0.3) < 0.06
