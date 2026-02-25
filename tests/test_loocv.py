from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.loocv import _fold_slug, _resolve_selected_folds, _to_float_metrics, run_loocv


def _make_loocv_cfg(tmp_path: Path) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="loocv.yaml")
    with open_dict(cfg):
        cfg.paths.root_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path / "outputs")
        cfg.data.processed_dir = str(tmp_path / "processed")
        cfg.data.raw_manifest_path = str(tmp_path / "manifest.csv")
        cfg.data.split.loocv.enabled = True
        cfg.data.split.loocv.fold_unit = "sample"
        cfg.data.split.loocv.holdout_id = "s0"
        cfg.data.split.loocv.validation_strategy = "heldout_fold_items"
    return cfg


def _write_manifest(path: Path) -> None:
    path.write_text(
        "sample_id,input_path,input_type,region_id\n"
        "s0,/tmp/f0.csv,csv,r0\n"
        "s1,/tmp/f1.csv,csv,r1\n"
    )


def _write_manifest_for_sample_ids(path: Path, sample_ids: list[str]) -> None:
    rows = ["sample_id,input_path,input_type,region_id"]
    rows.extend(
        f"{sample_id},/tmp/{idx}.csv,csv,r{idx}" for idx, sample_id in enumerate(sample_ids)
    )
    path.write_text("\n".join(rows) + "\n")


def test_fold_slug_is_collision_safe_for_sanitized_aliases() -> None:
    fold_a = "sampleA::region1"
    fold_b = "sampleA//region1"

    slug_a = _fold_slug(fold_a)
    slug_b = _fold_slug(fold_b)

    assert slug_a.startswith("sampleA_region1_")
    assert slug_b.startswith("sampleA_region1_")
    assert slug_a != slug_b
    assert slug_a == _fold_slug(fold_a)


@pytest.mark.parametrize("mode", ["single", "all"])
def test_loocv_rejects_unsafe_isolation_flags(tmp_path: Path, mode: str) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    _write_manifest(Path(cfg.data.raw_manifest_path))
    with open_dict(cfg):
        cfg.loocv.mode = mode
        cfg.loocv.per_fold_processed_dir = False
        cfg.loocv.force_precompute_per_fold = False
        cfg.loocv.fold_index = 0

    with pytest.raises(ValueError, match="Unsafe LOOCV configuration"):
        run_loocv(cfg)
    GlobalHydra.instance().clear()


def test_loocv_rejects_unsafe_isolation_flags_when_base_loocv_disabled(
    tmp_path: Path,
) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    _write_manifest(Path(cfg.data.raw_manifest_path))
    with open_dict(cfg):
        cfg.data.split.loocv.enabled = False
        cfg.loocv.mode = "single"
        cfg.loocv.per_fold_processed_dir = False
        cfg.loocv.force_precompute_per_fold = False
        cfg.loocv.fold_index = 0

    with pytest.raises(ValueError, match="Unsafe LOOCV configuration"):
        run_loocv(cfg)
    GlobalHydra.instance().clear()


def test_loocv_runs_with_safe_isolation_flags(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    _write_manifest(Path(cfg.data.raw_manifest_path))
    with open_dict(cfg):
        cfg.loocv.mode = "all"
        cfg.loocv.per_fold_processed_dir = True
        cfg.loocv.force_precompute_per_fold = False
        cfg.loocv.summary_filename = "summary.json"

    def _fake_train(_cfg):
        return {"val/loss": 1.23}, {}

    monkeypatch.setattr("src.loocv._discover_viable_fold_ids", lambda _cfg: ["s0", "s1"])
    monkeypatch.setattr("src.loocv.train", _fake_train)
    summary = run_loocv(cfg)
    summary_path = Path(cfg.paths.output_dir) / "summary.json"

    assert summary_path.exists()
    assert len(summary["folds"]) == 2
    assert "aggregate_metrics" in summary
    GlobalHydra.instance().clear()


def test_loocv_uses_unique_dirs_for_colliding_sanitized_fold_ids(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    _write_manifest_for_sample_ids(
        Path(cfg.data.raw_manifest_path), ["sampleA::region1", "sampleA//region1"]
    )
    with open_dict(cfg):
        cfg.loocv.mode = "all"
        cfg.loocv.per_fold_processed_dir = True
        cfg.loocv.force_precompute_per_fold = False

    seen_output_dirs: list[str] = []
    seen_processed_dirs: list[str] = []

    def _fake_train(_cfg):
        seen_output_dirs.append(str(_cfg.paths.output_dir))
        seen_processed_dirs.append(str(_cfg.data.processed_dir))
        return {"val/loss": 0.5}, {}

    monkeypatch.setattr(
        "src.loocv._discover_viable_fold_ids",
        lambda _cfg: ["sampleA::region1", "sampleA//region1"],
    )
    monkeypatch.setattr("src.loocv.train", _fake_train)
    summary = run_loocv(cfg)

    assert len(summary["folds"]) == 2
    assert len(set(seen_output_dirs)) == 2
    assert len(set(seen_processed_dirs)) == 2
    GlobalHydra.instance().clear()


def test_loocv_fails_fast_when_selected_fold_not_viable(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    _write_manifest(Path(cfg.data.raw_manifest_path))
    with open_dict(cfg):
        cfg.loocv.mode = "all"
        cfg.loocv.per_fold_processed_dir = True
        cfg.loocv.force_precompute_per_fold = False

    monkeypatch.setattr("src.loocv._discover_viable_fold_ids", lambda _cfg: ["s0"])
    monkeypatch.setattr(
        "src.loocv.train",
        lambda _cfg: pytest.fail("train() should not be called when preflight fails"),
    )

    with pytest.raises(
        ValueError,
        match="LOOCV preflight failed: selected folds are not viable after preprocessing filters",
    ):
        run_loocv(cfg)
    GlobalHydra.instance().clear()


def test_loocv_preflight_passes_and_runs_all_selected_folds(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    _write_manifest_for_sample_ids(Path(cfg.data.raw_manifest_path), ["s0", "s1", "s2"])
    with open_dict(cfg):
        cfg.loocv.mode = "all"
        cfg.loocv.per_fold_processed_dir = True
        cfg.loocv.force_precompute_per_fold = False

    call_count = {"n": 0}

    def _fake_train(_cfg):
        call_count["n"] += 1
        return {"val/loss": 0.75}, {}

    monkeypatch.setattr("src.loocv._discover_viable_fold_ids", lambda _cfg: ["s0", "s1", "s2"])
    monkeypatch.setattr("src.loocv.train", _fake_train)

    summary = run_loocv(cfg)

    assert call_count["n"] == 3
    assert len(summary["folds"]) == 3
    GlobalHydra.instance().clear()


def test_single_mode_warns_when_both_holdout_id_and_fold_index_are_set(
    tmp_path: Path,
) -> None:
    cfg = _make_loocv_cfg(tmp_path)
    with open_dict(cfg):
        cfg.loocv.mode = "single"
        cfg.data.split.loocv.holdout_id = "s0"
        cfg.loocv.fold_index = 1

    discovered = [
        {"fold_id": "s0", "sample_id": "s0", "region_id": None},
        {"fold_id": "s1", "sample_id": "s1", "region_id": None},
    ]
    with pytest.warns(
        UserWarning,
        match="loocv.fold_index takes priority",
    ):
        selected = _resolve_selected_folds(cfg, discovered)

    assert selected == ["s1"]


def test_to_float_metrics_warns_and_skips_non_convertible_values() -> None:
    metrics = {
        "train/loss": 0.5,
        "val/ok": "1.25",
        "val/bad": {"not": "numeric"},
    }

    with pytest.warns(UserWarning, match="Skipping non-numeric LOOCV metric 'val/bad'"):
        converted = _to_float_metrics(metrics)

    assert converted == {"train/loss": 0.5, "val/ok": 1.25}
