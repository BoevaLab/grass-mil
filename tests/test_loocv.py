from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.loocv import run_loocv


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

    monkeypatch.setattr("src.loocv.train", _fake_train)
    summary = run_loocv(cfg)
    summary_path = Path(cfg.paths.output_dir) / "summary.json"

    assert summary_path.exists()
    assert len(summary["folds"]) == 2
    assert "aggregate_metrics" in summary
    GlobalHydra.instance().clear()
