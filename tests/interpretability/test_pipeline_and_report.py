from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.interpretability.core.data import load_interpretability_dataset
from src.interpretability.pipeline import run_interpretability_pipeline
from src.interpretability.report_cli import run_report
from src.interpretability.reporting.plotly_builders import bundle_figures
from src.interpretability.reporting.render import render_interpretability_report


def _write_minimal_tables(tmp_path: Path) -> tuple[Path, Path]:
    node = pd.DataFrame(
        {
            "instance_id": [f"i{i}" for i in range(8)],
            "bag_id": ["b0"] * 4 + ["b1"] * 4,
            "cell_type": ["A", "A", "B", "B", "A", "B", "B", "A"],
            "score": [0.1, 0.2, 0.8, 0.7, 0.2, 0.9, 0.5, 0.3],
            "attention": [0.2, 0.3, 0.2, 0.3, 0.1, 0.5, 0.2, 0.2],
            "inst_emb_0": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
            "inst_emb_1": [0.0, 0.2, 0.1, 0.3, 1.0, 1.2, 1.1, 1.3],
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": ["i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7"],
            "target_id": ["i1", "i2", "i3", "i0", "i5", "i6", "i7", "i4"],
            "distance": [0.1, 0.2, 0.4, 0.3, 0.7, 0.8, 0.9, 0.6],
            "weight": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    instance_path = tmp_path / "instance_table.csv"
    spatial_path = tmp_path / "spatial_table.csv"
    node.to_csv(instance_path, index=False)
    edges.to_csv(spatial_path, index=False)
    return instance_path, spatial_path


def test_pipeline_and_report_render(monkeypatch, tmp_path: Path) -> None:
    instance_path, spatial_path = _write_minimal_tables(tmp_path)
    dataset = load_interpretability_dataset(
        instance_table_path=instance_path,
        spatial_table_path=spatial_path,
        id_column="instance_id",
        bag_id_column="bag_id",
        cell_type_column="cell_type",
    )
    cfg = {
        "reduction": {"enabled": True, "method": "pca", "params": {"n_components": 2}},
        "clustering": {
            "enabled": True,
            "method": "agglomerative",
            "params": {"n_clusters": 2, "linkage": "ward"},
        },
        "plugins": {
            "enabled": ["cluster_profiles", "attention_attribution"],
            "params": {
                "attention_attribution": {"attention_column": "attention", "score_column": "score"}
            },
        },
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")
    figs = bundle_figures(bundle)
    assert len(figs) >= 2

    monkeypatch.setattr(
        "src.interpretability.reporting.render.render_pdf_via_playwright",
        lambda html_path, pdf_path: pdf_path,  # type: ignore[lambda-assign]
    )
    monkeypatch.setattr(
        "src.interpretability.reporting.render.export_plotly_snapshots",
        lambda figures, output_dir, scale=2.0: {},  # type: ignore[lambda-assign]
    )
    out = render_interpretability_report(
        bundle,
        output_dir=tmp_path / "report",
        html_enabled=True,
        pdf_enabled=True,
    )
    assert out["html_path"] is not None
    assert out["pdf_path"] is not None


def test_report_cli_smoke(
    monkeypatch,
    cfg_interpret: DictConfig,
    tmp_path: Path,  # type: ignore[valid-type]
) -> None:
    instance_path, spatial_path = _write_minimal_tables(tmp_path)
    monkeypatch.setattr(
        "src.interpretability.reporting.render.render_pdf_via_playwright",
        lambda html_path, pdf_path: pdf_path,  # type: ignore[lambda-assign]
    )
    monkeypatch.setattr(
        "src.interpretability.reporting.render.export_plotly_snapshots",
        lambda figures, output_dir, scale=2.0: {},  # type: ignore[lambda-assign]
    )
    with open_dict(cfg_interpret):
        cfg_interpret.data.instance_table = str(instance_path)
        cfg_interpret.data.spatial_table = str(spatial_path)
        cfg_interpret.plugins.enabled = [
            "cluster_profiles",
            "attention_attribution",
            "neighborhood_enrichment",
            "filtration_curves",
        ]
        cfg_interpret.clustering.method = "agglomerative"
        cfg_interpret.clustering.params.n_clusters = 2
        cfg_interpret.clustering.params.linkage = "ward"
        cfg_interpret.report.pdf.enabled = True
    HydraConfig().set_config(cfg_interpret)
    summary, _ = run_report(cfg_interpret)
    assert Path(summary["output_dir"]).exists()
    assert Path(summary["output_dir"], "report_summary.json").exists()


def test_pdf_renderer_raises_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    from src.interpretability.reporting.pdf import render_pdf_via_playwright

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("missing playwright")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    with pytest.raises(RuntimeError, match="Playwright is unavailable"):
        render_pdf_via_playwright(
            html_path=tmp_path / "foo.html",
            pdf_path=tmp_path / "foo.pdf",
        )
