from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from grass_mil.interpretability.contracts import ClusteringResult
from grass_mil.interpretability.core.data import load_interpretability_dataset
from grass_mil.interpretability.pipeline import run_interpretability_pipeline
from grass_mil.interpretability.report_cli import run_report

pytest.importorskip("plotly.graph_objects")

from grass_mil.interpretability.reporting.plotly_builders import bundle_figures
from grass_mil.interpretability.reporting.render import render_interpretability_report


def _write_minimal_tables(tmp_path: Path) -> tuple[Path, Path]:
    node = pd.DataFrame(
        {
            "instance_id": [f"i{i}" for i in range(8)],
            "bag_id": ["b0"] * 4 + ["b1"] * 4,
            "cell_type": ["A", "A", "B", "B", "A", "B", "B", "A"],
            "comp_A": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "comp_B": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "score": [0.1, 0.2, 0.8, 0.7, 0.2, 0.9, 0.5, 0.3],
            "attention": [0.2, 0.3, 0.2, 0.3, 0.1, 0.5, 0.2, 0.2],
            "sample_id": ["sx"] * 4 + ["sy"] * 4,
            "condition": ["X"] * 4 + ["Y"] * 4,
            "center_x": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
            "center_y": [0.0, 0.2, 0.1, 0.3, 1.0, 1.2, 1.1, 1.3],
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
        "grass_mil.interpretability.reporting.render.render_pdf_via_playwright",
        lambda html_path, pdf_path: pdf_path,  # type: ignore[lambda-assign]
    )
    monkeypatch.setattr(
        "grass_mil.interpretability.reporting.render.export_plotly_snapshots",
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


def test_report_render_tolerates_snapshot_export_failure_when_pdf_disabled(
    monkeypatch, tmp_path: Path
) -> None:
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
            "enabled": ["cluster_profiles"],
            "params": {},
        },
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")

    def _raise_snapshot_error(figures, output_dir, scale=2.0):  # type: ignore[unused-argument]
        raise RuntimeError("kaleido unavailable")

    monkeypatch.setattr(
        "grass_mil.interpretability.reporting.render.export_plotly_snapshots",
        _raise_snapshot_error,
    )
    out = render_interpretability_report(
        bundle,
        output_dir=tmp_path / "report",
        html_enabled=True,
        pdf_enabled=False,
    )
    assert out["html_path"] is not None
    assert out["pdf_path"] is None
    assert out["snapshot_paths"] == {}


def test_pipeline_and_report_render_with_diff_neighborhood_plugin(
    monkeypatch, tmp_path: Path
) -> None:
    instance_path, spatial_path = _write_minimal_tables(tmp_path)
    dataset = load_interpretability_dataset(
        instance_table_path=instance_path,
        spatial_table_path=spatial_path,
        id_column="instance_id",
        bag_id_column="bag_id",
        cell_type_column="cell_type",
        condition_column="condition",
    )
    cfg = {
        "reduction": {"enabled": True, "method": "pca", "params": {"n_components": 2}},
        "clustering": {
            "enabled": True,
            "method": "agglomerative",
            "params": {"n_clusters": 2, "linkage": "ward"},
        },
        "plugins": {
            "enabled": ["diff_neighborhood_enrichment"],
            "params": {
                "diff_neighborhood_enrichment": {
                    "condition_column": "condition",
                    "permutation_group_column": "sample_id",
                    "n_perms": 0,
                    "undirected": True,
                }
            },
        },
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")
    assert "diff_neighborhood_enrichment" in bundle.plugin_results
    figs = dict(bundle_figures(bundle))
    assert "diff_neighborhood_enrichment_X_Y" in figs

    monkeypatch.setattr(
        "grass_mil.interpretability.reporting.render.export_plotly_snapshots",
        lambda figures, output_dir, scale=2.0: {},  # type: ignore[lambda-assign]
    )
    out = render_interpretability_report(
        bundle,
        output_dir=tmp_path / "report_diff",
        html_enabled=True,
        pdf_enabled=False,
    )
    assert out["html_path"] is not None


def test_pipeline_and_report_render_with_tissue_graph_plugin(monkeypatch, tmp_path: Path) -> None:
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
            "enabled": ["tissue_graph"],
            "params": {
                "tissue_graph": {
                    "sample_column": "sample_id",
                    "sample_value": "sx",
                    "id_column": "instance_id",
                }
            },
        },
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")
    assert "tissue_graph" in bundle.plugin_results
    figs = dict(bundle_figures(bundle))
    assert "tissue_graph_sample_id_sx" in figs

    monkeypatch.setattr(
        "grass_mil.interpretability.reporting.render.export_plotly_snapshots",
        lambda figures, output_dir, scale=2.0: {},  # type: ignore[lambda-assign]
    )
    out = render_interpretability_report(
        bundle,
        output_dir=tmp_path / "report_tissue_graph",
        html_enabled=True,
        pdf_enabled=False,
    )
    assert out["html_path"] is not None


def test_pipeline_fails_fast_when_plugin_required_inputs_missing_cluster_labels(
    tmp_path: Path,
) -> None:
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
        "clustering": {"enabled": False},
        "plugins": {"enabled": ["cluster_profiles"], "params": {}},
    }
    with pytest.raises(ValueError, match="missing required inputs: cluster_labels"):
        run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")


def test_pipeline_fails_fast_when_plugin_required_inputs_missing_spatial_table(
    tmp_path: Path,
) -> None:
    instance_path, _ = _write_minimal_tables(tmp_path)
    dataset = load_interpretability_dataset(
        instance_table_path=instance_path,
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
        "plugins": {"enabled": ["neighborhood_enrichment"], "params": {}},
    }
    with pytest.raises(ValueError, match="missing required inputs: spatial_table"):
        run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")


def test_pipeline_clusters_on_raw_embeddings_when_cluster_on_pca_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    instance_path, _ = _write_minimal_tables(tmp_path)
    dataset = load_interpretability_dataset(
        instance_table_path=instance_path,
        id_column="instance_id",
        bag_id_column="bag_id",
        cell_type_column="cell_type",
    )
    expected_raw = dataset.instance_table[["inst_emb_0", "inst_emb_1"]].to_numpy(dtype=float)
    captured: dict[str, np.ndarray] = {}

    def _capture_run_clustering(method, x, **params):  # type: ignore[no-untyped-def]
        captured["x"] = np.asarray(x)
        return ClusteringResult(
            method=str(method),
            labels=np.zeros((int(x.shape[0]),), dtype=int),
            params=dict(params),
            fitted_object=None,
        )

    monkeypatch.setattr(
        "grass_mil.interpretability.pipeline.run_clustering", _capture_run_clustering
    )
    cfg = {
        "reduction": {"enabled": True, "method": "pca", "params": {"n_components": 1}},
        "clustering": {
            "enabled": True,
            "method": "kmeans",
            "cluster_on_pca": False,
            "params": {"n_clusters": 2, "random_state": 42},
        },
        "plugins": {"enabled": [], "params": {}},
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")
    assert "x" in captured
    assert captured["x"].shape == expected_raw.shape
    assert np.allclose(captured["x"], expected_raw)
    assert bundle.cluster_feature_reduction is None


def test_pipeline_exposes_cluster_feature_reduction_when_cluster_on_pca_enabled(
    tmp_path: Path,
) -> None:
    instance_path, _ = _write_minimal_tables(tmp_path)
    dataset = load_interpretability_dataset(
        instance_table_path=instance_path,
        id_column="instance_id",
        bag_id_column="bag_id",
        cell_type_column="cell_type",
    )
    cfg = {
        "reduction": {"enabled": True, "method": "pca", "params": {"n_components": 2}},
        "clustering": {
            "enabled": True,
            "method": "agglomerative",
            "cluster_on_pca": True,
            "pca_components": 1,
            "params": {"n_clusters": 2, "linkage": "ward"},
        },
        "plugins": {"enabled": [], "params": {}},
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")
    assert bundle.cluster_feature_reduction is not None
    assert bundle.cluster_feature_reduction.method == "pca"
    assert bundle.cluster_feature_reduction.embedding.shape[1] == 1


def test_report_cli_smoke(
    monkeypatch,
    cfg_interpret: DictConfig,
    tmp_path: Path,  # type: ignore[valid-type]
) -> None:
    instance_path, spatial_path = _write_minimal_tables(tmp_path)
    monkeypatch.setattr(
        "grass_mil.interpretability.reporting.render.render_pdf_via_playwright",
        lambda html_path, pdf_path: pdf_path,  # type: ignore[lambda-assign]
    )
    monkeypatch.setattr(
        "grass_mil.interpretability.reporting.render.export_plotly_snapshots",
        lambda figures, output_dir, scale=2.0: {},  # type: ignore[lambda-assign]
    )
    with open_dict(cfg_interpret):
        cfg_interpret.data.instance_table = str(instance_path)
        cfg_interpret.data.spatial_table = str(spatial_path)
        # Keep smoke test independent of optional umap-learn dependency.
        cfg_interpret.reduction.method = "pca"
        cfg_interpret.reduction.params = {"n_components": 2, "random_state": 42}
        cfg_interpret.plugins.enabled = [
            "cluster_profiles",
            "attention_attribution",
            "neighborhood_enrichment",
            "diff_neighborhood_enrichment",
            "filtration_curves",
        ]
        cfg_interpret.clustering.method = "agglomerative"
        cfg_interpret.clustering.cluster_on_pca = False
        cfg_interpret.clustering.pca_components = None
        cfg_interpret.clustering.params = {"n_clusters": 2, "linkage": "ward"}
        cfg_interpret.report.pdf.enabled = True
    HydraConfig().set_config(cfg_interpret)
    summary, _ = run_report(cfg_interpret)
    assert Path(summary["output_dir"]).exists()
    assert Path(summary["output_dir"], "report_summary.json").exists()


def test_pdf_renderer_raises_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    from grass_mil.interpretability.reporting.pdf import render_pdf_via_playwright

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


def test_resolve_color_values_handles_numeric_and_categorical_columns() -> None:
    from grass_mil.interpretability.reporting.plotly_builders import _resolve_color_values

    numeric_values, numeric_ticks = _resolve_color_values(pd.Series([0.5, 1.5, 2.5]))
    np.testing.assert_allclose(numeric_values, [0.5, 1.5, 2.5])
    assert numeric_ticks == {}

    # A string column (``condition`` routinely is one) must be factorized rather
    # than raising, with the original labels preserved as colorbar ticks.
    cat_values, cat_ticks = _resolve_color_values(pd.Series(["X", "X", "Y"]))
    np.testing.assert_allclose(cat_values, [0.0, 0.0, 1.0])
    assert cat_ticks["ticktext"] == ["X", "Y"]
    assert cat_ticks["tickvals"] == [0, 1]


def test_multi_attribute_scatters_plot_string_condition_column(tmp_path: Path) -> None:
    node_path, spatial_path = _write_minimal_tables(tmp_path)
    dataset = load_interpretability_dataset(
        instance_table_path=node_path,
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
        "plugins": {"enabled": []},
    }
    bundle = run_interpretability_pipeline(dataset, cfg, artifacts_dir=tmp_path / "artifacts")

    figs = dict(bundle_figures(bundle))
    condition_figs = [name for name in figs if name.endswith("_condition")]
    assert condition_figs, f"expected a condition-colored scatter, got {sorted(figs)}"

    marker = figs[condition_figs[0]].data[0].marker
    assert list(marker.colorbar.ticktext) == ["X", "Y"]
