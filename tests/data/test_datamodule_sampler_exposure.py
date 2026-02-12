from pathlib import Path

from src.data.spatial_omics_datamodule import SpatialOmicsDataModule


def _minimal_datamodule(tmp_path: Path, sampler: dict | None) -> SpatialOmicsDataModule:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return SpatialOmicsDataModule(
        data_dir=str(data_dir),
        raw_manifest_path=str(data_dir / "raw" / "manifest.csv"),
        processed_dir=str(data_dir / "processed"),
        coord_scale_um=1.0,
        sample_unit="tile",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        reducer_scope="sample",
        keep_raw_molecular=False,
        force_precompute=False,
        min_cells=1,
        use_molecular_features=False,
        categorical_features={"include_labels": []},
        split={"train_val_test_split": [1.0, 0.0, 0.0], "split_by": "patch"},
        csv={
            "sep": ",",
            "coord_columns": ["x", "y"],
            "cell_id_column": "cell_id",
            "categorical_label_columns": [],
            "molecular_columns": None,
        },
        h5ad={
            "coord_columns": ["x", "y"],
            "cell_id_column": None,
            "categorical_label_columns": [],
            "molecular_layer": None,
        },
        sce={
            "assay_name": None,
            "coord_source": "colData",
            "coord_key": None,
            "coord_columns": ["x", "y"],
            "cell_id_column": None,
            "categorical_label_columns": [],
            "transpose_assay": True,
        },
        graph_builder={
            "name": "delaunay",
            "kwargs": {"edge_features": ["distance"], "neighbor_cutoff_um": 50.0},
        },
        feature_reducer={"name": "identity", "kwargs": {}},
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
        sampler=sampler,
        manifest={
            "sample_id": "sample_id",
            "input_path": "input_path",
            "input_type": "input_type",
            "region_id": "region_id",
            "polygons_path": "polygons_path",
        },
        graph_labels=None,
        transforms=[],
    )


def test_datamodule_sampler_defaults_to_identity(tmp_path: Path):
    dm = _minimal_datamodule(tmp_path, sampler=None)
    strategy = dm._ensure_sampler_strategy()
    assert strategy.__class__.__name__ == "IdentityBatchStrategy"
    assert dm.sampler_strategy is strategy


def test_datamodule_exposes_configured_sampler_strategy(tmp_path: Path):
    dm = _minimal_datamodule(
        tmp_path,
        sampler={
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": {
                "enabled": True,
                "depth": 2,
                "num_neighbors": 8,
                "subgraph_batch_size": 4,
                "replace": False,
                "shuffle_subgraphs": True,
            },
        },
    )
    strategy = dm._ensure_sampler_strategy()
    assert strategy.__class__.__name__ == "ShadowCustomStrategy"
    assert dm.sampler_strategy is strategy
    assert strategy.runtime.enabled is True
    assert strategy.runtime.depth == 2
    assert strategy.runtime.num_neighbors == 8
