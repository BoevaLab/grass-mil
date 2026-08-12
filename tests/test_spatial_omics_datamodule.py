import json
from pathlib import Path

import numpy as np
import pytest


def _write_polygon_wkt(path: Path, coords: np.ndarray) -> None:
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    polygon = Polygon(coords)
    path.write_text(polygon.wkt)


def test_spatial_omics_datamodule_precompute(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("scipy")

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ]
    )
    csv_path = raw_dir / "sample.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                np.arange(coords.shape[0]),
                coords[:, 0],
                coords[:, 1],
                np.random.rand(coords.shape[0]),
                np.random.rand(coords.shape[0]),
                ["A", "B", "A", "B"],
            ]
        ),
        delimiter=",",
        fmt="%s",
        header="cell_id,x,y,gene1,gene2,cell_type",
        comments="",
    )

    polygon_path = raw_dir / "polygons.wkt"
    _write_polygon_wkt(
        polygon_path,
        np.array([[-1, -1], [-1, 20], [20, 20], [20, -1], [-1, -1]]),
    )

    manifest_path = raw_dir / "manifest.csv"
    manifest_path.write_text(
        "sample_id,input_path,input_type,region_id,polygons_path\n"
        f"sample_1,{csv_path},csv,region_1,{polygon_path}\n"
    )

    graph_labels_path = raw_dir / "graph_labels.csv"
    graph_labels_path.write_text("region_id,outcome\nregion_1,1\n")

    from grass_mil.data.spatial_omics_datamodule import SpatialOmicsDataModule

    dm = SpatialOmicsDataModule(
        data_dir=str(data_dir),
        raw_manifest_path=str(manifest_path),
        processed_dir=str(data_dir / "processed"),
        coord_scale_um=1.0,
        sample_unit="tile",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        reducer_scope="sample",
        keep_raw_molecular=True,
        force_precompute=True,
        min_cells=1,
        use_molecular_features=True,
        categorical_features={
            "include_labels": ["cell_type"],
        },
        split={"train_val_test_split": [1.0, 0.0, 0.0], "split_by": "patch"},
        csv={
            "sep": ",",
            "coord_columns": ["x", "y"],
            "cell_id_column": "cell_id",
            "categorical_label_columns": ["cell_type"],
            "molecular_columns": ["gene1", "gene2"],
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
            "kwargs": {
                "edge_features": ["distance", "neighbor"],
                "neighbor_cutoff_um": 50.0,
            },
        },
        feature_reducer={"name": "identity", "kwargs": {}},
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
        manifest={
            "sample_id": "sample_id",
            "input_path": "input_path",
            "input_type": "input_type",
            "region_id": "region_id",
            "polygons_path": "polygons_path",
        },
        graph_labels={
            "label_file": str(graph_labels_path),
            "id_column": "region_id",
            "tasks": ["outcome"],
            "scope": "region",
        },
        transforms=[],
    )

    dm.prepare_data()
    dm.setup()

    dataset = dm.train_dataloader().dataset
    assert len(dataset) > 0

    metadata_path = data_dir / "processed" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["molecular_feature_dim"] == 2
    assert metadata["molecular_feature_names"] == ["gene1", "gene2"]
    assert metadata["categorical_labels"] == ["cell_type"]

    data = dataset[0]
    assert hasattr(data, "x")
    assert hasattr(data, "pos")
    assert data.patch_id.startswith("sample_1_region_1")
    assert hasattr(data, "edge_attr")
    assert hasattr(data, "graph_y")
    assert data.x.shape[1] == 2  # gene1, gene2
    assert hasattr(data, "categorical_codes")
    assert data.categorical_codes.shape[1] == 1
    assert data.categorical_slices["cell_type"] == 0

    index_payload = json.loads((data_dir / "processed" / "processed_index.json").read_text())
    assert all(
        entry.get("split") in {"train", "val", "test"} for entry in index_payload["entries"]
    )


def test_spatial_omics_datamodule_indices_no_molecular(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("scipy")

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    coords = np.array([[0.0, 0.0], [0.0, 10.0]])
    csv_path = raw_dir / "sample.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                np.arange(coords.shape[0]),
                coords[:, 0],
                coords[:, 1],
                ["A", "B"],
            ]
        ),
        delimiter=",",
        fmt="%s",
        header="cell_id,x,y,cell_type",
        comments="",
    )

    polygon_path = raw_dir / "polygons.wkt"
    _write_polygon_wkt(
        polygon_path,
        np.array([[-1, -1], [-1, 20], [20, 20], [20, -1], [-1, -1]]),
    )

    manifest_path = raw_dir / "manifest.csv"
    manifest_path.write_text(
        "sample_id,input_path,input_type,region_id,polygons_path\n"
        f"sample_1,{csv_path},csv,region_1,{polygon_path}\n"
    )

    from grass_mil.data.spatial_omics_datamodule import SpatialOmicsDataModule

    dm = SpatialOmicsDataModule(
        data_dir=str(data_dir),
        raw_manifest_path=str(manifest_path),
        processed_dir=str(data_dir / "processed"),
        coord_scale_um=1.0,
        sample_unit="tile",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        reducer_scope="sample",
        keep_raw_molecular=False,
        force_precompute=True,
        min_cells=1,
        use_molecular_features=False,
        categorical_features={
            "include_labels": ["cell_type"],
        },
        split={"train_val_test_split": [1.0, 0.0, 0.0], "split_by": "patch"},
        csv={
            "sep": ",",
            "coord_columns": ["x", "y"],
            "cell_id_column": "cell_id",
            "categorical_label_columns": ["cell_type"],
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
            "kwargs": {
                "edge_features": ["distance", "neighbor"],
                "neighbor_cutoff_um": 50.0,
            },
        },
        feature_reducer={"name": "identity", "kwargs": {}},
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
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

    dm.prepare_data()
    dm.setup()

    data = dm.train_dataloader().dataset[0]
    assert hasattr(data, "x")
    assert data.x.shape[1] == 0
    assert hasattr(data, "categorical_codes")
    assert data.categorical_codes.shape == (2, 1)


def test_h5ad_obsm_spatial_coords(tmp_path: Path) -> None:
    pytest.importorskip("anndata")

    import anndata as ad

    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.random.rand(2, 3)
    adata = ad.AnnData(X=x)
    adata.var_names = ["0", "1", "2"]
    adata.obsm["spatial"] = coords
    adata.obs["cell_type"] = ["A", "B"]

    p = tmp_path / "test.h5ad"
    adata.write_h5ad(p)

    from grass_mil.data.components.loaders import H5adConfig, load_h5ad_table

    table = load_h5ad_table(
        p,
        H5adConfig(
            coord_source="obsm",
            coord_key="spatial",
            coord_columns=("x", "y"),
            cell_id_column=None,
            categorical_label_columns=("cell_type",),
            molecular_layer=None,
            molecular_features=["0", "2"],
            use_molecular_features=True,
        ),
    )
    assert np.allclose(table.coords, coords)
    assert table.molecular_features.shape[1] == 2
    assert list(table.molecular_feature_names) == ["0", "2"]


def test_datamodule_requires_persisted_splits(tmp_path: Path) -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    index_path = processed_dir / "processed_index.json"

    entries = [
        {
            "path": str(tmp_path / f"graph_{i}.pt"),
            "sample_id": f"sample_{i % 3}",
            "region_id": f"region_{i % 2}",
            "patch_id": f"patch_{i}",
        }
        for i in range(12)
    ]
    index_path.write_text(
        json.dumps(
            {
                "entries": entries,
                "label_maps": {},
                "graph_label_maps": {},
                "reducer_state": {},
            }
        )
    )

    from grass_mil.data.spatial_omics_datamodule import SpatialOmicsDataModule

    dm = SpatialOmicsDataModule(
        data_dir=str(tmp_path),
        raw_manifest_path=str(tmp_path / "unused_manifest.csv"),
        processed_dir=str(processed_dir),
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
        split={"train_val_test_split": [0.5, 0.25, 0.25], "split_by": "patch"},
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
        graph_builder={"name": "delaunay", "kwargs": {}},
        feature_reducer={"name": "identity", "kwargs": {}},
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
        manifest=None,
        graph_labels=None,
        transforms=[],
        split_seed=7,
    )

    with pytest.raises(
        ValueError,
        match="processed_index.json must contain valid per-entry split labels",
    ):
        dm.setup()


def test_datamodule_uses_persisted_splits(tmp_path: Path) -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    index_path = processed_dir / "processed_index.json"
    entries = []
    for i, split in enumerate(["train", "train", "val", "test", "test"]):
        entries.append(
            {
                "path": str(tmp_path / f"graph_{i}.pt"),
                "sample_id": f"sample_{i}",
                "region_id": f"region_{i}",
                "patch_id": f"patch_{i}",
                "split": split,
            }
        )
    index_path.write_text(
        json.dumps(
            {
                "entries": entries,
                "label_maps": {},
                "graph_label_maps": {},
                "reducer_state": {},
            }
        )
    )

    from grass_mil.data.spatial_omics_datamodule import SpatialOmicsDataModule

    dm = SpatialOmicsDataModule(
        data_dir=str(tmp_path),
        raw_manifest_path=str(tmp_path / "unused_manifest.csv"),
        processed_dir=str(processed_dir),
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
        graph_builder={"name": "delaunay", "kwargs": {}},
        feature_reducer={"name": "identity", "fit_mode": "train_only", "kwargs": {}},
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
        manifest=None,
        graph_labels=None,
        transforms=[],
        split_seed=999,
    )
    dm.setup()
    assert len(dm.dataset_train.entries) == 2
    assert len(dm.dataset_val.entries) == 1
    assert len(dm.dataset_test.entries) == 2


def _make_unit(
    table_idx: int, sample_id: str, region_id: str, patch_idx: int, split: str = "train"
):
    from grass_mil.data.components.precompute import GraphUnitSpec

    return GraphUnitSpec(
        table_idx=table_idx,
        sample_id=sample_id,
        region_id=region_id,
        patch_idx=patch_idx,
        patch_id=f"{sample_id}_{region_id}_patch_{patch_idx}",
        indices=np.array([0, 1], dtype=int),
        split=split,
    )


def test_region_split_groups_all_patches_per_region() -> None:
    from grass_mil.data.components.precompute import _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(0, "s0", "r0", 1),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(1, "s1", "r1", 1),
        _make_unit(2, "s2", "r2", 0),
        _make_unit(2, "s2", "r2", 1),
    ]
    labels = _assign_split_labels(
        units=units, split_by="region", split_ratios=(0.5, 0.25, 0.25), seed=123
    )
    by_region = {}
    for unit, label in zip(units, labels):
        by_region.setdefault((unit.sample_id, unit.region_id), set()).add(label)
    assert all(len(splits) == 1 for splits in by_region.values())


def test_loocv_holdout_region_is_test_only_with_region_validation() -> None:
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(0, "s0", "r0", 1),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(1, "s1", "r1", 1),
        _make_unit(2, "s2", "r2", 0),
        _make_unit(2, "s2", "r2", 1),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="region",
        split_ratios=(0.8, 0.1, 0.1),
        seed=3,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="region",
            holdout_id="s1::r1",
            validation_strategy="heldout_fold_items",
            val_ratio=0.5,
            seed=11,
        ),
    )
    for unit, label in zip(units, labels):
        if unit.sample_id == "s1" and unit.region_id == "r1":
            assert label == "test"
        else:
            assert label in {"train", "val"}


def test_loocv_patch_validation_uses_only_train_fold_items() -> None:
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(0, "s0", "r0", 1),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(1, "s1", "r1", 1),
        _make_unit(2, "s2", "r2", 0),
        _make_unit(2, "s2", "r2", 1),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="region",
        split_ratios=(0.8, 0.1, 0.1),
        seed=5,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="region",
            holdout_id="s2::r2",
            validation_strategy="patches_from_train_items",
            val_ratio=0.4,
            seed=7,
        ),
    )
    # Held-out region must be test-only and never val.
    for unit, label in zip(units, labels):
        if unit.sample_id == "s2" and unit.region_id == "r2":
            assert label == "test"
    assert "val" in labels


def test_loocv_heldout_validation_allows_zero_val_ratio() -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(2, "s2", "r2", 0),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="region",
        split_ratios=(0.8, 0.1, 0.1),
        seed=3,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="region",
            holdout_id="s1::r1",
            validation_strategy="heldout_fold_items",
            val_ratio=0.0,
            seed=11,
        ),
    )
    assert "val" not in labels
    for unit, label in zip(units, labels):
        if unit.sample_id == "s1" and unit.region_id == "r1":
            assert label == "test"


def test_loocv_heldout_validation_forces_one_for_nonzero_ratio() -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(2, "s2", "r2", 0),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="region",
        split_ratios=(0.8, 0.1, 0.1),
        seed=3,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="region",
            holdout_id="s1::r1",
            validation_strategy="heldout_fold_items",
            val_ratio=0.01,
            seed=11,
        ),
    )
    assert labels.count("val") == 1


def test_loocv_patch_validation_allows_zero_val_ratio() -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(0, "s0", "r0", 1),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(1, "s1", "r1", 1),
        _make_unit(2, "s2", "r2", 0),
        _make_unit(2, "s2", "r2", 1),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="region",
        split_ratios=(0.8, 0.1, 0.1),
        seed=5,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="region",
            holdout_id="s2::r2",
            validation_strategy="patches_from_train_items",
            val_ratio=0.0,
            seed=7,
        ),
    )
    assert "val" not in labels


def test_loocv_patch_validation_forces_one_for_nonzero_ratio() -> None:
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(2, "s2", "r2", 0),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="region",
        split_ratios=(0.8, 0.1, 0.1),
        seed=5,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="region",
            holdout_id="s2::r2",
            validation_strategy="patches_from_train_items",
            val_ratio=0.01,
            seed=7,
        ),
    )
    assert labels.count("val") == 1


def test_loocv_supports_sample_fold_unit() -> None:
    from grass_mil.data.components.precompute import LoocvConfig, _assign_split_labels

    units = [
        _make_unit(0, "s0", "r0", 0),
        _make_unit(0, "s0", "r1", 0),
        _make_unit(1, "s1", "r0", 0),
        _make_unit(1, "s1", "r1", 0),
        _make_unit(2, "s2", "r0", 0),
        _make_unit(2, "s2", "r1", 0),
    ]
    labels = _assign_split_labels(
        units=units,
        split_by="sample",
        split_ratios=(0.8, 0.1, 0.1),
        seed=9,
        loocv_config=LoocvConfig(
            enabled=True,
            fold_unit="sample",
            holdout_id="s1",
            validation_strategy="heldout_fold_items",
            val_ratio=0.5,
            seed=13,
        ),
    )
    s1_labels = {label for unit, label in zip(units, labels) if unit.sample_id == "s1"}
    assert s1_labels == {"test"}


def _build_loocv_train_only_dm(tmp_path: Path, processed_dir: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("sample_id,input_path,input_type\ns0,/tmp/fake.csv,csv\n")

    from grass_mil.data.spatial_omics_datamodule import SpatialOmicsDataModule

    return SpatialOmicsDataModule(
        data_dir=str(tmp_path),
        raw_manifest_path=str(manifest_path),
        processed_dir=str(processed_dir),
        coord_scale_um=1.0,
        sample_unit="tile",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        reducer_scope="dataset",
        keep_raw_molecular=False,
        force_precompute=False,
        min_cells=1,
        use_molecular_features=False,
        categorical_features={"include_labels": []},
        split={
            "train_val_test_split": [0.8, 0.1, 0.1],
            "split_by": "sample",
            "loocv": {
                "enabled": True,
                "fold_unit": "sample",
                "holdout_id": "s0",
                "validation_strategy": "patches_from_train_items",
                "val_ratio": 0.2,
                "seed": 42,
            },
        },
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
        graph_builder={"name": "delaunay", "kwargs": {}},
        feature_reducer={
            "name": "identity",
            "fit_mode": "train_only",
            "kwargs": {},
        },
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
        manifest=None,
        graph_labels=None,
        transforms=[],
    )


def test_loocv_train_only_reducer_rejects_existing_processed_index(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "processed_index.json").write_text("{}")
    dm = _build_loocv_train_only_dm(tmp_path=tmp_path, processed_dir=processed_dir)

    with pytest.raises(ValueError, match="cannot reuse an existing processed_index"):
        dm.prepare_data()


def test_loocv_train_only_reducer_allows_fresh_processed_dir_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed_dir = tmp_path / "processed_fresh"
    dm = _build_loocv_train_only_dm(tmp_path=tmp_path, processed_dir=processed_dir)

    calls = {"precompute": 0}

    class _StubPreprocessor:
        def precompute(self) -> None:
            calls["precompute"] += 1

    monkeypatch.setattr(dm, "_build_preprocessor", lambda: _StubPreprocessor())
    dm.prepare_data()
    assert calls["precompute"] == 1


def _write_minimal_csv_and_manifest(
    *,
    raw_dir: Path,
    coords: np.ndarray,
    polygon_path: Path | None = None,
) -> Path:
    csv_path = raw_dir / "sample.csv"
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(coords.shape[0]), coords[:, 0], coords[:, 1]]),
        delimiter=",",
        fmt="%s",
        header="cell_id,x,y",
        comments="",
    )
    manifest_path = raw_dir / "manifest.csv"
    polygons_field = str(polygon_path) if polygon_path is not None else ""
    manifest_path.write_text(
        "sample_id,input_path,input_type,region_id,polygons_path\n"
        f"sample_1,{csv_path},csv,region_1,{polygons_field}\n"
    )
    return manifest_path


def _build_minimal_datamodule(
    *,
    data_dir: Path,
    manifest_path: Path,
    processed_dir: Path,
    coord_scale_um: float,
    force_precompute: bool,
    edge_features: list[str],
):
    from grass_mil.data.spatial_omics_datamodule import SpatialOmicsDataModule

    return SpatialOmicsDataModule(
        data_dir=str(data_dir),
        raw_manifest_path=str(manifest_path),
        processed_dir=str(processed_dir),
        coord_scale_um=coord_scale_um,
        sample_unit="full",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        reducer_scope="sample",
        keep_raw_molecular=False,
        force_precompute=force_precompute,
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
        graph_builder={"name": "delaunay", "kwargs": {"edge_features": edge_features}},
        feature_reducer={"name": "identity", "kwargs": {}},
        tiling={"tile_size_um": 50.0, "stride_um": 50.0, "min_cells": 1},
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


def test_coord_scale_um_scales_positions_and_distances(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("scipy")

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    manifest_path = _write_minimal_csv_and_manifest(raw_dir=raw_dir, coords=coords)

    dm = _build_minimal_datamodule(
        data_dir=data_dir,
        manifest_path=manifest_path,
        processed_dir=data_dir / "processed",
        coord_scale_um=2.0,
        force_precompute=True,
        edge_features=["distance"],
    )
    dm.prepare_data()
    dm.setup()
    data = dm.train_dataloader().dataset[0]

    assert np.allclose(data.pos.detach().cpu().numpy(), np.array([[0.0, 0.0], [6.0, 8.0]]))
    assert list(getattr(data, "edge_attr_names", [])) == ["distance"]
    assert np.allclose(data.edge_attr.detach().cpu().numpy(), np.full((2, 1), 10.0))


def test_coord_scale_um_scales_polygons_with_coordinates(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("scipy")

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    coords = np.array([[1.0, 1.0], [2.0, 2.0]])
    polygon_path = raw_dir / "polygons.wkt"
    _write_polygon_wkt(
        polygon_path,
        np.array([[0, 0], [0, 3], [3, 3], [3, 0], [0, 0]], dtype=float),
    )
    manifest_path = _write_minimal_csv_and_manifest(
        raw_dir=raw_dir,
        coords=coords,
        polygon_path=polygon_path,
    )

    dm = _build_minimal_datamodule(
        data_dir=data_dir,
        manifest_path=manifest_path,
        processed_dir=data_dir / "processed",
        coord_scale_um=2.0,
        force_precompute=True,
        edge_features=["distance"],
    )
    dm.prepare_data()
    dm.setup()
    data = dm.train_dataloader().dataset[0]

    assert int(data.num_nodes) == 2


def test_precompute_reuse_rejects_coord_scale_um_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("scipy")

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    processed_dir = data_dir / "processed"

    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    manifest_path = _write_minimal_csv_and_manifest(raw_dir=raw_dir, coords=coords)

    dm_first = _build_minimal_datamodule(
        data_dir=data_dir,
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        coord_scale_um=1.0,
        force_precompute=True,
        edge_features=["distance"],
    )
    dm_first.prepare_data()
    first_index = json.loads((processed_dir / "processed_index.json").read_text())
    assert float(first_index["preprocessing"]["coord_scale_um"]) == 1.0

    dm_second = _build_minimal_datamodule(
        data_dir=data_dir,
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        coord_scale_um=2.0,
        force_precompute=False,
        edge_features=["distance"],
    )
    with pytest.raises(ValueError, match="coord_scale_um=1.0"):
        dm_second.prepare_data()

    dm_third = _build_minimal_datamodule(
        data_dir=data_dir,
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        coord_scale_um=2.0,
        force_precompute=True,
        edge_features=["distance"],
    )
    dm_third.prepare_data()
    third_index = json.loads((processed_dir / "processed_index.json").read_text())
    assert float(third_index["preprocessing"]["coord_scale_um"]) == 2.0


def test_graph_dataset_migrates_legacy_categorical_index(tmp_path):
    """Processed caches written before the rename must still load."""
    import json

    import torch
    from torch_geometric.data import Data

    from grass_mil.data.components.datasets import SpatialOmicsGraphDataset

    graph_path = tmp_path / "legacy.pt"
    data = Data(x=torch.randn(3, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
    data.num_nodes = 3
    data.categorical_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    torch.save(data, graph_path)

    index_path = tmp_path / "processed_index.json"
    index_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "path": str(graph_path),
                        "sample_id": "s0",
                        "region_id": "r0",
                        "patch_id": "p0",
                        "split": "train",
                    }
                ]
            }
        )
    )

    loaded = SpatialOmicsGraphDataset(index_path)[0]
    assert torch.equal(loaded.categorical_codes.view(-1), torch.tensor([0, 1, 2]))
    assert getattr(loaded, "categorical_index", None) is None
