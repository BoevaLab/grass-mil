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

    from src.data.spatial_omics_datamodule import SpatialOmicsDataModule

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
        graph_builder={"name": "delaunay", "kwargs": {}},
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
        transforms=[
            {
                "name": "src.data.components.transforms.CompositionVector",
                "kwargs": {"label_attr": "label_cell_type", "num_classes": 2, "keep_original": True},
            }
        ],
    )

    dm.prepare_data()
    dm.setup()

    dataset = dm.train_dataloader().dataset
    assert len(dataset) > 0

    data = dataset[0]
    assert hasattr(data, "x")
    assert hasattr(data, "pos")
    assert hasattr(data, "label_cell_type")
    assert data.patch_id.startswith("sample_1")
    assert hasattr(data, "edge_attr")
    assert hasattr(data, "graph_y")
    assert data.x.shape[0] == 1