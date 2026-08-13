from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lightning")

from grass_mil.data.components.feature_reducers import FeatureReducerConfig
from grass_mil.data.components.graph_builders import GraphBuilderConfig
from grass_mil.data.components.loaders import CsvConfig, H5adConfig, SceConfig
from grass_mil.data.components.patching import TileConfig
from grass_mil.data.components.precompute import (
    CategoricalFeatureConfig,
    ManifestConfig,
    PrecomputeConfig,
    SpatialOmicsPreprocessor,
)


def test_tsv_loader_preserves_use_molecular_features_flag(tmp_path: Path) -> None:
    tsv_path = tmp_path / "cells.tsv"
    np.savetxt(
        tsv_path,
        np.array(
            [
                ["c0", "0.0", "1.0"],
                ["c1", "2.0", "3.0"],
            ],
            dtype=str,
        ),
        delimiter="\t",
        fmt="%s",
        header="cell_id\tx\ty",
        comments="",
    )

    preprocessor = SpatialOmicsPreprocessor(
        manifest_config=ManifestConfig(),
        precompute_config=PrecomputeConfig.from_args(
            raw_manifest_path=str(tmp_path / "manifest.csv"),
            processed_dir=str(tmp_path / "processed"),
            coord_scale_um=1.0,
            sample_unit="tile",
            reducer_scope="sample",
            reducer_fit_mode="train_only",
            keep_raw_molecular=False,
            force=False,
            min_cells=1,
            use_molecular_features=False,
            split_by="patch",
            split_ratios=(1.0, 0.0, 0.0),
            split_seed=42,
        ),
        csv_config=CsvConfig(
            coord_columns=("x", "y"),
            cell_id_column="cell_id",
            categorical_label_columns=[],
            molecular_columns=None,
            use_molecular_features=False,
            sep=",",
        ),
        h5ad_config=H5adConfig(
            coord_source="obsm",
            coord_key="spatial",
            coord_columns=("x", "y"),
            cell_id_column=None,
            categorical_label_columns=[],
            molecular_layer=None,
            molecular_features=None,
            use_molecular_features=False,
        ),
        sce_config=SceConfig(
            assay_name=None,
            coord_source="colData",
            coord_key=None,
            coord_columns=("x", "y"),
            cell_id_column=None,
            categorical_label_columns=[],
            molecular_features=None,
            use_molecular_features=False,
        ),
        graph_builder_config=GraphBuilderConfig(name="delaunay", kwargs={}),
        feature_reducer_config=FeatureReducerConfig(name="identity", kwargs={}),
        categorical_feature_config=CategoricalFeatureConfig(include_labels=[]),
        tile_config=TileConfig(tile_size_um=10.0, stride_um=10.0, min_cells=1),
    )

    row = {
        "sample_id": "s1",
        "input_path": str(tsv_path),
        "input_type": "tsv",
        "region_id": "r1",
        "polygons_path": "",
    }
    table = preprocessor._load_table(row)
    assert table.molecular_features is None
    assert table.coords.shape == (2, 2)
