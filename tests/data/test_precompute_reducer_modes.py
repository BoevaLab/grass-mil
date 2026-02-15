import numpy as np
import pytest

pytest.importorskip("lightning")

from src.data.components.feature_reducers import FeatureReducerConfig
from src.data.components.graph_builders import GraphBuilderConfig
from src.data.components.loaders import CsvConfig, H5adConfig, SceConfig
from src.data.components.patching import TileConfig
from src.data.components.precompute import (
    CategoricalFeatureConfig,
    GraphUnitSpec,
    ManifestConfig,
    PrecomputeConfig,
    PreparedSample,
    SpatialOmicsPreprocessor,
)
from src.data.components.spatial_types import SpatialOmicsTable


def _make_preprocessor(fit_mode: str) -> SpatialOmicsPreprocessor:
    return SpatialOmicsPreprocessor(
        manifest_config=ManifestConfig(),
        precompute_config=PrecomputeConfig.from_args(
            raw_manifest_path="/tmp/unused_manifest.csv",
            processed_dir="/tmp/unused_processed",
            coord_scale_um=1.0,
            sample_unit="tile",
            reducer_scope="dataset",
            reducer_fit_mode=fit_mode,
            keep_raw_molecular=False,
            force=False,
            min_cells=1,
            use_molecular_features=True,
            split_by="patch",
            split_ratios=(0.8, 0.1, 0.1),
            split_seed=42,
        ),
        csv_config=CsvConfig(
            coord_columns=("x", "y"),
            cell_id_column="cell_id",
            categorical_label_columns=[],
            molecular_columns=None,
            use_molecular_features=True,
        ),
        h5ad_config=H5adConfig(
            coord_source="obsm",
            coord_key="spatial",
            coord_columns=("x", "y"),
            cell_id_column=None,
            categorical_label_columns=[],
            molecular_layer=None,
            molecular_features=None,
            use_molecular_features=True,
        ),
        sce_config=SceConfig(
            assay_name=None,
            coord_source="colData",
            coord_key=None,
            coord_columns=("x", "y"),
            cell_id_column=None,
            categorical_label_columns=[],
            use_molecular_features=True,
        ),
        graph_builder_config=GraphBuilderConfig(name="delaunay", kwargs={}),
        feature_reducer_config=FeatureReducerConfig(name="pca", kwargs={"n_components": 1}),
        categorical_feature_config=CategoricalFeatureConfig(include_labels=[]),
        tile_config=TileConfig(tile_size_um=10.0, stride_um=10.0, min_cells=1),
    )


def _make_prepared_and_units():
    table_train = SpatialOmicsTable(
        coords=np.zeros((3, 2), dtype=float),
        molecular_features=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float),
        molecular_feature_names=["a", "b"],
        categorical_labels={},
        cell_ids=np.array(["a0", "a1", "a2"]),
        sample_id="s0",
        region_id="r0",
    )
    table_test = SpatialOmicsTable(
        coords=np.zeros((3, 2), dtype=float),
        molecular_features=np.array([[10.0, 11.0], [10.0, 11.0], [10.0, 11.0]], dtype=float),
        molecular_feature_names=["a", "b"],
        categorical_labels={},
        cell_ids=np.array(["b0", "b1", "b2"]),
        sample_id="s1",
        region_id="r1",
    )
    prepared = [
        PreparedSample(
            table_idx=0,
            table=table_train,
            sample_id="s0",
            region_id="r0",
            patch_indices=[np.array([0, 1, 2])],
        ),
        PreparedSample(
            table_idx=1,
            table=table_test,
            sample_id="s1",
            region_id="r1",
            patch_indices=[np.array([0, 1, 2])],
        ),
    ]
    units = [
        GraphUnitSpec(
            table_idx=0,
            sample_id="s0",
            region_id="r0",
            patch_idx=0,
            patch_id="s0_r0_patch_0",
            indices=np.array([0, 1, 2]),
            split="train",
        ),
        GraphUnitSpec(
            table_idx=1,
            sample_id="s1",
            region_id="r1",
            patch_idx=0,
            patch_id="s1_r1_patch_0",
            indices=np.array([0, 1, 2]),
            split="test",
        ),
    ]
    return prepared, units


def test_prepare_reducer_global_fits_on_all_data():
    preprocessor = _make_preprocessor("global")
    prepared, units = _make_prepared_and_units()
    state, reducer = preprocessor._prepare_reducer(prepared, units)

    assert state["fit_mode"] == "global"
    assert state["fitted_on"] == "global"
    assert "global" in state
    assert np.isclose(float(reducer.mean_[0, 0]), 5.0)


def test_prepare_reducer_train_only_fits_on_train_split():
    preprocessor = _make_preprocessor("train_only")
    prepared, units = _make_prepared_and_units()
    state, reducer = preprocessor._prepare_reducer(prepared, units)

    assert state["fit_mode"] == "train_only"
    assert state["fitted_on"] == "train_only"
    assert "global" in state
    assert np.isclose(float(reducer.mean_[0, 0]), 0.0)
