"""Inputs, graphs, units, labels, splits and the datamodule."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from grass_mil.data import LabelTable, SpatialOmicsDataModule
from grass_mil.data.graphs import GraphConfig, delaunay_edges, edge_features, log_distance_feature
from grass_mil.data.labels import class_index
from grass_mil.data.loaders import InputConfig, input_type_of, load_cell_table
from grass_mil.data.precompute import ProcessedIndex, read_manifest
from grass_mil.data.splits import inner_folds, load_folds, random_split
from grass_mil.data.units import TilingConfig, grid_tiles


def _datamodule(cohort, tmp_path, **kwargs):
    defaults = dict(
        manifest_path=str(cohort.manifest),
        processed_dir=str(tmp_path / "processed"),
        input={"cell_id_column": "cell_id", "features": ["area"]},
        graph={"neighbor_cutoff": 55},
        labels={"file": str(cohort.labels), "columns": ["label"], "patient_column": "patient_id"},
        sampler={"depth": 2, "roots_per_unit": 16, "batch_size": 8},
    )
    defaults.update(kwargs)
    return SpatialOmicsDataModule(**defaults)


# ---------------------------------------------------------------- inputs
def test_input_type_from_suffix_or_manifest():
    assert input_type_of("a/b.csv") == "csv"
    assert input_type_of("a/b.h5ad") == "h5ad"
    assert input_type_of("a/b.rds") == "sce"
    assert input_type_of("a/b.txt", "csv") == "csv"
    with pytest.raises(ValueError):
        input_type_of("a/b.parquet")


def test_csv_features_all_excludes_ids_coordinates_and_cell_type(cohort):
    path = cohort.manifest.parent / f"{cohort.regions[0]}.csv"
    table = load_cell_table(path, "csv", InputConfig.from_dict({"cell_id_column": "cell_id"}))
    assert table.feature_names == ["area"]
    assert table.cell_types is not None and set(table.cell_types) <= {"A", "B", "C", "D"}
    assert table.cell_ids[0] == "c0"


def test_h5ad_input(tmp_path):
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    adata = anndata.AnnData(
        X=rng.random((30, 3)).astype(np.float32),
        obs=pd.DataFrame(
            {"cell_type": rng.choice(["a", "b"], 30)}, index=[f"c{i}" for i in range(30)]
        ),
    )
    adata.var_names = ["g1", "g2", "g3"]
    adata.obsm["spatial"] = rng.random((30, 2)) * 100
    adata.write_h5ad(tmp_path / "s.h5ad")
    table = load_cell_table(
        tmp_path / "s.h5ad", "h5ad", InputConfig.from_dict({"features": ["g2"]})
    )
    assert table.features.shape == (30, 1) and table.feature_names == ["g2"]
    assert table.coords.shape == (30, 2)


def test_manifest_ids_stay_strings_and_region_defaults_to_sample(tmp_path):
    (tmp_path / "m.csv").write_text("sample_id,input_path,region_id\n001,a.csv,\n002,b.csv,7\n")
    df = read_manifest(tmp_path / "m.csv")
    assert list(df["sample_id"]) == ["001", "002"]
    assert list(df["region_id"]) == ["001", "7"]
    assert df["input_path"].iloc[0] == str((tmp_path / "a.csv").resolve())


def test_manifest_rejects_repeated_regions(tmp_path):
    (tmp_path / "m.csv").write_text("sample_id,input_path,region_id\na,a.csv,r\nb,b.csv,r\n")
    with pytest.raises(ValueError, match="unique"):
        read_manifest(tmp_path / "m.csv")


# ---------------------------------------------------------------- graphs
def test_delaunay_edges_are_symmetric_and_unique():
    coords = np.random.default_rng(0).random((50, 2))
    e = delaunay_edges(coords).numpy()
    pairs = set(map(tuple, e.T))
    assert len(pairs) == e.shape[1]
    assert all((b, a) in pairs for a, b in pairs)
    from scipy.spatial import Delaunay

    expected = set()
    for simplex in Delaunay(coords).simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    expected.add((int(simplex[i]), int(simplex[j])))
    assert pairs == expected


def test_edge_feature_columns():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 200.0]])
    e = delaunay_edges(coords)
    cfg = GraphConfig.from_dict(
        {"edge_features": ["log_distance", "neighbor", "distance"], "neighbor_cutoff": 15}
    )
    attr = edge_features(coords, e, cfg).numpy()
    lengths = attr[:, 2]
    assert np.allclose(attr[:, 1], lengths <= 15)
    assert np.allclose(attr[:, 0], log_distance_feature(lengths, 2.0, 5.0))
    assert log_distance_feature(np.array([np.exp(2.0), np.exp(5.0), 1e6]), 2.0, 5.0).tolist() == [
        0.0,
        1.0,
        1.0,
    ]


def test_grid_tiles_cover_every_cell_once():
    coords = np.random.default_rng(0).random((200, 2)) * 500
    tiles = grid_tiles(coords, TilingConfig(size=100, stride=100))
    covered = np.concatenate(tiles)
    assert sorted(covered.tolist()) == list(range(200))


# ---------------------------------------------------------------- labels
def test_label_table_encodes_text_labels_and_patients(tmp_path):
    (tmp_path / "l.csv").write_text("region_id,grade,patient_id\nr1,high,p1\nr2,low,p1\nr3,,p2\n")
    table = LabelTable.read(tmp_path / "l.csv", id_column="region_id", patient_column="patient_id")
    assert table.codes == {"grade": ["high", "low"]}
    assert table.target("r2").tolist() == [1.0]
    assert "r3" not in table  # unlabelled
    assert table.patient("r1") == "p1"
    assert table.resolve("x", "r2") == "r2"


def test_class_index_rounds_halves_up():
    assert [class_index(v) for v in (0.0, 0.49, 0.5, 1.0, 2.0)] == [0, 0, 1, 1, 2]


# ---------------------------------------------------------------- splits
def test_random_split_keeps_groups_together_and_stratifies():
    groups = {f"u{i}": f"g{i // 2}" for i in range(20)}
    strata = {f"g{j}": str(j % 2) for j in range(10)}
    split = random_split(groups, (0.6, 0.2, 0.2), seed=1, strata=strata)
    for g in set(groups.values()):
        assert len({split[u] for u in groups if groups[u] == g}) == 1
    for name in ("train", "val", "test"):
        classes = {strata[groups[u]] for u in split if split[u] == name}
        assert classes == {"0", "1"}


def test_inner_folds_are_group_aware_and_stratified():
    ids = [f"u{i}" for i in range(24)]
    groups = {u: f"p{i // 2}" for i, u in enumerate(ids)}
    classes = {u: (i // 2) % 2 for i, u in enumerate(ids)}
    folds = inner_folds(ids, groups=groups, classes=classes, k=3, seed=0)
    assert len(folds) == 3
    for train, val in folds:
        assert not {groups[u] for u in train} & {groups[u] for u in val}
        assert {classes[u] for u in val} == {0, 1}
    assert inner_folds(ids, groups=groups, classes=None, k=3, seed=0)


def test_load_folds_rejects_repeated_ids(tmp_path):
    (tmp_path / "f.json").write_text(json.dumps([["a", "b"], ["b"]]))
    with pytest.raises(ValueError):
        load_folds(tmp_path / "f.json")


# ------------------------------------------------------------ datamodule
def test_datamodule_builds_reuses_and_refuses_stale_graphs(cohort, tmp_path):
    dm = _datamodule(cohort, tmp_path)
    dm.prepare_data()
    dm.setup()
    assert dm.input_spec.feature_dim == 1
    assert dm.input_spec.cell_types == ("A", "B", "C", "D")
    assert dm.input_spec.edge_dim == 2
    sizes = [len(dm.dataset_train), len(dm.dataset_val), len(dm.dataset_test)]
    assert sum(sizes) == len(cohort.regions) and min(sizes) > 0
    data = dm.dataset_train[0]
    assert data.cell_type.dtype == torch.long and data.graph_y.shape == (1, 1)
    _datamodule(cohort, tmp_path).prepare_data()  # same settings: reused
    with pytest.raises(ValueError, match="different settings"):
        _datamodule(cohort, tmp_path, input={"features": None}).prepare_data()


def test_explicit_split_by_region_or_patientless_ids(cohort, tmp_path):
    dm = _datamodule(
        cohort,
        tmp_path,
        split={"train": cohort.regions[:6], "val": cohort.regions[6:8], "test": []},
    )
    dm.prepare_data()
    dm.setup()
    assert [e.unit_id for e in dm.dataset_train.entries] == cohort.regions[:6]
    assert len(dm.dataset_test) == 0
    with pytest.raises(ValueError, match="no processed unit"):
        bad = _datamodule(cohort, tmp_path, split={"train": ["nope"]})
        bad.setup()


def test_tile_units_inherit_region_labels(cohort, tmp_path):
    dm = _datamodule(
        cohort, tmp_path, unit="tile", tiling={"size": 300, "stride": 300}, min_cells=5
    )
    dm.prepare_data()
    dm.setup()
    entries = dm.all_units().entries
    assert all("_tile" in e.unit_id for e in entries)
    data = dm.all_units()[0]
    assert data.graph_y.item() == cohort.label_of[entries[0].region_id]


def test_pca_reducer_end_to_end_and_reuse_of_its_basis(cohort, tmp_path):
    reducer = {"name": "pca", "n_components": 1}
    dm = _datamodule(cohort, tmp_path, feature_reducer=reducer)
    dm.prepare_data()
    index = ProcessedIndex.read(tmp_path / "processed")
    assert index.feature_names == ["pc1"]
    assert len(index.feature_reducer["state"]["mean"]) == 1
    other = _datamodule(
        cohort,
        tmp_path / "other",
        feature_reducer={
            **reducer,
            "state_from": str(tmp_path / "processed" / "processed_index.json"),
        },
    )
    other.prepare_data()
    other.setup()
    a = dm.all_units()[0].x
    b = other.all_units()[0].x
    assert torch.allclose(a, b)


def test_train_fitted_reducer_is_refused_for_explicit_splits(cohort, tmp_path):
    with pytest.raises(ValueError, match="fit_on=train"):
        _datamodule(
            cohort,
            tmp_path,
            feature_reducer={"name": "pca", "fit_on": "train"},
            split={"train": cohort.regions},
        )
