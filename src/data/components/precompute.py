from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from .feature_reducers import FeatureReducerConfig, get_feature_reducer
from .graph_builders import GraphBuilderConfig, GraphBuildResult, get_graph_builder
from .loaders import (
    CsvConfig,
    H5adConfig,
    SceConfig,
    load_csv_table,
    load_h5ad_table,
    load_polygons_from_path,
    load_sce_table,
)
from .patching import TileConfig, build_grid_tiles, build_patches_from_polygons, filter_coords_by_union
from .spatial_types import SpatialOmicsTable


@dataclass
class ManifestConfig:
    sample_id: str = "sample_id"
    input_path: str = "input_path"
    input_type: str = "input_type"
    region_id: str = "region_id"
    polygons_path: str = "polygons_path"


@dataclass
class GraphLabelConfig:
    label_file: Optional[str] = None
    id_column: str = "id"
    tasks: Optional[Sequence[str]] = None
    scope: str = "region"  # region | patch | sample


@dataclass
class PrecomputeConfig:
    raw_manifest_path: str
    processed_dir: str
    coord_scale_um: float
    sample_unit: str
    reducer_scope: str
    keep_raw_molecular: bool
    force: bool
    min_cells: int
    use_molecular_features: bool


@dataclass
class CategoricalFeatureConfig:
    include_labels: Sequence[str]


class SpatialOmicsPreprocessor:
    def __init__(
        self,
        manifest_config: ManifestConfig,
        precompute_config: PrecomputeConfig,
        csv_config: CsvConfig,
        h5ad_config: H5adConfig,
        sce_config: SceConfig,
        graph_builder_config: GraphBuilderConfig,
        feature_reducer_config: FeatureReducerConfig,
        categorical_feature_config: CategoricalFeatureConfig,
        tile_config: TileConfig,
        graph_label_config: Optional[GraphLabelConfig] = None,
    ) -> None:
        self.manifest_config = manifest_config
        self.precompute_config = precompute_config
        self.csv_config = csv_config
        self.h5ad_config = h5ad_config
        self.sce_config = sce_config
        self.graph_builder_config = graph_builder_config
        self.feature_reducer_config = feature_reducer_config
        self.categorical_feature_config = categorical_feature_config
        self.tile_config = tile_config
        self.graph_label_config = graph_label_config or GraphLabelConfig()

    def precompute(self) -> Path:
        processed_dir = Path(self.precompute_config.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        index_path = processed_dir / "processed_index.json"

        if index_path.exists() and not self.precompute_config.force:
            return index_path

        manifest_rows = self._load_manifest(Path(self.precompute_config.raw_manifest_path))
        tables = [self._load_table(row) for row in manifest_rows]
        label_maps = self._build_label_maps(tables)
        graph_label_maps: Dict[str, Dict[str, int]] = {}
        graph_label_df = self._load_graph_labels()

        reducer_state: Dict[str, object] = {"scope": self.precompute_config.reducer_scope}
        reducer = get_feature_reducer(self.feature_reducer_config)

        if self.precompute_config.use_molecular_features and self.precompute_config.reducer_scope == "dataset":
            all_features = np.concatenate(
                [t.molecular_features for t in tables if t.molecular_features is not None], axis=0
            )
            reducer.fit(all_features)
            reducer_state["global"] = reducer.state_dict()

        entries: List[Dict[str, object]] = []
        reducer_state["by_sample"] = {}

        for row, table in zip(manifest_rows, tables):
            sample_id = table.sample_id or str(row[self.manifest_config.sample_id])
            region_id = table.region_id or row.get(self.manifest_config.region_id)
            polygons_path = row.get(self.manifest_config.polygons_path)

            table = _apply_coord_scale(table, self.precompute_config.coord_scale_um)
            polygons = load_polygons_from_path(polygons_path) if polygons_path else None

            if polygons:
                union_idx = filter_coords_by_union(table.coords, polygons)
                table = _subset_table(table, union_idx)

            embedded = None
            if self.precompute_config.use_molecular_features:
                reducer_instance = reducer
                if self.precompute_config.reducer_scope == "sample":
                    reducer_instance = get_feature_reducer(self.feature_reducer_config)
                    reducer_instance.fit(table.molecular_features)
                    reducer_state["by_sample"][sample_id] = reducer_instance.state_dict()

                embedded = reducer_instance.transform(table.molecular_features)

            if polygons and self.precompute_config.sample_unit != "full":
                patch_indices = build_patches_from_polygons(table.coords, polygons)
            elif self.precompute_config.sample_unit == "tile":
                patch_indices = build_grid_tiles(table.coords, self.tile_config)
            else:
                patch_indices = [np.arange(table.coords.shape[0])]

            patch_indices = [idx for idx in patch_indices if idx.size >= self.precompute_config.min_cells]

            for patch_idx, indices in enumerate(patch_indices):
                patch_id = _build_patch_id(sample_id, region_id, patch_idx)
                data = self._build_pyg_data(
                    table=table,
                    indices=indices,
                    embedded=embedded[indices] if embedded is not None else None,
                    sample_id=sample_id,
                    region_id=region_id,
                    patch_id=patch_id,
                    label_maps=label_maps,
                    graph_label_df=graph_label_df,
                    graph_label_maps=graph_label_maps,
                )

                graph_path = processed_dir / "graphs" / sample_id
                graph_path.mkdir(parents=True, exist_ok=True)
                data_path = graph_path / f"{patch_id}.pt"
                torch.save(data, data_path)
                entries.append(
                    {
                        "path": str(data_path),
                        "sample_id": sample_id,
                        "region_id": region_id,
                        "patch_id": patch_id,
                    }
                )

        index_payload = {
            "entries": entries,
            "label_maps": label_maps,
            "graph_label_maps": graph_label_maps,
            "reducer_state": reducer_state,
            "categorical_features": {
                "include_labels": list(self.categorical_feature_config.include_labels),
            },
        }
        index_path.write_text(json.dumps(index_payload, indent=2))
        return index_path

    def _load_manifest(self, path: Path) -> List[Dict[str, str]]:
        import pandas as pd

        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        required = [
            self.manifest_config.sample_id,
            self.manifest_config.input_path,
            self.manifest_config.input_type,
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Manifest missing required column: {col}")
        self._validate_manifest_regions(df)
        return df.to_dict(orient="records")

    def _validate_manifest_regions(self, df) -> None:
        sample_col = self.manifest_config.sample_id
        region_col = self.manifest_config.region_id
        if region_col not in df.columns:
            return

        duplicates = df[sample_col].duplicated(keep=False)
        if not duplicates.any():
            return

        for sample_id, group in df[duplicates].groupby(sample_col):
            region_vals = group[region_col].fillna("").astype(str).str.strip()
            if region_vals.eq("").any():
                raise ValueError(
                    f"Manifest has duplicate sample_id '{sample_id}' with empty region_id."
                )
            if region_vals.duplicated().any():
                raise ValueError(
                    f"Manifest has duplicate sample_id '{sample_id}' with repeated region_id values."
                )

    def _load_table(self, row: Dict[str, str]) -> SpatialOmicsTable:
        sample_id = row[self.manifest_config.sample_id]
        region_id = row.get(self.manifest_config.region_id)
        input_path = row[self.manifest_config.input_path]
        input_type = row[self.manifest_config.input_type].lower()

        if input_type in {"csv", "tsv"}:
            cfg = self.csv_config
            if input_type == "tsv":
                cfg = CsvConfig(
                    coord_columns=cfg.coord_columns,
                    cell_id_column=cfg.cell_id_column,
                    categorical_label_columns=cfg.categorical_label_columns,
                    molecular_columns=cfg.molecular_columns,
                    sep="\t",
                )
            return load_csv_table(input_path, cfg, sample_id=sample_id, region_id=region_id)
        if input_type == "h5ad":
            return load_h5ad_table(input_path, self.h5ad_config, sample_id=sample_id, region_id=region_id)
        if input_type in {"sce", "rds"}:
            return load_sce_table(input_path, self.sce_config, sample_id=sample_id, region_id=region_id)
        raise ValueError(f"Unsupported input type: {input_type}")

    def _build_label_maps(self, tables: Sequence[SpatialOmicsTable]) -> Dict[str, Dict[str, int]]:
        label_maps: Dict[str, Dict[str, int]] = {}
        for table in tables:
            for label_name, labels in table.categorical_labels.items():
                label_maps.setdefault(label_name, {})
                for label in labels:
                    if label not in label_maps[label_name]:
                        label_maps[label_name][label] = len(label_maps[label_name])
        return label_maps

    def _load_graph_labels(self):
        if not self.graph_label_config.label_file:
            return None
        import pandas as pd

        label_path = Path(self.graph_label_config.label_file)
        if not label_path.exists():
            raise FileNotFoundError(f"Graph label file not found: {label_path}")
        df = pd.read_csv(label_path)
        if self.graph_label_config.id_column not in df.columns:
            raise ValueError(
                f"Graph label file missing id column '{self.graph_label_config.id_column}'"
            )
        df = df.set_index(self.graph_label_config.id_column)
        return df

    def _build_pyg_data(
        self,
        table: SpatialOmicsTable,
        indices: np.ndarray,
        embedded: Optional[np.ndarray],
        sample_id: str,
        region_id: Optional[str],
        patch_id: str,
        label_maps: Dict[str, Dict[str, int]],
        graph_label_df,
        graph_label_maps: Dict[str, Dict[str, int]],
    ) -> Data:
        coords = table.coords[indices]
        graph_result: GraphBuildResult = get_graph_builder(self.graph_builder_config).build(coords)

        data_kwargs = {
            "pos": torch.from_numpy(coords).float(),
            "edge_index": graph_result.edge_index,
        }
        if embedded is not None:
            data_kwargs["x"] = torch.from_numpy(embedded).float()
        data = Data(**data_kwargs)
        if graph_result.edge_attr is not None:
            data.edge_attr = graph_result.edge_attr.float()
            data.edge_attr_names = graph_result.edge_attr_names

        if self.precompute_config.keep_raw_molecular and table.molecular_features is not None:
            data.x_raw = torch.from_numpy(table.molecular_features[indices]).float()

        for label_name, labels in table.categorical_labels.items():
            mapping = label_maps.get(label_name, {})
            encoded = np.array([mapping[val] for val in labels[indices]], dtype=np.int64)
            setattr(data, f"label_{label_name}", torch.from_numpy(encoded))
        data.cell_label_names = list(table.categorical_labels.keys())
        self._attach_categorical_indices(data, table.categorical_labels, label_maps, indices)

        if graph_label_df is not None:
            label_id = _select_graph_label_id(
                scope=self.graph_label_config.scope,
                sample_id=sample_id,
                region_id=region_id,
                patch_id=patch_id,
            )
            if label_id in graph_label_df.index:
                row = graph_label_df.loc[label_id]
                tasks = self.graph_label_config.tasks or list(row.index)
                graph_values = []
                for task in tasks:
                    value = row[task]
                    if isinstance(value, str):
                        graph_label_maps.setdefault(task, {})
                        mapping = graph_label_maps[task]
                        if value not in mapping:
                            mapping[value] = len(mapping)
                        graph_values.append(mapping[value])
                    else:
                        graph_values.append(value)
                data.graph_y = torch.tensor(graph_values, dtype=torch.float).unsqueeze(0)
                data.graph_label_names = tasks

        data.sample_id = sample_id
        data.region_id = region_id
        data.patch_id = patch_id
        return data

    def _attach_categorical_indices(
        self,
        data: Data,
        labels: Dict[str, np.ndarray],
        label_maps: Dict[str, Dict[str, int]],
        indices: np.ndarray,
    ) -> None:
        config = self.categorical_feature_config
        if not config.include_labels:
            return

        encoded_cols: List[np.ndarray] = []
        slices: Dict[str, int] = {}
        for col_idx, label_name in enumerate(config.include_labels):
            if label_name not in labels:
                raise ValueError(f"Categorical label '{label_name}' not found in data.")
            mapping = label_maps.get(label_name, {})
            encoded = np.array([mapping[val] for val in labels[label_name][indices]], dtype=np.int64)
            encoded_cols.append(encoded)
            slices[label_name] = col_idx

        categorical_index = np.stack(encoded_cols, axis=1)
        data.categorical_index = torch.from_numpy(categorical_index).long()
        data.categorical_labels = list(config.include_labels)
        data.categorical_slices = slices


def _apply_coord_scale(table: SpatialOmicsTable, coord_scale_um: float) -> SpatialOmicsTable:
    if coord_scale_um == 1.0:
        return table
    scaled = table.coords * coord_scale_um
    return SpatialOmicsTable(
        coords=scaled,
        molecular_features=table.molecular_features,
        categorical_labels=table.categorical_labels,
        cell_ids=table.cell_ids,
        sample_id=table.sample_id,
        region_id=table.region_id,
    )


def _subset_table(table: SpatialOmicsTable, indices: np.ndarray) -> SpatialOmicsTable:
    return SpatialOmicsTable(
        coords=table.coords[indices],
        molecular_features=table.molecular_features[indices],
        categorical_labels={k: v[indices] for k, v in table.categorical_labels.items()},
        cell_ids=table.cell_ids[indices],
        sample_id=table.sample_id,
        region_id=table.region_id,
    )


def _select_graph_label_id(
    scope: str,
    sample_id: str,
    region_id: Optional[str],
    patch_id: str,
) -> str:
    if scope == "patch":
        return patch_id
    if scope == "sample":
        return sample_id
    if region_id is None:
        raise ValueError("region_id required for region-scope graph labels")
    return region_id


def _build_patch_id(sample_id: str, region_id: Optional[str], patch_idx: int) -> str:
    if region_id:
        return f"{sample_id}_{region_id}_patch_{patch_idx}"
    return f"{sample_id}_patch_{patch_idx}"
