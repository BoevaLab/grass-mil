from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
from .patching import (
    TileConfig,
    build_grid_tiles,
    build_patches_from_polygons,
    filter_coords_by_union,
)
from .spatial_types import SpatialOmicsTable


@dataclass
class ManifestConfig:
    sample_id: str = "sample_id"
    input_path: str = "input_path"
    input_type: str = "input_type"
    region_id: str = "region_id"
    polygons_path: str = "polygons_path"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "ManifestConfig":
        return cls(**(data or {}))


@dataclass
class GraphLabelConfig:
    label_file: Optional[str] = None
    id_column: str = "id"
    tasks: Optional[Sequence[str]] = None
    scope: str = "region"  # region | patch | sample

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "GraphLabelConfig":
        return cls(**(data or {}))


@dataclass
class PrecomputeConfig:
    raw_manifest_path: str
    processed_dir: str
    coord_scale_um: float
    sample_unit: str
    reducer_scope: str
    reducer_fit_mode: str
    keep_raw_molecular: bool
    force: bool
    min_cells: int
    use_molecular_features: bool
    split_by: str
    split_ratios: Tuple[float, float, float]
    split_seed: int
    loocv: "LoocvConfig"

    @classmethod
    def from_args(
        cls,
        raw_manifest_path: str,
        processed_dir: str,
        coord_scale_um: float,
        sample_unit: str,
        reducer_scope: str,
        reducer_fit_mode: str,
        keep_raw_molecular: bool,
        force: bool,
        min_cells: int,
        use_molecular_features: bool,
        split_by: str,
        split_ratios: Tuple[float, float, float],
        split_seed: int,
        loocv: Optional[Dict[str, object]] = None,
    ) -> "PrecomputeConfig":
        return cls(
            raw_manifest_path=raw_manifest_path,
            processed_dir=processed_dir,
            coord_scale_um=coord_scale_um,
            sample_unit=sample_unit,
            reducer_scope=reducer_scope,
            reducer_fit_mode=reducer_fit_mode,
            keep_raw_molecular=keep_raw_molecular,
            force=force,
            min_cells=min_cells,
            use_molecular_features=use_molecular_features,
            split_by=split_by,
            split_ratios=split_ratios,
            split_seed=split_seed,
            loocv=LoocvConfig.from_dict(loocv),
        )


@dataclass
class LoocvConfig:
    enabled: bool = False
    fold_unit: str = "region"  # region | sample
    holdout_id: Optional[str] = None
    validation_strategy: str = (
        "heldout_fold_items"  # heldout_fold_items | patches_from_train_items
    )
    val_ratio: float = 0.1
    seed: int = 42

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "LoocvConfig":
        return cls(**(data or {}))


@dataclass
class CategoricalFeatureConfig:
    include_labels: Sequence[str]

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "CategoricalFeatureConfig":
        data = data or {}
        return cls(include_labels=data.get("include_labels", []))


@dataclass
class PreparedSample:
    table_idx: int
    table: SpatialOmicsTable
    sample_id: str
    region_id: Optional[str]
    patch_indices: List[np.ndarray]


@dataclass
class GraphUnitSpec:
    table_idx: int
    sample_id: str
    region_id: Optional[str]
    patch_idx: int
    patch_id: str
    indices: np.ndarray
    split: str


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
        self._manifest_dir: Optional[Path] = None

    def precompute(self) -> Path:
        processed_dir = Path(self.precompute_config.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        index_path = processed_dir / "processed_index.json"
        metadata_path = processed_dir / "metadata.json"

        if index_path.exists() and not self.precompute_config.force:
            return index_path

        manifest_rows = self._load_manifest(
            Path(self.precompute_config.raw_manifest_path)
        )
        raw_tables = self._load_tables(manifest_rows)

        prepared = self._prepare_samples(manifest_rows, raw_tables)
        units = self._build_unit_specs(prepared)
        label_maps = self._build_label_maps([item.table for item in prepared])

        graph_label_maps: Dict[str, Dict[str, int]] = {}
        graph_label_df = self._load_graph_labels()

        reducer_state, reducer = self._prepare_reducer(prepared, units)

        table_embeddings: Dict[int, Optional[np.ndarray]] = {}
        for sample in prepared:
            table_embeddings[sample.table_idx] = self._compute_molecular_embedding(
                table=sample.table,
                reducer=reducer,
                reducer_state=reducer_state,
                sample_id=sample.sample_id,
            )

        entries: List[Dict[str, object]] = []
        for unit in units:
            sample = prepared[unit.table_idx]
            embedded = table_embeddings[unit.table_idx]
            data = self._build_pyg_data(
                table=sample.table,
                indices=unit.indices,
                embedded=embedded[unit.indices] if embedded is not None else None,
                sample_id=unit.sample_id,
                region_id=unit.region_id,
                patch_id=unit.patch_id,
                label_maps=label_maps,
                graph_label_df=graph_label_df,
                graph_label_maps=graph_label_maps,
            )

            graph_path = processed_dir / "graphs" / unit.sample_id
            graph_path.mkdir(parents=True, exist_ok=True)
            data_path = graph_path / f"{unit.patch_id}.pt"
            torch.save(data, data_path)
            entries.append(
                {
                    "path": str(data_path),
                    "sample_id": unit.sample_id,
                    "region_id": unit.region_id,
                    "patch_id": unit.patch_id,
                    "split": unit.split,
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
            "split": {
                "split_by": self.precompute_config.split_by,
                "ratios": list(self.precompute_config.split_ratios),
                "seed": int(self.precompute_config.split_seed),
                "loocv": {
                    "enabled": bool(self.precompute_config.loocv.enabled),
                    "fold_unit": str(self.precompute_config.loocv.fold_unit),
                    "holdout_id": self.precompute_config.loocv.holdout_id,
                    "validation_strategy": str(
                        self.precompute_config.loocv.validation_strategy
                    ),
                    "val_ratio": float(self.precompute_config.loocv.val_ratio),
                    "seed": int(self.precompute_config.loocv.seed),
                },
            },
        }
        index_path.write_text(json.dumps(index_payload, indent=2))

        metadata = self._build_metadata(
            entries=entries,
            tables=[item.table for item in prepared],
            label_maps=label_maps,
            graph_label_maps=graph_label_maps,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return index_path

    def _prepare_samples(
        self,
        manifest_rows: List[Dict[str, str]],
        tables: List[SpatialOmicsTable],
    ) -> List[PreparedSample]:
        prepared: List[PreparedSample] = []
        for table_idx, (row, table) in enumerate(zip(manifest_rows, tables)):
            sample_id = table.sample_id or str(row[self.manifest_config.sample_id])
            region_id = table.region_id or row.get(self.manifest_config.region_id)
            polygons_path = self._resolve_manifest_relative_path(
                row.get(self.manifest_config.polygons_path)
            )

            table = _apply_coord_scale(table, self.precompute_config.coord_scale_um)
            table, polygons = self._apply_polygons(table, polygons_path)
            patch_indices = self._compute_patch_indices(table, polygons)
            prepared.append(
                PreparedSample(
                    table_idx=table_idx,
                    table=table,
                    sample_id=sample_id,
                    region_id=region_id,
                    patch_indices=patch_indices,
                )
            )
        return prepared

    def _build_unit_specs(
        self, prepared: Sequence[PreparedSample]
    ) -> List[GraphUnitSpec]:
        units: List[GraphUnitSpec] = []
        for sample in prepared:
            for patch_idx, indices in enumerate(sample.patch_indices):
                patch_id = _build_patch_id(
                    sample.sample_id, sample.region_id, patch_idx
                )
                units.append(
                    GraphUnitSpec(
                        table_idx=sample.table_idx,
                        sample_id=sample.sample_id,
                        region_id=sample.region_id,
                        patch_idx=patch_idx,
                        patch_id=patch_id,
                        indices=indices,
                        split="train",  # assigned below
                    )
                )

        split_labels = _assign_split_labels(
            units=units,
            split_by=self.precompute_config.split_by,
            split_ratios=self.precompute_config.split_ratios,
            seed=self.precompute_config.split_seed,
            loocv_config=self.precompute_config.loocv,
        )
        for i, split in enumerate(split_labels):
            units[i].split = split
        return units

    def _load_tables(
        self, manifest_rows: List[Dict[str, str]]
    ) -> List[SpatialOmicsTable]:
        return [self._load_table(row) for row in manifest_rows]

    def _prepare_reducer(
        self,
        prepared: Sequence[PreparedSample],
        units: Sequence[GraphUnitSpec],
    ) -> tuple[Dict[str, object], object]:
        reducer_state: Dict[str, object] = {
            "scope": self.precompute_config.reducer_scope,
            "fit_mode": self.precompute_config.reducer_fit_mode,
        }
        reducer = get_feature_reducer(self.feature_reducer_config)

        reducer_state["by_sample"] = {}
        if not self.precompute_config.use_molecular_features:
            return reducer_state, reducer

        if self.precompute_config.reducer_scope != "dataset":
            return reducer_state, reducer

        fit_mode = str(self.precompute_config.reducer_fit_mode).strip().lower()
        if fit_mode not in {"global", "train_only"}:
            raise ValueError(
                "feature_reducer.fit_mode must be one of ['global', 'train_only']."
            )

        if fit_mode == "global":
            features = [
                sample.table.molecular_features
                for sample in prepared
                if sample.table.molecular_features is not None
            ]
            fitted_on = "global"
        else:
            selected_by_table: Dict[int, List[np.ndarray]] = {}
            for unit in units:
                if unit.split != "train":
                    continue
                selected_by_table.setdefault(unit.table_idx, []).append(unit.indices)
            features = []
            for sample in prepared:
                if sample.table.molecular_features is None:
                    continue
                train_indices = selected_by_table.get(sample.table_idx, [])
                if not train_indices:
                    continue
                uniq = np.unique(np.concatenate(train_indices, axis=0))
                if uniq.size == 0:
                    continue
                features.append(sample.table.molecular_features[uniq])
            fitted_on = "train_only"

        if not features:
            raise ValueError(
                "No molecular features available for reducer fitting with the current split and fit_mode."
            )

        reducer.fit(np.concatenate(features, axis=0))
        reducer_state["global"] = reducer.state_dict()
        reducer_state["fitted_on"] = fitted_on
        return reducer_state, reducer

    def _apply_polygons(
        self, table: SpatialOmicsTable, polygons_path: Optional[str]
    ) -> tuple[SpatialOmicsTable, Optional[Sequence[object]]]:
        polygons = load_polygons_from_path(polygons_path) if polygons_path else None
        if polygons:
            union_idx = filter_coords_by_union(table.coords, polygons)
            table = _subset_table(table, union_idx)
        return table, polygons

    def _compute_molecular_embedding(
        self,
        table: SpatialOmicsTable,
        reducer,
        reducer_state: Dict[str, object],
        sample_id: str,
    ) -> Optional[np.ndarray]:
        if not self.precompute_config.use_molecular_features:
            return None
        reducer_instance = reducer
        if self.precompute_config.reducer_scope == "sample":
            reducer_instance = get_feature_reducer(self.feature_reducer_config)
            reducer_instance.fit(table.molecular_features)
            reducer_state["by_sample"][sample_id] = reducer_instance.state_dict()
        return reducer_instance.transform(table.molecular_features)

    def _compute_patch_indices(
        self, table: SpatialOmicsTable, polygons: Optional[Sequence[object]]
    ) -> List[np.ndarray]:
        if polygons and self.precompute_config.sample_unit != "full":
            patch_indices = build_patches_from_polygons(table.coords, polygons)
        elif self.precompute_config.sample_unit == "tile":
            patch_indices = build_grid_tiles(table.coords, self.tile_config)
        else:
            patch_indices = [np.arange(table.coords.shape[0])]
        return [
            idx for idx in patch_indices if idx.size >= self.precompute_config.min_cells
        ]

    def _load_manifest(self, path: Path) -> List[Dict[str, str]]:
        import pandas as pd

        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        self._manifest_dir = path.parent
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
        input_path = self._resolve_manifest_relative_path(input_path)
        input_type = row[self.manifest_config.input_type].lower()

        if input_type in {"csv", "tsv"}:
            cfg = self.csv_config
            if input_type == "tsv":
                cfg = CsvConfig(
                    coord_columns=cfg.coord_columns,
                    cell_id_column=cfg.cell_id_column,
                    categorical_label_columns=cfg.categorical_label_columns,
                    molecular_columns=cfg.molecular_columns,
                    use_molecular_features=cfg.use_molecular_features,
                    sep="\t",
                )
            return load_csv_table(
                input_path, cfg, sample_id=sample_id, region_id=region_id
            )
        if input_type == "h5ad":
            return load_h5ad_table(
                input_path, self.h5ad_config, sample_id=sample_id, region_id=region_id
            )
        if input_type in {"sce", "rds"}:
            return load_sce_table(
                input_path, self.sce_config, sample_id=sample_id, region_id=region_id
            )
        raise ValueError(f"Unsupported input type: {input_type}")

    def _resolve_manifest_relative_path(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        # handle pandas NaN and common empty sentinels
        try:
            import pandas as pd

            if pd.isna(value):
                return None
        except Exception:
            pass

        value = str(value).strip()
        if value == "":
            return None
        if value.lower() in {"nan", "none", "null"}:
            return None
        p = Path(value)
        if p.is_absolute():
            return str(p)
        if self._manifest_dir is None:
            return str(p)
        return str((self._manifest_dir / p).resolve())

    def _build_label_maps(
        self, tables: Sequence[SpatialOmicsTable]
    ) -> Dict[str, Dict[str, int]]:
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
        graph_result: GraphBuildResult = get_graph_builder(
            self.graph_builder_config
        ).build(coords)

        data_kwargs = {
            "pos": torch.from_numpy(coords).float(),
            "edge_index": graph_result.edge_index,
        }
        if embedded is not None:
            data_kwargs["x"] = torch.from_numpy(embedded).float()
        else:
            data_kwargs["x"] = torch.zeros((coords.shape[0], 0), dtype=torch.float)
        data = Data(**data_kwargs)
        if graph_result.edge_attr is not None:
            data.edge_attr = graph_result.edge_attr.float()
            data.edge_attr_names = graph_result.edge_attr_names

        if (
            self.precompute_config.keep_raw_molecular
            and table.molecular_features is not None
        ):
            data.x_raw = torch.from_numpy(table.molecular_features[indices]).float()

        self._attach_categorical_indices(
            data, table.categorical_labels, label_maps, indices
        )

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
                data.graph_y = torch.tensor(graph_values, dtype=torch.float).unsqueeze(
                    0
                )
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
            encoded = np.array(
                [mapping[val] for val in labels[label_name][indices]], dtype=np.int64
            )
            encoded_cols.append(encoded)
            slices[label_name] = col_idx

        categorical_index = np.stack(encoded_cols, axis=1)
        data.categorical_index = torch.from_numpy(categorical_index).long()
        data.categorical_labels = list(config.include_labels)
        data.categorical_slices = slices

    def _build_metadata(
        self,
        entries: List[Dict[str, object]],
        tables: List[SpatialOmicsTable],
        label_maps: Dict[str, Dict[str, int]],
        graph_label_maps: Dict[str, Dict[str, int]],
    ) -> Dict[str, object]:
        molecular_dim = 0
        molecular_feature_names = None
        if self.precompute_config.use_molecular_features:
            for table in tables:
                if table.molecular_features is not None:
                    molecular_dim = int(table.molecular_features.shape[1])
                    molecular_feature_names = (
                        list(table.molecular_feature_names)
                        if table.molecular_feature_names is not None
                        else None
                    )
                    break
        categorical_cardinalities = {
            label: len(mapping) for label, mapping in label_maps.items()
        }
        return {
            "num_graphs": len(entries),
            "num_samples": len({entry["sample_id"] for entry in entries}),
            "molecular_feature_dim": molecular_dim,
            "molecular_feature_names": molecular_feature_names,
            "categorical_labels": list(self.categorical_feature_config.include_labels),
            "categorical_cardinalities": categorical_cardinalities,
            "label_maps": label_maps,
            "graph_label_maps": graph_label_maps,
            "use_molecular_features": self.precompute_config.use_molecular_features,
            "reducer_scope": self.precompute_config.reducer_scope,
            "reducer_fit_mode": self.precompute_config.reducer_fit_mode,
            "sample_unit": self.precompute_config.sample_unit,
            "split_by": self.precompute_config.split_by,
            "split_ratios": list(self.precompute_config.split_ratios),
            "split_seed": int(self.precompute_config.split_seed),
        }


def _apply_coord_scale(
    table: SpatialOmicsTable, coord_scale_um: float
) -> SpatialOmicsTable:
    if coord_scale_um == 1.0:
        return table
    scaled = table.coords * coord_scale_um
    return SpatialOmicsTable(
        coords=scaled,
        molecular_features=table.molecular_features,
        molecular_feature_names=table.molecular_feature_names,
        categorical_labels=table.categorical_labels,
        cell_ids=table.cell_ids,
        sample_id=table.sample_id,
        region_id=table.region_id,
    )


def _subset_table(table: SpatialOmicsTable, indices: np.ndarray) -> SpatialOmicsTable:
    return SpatialOmicsTable(
        coords=table.coords[indices],
        molecular_features=(
            None
            if table.molecular_features is None
            else table.molecular_features[indices]
        ),
        molecular_feature_names=table.molecular_feature_names,
        categorical_labels={k: v[indices] for k, v in table.categorical_labels.items()},
        cell_ids=table.cell_ids[indices],
        sample_id=table.sample_id,
        region_id=table.region_id,
    )


def _split_indices(
    total: int,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    train_r, val_r, test_r = ratios
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError("train_val_test_split must sum to 1.0")
    if total <= 0:
        return [], [], []

    ratio_arr = np.asarray([train_r, val_r, test_r], dtype=float)
    raw_counts = ratio_arr * float(total)
    counts = np.floor(raw_counts).astype(int)

    remainder = int(total - counts.sum())
    if remainder > 0:
        fractions = raw_counts - counts
        fractions[ratio_arr <= 0.0] = -1.0
        order = np.argsort(-fractions)
        ptr = 0
        while remainder > 0 and ptr < len(order):
            idx = int(order[ptr])
            if fractions[idx] < 0:
                break
            counts[idx] += 1
            remainder -= 1
            ptr += 1
        while remainder > 0:
            counts[0] += 1
            remainder -= 1

    if ratio_arr[0] > 0.0 and counts[0] == 0:
        donor_idx = int(np.argmax(counts[1:]) + 1) if counts[1:].sum() > 0 else -1
        if donor_idx >= 0 and counts[donor_idx] > 0:
            counts[donor_idx] -= 1
            counts[0] += 1
        else:
            counts[0] = 1
            overflow = int(counts.sum() - total)
            for idx in (2, 1):
                if overflow <= 0:
                    break
                take = min(overflow, counts[idx])
                counts[idx] -= take
                overflow -= take

    # Keep val split non-empty whenever val ratio is non-zero.
    if ratio_arr[1] > 0.0 and counts[1] == 0 and total > 0:
        donor_idx = -1
        if counts[2] > 0:
            donor_idx = 2
        elif counts[0] > 1 or ratio_arr[0] <= 0.0:
            donor_idx = 0
        elif counts[0] > 0:
            # Tiny totals can make non-empty train and val impossible.
            donor_idx = 0
        if donor_idx >= 0 and counts[donor_idx] > 0:
            counts[donor_idx] -= 1
            counts[1] += 1

    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(counts[0])
    val_end = train_end + int(counts[1])
    return (
        indices[:train_end].tolist(),
        indices[train_end:val_end].tolist(),
        indices[val_end:].tolist(),
    )


def _assign_split_labels(
    units: Sequence[GraphUnitSpec],
    split_by: str,
    split_ratios: Tuple[float, float, float],
    seed: int,
    loocv_config: Optional[LoocvConfig] = None,
) -> List[str]:
    labels = ["train"] * len(units)
    if not units:
        return labels

    loocv = loocv_config or LoocvConfig()
    if loocv.enabled:
        return _assign_loocv_split_labels(units, loocv)

    if split_by == "sample":
        by_sample: Dict[str, List[int]] = {}
        for idx, unit in enumerate(units):
            by_sample.setdefault(unit.sample_id, []).append(idx)
        sample_keys = list(by_sample)
        train_idx, val_idx, test_idx = _split_indices(
            total=len(sample_keys), ratios=split_ratios, seed=seed
        )
        for i in train_idx:
            for unit_idx in by_sample[sample_keys[i]]:
                labels[unit_idx] = "train"
        for i in val_idx:
            for unit_idx in by_sample[sample_keys[i]]:
                labels[unit_idx] = "val"
        for i in test_idx:
            for unit_idx in by_sample[sample_keys[i]]:
                labels[unit_idx] = "test"
        return labels

    if split_by == "region":
        by_region: Dict[Tuple[str, Optional[str]], List[int]] = {}
        for idx, unit in enumerate(units):
            by_region.setdefault((unit.sample_id, unit.region_id), []).append(idx)
        region_keys = list(by_region)
        train_idx, val_idx, test_idx = _split_indices(
            total=len(region_keys), ratios=split_ratios, seed=seed
        )
        for i in train_idx:
            for unit_idx in by_region[region_keys[i]]:
                labels[unit_idx] = "train"
        for i in val_idx:
            for unit_idx in by_region[region_keys[i]]:
                labels[unit_idx] = "val"
        for i in test_idx:
            for unit_idx in by_region[region_keys[i]]:
                labels[unit_idx] = "test"
        return labels

    if split_by != "patch":
        raise ValueError("split_by must be one of ['sample', 'region', 'patch']")

    train_idx, val_idx, test_idx = _split_indices(
        total=len(units), ratios=split_ratios, seed=seed
    )
    for i in train_idx:
        labels[i] = "train"
    for i in val_idx:
        labels[i] = "val"
    for i in test_idx:
        labels[i] = "test"
    return labels


def _assign_loocv_split_labels(
    units: Sequence[GraphUnitSpec], loocv: LoocvConfig
) -> List[str]:
    if loocv.fold_unit not in {"region", "sample"}:
        raise ValueError("split.loocv.fold_unit must be one of ['region', 'sample'].")
    if loocv.validation_strategy not in {
        "heldout_fold_items",
        "patches_from_train_items",
    }:
        raise ValueError(
            "split.loocv.validation_strategy must be one of "
            "['heldout_fold_items', 'patches_from_train_items']."
        )
    if loocv.val_ratio < 0.0 or loocv.val_ratio >= 1.0:
        raise ValueError("split.loocv.val_ratio must be in [0.0, 1.0).")
    if not loocv.holdout_id:
        raise ValueError(
            "split.loocv.holdout_id must be set when split.loocv.enabled=true."
        )

    by_fold_item: Dict[object, List[int]] = {}
    fold_alias_to_key: Dict[str, object] = {}
    ambiguous_region_aliases = set()

    for idx, unit in enumerate(units):
        key = (
            (unit.sample_id, unit.region_id)
            if loocv.fold_unit == "region"
            else unit.sample_id
        )
        by_fold_item.setdefault(key, []).append(idx)

        canonical = _loocv_fold_id_from_key(key=key, fold_unit=loocv.fold_unit)
        fold_alias_to_key[canonical] = key
        if loocv.fold_unit == "region" and unit.region_id is not None:
            alias = str(unit.region_id)
            existing = fold_alias_to_key.get(alias)
            if existing is None:
                fold_alias_to_key[alias] = key
            elif existing != key:
                ambiguous_region_aliases.add(alias)

    for alias in ambiguous_region_aliases:
        if alias in fold_alias_to_key:
            del fold_alias_to_key[alias]

    if len(by_fold_item) < 2:
        raise ValueError(
            "LOOCV requires at least two fold items after grouping by "
            f"'{loocv.fold_unit}'. Found {len(by_fold_item)}."
        )
    if loocv.holdout_id not in fold_alias_to_key:
        available = sorted(
            _loocv_fold_id_from_key(key=key, fold_unit=loocv.fold_unit)
            for key in by_fold_item
        )
        raise ValueError(
            "split.loocv.holdout_id not found in grouped fold items. "
            f"Provided '{loocv.holdout_id}'. Available canonical ids: {available}"
        )

    holdout_key = fold_alias_to_key[loocv.holdout_id]
    labels = ["train"] * len(units)
    for unit_idx in by_fold_item[holdout_key]:
        labels[unit_idx] = "test"

    remaining_keys = [key for key in by_fold_item if key != holdout_key]
    if not remaining_keys:
        raise ValueError("LOOCV left no non-test fold items.")

    if loocv.validation_strategy == "heldout_fold_items":
        val_group_indices = _pick_validation_fold_items(
            total=len(remaining_keys),
            val_ratio=float(loocv.val_ratio),
            seed=int(loocv.seed),
        )
        for group_idx in val_group_indices:
            for unit_idx in by_fold_item[remaining_keys[group_idx]]:
                labels[unit_idx] = "val"
        return labels

    warnings.warn(
        "Using patch-level validation from training fold items. "
        "This is useful in low-data settings, but validation is less independent "
        "than held-out fold-item validation.",
        stacklevel=2,
    )
    candidate_indices = [
        unit_idx for key in remaining_keys for unit_idx in by_fold_item[key]
    ]
    train_idx, val_idx, _ = _split_indices(
        total=len(candidate_indices),
        ratios=(1.0 - float(loocv.val_ratio), float(loocv.val_ratio), 0.0),
        seed=int(loocv.seed),
    )
    for local_idx in train_idx:
        labels[candidate_indices[local_idx]] = "train"
    for local_idx in val_idx:
        labels[candidate_indices[local_idx]] = "val"
    return labels


def _pick_validation_fold_items(total: int, val_ratio: float, seed: int) -> List[int]:
    if total <= 1:
        raise ValueError(
            "LOOCV validation strategy 'heldout_fold_items' requires at least two "
            "non-test fold items. Use 'patches_from_train_items' for tiny-data regimes."
        )
    _, val_idx, _ = _split_indices(
        total=total,
        ratios=(1.0 - float(val_ratio), float(val_ratio), 0.0),
        seed=seed,
    )
    if val_idx:
        return val_idx
    if float(val_ratio) == 0.0:
        return []
    return [int(np.random.default_rng(seed).integers(0, total))]


def _loocv_fold_id_from_key(key: object, fold_unit: str) -> str:
    if fold_unit == "sample":
        return str(key)
    sample_id, region_id = key
    region_value = "__NONE__" if region_id is None else str(region_id)
    return f"{sample_id}::{region_value}"


def discover_viable_loocv_fold_ids(
    preprocessor: SpatialOmicsPreprocessor, fold_unit: str
) -> List[str]:
    if fold_unit not in {"region", "sample"}:
        raise ValueError("split.loocv.fold_unit must be one of ['region', 'sample'].")

    manifest_rows = preprocessor._load_manifest(
        Path(preprocessor.precompute_config.raw_manifest_path)
    )
    tables = preprocessor._load_tables(manifest_rows)
    prepared = preprocessor._prepare_samples(manifest_rows, tables)

    fold_keys: Set[object] = set()
    for sample in prepared:
        if not sample.patch_indices:
            continue
        key = (
            (sample.sample_id, sample.region_id)
            if fold_unit == "region"
            else sample.sample_id
        )
        fold_keys.add(key)

    return sorted(
        _loocv_fold_id_from_key(key=key, fold_unit=fold_unit) for key in fold_keys
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
