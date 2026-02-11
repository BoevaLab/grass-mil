from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightning as L

from .components.datasets import SpatialOmicsGraphDataset, TransformDataset
from .components.feature_reducers import FeatureReducerConfig
from .components.graph_builders import GraphBuilderConfig
from .components.loaders import CsvConfig, H5adConfig, SceConfig
from .components.patching import TileConfig
from .components.precompute import (
    CategoricalFeatureConfig,
    GraphLabelConfig,
    ManifestConfig,
    PrecomputeConfig,
    SpatialOmicsPreprocessor,
)
from .components.samplers import (
    BaseSamplerStrategy,
    SamplerConfig,
    get_sampler_strategy,
)
from .components.transforms import instantiate_transforms


@dataclass
class SplitConfig:
    train_val_test_split: Tuple[float, float, float]
    split_by: str


class SpatialOmicsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        raw_manifest_path: str,
        processed_dir: str,
        coord_scale_um: float,
        sample_unit: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        reducer_scope: str,
        keep_raw_molecular: bool,
        force_precompute: bool,
        min_cells: int,
        use_molecular_features: bool,
        categorical_features: Dict[str, Any],
        split: Dict[str, Any],
        csv: Dict[str, Any],
        h5ad: Dict[str, Any],
        sce: Dict[str, Any],
        graph_builder: Dict[str, Any],
        feature_reducer: Dict[str, Any],
        tiling: Dict[str, Any],
        sampler: Optional[Dict[str, Any]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        graph_labels: Optional[Dict[str, Any]] = None,
        transforms: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.raw_manifest_path = raw_manifest_path
        self.processed_dir = processed_dir

        self.dataset_train: Optional[SpatialOmicsGraphDataset] = None
        self.dataset_val: Optional[SpatialOmicsGraphDataset] = None
        self.dataset_test: Optional[SpatialOmicsGraphDataset] = None

        self.split_config = SplitConfig(
            train_val_test_split=tuple(split["train_val_test_split"]),
            split_by=split["split_by"],
        )
        self.csv_config = CsvConfig.from_dict(csv, use_molecular_features)
        self.h5ad_config = H5adConfig.from_dict(h5ad, use_molecular_features)
        self.sce_config = SceConfig.from_dict(sce, use_molecular_features)
        self.graph_builder_config = GraphBuilderConfig.from_dict(graph_builder)
        self.feature_reducer_config = FeatureReducerConfig.from_dict(feature_reducer)
        self.tile_config = TileConfig.from_dict(tiling)
        self.sampler_config = SamplerConfig.from_dict(sampler)
        self.sampler_strategy: Optional[BaseSamplerStrategy] = None
        self.manifest_config = ManifestConfig.from_dict(manifest)
        self.graph_label_config = GraphLabelConfig.from_dict(graph_labels)
        self.transforms = instantiate_transforms(transforms)

        self.precompute_config = PrecomputeConfig.from_args(
            raw_manifest_path=self.raw_manifest_path,
            processed_dir=self.processed_dir,
            coord_scale_um=coord_scale_um,
            sample_unit=sample_unit,
            reducer_scope=reducer_scope,
            keep_raw_molecular=keep_raw_molecular,
            force=force_precompute,
            min_cells=min_cells,
            use_molecular_features=use_molecular_features,
        )
        self.categorical_feature_config = CategoricalFeatureConfig.from_dict(
            categorical_features
        )

    def prepare_data(self) -> None:
        preprocessor = self._build_preprocessor()
        preprocessor.precompute()

    def _build_preprocessor(self) -> SpatialOmicsPreprocessor:
        return SpatialOmicsPreprocessor(
            manifest_config=self.manifest_config,
            precompute_config=self.precompute_config,
            csv_config=self.csv_config,
            h5ad_config=self.h5ad_config,
            sce_config=self.sce_config,
            graph_builder_config=self.graph_builder_config,
            feature_reducer_config=self.feature_reducer_config,
            categorical_feature_config=self.categorical_feature_config,
            tile_config=self.tile_config,
            graph_label_config=self.graph_label_config,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        index_path = Path(self.processed_dir) / "processed_index.json"
        base_dataset = SpatialOmicsGraphDataset(index_path)

        if self.split_config.split_by == "patch":
            entries = base_dataset.entries
        else:
            entries = _group_entries_by_sample(base_dataset.entries)

        train_idx, val_idx, test_idx = _split_indices(
            len(entries),
            self.split_config.train_val_test_split,
        )

        self.dataset_train = _subset_dataset(
            base_dataset, entries, train_idx, self.transforms
        )
        self.dataset_val = _subset_dataset(
            base_dataset, entries, val_idx, self.transforms
        )
        self.dataset_test = _subset_dataset(
            base_dataset, entries, test_idx, self.transforms
        )
        self._ensure_sampler_strategy()

    def _ensure_sampler_strategy(self) -> BaseSamplerStrategy:
        if self.sampler_strategy is None:
            self.sampler_strategy = get_sampler_strategy(self.sampler_config)
        return self.sampler_strategy

    def train_dataloader(self) -> Any:
        sampler = self._ensure_sampler_strategy()
        return sampler.build_dataset_loader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> Any:
        sampler = self._ensure_sampler_strategy()
        return sampler.build_dataset_loader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> Any:
        sampler = self._ensure_sampler_strategy()
        return sampler.build_dataset_loader(
            dataset=self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


def _split_indices(
    total: int, ratios: Tuple[float, float, float]
) -> Tuple[list[int], list[int], list[int]]:
    import numpy as np

    train_r, val_r, test_r = ratios
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError("train_val_test_split must sum to 1.0")
    indices = np.arange(total)
    np.random.shuffle(indices)
    train_end = int(total * train_r)
    val_end = train_end + int(total * val_r)
    return (
        indices[:train_end].tolist(),
        indices[train_end:val_end].tolist(),
        indices[val_end:].tolist(),
    )


def _group_entries_by_sample(entries):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.sample_id, []).append(entry)
    return list(grouped.values())


def _subset_dataset(
    dataset: SpatialOmicsGraphDataset, entries, indices, transforms
) -> SpatialOmicsGraphDataset:
    if not entries:
        return dataset
    if isinstance(entries[0], list):
        selected_entries = [e for i in indices for e in entries[i]]
    else:
        selected_entries = [entries[i] for i in indices]

    subset = SpatialOmicsGraphDataset(dataset.index_path)
    subset.entries = selected_entries
    subset.label_maps = dataset.label_maps
    subset.reducer_state = dataset.reducer_state
    subset.graph_label_maps = dataset.graph_label_maps
    if transforms:
        return TransformDataset(subset, transforms)
    return subset
