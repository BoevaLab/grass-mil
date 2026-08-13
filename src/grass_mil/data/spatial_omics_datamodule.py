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
    LoocvConfig,
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
    loocv: LoocvConfig


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
        val_sampler: Optional[Dict[str, Any]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        graph_labels: Optional[Dict[str, Any]] = None,
        transforms: Optional[list[Dict[str, Any]]] = None,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.raw_manifest_path = raw_manifest_path
        self.processed_dir = processed_dir

        self.dataset_train: Optional[SpatialOmicsGraphDataset] = None
        self.dataset_val: Optional[SpatialOmicsGraphDataset] = None
        self.dataset_test: Optional[SpatialOmicsGraphDataset] = None
        self.split_seed = int(split_seed)

        self.split_config = SplitConfig(
            train_val_test_split=tuple(split["train_val_test_split"]),
            split_by=split["split_by"],
            loocv=LoocvConfig.from_dict(split.get("loocv")),
        )
        self.csv_config = CsvConfig.from_dict(csv, use_molecular_features)
        self.h5ad_config = H5adConfig.from_dict(h5ad, use_molecular_features)
        self.sce_config = SceConfig.from_dict(sce, use_molecular_features)
        self.graph_builder_config = GraphBuilderConfig.from_dict(graph_builder)
        self.feature_reducer_config = FeatureReducerConfig.from_dict(feature_reducer)
        self.tile_config = TileConfig.from_dict(tiling)
        self.sampler_config = SamplerConfig.from_dict(sampler)
        self.val_sampler_config = (
            SamplerConfig.from_dict(val_sampler) if val_sampler is not None else None
        )
        self.sampler_strategy: Optional[BaseSamplerStrategy] = None
        self.val_sampler_strategy: Optional[BaseSamplerStrategy] = None
        self.manifest_config = ManifestConfig.from_dict(manifest)
        self.graph_label_config = GraphLabelConfig.from_dict(graph_labels)
        self.transforms = instantiate_transforms(transforms)

        self.precompute_config = PrecomputeConfig.from_args(
            raw_manifest_path=self.raw_manifest_path,
            processed_dir=self.processed_dir,
            coord_scale_um=coord_scale_um,
            sample_unit=sample_unit,
            reducer_scope=reducer_scope,
            reducer_fit_mode=str(feature_reducer.get("fit_mode", "train_only")),
            keep_raw_molecular=keep_raw_molecular,
            force=force_precompute,
            min_cells=min_cells,
            use_molecular_features=use_molecular_features,
            split_by=self.split_config.split_by,
            split_ratios=self.split_config.train_val_test_split,
            split_seed=self.split_seed,
            loocv=split.get("loocv"),
        )
        self.categorical_feature_config = CategoricalFeatureConfig.from_dict(
            categorical_features
        )

    def prepare_data(self) -> None:
        if (
            self.split_config.loocv.enabled
            and self.hparams.reducer_scope == "dataset"
            and str(self.hparams.feature_reducer.get("fit_mode", "train_only"))
            .strip()
            .lower()
            == "train_only"
        ):
            processed_index_path = Path(self.processed_dir) / "processed_index.json"
            if (
                not bool(self.hparams.force_precompute)
                and processed_index_path.exists()
            ):
                raise ValueError(
                    "LOOCV with reducer_scope='dataset' and "
                    "feature_reducer.fit_mode='train_only' cannot reuse "
                    "an existing "
                    "processed_index when data.force_precompute=false. Set "
                    "data.force_precompute=true or use a fresh fold-specific "
                    "processed_dir."
                )
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

        if not _entries_have_persisted_splits(base_dataset.entries):
            raise ValueError(
                "processed_index.json must contain valid per-entry split labels "
                "('train', 'val', 'test'). Run data precompute/prepare_data() "
                "to regenerate processed artifacts."
            )

        self.dataset_train = _subset_dataset_from_entries(
            base_dataset,
            [entry for entry in base_dataset.entries if entry.split == "train"],
            self.transforms,
        )
        self.dataset_val = _subset_dataset_from_entries(
            base_dataset,
            [entry for entry in base_dataset.entries if entry.split == "val"],
            self.transforms,
        )
        self.dataset_test = _subset_dataset_from_entries(
            base_dataset,
            [entry for entry in base_dataset.entries if entry.split == "test"],
            self.transforms,
        )
        self._ensure_sampler_strategy()
        self._ensure_val_sampler_strategy()

    def _ensure_sampler_strategy(self) -> BaseSamplerStrategy:
        if self.sampler_strategy is None:
            self.sampler_strategy = get_sampler_strategy(self.sampler_config)
        return self.sampler_strategy

    def _ensure_val_sampler_strategy(self) -> BaseSamplerStrategy:
        if self.val_sampler_config is None:
            return self._ensure_sampler_strategy()
        if self.val_sampler_strategy is None:
            self.val_sampler_strategy = get_sampler_strategy(self.val_sampler_config)
        return self.val_sampler_strategy

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
        sampler = self._ensure_val_sampler_strategy()
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

    def predict_dataloader(self) -> Any:
        """Inference loader, over the test split.

        Required by `grass-mil-predict`: `Trainer.predict` raises
        `MisconfigurationException` without it. Predicting on the held-out split
        mirrors `test_dataloader` so exported tables line up with test metrics.
        To score a different set of graphs, point `data.processed_dir` at a
        precompute whose entries carry the split you want.
        """
        return self.test_dataloader()


def _subset_dataset_from_entries(
    dataset: SpatialOmicsGraphDataset,
    selected_entries,
    transforms,
) -> SpatialOmicsGraphDataset:
    subset = SpatialOmicsGraphDataset(dataset.index_path)
    subset.entries = list(selected_entries)
    subset.label_maps = dataset.label_maps
    subset.reducer_state = dataset.reducer_state
    subset.graph_label_maps = dataset.graph_label_maps
    if transforms:
        return TransformDataset(subset, transforms)
    return subset


def _entries_have_persisted_splits(entries) -> bool:
    if not entries:
        return True
    valid = {"train", "val", "test"}
    return all(getattr(entry, "split", None) in valid for entry in entries)
