"""Lightning datamodule: processed graph units, labels, splits and ego-graph loaders."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import lightning as L
import torch
from torch.utils.data import Dataset

from grass_mil.utils import plain

from .graphs import GraphConfig
from .labels import LabelTable
from .loaders import InputConfig
from .precompute import BuildConfig, ProcessedIndex, UnitEntry, build_graphs
from .reducers import ReducerConfig
from .samplers import EgoGraphLoader, SamplerConfig
from .splits import random_split, read_ids
from .units import TilingConfig

__all__ = ["SpatialOmicsDataModule", "GraphUnitDataset", "InputSpec"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputSpec:
    """The node and edge inputs a model is built for; it scores only data with the same spec.

    ``preprocessing`` fingerprints the graph settings and the feature basis
    (PCA), which the names alone do not identify.
    """

    feature_names: tuple[str, ...]
    cell_types: tuple[str, ...]
    edge_features: tuple[str, ...]
    preprocessing: str = ""

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def edge_dim(self) -> int:
        return len(self.edge_features)

    def as_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> InputSpec:
        return cls(
            tuple(d["feature_names"]),
            tuple(d["cell_types"]),
            tuple(d["edge_features"]),
            str(d.get("preprocessing", "")),
        )

    def check_matches(self, other: InputSpec, what: str) -> None:
        """Raise when ``other`` (the data) differs from this spec (the model)."""
        differing = [f.name for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)]
        if differing:
            raise ValueError(
                f"{what} was built for other inputs than this cohort's ({', '.join(differing)} "
                "differ). Use the model's cell types, features, graph settings and, for PCA "
                "features, data.feature_reducer.state_from."
            )


class GraphUnitDataset(Dataset):
    """Graph units with their ids and, when labelled, a ``graph_y`` target row."""

    def __init__(self, entries: Sequence[UnitEntry], labels: LabelTable | None = None) -> None:
        self.entries = list(entries)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.entries)

    def label_id(self, entry: UnitEntry) -> str | None:
        if self.labels is None:
            return None
        return self.labels.resolve(entry.unit_id, entry.region_id, entry.sample_id)

    def load(self, idx: int) -> dict[str, Any]:
        # The files are written by this package's own precompute step and hold
        # PyG objects, which torch.load(weights_only=True) cannot unpickle.
        return torch.load(self.entries[idx].path, weights_only=False)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        data = self.load(idx)["data"]
        data.unit_id = entry.unit_id
        data.region_id = entry.region_id
        data.sample_id = entry.sample_id
        label_id = self.label_id(entry)
        if label_id is not None:
            data.graph_y = torch.from_numpy(self.labels.target(label_id)).view(1, -1)
        return data

    def select(self, ids: Sequence[str] | None) -> GraphUnitDataset:
        """The units matching any of ``ids`` (unit, region or sample ids); all for ``None``."""
        if ids is None:
            return self
        wanted = set(map(str, ids))
        return GraphUnitDataset([e for e in self.entries if e.ids & wanted], self.labels)


class SpatialOmicsDataModule(L.LightningDataModule):
    """Cell graphs of a cohort, cut into units (regions, tiles or polygons).

    Splits are drawn when the data are set up, not when graphs are built, so
    one set of processed graphs serves pretraining, every cross-validation fold
    and prediction. With explicit ``split.train/val/test`` lists (unit, region
    or sample ids, or files listing them) only the listed units are used;
    otherwise units are split at random by ``split.ratios``, grouped by patient
    (or region when the label file has no patient column).
    """

    def __init__(
        self,
        manifest_path: str,
        processed_dir: str,
        unit: str = "region",
        coord_scale_um: float = 1.0,
        min_cells: int = 10,
        tiling: Mapping[str, Any] | None = None,
        input: Mapping[str, Any] | None = None,
        graph: Mapping[str, Any] | None = None,
        feature_reducer: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
        labelled_only: bool = True,
        split: Mapping[str, Any] | None = None,
        sampler: Mapping[str, Any] | None = None,
        val_roots_per_unit: int | None = 1024,
        num_workers: int = 0,
        pin_memory: bool = False,
        force_precompute: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {k: plain(v) for k, v in locals().items() if k not in ("self", "__class__")},
            logger=False,
        )
        self.build_cfg = BuildConfig(
            unit=str(unit),
            coord_scale_um=float(coord_scale_um),
            min_cells=int(min_cells),
            tiling=TilingConfig.from_dict(tiling),
            input=InputConfig.from_dict(input),
            graph=GraphConfig.from_dict(graph),
            feature_reducer=ReducerConfig.from_dict(feature_reducer),
        )
        self.manifest_path = str(manifest_path)
        self.processed_dir = str(processed_dir)
        self.labels_cfg = dict(labels or {})
        self.labelled_only = bool(labelled_only)
        self.split_cfg = dict(split or {})
        self.sampler_cfg = SamplerConfig.from_dict(sampler)
        self.val_roots_per_unit = val_roots_per_unit
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.force_precompute = bool(force_precompute)
        unknown = set(self.split_cfg) - {"ratios", "seed", "train", "val", "test"}
        if unknown:
            raise ValueError(f"Unknown data.split keys: {sorted(unknown)}")
        if self.build_cfg.feature_reducer.fit_on == "train" and self._explicit():
            raise ValueError(
                "feature_reducer.fit_on=train cannot be combined with explicit splits (as used "
                "by cross-validation): every fold would need its own graphs. Use fit_on=all."
            )

        self.index: ProcessedIndex | None = None
        self.labels: LabelTable | None = None
        self.dataset_train: GraphUnitDataset | None = None
        self.dataset_val: GraphUnitDataset | None = None
        self.dataset_test: GraphUnitDataset | None = None

    # ----------------------------------------------------------------- setup
    def _explicit(self) -> bool:
        return any(self.split_cfg.get(k) is not None for k in ("train", "val", "test"))

    def read_labels(self) -> LabelTable | None:
        if self.labels is None and self.labels_cfg.get("file"):
            columns = self.labels_cfg.get("columns")
            self.labels = LabelTable.read(
                self.labels_cfg["file"],
                id_column=str(self.labels_cfg.get("id_column", "region_id")),
                target_columns=list(columns) if columns else None,
                patient_column=self.labels_cfg.get("patient_column"),
            )
        return self.labels

    def prepare_data(self) -> None:
        fit_on_train = self.build_cfg.feature_reducer.fit_on == "train"
        build_graphs(
            self.manifest_path,
            self.processed_dir,
            self.build_cfg,
            force=self.force_precompute,
            # The same units and split as setup() draws, so PCA sees training cells only.
            training_units=(
                (lambda units: set(self._split(self._usable(units))["train"]))
                if fit_on_train
                else None
            ),
            split_key=repr(sorted(self.split_cfg.items())) if fit_on_train else "",
        )
        # A forced rebuild happens once per process, not once per setup().
        self.force_precompute = False

    def _usable(self, units: Sequence[UnitEntry]) -> list[UnitEntry]:
        """The units a run uses: every unit, or only labelled ones with ``labelled_only``."""
        labels = self.read_labels()
        if not self.labelled_only or labels is None:
            return list(units)
        labelled = [u for u in units if labels.resolve(u.unit_id, u.region_id, u.sample_id)]
        if len(labelled) < len(units):
            log.warning(f"{len(units) - len(labelled)} unit(s) have no label and are skipped.")
        return labelled

    def group_of(self, entry: UnitEntry) -> str:
        """The unit's patient, or its region when the labels name no patient."""
        labels = self.read_labels()
        label_id = (
            labels.resolve(entry.unit_id, entry.region_id, entry.sample_id) if labels else None
        )
        if label_id is not None and self.labels_cfg.get("patient_column"):
            return labels.patient(label_id)
        return entry.region_id

    def _strata(self, units: Sequence[UnitEntry], groups: Mapping[str, str]) -> dict | None:
        """The class of every group, when each group has one integer class (at most 10)."""
        labels = self.read_labels()
        if labels is None:
            return None
        values: dict[str, set[float]] = {}
        for u in units:
            label_id = labels.resolve(u.unit_id, u.region_id, u.sample_id)
            value = float(labels.targets.loc[label_id].iloc[0]) if label_id else float("nan")
            values.setdefault(groups[u.unit_id], set()).add(value)
        classes = {g: next(iter(v)) for g, v in values.items() if len(v) == 1}
        if len(classes) < len(values) or not all(float(c).is_integer() for c in classes.values()):
            return None
        return {g: str(c) for g, c in classes.items()} if len(set(classes.values())) <= 10 else None

    def _split(self, units: Sequence[UnitEntry]) -> dict[str, list[str]]:
        """Unit ids of the train, val and test splits."""
        if self._explicit():
            out, seen = {}, {}
            for split in ("train", "val", "test"):
                wanted = set(read_ids(self.split_cfg.get(split)) or [])
                members = [u for u in units if u.ids & wanted]
                missing = sorted(wanted - {i for u in members for i in u.ids})
                if missing:
                    raise ValueError(
                        f"data.split.{split} names {len(missing)} id(s) with no processed unit "
                        f"(first: {missing[:5]})."
                    )
                for u in members:
                    if u.unit_id in seen:
                        raise ValueError(
                            f"Unit {u.unit_id} is in both data.split.{seen[u.unit_id]} and "
                            f"data.split.{split}."
                        )
                    seen[u.unit_id] = split
                out[split] = [u.unit_id for u in members]
            return out
        groups = {u.unit_id: self.group_of(u) for u in units}
        assignment = random_split(
            groups,
            self.split_cfg.get("ratios", (0.8, 0.1, 0.1)),
            int(self.split_cfg.get("seed", 0)),
            self._strata(units, groups),
        )
        return {s: [u for u, a in assignment.items() if a == s] for s in ("train", "val", "test")}

    def setup(self, stage: str | None = None) -> None:
        self.index = ProcessedIndex.read(self.processed_dir)
        labels = self.read_labels()
        units = self._usable(self.index.units)
        by_id = {u.unit_id: u for u in units}
        split = self._split(units)
        self.dataset_train = GraphUnitDataset([by_id[u] for u in split["train"]], labels)
        self.dataset_val = GraphUnitDataset([by_id[u] for u in split["val"]], labels)
        self.dataset_test = GraphUnitDataset([by_id[u] for u in split["test"]], labels)

    def all_units(self) -> GraphUnitDataset:
        """Every processed unit (labelled or not), regardless of the split."""
        if self.index is None:
            self.index = ProcessedIndex.read(self.processed_dir)
        return GraphUnitDataset(self.index.units, self.read_labels())

    def unit_table(self, dataset: GraphUnitDataset | None = None):
        """Region, sample and (when labelled) patient id of every unit, indexed by unit id."""
        import pandas as pd

        dataset = dataset if dataset is not None else self.all_units()
        rows = []
        for entry in dataset.entries:
            row = {
                "unit_id": entry.unit_id,
                "region_id": entry.region_id,
                "sample_id": entry.sample_id,
            }
            label_id = dataset.label_id(entry)
            if label_id is not None and self.labels_cfg.get("patient_column"):
                row["patient_id"] = dataset.labels.patient(label_id)
            rows.append(row)
        return pd.DataFrame(rows).set_index("unit_id")

    @property
    def input_spec(self) -> InputSpec:
        if self.index is None:
            self.index = ProcessedIndex.read(self.processed_dir)
        return InputSpec(
            feature_names=tuple(self.index.feature_names),
            cell_types=tuple(self.index.cell_types),
            edge_features=tuple(self.index.edge_features),
            preprocessing=self.index.preprocessing,
        )

    # ---------------------------------------------------------------- loaders
    def loader(
        self,
        dataset: GraphUnitDataset,
        *,
        mode: str = "chunks",
        roots_per_unit: int | None = None,
        seed: int | None = None,
        shuffle: bool = False,
    ) -> EgoGraphLoader:
        """An ego-graph loader over ``dataset`` with this datamodule's sampling settings."""
        return EgoGraphLoader(
            dataset,
            self.sampler_cfg,
            mode=mode,
            roots_per_unit=roots_per_unit,
            seed=seed,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> EgoGraphLoader:
        mode = "merged" if self.sampler_cfg.merge_units else "bags"
        return self.loader(
            self.dataset_train,
            mode=mode,
            roots_per_unit=self.sampler_cfg.roots_per_unit,
            shuffle=True,
        )

    def val_dataloader(self) -> EgoGraphLoader:
        # A fixed root subset per unit, so epoch-to-epoch differences reflect the model.
        return self.loader(self.dataset_val, roots_per_unit=self.val_roots_per_unit, seed=0)

    def test_dataloader(self) -> EgoGraphLoader:
        return self.loader(self.dataset_test)
