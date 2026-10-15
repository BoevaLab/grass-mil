"""Building the cell graphs of a cohort once and storing them on disk.

The manifest lists one row per region (``sample_id``, ``input_path`` and
optionally ``region_id``, ``input_type``, ``polygons_path``). Every region is
cut into graph units (the whole region, square tiles, or polygons), each unit
becomes a PyG ``Data`` with

* ``x``: continuous node features (possibly zero columns),
* ``cell_type``: integer cell-type codes (when a cell-type column is set),
* ``pos``, ``edge_index`` and ``edge_attr`` of the Delaunay graph,

and ``processed_index.json`` records the units, the cell-type vocabulary, the
feature names and a fingerprint of every setting that shaped the graphs. A
later run with different settings refuses to reuse the directory instead of
silently training on stale graphs. Labels are not part of the graphs; they are
attached when units are loaded (:mod:`grass_mil.data.datamodule`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .graphs import GraphConfig, delaunay_edges, edge_features
from .loaders import CellTable, InputConfig, input_type_of, load_cell_table, load_polygons
from .reducers import Pca, ReducerConfig, load_reducer_state
from .units import TilingConfig, grid_tiles, inside_any, polygon_members

__all__ = ["BuildConfig", "UnitEntry", "ProcessedIndex", "read_manifest", "build_graphs"]

log = logging.getLogger(__name__)

INDEX_NAME = "processed_index.json"
INDEX_VERSION = 3
UNIT_KINDS = ("region", "tile", "polygon")


@dataclass(frozen=True)
class BuildConfig:
    """Every setting that shapes the processed graphs."""

    unit: str = "region"
    coord_scale_um: float = 1.0
    min_cells: int = 10
    tiling: TilingConfig = field(default_factory=TilingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    feature_reducer: ReducerConfig = field(default_factory=ReducerConfig)

    def __post_init__(self) -> None:
        if self.unit not in UNIT_KINDS:
            raise ValueError(f"data.unit must be one of {list(UNIT_KINDS)}, got '{self.unit}'.")


@dataclass(frozen=True)
class UnitEntry:
    unit_id: str
    region_id: str
    sample_id: str
    path: str
    n_cells: int

    @property
    def ids(self) -> set[str]:
        """The ids that select this unit: its own, its region's and its sample's."""
        return {self.unit_id, self.region_id, self.sample_id}


@dataclass
class ProcessedIndex:
    units: list[UnitEntry]
    cell_types: list[str]
    feature_names: list[str]
    edge_features: list[str]
    fingerprint: str
    feature_reducer: dict
    preprocessing: str

    @classmethod
    def read(cls, processed_dir: str | Path) -> ProcessedIndex:
        path = Path(processed_dir) / INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(f"No processed graphs at {path}; run prepare_data() first.")
        raw = json.loads(path.read_text())
        if raw.get("version") != INDEX_VERSION:
            raise ValueError(
                f"{path} was written by another version of grass-mil; "
                "rebuild it with data.force_precompute=true."
            )
        return cls(
            units=[UnitEntry(**u) for u in raw["units"]],
            cell_types=list(raw["cell_types"]),
            feature_names=list(raw["feature_names"]),
            edge_features=list(raw["edge_features"]),
            fingerprint=str(raw["fingerprint"]),
            feature_reducer=dict(raw.get("feature_reducer") or {}),
            preprocessing=str(raw["preprocessing"]),
        )

    def write(self, processed_dir: str | Path) -> Path:
        payload = {"version": INDEX_VERSION, **asdict(self)}
        path = Path(processed_dir) / INDEX_NAME
        path.write_text(json.dumps(payload, indent=1))
        return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Manifest rows with resolved input paths and a ``region_id`` per row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for column in ("sample_id", "input_path"):
        if column not in df.columns:
            raise ValueError(f"Manifest {path} has no '{column}' column.")
    df = df.apply(lambda col: col.str.strip())
    if "region_id" not in df.columns:
        df["region_id"] = ""
    df["region_id"] = df["region_id"].where(df["region_id"] != "", df["sample_id"])
    if df["region_id"].duplicated().any():
        dup = sorted(df.loc[df["region_id"].duplicated(), "region_id"])[:5]
        raise ValueError(f"Manifest {path}: region ids must be unique; repeated: {dup}")

    def _resolve(value: str) -> str:
        if not value:
            return ""
        p = Path(value)
        return str(p if p.is_absolute() else (path.parent / p).resolve())

    df["input_path"] = df["input_path"].map(_resolve)
    if "polygons_path" in df.columns:
        df["polygons_path"] = df["polygons_path"].map(_resolve)
    return df


def fingerprint(cfg: BuildConfig, manifest: pd.DataFrame, split_key: str = "") -> str:
    h = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True, default=list).encode())
    h.update(manifest.to_csv(index=False).encode())
    h.update(split_key.encode())
    return h.hexdigest()[:16]


def preprocessing_fingerprint(graph: GraphConfig, reducer_state) -> str:
    """What a model's inputs depend on beyond feature names: graph settings and feature basis."""
    payload = json.dumps({"graph": asdict(graph), "reducer": reducer_state}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _units_of(table: CellTable, row: pd.Series, cfg: BuildConfig) -> list[tuple[str, np.ndarray]]:
    region = str(row["region_id"])
    polygons_path = row.get("polygons_path") or None
    polygons = load_polygons(polygons_path) if polygons_path else None
    if polygons and cfg.coord_scale_um != 1.0:
        from shapely import affinity

        scale = cfg.coord_scale_um
        polygons = [affinity.scale(p, xfact=scale, yfact=scale, origin=(0, 0)) for p in polygons]
    if cfg.unit == "polygon":
        if not polygons:
            raise ValueError(f"data.unit=polygon but region {region} has no polygons_path.")
        parts = [(f"{region}_polygon{k:03d}", idx) for k, idx in enumerate(polygon_members(table.coords, polygons))]
    else:
        keep = inside_any(table.coords, polygons) if polygons else np.arange(len(table))
        if cfg.unit == "tile":
            tiles = grid_tiles(table.coords[keep], cfg.tiling)
            parts = [(f"{region}_tile{k:03d}", keep[idx]) for k, idx in enumerate(tiles)]
        else:
            parts = [(region, keep)]
    return [(uid, idx) for uid, idx in parts if idx.size >= cfg.min_cells]


def build_graphs(
    manifest_path: str | Path,
    processed_dir: str | Path,
    cfg: BuildConfig,
    *,
    force: bool = False,
    training_units: Callable[[list[UnitEntry]], set[str]] | None = None,
    split_key: str = "",
) -> ProcessedIndex:
    """Build (or reuse) the processed graphs of a cohort.

    ``training_units`` returns the ids of the training units given all units;
    it is needed only for ``feature_reducer.fit_on=train``, in which case
    ``split_key`` describes that split so a different split triggers a rebuild.
    """
    processed_dir = Path(processed_dir)
    manifest = read_manifest(manifest_path)
    fp = fingerprint(cfg, manifest, split_key)
    if (processed_dir / INDEX_NAME).exists() and not force:
        index = ProcessedIndex.read(processed_dir)
        if index.fingerprint != fp:
            raise ValueError(
                f"{processed_dir} holds graphs built with different settings or manifest. "
                "Set data.force_precompute=true to rebuild, or use another data.processed_dir."
            )
        return index

    log.info(f"Building cell graphs for {len(manifest)} region(s) into {processed_dir}")
    tables: list[tuple[pd.Series, CellTable, list[tuple[str, np.ndarray]]]] = []
    for _, row in manifest.iterrows():
        kind = input_type_of(row["input_path"], row.get("input_type") or None)
        table = load_cell_table(row["input_path"], kind, cfg.input)
        table.coords = table.coords * float(cfg.coord_scale_um)
        tables.append((row, table, _units_of(table, row, cfg)))

    cell_types: list[str] = []
    if cfg.input.cell_type_column:
        cell_types = sorted({str(t) for _, table, _ in tables for t in table.cell_types})
    code_of = {name: i for i, name in enumerate(cell_types)}
    feature_names = tables[0][1].feature_names if tables else []
    if any(table.feature_names != feature_names for _, table, _ in tables):
        raise ValueError("All regions must provide the same feature columns.")

    entries = [
        UnitEntry(uid, str(row["region_id"]), str(row["sample_id"]), "", int(idx.size))
        for row, _, units in tables
        for uid, idx in units
    ]
    reducer_state, project = _fit_reducer(tables, entries, cfg.feature_reducer, training_units)
    if cfg.feature_reducer.name == "pca" and feature_names:
        feature_names = [f"pc{i + 1}" for i in range(min(cfg.feature_reducer.n_components, len(feature_names)))]

    graph_dir = processed_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    written: list[UnitEntry] = []
    for row, table, units in tables:
        features = project(table)
        for uid, idx in units:
            path = graph_dir / f"{_safe(uid)}.pt"
            data = _unit_graph(table, features, idx, code_of, cfg.graph)
            torch.save({"data": data, "cell_ids": table.cell_ids[idx].tolist()}, path)
            written.append(UnitEntry(uid, str(row["region_id"]), str(row["sample_id"]), str(path), int(idx.size)))
    index = ProcessedIndex(
        units=written,
        cell_types=cell_types,
        feature_names=list(feature_names),
        edge_features=list(cfg.graph.edge_features),
        fingerprint=fp,
        feature_reducer={**asdict(cfg.feature_reducer), "state": reducer_state},
        preprocessing=preprocessing_fingerprint(cfg.graph, reducer_state),
    )
    index.write(processed_dir)
    log.info(f"Wrote {len(written)} graph unit(s) to {processed_dir}")
    return index


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _fit_reducer(tables, entries, cfg: ReducerConfig, training_units):
    """The fitted reducer state and a ``table -> features`` projection."""
    if cfg.name == "identity" or not any(t.features is not None for _, t, _ in tables):
        return None, lambda table: table.features
    if cfg.state_from:
        pca = load_reducer_state(cfg.state_from)
    elif cfg.fit_on == "train":
        if training_units is None:
            raise ValueError("feature_reducer.fit_on=train needs a training split.")
        train = training_units(entries)
        parts = [t.features[idx] for _, t, units in tables for uid, idx in units if uid in train]
        pca = Pca.fit(np.concatenate(parts), cfg.n_components)
    else:
        pca = Pca.fit(np.concatenate([t.features for _, t, _ in tables]), cfg.n_components)
    return pca.state(), lambda table: pca.transform(table.features)


def _unit_graph(
    table: CellTable,
    features: np.ndarray | None,
    idx: np.ndarray,
    code_of: dict[str, int],
    graph: GraphConfig,
) -> Data:
    coords = table.coords[idx]
    edge_index = delaunay_edges(coords)
    x = (
        torch.from_numpy(np.ascontiguousarray(features[idx], dtype=np.float32))
        if features is not None
        else torch.zeros((idx.size, 0))
    )
    data = Data(
        x=x,
        pos=torch.from_numpy(coords).float(),
        edge_index=edge_index,
        edge_attr=edge_features(coords, edge_index, graph),
        num_nodes=int(idx.size),
    )
    if code_of:
        data.cell_type = torch.tensor([code_of[str(t)] for t in table.cell_types[idx]])
    return data
