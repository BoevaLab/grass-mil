"""Reading per-sample cell tables (CSV/TSV, h5ad, SingleCellExperiment).

Every format is reduced to a :class:`CellTable`: coordinates, an optional
cell-type label per cell, optional continuous features and cell ids. Which
columns are read is described by one :class:`InputConfig` for all formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["CellTable", "InputConfig", "load_cell_table", "load_polygons", "input_type_of"]


@dataclass(frozen=True)
class InputConfig:
    """Columns of the per-sample cell tables.

    Attributes:
        coord_columns: x/y columns (CSV/TSV; h5ad ``obs`` and SCE ``colData``
            when no embedding key is set).
        obsm_key: h5ad coordinates from ``obsm[obsm_key]``; ``None`` reads
            ``coord_columns`` from ``obs``.
        reduced_dim: SCE coordinates from ``reducedDim(sce, reduced_dim)``;
            ``None`` reads ``coord_columns`` from ``colData``.
        cell_id_column: Column of cell ids (``None``: row names / row number).
        cell_type_column: Column of cell-type labels (``None``: no cell types).
        features: Continuous node features: ``"all"`` (every remaining CSV
            column, every h5ad variable / SCE feature), a list of names, or
            ``None`` / ``[]`` for none (cell type only).
        h5ad_layer: Read h5ad features from this layer instead of ``X``.
        sce_assay: SCE assay name (``None``: the first assay).
        sep: CSV separator (``.tsv`` files always use a tab).
    """

    coord_columns: tuple[str, str] = ("x", "y")
    obsm_key: str | None = "spatial"
    reduced_dim: str | None = None
    cell_id_column: str | None = None
    cell_type_column: str | None = "cell_type"
    features: str | tuple[str, ...] | None = "all"
    h5ad_layer: str | None = None
    sce_assay: str | None = None
    sep: str = ","

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> InputConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown data.input keys: {sorted(unknown)}")
        if "coord_columns" in cfg:
            cfg["coord_columns"] = tuple(cfg["coord_columns"])
        features = cfg.get("features", "all")
        if features is not None and not isinstance(features, str):
            cfg["features"] = tuple(str(f) for f in features) or None
        elif isinstance(features, str) and features != "all":
            raise ValueError("data.input.features must be 'all', a list of names or null.")
        return cls(**cfg)


@dataclass
class CellTable:
    coords: np.ndarray  # (n, 2) float64
    cell_ids: np.ndarray  # (n,) str
    cell_types: np.ndarray | None = None  # (n,) str
    features: np.ndarray | None = None  # (n, f) float32
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def subset(self, indices: np.ndarray) -> CellTable:
        return CellTable(
            coords=self.coords[indices],
            cell_ids=self.cell_ids[indices],
            cell_types=None if self.cell_types is None else self.cell_types[indices],
            features=None if self.features is None else self.features[indices],
            feature_names=list(self.feature_names),
        )


_SUFFIX_TYPES = {".csv": "csv", ".tsv": "tsv", ".txt": "tsv", ".h5ad": "h5ad", ".rds": "sce"}


def input_type_of(path: str | Path, declared: str | None = None) -> str:
    """``csv`` | ``tsv`` | ``h5ad`` | ``sce``, from the manifest or the file suffix."""
    if declared:
        kind = str(declared).strip().lower()
        kind = {"rds": "sce"}.get(kind, kind)
    else:
        kind = _SUFFIX_TYPES.get(Path(path).suffix.lower(), "")
    if kind not in {"csv", "tsv", "h5ad", "sce"}:
        raise ValueError(f"Cannot tell the input type of {path}; set input_type in the manifest.")
    return kind


def load_cell_table(path: str | Path, kind: str, cfg: InputConfig) -> CellTable:
    if kind in {"csv", "tsv"}:
        return _load_csv(path, cfg, sep="\t" if kind == "tsv" else cfg.sep)
    if kind == "h5ad":
        return _load_h5ad(path, cfg)
    if kind == "sce":
        return _load_sce(path, cfg)
    raise ValueError(f"Unsupported input type '{kind}'.")


def _select_features(names: Sequence[str], cfg: InputConfig, *, exclude: set[str]) -> list[str]:
    if cfg.features is None:
        return []
    if cfg.features == "all":
        return [n for n in names if n not in exclude]
    missing = [n for n in cfg.features if n not in set(names)]
    if missing:
        raise ValueError(f"Feature(s) not found: {missing[:10]}")
    return list(cfg.features)


def _cell_types(values: pd.Series | None) -> np.ndarray | None:
    return None if values is None else values.astype(str).to_numpy()


def _load_csv(path: str | Path, cfg: InputConfig, *, sep: str) -> CellTable:
    df = pd.read_csv(path, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    needed = [*cfg.coord_columns, *([cfg.cell_type_column] if cfg.cell_type_column else [])]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}")
    exclude = {*cfg.coord_columns, cfg.cell_type_column, cfg.cell_id_column}
    names = _select_features(list(df.columns), cfg, exclude=exclude)
    return CellTable(
        coords=df[list(cfg.coord_columns)].to_numpy(dtype=np.float64),
        cell_ids=(
            df[cfg.cell_id_column].astype(str).to_numpy()
            if cfg.cell_id_column in df.columns
            else np.arange(len(df)).astype(str)
        ),
        cell_types=_cell_types(df[cfg.cell_type_column] if cfg.cell_type_column else None),
        features=df[names].to_numpy(dtype=np.float32) if names else None,
        feature_names=names,
    )


def _load_h5ad(path: str | Path, cfg: InputConfig) -> CellTable:
    import anndata as ad

    adata = ad.read_h5ad(path)
    if cfg.obsm_key:
        if cfg.obsm_key not in adata.obsm:
            raise ValueError(f"{path} has no obsm['{cfg.obsm_key}']")
        coords = np.asarray(adata.obsm[cfg.obsm_key], dtype=np.float64)[:, :2]
    else:
        coords = adata.obs[list(cfg.coord_columns)].to_numpy(dtype=np.float64)
    if cfg.cell_type_column and cfg.cell_type_column not in adata.obs:
        raise ValueError(f"{path} has no obs column '{cfg.cell_type_column}'")
    names = _select_features(list(map(str, adata.var_names)), cfg, exclude=set())
    features = None
    if names:
        matrix = adata[:, names].layers[cfg.h5ad_layer] if cfg.h5ad_layer else adata[:, names].X
        features = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
        features = features.astype(np.float32)
    return CellTable(
        coords=coords,
        cell_ids=(
            adata.obs[cfg.cell_id_column].astype(str).to_numpy()
            if cfg.cell_id_column
            else adata.obs_names.astype(str).to_numpy()
        ),
        cell_types=_cell_types(adata.obs[cfg.cell_type_column] if cfg.cell_type_column else None),
        features=features,
        feature_names=names,
    )


def _load_sce(path: str | Path, cfg: InputConfig) -> CellTable:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import default_converter, numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError as exc:
        raise ImportError(
            "Reading SingleCellExperiment (.rds) inputs needs rpy2 and an R installation "
            "with Bioconductor (pip install 'grass-mil[sce]')."
        ) from exc

    r = ro.r
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter):
        sce = r["readRDS"](str(path))
        coldata = r["as.data.frame"](r["colData"](sce))
        if cfg.reduced_dim:
            coords = np.asarray(r["reducedDim"](sce, cfg.reduced_dim), dtype=np.float64)[:, :2]
        else:
            coords = coldata[list(cfg.coord_columns)].to_numpy(dtype=np.float64)
        assay = r["assay"](sce, cfg.sce_assay) if cfg.sce_assay else r["assay"](sce)
        all_names = [str(n) for n in r["rownames"](sce)]
        names = _select_features(all_names, cfg, exclude=set())
        features = None
        if names:
            rows = [all_names.index(n) for n in names]
            features = np.asarray(assay, dtype=np.float32)[rows, :].T
        cell_ids = (
            coldata[cfg.cell_id_column].astype(str).to_numpy()
            if cfg.cell_id_column
            else np.asarray(r["colnames"](sce), dtype=str)
        )
    if cfg.cell_type_column and cfg.cell_type_column not in coldata.columns:
        raise ValueError(f"{path} has no colData column '{cfg.cell_type_column}'")
    return CellTable(
        coords=coords,
        cell_ids=cell_ids,
        cell_types=_cell_types(coldata[cfg.cell_type_column] if cfg.cell_type_column else None),
        features=features,
        feature_names=names,
    )


def load_polygons(path: str | Path) -> list[Any]:
    """Polygons from a GeoJSON (``.json``/``.geojson``) or WKT (one per line) file."""
    import json

    from shapely import wkt
    from shapely.geometry import shape

    path = Path(path)
    if path.suffix.lower() in {".json", ".geojson"}:
        raw = json.loads(path.read_text())
        if raw.get("type") == "FeatureCollection":
            polygons = [shape(feature["geometry"]) for feature in raw["features"]]
        else:
            polygons = [shape(raw)]
    elif path.suffix.lower() in {".wkt", ".txt"}:
        polygons = [wkt.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported polygon file {path}: use GeoJSON or WKT.")
    if not polygons:
        raise ValueError(f"No polygons in {path}")
    return polygons
