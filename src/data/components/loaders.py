from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .spatial_types import SpatialOmicsTable


@dataclass
class CsvConfig:
    coord_columns: Tuple[str, str]
    cell_id_column: Optional[str]
    categorical_label_columns: Sequence[str]
    molecular_columns: Optional[Sequence[str]]
    sep: str = ","


@dataclass
class H5adConfig:
    coord_columns: Tuple[str, str]
    cell_id_column: Optional[str]
    categorical_label_columns: Sequence[str]
    molecular_layer: Optional[str]


@dataclass
class SceConfig:
    assay_name: Optional[str]
    coord_source: str  # "colData" or "reducedDims"
    coord_key: Optional[str]
    coord_columns: Optional[Tuple[str, str]]
    cell_id_column: Optional[str]
    categorical_label_columns: Sequence[str]
    transpose_assay: bool = True


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_csv_table(
    path: str | Path,
    config: CsvConfig,
    sample_id: Optional[str] = None,
    region_id: Optional[str] = None,
) -> SpatialOmicsTable:
    df = pd.read_csv(path, sep=config.sep)
    df = _normalize_columns(df)

    x_col, y_col = config.coord_columns
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Missing coord columns {config.coord_columns} in {path}")

    if config.cell_id_column and config.cell_id_column in df.columns:
        cell_ids = df[config.cell_id_column].astype(str).to_numpy()
    else:
        cell_ids = df.index.astype(str).to_numpy()

    coord_df = df[[x_col, y_col]].astype(float)
    coords = coord_df.to_numpy()

    categorical_labels: Dict[str, np.ndarray] = {}
    for label_col in config.categorical_label_columns:
        if label_col not in df.columns:
            raise ValueError(f"Missing categorical label column {label_col} in {path}")
        categorical_labels[label_col] = df[label_col].astype(str).to_numpy()

    if config.molecular_columns is None:
        exclude = {x_col, y_col, *(config.categorical_label_columns or [])}
        if config.cell_id_column:
            exclude.add(config.cell_id_column)
        molecular_cols = [c for c in df.columns if c not in exclude]
    else:
        molecular_cols = list(config.molecular_columns)

    if not molecular_cols:
        raise ValueError(f"No molecular feature columns found in {path}")

    molecular_features = df[molecular_cols].astype(float).to_numpy()

    return SpatialOmicsTable(
        coords=coords,
        molecular_features=molecular_features,
        categorical_labels=categorical_labels,
        cell_ids=cell_ids,
        sample_id=sample_id,
        region_id=region_id,
    )


def load_h5ad_table(
    path: str | Path,
    config: H5adConfig,
    sample_id: Optional[str] = None,
    region_id: Optional[str] = None,
) -> SpatialOmicsTable:
    import anndata as ad

    adata = ad.read_h5ad(path)
    x_col, y_col = config.coord_columns
    if x_col not in adata.obs.columns or y_col not in adata.obs.columns:
        raise ValueError(f"Missing coord columns {config.coord_columns} in {path}")

    coords = adata.obs[[x_col, y_col]].to_numpy(dtype=float)

    if config.cell_id_column and config.cell_id_column in adata.obs.columns:
        cell_ids = adata.obs[config.cell_id_column].astype(str).to_numpy()
    else:
        cell_ids = adata.obs_names.astype(str).to_numpy()

    categorical_labels: Dict[str, np.ndarray] = {}
    for label_col in config.categorical_label_columns:
        if label_col not in adata.obs.columns:
            raise ValueError(f"Missing categorical label column {label_col} in {path}")
        categorical_labels[label_col] = adata.obs[label_col].astype(str).to_numpy()

    if config.molecular_layer:
        data = adata.layers[config.molecular_layer]
    else:
        data = adata.X

    if hasattr(data, "toarray"):
        data = data.toarray()
    molecular_features = np.asarray(data, dtype=float)

    return SpatialOmicsTable(
        coords=coords,
        molecular_features=molecular_features,
        categorical_labels=categorical_labels,
        cell_ids=cell_ids,
        sample_id=sample_id,
        region_id=region_id,
    )


def load_sce_table(
    path: str | Path,
    config: SceConfig,
    sample_id: Optional[str] = None,
    region_id: Optional[str] = None,
) -> SpatialOmicsTable:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
    except ImportError as exc:
        raise ImportError(
            "rpy2 is required to load SingleCellExperiment objects. "
            "Install rpy2 and make sure R + Bioconductor are available."
        ) from exc

    pandas2ri.activate()
    numpy2ri.activate()

    r = ro.r
    sce = r["readRDS"](str(path))

    if config.assay_name:
        assay = r["assay"](sce, config.assay_name)
    else:
        assay = r["assay"](sce)
    molecular_features = np.asarray(assay, dtype=float)
    if config.transpose_assay:
        molecular_features = molecular_features.T

    coldata_df = None
    try:
        coldata_df = pandas2ri.rpy2py(r["as.data.frame"](r["colData"](sce)))
    except Exception:
        coldata_df = None

    if config.coord_source == "colData":
        if coldata_df is None:
            raise ValueError("coord_source=colData but colData could not be loaded from SCE.")
        if not config.coord_columns:
            raise ValueError("coord_columns must be set when coord_source=colData.")
        x_col, y_col = config.coord_columns
        if x_col not in coldata_df.columns or y_col not in coldata_df.columns:
            raise ValueError(f"Missing coord columns {config.coord_columns} in colData for {path}")
        coords = coldata_df[[x_col, y_col]].to_numpy(dtype=float)
    elif config.coord_source == "reducedDims":
        if not config.coord_key:
            raise ValueError("coord_key must be set when coord_source=reducedDims.")
        reduced = r["reducedDim"](sce, config.coord_key)
        coords = np.asarray(reduced, dtype=float)
        if coords.shape[1] < 2:
            raise ValueError(f"reducedDim '{config.coord_key}' has <2 columns.")
        coords = coords[:, :2]
    else:
        raise ValueError("coord_source must be 'colData' or 'reducedDims'.")

    if config.cell_id_column and coldata_df is not None and config.cell_id_column in coldata_df.columns:
        cell_ids = coldata_df[config.cell_id_column].astype(str).to_numpy()
    else:
        cell_ids = np.asarray(r["colnames"](sce), dtype=str)

    categorical_labels: Dict[str, np.ndarray] = {}
    if config.categorical_label_columns and coldata_df is None:
        raise ValueError("categorical_label_columns requested but colData could not be loaded.")
    for label_col in config.categorical_label_columns:
        if label_col not in coldata_df.columns:
            raise ValueError(f"Missing categorical label column {label_col} in colData for {path}")
        categorical_labels[label_col] = coldata_df[label_col].astype(str).to_numpy()

    return SpatialOmicsTable(
        coords=coords,
        molecular_features=molecular_features,
        categorical_labels=categorical_labels,
        cell_ids=cell_ids,
        sample_id=sample_id,
        region_id=region_id,
    )


def load_polygons_from_path(path: str | Path) -> List[object]:
    from shapely.geometry import shape
    from shapely import wkt

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Polygon file not found: {path}")

    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            polygons = pickle.load(f)
        return _ensure_polygon_list(polygons)

    if path.suffix in {".wkt", ".txt"}:
        polygons = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            polygons.append(wkt.loads(line.strip()))
        return _ensure_polygon_list(polygons)

    if path.suffix in {".json", ".geojson"}:
        raw = json.loads(path.read_text())
        if raw.get("type") == "FeatureCollection":
            polygons = [shape(feature["geometry"]) for feature in raw["features"]]
        elif raw.get("type") in {"Polygon", "MultiPolygon"}:
            polygons = [shape(raw)]
        else:
            raise ValueError(f"Unsupported GeoJSON type in {path}")
        return _ensure_polygon_list(polygons)

    raise ValueError(f"Unsupported polygon file type: {path.suffix}")


def _ensure_polygon_list(polygons: Iterable[object]) -> List[object]:
    polygons_list = list(polygons)
    if not polygons_list:
        raise ValueError("No polygons found in polygon file")
    return polygons_list
