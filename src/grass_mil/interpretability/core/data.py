from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from grass_mil.contracts import EmbeddingSet, InterpretabilityDataset


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path}. Expected .csv or .parquet")


def load_interpretability_dataset(
    *,
    instance_table_path: Path,
    bag_table_path: Optional[Path] = None,
    spatial_table_path: Optional[Path] = None,
    cell_table_path: Optional[Path] = None,
    cell_edge_table_path: Optional[Path] = None,
    id_column: str = "instance_id",
    bag_id_column: str = "bag_id",
    cell_type_column: Optional[str] = None,
    condition_column: Optional[str] = None,
) -> InterpretabilityDataset:
    instance_table = load_table(instance_table_path)
    bag_table = load_table(bag_table_path) if bag_table_path else None
    spatial_table = load_table(spatial_table_path) if spatial_table_path else None
    cell_table = load_table(cell_table_path) if cell_table_path else None
    cell_edge_table = load_table(cell_edge_table_path) if cell_edge_table_path else None
    return InterpretabilityDataset(
        instance_table=instance_table,
        bag_table=bag_table,
        spatial_table=spatial_table,
        cell_table=cell_table,
        cell_edge_table=cell_edge_table,
        id_column=id_column,
        bag_id_column=bag_id_column,
        cell_type_column=cell_type_column,
        condition_column=condition_column,
    )


def extract_embedding_set(
    dataset: InterpretabilityDataset,
    *,
    embedding_prefixes: Iterable[str] = ("inst_emb_", "emb_", "graph_emb_"),
) -> EmbeddingSet:
    columns = list(dataset.instance_table.columns)
    emb_cols: list[str] = []
    for prefix in embedding_prefixes:
        emb_cols = [c for c in columns if c.startswith(prefix)]
        if emb_cols:
            break
    if not emb_cols:
        raise ValueError(
            "No embedding columns found in instance table. "
            "Expected prefixes: " + ", ".join(embedding_prefixes)
        )
    matrix = dataset.instance_table[emb_cols].to_numpy(dtype=float)
    return EmbeddingSet(
        matrix=np.asarray(matrix), feature_columns=emb_cols, id_column=dataset.id_column
    )
