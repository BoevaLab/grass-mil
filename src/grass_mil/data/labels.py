"""Unit labels and patient grouping, read once from a label CSV.

The label file has one row per labelled id (a region, a tile or a sample), the
target column(s) and optionally a patient column used for grouping. Every part
of the package (training, cross-validation, prediction, niche analysis) reads
labels through :class:`LabelTable`, so they are parsed and encoded in exactly
one way: numeric columns are kept as floats, text columns are encoded as
integer codes in sorted order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["LabelTable", "class_index"]


@dataclass
class LabelTable:
    """Targets and patient of every labelled id.

    ``targets`` is indexed by id (str) with one float column per target;
    ``patients`` maps id to patient (every id is its own patient when the file
    has no patient column). ``codes`` records the encoding of text columns.
    """

    targets: pd.DataFrame
    patients: pd.Series
    codes: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        id_column: str,
        target_columns: Sequence[str] | None = None,
        patient_column: str | None = None,
    ) -> LabelTable:
        df = pd.read_csv(path, dtype={id_column: str})
        df.columns = [str(c).strip() for c in df.columns]
        if id_column not in df.columns:
            raise ValueError(f"Label file {path} has no id column '{id_column}'.")
        if patient_column and patient_column not in df.columns:
            raise ValueError(f"Label file {path} has no patient column '{patient_column}'.")
        df[id_column] = df[id_column].str.strip()
        if df[id_column].duplicated().any():
            dup = sorted(df.loc[df[id_column].duplicated(), id_column])[:5]
            raise ValueError(f"Label file {path} repeats ids: {dup}")
        df = df.set_index(id_column)
        if target_columns is None:
            # Default: the first column that is not the patient column.
            candidates = [c for c in df.columns if c != patient_column]
            if not candidates:
                raise ValueError(f"Label file {path} has no target column.")
            target_columns = candidates[:1]
        missing = [c for c in target_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Label file {path} has no column(s) {missing}.")

        targets = pd.DataFrame(index=df.index)
        codes: dict[str, list[str]] = {}
        for column in target_columns:
            values = df[column]
            if not pd.api.types.is_numeric_dtype(values):
                categories = sorted(values.dropna().astype(str).unique())
                codes[column] = categories
                values = values.astype("string").map({c: i for i, c in enumerate(categories)})
            targets[column] = pd.to_numeric(values, errors="raise").astype(float)
        targets = targets.dropna(how="any")
        if patient_column:
            patients = df.loc[targets.index, patient_column].astype(str)
        else:
            patients = pd.Series(targets.index, index=targets.index, dtype=str)
        return cls(targets=targets, patients=patients, codes=codes)

    @property
    def target_columns(self) -> list[str]:
        return list(self.targets.columns)

    def __contains__(self, unit_id: str) -> bool:
        return str(unit_id) in self.targets.index

    def resolve(self, *ids: str | None) -> str | None:
        """The first of ``ids`` that has a label (unit id, then region, then sample)."""
        for value in ids:
            if value is not None and str(value) in self.targets.index:
                return str(value)
        return None

    def target(self, label_id: str) -> np.ndarray:
        return self.targets.loc[str(label_id)].to_numpy(dtype=np.float32)

    def patient(self, label_id: str) -> str:
        return str(self.patients.loc[str(label_id)])

    def classes(self, label_id: str) -> int:
        """Integer class of a classification label (first target column)."""
        return class_index(float(self.targets.loc[str(label_id)].iloc[0]))


def class_index(value: float) -> int:
    """Integer class of a label value: the nearest integer, halves rounded up (0.5 -> 1)."""
    return int(np.floor(float(value) + 0.5))
