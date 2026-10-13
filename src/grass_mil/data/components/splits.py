"""Explicit splits, fold files and patient-grouped inner folds.

The preprint's cross-validation uses fixed, patient-grouped outer folds stored
as a JSON list of region-id lists, and patient-grouped stratified inner folds
built from the development regions of each outer fold. These helpers read and
build those splits; the datamodule applies them through ``data.split.explicit``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ExplicitSplit",
    "LabelTable",
    "load_fold_json",
    "load_labels",
    "make_inner_folds",
    "write_fold_json",
]


@dataclass
class ExplicitSplit:
    """Region (or sample) ids per split; ids not listed are excluded."""

    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    id_kind: str = "region"  # region | sample | patch

    @classmethod
    def from_dict(cls, cfg: Optional[Mapping[str, Any]]) -> Optional["ExplicitSplit"]:
        if not cfg:
            return None
        train = cfg.get("train")
        val = cfg.get("val")
        test = cfg.get("test")
        if train is None and val is None and test is None:
            return None
        id_kind = str(cfg.get("id_kind", "region"))
        if id_kind not in {"region", "sample", "patch"}:
            raise ValueError("split.explicit.id_kind must be 'region', 'sample' or 'patch'.")

        def _ids(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, (str, Path)):
                path = Path(str(value))
                if path.suffix.lower() == ".json":
                    return [str(v) for v in json.loads(path.read_text())]
                return [line.strip() for line in path.read_text().splitlines() if line.strip()]
            return [str(v) for v in value]

        split = cls(train=_ids(train), val=_ids(val), test=_ids(test), id_kind=id_kind)
        overlap = (set(split.train) & set(split.val)) | (set(split.train) & set(split.test))
        overlap |= set(split.val) & set(split.test)
        if overlap and not (set(split.val) == set(split.test) and not (set(split.train) & overlap)):
            # val == test is tolerated (refit runs score the held-out fold as "val"
            # without using it for selection); any other overlap is a leak.
            raise ValueError(f"split.explicit lists overlap: {sorted(overlap)[:10]}")
        return split

    def lookup(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name in ("train", "val", "test"):
            for rid in getattr(self, name):
                out.setdefault(rid, name)
        return out


def load_fold_json(path: str | Path) -> List[List[str]]:
    """Outer folds: a JSON list of lists of region ids."""
    folds = json.loads(Path(path).read_text())
    if not isinstance(folds, list) or not all(isinstance(f, list) for f in folds):
        raise ValueError(f"{path} must contain a JSON list of lists of region ids.")
    folds = [[str(r) for r in fold] for fold in folds]
    seen: set[str] = set()
    for fold in folds:
        dup = seen & set(fold)
        if dup:
            raise ValueError(f"Region ids appear in more than one fold: {sorted(dup)[:10]}")
        seen |= set(fold)
    return folds


def write_fold_json(folds: Sequence[Sequence[str]], path: str | Path) -> None:
    Path(path).write_text(json.dumps([[str(r) for r in f] for f in folds]))


@dataclass
class LabelTable:
    """Per-region label and grouping information."""

    labels: pd.Series  # index: region id (str) -> int label
    groups: pd.Series  # index: region id (str) -> group id (patient), str

    def label_of(self, rid: str) -> int:
        return int(self.labels.loc[str(rid)])

    def group_of(self, rid: str) -> str:
        return str(self.groups.loc[str(rid)])

    def subset(self, ids: Sequence[str]) -> "LabelTable":
        ids = [str(r) for r in ids]
        return LabelTable(labels=self.labels.loc[ids], groups=self.groups.loc[ids])


def load_labels(
    label_file: str | Path,
    *,
    id_column: str,
    label_column: str,
    patient_column: Optional[str] = None,
    drop_unlabelled: bool = True,
) -> LabelTable:
    """Read a label CSV into region -> label and region -> group series.

    Without ``patient_column`` every region is its own group (no grouping).
    Non-numeric labels are mapped to codes in sorted order.
    """
    df = pd.read_csv(label_file)
    df.columns = [str(c).strip() for c in df.columns]
    for col in (id_column, label_column) + ((patient_column,) if patient_column else ()):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {label_file}; columns: {list(df.columns)}")
    df[id_column] = df[id_column].astype(str)
    df = df.drop_duplicates(id_column).set_index(id_column)
    values = df[label_column]
    if drop_unlabelled:
        values = values.dropna()
    if not np.issubdtype(values.dtype, np.number):
        codes = {v: i for i, v in enumerate(sorted(values.astype(str).unique()))}
        values = values.astype(str).map(codes)
    labels = values.astype(float).round().astype(int)
    if patient_column:
        groups = df.loc[labels.index, patient_column].astype(str)
    else:
        groups = pd.Series(labels.index, index=labels.index, dtype=str)
    return LabelTable(labels=labels, groups=groups)


def make_inner_folds(
    dev_ids: Sequence[str],
    table: LabelTable,
    *,
    k: int = 3,
    seed: int = 0,
    max_reseeds: int = 20,
) -> List[Tuple[List[str], List[str]]]:
    """Stratified, group-aware inner folds over the SORTED development ids.

    Uses ``StratifiedGroupKFold(k, shuffle=True, random_state=seed)``; sorting the
    ids first makes the folds a deterministic function of the id set, which the
    preprint relies on for reproducible epoch selection. With very few groups
    sklearn's greedy assignment can leave a fold without one class, which makes
    AUROC undefined; such a split is rejected and the next seed is tried
    (``max_reseeds`` times) before giving up. Large cohorts keep ``seed``.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    ids = np.array(sorted(str(r) for r in dev_ids))
    y = np.array([table.label_of(r) for r in ids])
    g = np.array([table.group_of(r) for r in ids])
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError("Inner folds need at least two classes among the development regions.")
    if len(np.unique(g)) < int(k):
        raise ValueError(
            f"Inner folds need at least k={k} patient groups; got {len(np.unique(g))}."
        )
    last_error: Optional[str] = None
    for attempt in range(int(max_reseeds) + 1):
        splitter = StratifiedGroupKFold(
            n_splits=int(k), shuffle=True, random_state=int(seed) + attempt
        )
        folds: List[Tuple[List[str], List[str]]] = []
        ok = True
        for tr, va in splitter.split(np.zeros((len(ids), 1)), y, g):
            if len(np.unique(y[tr])) < n_classes or len(np.unique(y[va])) < 2:
                ok = False
                break
            folds.append((ids[tr].tolist(), ids[va].tolist()))
        if ok and len(folds) == int(k):
            return folds
        last_error = f"seed {int(seed) + attempt} left a fold without one class"
    raise ValueError(
        f"Could not build {k} stratified patient-grouped inner folds with every class in "
        f"every split after {max_reseeds + 1} seeds ({last_error}). Use fewer inner folds "
        "or more development patients."
    )
