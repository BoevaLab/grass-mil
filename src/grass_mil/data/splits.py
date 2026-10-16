"""Train/validation/test splits, outer fold files and inner cross-validation folds.

All splits are over groups of units: units of the same patient (or, without a
patient column, of the same region) always fall on the same side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["read_ids", "load_folds", "random_split", "inner_folds"]


def read_ids(value: Any) -> list[str] | None:
    """Ids from a list, or from a ``.json`` (list) or text (one per line) file."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(str(value))
        if path.suffix.lower() == ".json":
            return [str(v) for v in json.loads(path.read_text())]
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [str(v) for v in value]


def load_folds(path: str | Path) -> list[list[str]]:
    """Outer folds: a JSON list of lists of unit ids, each id in at most one fold."""
    folds = json.loads(Path(path).read_text())
    if not isinstance(folds, list) or not all(isinstance(f, list) for f in folds):
        raise ValueError(f"{path} must hold a JSON list of lists of unit ids.")
    folds = [[str(r) for r in fold] for fold in folds]
    seen: set[str] = set()
    for fold in folds:
        repeated = seen & set(fold)
        if repeated:
            raise ValueError(f"{path}: ids in more than one fold: {sorted(repeated)[:10]}")
        seen |= set(fold)
    return folds


def random_split(
    groups: Mapping[str, str],
    ratios: Sequence[float],
    seed: int,
    strata: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign every unit to ``train``/``val``/``test`` by shuffling its group.

    ``groups`` maps unit id to group id; ``strata`` (optional) maps group id to
    a stratum (the class), and every stratum is split separately so each split
    holds every class when possible. Group counts per split follow ``ratios``
    (largest remainder); a split with a positive ratio gets at least one group
    of a stratum when the stratum has enough groups.
    """
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isclose(ratios.sum(), 1.0) or (ratios < 0).any():
        raise ValueError("data.split.ratios must be three non-negative numbers summing to 1.")
    rng = np.random.default_rng(int(seed))
    by_stratum: dict[str, list[str]] = {}
    for name in sorted(set(groups.values())):
        by_stratum.setdefault(str(strata[name]) if strata else "", []).append(name)
    split_of_group: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        names = by_stratum[stratum]
        order = rng.permutation(len(names))
        raw = ratios * len(names)
        counts = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - counts))[: len(names) - int(counts.sum())]:
            counts[i] += 1
        for i in np.flatnonzero((ratios > 0) & (counts == 0)):
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[i] += 1
        start = 0
        for split, count in zip(("train", "val", "test"), counts):
            for j in order[start : start + count]:
                split_of_group[names[j]] = split
            start += count
    return {unit: split_of_group[group] for unit, group in groups.items()}


def inner_folds(
    ids: Sequence[str],
    *,
    groups: Mapping[str, str],
    classes: Mapping[str, int] | None,
    k: int,
    seed: int,
    max_reseeds: int = 20,
) -> list[tuple[list[str], list[str]]]:
    """``k`` group-aware inner folds over the sorted ids.

    With ``classes`` (classification) the folds are stratified, and a split
    leaving a fold without one of the classes is redrawn with the next seed (up
    to ``max_reseeds`` times), because AUROC is undefined on such a fold.
    Sorting the ids first makes the folds a function of the id set alone.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    ids = np.array(sorted(str(r) for r in ids))
    g = np.array([groups[r] for r in ids])
    if len(np.unique(g)) < k:
        raise ValueError(f"Inner folds need at least k={k} groups; got {len(np.unique(g))}.")
    if classes is None:
        splitter = GroupKFold(n_splits=int(k))
        return [(ids[tr].tolist(), ids[va].tolist()) for tr, va in splitter.split(ids, groups=g)]

    y = np.array([classes[r] for r in ids])
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError("Inner folds need at least two classes among the development units.")
    for attempt in range(int(max_reseeds) + 1):
        splitter = StratifiedGroupKFold(n_splits=int(k), shuffle=True, random_state=seed + attempt)
        folds = [(tr, va) for tr, va in splitter.split(ids, y, g)]
        if all(len(np.unique(y[tr])) == n_classes and len(np.unique(y[va])) > 1 for tr, va in folds):
            return [(ids[tr].tolist(), ids[va].tolist()) for tr, va in folds]
    raise ValueError(
        f"Could not build {k} stratified, group-aware inner folds with every class in every "
        f"split after {max_reseeds + 1} seeds. Use fewer inner folds or more patients."
    )
