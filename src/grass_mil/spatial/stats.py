"""Group comparisons and summaries shared by every analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["bh", "bootstrap_mean", "clr", "patient_labels", "group_tests"]


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaNs kept)."""
    from scipy.stats import false_discovery_control

    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = false_discovery_control(p[ok], method="bh")
    return q


def bootstrap_mean(
    values, *, clusters: Sequence | None = None, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of the finite ``values`` and its 95% percentile bootstrap interval.

    With ``clusters`` (for example the patient of every value) whole clusters
    are resampled, so values of one patient do not count as independent. The
    interval is NaN with fewer than two clusters.
    """
    v = np.asarray(values, dtype=float)
    c = np.arange(v.size) if clusters is None else np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    _, c = np.unique(c, return_inverse=True)
    if c.max() == 0:
        return float(v.mean()), float("nan"), float("nan")
    sums, sizes = np.bincount(c, weights=v), np.bincount(c)
    draws = np.random.default_rng(seed).integers(0, sums.size, size=(int(n_boot), sums.size))
    means = sums[draws].sum(1) / sizes[draws].sum(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def clr(shares: np.ndarray, pseudo: float = 1e-3) -> np.ndarray:
    """Centred log-ratio of composition rows."""
    logs = np.log(np.asarray(shares, dtype=float) + pseudo)
    return logs - logs.mean(axis=1, keepdims=True)


def patient_labels(positive: Mapping[str, bool], patient_of: Mapping[str, str]) -> pd.Series:
    """Positive patients (any of their units positive), indexed by patient."""
    units = pd.Series({u: bool(p) for u, p in positive.items()})
    return units.groupby([patient_of.get(u, u) for u in units.index]).max()


def group_tests(values: pd.DataFrame, y: pd.Series, *, min_per_group: int = 3) -> pd.DataFrame:
    """Per column, positive (``y`` true) against other rows: means, AUROC and a
    two-sided Mann-Whitney U test with BH correction over the columns."""
    from scipy.stats import mannwhitneyu

    y = y.reindex(values.index).fillna(False).astype(bool)
    rows = []
    for column in values.columns:
        a = values.loc[y, column].dropna()
        b = values.loc[~y, column].dropna()
        row = {
            "metric": str(column),
            "n_pos": len(a),
            "n_neg": len(b),
            "mean_pos": float(a.mean()) if len(a) else float("nan"),
            "mean_neg": float(b.mean()) if len(b) else float("nan"),
            "auroc": float("nan"),
            "p": float("nan"),
        }
        if len(a) >= min_per_group and len(b) >= min_per_group:
            test = mannwhitneyu(a, b)
            row["auroc"] = float(test.statistic / (len(a) * len(b)))
            row["p"] = float(test.pvalue)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["diff"] = table["mean_pos"] - table["mean_neg"]
    table["q"] = bh(table["p"].to_numpy())
    return table.sort_values("p").reset_index(drop=True)
