"""Group comparisons and summaries shared by every analysis."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["bh", "bootstrap_mean", "clr", "patient_mean", "mann_whitney_table"]


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaNs kept)."""
    from scipy.stats import false_discovery_control

    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = false_discovery_control(p[ok], method="bh")
    return q


def bootstrap_mean(values, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap interval of the finite ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.random.default_rng(seed).integers(0, v.size, size=(int(n_boot), v.size))
    means = v[draws].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def clr(shares: np.ndarray, pseudo: float = 1e-3) -> np.ndarray:
    """Centred log-ratio of composition rows."""
    logs = np.log(np.asarray(shares, dtype=float) + pseudo)
    return logs - logs.mean(axis=1, keepdims=True)


def patient_mean(per_unit: pd.DataFrame, patient_of: Mapping[str, str]) -> pd.DataFrame:
    """Mean of every column over the units of each patient (index: unit id)."""
    patients = [patient_of.get(str(u), str(u)) for u in per_unit.index]
    return per_unit.groupby(patients).mean(numeric_only=True)


def mann_whitney_table(
    values: pd.DataFrame, y: pd.Series, *, min_per_group: int = 3
) -> pd.DataFrame:
    """Two-sided Mann-Whitney U per column between ``y == 1`` and ``y == 0`` (BH-adjusted)."""
    from scipy.stats import mannwhitneyu

    y = y.reindex(values.index)
    rows = []
    for column in values.columns:
        a = values.loc[y == 1, column].dropna()
        b = values.loc[y == 0, column].dropna()
        enough = len(a) >= min_per_group and len(b) >= min_per_group
        rows.append(
            {
                "metric": str(column),
                "n_pos": len(a),
                "n_neg": len(b),
                "mean_pos": float(a.mean()) if len(a) else float("nan"),
                "mean_neg": float(b.mean()) if len(b) else float("nan"),
                "p": float(mannwhitneyu(a, b).pvalue) if enough else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    table["diff"] = table["mean_pos"] - table["mean_neg"]
    table["q"] = bh(table["p"].to_numpy())
    return table.sort_values("p").reset_index(drop=True)
