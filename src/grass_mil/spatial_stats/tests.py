"""Per-region -> per-patient summaries and two-group tests with BH correction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = ["bh_adjust", "mann_whitney_table", "patient_mean", "clr"]


def bh_adjust(p) -> np.ndarray:
    from scipy.stats import false_discovery_control

    p = np.asarray(p, dtype=float)
    q = np.full(p.size, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        q[ok] = false_discovery_control(p[ok], method="bh")
    return q


def patient_mean(per_region: pd.DataFrame, patient_of: Dict[str, str]) -> pd.DataFrame:
    """Mean of every column per patient (index = region id)."""
    df = per_region.copy()
    df["__patient"] = [patient_of.get(str(r), str(r)) for r in df.index]
    return df.groupby("__patient").mean(numeric_only=True)


def clr(M: np.ndarray, pseudo: float = 1e-4) -> np.ndarray:
    L = np.log(np.asarray(M, dtype=float) + pseudo)
    return L - L.mean(axis=1, keepdims=True)


def mann_whitney_table(
    values: pd.DataFrame, y: pd.Series, *, min_per_group: int = 3
) -> pd.DataFrame:
    """Mann-Whitney U (two-sided) per column between y == 1 and y == 0, BH-adjusted."""
    from scipy import stats

    yy = y.reindex(values.index)
    rows = []
    for col in values.columns:
        v = values[col]
        a = v[yy == 1].dropna()
        b = v[yy == 0].dropna()
        ok = len(a) >= int(min_per_group) and len(b) >= int(min_per_group)
        rows.append(
            dict(
                metric=str(col),
                n_pos=int(len(a)),
                n_neg=int(len(b)),
                mean_pos=float(a.mean()) if len(a) else float("nan"),
                mean_neg=float(b.mean()) if len(b) else float("nan"),
                diff=float(a.mean() - b.mean()) if len(a) and len(b) else float("nan"),
                p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                if ok
                else float("nan"),
            )
        )
    T = pd.DataFrame(rows)
    T["q"] = bh_adjust(T["p"].to_numpy())
    return T.sort_values("p").reset_index(drop=True)
