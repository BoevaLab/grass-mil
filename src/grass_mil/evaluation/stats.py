"""AUROC statistics for held-out (out-of-fold) scores.

Fold models of a cross-validation have different score scales, so pooled
AUROCs are computed on scores standardised within fold (z-score or rank);
per-fold means are reported alongside. The DeLong test uses the Sun & Xu
structural components for two correlated AUROCs on the same labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "AurocCI",
    "DeLongResult",
    "auroc",
    "bootstrap_auroc_ci",
    "delong_test",
    "per_fold_auroc",
    "rank_within_fold",
    "stratified_auroc",
    "z_within_fold",
]


def _arr(y, s) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores differ in length")
    return y, s


def auroc(y, s) -> float:
    """AUROC via the Mann-Whitney statistic (ties count one half); nan if one class."""
    y, s = _arr(y, s)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    )


@dataclass
class AurocCI:
    auroc: float
    lower: float
    upper: float
    n_boot: int
    n_valid: int


def bootstrap_auroc_ci(
    y, s, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05, groups=None
) -> AurocCI:
    """Percentile bootstrap CI over samples (or over ``groups``, e.g. patients)."""
    y, s = _arr(y, s)
    rng = np.random.default_rng(seed)
    point = auroc(y, s)
    vals = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(int(n_boot)):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([members[uniq[i]] for i in pick])
            if 0 < y[idx].sum() < idx.size:
                vals.append(auroc(y[idx], s[idx]))
    else:
        n = y.size
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            if 0 < y[idx].sum() < n:
                vals.append(auroc(y[idx], s[idx]))
    if not vals:
        return AurocCI(point, float("nan"), float("nan"), int(n_boot), 0)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return AurocCI(point, float(lo), float(hi), int(n_boot), len(vals))


@dataclass
class DeLongResult:
    auroc_1: float
    auroc_2: float
    delta: float
    z: float
    p_value: float
    variance: float


def _structural_components(s: np.ndarray, pos: np.ndarray, neg: np.ndarray):
    sp, sn = s[pos], s[neg]
    v10 = np.array([(np.sum(sn < x) + 0.5 * np.sum(sn == x)) / sn.size for x in sp])
    v01 = np.array([(np.sum(sp > x) + 0.5 * np.sum(sp == x)) / sp.size for x in sn])
    return v10, v01


def delong_test(y, s1, s2) -> DeLongResult:
    """DeLong test for two correlated AUROCs on the same labels (Sun & Xu components)."""
    from scipy import stats

    y, s1 = _arr(y, s1)
    _, s2 = _arr(y, s2)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if pos.size < 2 or neg.size < 2:
        raise ValueError("DeLong needs at least two positives and two negatives.")
    a10, a01 = _structural_components(s1, pos, neg)
    b10, b01 = _structural_components(s2, pos, neg)
    auc1, auc2 = float(a10.mean()), float(b10.mean())
    s10 = np.cov(np.stack([a10, b10]))
    s01 = np.cov(np.stack([a01, b01]))
    S = s10 / pos.size + s01 / neg.size
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    z = (auc1 - auc2) / np.sqrt(var) if var > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return DeLongResult(auc1, auc2, auc1 - auc2, float(z), p, var)


def z_within_fold(scores, folds) -> np.ndarray:
    """Standardise scores to zero mean / unit sd within each fold."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    f = np.asarray(folds).reshape(-1)
    out = np.zeros_like(s)
    for k in np.unique(f):
        m = f == k
        sd = s[m].std()
        out[m] = (s[m] - s[m].mean()) / (sd if sd > 0 else 1.0)
    return out


def rank_within_fold(scores, folds) -> np.ndarray:
    """Map scores to (rank - 0.5) / n within each fold (uniform on (0, 1))."""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=float).reshape(-1)
    f = np.asarray(folds).reshape(-1)
    out = np.zeros_like(s)
    for k in np.unique(f):
        m = f == k
        out[m] = (rankdata(s[m]) - 0.5) / m.sum()
    return out


def per_fold_auroc(y, s, folds) -> Dict[str, float]:
    """AUROC within each fold plus their mean and sd (the preprint's headline number)."""
    y, s = _arr(y, s)
    f = np.asarray(folds).reshape(-1)
    vals = {}
    for k in sorted(np.unique(f), key=str):
        m = f == k
        vals[str(k)] = auroc(y[m], s[m])
    arr = np.array([v for v in vals.values() if v == v])
    out = {f"fold_{k}": v for k, v in vals.items()}
    out["mean"] = float(arr.mean()) if arr.size else float("nan")
    out["sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    out["n_folds"] = int(arr.size)
    return out


def stratified_auroc(
    y, s, strata, *, n_boot: int = 2000, seed: int = 0
) -> Dict[str, Dict[str, float]]:
    """AUROC (with bootstrap CI) within each level of a stratifying variable."""
    y, s = _arr(y, s)
    st = np.asarray(strata).reshape(-1)
    out: Dict[str, Dict[str, float]] = {}
    for level in sorted(np.unique(st), key=str):
        m = st == level
        ci = bootstrap_auroc_ci(y[m], s[m], n_boot=n_boot, seed=seed)
        out[str(level)] = {
            "n": int(m.sum()),
            "n_pos": int(y[m].sum()),
            "auroc": ci.auroc,
            "lower": ci.lower,
            "upper": ci.upper,
        }
    return out
