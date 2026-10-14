"""Sensitivity / specificity / PPV / NPV at pre-specified operating rules.

Thresholds are quantile rules applied *within fold* (flag the top fraction q
of units by score), so they do not depend on the fold models' score scales.
Threshold-free summaries (Youden, rule-in, rule-out) are chosen on all data
and reported as optimistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

__all__ = [
    "Confusion",
    "confusion_counts",
    "flag_top_within_fold",
    "likelihood_ratios",
    "metrics_from_confusion",
    "operating_point",
    "operating_point_bootstrap",
    "ppv_npv_at_prevalence",
    "threshold_points",
]


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int


def confusion_counts(pred, y) -> Confusion:
    pred = np.asarray(pred).astype(int).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    return Confusion(
        tp=int(((pred == 1) & (y == 1)).sum()),
        fp=int(((pred == 1) & (y == 0)).sum()),
        fn=int(((pred == 0) & (y == 1)).sum()),
        tn=int(((pred == 0) & (y == 0)).sum()),
    )


def metrics_from_confusion(c: Confusion) -> Dict[str, float]:
    n = c.tp + c.fp + c.fn + c.tn
    return {
        "sensitivity": c.tp / (c.tp + c.fn) if (c.tp + c.fn) else float("nan"),
        "specificity": c.tn / (c.tn + c.fp) if (c.tn + c.fp) else float("nan"),
        "ppv": c.tp / (c.tp + c.fp) if (c.tp + c.fp) else float("nan"),
        "npv": c.tn / (c.tn + c.fn) if (c.tn + c.fn) else float("nan"),
        "flagged": (c.tp + c.fp) / n if n else float("nan"),
        "accuracy": (c.tp + c.tn) / n if n else float("nan"),
        **asdict(c),
    }


def flag_top_within_fold(score, folds, q: float) -> np.ndarray:
    """Flag the top fraction ``q`` of units within each fold (threshold = fold quantile)."""
    s = np.asarray(score, dtype=float).reshape(-1)
    f = np.asarray(folds).reshape(-1)
    pred = np.zeros(s.size, dtype=int)
    for k in np.unique(f):
        m = f == k
        thr = np.quantile(s[m], 1.0 - float(q))
        pred[m] = (s[m] >= thr).astype(int)
    return pred


def operating_point(pred, y) -> Dict[str, float]:
    return metrics_from_confusion(confusion_counts(pred, y))


def operating_point_bootstrap(
    pred, y, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> Dict[str, List[float]]:
    """Percentile bootstrap CIs over units for sensitivity, specificity, PPV, NPV."""
    pred = np.asarray(pred).astype(int).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    rng = np.random.default_rng(seed)
    keys = ("sensitivity", "specificity", "ppv", "npv")
    draws = {k: [] for k in keys}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, y.size, y.size)
        m = operating_point(pred[idx], y[idx])
        for k in keys:
            if m[k] == m[k]:
                draws[k].append(m[k])
    return {
        k: [float(v) for v in np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])]
        if vals
        else [float("nan"), float("nan")]
        for k, vals in draws.items()
    }


def ppv_npv_at_prevalence(
    sensitivity: float, specificity: float, prevalence: float
) -> Dict[str, float]:
    p = float(prevalence)
    ppv_den = sensitivity * p + (1 - specificity) * (1 - p)
    npv_den = specificity * (1 - p) + (1 - sensitivity) * p
    return {
        "prevalence": p,
        "ppv": sensitivity * p / ppv_den if ppv_den > 0 else float("nan"),
        "npv": specificity * (1 - p) / npv_den if npv_den > 0 else float("nan"),
    }


def likelihood_ratios(sensitivity: float, specificity: float) -> Dict[str, float]:
    return {
        "lr_plus": sensitivity / (1 - specificity) if specificity < 1 else float("inf"),
        "lr_minus": (1 - sensitivity) / specificity if specificity > 0 else float("inf"),
    }


def threshold_points(
    score, y, *, rule_out_sens: float = 0.9, rule_in_spec: float = 0.9
) -> Dict[str, float]:
    """Youden-optimal, rule-out and rule-in thresholds chosen on ALL data (optimistic)."""
    from sklearn.metrics import roc_curve

    s = np.asarray(score, dtype=float).reshape(-1)
    yy = np.asarray(y).astype(int).reshape(-1)
    fpr, tpr, thr = roc_curve(yy, s)
    out: Dict[str, float] = {}
    j = int(np.argmax(tpr - fpr))
    out["youden"] = float(thr[j])
    ro = np.where(tpr >= rule_out_sens)[0]
    out["rule_out"] = float(thr[ro[0]]) if ro.size else float("nan")
    ri = np.where(fpr <= 1 - rule_in_spec)[0]
    out["rule_in"] = float(thr[ri[-1]]) if ri.size else float("nan")
    return out
