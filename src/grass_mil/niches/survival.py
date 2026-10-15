"""Niche shares versus survival.

Per patient, the share of every niche is averaged over the patient's units and
summarised as the z-scored log share. Each niche gets a univariate Cox hazard
ratio (per standard deviation of the log share) with a Wald test and BH
correction. Optionally, patients are split into tertiles of a niche score (a
signed sum of z-scored log shares, e.g. ``["Niche 3", "-Niche 4"]``) and
compared with Kaplan-Meier curves and a log-rank test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from grass_mil.spatial.stats import bh
from grass_mil.utils import checked

__all__ = ["SurvivalConfig", "niche_survival", "cox_univariate", "kaplan_meier", "logrank"]


@dataclass(frozen=True)
class SurvivalConfig:
    file: str | None = None
    id_column: str = "patient_id"
    time_column: str = "time"
    event_column: str = "event"
    pseudo_share: float = 1e-3
    score: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> SurvivalConfig:
        cfg = checked(cls, cfg, "survival")
        cfg["score"] = tuple(cfg.get("score") or ())
        return cls(**cfg)


def cox_univariate(
    x: np.ndarray, time: np.ndarray, event: np.ndarray, max_iter: int = 50
) -> tuple[float, float, float, float]:
    """Cox model with one covariate (Breslow ties): ``(hr, lo, hi, p)`` with a 95% Wald interval.

    Newton-Raphson with step halving; NaN when the fit does not converge (for
    example a covariate that separates events perfectly).
    """
    from scipy.stats import norm

    order = np.argsort(-time, kind="stable")
    x, died, t = x[order], event[order] > 0, time[order]
    # Risk set of subject i: everyone with time >= t_i, including later ties.
    last = np.searchsorted(-t, -t, side="right") - 1

    def fit(beta: float) -> tuple[float, float, float]:
        w = np.exp(beta * x)
        s0, s1, s2 = (np.cumsum(w * x**power)[last][died] for power in (0, 1, 2))
        mean = s1 / s0
        loglik = float(np.sum(beta * x[died] - np.log(s0)))
        return loglik, float(np.sum(x[died] - mean)), float(np.sum(s2 / s0 - mean**2))

    beta = 0.0
    loglik, grad, info = fit(beta)
    for _ in range(max_iter):
        if not info > 0:
            break
        step = grad / info
        while (trial := fit(beta + step))[0] < loglik and abs(step) > 1e-12:
            step /= 2.0
        beta += step
        loglik, grad, info = trial
        if abs(step) < 1e-9 and info > 0:
            se = 1.0 / np.sqrt(info)
            hr, lo, hi = np.exp([beta, beta - 1.96 * se, beta + 1.96 * se])
            return float(hr), float(lo), float(hi), float(2 * norm.sf(abs(beta / se)))
    return (float("nan"),) * 4


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step function ``(times, survival)`` starting at ``(0, 1)``."""
    times = np.unique(time[event > 0])
    surv, s = [1.0], 1.0
    for t in times:
        at_risk = (time >= t).sum()
        s *= 1.0 - ((time == t) & (event > 0)).sum() / at_risk
        surv.append(s)
    return np.r_[0.0, times], np.asarray(surv)


def logrank(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> float:
    """p-value of the k-group log-rank test."""
    from scipy.stats import chi2

    groups = np.unique(group)
    o_minus_e = np.zeros(len(groups))
    cov = np.zeros((len(groups), len(groups)))
    for t in np.unique(time[event > 0]):
        at_risk = time >= t
        n = at_risk.sum()
        d = ((time == t) & (event > 0)).sum()
        n_g = np.array([(at_risk & (group == g)).sum() for g in groups])
        d_g = np.array([((time == t) & (event > 0) & (group == g)).sum() for g in groups])
        o_minus_e += d_g - d * n_g / n
        if n > 1:
            f = d * (n - d) / (n * n * (n - 1))
            cov += f * (np.diag(n_g * n) - np.outer(n_g, n_g))
    stat = float(o_minus_e[:-1] @ np.linalg.pinv(cov[:-1, :-1]) @ o_minus_e[:-1])
    return float(chi2.sf(stat, len(groups) - 1))


def niche_survival(patient_shares: pd.DataFrame, outcomes: pd.DataFrame, cfg: SurvivalConfig):
    """Cox table per niche and, with ``cfg.score``, the patients' score tertiles.

    ``patient_shares``: patients x niches; ``outcomes``: patients x [time, event].
    """
    data = patient_shares.join(outcomes, how="inner").dropna(subset=["time", "event"])
    niches = [c for c in patient_shares.columns if c != "Background"]
    z = np.log(data[niches] + cfg.pseudo_share)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    time, event = data["time"].to_numpy(float), data["event"].to_numpy(float)
    rows = []
    for niche in niches:
        hr, lo, hi, p = cox_univariate(z[niche].fillna(0).to_numpy(), time, event)
        rows.append({"niche": niche, "hazard_ratio": hr, "hr_lo": lo, "hr_hi": hi, "p": p})
    table = pd.DataFrame(rows)
    table["q"] = bh(table["p"].to_numpy())
    groups = None
    if cfg.score:
        score = np.zeros(len(data))
        for term in cfg.score:
            sign, name = (-1.0, term[1:]) if term.startswith("-") else (1.0, term)
            if name not in z.columns:
                raise ValueError(f"survival.score names an unknown niche '{name}'.")
            score += sign * z[name].fillna(0).to_numpy()
        tertile = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        groups = pd.DataFrame(
            {"score": score, "tertile": tertile, "time": time, "event": event}, index=data.index
        )
        groups.attrs["logrank_p"] = logrank(time, event, tertile)
    return table, groups
