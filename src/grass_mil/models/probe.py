"""Initialising a linear instance head from a probe on frozen ego-graph embeddings.

Every ego-graph inherits the label of its unit. A standardised, L2-penalised
logistic regression (classification), ridge regression (regression) or Cox
model (survival, Breslow ties) is fitted on the embeddings of the training
units and folded, scaler included, into the weights of a
``Linear(embedding_dim, num_logits)`` head. For two classes the class-0 row is
zero and the class-1 row carries the decision function, so the softmax of the
two logits equals the logistic probability. ``C`` is the inverse penalty
strength in every case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from grass_mil.utils import checked

__all__ = ["ProbeConfig", "collect_embeddings", "fit_linear_head"]


@dataclass(frozen=True)
class ProbeConfig:
    roots_per_unit: int | None = 96
    seed: int = 0
    C: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 3000

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> ProbeConfig:
        cfg = checked(cls, cfg, "finetune.probe")
        return cls(**cfg)


@torch.no_grad()
def collect_embeddings(
    encoder: torch.nn.Module, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings and inherited targets of every ego-graph of ``loader``."""
    was_training = encoder.training
    encoder.eval()
    embeddings, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        embeddings.append(encoder(batch).float().cpu().numpy())
        targets.append(batch.graph_y.float().cpu().numpy())
    encoder.train(was_training)
    if not embeddings:
        raise ValueError("The probe found no ego-graphs in the training split.")
    return np.concatenate(embeddings), np.concatenate(targets)


def fit_linear_head(
    embeddings: np.ndarray,
    targets: np.ndarray,
    *,
    kind: str,
    num_logits: int,
    cfg: ProbeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight [num_logits, dim], bias [num_logits])`` of the folded probe."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(embeddings)
    z = scaler.transform(embeddings)
    if kind == "classification":
        y = targets[:, 0].astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("The probe's training split holds a single class.")
        model = LogisticRegression(C=cfg.C, class_weight=cfg.class_weight, max_iter=cfg.max_iter)
        model.fit(z, y)
        coef, intercept = model.coef_, model.intercept_
    elif kind == "regression":
        model = Ridge(alpha=1.0 / cfg.C).fit(z, targets)
        coef, intercept = np.atleast_2d(model.coef_), np.atleast_1d(model.intercept_)
    elif kind == "survival":
        coef, intercept = _cox(z, targets[:, 0], targets[:, 1], cfg)[None, :], np.zeros(1)
    else:
        raise ValueError(f"No probe for target type '{kind}'.")
    w = coef / scaler.scale_[None, :]
    b = intercept - (coef * scaler.mean_[None, :] / scaler.scale_[None, :]).sum(axis=1)
    if coef.shape[0] not in (1, num_logits):
        raise ValueError(
            f"The probe saw {coef.shape[0]} classes but the head has {num_logits} logits."
        )
    weight = torch.zeros(num_logits, embeddings.shape[1])
    bias = torch.zeros(num_logits)
    if kind == "classification" and coef.shape[0] == 1:
        weight[1], bias[1] = torch.from_numpy(w[0]).float(), float(b[0])
    else:
        weight[:], bias[:] = torch.from_numpy(w).float(), torch.from_numpy(b).float()
    return weight, bias


def _cox(z: np.ndarray, time: np.ndarray, event: np.ndarray, cfg: ProbeConfig) -> np.ndarray:
    """Coefficients minimising the Cox negative log partial likelihood + ``|w|^2 / (2C)``."""
    from .objectives import cox_loss

    n_events = float((event > 0).sum())
    if n_events == 0:
        raise ValueError("The probe's training split holds no event.")
    x, t, e = (torch.as_tensor(a, dtype=torch.float64) for a in (z, time, event))
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=cfg.max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = cox_loss(x @ w, t, e) + (w @ w) / (2.0 * cfg.C * n_events)
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach().numpy()
