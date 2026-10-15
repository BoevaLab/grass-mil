"""Unit-level prediction tables, shared by training, cross-validation and prediction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

__all__ = ["prediction_frame", "patient_frame", "frame_metrics", "score_column"]


def score_column(target_type: str) -> str:
    """The column holding the score that metrics such as AUROC are computed on."""
    return {"classification": "prob_1", "regression": "pred_0", "survival": "risk"}[target_type]


def prediction_frame(
    unit_ids: Sequence[str],
    logits: torch.Tensor,
    *,
    target_type: str,
    targets: torch.Tensor | None = None,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per unit: ids, class probabilities / predictions / risk, and targets.

    ``meta`` (indexed by unit id) adds columns such as ``region_id``,
    ``sample_id`` and ``patient_id``.
    """
    logits = logits.detach().float().cpu()
    df = pd.DataFrame({"unit_id": [str(u) for u in unit_ids]})
    if target_type == "classification":
        probs = torch.softmax(logits, dim=1).numpy()
        for c in range(probs.shape[1]):
            df[f"prob_{c}"] = probs[:, c]
        df["pred"] = probs.argmax(1)
    elif target_type == "regression":
        for c in range(logits.shape[1]):
            df[f"pred_{c}"] = logits[:, c].numpy()
    else:
        df["risk"] = logits[:, 0].numpy()
    if targets is not None:
        t = targets.detach().float().cpu().numpy()
        if target_type == "classification":
            df["label"] = np.floor(t[:, 0] + 0.5).astype(int)
        elif target_type == "survival":
            df["time"], df["event"] = t[:, 0], t[:, 1]
        else:
            for c in range(t.shape[1]):
                df[f"target_{c}"] = t[:, c]
    if meta is not None:
        df = df.join(meta, on="unit_id")
    return df


def patient_frame(units: pd.DataFrame, target_type: str) -> pd.DataFrame:
    """Patient scores as the mean of the patient's unit scores."""
    if "patient_id" not in units.columns:
        raise ValueError("Patient-level scores need a patient_id column.")
    value_cols = [c for c in units.columns if c.startswith(("prob_", "pred_")) or c == "risk"]
    agg: dict[str, tuple[str, str]] = {c: (c, "mean") for c in value_cols}
    agg["n_units"] = ("unit_id", "nunique")
    for c in ("label", "time", "event", *[c for c in units.columns if c.startswith("target_")]):
        if c in units.columns:
            agg[c] = (c, "first")
    for c in ("outer",):
        if c in units.columns:
            agg[c] = (c, "first")
    out = units.groupby("patient_id").agg(**agg).reset_index()
    if target_type == "classification":
        probs = out[[c for c in out.columns if c.startswith("prob_")]].to_numpy()
        out["pred"] = probs.argmax(1)
    return out


def frame_metrics(frame: pd.DataFrame, target_type: str) -> dict[str, float]:
    """Region-level metrics of a prediction table (unit or patient rows)."""
    from grass_mil.models.objectives import region_metrics

    if target_type == "classification":
        probs = frame[[c for c in frame.columns if c.startswith("prob_")]].to_numpy()
        logits = torch.log(torch.as_tensor(probs, dtype=torch.float32).clamp_min(1e-12))
        targets = torch.as_tensor(frame[["label"]].to_numpy(), dtype=torch.float32)
    elif target_type == "regression":
        logits = torch.as_tensor(
            frame[[c for c in frame.columns if c.startswith("pred_")]].to_numpy(),
            dtype=torch.float32,
        )
        targets = torch.as_tensor(
            frame[[c for c in frame.columns if c.startswith("target_")]].to_numpy(),
            dtype=torch.float32,
        )
    else:
        logits = torch.as_tensor(frame[["risk"]].to_numpy(), dtype=torch.float32)
        targets = torch.as_tensor(frame[["time", "event"]].to_numpy(), dtype=torch.float32)
    return region_metrics(target_type, logits, targets)
