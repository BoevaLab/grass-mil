"""Per-cell attention-weighted margins and the attention feature space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["RelattnResult", "relattn_margins", "attention_space_from_checkpoint"]


@dataclass
class RelattnResult:
    margin: np.ndarray  # (N,) N_r * [A1 (l1 - b1) - A0 (l0 - b0)]  (binary)
    attention: np.ndarray  # (N, C) attention normalised within region per class
    contributions: np.ndarray  # (N, C) A_c * (l_c - b_c)


def relattn_margins(
    instance_logits: np.ndarray,
    attention_logits: np.ndarray,
    bias: Sequence[float],
    region_ids: np.ndarray,
    *,
    positive_class: int = 1,
) -> RelattnResult:
    """Attention-weighted, region-size-scaled cell margins (``relattn``).

    Attention is a softmax over the region's cells per class, so a cell's
    contribution ``A[i,c] (l[i,c] - b_c)`` sums over the region to the bag
    logit minus bias. Multiplying by the region's cell count ``N_r`` makes the
    quantity comparable across regions of different size (a uniformly attended
    cell gets weight 1). For a binary head the margin contrasts the positive
    class with class 0; for more classes it is one-vs-rest.
    """
    LG = np.asarray(instance_logits, dtype=np.float64)
    AR = np.asarray(attention_logits, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64).reshape(-1)
    rid = np.asarray(region_ids)
    if LG.shape != AR.shape:
        raise ValueError(f"instance_logits {LG.shape} and attention_logits {AR.shape} differ")
    n, c = LG.shape
    if b.size != c:
        raise ValueError(f"bias has {b.size} entries for {c} classes")
    A = np.zeros_like(AR)
    scale = np.zeros(n, dtype=np.float64)
    for r in np.unique(rid):
        m = rid == r
        ar = AR[m]
        a = np.exp(ar - ar.max(axis=0, keepdims=True))
        a /= a.sum(axis=0, keepdims=True)
        A[m] = a
        scale[m] = float(m.sum())
    contrib = A * (LG - b[None, :])
    pos = int(positive_class)
    if c == 1:
        margin = contrib[:, 0]
    elif c == 2:
        margin = contrib[:, pos] - contrib[:, 1 - pos]
    else:
        rest = (contrib.sum(axis=1) - contrib[:, pos]) / (c - 1)
        margin = contrib[:, pos] - rest
    return RelattnResult(
        margin=(margin * scale).astype(np.float32),
        attention=A.astype(np.float32),
        contributions=contrib.astype(np.float32),
    )


def attention_space_from_checkpoint(
    embeddings: np.ndarray, ckpt_path: str, *, batch_size: int = 65536
) -> np.ndarray:
    """The attention head's pre-tanh activation for every cell, from a checkpoint.

    Rebuilds the projected gated attention head from the ``attention.*`` weights
    (dimensions inferred from the tensors) and applies ``attention_space``.
    """
    import torch

    from grass_mil.inference.model_loading import read_checkpoint_state
    from grass_mil.models.components.attention import AttnNetGatedProjected

    state = read_checkpoint_state(ckpt_path)
    att = {k[len("attention.") :]: v for k, v in state.items() if k.startswith("attention.")}
    if "projection.0.weight" not in att:
        raise ValueError(
            f"{ckpt_path} has no projected gated attention (attention.projection.0.weight)."
        )
    proj_dim, in_dim = att["projection.0.weight"].shape
    hidden = att["attention_a.0.weight"].shape[0]
    n_classes = att["attention_c.weight"].shape[0]
    module = AttnNetGatedProjected(
        input_dim=int(in_dim),
        projection_dim=int(proj_dim),
        hidden_dim=int(hidden),
        n_classes=int(n_classes),
    )
    module.load_state_dict(att, strict=True)
    module.eval()
    out = []
    E = np.asarray(embeddings, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, E.shape[0], int(batch_size)):
            out.append(
                module.attention_space(torch.from_numpy(E[i : i + int(batch_size)])).numpy()
            )
    return np.concatenate(out) if out else np.zeros((0, int(hidden)), dtype=np.float32)
