"""Contact fractions on the cell graph."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["contact_fractions"]


def contact_fractions(
    labels: np.ndarray,
    edge_index: np.ndarray,
    *,
    group_names: Sequence[str],
    reference: Sequence[str],
    partners: Sequence[str],
    infiltrating: Sequence[str] = (),
    min_cells: int = 5,
) -> Dict[str, float]:
    """Fraction of reference cells touching >= 1 partner cell, homotypic contact of
    the reference group, and for each infiltrating group the fraction of its cells
    whose neighbours are mostly reference cells."""
    labels = np.asarray(labels).astype(int)
    ei = np.asarray(edge_index).astype(int)
    names = list(group_names)
    gl = np.asarray(names, dtype=object)[labels]
    n = labels.size
    src = np.concatenate([ei[0], ei[1]])
    dst = np.concatenate([ei[1], ei[0]])
    ref = np.isin(gl, list(reference))
    out: Dict[str, float] = {}
    for x in partners:
        has = np.zeros(n, dtype=bool)
        has[src[gl[dst] == x]] = True
        out[f"contact_{x}"] = float(has[ref].mean()) if ref.sum() else float("nan")
    has = np.zeros(n, dtype=bool)
    has[src[ref[dst]]] = True
    out["homotypic"] = float(has[ref].mean()) if ref.sum() else float("nan")
    n_ref_nb = np.zeros(n)
    deg = np.zeros(n)
    np.add.at(n_ref_nb, src, ref[dst].astype(float))
    np.add.at(deg, src, 1.0)
    frac_ref = n_ref_nb / np.maximum(deg, 1.0)
    out["reference_fraction"] = float(ref.mean())
    for x in infiltrating:
        mk = gl == x
        out[f"{x}_intra_reference"] = (
            float((frac_ref[mk] > 0.5).mean()) if mk.sum() >= int(min_cells) else float("nan")
        )
    if infiltrating:
        imm = np.isin(gl, list(infiltrating))
        out["infiltration"] = float((frac_ref[imm] > 0.5).mean()) if imm.sum() else float("nan")
        out["infiltrating_fraction"] = float(imm.mean())
    return out
