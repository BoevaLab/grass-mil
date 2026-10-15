"""Per-niche summary and per-unit niche composition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grass_mil.spatial.stats import bootstrap_mean

from .cells import Cells
from .discovery import BACKGROUND, niche_name

__all__ = ["niche_table", "unit_niche_shares"]


def unit_niche_shares(niches: np.ndarray, cells: Cells) -> pd.DataFrame:
    """Fraction of every unit's cells in every niche (rows: units)."""
    frame = pd.crosstab(
        pd.Series(cells.unit_of_cell, name="unit_id"),
        pd.Series(niches, name="niche"),
        normalize="index",
    )
    frame.columns = [niche_name(c) for c in frame.columns]
    return frame.reindex(cells.units).fillna(0.0)


def niche_table(
    niches: np.ndarray,
    cells: Cells,
    margin: np.ndarray,
    *,
    positive: dict[str, bool],
    present_threshold: float,
    enrichment_threshold: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """One row per niche (Background last).

    * ``share``: fraction of the analysed cells in the niche;
    * ``n_units``: units where it makes up more than ``present_threshold``;
    * ``association``: fraction of the niche's cells that come from positive
      units (cell-weighted; compare with ``base_rate``);
    * ``margin`` [``margin_lo``, ``margin_hi``]: mean over units of the mean
      cell margin within the niche, with a 95% bootstrap interval over units;
    * ``sign_consistency``: fraction of those units where the niche's margin
      has the sign of the overall margin;
    * ``frac_<cell type>``: composition; ``enriched``: cell types over-represented
      by more than ``enrichment_threshold`` relative to all analysed cells.
    """
    names = cells.cell_type_names
    k = len(names)
    has_types = k > 0 and (cells.cell_type >= 0).all()
    cohort = np.bincount(cells.cell_type, minlength=k) / cells.n if has_types else None
    is_pos = np.array([positive.get(u, False) for u in cells.unit_of_cell])
    shares = unit_niche_shares(niches, cells)
    rows = []
    for niche in sorted(set(niches.tolist()) - {BACKGROUND}) + (
        [BACKGROUND] if (niches == BACKGROUND).any() else []
    ):
        m = niches == niche
        per_unit = pd.Series(margin[m]).groupby(cells.unit_of_cell[m]).mean()
        mean, lo, hi = bootstrap_mean(per_unit.to_numpy(), n_boot=n_boot, seed=seed)
        row = {
            "niche": niche_name(niche),
            "niche_id": int(niche),
            "n_cells": int(m.sum()),
            "share": float(m.mean()),
            "n_units": int((shares[niche_name(niche)] > present_threshold).sum()),
            "association": float(is_pos[m].mean()),
            "margin": mean,
            "margin_lo": lo,
            "margin_hi": hi,
            "sign_consistency": float((np.sign(per_unit) == np.sign(mean)).mean())
            if len(per_unit)
            else float("nan"),
        }
        if has_types:
            comp = np.bincount(cells.cell_type[m], minlength=k) / m.sum()
            ratio = comp / np.maximum(cohort, 1e-12)
            row["composition"] = ", ".join(
                f"{names[j]} {100 * comp[j]:.0f}%" for j in np.argsort(-comp)[:3] if comp[j] > 0
            )
            row["enriched"] = ", ".join(
                f"{names[j]} x{ratio[j]:.1f}"
                for j in np.argsort(-ratio)
                if ratio[j] > enrichment_threshold
            )
            row.update({f"frac_{names[j]}": float(comp[j]) for j in range(k)})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["base_rate"] = float(is_pos.mean())
    return table
