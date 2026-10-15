"""``grass-mil-spatial-stats``: model-free spatial statistics, compared between classes."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from grass_mil.inference.cell_cache import CellCache
from grass_mil.niches.cells import load_cells, resolve_units

from .graph_stats import contact_counts, cross_l, enrichment_zscores
from .neighbourhoods import fit_neighbourhoods, window_composition
from .stats import clr, mann_whitney_table, patient_mean

__all__ = ["run_spatial_stats", "main"]

log = logging.getLogger(__name__)

ANALYSES = ("enrichment", "neighbourhoods", "contacts", "cross_l")


def run_spatial_stats(cfg: DictConfig) -> Path:
    bad = set(cfg.analyses) - set(ANALYSES)
    if bad:
        raise ValueError(f"Unknown analyses {sorted(bad)}; expected {list(ANALYSES)}.")
    out = Path(str(cfg.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    cache = CellCache.open(str(cfg.cells_dir))
    units_sel = cfg.units if isinstance(cfg.units, str) else OmegaConf.to_container(cfg.units)
    cells = load_cells(cache, resolve_units(units_sel, cache))
    if (cells.cell_type < 0).any():
        raise ValueError("Spatial statistics need cell types.")
    mapping = dict(cfg.group_map or {})
    groups = sorted({mapping.get(n, n) for n in cells.cell_type_names})
    to_group = np.array([groups.index(mapping.get(n, n)) for n in cells.cell_type_names])
    labels = to_group[cells.cell_type]
    g = len(groups)
    reference = [groups.index(r) for r in cfg.reference]
    partners = [j for j in range(g) if j not in reference]
    rng = np.random.default_rng(int(cfg.seed))
    positive = cells.positive(int(cfg.positive_class))
    patient_of = {u: cells.patients.get(u, u) for u in cells.units}
    y = pd.Series({patient_of[u]: int(positive[u]) for u in cells.units})

    per_unit: dict[str, dict[str, dict[str, float]]] = {a: {} for a in cfg.analyses}
    windows = []
    for unit in cells.units:
        m = np.flatnonzero(cells.unit_of_cell == unit)
        local = np.full(cells.n, -1)
        local[m] = np.arange(m.size)
        e = cells.edges[:, np.isin(cells.edges[0], m)]
        e = local[e]
        lab, pos = labels[m], cells.pos[m]
        if "enrichment" in cfg.analyses:
            z = enrichment_zscores(lab, e, g, n_perms=int(cfg.n_perms), rng=rng)
            per_unit["enrichment"][unit] = {
                f"{groups[a]}|{groups[b]}": z[a, b] for a in range(g) for b in range(a, g)
            }
        if "neighbourhoods" in cfg.analyses:
            windows.append(window_composition(pos, lab, g, int(cfg.neighbourhoods.window)))
        if "contacts" in cfg.analyses and reference:
            counts = contact_counts(lab, e, g)
            ref_ends = counts[reference].sum()
            per_unit["contacts"][unit] = {
                f"{groups[r]}->{groups[p]}": counts[r, p] / ref_ends if ref_ends else np.nan
                for r in reference
                for p in range(g)
            }
        if "cross_l" in cfg.analyses and reference:
            row = {}
            for r in reference:
                for p in [*reference, *partners]:
                    if (lab == r).sum() >= 2 and (lab == p).sum() >= 2:
                        curve = cross_l(
                            pos, lab == r, lab == p, np.asarray(cfg.radii, dtype=float)
                        )
                        row.update(
                            {
                                f"{groups[r]}->{groups[p]}@{rad:g}": v
                                for rad, v in zip(cfg.radii, curve)
                            }
                        )
            per_unit["cross_l"][unit] = row
    if "neighbourhoods" in cfg.analyses:
        n_cn = int(cfg.neighbourhoods.n)
        cn_labels, centres = fit_neighbourhoods(
            windows, n_cn, int(cfg.neighbourhoods.sample_size), int(cfg.seed)
        )
        per_unit["neighbourhoods"] = {
            u: {f"CN{k + 1}": float(np.mean(lab == k)) for k in range(n_cn)}
            for u, lab in zip(cells.units, cn_labels)
        }
        pd.DataFrame(centres, columns=groups, index=[f"CN{k + 1}" for k in range(n_cn)]).to_csv(
            out / "neighbourhood_centres.csv"
        )

    for analysis, rows in per_unit.items():
        if not rows:
            continue
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "unit_id"
        frame.to_csv(out / f"{analysis}_per_unit.csv")
        patients = patient_mean(frame, patient_of)
        if analysis == "neighbourhoods":
            patients = pd.DataFrame(
                clr(patients.to_numpy()), index=patients.index, columns=patients.columns
            )
        mann_whitney_table(patients, y, min_per_group=int(cfg.min_patients)).to_csv(
            out / f"{analysis}_tests.csv", index=False
        )
    log.info(f"Spatial statistics of {len(cells.units)} units written to {out}")
    return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="spatial_stats.yaml")
def main(cfg: DictConfig) -> None:
    run_spatial_stats(cfg)


if __name__ == "__main__":
    main()
