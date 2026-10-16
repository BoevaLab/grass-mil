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

from .graph_stats import contact_counts, cross_l, enrichment
from .neighbourhoods import fit_neighbourhoods, window_composition
from .stats import clr, group_tests, patient_labels

__all__ = ["run_spatial_stats", "main"]

log = logging.getLogger(__name__)

ANALYSES = ("enrichment", "neighbourhoods", "contacts", "cross_l")


def run_spatial_stats(cfg: DictConfig) -> Path:
    bad = set(cfg.analyses) - set(ANALYSES)
    if bad:
        raise ValueError(f"Unknown analyses {sorted(bad)}; expected {list(ANALYSES)}.")
    if cfg.compare_by not in ("patient", "unit"):
        raise ValueError("compare_by must be 'patient' or 'unit'.")
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
    reference_pairs = [(r, p) for r in reference for p in range(g)]
    radii = np.asarray(cfg.radii, dtype=float)
    rng = np.random.default_rng(int(cfg.seed))
    if cfg.compare_by == "patient":
        patient_of = {u: cells.patient_of(u) for u in cells.units}
    else:
        patient_of = {u: u for u in cells.units}
        if len({cells.patient_of(u) for u in cells.units}) < len(cells.units):
            log.warning(
                "compare_by=unit tests units of the same patient as independent samples; "
                "the p-values are anti-conservative."
            )
    y = patient_labels(cells.positive(int(cfg.positive_class)), patient_of)

    per_unit: dict[str, dict[str, dict[str, float]]] = {a: {} for a in cfg.analyses}
    windows = []
    for unit in cells.units:
        block, edges = cells.unit_graph(unit)
        lab, pos = labels[block], cells.pos[block]
        if "enrichment" in cfg.analyses:
            _, ratio = enrichment(lab, edges, g, n_perms=int(cfg.n_perms), rng=rng)
            per_unit["enrichment"][unit] = {
                f"{groups[a]}|{groups[b]}": ratio[a, b] for a in range(g) for b in range(a, g)
            }
        if "neighbourhoods" in cfg.analyses:
            windows.append(window_composition(pos, lab, g, int(cfg.neighbourhoods.window)))
        if "contacts" in cfg.analyses and reference:
            counts = contact_counts(lab, edges, g)
            ends = counts[reference].sum()
            per_unit["contacts"][unit] = {
                f"{groups[r]}->{groups[p]}": counts[r, p] / ends if ends else np.nan
                for r, p in reference_pairs
            }
        if "cross_l" in cfg.analyses and reference:
            curves = cross_l(pos, lab, reference_pairs, radii)
            per_unit["cross_l"][unit] = {
                f"{groups[r]}->{groups[p]}@{radius:g}": value
                for (r, p), curve in zip(reference_pairs, curves)
                for radius, value in zip(radii, curve)
            }
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
        compared = frame.groupby([patient_of[u] for u in frame.index]).mean()
        if analysis == "neighbourhoods":
            compared = pd.DataFrame(
                clr(compared.to_numpy()), index=compared.index, columns=compared.columns
            )
        group_tests(compared, y, min_per_group=int(cfg.min_patients)).to_csv(
            out / f"{analysis}_tests.csv", index=False
        )
    log.info(f"Spatial statistics of {len(cells.units)} units written to {out}")
    return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="spatial_stats.yaml")
def main(cfg: DictConfig) -> None:
    run_spatial_stats(cfg)


if __name__ == "__main__":
    main()
