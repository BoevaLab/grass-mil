"""Classical spatial descriptors of every niche.

Each statistic is computed on the cells of one niche within one unit (the
niche's subgraph: its cells and the cell-graph edges between them) and then
averaged over the units in which the niche has at least ``min_cells`` cells:

* ``moran``: Moran's I of every cell-type indicator (are cells of a type
  clustered within the niche?), mean and s.e.m. over units, with a
  label-permutation p-value of the mean and BH correction over niche x type;
* ``enrichment``: neighbourhood-enrichment z-score of every cell-type pair;
* ``cross_l``: Ripley's centred cross-L of every cell-type pair on a shared
  radius grid; ``r_star`` is the radius maximising the mean absolute value
  over pairs, and the table reports ``L(r*) - r*``;
* ``contacts``: ``log2`` of a pair's share of the contacts inside the niche
  over its share of all contacts of the analysed cells (all units pooled),
  masked below ``min_contacts`` contacts inside the niche.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np
import pandas as pd

from grass_mil.spatial.graph_stats import contact_counts, cross_l, enrichment_zscores, morans_i
from grass_mil.spatial.stats import bh

from .cells import Cells
from .discovery import BACKGROUND, niche_name

__all__ = ["DescriptorConfig", "niche_descriptors"]

ANALYSES = ("moran", "enrichment", "cross_l", "contacts")


@dataclass(frozen=True)
class DescriptorConfig:
    analyses: tuple[str, ...] = ANALYSES
    min_cells: int = 20
    n_perms: int = 100
    n_radii: int = 20
    max_radius: float | None = None  # null = a quarter of the median unit diagonal
    min_contacts: int = 100
    max_edge_length: float | None = None  # drop longer cell-graph edges first
    seed: int = 0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> DescriptorConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown descriptors keys: {sorted(unknown)}")
        cfg["analyses"] = tuple(cfg.get("analyses", ANALYSES))
        bad = set(cfg["analyses"]) - set(ANALYSES)
        if bad:
            raise ValueError(
                f"Unknown descriptor analyses {sorted(bad)}; expected {list(ANALYSES)}."
            )
        return cls(**cfg)


def _subgraphs(cells: Cells, niches: np.ndarray, edges: np.ndarray, min_cells: int):
    """``(niche, unit, cell indices, local edges)`` of every niche subgraph with enough cells."""
    _, unit_index = np.unique(cells.unit_of_cell, return_inverse=True)
    key = niches.astype(np.int64) * (int(unit_index.max()) + 1) + unit_index
    order = np.argsort(key, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(key[order]) != 0])
    # Edges inside one subgraph (same niche; units never share edges), grouped by key.
    inside = edges[:, key[edges[0]] == key[edges[1]]]
    edge_key = key[inside[0]]
    edge_order = np.argsort(edge_key, kind="stable")
    inside, edge_key = inside[:, edge_order], edge_key[edge_order]
    local = np.empty(cells.n, dtype=np.int64)
    for idx in np.split(order, starts[1:]):
        if idx.size < min_cells or niches[idx[0]] == BACKGROUND:
            continue
        k = key[idx[0]]
        lo, hi = np.searchsorted(edge_key, [k, k + 1])
        local[idx] = np.arange(idx.size)
        yield int(niches[idx[0]]), str(cells.unit_of_cell[idx[0]]), idx, local[inside[:, lo:hi]]


def niche_descriptors(
    niches: np.ndarray, cells: Cells, cfg: DescriptorConfig
) -> dict[str, pd.DataFrame]:
    if (cells.cell_type < 0).any():
        return {}
    names = cells.cell_type_names
    k = len(names)
    edges = cells.edges
    if cfg.max_edge_length is not None:
        length = np.linalg.norm(cells.pos[edges[0]] - cells.pos[edges[1]], axis=1)
        edges = edges[:, length <= cfg.max_edge_length]
    rng = np.random.default_rng(cfg.seed)
    radii = _radii(cells, cfg)
    pairs = [(a, b) for a in range(k) for b in range(a, k)]
    collected: dict[str, list] = {a: [] for a in cfg.analyses}
    for niche, unit, idx, local_edges in _subgraphs(cells, niches, edges, cfg.min_cells):
        types = cells.cell_type[idx]
        if "moran" in cfg.analyses:
            onehot = np.eye(k)[types]
            observed = morans_i(onehot, local_edges)
            null = np.stack(
                [
                    morans_i(onehot[rng.permutation(idx.size)], local_edges)
                    for _ in range(cfg.n_perms)
                ]
            )
            collected["moran"].append((niche, observed, null))
        if "enrichment" in cfg.analyses:
            z = enrichment_zscores(types, local_edges, k, n_perms=cfg.n_perms, rng=rng)
            collected["enrichment"].append((niche, z))
        if "cross_l" in cfg.analyses:
            coords = cells.pos[idx]
            curves = np.full((len(pairs), radii.size), np.nan)
            counts = np.bincount(types, minlength=k)
            for p, (a, b) in enumerate(pairs):
                if counts[a] >= 2 and counts[b] >= 2:
                    curves[p] = cross_l(coords, types == a, types == b, radii)
            collected["cross_l"].append((niche, curves))
    out: dict[str, pd.DataFrame] = {}
    if "moran" in cfg.analyses:
        out["moran"] = _moran_table(collected["moran"], names)
    if "enrichment" in cfg.analyses:
        out["enrichment"] = _pair_table(collected["enrichment"], names, "z")
    if "cross_l" in cfg.analyses:
        out["cross_l"] = _cross_l_table(collected["cross_l"], names, pairs, radii)
    if "contacts" in cfg.analyses:
        out["contacts"] = _contact_table(niches, cells.cell_type, edges, names, cfg.min_contacts)
    return out


def _radii(cells: Cells, cfg: DescriptorConfig) -> np.ndarray:
    if cfg.max_radius is not None:
        top = float(cfg.max_radius)
    else:
        diagonals = [
            float(np.linalg.norm(np.ptp(cells.pos[cells.unit_of_cell == u], axis=0)))
            for u in cells.units
        ]
        top = 0.25 * float(np.median(diagonals))
    return np.linspace(top / cfg.n_radii, top, cfg.n_radii)


def _moran_table(records, names) -> pd.DataFrame:
    rows = []
    for niche in sorted({r[0] for r in records}):
        obs = np.stack([r[1] for r in records if r[0] == niche])  # (units, k)
        null = np.stack([r[2] for r in records if r[0] == niche])  # (units, perms, k)
        for j, name in enumerate(names):
            values = obs[:, j]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            mean = float(values[finite].mean())
            null_mean = np.nanmean(null[finite][:, :, j], axis=0)
            centre = float(np.nanmean(null_mean))
            extreme = np.abs(null_mean - centre) >= abs(mean - centre)
            rows.append(
                {
                    "niche": niche_name(niche),
                    "cell_type": name,
                    "morans_i": mean,
                    "sem": float(values[finite].std(ddof=1) / np.sqrt(finite.sum()))
                    if finite.sum() > 1
                    else float("nan"),
                    "n_units": int(finite.sum()),
                    "p": float((1 + extreme.sum()) / (1 + null_mean.size)),
                }
            )
    table = pd.DataFrame(rows, columns=["niche", "cell_type", "morans_i", "sem", "n_units", "p"])
    table["q"] = bh(table["p"].to_numpy())
    return table


def _pair_table(records, names, value: str) -> pd.DataFrame:
    rows = []
    for niche in sorted({r[0] for r in records}):
        stack = np.stack([r[1] for r in records if r[0] == niche])
        with np.errstate(all="ignore"):
            mean = np.nanmean(stack, axis=0)
        n = np.isfinite(stack).sum(0)
        for a in range(len(names)):
            for b in range(a, len(names)):
                if n[a, b]:
                    rows.append(
                        {
                            "niche": niche_name(niche),
                            "type_a": names[a],
                            "type_b": names[b],
                            value: float(mean[a, b]),
                            "n_units": int(n[a, b]),
                        }
                    )
    return pd.DataFrame(rows, columns=["niche", "type_a", "type_b", value, "n_units"])


def _cross_l_table(records, names, pairs, radii) -> pd.DataFrame:
    rows = []
    for niche in sorted({r[0] for r in records}):
        stack = np.stack([r[1] for r in records if r[0] == niche])  # (units, pairs, radii)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stack, axis=0)
            strength = np.nanmean(np.abs(mean), axis=0)
        if not np.isfinite(strength).any():
            continue
        r_idx = int(np.nanargmax(strength))
        for p, (a, b) in enumerate(pairs):
            if np.isfinite(mean[p, r_idx]):
                rows.append(
                    {
                        "niche": niche_name(niche),
                        "type_a": names[a],
                        "type_b": names[b],
                        "r_star": float(radii[r_idx]),
                        "centred_l": float(mean[p, r_idx]),
                        "n_units": int(np.isfinite(stack[:, p, r_idx]).sum()),
                    }
                )
    return pd.DataFrame(
        rows, columns=["niche", "type_a", "type_b", "r_star", "centred_l", "n_units"]
    )


def _contact_table(niches, types, edges, names, min_contacts: int) -> pd.DataFrame:
    k = len(names)
    tissue = contact_counts(types, edges, k)
    tissue_frac = tissue / max(tissue.sum(), 1.0)
    rows = []
    same = niches[edges[0]] == niches[edges[1]]
    for niche in sorted(set(niches.tolist()) - {BACKGROUND}):
        inside = edges[:, same & (niches[edges[0]] == niche)]
        counts = contact_counts(types, inside, k)
        frac = counts / max(counts.sum(), 1.0)
        for a in range(k):
            for b in range(a, k):
                n_contacts = int(counts[a, b] if a != b else counts[a, b] / 2)
                enough = n_contacts >= min_contacts and tissue_frac[a, b] > 0
                rows.append(
                    {
                        "niche": niche_name(niche),
                        "type_a": names[a],
                        "type_b": names[b],
                        "n_contacts": n_contacts,
                        "log2_enrichment": float(np.log2(frac[a, b] / tissue_frac[a, b]))
                        if enough
                        else float("nan"),
                    }
                )
    return pd.DataFrame(
        rows, columns=["niche", "type_a", "type_b", "n_contacts", "log2_enrichment"]
    )
