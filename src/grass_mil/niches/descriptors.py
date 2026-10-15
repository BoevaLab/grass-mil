"""Classical spatial descriptors of every niche.

Each statistic is computed on the cells of one niche within one unit (the
niche's subgraph: its cells and the cell-graph edges between them) and then
averaged over the units in which the niche has at least ``min_cells`` cells:

* ``moran``: Moran's I of every cell-type indicator (are cells of a type
  clustered within the niche?), mean and s.e.m. over units, with a z-test of
  the mean against its expectation under random labelling (exact moments per
  unit) and BH correction over niche x type;
* ``enrichment``: neighbourhood enrichment of every cell-type pair, as the
  permutation z-score and the size-free ``log2`` observed/expected ratio;
* ``cross_l``: Ripley's cross-L of every cell-type pair relative to random
  labelling of the niche's cells, ``L_ab(r) - L(r)``, on a shared radius grid;
  ``r_star`` is the radius where the mean absolute value over pairs peaks;
* ``contacts``: ``log2`` of a pair's share of the contacts inside the niche
  over its share of all contacts of the analysed cells (all units pooled),
  masked below ``min_contacts`` contacts inside the niche. A pair of the
  niche's dominant cell type is enriched by composition alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from grass_mil.spatial.graph_stats import contact_counts, cross_l, enrichment, morans_i
from grass_mil.spatial.stats import bh
from grass_mil.utils import checked

from .cells import Cells
from .discovery import BACKGROUND, niche_name

__all__ = ["DescriptorConfig", "niche_descriptors", "short_edges"]

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
        cfg = checked(cls, cfg, "descriptors")
        cfg["analyses"] = tuple(cfg.get("analyses", ANALYSES))
        bad = set(cfg["analyses"]) - set(ANALYSES)
        if bad:
            raise ValueError(
                f"Unknown descriptor analyses {sorted(bad)}; expected {list(ANALYSES)}."
            )
        return cls(**cfg)


def short_edges(cells: Cells, max_length: float | None) -> np.ndarray:
    """The cell-graph edges no longer than ``max_length`` (all edges for ``None``)."""
    if max_length is None:
        return cells.edges
    a, b = cells.edges
    return cells.edges[:, np.linalg.norm(cells.pos[a] - cells.pos[b], axis=1) <= max_length]


def _subgraphs(cells: Cells, niches: np.ndarray, edges: np.ndarray, min_cells: int):
    """``(niche, cell indices, local edges)`` of every niche subgraph with enough cells."""
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
        lo, hi = np.searchsorted(edge_key, [key[idx[0]], key[idx[0]] + 1])
        local[idx] = np.arange(idx.size)
        yield int(niches[idx[0]]), idx, local[inside[:, lo:hi]]


def niche_descriptors(
    niches: np.ndarray, cells: Cells, cfg: DescriptorConfig
) -> dict[str, pd.DataFrame]:
    if (cells.cell_type < 0).any():
        return {}
    names = cells.cell_type_names
    k = len(names)
    edges = short_edges(cells, cfg.max_edge_length)
    rng = np.random.default_rng(cfg.seed)
    radii = _radii(cells, cfg)
    pairs = [(a, b) for a in range(k) for b in range(a, k)]
    first, second = np.array(pairs).T
    records: dict[str, dict[int, list]] = {a: {} for a in cfg.analyses if a != "contacts"}
    for niche, idx, local_edges in _subgraphs(cells, niches, edges, cfg.min_cells):
        types = cells.cell_type[idx]
        values = {}
        if "moran" in records:
            i, expected, var = morans_i(np.eye(k)[types], local_edges)
            values["moran"] = np.stack([i, np.full(k, expected), var])
        if "enrichment" in records:
            z, ratio = enrichment(types, local_edges, k, n_perms=cfg.n_perms, rng=rng)
            values["enrichment"] = np.stack([z[first, second], ratio[first, second]])
        if "cross_l" in records:
            values["cross_l"] = cross_l(cells.pos[idx], types, pairs, radii)
        for name, value in values.items():
            records[name].setdefault(niche, []).append(value)
    stacks = {a: {n: np.stack(v) for n, v in sorted(r.items())} for a, r in records.items()}
    out: dict[str, pd.DataFrame] = {}
    if "moran" in stacks:
        out["moran"] = _moran_table(stacks["moran"], names)
    if "enrichment" in stacks:
        out["enrichment"] = _pair_table(stacks["enrichment"], names, pairs, ["z", "log2_ratio"])
    if "cross_l" in stacks:
        out["cross_l"] = _cross_l_table(stacks["cross_l"], names, pairs, radii)
    if "contacts" in cfg.analyses:
        out["contacts"] = _contact_table(niches, cells.cell_type, edges, names, cfg.min_contacts)
    return out


def _radii(cells: Cells, cfg: DescriptorConfig) -> np.ndarray:
    if cfg.max_radius is not None:
        top = float(cfg.max_radius)
    else:
        diagonals = [np.linalg.norm(np.ptp(cells.pos[s], axis=0)) for s in cells.slices.values()]
        top = 0.25 * float(np.median(diagonals))
    return np.linspace(top / cfg.n_radii, top, cfg.n_radii)


def _moran_table(stacks: dict[int, np.ndarray], names: Sequence[str]) -> pd.DataFrame:
    from scipy.stats import norm

    rows = []
    for niche, stack in stacks.items():  # (units, [I, E, Var], types)
        for j, name in enumerate(names):
            i, expected, var = stack[:, 0, j], stack[:, 1, j], stack[:, 2, j]
            ok = np.isfinite(i) & np.isfinite(var)
            u = int(ok.sum())
            if not u:
                continue
            z = (i[ok].mean() - expected[ok].mean()) / (np.sqrt(var[ok].sum()) / u)
            rows.append(
                {
                    "niche": niche_name(niche),
                    "cell_type": name,
                    "morans_i": float(i[ok].mean()),
                    "expected": float(expected[ok].mean()),
                    "sem": float(i[ok].std(ddof=1) / np.sqrt(u)) if u > 1 else float("nan"),
                    "n_units": u,
                    "z": float(z),
                    "p": float(2 * norm.sf(abs(z))),
                }
            )
    columns = ["niche", "cell_type", "morans_i", "expected", "sem", "n_units", "z", "p"]
    table = pd.DataFrame(rows, columns=columns)
    table["q"] = bh(table["p"].to_numpy())
    return table


def _pair_table(
    stacks: dict[int, np.ndarray], names: Sequence[str], pairs, columns: Sequence[str]
) -> pd.DataFrame:
    """Mean over units of every value of every cell-type pair; ``stacks`` hold
    ``(units, len(columns), len(pairs))`` per niche."""
    rows = []
    for niche, stack in stacks.items():
        with np.errstate(all="ignore"):
            mean = np.nanmean(stack, axis=0)
        n = np.isfinite(stack[:, 0]).sum(0)
        for p, (a, b) in enumerate(pairs):
            if n[p]:
                row = {"niche": niche_name(niche), "type_a": names[a], "type_b": names[b]}
                row.update({c: float(mean[j, p]) for j, c in enumerate(columns)})
                rows.append({**row, "n_units": int(n[p])})
    return pd.DataFrame(rows, columns=["niche", "type_a", "type_b", *columns, "n_units"])


def _cross_l_table(stacks, names, pairs, radii) -> pd.DataFrame:
    at_r_star, r_star = {}, {}
    for niche, stack in stacks.items():  # (units, pairs, radii)
        with np.errstate(all="ignore"):
            strength = np.nanmean(np.abs(np.nanmean(stack, axis=0)), axis=0)
        if np.isfinite(strength).any():
            r = int(np.nanargmax(strength))
            at_r_star[niche], r_star[niche] = stack[:, None, :, r], float(radii[r])
    table = _pair_table(at_r_star, names, pairs, ["cross_l"])
    table.insert(3, "r_star", table["niche"].map({niche_name(n): r for n, r in r_star.items()}))
    return table


def _contact_table(niches, types, edges, names, min_contacts: int) -> pd.DataFrame:
    k = len(names)
    tissue = contact_counts(types, edges, k)
    tissue_frac = tissue / max(tissue.sum(), 1.0)
    rows = []
    same = niches[edges[0]] == niches[edges[1]]
    for niche in sorted(set(niches.tolist()) - {BACKGROUND}):
        counts = contact_counts(types, edges[:, same & (niches[edges[0]] == niche)], k)
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
