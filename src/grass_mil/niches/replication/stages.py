"""Replication of niches across the models of a cross-validation (optional analysis).

1. ``cluster``: discover the niches of every fold model on the units it held out.
2. ``match``: group the niches of different models into metaclusters by the
   Hellinger distance of their cell-type compositions, with average linkage
   that never merges two niches of the same model. The distance threshold is
   fixed or chosen to maximise the metaclusters found in ``min_models`` models.
3. ``transfer``: every model labels every unit with its own niches (mapped to
   metaclusters); reports train vs held-out behaviour, agreement between models
   and consensus metacluster shares (descriptive: most models trained on most
   units, so only the held-out shares of ``match`` are tested against labels).
4. ``basis``: one label-free k-means partition of a single space, attributed by
   every model, so the margins of different models refer to the same cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache
from grass_mil.spatial.stats import bootstrap_mean

from ..attribution import cell_margins
from ..cells import Cells, load_cells
from ..discovery import BACKGROUND, ClusteringConfig, NicheModel, fit_niches
from ..tables import niche_table

__all__ = [
    "FoldNiches",
    "cluster_folds",
    "hellinger",
    "constrained_linkage",
    "match_niches",
    "held_out_shares",
    "transfer_niches",
    "shared_basis",
]

log = logging.getLogger(__name__)


@dataclass
class FoldNiches:
    records: pd.DataFrame  # one row per (model, niche)
    labels: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # unit -> niche ids (held-out model)
    model_of_unit: dict[str, str] = field(default_factory=dict)
    models: dict[str, NicheModel] = field(default_factory=dict)
    cell_types: list[str] = field(default_factory=list)
    patients: dict[str, str] = field(default_factory=dict)
    positive: dict[str, bool] = field(default_factory=dict)


def _margins(cells: Cells, positive_class: int) -> np.ndarray:
    return cell_margins(
        cells.instance_logits,
        cells.attention_logits,
        cells.reference_logits,
        cells.unit_of_cell,
        positive_class=positive_class,
    )


def cluster_folds(
    caches: dict[str, CellCache],
    cfg: ClusteringConfig,
    *,
    positive_class: int,
    present_threshold: float,
    enrichment_threshold: float,
    n_boot: int,
) -> FoldNiches:
    frames, out = [], FoldNiches(records=pd.DataFrame())
    for name, cache in caches.items():
        cells = load_cells(cache, sorted(cache.held_out), spaces=[cfg.space])
        margin = _margins(cells, positive_class)
        model, niches = fit_niches(cells.space[cfg.space], margin, cells.unit_of_cell, cfg)
        positive = cells.positive(positive_class)
        table = niche_table(
            niches,
            cells,
            margin,
            positive=positive,
            present_threshold=present_threshold,
            enrichment_threshold=enrichment_threshold,
            n_boot=n_boot,
            seed=cfg.seed,
        )
        table = table[table["niche_id"] != BACKGROUND].copy()
        table.insert(0, "model", name)
        table["id"] = name + ":" + table["niche"]
        table["association_vs_base"] = table["association"] - table.attrs.get("base_rate", np.nan)
        frames.append(table)
        for unit in cells.units:
            out.labels[unit] = niches[cells.unit_of_cell == unit]
            out.model_of_unit[unit] = name
        out.models[name] = model
        out.cell_types = cells.cell_type_names
        out.patients.update(cells.patients)
        out.positive.update(positive)
        log.info(f"[{name}] {len(cells.units)} held-out units, {model.n_niches} niches")
    out.records = pd.concat(frames, ignore_index=True)
    return out


def hellinger(compositions: np.ndarray) -> np.ndarray:
    """Pairwise Hellinger distance between composition rows (0 to 1)."""
    h = np.sqrt(np.clip(compositions, 0.0, None))
    return np.sqrt(0.5 * ((h[:, None, :] - h[None, :, :]) ** 2).sum(-1))


def constrained_linkage(
    distance: np.ndarray, model: Sequence[str], threshold: float
) -> list[list[int]]:
    """Greedy average linkage below ``threshold``; groups sharing a model never merge."""
    model = np.asarray(model)
    groups = [[i] for i in range(distance.shape[0])]
    while True:
        best = None
        for a in range(len(groups)):
            models_a = set(model[groups[a]])
            for b in range(a + 1, len(groups)):
                if models_a & set(model[groups[b]]):
                    continue
                d = float(distance[np.ix_(groups[a], groups[b])].mean())
                if d < threshold and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            return groups
        _, a, b = best
        groups[a] += groups.pop(b)


def match_niches(
    folds: FoldNiches, *, threshold: float | None, scan: Sequence[float], min_models: int
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame | None]:
    """Metacluster of every niche record, the metacluster table and the threshold used."""
    records = folds.records.reset_index(drop=True).copy()
    comp_cols = [f"frac_{c}" for c in folds.cell_types]
    distance = hellinger(records[comp_cols].to_numpy(float))
    scan_table = None
    if threshold is None:
        rows, best = [], None
        for t in scan:
            groups = constrained_linkage(distance, records["model"], float(t))
            replicated = sum(len(g) >= min_models for g in groups)
            rows.append(
                {"threshold": float(t), "n_metaclusters": len(groups), "n_replicated": replicated}
            )
            if best is None or (replicated, -len(groups)) > best[0]:
                best = ((replicated, -len(groups)), float(t), groups)
        _, threshold, groups = best
        scan_table = pd.DataFrame(rows)
    else:
        groups = constrained_linkage(distance, records["model"], float(threshold))
    records["metacluster"] = 0
    for k, members in enumerate(sorted(groups, key=len, reverse=True), 1):
        records.loc[members, "metacluster"] = k
    rows = []
    for k, g in records.groupby("metacluster"):
        comp = np.average(g[comp_cols].to_numpy(float), axis=0, weights=g["n_cells"])
        idx = g.index.to_numpy()
        rows.append(
            {
                "metacluster": f"M{k}",
                "n_models": g["model"].nunique(),
                "replicated": g["model"].nunique() >= min_models,
                "members": ", ".join(g["id"]),
                "within_distance": float(
                    distance[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
                )
                if len(idx) > 1
                else 0.0,
                "share": float(g["share"].mean()),
                "association_vs_base": float(g["association_vs_base"].mean()),
                "margin": float(g["margin"].mean()),
                "margin_sd": float(g["margin"].std(ddof=1)) if len(g) > 1 else float("nan"),
                "sign_agreement": float(
                    (np.sign(g["margin"]) == np.sign(g["margin"].mean())).mean()
                ),
                "composition": ", ".join(
                    f"{folds.cell_types[j]} {100 * comp[j]:.0f}%" for j in np.argsort(-comp)[:3]
                ),
                **{c: float(v) for c, v in zip(comp_cols, comp)},
            }
        )
    return records, pd.DataFrame(rows), float(threshold), scan_table


def held_out_shares(folds: FoldNiches, records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metacluster shares of every unit (labelled by the model that held it out) and per patient."""
    meta_of = {(r.model, r.niche_id): f"M{r.metacluster}" for r in records.itertuples()}
    columns = sorted(set(meta_of.values()), key=lambda m: int(m[1:]))
    rows = {}
    for unit, labels in folds.labels.items():
        model = folds.model_of_unit[unit]
        names = pd.Series([meta_of.get((model, int(n)), "Background") for n in labels])
        rows[unit] = names.value_counts(normalize=True).reindex(columns, fill_value=0.0)
    units = pd.DataFrame(rows).T
    patients = units.groupby([folds.patients.get(u, u) for u in units.index]).mean()
    return units, patients


def _aligned(caches: dict[str, CellCache], units: list[str], spaces=()) -> dict[str, Cells]:
    """The same units' cells from every cache, checked to be in the same order."""
    out = {name: load_cells(cache, units, spaces=spaces) for name, cache in caches.items()}
    first = next(iter(out.values()))
    if any(not np.array_equal(c.cell_ids, first.cell_ids) for c in out.values()):
        raise ValueError("The caches hold different cells for the same units.")
    return out


def transfer_niches(
    caches: dict[str, CellCache], folds: FoldNiches, records: pd.DataFrame, *, positive_class: int
) -> dict[str, pd.DataFrame]:
    """Every model labels every unit with its niches, mapped to metaclusters."""
    from sklearn.metrics import adjusted_rand_score

    meta_of = {(r.model, r.niche_id): int(r.metacluster) for r in records.itertuples()}
    names = list(caches)
    units = sorted(set().union(*(c.units for c in caches.values())))
    spaces = sorted({m.cfg.space for m in folds.models.values()})
    cells_of = _aligned(caches, units, spaces)
    labels, rows = {}, []
    for name in names:
        cells = cells_of[name]
        niches = folds.models[name].predict(cells.space[folds.models[name].cfg.space])
        meta = np.array([meta_of.get((name, int(n)), 0) for n in niches])
        labels[name] = meta
        held = np.isin(cells.unit_of_cell, caches[name].held_out)
        positive = np.array([folds.positive.get(u, False) for u in cells.unit_of_cell])
        margin = _margins(cells, positive_class)
        for k in sorted(set(meta.tolist()) - {0}):
            for split, mask in (("held_out", held), ("train", ~held)):
                m = (meta == k) & mask
                if m.any():
                    rows.append(
                        {
                            "metacluster": f"M{k}",
                            "model": name,
                            "split": split,
                            "share": float(m.sum() / mask.sum()),
                            "association_vs_base": float(
                                positive[m].mean() - positive[mask].mean()
                            ),
                            "margin": float(
                                pd.Series(margin[m]).groupby(cells.unit_of_cell[m]).mean().mean()
                            ),
                        }
                    )
    ari = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            both = (labels[a] > 0) & (labels[b] > 0)
            if both.sum() > 10:
                ari.loc[a, b] = adjusted_rand_score(labels[a][both], labels[b][both])
    metas = sorted(set(np.concatenate(list(labels.values())).tolist()) - {0})
    unit_of_cell = cells_of[names[0]].unit_of_cell
    votes = pd.DataFrame(
        {f"M{k}": np.mean([labels[n] == k for n in names], axis=0) for k in metas}
    )
    consensus = votes.groupby(unit_of_cell).mean().reindex(units)
    return {
        "by_model": pd.DataFrame(rows),
        "agreement": ari,
        "unit_shares": consensus,
        "patient_shares": consensus.groupby([folds.patients.get(u, u) for u in units]).mean(),
    }


def shared_basis(
    caches: dict[str, CellCache],
    *,
    space_model: str | None,
    space: str,
    k: int,
    sample_size: int,
    seed: int,
    n_boot: int,
    positive_class: int,
) -> dict[str, pd.DataFrame]:
    """A k-means partition of one model's space, attributed by every model."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    basis_name = space_model or next(iter(caches))
    units = sorted(caches[basis_name].units)
    cells_of = _aligned(caches, units)
    base = load_cells(caches[basis_name], units, spaces=[space])
    x = base.space[space]
    sample = np.random.default_rng(seed).choice(len(x), min(sample_size, len(x)), replace=False)
    scaler = StandardScaler().fit(x[sample])
    kmeans = MiniBatchKMeans(k, random_state=seed, n_init=10).fit(scaler.transform(x[sample]))
    groups = kmeans.predict(scaler.transform(x)) + 1
    positive = base.positive(positive_class)
    is_pos = np.array([positive.get(u, False) for u in base.unit_of_cell])
    frames = []
    for name, cells in cells_of.items():
        frame = (
            pd.DataFrame(
                {
                    "unit_id": cells.unit_of_cell,
                    "group": groups,
                    "margin": _margins(cells, positive_class),
                }
            )
            .groupby(["unit_id", "group"], as_index=False)["margin"]
            .mean()
        )
        held = frame["unit_id"].isin(caches[name].held_out)
        frame.insert(0, "model", name)
        frame.insert(3, "split", np.where(held, "held_out", "train"))
        frames.append(frame)
    pairs = pd.concat(frames, ignore_index=True)
    rows = []
    k_types = len(base.cell_type_names)
    for group in range(1, k + 1):
        m = groups == group
        if not m.any():
            continue
        comp = (
            np.bincount(base.cell_type[m], minlength=k_types) / m.sum() if k_types else np.zeros(0)
        )
        row = {
            "group": f"G{group}",
            "share": float(m.mean()),
            "association": float(is_pos[m].mean()),
            "composition": ", ".join(
                f"{base.cell_type_names[j]} {100 * comp[j]:.0f}%" for j in np.argsort(-comp)[:3]
            ),
        }
        for split in ("held_out", "train"):
            chosen = pairs[(pairs.group == group) & (pairs.split == split)]
            # Every model scores the same units: resample patients, not (model, unit) pairs.
            row[f"margin_{split}"], row[f"margin_{split}_lo"], row[f"margin_{split}_hi"] = (
                bootstrap_mean(
                    chosen["margin"],
                    clusters=[base.patient_of(u) for u in chosen["unit_id"]],
                    n_boot=n_boot,
                    seed=seed,
                )
            )
        rows.append(row)
    return {"groups": pd.DataFrame(rows), "pairs": pairs}
