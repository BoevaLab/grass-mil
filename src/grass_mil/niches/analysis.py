"""Niche analysis of one model's cells: discovery, attribution, descriptors, figures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache
from grass_mil.spatial.graph_stats import colocalisation

from . import figures
from .attribution import cell_margins
from .cells import Cells, load_cells
from .descriptors import DescriptorConfig, niche_descriptors
from .discovery import BACKGROUND, ClusteringConfig, NicheModel, fit_niches, niche_name
from .survival import SurvivalConfig, niche_survival
from .tables import niche_table, unit_niche_shares

__all__ = ["NicheAnalysis", "analyse_niches", "write_niche_outputs"]

log = logging.getLogger(__name__)


@dataclass
class NicheAnalysis:
    cells: Cells
    model: NicheModel
    niches: np.ndarray  # niche id per cell (0 = Background)
    margin: np.ndarray
    table: pd.DataFrame
    unit_shares: pd.DataFrame
    descriptors: dict[str, pd.DataFrame]
    colocalisation: pd.DataFrame
    positive: dict[str, bool]
    survival: tuple[pd.DataFrame, pd.DataFrame | None] | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def niche_names(self) -> np.ndarray:
        return np.array([niche_name(n) for n in self.niches], dtype=object)


def analyse_niches(
    cache: CellCache,
    units: list[str],
    *,
    clustering: ClusteringConfig,
    descriptors: DescriptorConfig,
    survival: SurvivalConfig,
    positive_class: int = 1,
    present_threshold: float = 0.02,
    enrichment_threshold: float = 1.5,
    n_boot: int = 2000,
) -> NicheAnalysis:
    cells = load_cells(cache, units)
    missing = [u for u in units if u not in cells.targets]
    if missing:
        raise ValueError(
            f"{len(missing)} unit(s) have no label in the cache (first: {missing[:5]})."
        )
    margin = cell_margins(
        cells.instance_logits,
        cells.attention_logits,
        cells.head_bias,
        cells.unit_of_cell,
        positive_class=positive_class,
    )
    model, niches = fit_niches(cells.space[clustering.space], margin, clustering)
    positive = cells.positive(positive_class)
    table = niche_table(
        niches,
        cells,
        margin,
        positive=positive,
        present_threshold=present_threshold,
        enrichment_threshold=enrichment_threshold,
        n_boot=n_boot,
        seed=clustering.seed,
    )
    shares = unit_niche_shares(niches, cells)
    named = [n for n in table["niche"] if n != "Background"]
    coloc = colocalisation(np.where(niches == BACKGROUND, -1, niches - 1), cells.edges, len(named))
    result = NicheAnalysis(
        cells=cells,
        model=model,
        niches=niches,
        margin=margin,
        table=table,
        unit_shares=shares,
        descriptors=niche_descriptors(niches, cells, descriptors),
        colocalisation=pd.DataFrame(coloc, index=named, columns=named),
        positive=positive,
        settings={
            "units": units,
            "n_cells": cells.n,
            "positive_class": positive_class,
            "present_threshold": present_threshold,
            "enrichment_threshold": enrichment_threshold,
            "clustering": model.settings(),
        },
    )
    if survival.file:
        outcomes = pd.read_csv(survival.file, dtype={survival.id_column: str}).set_index(
            survival.id_column
        )
        outcomes = outcomes[[survival.time_column, survival.event_column]].set_axis(
            ["time", "event"], axis=1
        )
        patients = shares.groupby([cells.patients.get(u, u) for u in shares.index]).mean()
        result.survival = niche_survival(patients, outcomes, survival)
    return result


def write_niche_outputs(
    result: NicheAnalysis,
    out_dir: str | Path,
    *,
    title: str,
    positive_name: str,
    render_units: int,
    figures_enabled: bool = True,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = result.cells
    result.table.to_csv(out / "niches.csv", index=False)
    result.unit_shares.to_csv(out / "unit_niche_shares.csv")
    pd.DataFrame(
        {
            "unit_id": cells.unit_of_cell,
            "cell_id": cells.cell_ids,
            "x": cells.pos[:, 0],
            "y": cells.pos[:, 1],
            "cell_type": [cells.cell_type_names[c] if c >= 0 else "" for c in cells.cell_type],
            "niche": result.niche_names,
            "margin": result.margin,
        }
    ).to_csv(out / "cells.csv", index=False)
    result.colocalisation.to_csv(out / "niche_colocalisation.csv")
    desc_dir = out / "descriptors"
    desc_dir.mkdir(exist_ok=True)
    for name, frame in result.descriptors.items():
        frame.to_csv(desc_dir / f"{name}.csv", index=False)
    if result.survival is not None:
        result.survival[0].to_csv(out / "survival_cox.csv", index=False)
        if result.survival[1] is not None:
            result.survival[1].to_csv(out / "survival_groups.csv")
    joblib.dump(result.model, out / "niche_model.joblib")
    (out / "settings.json").write_text(json.dumps(result.settings, indent=1, default=str))
    (out / "report.md").write_text(_report(result, title=title, positive_name=positive_name))
    if figures_enabled:
        _figures(result, out, positive_name=positive_name, render_units=render_units)
    return out


def _figures(result: NicheAnalysis, out: Path, *, positive_name: str, render_units: int) -> None:
    from sklearn.decomposition import PCA

    cells = result.cells
    names = cells.cell_type_names
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    figures.niche_panel(
        result.table, names, fig_dir / "niche_panel.png", positive_name=positive_name
    )
    sample = result.model.sample
    x = result.model.coordinates(cells.space[result.model.cfg.space][sample])
    pca = PCA(n_components=2, random_state=0).fit(x)
    ev = pca.explained_variance_ratio_
    is_pos = np.array([result.positive.get(u, False) for u in cells.unit_of_cell[sample]])
    figures.embedding_panel(
        pca.transform(x),
        result.niche_names[sample],
        result.margin[sample],
        is_pos,
        fig_dir / "embedding.png",
        positive_name=positive_name,
        axis_labels=(f"PC 1 ({100 * ev[0]:.0f}%)", f"PC 2 ({100 * ev[1]:.0f}%)"),
    )
    figures.unit_composition(
        result.unit_shares,
        result.positive,
        fig_dir / "unit_composition.png",
        positive_name=positive_name,
    )
    if len(result.colocalisation):
        figures.colocalisation_heatmap(
            result.colocalisation.to_numpy(),
            list(result.colocalisation.index),
            fig_dir / "niche_colocalisation.png",
        )
    desc = result.descriptors
    if "moran" in desc:
        figures.moran_heatmap(desc["moran"], fig_dir / "moran.png")
    if "enrichment" in desc:
        figures.pair_heatmaps(
            desc["enrichment"],
            "z",
            names,
            fig_dir / "enrichment.png",
            label="neighbourhood enrichment z",
        )
    if "cross_l" in desc:
        figures.pair_heatmaps(
            desc["cross_l"], "centred_l", names, fig_dir / "cross_l.png", label="L(r*) - r*"
        )
    if "contacts" in desc:
        figures.pair_heatmaps(
            desc["contacts"],
            "log2_enrichment",
            names,
            fig_dir / "contacts.png",
            label="log2 contact enrichment",
        )
    if result.survival is not None and result.survival[1] is not None:
        figures.survival_curves(
            result.survival[1], fig_dir / "survival.png", score_label="niche score tertiles"
        )
    if render_units > 0:
        _unit_maps(result, fig_dir / "units", render_units)


def _unit_maps(result: NicheAnalysis, out: Path, n: int) -> None:
    """Maps of the units the model scores most and least positive."""
    cells = result.cells
    out.mkdir(exist_ok=True)
    unit_margin = pd.Series(result.margin).groupby(cells.unit_of_cell).mean()
    order = unit_margin.sort_values().index.tolist()
    chosen = list(dict.fromkeys(order[-((n + 1) // 2) :][::-1] + order[: n // 2]))
    for unit in chosen:
        m = cells.unit_of_cell == unit
        figures.unit_maps(
            cells.pos[m],
            cells.cell_type[m],
            cells.cell_type_names,
            result.margin[m],
            result.niche_names[m],
            out / f"{''.join(c if c.isalnum() or c in '._-' else '_' for c in unit)}.png",
            title=unit,
        )


def _fmt(value: float, spec: str = "+.2f") -> str:
    return "" if value is None or not np.isfinite(value) else format(value, spec)


def _report(result: NicheAnalysis, *, title: str, positive_name: str) -> str:
    s = result.settings
    c = s["clustering"]
    t = result.table
    base = t.attrs.get("base_rate", float("nan"))
    lines = [
        f"# {title}",
        "",
        f"{len(s['units'])} units, {s['n_cells']} cells. Clustering: {c['method']} in the {c['space']} space "
        f"(PCA {c['pca_components'] or 'none'}, cosine {c['cosine']}, fitted on {c['n_fitted']} cells, "
        f"{c['assign_k']}-NN assignment): {c['n_niches']} niches. Fraction of cells from {positive_name} units: {base:.3f}.",
        "",
        f"| Niche | Cells | Units (>{100 * s['present_threshold']:.0f}%) | Composition | Enriched (>{s['enrichment_threshold']}x) | "
        f"Association | Margin [95% CI] | Sign consistency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['niche']} | {100 * r['share']:.1f}% | {r['n_units']} | {r.get('composition', '')} | {r.get('enriched', '')} | "
            f"{r['association']:.2f} | {_fmt(r['margin'])} [{_fmt(r['margin_lo'])}, {_fmt(r['margin_hi'])}] | {_fmt(r['sign_consistency'], '.2f')} |"
        )
    moran = result.descriptors.get("moran")
    if moran is not None and len(moran):
        top = moran[moran["q"] < 0.05].sort_values("morans_i", ascending=False).head(10)
        if len(top):
            lines += ["", "## Clustered cell types within niches (Moran's I, q < 0.05)", ""]
            lines += [
                f"- {r.niche}: {r.cell_type} I = {r.morans_i:.2f} ± {r.sem:.2f} ({r.n_units} units)"
                for r in top.itertuples()
            ]
    if result.survival is not None:
        cox = result.survival[0].sort_values("p").head(10)
        lines += [
            "",
            "## Niche share and survival (Cox, per SD of log share)",
            "",
            "| Niche | HR [95% CI] | p | q |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r.niche} | {r.hazard_ratio:.2f} [{r.hr_lo:.2f}, {r.hr_hi:.2f}] | {r.p:.3g} | {r.q:.2g} |"
            for r in cox.itertuples()
        ]
        if result.survival[1] is not None:
            lines.append(
                f"\nScore tertiles: log-rank p = {result.survival[1].attrs['logrank_p']:.3g}"
            )
    return "\n".join(lines) + "\n"
