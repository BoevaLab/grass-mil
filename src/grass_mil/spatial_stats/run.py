"""Run the classical spatial statistics over a cell cache and compare groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache

from .cellular_neighbourhoods import cellular_neighbourhoods, window_composition
from .contacts import contact_fractions
from .cross_k import cross_k
from .nhood_enrichment import nhood_enrichment_zscores
from .tests import clr, mann_whitney_table, patient_mean

__all__ = ["run_spatial_stats"]


def _resolve_groups(cell_type_names: Sequence[str], group_map: Optional[Dict[str, str]]):
    """Fine cell types -> coarse groups (identity by default)."""
    names = list(cell_type_names)
    if not group_map:
        groups = sorted(set(names))
    else:
        groups = sorted({group_map.get(n, n) for n in names})
    gi = {g: i for i, g in enumerate(groups)}
    fine_to_group = np.array([gi[(group_map or {}).get(n, n)] for n in names], dtype=int)
    return groups, fine_to_group


def run_spatial_stats(
    cache: CellCache,
    *,
    labels_of: Dict[str, int],
    patient_of: Optional[Dict[str, str]] = None,
    out_dir: str | Path,
    regions: Optional[Sequence[str]] = None,
    group_map: Optional[Dict[str, str]] = None,
    reference: Optional[Sequence[str]] = None,
    partners: Optional[Sequence[str]] = None,
    infiltrating: Sequence[str] = (),
    n_perms: int = 200,
    nhood_backend: str = "numpy",
    window_k: int = 10,
    n_cn: int = 10,
    cn_sample_size: int = 200_000,
    radii: Sequence[float] = (20.0, 50.0, 100.0),
    seed: int = 0,
    min_per_group: int = 10,
    positive_name: str = "positive",
    analyses: Sequence[str] = ("nhood", "cn", "contacts", "crossk"),
    log=None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    regions = sorted(str(r) for r in (regions if regions is not None else cache.regions))
    regions = [r for r in regions if r in labels_of]
    names = cache.cell_type_names or []
    groups, fine_to_group = _resolve_groups(names, group_map)
    G = len(groups)
    pat_of = patient_of or {r: r for r in regions}
    y_pat = pd.Series({pat_of.get(r, r): int(labels_of[r]) for r in regions})
    reference = list(reference) if reference else []
    partners = list(partners) if partners else [g for g in groups if g not in reference]
    analyses = [a.lower() for a in analyses]

    Z: Dict[str, np.ndarray] = {}
    windows: List[np.ndarray] = []
    contacts: List[Dict[str, Any]] = []
    crossk: List[Dict[str, Any]] = []
    for i, r in enumerate(regions):
        d = cache.load(r, keys=["ct", "pos", "edge_index"])
        g = fine_to_group[d["ct"].astype(int)]
        pos = d["pos"].astype(float)
        ei = d["edge_index"].astype(int)
        if "nhood" in analyses:
            Z[r] = nhood_enrichment_zscores(
                g, ei, n_groups=G, n_perms=n_perms, seed=seed, backend=nhood_backend, coords=pos
            )
        if "cn" in analyses:
            windows.append(window_composition(pos, g, n_groups=G, k=window_k))
        if "contacts" in analyses and reference:
            rec = {"region_id": r}
            rec.update(
                contact_fractions(
                    g,
                    ei,
                    group_names=groups,
                    reference=reference,
                    partners=partners,
                    infiltrating=infiltrating,
                )
            )
            contacts.append(rec)
        if "crossk" in analyses and reference:
            rec = {"region_id": r}
            rec.update(
                cross_k(
                    pos,
                    g,
                    group_names=groups,
                    reference=reference,
                    partners=partners + ["reference"],
                    radii=radii,
                )
            )
            crossk.append(rec)
        if log is not None and (i + 1) % 50 == 0:
            log.info(f"  {i + 1}/{len(regions)} regions")

    md = [
        f"# Classical spatial statistics ({len(regions)} regions, {len(y_pat)} patients; per region -> patient mean; Mann-Whitney {positive_name} vs rest, BH within analysis)",
        "",
    ]
    results: Dict[str, Any] = {"regions": len(regions), "groups": groups}
    if "nhood" in analyses and Z:
        ZT = np.stack([Z[r] for r in regions])
        rows = {}
        for a in range(G):
            for b in range(a, G):
                rows[f"{groups[a]}|{groups[b]}"] = ZT[:, a, b]
        per_region = pd.DataFrame(rows, index=regions)
        per_region.to_csv(out / "nhood_zscores_per_region.csv")
        T = mann_whitney_table(
            patient_mean(per_region, pat_of), y_pat, min_per_group=min_per_group
        )
        T[["a", "b"]] = T["metric"].str.split("|", expand=True)
        T.to_csv(out / "nhood_tests.csv", index=False)
        results["nhood_tests"] = str(out / "nhood_tests.csv")
        md += [
            f"## Neighbourhood enrichment (z-scores on the cell graph, {n_perms} label permutations, {nhood_backend}): pairs with q < 0.1 (top 20 by p otherwise)",
            "",
            "| Pair | z pos | z neg | diff | p | q |",
            "|---|---|---|---|---|---|",
        ]
        sel = T[T.q < 0.1] if (T.q < 0.1).sum() else T.head(20)
        for _, r in sel.iterrows():
            md.append(
                f"| {r.a}-{r.b} | {r.mean_pos:+.2f} | {r.mean_neg:+.2f} | {r['diff']:+.2f} | {r.p:.3g} | {r.q:.2g} |"
            )
    if "cn" in analyses and windows:
        cn_labels, centres = cellular_neighbourhoods(
            windows, n_cn=n_cn, sample_size=cn_sample_size, seed=seed
        )
        CN = pd.DataFrame(
            {f"CN{k}": [float(np.mean(lab == k)) for lab in cn_labels] for k in range(n_cn)},
            index=regions,
        )
        CN.to_csv(out / "cn_shares_per_region.csv")
        CNp = patient_mean(CN, pat_of)
        L = pd.DataFrame(clr(CNp.to_numpy()), index=CNp.index, columns=CNp.columns)
        T = mann_whitney_table(L, y_pat, min_per_group=min_per_group).rename(
            columns={"mean_pos": "clr_pos", "mean_neg": "clr_neg", "diff": "clr_diff"}
        )
        share_pos = CNp[y_pat.reindex(CNp.index) == 1].mean()
        share_neg = CNp[y_pat.reindex(CNp.index) == 0].mean()
        T["share_pos"] = T["metric"].map(share_pos)
        T["share_neg"] = T["metric"].map(share_neg)
        T["composition"] = T["metric"].map(
            {
                f"CN{k}": ", ".join(
                    f"{groups[j]} {100 * centres[k][j]:.0f}%" for j in np.argsort(-centres[k])[:3]
                )
                for k in range(n_cn)
            }
        )
        T.to_csv(out / "cn_tests.csv", index=False)
        pd.DataFrame(centres, columns=groups, index=[f"CN{k}" for k in range(n_cn)]).to_csv(
            out / "cn_centres.csv"
        )
        results["cn_tests"] = str(out / "cn_tests.csv")
        md += [
            "",
            f"## Cellular neighbourhoods ({window_k}-NN windows, k-means k={n_cn}), patient-level CLR shares",
            "",
            "| CN | composition | share pos | share neg | CLR diff | p | q |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in T.iterrows():
            md.append(
                f"| {r.metric} | {r.composition} | {100 * r.share_pos:.1f}% | {100 * r.share_neg:.1f}% | {r.clr_diff:+.2f} | {r.p:.3g} | {r.q:.2g} |"
            )
    if contacts:
        C = pd.DataFrame(contacts).set_index("region_id")
        C.to_csv(out / "contacts_per_region.csv")
        T = mann_whitney_table(patient_mean(C, pat_of), y_pat, min_per_group=min_per_group)
        T.to_csv(out / "contact_tests.csv", index=False)
        results["contact_tests"] = str(out / "contact_tests.csv")
        md += [
            "",
            f"## Contact fractions (reference = {', '.join(reference)})",
            "",
            "| Metric | pos | neg | p | q |",
            "|---|---|---|---|---|",
        ]
        for _, r in T.iterrows():
            md.append(
                f"| {r.metric} | {r.mean_pos:.3f} | {r.mean_neg:.3f} | {r.p:.3g} | {r.q:.2g} |"
            )
    if crossk:
        X = pd.DataFrame(crossk).set_index("region_id")
        X.to_csv(out / "crossk_per_region.csv")
        T = mann_whitney_table(patient_mean(X, pat_of), y_pat, min_per_group=min_per_group)
        T.to_csv(out / "crossk_tests.csv", index=False)
        results["crossk_tests"] = str(out / "crossk_tests.csv")
        md += [
            "",
            f"## Cross-K (Ripley) reference -> partner, normalised by pi r^2 (1 = complete spatial randomness); radii {list(radii)}",
            "",
            "| Metric | pos | neg | p | q |",
            "|---|---|---|---|---|",
        ]
        for _, r in T.iterrows():
            md.append(
                f"| {r.metric} | {r.mean_pos:.2f} | {r.mean_neg:.2f} | {r.p:.3g} | {r.q:.2g} |"
            )
    (out / "report_spatial.md").write_text("\n".join(md) + "\n")
    results["report"] = str(out / "report_spatial.md")
    (out / "spatial_stats_summary.json").write_text(json.dumps(results, indent=1, default=str))
    return results
