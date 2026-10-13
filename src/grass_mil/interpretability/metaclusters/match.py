"""Stage 2: match fold-clusters across folds into metaclusters.

Distance between two fold-clusters = Hellinger distance of their cell-type
compositions (or Aitchison / CLR euclidean). Constrained average linkage never
merges two clusters of the same fold (cannot-link). The threshold is either
fixed or chosen by a scan that maximises the number of metaclusters replicated
in at least ``min_folds`` folds (ties -> fewer metaclusters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MatchResult",
    "aitchison_matrix",
    "constrained_average_linkage",
    "hellinger_matrix",
    "match_metaclusters",
    "metacluster_table",
    "patient_shares",
    "patient_tests",
    "scan_threshold",
]


def hellinger_matrix(comp: np.ndarray) -> np.ndarray:
    """Pairwise Hellinger distance between composition rows (in [0, 1])."""
    H = np.sqrt(np.clip(np.asarray(comp, dtype=float), 0.0, None))
    return np.sqrt(0.5 * ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1))


def aitchison_matrix(comp: np.ndarray, pseudo: float = 1e-3) -> np.ndarray:
    L = np.log(np.asarray(comp, dtype=float) + pseudo)
    L = L - L.mean(axis=1, keepdims=True)
    return np.sqrt(((L[:, None, :] - L[None, :, :]) ** 2).sum(-1))


def constrained_average_linkage(
    D: np.ndarray, fold: Sequence[Any], threshold: float
) -> List[List[int]]:
    """Agglomerate greedily by average distance below ``threshold``; groups sharing a fold never merge."""
    fold = np.asarray(fold)
    n = D.shape[0]
    groups: List[List[int]] = [[i] for i in range(n)]
    while True:
        best = None
        for a in range(len(groups)):
            fa = set(fold[groups[a]].tolist())
            for b in range(a + 1, len(groups)):
                if fa & set(fold[groups[b]].tolist()):
                    continue
                d = float(np.mean([D[i, j] for i in groups[a] for j in groups[b]]))
                if d < threshold and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        groups[a] = groups[a] + groups[b]
        del groups[b]
    return groups


def scan_threshold(
    D: np.ndarray, fold: Sequence[Any], *, grid: Sequence[float], min_folds: int = 3
) -> Tuple[float, List[List[int]], pd.DataFrame]:
    """Pick the threshold maximising replicated metaclusters, ties -> fewer metaclusters."""
    rows = []
    best = None
    for thr in grid:
        groups = constrained_average_linkage(D, fold, float(thr))
        rep = sum(len(g) >= int(min_folds) for g in groups)
        key = (rep, -len(groups))
        rows.append({"threshold": float(thr), "n_metaclusters": len(groups), "n_replicated": rep})
        if best is None or key > best[0]:
            best = (key, float(thr), groups)
    return best[1], best[2], pd.DataFrame(rows)


@dataclass
class MatchResult:
    threshold: float
    distance: str
    records: pd.DataFrame  # fold-cluster records with a `meta` column
    table: pd.DataFrame  # one row per metacluster
    D: np.ndarray
    scan: Optional[pd.DataFrame] = None
    patient_shares: Optional[pd.DataFrame] = None
    patient_tests: Optional[pd.DataFrame] = None
    region_shares: Optional[pd.DataFrame] = None
    settings: Dict[str, Any] = field(default_factory=dict)


def metacluster_table(
    records: pd.DataFrame, D: np.ndarray, cohort: np.ndarray, names: Sequence[str]
) -> pd.DataFrame:
    comp_cols = [f"comp_{n}" for n in names]
    rows = []
    for k in sorted(records["meta"].unique()):
        g = records[records["meta"] == k]
        comp = np.average(
            g[comp_cols].to_numpy(dtype=float), axis=0, weights=g["n_cells"].to_numpy(dtype=float)
        )
        enr = comp / np.maximum(np.asarray(cohort, dtype=float), 1e-9)
        idx = g.index.to_list()
        within = (
            float(np.mean([D[i, j] for i in idx for j in idx if i < j])) if len(idx) > 1 else 0.0
        )
        rows.append(
            dict(
                meta=int(k),
                n_folds=int(g["fold"].nunique()),
                n_members=int(len(g)),
                folds=",".join(str(f) for f in sorted(g["fold"].unique())),
                members=", ".join(g["id"]),
                within_dist=within,
                share_mean=float(g["share"].mean()),
                cores=int(g["cores"].sum()),
                pos_cores=int(g["pos_cores"].sum()),
                neg_cores=int(g["neg_cores"].sum()),
                emp_mean=float(g["emp"].mean()),
                emp_rel_mean=float(g["emp_rel"].mean()),
                emp_rel_sd=float(g["emp_rel"].std(ddof=1)) if len(g) > 1 else float("nan"),
                emp_per_fold=";".join(f"{f}:{e:+.2f}" for f, e in zip(g["fold"], g["emp_rel"])),
                margin_mean=float(g["margin"].mean()),
                margin_sd=float(g["margin"].std(ddof=1)) if len(g) > 1 else float("nan"),
                margin_per_fold=";".join(f"{f}:{m:+.2f}" for f, m in zip(g["fold"], g["margin"])),
                sign_agree=float(np.mean(np.sign(g["margin"]) == np.sign(g["margin"].mean()))),
                ci_excl0=int(((g["margin_lo"] > 0) | (g["margin_hi"] < 0)).sum()),
                emp_margin_agree=float(np.mean(np.sign(g["emp_rel"]) == np.sign(g["margin"]))),
                lift_mean=float(g["lift"].mean()),
                top_types=", ".join(
                    f"{names[j]} {100 * comp[j]:.0f}%" for j in np.argsort(-comp)[:4]
                ),
                top_enriched=", ".join(
                    f"{names[j]} x{enr[j]:.1f}" for j in np.argsort(-enr)[:4] if enr[j] > 1.5
                ),
                **{c: float(v) for c, v in zip(comp_cols, comp)},
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_folds", "emp_rel_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )


def patient_shares(
    region_labels: Dict[str, np.ndarray],
    region_fold: Dict[str, str],
    id_to_meta: Dict[str, int],
    patient_of: Dict[str, str],
    metas: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-region and per-patient metacluster shares (each region labelled once)."""
    cols = [f"M{k}" for k in metas]
    regions = sorted(region_labels)
    SH = pd.DataFrame(0.0, index=regions, columns=cols)
    for r in regions:
        lab = region_labels[r]
        f = region_fold[r]
        for c in set(lab.tolist()) - {-1}:
            k = id_to_meta.get(f"{f}_C{c}")
            if k is not None:
                SH.loc[r, f"M{k}"] += float(np.mean(lab == c))
    SH.index.name = "region_id"
    pat = SH.copy()
    pat["patient_id"] = [patient_of.get(r, r) for r in regions]
    return SH, pat.groupby("patient_id").mean()


def patient_tests(
    shares: pd.DataFrame, y: pd.Series, *, present_threshold: float = 0.0
) -> pd.DataFrame:
    """Mann-Whitney (BH) and AUROC of each metacluster share between classes."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    yy = y.reindex(shares.index)
    ok = yy.notna()
    shares = shares.loc[ok]
    yy = yy.loc[ok].astype(int).to_numpy()
    rows = []
    for col in shares.columns:
        v = shares[col].to_numpy(dtype=float)
        a, b = v[yy == 1], v[yy == 0]
        both = a.size > 2 and b.size > 2
        rows.append(
            dict(
                meta=int(col[1:]),
                share_pos=float(a.mean()) if a.size else float("nan"),
                share_neg=float(b.mean()) if b.size else float("nan"),
                present_pos=float((a > present_threshold).mean()) if a.size else float("nan"),
                present_neg=float((b > present_threshold).mean()) if b.size else float("nan"),
                auroc=float(roc_auc_score(yy, v)) if both else float("nan"),
                p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                if both
                else float("nan"),
            )
        )
    T = pd.DataFrame(rows)
    T["q"] = _bh(T["p"].to_numpy())
    return T.sort_values("p").reset_index(drop=True)


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full(p.size, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        from scipy.stats import false_discovery_control

        q[ok] = false_discovery_control(p[ok], method="bh")
    return q


def match_metaclusters(
    records: pd.DataFrame,
    *,
    cell_type_names: Sequence[str],
    cohort_frequency: np.ndarray,
    distance: str = "hellinger",
    threshold: Optional[float] = None,
    scan_grid: Sequence[float] = tuple(np.linspace(0.15, 0.6, 46)),
    min_folds: int = 3,
    region_labels: Optional[Dict[str, np.ndarray]] = None,
    region_fold: Optional[Dict[str, str]] = None,
    patient_of: Optional[Dict[str, str]] = None,
    labels_of: Optional[Dict[str, int]] = None,
) -> MatchResult:
    records = records.reset_index(drop=True).copy()
    comp_cols = [f"comp_{n}" for n in cell_type_names]
    comp = records[comp_cols].to_numpy(dtype=float)
    if distance == "hellinger":
        D = hellinger_matrix(comp)
    elif distance == "aitchison":
        D = aitchison_matrix(comp)
    else:
        raise ValueError("distance must be hellinger or aitchison")
    fold = records["fold"].to_numpy()
    scan = None
    if threshold is None:
        threshold, groups, scan = scan_threshold(D, fold, grid=scan_grid, min_folds=min_folds)
    else:
        groups = constrained_average_linkage(D, fold, float(threshold))
    meta = np.zeros(len(records), dtype=int)
    for k, g in enumerate(sorted(groups, key=lambda x: -len(x)), 1):
        meta[g] = k
    records["meta"] = meta
    table = metacluster_table(records, D, cohort_frequency, cell_type_names)
    result = MatchResult(
        threshold=float(threshold),
        distance=distance,
        records=records,
        table=table,
        D=D,
        scan=scan,
        settings={
            "min_folds": min_folds,
            "n_metaclusters": int(len(groups)),
            "n_replicated": int(sum(len(g) >= min_folds for g in groups)),
        },
    )
    if region_labels is not None and region_fold is not None:
        id_to_meta = dict(zip(records["id"], records["meta"]))
        pat_of = patient_of or {r: r for r in region_labels}
        region_sh, pat_sh = patient_shares(
            region_labels, region_fold, id_to_meta, pat_of, sorted(records["meta"].unique())
        )
        result.region_shares = region_sh
        result.patient_shares = pat_sh
        if labels_of is not None:
            y_pat = pd.Series(
                {pat_of.get(r, r): int(v) for r, v in labels_of.items() if r in region_labels}
            )
            result.patient_tests = patient_tests(pat_sh, y_pat)
    return result


def write_match_outputs(
    result: MatchResult,
    out_dir: str | Path,
    *,
    cell_type_names: Sequence[str],
    figures: bool = True,
    positive_name: str = "positive",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    result.records.to_csv(out / "fold_clusters_matched.csv", index=False)
    result.table.to_csv(out / "metaclusters.csv", index=False)
    paths["metaclusters"] = str(out / "metaclusters.csv")
    np.save(out / "fold_cluster_distances.npy", result.D)
    if result.scan is not None:
        result.scan.to_csv(out / "threshold_scan.csv", index=False)
    if result.patient_shares is not None:
        result.patient_shares.to_csv(out / "metacluster_patient_shares.csv")
        result.region_shares.to_csv(out / "metacluster_region_shares.csv")
    if result.patient_tests is not None:
        result.patient_tests.to_csv(out / "metacluster_patient_tests.csv", index=False)
    MT = result.table
    md = [
        f"# Cross-fold niche replication: {result.distance} distance on composition, cannot-link within fold, threshold {result.threshold:.3f} -> {result.settings['n_metaclusters']} metaclusters, {result.settings['n_replicated']} replicated in >= {result.settings['min_folds']} folds ({len(result.records)} fold-clusters)",
        "",
        "| Meta | folds | share | regions (pos/neg) | composition | enriched | within-dist | emp. minus base, per fold | mean (sd) | margin per fold | mean (sd) | sign agree | CI excl 0 | emp/margin agree |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in MT.iterrows():
        md.append(
            f"| M{r.meta} | {r.n_folds} ({r.folds}) | {100 * r.share_mean:.1f}% | {r.cores} ({r.pos_cores}/{r.neg_cores}) | {r.top_types} | {r.top_enriched} | {r.within_dist:.2f} | {r.emp_per_fold} | {r.emp_rel_mean:+.2f} ({r.emp_rel_sd:.2f}) | {r.margin_per_fold} | {r.margin_mean:+.2f} ({r.margin_sd:.2f}) | {r.sign_agree:.2f} | {r.ci_excl0}/{r.n_members} | {r.emp_margin_agree:.2f} |"
        )
    if result.patient_tests is not None:
        nf = dict(zip(MT.meta, MT.n_folds))
        md += [
            "",
            f"## Patient-level metacluster shares (each region labelled once, by the model that held it out); {positive_name} vs rest",
            "",
            "| Meta | folds | share pos | share neg | present pos / neg | AUROC | p | q |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in result.patient_tests.iterrows():
            md.append(
                f"| M{r.meta} | {nf.get(r.meta, '')} | {100 * r.share_pos:.2f}% | {100 * r.share_neg:.2f}% | {100 * r.present_pos:.0f}% / {100 * r.present_neg:.0f}% | {r.auroc:.3f} | {r.p:.3g} | {r.q:.2g} |"
            )
    (out / "report.md").write_text("\n".join(md) + "\n")
    paths["report"] = str(out / "report.md")
    if figures:
        try:
            _overview_figure(
                result,
                out / "metaclusters_overview.png",
                cell_type_names=cell_type_names,
                positive_name=positive_name,
            )
            paths["overview"] = str(out / "metaclusters_overview.png")
        except Exception:  # pragma: no cover - matplotlib missing
            pass
    return paths


def _overview_figure(
    result: MatchResult,
    path: Path,
    *,
    cell_type_names: Sequence[str],
    positive_name: str,
    max_types: int = 20,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MT = result.table
    if MT.empty:
        return
    comp_cols = [f"comp_{n}" for n in cell_type_names]
    cohort = MT[comp_cols].to_numpy(dtype=float).mean(0)
    typ = np.argsort(-cohort)[:max_types]
    H = MT[[comp_cols[j] for j in typ]].to_numpy(dtype=float) * 100
    m = len(MT)
    folds = sorted(result.records["fold"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(
        1, 3, figsize=(17, 0.5 * m + 2.4), gridspec_kw=dict(width_ratios=[2.6, 1.1, 1.3])
    )
    im = ax[0].imshow(H, aspect="auto", cmap="viridis", vmin=0, vmax=min(100, H.max()))
    ax[0].set_xticks(range(len(typ)))
    ax[0].set_xticklabels([cell_type_names[j] for j in typ], rotation=90, fontsize=7)
    ax[0].set_yticks(range(m))
    ax[0].set_yticklabels(
        [
            f"M{a} ({b} folds, {100 * s:.1f}%)"
            for a, b, s in zip(MT.meta, MT.n_folds, MT.share_mean)
        ],
        fontsize=7,
    )
    plt.colorbar(im, ax=ax[0], label="% of metacluster cells", fraction=0.03)
    yy = np.arange(m)[::-1]
    for k, (_, r) in enumerate(MT.iterrows()):
        g = result.records[result.records["meta"] == r.meta]
        for f, e, mg, lo, hi in zip(
            g["fold"], g["emp_rel"], g["margin"], g["margin_lo"], g["margin_hi"]
        ):
            fi = folds.index(f)
            off = (fi - len(folds) / 2) * 0.12
            ax[1].scatter(e, yy[k] + off, s=20, color=cmap(fi % 10), zorder=3)
            ax[2].errorbar(
                mg,
                yy[k] + off,
                xerr=[[mg - lo], [hi - mg]],
                fmt="o",
                ms=3.5,
                color=cmap(fi % 10),
                capsize=1.5,
                lw=0.8,
            )
    for a in ax[1:]:
        a.axvline(0, color="k", lw=0.8)
        a.set_yticks(yy)
        a.set_yticklabels([])
        a.set_ylim(-0.7, m - 0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax[1].set_xlabel(f"empirical {positive_name} fraction minus fold base rate")
    ax[2].set_xlabel(f"attention-weighted margin (+ = {positive_name})")
    ax[1].set_title("association, one dot per fold", fontsize=9)
    ax[2].set_title("attribution, one point per fold", fontsize=9)
    for fi, f in enumerate(folds):
        ax[2].scatter([], [], color=cmap(fi % 10), label=str(f))
    ax[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
