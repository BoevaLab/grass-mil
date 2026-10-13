"""Per-niche summary table (the preprint's cluster table)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["niche_table", "spearman_concordance", "bootstrap_mean"]


def bootstrap_mean(values, *, n_boot: int = 2000, seed: int = 0):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    g = np.random.default_rng(seed)
    bs = [g.choice(v, v.size, replace=True).mean() for _ in range(int(n_boot))]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def niche_table(
    labels: np.ndarray,
    *,
    cell_types: np.ndarray,
    region_ids: np.ndarray,
    region_labels: Dict[str, int],
    margin: np.ndarray,
    attention_pos: np.ndarray,
    cell_type_names: Optional[Sequence[str]] = None,
    present_threshold: float = 0.02,
    enrichment_threshold: float = 1.5,
    n_boot: int = 2000,
    seed: int = 0,
    include_noise: bool = True,
) -> pd.DataFrame:
    """One row per niche: share, cores present, composition, enrichment, empirical
    fraction of cells from positive regions, share-AUROC, relattn margin with a
    bootstrap CI over regions, signed share, sign consistency and attention lift.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    ct = np.asarray(cell_types).astype(int)
    rid = np.asarray(region_ids)
    M = np.asarray(margin, dtype=float)
    A1 = np.asarray(attention_pos, dtype=float)
    K = int(ct.max()) + 1 if ct.size else 0
    if cell_type_names is not None:
        K = max(K, len(cell_type_names))
    names = list(cell_type_names) if cell_type_names is not None else [str(i) for i in range(K)]
    regions = sorted(set(rid.tolist()))
    y_region = np.array([int(region_labels[r]) for r in regions])
    y_cell = np.array([int(region_labels[r]) for r in rid])
    cohort = np.bincount(ct, minlength=K) / max(1, ct.size)
    region_mask = {r: rid == r for r in regions}
    region_size = {r: int(m.sum()) for r, m in region_mask.items()}
    region_margin = {r: float(M[m].sum()) for r, m in region_mask.items()}
    clusters = sorted(set(labels.tolist()) - {-1})
    if include_noise and (labels == -1).any():
        clusters = clusters + [-1]
    rows: List[dict] = []
    for cl in clusters:
        msk = labels == cl
        comp = np.bincount(ct[msk], minlength=K) / max(1, msk.sum())
        enr = comp / np.maximum(cohort, 1e-9)
        vals, shares, signs, lifts = [], [], [], []
        per_region_share = np.zeros(len(regions))
        for j, r in enumerate(regions):
            mm = msk & region_mask[r]
            s = mm.sum() / max(1, region_size[r])
            per_region_share[j] = s
            if mm.any():
                vals.append(M[mm].mean())
                mcr = float(M[mm].sum())
                Mr = region_margin[r]
                if abs(Mr) > 1e-6:
                    shares.append(mcr / Mr)
                    signs.append(float(np.sign(mcr) == np.sign(Mr)))
                lifts.append(A1[mm].sum() / s)
        present = per_region_share > present_threshold
        m_, lo, hi = bootstrap_mean(vals, n_boot=n_boot, seed=seed)
        ss = bootstrap_mean(shares, n_boot=n_boot, seed=seed)[0] if shares else float("nan")
        both = 0 < y_region.sum() < y_region.size
        rows.append(
            dict(
                cluster="Noise" if cl == -1 else f"C{cl}",
                cluster_id=int(cl),
                share=float(msk.mean()),
                n_cells=int(msk.sum()),
                cores_present=int(present.sum()),
                pos_cores_present=int((present & (y_region == 1)).sum()),
                neg_cores_present=int((present & (y_region == 0)).sum()),
                emp_frac_pos=float(y_cell[msk].mean()) if msk.any() else float("nan"),
                share_auroc=float(roc_auc_score(y_region, per_region_share))
                if both
                else float("nan"),
                relattn=m_,
                relattn_lo=lo,
                relattn_hi=hi,
                signed_share=ss,
                sign_consistency=float(np.mean(signs)) if signs else float("nan"),
                lift=float(np.mean(lifts)) if lifts else float("nan"),
                top_types=", ".join(
                    f"{names[j]} {100 * comp[j]:.0f}%" for j in np.argsort(-comp)[:4]
                ),
                top_enriched=", ".join(
                    f"{names[j]} x{enr[j]:.1f}"
                    for j in np.argsort(-enr)[:4]
                    if enr[j] > enrichment_threshold
                ),
                **{f"comp_{names[j]}": float(comp[j]) for j in range(K)},
            )
        )
    return pd.DataFrame(rows)


def spearman_concordance(table: pd.DataFrame) -> Dict[str, float]:
    """Spearman correlations between the empirical association and the margin."""
    from scipy.stats import spearmanr

    tt = table[table["cluster"] != "Noise"]
    out = {"n_clusters": int(len(tt))}
    if len(tt) >= 3:
        out["spearman_emp_relattn"] = float(spearmanr(tt["emp_frac_pos"], tt["relattn"])[0])
        if tt["share_auroc"].notna().sum() >= 3:
            out["spearman_share_auroc_relattn"] = float(
                spearmanr(tt["share_auroc"], tt["relattn"], nan_policy="omit")[0]
            )
    return out
