"""Stage 4: a shared, label-free k-means basis with per-model attribution.

Fit ``MiniBatchKMeans(k)`` once on standardised embeddings of ONE space (a
chosen model, or a pretrained encoder cache), label every cell, then report
per niche the relattn margin of every fold model split into held-out / train /
all (model, region) pairs with bootstrap intervals, next to the empirical
association. Comparable across models because the niches are the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache
from grass_mil.interpretability.niches.margins import relattn_margins
from grass_mil.interpretability.niches.pipeline import load_cells
from grass_mil.interpretability.niches.tables import bootstrap_mean

__all__ = ["BasisResult", "run_shared_basis", "write_basis_outputs"]


@dataclass
class BasisResult:
    labels: np.ndarray  # (N_all,) niche id
    region_ids: np.ndarray
    pairs: pd.DataFrame  # (model, region, niche, set, relattn, lift, n)
    table: pd.DataFrame
    basis_model: Any
    scaler: Any
    settings: Dict[str, Any]


def run_shared_basis(
    caches: Dict[str, CellCache],
    *,
    basis_cache: Optional[CellCache] = None,
    labels_of: Dict[str, int],
    k: int = 20,
    sample_size: int = 300_000,
    seed: int = 0,
    n_init: int = 10,
    batch_size: int = 4096,
    regions: Optional[Sequence[str]] = None,
    positive_class: int = 1,
    n_boot: int = 2000,
    log=None,
) -> BasisResult:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    models = list(caches)
    basis = basis_cache if basis_cache is not None else caches[models[0]]
    regions = sorted(str(r) for r in (regions if regions is not None else basis.regions))
    cells_b = load_cells(basis, regions)
    RID = cells_b.region_ids
    E = cells_b.E
    rng = np.random.default_rng(int(seed))
    sub = rng.choice(len(E), min(int(sample_size), len(E)), replace=False)
    scaler = StandardScaler().fit(E[sub])
    km = MiniBatchKMeans(
        int(k), random_state=int(seed), n_init=int(n_init), batch_size=int(batch_size)
    ).fit(scaler.transform(E[sub]))
    labels = km.predict(scaler.transform(E)).astype(int)
    fold_of_region = {str(r): name for name, c in caches.items() for r in c.held_out}
    rows = []
    for name, cache in caches.items():
        cells = load_cells(cache, regions)
        if not np.array_equal(cells.region_ids, RID):
            raise ValueError(f"Cache {name} does not cover the same cells as the basis cache.")
        rel = relattn_margins(cells.LG, cells.AR, cells.bias, RID, positive_class=positive_class)
        a_pos = rel.attention[:, min(positive_class, rel.attention.shape[1] - 1)]
        for r in regions:
            m = RID == r
            lab = labels[m]
            mm = rel.margin[m]
            aa = a_pos[m]
            which = "heldout" if fold_of_region.get(r) == name else "train"
            for cl in np.unique(lab):
                msk = lab == cl
                rows.append(
                    dict(
                        model=name,
                        region=r,
                        niche=int(cl),
                        set=which,
                        relattn=float(mm[msk].mean()),
                        lift=float(aa[msk].sum() / msk.mean()),
                        n=int(msk.sum()),
                    )
                )
        if log is not None:
            log.info(f"[{name}] basis attribution over {len(regions)} regions")
    pairs = pd.DataFrame(rows)
    names = cells_b.cell_type_names
    K = len(names)
    ct = cells_b.ct
    y_cell = np.array([int(labels_of[r]) for r in RID])
    cohort = np.bincount(ct, minlength=K) / max(1, ct.size)
    out = []
    for cl in sorted(np.unique(labels)):
        msk = labels == cl
        comp = np.bincount(ct[msk], minlength=K) / max(1, msk.sum())
        enr = comp / np.maximum(cohort, 1e-9)
        rec = dict(
            niche=f"N{int(cl)}",
            niche_id=int(cl),
            share=float(msk.mean()),
            regions=int(pairs[pairs.niche == cl].region.nunique()),
            emp_frac_pos=float(y_cell[msk].mean()),
            top_types=", ".join(f"{names[j]} {100 * comp[j]:.0f}%" for j in np.argsort(-comp)[:3]),
            top_enriched=", ".join(
                f"{names[j]} x{enr[j]:.1f}" for j in np.argsort(-enr)[:3] if enr[j] > 1.5
            ),
        )
        for s in ("heldout", "train"):
            v = pairs[(pairs.niche == cl) & (pairs.set == s)]
            rec[f"relattn_{s}"], rec[f"relattn_{s}_lo"], rec[f"relattn_{s}_hi"] = bootstrap_mean(
                v.relattn, n_boot=n_boot, seed=seed
            )
            rec[f"lift_{s}"] = float(v.lift.mean()) if len(v) else float("nan")
        v = pairs[pairs.niche == cl]
        rec["relattn_all"], rec["relattn_all_lo"], rec["relattn_all_hi"] = bootstrap_mean(
            v.relattn, n_boot=n_boot, seed=seed
        )
        rec["lift_all"] = float(v.lift.mean()) if len(v) else float("nan")
        rec.update({f"comp_{names[j]}": float(comp[j]) for j in range(K)})
        out.append(rec)
    table = pd.DataFrame(out)
    return BasisResult(
        labels=labels,
        region_ids=RID,
        pairs=pairs,
        table=table,
        basis_model=km,
        scaler=scaler,
        settings={
            "k": int(k),
            "sample_size": int(sub.size),
            "seed": int(seed),
            "basis": basis.index.get("model_name"),
            "models": models,
            "n_boot": int(n_boot),
        },
    )


def write_basis_outputs(
    result: BasisResult,
    out_dir: str | Path,
    *,
    positive_name: str = "positive",
    figures: bool = True,
) -> Dict[str, str]:
    import joblib

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "basis_labels.npy", result.labels)
    joblib.dump(
        {"kmeans": result.basis_model, "scaler": result.scaler, "settings": result.settings},
        out / "basis_model.pkl",
    )
    result.pairs.to_csv(out / "basis_attribution_pairs.csv", index=False)
    result.table.to_csv(out / "basis_attribution.csv", index=False)
    T = result.table
    md = [
        f"# Shared k-means basis (k = {result.settings['k']}, fitted on {result.settings['basis']}) with per-model attribution",
        "",
        f"Margins = relattn, region-equal mean, 95% bootstrap over (model, region) pairs; held-out = region scored by the model that held it out; {len(result.settings['models'])} models.",
        "",
        "| Niche | Share | Regions | Top types | Top enriched | Emp. frac. pos | Margin HELD-OUT | Margin TRAIN | Margin ALL | Lift held-out / train |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, t in T.sort_values("emp_frac_pos", ascending=False).iterrows():
        md.append(
            f"| {t.niche} | {100 * t.share:.1f}% | {t.regions} | {t.top_types} | {t.top_enriched} | {t.emp_frac_pos:.2f} | {t.relattn_heldout:+.2f} [{t.relattn_heldout_lo:+.2f}, {t.relattn_heldout_hi:+.2f}] | {t.relattn_train:+.2f} [{t.relattn_train_lo:+.2f}, {t.relattn_train_hi:+.2f}] | {t.relattn_all:+.2f} [{t.relattn_all_lo:+.2f}, {t.relattn_all_hi:+.2f}] | {t.lift_heldout:.2f} / {t.lift_train:.2f} |"
        )
    if len(T) >= 3:
        from scipy.stats import spearmanr

        for s in ("heldout", "train", "all"):
            md.append(
                f"\nSpearman(emp_frac_pos, margin_{s}) over {len(T)} niches: rho={spearmanr(T.emp_frac_pos, T[f'relattn_{s}'], nan_policy='omit')[0]:+.2f}"
            )
    (out / "report_basis.md").write_text("\n".join(md) + "\n")
    paths = {"report": str(out / "report_basis.md")}
    if figures:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            T2 = T.sort_values("emp_frac_pos", ascending=False).reset_index(drop=True)
            base = float(np.average(T2.emp_frac_pos, weights=T2.share))
            yy = np.arange(len(T2))[::-1]
            fig, ax = plt.subplots(
                1, 2, figsize=(10, 0.45 * len(T2) + 1.8), gridspec_kw=dict(width_ratios=[1, 1.6])
            )
            ax[0].barh(
                yy,
                T2.emp_frac_pos,
                color=["#b2182b" if v > base else "#2166ac" for v in T2.emp_frac_pos],
                height=0.6,
            )
            ax[0].axvline(base, color="k", ls=":", lw=1)
            ax[0].set_yticks(yy)
            ax[0].set_yticklabels(
                [
                    f"{n} ({100 * s:.1f}%): {t.split(',')[0]}"
                    for n, s, t in zip(T2.niche, T2.share, T2.top_enriched)
                ],
                fontsize=7,
            )
            ax[0].set_xlabel(f"fraction of niche cells from {positive_name} regions")
            for s, col, off in (
                ("heldout", "#1b9e77", 0.22),
                ("train", "#d95f02", 0.0),
                ("all", "#7570b3", -0.22),
            ):
                m_, lo, hi = T2[f"relattn_{s}"], T2[f"relattn_{s}_lo"], T2[f"relattn_{s}_hi"]
                ax[1].errorbar(
                    m_,
                    yy + off,
                    xerr=[m_ - lo, hi - m_],
                    fmt="o",
                    ms=4,
                    color=col,
                    capsize=2,
                    lw=1,
                    label=s,
                )
            ax[1].axvline(0, color="k", lw=0.8)
            ax[1].set_yticks(yy)
            ax[1].set_yticklabels([])
            ax[1].set_xlabel(f"attention-weighted margin (relattn), + = {positive_name}")
            ax[1].legend(fontsize=7, loc="lower right")
            for a in ax:
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(out / "basis_attribution.png", dpi=160)
            plt.close(fig)
            paths["figure"] = str(out / "basis_attribution.png")
        except Exception:  # pragma: no cover
            pass
    return paths
