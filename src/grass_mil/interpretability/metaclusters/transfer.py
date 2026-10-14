"""Stage 3: transfer every fold model's clustering to ALL regions in its own space.

Each fold's saved cluster model (reducer + k-NN assigner fitted on its held-out
sample) labels every region's cells under that model's embedding; fold-cluster
ids map to metaclusters. Reports, per metacluster and model, share /
association / margin on TRAIN vs HELD-OUT regions, the pairwise ARI of the
models' metacluster labels on cells both assign, and consensus patient shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from grass_mil.inference.cell_cache import CellCache
from grass_mil.interpretability.niches.margins import relattn_margins
from grass_mil.interpretability.niches.pipeline import load_cells

from .match import patient_tests

__all__ = ["TransferResult", "run_transfer", "write_transfer_outputs"]


@dataclass
class TransferResult:
    meta_labels: np.ndarray  # (n_models, N_all) metacluster id or -1
    region_ids: np.ndarray  # (N_all,)
    models: List[str]
    by_model: pd.DataFrame
    ari: pd.DataFrame
    consensus_region_shares: pd.DataFrame
    consensus_patient_shares: Optional[pd.DataFrame] = None
    consensus_patient_tests: Optional[pd.DataFrame] = None
    refit_agreement: Dict[str, float] = field(default_factory=dict)


def run_transfer(
    caches: Dict[str, CellCache],
    *,
    cluster_dir: str | Path,
    matched_records: pd.DataFrame,
    labels_of: Dict[str, int],
    patient_of: Optional[Dict[str, str]] = None,
    regions: Optional[Sequence[str]] = None,
    positive_class: int = 1,
    min_cells: int = 100,
    present_threshold: float = 0.02,
    log=None,
) -> TransferResult:
    import joblib
    from sklearn.metrics import adjusted_rand_score

    cluster_dir = Path(cluster_dir)
    models = list(caches)
    first = caches[models[0]]
    regions = sorted(str(r) for r in (regions if regions is not None else first.regions))
    cells0 = load_cells(first, regions)
    RID = cells0.region_ids
    N = RID.size
    y_cell = np.array([int(labels_of[r]) for r in RID])
    fold_of_region = {}
    for name, cache in caches.items():
        for r in cache.held_out:
            fold_of_region[str(r)] = name
    split_model = np.array([fold_of_region.get(r, "") for r in RID], dtype=object)
    meta_of = {(str(r.fold), int(r.cluster)): int(r.meta) for _, r in matched_records.iterrows()}
    ML = np.full((len(models), N), -1, dtype=int)
    MG = np.zeros((len(models), N), dtype=np.float32)
    refit_agreement: Dict[str, float] = {}
    for mi, name in enumerate(models):
        cache = caches[name]
        cells = load_cells(cache, regions)
        if not np.array_equal(cells.region_ids, RID):
            raise ValueError(f"Cache {name} does not cover the same cells as {models[0]}.")
        model = joblib.load(cluster_dir / f"cluster_model_{name}.pkl")
        labels_all = model.predict(cells.E)
        ref_path = cluster_dir / f"labels_{name}.npy"
        if ref_path.exists():
            held = split_model == name
            ref = np.load(ref_path)
            if ref.size == int(held.sum()):
                refit_agreement[name] = float((labels_all[held] == ref).mean())
                labels_all[held] = ref  # the held-out cells keep their stage-1 labels exactly
        ML[mi] = np.array([meta_of.get((name, int(c)), -1) if c >= 0 else -1 for c in labels_all])
        rel = relattn_margins(cells.LG, cells.AR, cells.bias, RID, positive_class=positive_class)
        MG[mi] = rel.margin
        if log is not None:
            log.info(
                f"[{name}] transferred to {len(regions)} regions; refit agreement {refit_agreement.get(name, float('nan')):.3f}"
            )
    metas = sorted(set(matched_records["meta"].astype(int)))
    rows = []
    for k in metas:
        for mi, name in enumerate(models):
            mk = ML[mi] == k
            if mk.sum() < int(min_cells):
                continue
            for split_name, sm in (
                ("train", split_model != name),
                ("heldout", split_model == name),
            ):
                ms = mk & sm
                if not sm.any():
                    continue
                base = float(y_cell[sm].mean())
                present = [
                    r
                    for r in regions
                    if ((split_model[RID == r][0] == name) == (split_name == "heldout"))
                    and ms[RID == r].mean() >= present_threshold
                ]
                vals = [
                    float(MG[mi][ms & (RID == r)].mean())
                    for r in present
                    if (ms & (RID == r)).any()
                ]
                rows.append(
                    dict(
                        meta=k,
                        model=name,
                        split=split_name,
                        share=float(ms.sum() / sm.sum()),
                        cores=len(present),
                        pos_cores=int(sum(labels_of[r] for r in present)),
                        emp=float(y_cell[ms].mean()) if ms.any() else float("nan"),
                        emp_rel=(float(y_cell[ms].mean()) - base) if ms.any() else float("nan"),
                        margin=float(np.mean(vals)) if vals else float("nan"),
                    )
                )
    by_model = pd.DataFrame(rows)
    ari = pd.DataFrame(np.nan, index=models, columns=models)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            both = (ML[i] >= 0) & (ML[j] >= 0)
            if both.sum() > 10:
                ari.loc[a, b] = adjusted_rand_score(ML[i][both], ML[j][both])
    region_shares = pd.DataFrame(
        {
            f"M{k}": [
                float(np.mean([np.mean(ML[i][RID == r] == k) for i in range(len(models))]))
                for r in regions
            ]
            for k in metas
        },
        index=regions,
    )
    region_shares.index.name = "region_id"
    result = TransferResult(
        meta_labels=ML,
        region_ids=RID,
        models=models,
        by_model=by_model,
        ari=ari,
        consensus_region_shares=region_shares,
        refit_agreement=refit_agreement,
    )
    pat_of = patient_of or {r: r for r in regions}
    pat = region_shares.copy()
    pat["patient_id"] = [pat_of.get(r, r) for r in regions]
    result.consensus_patient_shares = pat.groupby("patient_id").mean()
    y_pat = pd.Series({pat_of.get(r, r): int(labels_of[r]) for r in regions})
    result.consensus_patient_tests = patient_tests(
        result.consensus_patient_shares, y_pat, present_threshold=0.005
    )
    return result


def write_transfer_outputs(
    result: TransferResult,
    out_dir: str | Path,
    *,
    metaclusters: pd.DataFrame,
    positive_name: str = "positive",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "transfer_metalabels.npy", result.meta_labels)
    pd.DataFrame({"region_id": result.region_ids}).to_csv(out / "transfer_cells.csv", index=False)
    result.by_model.to_csv(out / "transfer_meta_by_model.csv", index=False)
    result.ari.to_csv(out / "transfer_model_ari.csv")
    result.consensus_region_shares.to_csv(out / "transfer_region_shares.csv")
    if result.consensus_patient_shares is not None:
        result.consensus_patient_shares.to_csv(out / "transfer_patient_shares.csv")
    if result.consensus_patient_tests is not None:
        result.consensus_patient_tests.to_csv(out / "transfer_patient_tests.csv", index=False)
    MT = metaclusters.set_index("meta")
    models = result.models
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    md = [
        "# Metaclusters transferred to all regions by each fold model",
        "",
        "Stage-1 refit agreement (held-out cells relabelled by the saved model): "
        + "; ".join(f"{k} {v:.3f}" for k, v in result.refit_agreement.items()),
        "",
        "Pairwise ARI of metacluster labels between models on cells both assign: "
        + "; ".join(f"{a}-{b} {result.ari.loc[a, b]:.2f}" for a, b in pairs),
        "",
        "| Meta | folds | composition | model | share train / held-out | regions train (pos) / held-out (pos) | emp-base train / held-out | margin train / held-out |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in MT.index:
        g = result.by_model[result.by_model["meta"] == k]
        first = True
        for m in models:
            a = g[(g.model == m) & (g.split == "train")]
            h = g[(g.model == m) & (g.split == "heldout")]
            if a.empty or h.empty:
                continue
            a, h = a.iloc[0], h.iloc[0]
            md.append(
                f"| {'M' + str(k) if first else ''} | {MT.loc[k, 'n_folds'] if first else ''} | {MT.loc[k, 'top_types'] if first else ''} | {m} | {100 * a.share:.2f}% / {100 * h.share:.2f}% | {a.cores} ({a.pos_cores}) / {h.cores} ({h.pos_cores}) | {a.emp_rel:+.2f} / {h.emp_rel:+.2f} | {a.margin:+.2f} / {h.margin:+.2f} |"
            )
            first = False
    if result.consensus_patient_tests is not None:
        md += [
            "",
            "## Consensus patient-level shares (mean over the models' assignments of every region)",
            "",
            "| Meta | folds | composition | share pos | share neg | present pos / neg | AUROC | p | q |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in result.consensus_patient_tests.iterrows():
            md.append(
                f"| M{r.meta} | {MT.loc[r.meta, 'n_folds'] if r.meta in MT.index else ''} | {MT.loc[r.meta, 'top_types'] if r.meta in MT.index else ''} | {100 * r.share_pos:.2f}% | {100 * r.share_neg:.2f}% | {100 * r.present_pos:.0f}% / {100 * r.present_neg:.0f}% | {r.auroc:.3f} | {r.p:.3g} | {r.q:.2g} |"
            )
    (out / "report_transfer.md").write_text("\n".join(md) + "\n")
    return {"report": str(out / "report_transfer.md")}
