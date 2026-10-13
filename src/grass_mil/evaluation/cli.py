"""``grass-mil-scores``: AUROC CIs, DeLong comparisons and operating points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from grass_mil._paths import setup_project_root

setup_project_root()

from grass_mil.evaluation.operating_points import (  # noqa: E402
    flag_top_within_fold,
    likelihood_ratios,
    operating_point,
    operating_point_bootstrap,
    ppv_npv_at_prevalence,
    threshold_points,
)
from grass_mil.evaluation.stats import (  # noqa: E402
    auroc,
    bootstrap_auroc_ci,
    delong_test,
    per_fold_auroc,
    rank_within_fold,
    stratified_auroc,
    z_within_fold,
)
from grass_mil.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ["run_scores", "main"]


def _prepare_table(cfg: DictConfig) -> pd.DataFrame:
    df = pd.read_csv(str(cfg.input))
    df.columns = [str(c).strip() for c in df.columns]
    label = str(cfg.label_column)
    scores = [str(s) for s in cfg.scores]
    fold = cfg.get("fold_column")
    patient = cfg.get("patient_column")
    strata = [str(s) for s in (cfg.get("strata") or [])]
    for col in [label] + scores + ([str(fold)] if fold else []) + strata:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in {cfg.input}; columns: {list(df.columns)}")
    df = df.dropna(subset=[label] + scores)
    if patient and str(patient) in df.columns and bool(cfg.get("patient_level", True)):
        agg: Dict[str, Any] = {s: "mean" for s in scores}
        agg[label] = "max"
        if fold:
            agg[str(fold)] = "first"
        for s in strata:
            agg[s] = "first"
        agg["n_units"] = ("__one", "sum")
        df = df.assign(__one=1)
        grouped = df.groupby(str(patient))
        out = grouped.agg({k: v for k, v in agg.items() if k != "n_units"})
        out["n_units"] = grouped["__one"].sum()
        df = out.reset_index()
    df[label] = (df[label].astype(float) >= 0.5).astype(int)
    return df


def _standardise(cfg: DictConfig, df: pd.DataFrame, col: str) -> np.ndarray:
    how = str(cfg.get("standardise", "z")).lower()
    fold = cfg.get("fold_column")
    values = df[col].to_numpy(dtype=float)
    if not fold or how == "none":
        return values
    folds = df[str(fold)].to_numpy()
    if how == "z":
        return z_within_fold(values, folds)
    if how == "rank":
        return rank_within_fold(values, folds)
    raise ValueError("standardise must be z, rank or none")


def run_scores(cfg: DictConfig) -> Dict[str, Any]:
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare_table(cfg)
    label = str(cfg.label_column)
    y = df[label].to_numpy(dtype=int)
    fold = cfg.get("fold_column")
    folds = df[str(fold)].to_numpy() if fold else None
    n_boot = int(cfg.n_boot)
    seed = int(cfg.seed)
    unit = (
        "patients"
        if cfg.get("patient_column") and bool(cfg.get("patient_level", True))
        else "units"
    )
    prevalence = float(y.mean())
    md: List[str] = [
        f"# Held-out score evaluation ({len(y)} {unit}, {int(y.sum())} positive, prevalence {100 * prevalence:.0f} %)",
        "",
        f"Scores standardised within fold ({cfg.get('standardise', 'z')}) before pooling; "
        f"{n_boot} bootstrap resamples, seed {seed}.",
        "",
    ]

    scores = [str(s) for s in cfg.scores]
    std_scores = {s: _standardise(cfg, df, s) for s in scores}
    summary_rows = []
    md += [
        "## AUROC",
        "",
        "| score | pooled AUROC [95% CI] | per-fold mean ± sd | raw pooled |",
        "|---|---|---|---|",
    ]
    for s in scores:
        ci = bootstrap_auroc_ci(y, std_scores[s], n_boot=n_boot, seed=seed)
        row = {
            "score": s,
            "auroc": ci.auroc,
            "lower": ci.lower,
            "upper": ci.upper,
            "raw_pooled_auroc": auroc(y, df[s].to_numpy(float)),
        }
        pf_txt = "–"
        if folds is not None:
            pf = per_fold_auroc(y, df[s].to_numpy(float), folds)
            row.update({f"perfold_{k}": v for k, v in pf.items()})
            pf_txt = f"{pf['mean']:.3f} ± {pf['sd']:.3f} ({pf['n_folds']} folds)"
        summary_rows.append(row)
        md.append(
            f"| {s} | {ci.auroc:.3f} [{ci.lower:.3f}, {ci.upper:.3f}] | {pf_txt} | {row['raw_pooled_auroc']:.3f} |"
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "scores_summary.csv", index=False)

    delong_rows = []
    if len(scores) > 1:
        md += [
            "",
            "## DeLong comparisons (pooled, standardised scores)",
            "",
            "| score 1 | score 2 | ΔAUROC | z | p |",
            "|---|---|---|---|---|",
        ]
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                d = delong_test(y, std_scores[scores[i]], std_scores[scores[j]])
                delong_rows.append(
                    {
                        "score_1": scores[i],
                        "score_2": scores[j],
                        "auroc_1": d.auroc_1,
                        "auroc_2": d.auroc_2,
                        "delta": d.delta,
                        "z": d.z,
                        "p_value": d.p_value,
                    }
                )
                md.append(
                    f"| {scores[i]} | {scores[j]} | {d.delta:+.3f} | {d.z:.2f} | {d.p_value:.3g} |"
                )
        pd.DataFrame(delong_rows).to_csv(out_dir / "delong.csv", index=False)

    strata_rows = []
    for col in [str(c) for c in (cfg.get("strata") or [])]:
        md += [
            "",
            f"## AUROC by {col} (first score: {scores[0]})",
            "",
            "| level | n | positives | AUROC [95% CI] |",
            "|---|---|---|---|",
        ]
        for level, r in stratified_auroc(
            y, std_scores[scores[0]], df[col].to_numpy(), n_boot=n_boot, seed=seed
        ).items():
            strata_rows.append({"stratum": col, "level": level, **r})
            md.append(
                f"| {level} | {r['n']} | {r['n_pos']} | {r['auroc']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}] |"
            )
    if strata_rows:
        pd.DataFrame(strata_rows).to_csv(out_dir / "strata.csv", index=False)

    op_rows = []
    rules = [float(q) for q in cfg.operating_rules]
    fold_key = folds if folds is not None else np.zeros(len(y), dtype=int)
    for s in scores:
        sc = std_scores[s]
        md += [
            "",
            f"## Operating points: {s}",
            "",
            "| rule | flagged | sensitivity | specificity | PPV | NPV | LR+ | LR− | TP/FP/FN/TN |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        points = [
            (f"top {100 * q:.0f} % within fold", flag_top_within_fold(sc, fold_key, q), False)
            for q in rules
        ]
        thr = threshold_points(sc, y)
        for name, key in (
            ("Youden (all data, optimistic)", "youden"),
            ("rule-out: sens ≥ 90 % (optimistic)", "rule_out"),
            ("rule-in: spec ≥ 90 % (optimistic)", "rule_in"),
        ):
            t = thr[key]
            if t == t:
                points.append((name, (sc >= t).astype(int), True))
        for name, pred, optimistic in points:
            m = operating_point(pred, y)
            ci = operating_point_bootstrap(pred, y, n_boot=n_boot, seed=seed)
            lr = likelihood_ratios(m["sensitivity"], m["specificity"])
            op_rows.append(
                {
                    "score": s,
                    "rule": name,
                    "optimistic": optimistic,
                    **m,
                    **{f"{k}_lower": v[0] for k, v in ci.items()},
                    **{f"{k}_upper": v[1] for k, v in ci.items()},
                    **lr,
                }
            )
            md.append(
                f"| {name} | {100 * m['flagged']:.0f} % | {100 * m['sensitivity']:.0f} % [{100 * ci['sensitivity'][0]:.0f}, {100 * ci['sensitivity'][1]:.0f}] | "
                f"{100 * m['specificity']:.0f} % [{100 * ci['specificity'][0]:.0f}, {100 * ci['specificity'][1]:.0f}] | "
                f"{100 * m['ppv']:.0f} % [{100 * ci['ppv'][0]:.0f}, {100 * ci['ppv'][1]:.0f}] | {100 * m['npv']:.0f} % [{100 * ci['npv'][0]:.0f}, {100 * ci['npv'][1]:.0f}] | "
                f"{lr['lr_plus']:.2f} | {lr['lr_minus']:.2f} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} |"
            )
    pd.DataFrame(op_rows).to_csv(out_dir / "operating_points.csv", index=False)

    prev_rows = []
    if rules:
        s = scores[0]
        q = rules[len(rules) // 2]
        m = operating_point(flag_top_within_fold(std_scores[s], fold_key, q), y)
        md += [
            "",
            f"## PPV / NPV at other prevalences ({s}, top {100 * q:.0f} % rule: sensitivity {100 * m['sensitivity']:.0f} %, specificity {100 * m['specificity']:.0f} %)",
            "",
            "| prevalence | PPV | NPV |",
            "|---|---|---|",
        ]
        for p in [float(v) for v in cfg.prevalences]:
            r = ppv_npv_at_prevalence(m["sensitivity"], m["specificity"], p)
            prev_rows.append({"score": s, "rule_q": q, **r})
            md.append(f"| {100 * p:.0f} % | {100 * r['ppv']:.0f} % | {100 * r['npv']:.0f} % |")
        pd.DataFrame(prev_rows).to_csv(out_dir / "ppv_npv_by_prevalence.csv", index=False)

    report = out_dir / str(cfg.report_filename)
    report.write_text("\n".join(md) + "\n")
    summary = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "scores": summary_rows,
        "delong": delong_rows,
        "report": str(report),
    }
    (out_dir / "scores_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log.info("\n".join(md[:12]))
    return summary


@hydra.main(version_base="1.3", config_path="../configs", config_name="scores.yaml")
def main(cfg: DictConfig) -> None:
    run_scores(cfg)


if __name__ == "__main__":
    main()
