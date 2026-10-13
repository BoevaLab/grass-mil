# Evaluation Statistics (`grass-mil-scores`)

Held-out scores of a cross-validation come from different fold models with
different score scales. `grass-mil-scores` standardises them within fold before
pooling, reports both the per-fold mean and the pooled AUROC with bootstrap
intervals, compares scores with the DeLong test and tabulates operating points
at pre-specified rules.

```bash
grass-mil-scores input=runs/nested_ftall_reg/oof_predictions.csv output_dir=runs/scores
grass-mil-scores input=patients.csv scores=[grass,clinical] strata=[subtype] patient_column=null
```

Input: a CSV with one row per unit, a `label_column` (default `label`), one or
more score columns (`scores`, default `[prob_class_1]`), optionally a fold column
(`fold_column`, default `outer`), a patient column (`patient_column`, default
`patient_id`; with `patient_level: true` the scores are averaged per patient
first) and strata columns.

| output | content |
|---|---|
| `scores_summary.csv` | per score: pooled AUROC on within-fold standardised scores with a percentile bootstrap CI (`n_boot`, `seed`), per-fold AUROC mean and sd, raw pooled AUROC |
| `delong.csv` | every pair of scores: delta AUROC, z, p (Sun & Xu structural components; same labels) |
| `strata.csv` | AUROC with CI within each level of each stratum (first score) |
| `operating_points.csv` | per score and rule: sensitivity, specificity, PPV, NPV with bootstrap CIs, LR+, LR-, confusion counts; rules = flag the top `q` within each fold (`operating_rules`) plus Youden / rule-in / rule-out thresholds chosen on all data (flagged optimistic) |
| `ppv_npv_by_prevalence.csv` | PPV and NPV of the middle rule at other prevalences |
| `report_scores.md` | all of the above as tables |

`standardise=z|rank|none` selects the within-fold transform. The functions are
importable from `grass_mil.evaluation` (`auroc`, `bootstrap_auroc_ci`,
`delong_test`, `z_within_fold`, `rank_within_fold`, `per_fold_auroc`,
`stratified_auroc`, `flag_top_within_fold`, `operating_point`,
`operating_point_bootstrap`, `likelihood_ratios`, `ppv_npv_at_prevalence`,
`threshold_points`).
