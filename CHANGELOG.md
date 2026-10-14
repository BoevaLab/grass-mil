# Changelog

## 1.0.0 (2026-10)

Public release accompanying the preprint
[GRASS-MIL: graph-based representation and discovery of phenotype-associated spatial structures with multiple instance learning](https://www.biorxiv.org/content/10.64898/2026.10.08.757722v1).

### Method (defaults match the preprint; every setting stays configurable)

- Delaunay cell graph with the bounded log edge length `clip((ln d - 2)/3, 0, 1)` as the edge feature (`log_distance`).
- GINE ego-graph encoder (4 x 256, learnable epsilon, layer norm, max pooling, xavier-initialised cell-type embedding) as `model=grass_mil`.
- Projected gated attention with one channel per class (32 / 8, kaiming) and logit-space pooling, so every bag logit is an exact sum of per-cell contributions.
- One region = one bag: all sampled ego-graphs of a region in one batch, one attention softmax, one loss; region-level validation metrics and selection.
- Training regimes as presets (`regime=mil|mil_frozen|linear_probehead|linear_ftall|linear_ftall_reg`) with per-component learning-rate scales, head pre-fit by a logistic probe, class weights from the training split, per-step cosine schedule, gradient accumulation over regions, AMP-safe scheduler stepping, elementwise feature drop with an unassigned token.
- BGRL "A3" pretraining preset (`regime=pretrain_a3`): roots drawn from the union of all regions, momentum 0.99 -> 1.0, DropImportance + feature noise, checkpoint every 5000 steps.
- Import of legacy SPACE-GM style BGRL checkpoints (`model.encoder_init_map=spacegm_bgrl`).

### Analyses

- `grass-mil-probe`, `grass-mil-nested-cv`: nested patient-grouped cross-validation with explicit fold files, inner-fold epoch selection, refit and out-of-fold predictions (region and patient level).
- `grass-mil-extract`: per-cell cache (embeddings, instance logits, attention) of trained models over every cell.
- `grass-mil-niches`: niche discovery on a model's cells (cosine HDBSCAN by default, any registry backend), attention-weighted margins, per-niche tables, co-localisation, figures.
- `grass-mil-metaclusters`: cross-fold niche replication (constrained matching on composition), whole-dataset transfer, shared label-free basis.
- `grass-mil-scores`: bootstrap AUROC intervals, DeLong tests, within-fold standardisation, operating points (sensitivity, specificity, PPV, NPV, likelihood ratios).
- `grass-mil-spatial-stats`: neighbourhood enrichment, cellular neighbourhoods, contact fractions and cross-K per region with patient-level group tests.
- Patient-level aggregation in `grass-mil-predict` (mean of region probabilities or one merged bag).

### Housekeeping

- Removed stale component configs, the dummy dataset, the embeddings shim and internal audit notes; added `CITATION.cff`; `hydra-core` 1.3.5+.
