# Inference Utility Suite

This document describes the artifact-oriented inference workflow implemented in
`src/inference/predict.py`.

## Purpose

The suite extends checkpoint evaluation with:

- optionally deterministic prediction export
- task-aware metrics for categorical/regression/survival tasks
- graph embedding export for downstream interpretability workflows
- optional node embedding export with bag-id mapping
- single-pass collection for predictions + embeddings (no second dataloader pass)

## CLI Entry Point

```bash
python src/inference/predict.py \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  task=finetune_mil \
  model=supervised_module \
  data=spatial_omics
```

Base config:

- `configs/inference/predict.yaml`

Sub-config groups:

- `configs/inference/metrics/default.yaml`
- `configs/inference/aggregation/default.yaml`
- `configs/inference/embeddings/default.yaml`

## Aggregation Subsampling Semantics

`aggregation.subsample_fraction` supports efficient inference-time subsampling:

- when `< 1.0`, `src/inference/predict.py` injects predict-time sampler runtime
  subsampling so fewer roots/subgraphs are forwarded through the model
- sampling is uniform at random over candidate roots, with optional reproducibility
  via `aggregation.subsample_seed`
- this path applies to aggregation modes `mean`, `max`, and `attention_weighted`
  while avoiding post-aggregation double subsampling
- if subsampling is requested but cannot be effectively applied preforward
  (for example, `identity` sampler or `runtime.enabled=false`), inference fails
  fast with a remediation error
- `aggregation.subsample_fraction=1.0` disables subsampling and keeps full usage
- `aggregation.subsample_fraction=0.0` is invalid

Inference summary metadata includes:

- `aggregation.preforward_subsampling_applied` (backward-compatible bool)
- `aggregation.preforward_subsampling` (structured decision payload)

## Output Schema

Outputs are written under:

- `${paths.output_dir}/${predict.output_subdir}`

### `predictions.csv`

Columns:

- `bag_id`
- `logit_*`
- optional `target_*`
- optional `attention` (JSON-encoded list)

Rows are sorted by `bag_id` for deterministic export.

### `metrics.json`

Structure:

- `global`: overall metrics
- `per_group`: per-bag metrics keyed by `bag_id` (when `metrics.per_group=true`)

Task metric sets:

- binary/categorical: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- regression: `r2`, `mae`, `rmse`
- survival: `c_index`

### `graph_embeddings.csv`

Columns:

- `bag_id`
- `graph_emb_*`

### `node_embeddings.csv` (optional)

Enabled via `embeddings.extract_node=true`.

Columns:

- `bag_id`
- `node_emb_*`

### `inference_summary.json`

Quick run summary with artifact paths and row count.

## Legacy Parity Mapping

High-value legacy utilities from `working_version` are mapped as:

- prediction collectors (`collect_predict_for_all_nodes*`) -> unified
  `collect_inference_payload` in `src/inference/collectors.py`
- full-graph aggregation (`full_graph_*`) -> `src/inference/aggregation.py`
- evaluation metrics (`graph_*_evaluate_fn`) -> `src/inference/metrics.py`
- embedding collectors -> `src/inference/collectors.py` + `src/inference/embeddings.py`
