# Inference Utility Suite

This document describes the artifact-oriented inference workflow implemented in
`src/grass_mil/inference/predict.py`.

## Purpose

The suite extends checkpoint evaluation with:

- optionally deterministic prediction export
- task-aware metrics for categorical/regression/survival tasks
- graph embedding export for downstream interpretability workflows
- optional node embedding export with bag-id mapping
- single-pass collection for predictions + embeddings (no second dataloader pass)

## CLI Entry Point

```bash
grass-mil-predict \
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
- `configs/inference/interpretability/default.yaml`

## Aggregation Subsampling Semantics

`aggregation.subsample_fraction` supports efficient inference-time subsampling:

- when `< 1.0`, `src/grass_mil/inference/predict.py` injects predict-time sampler runtime
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

### `instance_table.csv` (optional)

Enabled via `interpretability.enabled=true`.

Columns include:

- `instance_id`, `bag_id`, `patch_id`, `region_id`, `sample_id`
- `score`, `logit_*`, `attention`
- `inst_emb_*`
- `comp_*` (when composition metadata is available)
- `center_x`, `center_y` (when centroid metadata is available)

Notes:

- `attention` is normalized within each bag.
- when attention logits are unavailable, `attention` defaults to uniform within bag.
- `score` export is configurable via `interpretability.score_mode` and
  `interpretability.score_logit_index`:
  - `score_mode=sigmoid` (default): `sigmoid(logit[index])`
  - `score_mode=identity`: raw `logit[index]` (for hazard/log-risk style outputs)
  - `score_mode=softmax`: class probability `softmax(logits)[index]`
- by default, missing composition metadata raises (`interpretability.require_composition=true`).

### `spatial_table.csv` (optional)

Enabled via `interpretability.spatial.enabled=true`.

Columns:

- `source_id`
- `target_id`
- `distance`

Edges are taken from original subgraph/root connectivity by reading `instance_graphs`
stored alongside prediction payloads.
No kNN graph is reconstructed during interpretability export.

Unit contract:

- Preprocessing converts coordinates/distances to micrometers using `data.coord_scale_um`.
- `data.coord_scale_um=1.0` means input coordinates are already in micrometers.
- Exported `distance` values are consumed as micrometers by interpretability tier-2 analyses.
- If `data.coord_scale_um` changes, regenerate processed artifacts before running prediction/export.

## Legacy Parity Mapping

High-value legacy utilities from `working_version` are mapped as:

- prediction collectors (`collect_predict_for_all_nodes*`) -> unified
  `collect_inference_payload` in `src/grass_mil/inference/collectors.py`
- full-graph aggregation (`full_graph_*`) -> `src/grass_mil/inference/aggregation.py`
- evaluation metrics (`graph_*_evaluate_fn`) -> `src/grass_mil/inference/metrics.py`
- embedding collectors -> `src/grass_mil/inference/collectors.py` + `src/grass_mil/inference/embeddings.py`
- interpretability table bridge -> `src/grass_mil/inference/interpretability_export.py`
