# Training And Inference Workflows

This guide provides flow-through instructions for all major `grass-mil` workflows.

Interpretability workflows are documented in:

- `docs/interpretability_suite.md`
- `docs/interpretability_extension_guide.md`

## 0) Environment And Dependency Prerequisites

From `/Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PROJECT_ROOT=/Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil
```

Dependency notes by data type:

- `csv` / `tsv`: requires `pandas`, `numpy`.
- `h5ad`: requires `anndata`.
- `sce` / `rds`: requires `rpy2` plus an R runtime and Bioconductor-compatible SCE objects.
- Polygon masking/patching: requires `shapely`.
- Graph building (`delaunay`): requires `scipy`.
- Graph runtime/samplers: requires `torch-geometric`; `shadow_custom` transform path also needs `torch-sparse`.

## 1) Running A Default Pipeline With New Data

### Step 1: Prepare The Manifest

Default manifest path is `data/raw/manifest.csv` and default column names come from `configs/data/spatial_omics.yaml`.

Required columns:

- `sample_id`
- `input_path`
- `input_type` (`csv`, `tsv`, `h5ad`, `sce`, `rds`)

Optional columns:

- `region_id` (used as default bag key for supervised training)
- `polygons_path` (polygon file for filtering/patch segmentation)

Example:

```csv
sample_id,input_path,input_type,region_id,polygons_path
s1,data/raw/s1.csv,csv,r1,
s2,data/raw/s2.h5ad,h5ad,r2,data/raw/s2_polygons.geojson
```

Path resolution behavior:

- Absolute paths are used as-is.
- Relative paths are resolved relative to the manifest directory.

### Step 2: Configure Input-Specific Data Parsing

All options below live under `data.*` in Hydra.

Spatial unit expectation:

- Normalize spatial coordinates to micrometers during preprocessing.
- `data.coord_scale_um` is the preprocessing conversion factor to micrometers.
- `data.coord_scale_um=1.0` means input coordinates are already in micrometers.
- This ensures downstream `spatial_table.distance` values are in micrometers and compatible with interpretability defaults (including filtration thresholds).
- If `data.coord_scale_um` changes, regenerate processed artifacts (`data.force_precompute=true` or use a new `data.processed_dir`).

#### CSV/TSV inputs

Key options (`data.csv.*`):

- `sep` (default `,`)
- `coord_columns` (default `['x','y']`)
- `cell_id_column`
- `categorical_label_columns`
- `molecular_columns` (`null` means infer by excluding coords/cell id/categorical labels)

Example override:

```bash
grass-mil-train \
  data.csv.coord_columns="[x_um,y_um]" \
  data.csv.cell_id_column=cell_id \
  data.csv.categorical_label_columns="[cell_type]"
```

#### H5AD inputs

Key options (`data.h5ad.*`):

- `coord_source`: `obsm` or `obs`
- `coord_key` (for `obsm`, default `spatial`)
- `coord_columns` (for `obs`)
- `cell_id_column`
- `categorical_label_columns`
- `molecular_layer`
- `molecular_features`

#### SCE/RDS inputs

Key options (`data.sce.*`):

- `assay_name`
- `coord_source`: `colData` or `reducedDims`
- `coord_key` (required for `reducedDims`)
- `coord_columns` (required for `colData`)
- `cell_id_column`
- `categorical_label_columns`
- `transpose_assay`
- `molecular_features`

### Step 3: (Optional) Graph Labels And Polygon Files

#### Polygon files

Supported polygon formats:

- `.pkl` / `.pickle`
- `.wkt` / `.txt` (line-delimited WKT)
- `.json` / `.geojson`

`polygons_path` behavior:

- Coordinates are first filtered by polygon union.
- Patch extraction uses polygons when `sample_unit != full`; otherwise tiling/full logic applies.

#### Graph label file

Configure `data.graph_labels.*`:

- `label_file`: CSV path with labels
- `id_column`: column used as lookup key
- `tasks`: task columns (or all columns if `null`)
- `scope`: `region` | `patch` | `sample`

When enabled, labels are attached as `graph_y` on each graph unit.

### Step 4: Precompute And Split Control

Precompute is triggered from `SpatialOmicsDataModule.prepare_data()` during training/eval startup.

Primary controls:

- `data.force_precompute`: force rebuild even if index exists
- `data.sample_unit`: `tile` or `full`
- `data.tiling.*`: tile geometry and tile min-cell threshold
- `data.min_cells`: global patch retention threshold
- `data.split.train_val_test_split`: must sum to `1.0`
- `data.split.split_by`: `sample`, `region`, or `patch`
- `data.split.loocv.*`: optional fold-wise holdout overrides
- `data.reducer_scope`: `sample` or `dataset`
- `data.feature_reducer.fit_mode`: `train_only` or `global` (only relevant for dataset scope)

Important split persistence behavior:

- Train/val/test labels are assigned during precompute and persisted in
  `processed_index.json`.
- The datamodule consumes persisted split labels and does not recompute them at
  runtime.
- If you change any `data.split.*` value, you must regenerate processed
  artifacts (`data.force_precompute=true` or a new `data.processed_dir`),
  otherwise old persisted splits are reused.

Typical fresh-data command:

```bash
grass-mil-train \
  task=finetune_mean \
  model=supervised_module \
  data=spatial_omics \
  data.raw_manifest_path=/absolute/path/to/manifest.csv \
  data.processed_dir=/absolute/path/to/processed/spatial_omics \
  data.force_precompute=true
```

### Step 4b: Run LOOCV (Single Fold Or Full Loop)

`src/grass_mil/loocv.py` reuses the train stack and runs fold-specific jobs by overriding
`data.split.loocv.*`.

Single-fold by explicit fold id:

```bash
grass-mil-loocv \
  loocv.mode=single \
  data.split.loocv.enabled=true \
  data.split.loocv.fold_unit=region \
  data.split.loocv.holdout_id="sampleA::region3" \
  data.split.loocv.validation_strategy=heldout_fold_items \
  data.split.loocv.val_ratio=0.2
```

Single-fold by fold index (useful for array jobs):

```bash
grass-mil-loocv \
  loocv.mode=single \
  loocv.fold_index=0 \
  data.split.loocv.enabled=true \
  data.split.loocv.fold_unit=sample
```

If both `data.split.loocv.holdout_id` and `loocv.fold_index` are set in single
mode, `loocv.fold_index` takes priority and a warning is emitted.

Full loop across all discovered folds:

```bash
grass-mil-loocv \
  loocv.mode=all \
  data.split.loocv.enabled=true \
  data.split.loocv.fold_unit=region \
  data.split.loocv.validation_strategy=patches_from_train_items \
  data.split.loocv.val_ratio=0.15
```

Validation strategy options:

- `heldout_fold_items`: validation is held out from remaining fold items.
- `patches_from_train_items`: validation patches are sampled from training fold items (weaker independence, useful in low-data settings).

Safety note:

- For `data.reducer_scope=dataset` and `data.feature_reducer.fit_mode=train_only`, LOOCV requires fold-specific precompute (`data.force_precompute=true` or per-fold `processed_dir`).
- To prevent stale split reuse across folds, at least one of `loocv.force_precompute_per_fold` or `loocv.per_fold_processed_dir` must be `true`.

### Step 5: Run Default Training And Evaluation

Default training (mean aggregation):

```bash
grass-mil-train task=finetune_mean model=supervised_module data=spatial_omics
```

Evaluation of a selected checkpoint:

```bash
grass-mil-eval \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  task=finetune_mean \
  model=supervised_module \
  data=spatial_omics
```

### Step 6: Inspect Output Artifacts

Data artifacts (`data.processed_dir`):

- `processed_index.json`: graph entry index, label maps, reducer state, split metadata
- `metadata.json`: summary of feature dimensions, labels, and split/reducer metadata
- `graphs/<sample_id>/*.pt`: serialized PyG `Data` units

Training artifacts (`paths.output_dir`):

- `train.log` (Hydra job logging)
- `checkpoints/last.ckpt`
- `checkpoints/epoch_*.ckpt` (if checkpoint callback enabled)

## 2) Pretraining With SSL, Then Using It In Fine-Tuning

### Step 1: Run SSL Pretraining (BGRL)

```bash
grass-mil-train task=pretrain_bgrl model=bgrl_module data=spatial_omics
```

Default pretraining task includes:

- `scheduler=cosine_step` override
- BGRL augment settings (`drop_edge_*`, `drop_feat_*`)
- Momentum schedule (`momentum`, `momentum_min`, `warmup_steps`, `total_steps`)

### Step 2: Choose A Checkpoint

Use either:

- `checkpoints/last.ckpt` for latest state
- a specific `epoch_*.ckpt`

If your callback tracks best metric, use best checkpoint path from training logs.

### Step 3: Fine-Tune Using The Pretrained Encoder

```bash
grass-mil-train \
  task=finetune_mil \
  model=supervised_module \
  data=spatial_omics \
  model.init_from_ckpt=/absolute/path/to/pretrain.ckpt \
  model.encoder_init_map=auto_bgrl_or_identity \
  model.init_strict=false
```

Optional encoder freezing:

```bash
grass-mil-train \
  task=finetune_mil \
  model=supervised_module \
  model.init_from_ckpt=/absolute/path/to/pretrain.ckpt \
  model.freeze_encoder=true
```

### Checkpoint Mapping Behavior And Failure Cases

Mapping behavior (`encoder_init_map`):

- `auto_bgrl_or_identity`: remaps known BGRL encoder prefixes such as `online_encoder.*` and `ssl_model.online_encoder.*` to `encoder.*`
- `identity`: no key remapping

Common failures:

1. `FileNotFoundError` if `model.init_from_ckpt` path is invalid.
2. Missing/unexpected key mismatch with `model.init_strict=true`.
3. Shape mismatch when encoder dimensions differ between pretrain and fine-tune configs.
4. Invalid `encoder_init_map` value (must be `identity` or `auto_bgrl_or_identity`).

## 3) Adjusting Model Settings (Aggregation, Backbone, Loss, MIL Options)

### A) Mean vs MIL Aggregation

Mean aggregation:

```bash
grass-mil-train task=finetune_mean model=supervised_module
```

MIL attention aggregation:

```bash
grass-mil-train task=finetune_mil model=supervised_module
```

Notes:

- `task=finetune_mil` activates attention path and enables MIL-only options.
- `task=finetune_mean` ignores MIL-only controls.

### B) Backbone And Encoder Adjustments

Switch backbone type:

```bash
grass-mil-train model.encoder.conv_type=gin
grass-mil-train model.encoder.conv_type=gcn
grass-mil-train model.encoder.conv_type=gat
grass-mil-train model.encoder.conv_type=graphsage
grass-mil-train model.encoder.conv_type=gine
```

Important constraints:

1. `conv_type=gine` requires `model.encoder.edge_attr_dim` and runtime `edge_attr`.
2. `conv_type=gcn` can use scalar edge weights from `edge_attr` when `model.encoder.use_edge_attr=true`; choose feature by `model.encoder.edge_weight_index`.
3. `conv_type=gat` requires `hidden_dim % gat_heads == 0`.
4. `use_edge_attr=true` on unsupported backbones is ignored with warning.

### C) Pooling, JK, And Core Encoder Hyperparameters

Common overrides:

```bash
grass-mil-train \
  model.encoder.hidden_dim=256 \
  model.encoder.out_dim=256 \
  model.encoder.num_layers=4 \
  model.encoder.dropout=0.2 \
  model.encoder.norm=layernorm \
  model.encoder.jk=concat \
  model.encoder.pooling=set2set \
  model.encoder.set2set_steps=4
```

### D) Attention Variants

```bash
grass-mil-train task=finetune_mil \
  model.attention.attention_type=gated

grass-mil-train task=finetune_mil \
  model.attention.attention_type=gated_projected \
  model.attention.projection_dim=128
```

### E) MIL-Only Optional Terms

Region accumulation (manual optimizer stepping over region hyperbatches):

```bash
grass-mil-train task=finetune_mil \
  task.region_accumulation.enabled=true \
  task.region_accumulation.hyperbatch_size=8
```

Node auxiliary loss:

```bash
grass-mil-train task=finetune_mil \
  task.node_aux.enabled=true \
  task.node_aux.target_mode=attention_shaped_ti \
  task.node_aux.loss_mode=weighted_bce \
  task.node_aux.weight=0.2
```

Entropy regularization:

```bash
grass-mil-train task=finetune_mil \
  task.entropy_reg.enabled=true \
  task.entropy_reg.mode=attention \
  task.entropy_reg.weight=0.01
```

Dual LR groups for MIL backbone/attention:

```bash
grass-mil-train task=finetune_mil \
  task.optimization.backbone_lr=1e-3 \
  task.optimization.attention_lr=4e-3
```

### F) Loss And Target-Type Switching

Binary classification (default):

```bash
grass-mil-train task.target_type=binary task.loss=categorical_bce
```

Regression:

```bash
grass-mil-train task.target_type=regression task.loss=regression_mse
grass-mil-train task.target_type=regression task.loss=regression_huber
```

Survival:

```bash
grass-mil-train task.target_type=survival task.loss=survival_coxsgd
```

Target column selection:

```bash
grass-mil-train task.target_columns="[outcome]"
```

Target format requirements:

- Binary/regression: `graph_y` shape compatible with output logits.
- Survival: requires at least two columns in `graph_y` interpreted as `[time, event]`.

## 4) Inference Approaches

### A) Supported CLI Inference Path (Checkpoint Evaluation)

`src/grass_mil/eval.py` is the official CLI inference/evaluation entrypoint.

```bash
grass-mil-eval \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  task=finetune_mil \
  model=supervised_module \
  data=spatial_omics
```

This runs `trainer.test(...)` and logs test metrics.

### B) Inference Utility Suite (Prediction + Metrics + Embeddings)

Use `src/grass_mil/inference/predict.py` for artifact-centric inference runs:

```bash
grass-mil-predict \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  task=finetune_mil \
  model=supervised_module \
  data=spatial_omics \
  aggregation.mode=mean \
  embeddings.extract_node=false
```

Internally, this utility performs a single `trainer.predict(...)` pass and
collects prediction + embedding payloads together.

By default this writes:

- prediction table: `${paths.output_dir}/${predict.output_subdir}/${output.predictions_filename}`
- metrics report: `${paths.output_dir}/${predict.output_subdir}/${output.metrics_filename}`
- graph embeddings: `${paths.output_dir}/${predict.output_subdir}/${embeddings.filename}`
- summary payload: `${paths.output_dir}/${predict.output_subdir}/${output.summary_filename}`

### C) Advanced Predict Path (Python API)

`SupervisedModule.predict_step(...)` still supports direct Python-level prediction output.

```python
import hydra
from pathlib import Path
from omegaconf import open_dict

# Compose exactly as train/eval does.
project_root = Path("/Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil")
with hydra.initialize_config_dir(
    config_dir=str(project_root / "configs"),
    version_base="1.3",
):
    cfg = hydra.compose(
        config_name="eval.yaml",
        overrides=[
            "task=finetune_mil",
            "model=supervised_module",
            "data=spatial_omics",
        ],
    )

# Ensure path roots are concrete for interpolation.
with open_dict(cfg):
    cfg.paths.root_dir = str(project_root)

datamodule = hydra.utils.instantiate(cfg.data)
model = hydra.utils.instantiate(cfg.model)
trainer = hydra.utils.instantiate(cfg.trainer)

preds = trainer.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path="/absolute/path/to/checkpoint.ckpt",
)

# Each predict output dict can contain:
# - bag_ids
# - bag_logits
# - bag_targets (if present in batch)
# - bag_attention (MIL mode)
# - row_region_ids / row_sample_ids
# - instance_* fields (when instance payload emission is enabled)
# - embedding_* fields (when embedding payload emission is enabled)
print(preds[0].keys())
```

Expected fields from `predict_step`:

- `bag_ids`
- `bag_logits`
- `row_region_ids`
- `row_sample_ids`
- optional `bag_targets` (if labels present)
- optional `bag_attention` (when MIL aggregation is active)
- optional `instance_logits`, `instance_patch_ids`, `instance_region_ids`, `instance_sample_ids`
- optional `instance_attention_logits` (MIL + instance payload enabled)
- optional `embedding_bag_ids`, `graph_embeddings`, `embedding_bag_counts`
- optional `node_embeddings`, `node_bag_ids` (when node embedding extraction enabled)

## 5) Anything Else This Repo Can Do

See the capability matrix in [`capabilities_and_modes.md`](capabilities_and_modes.md) for details. Highlights:

1. Runtime shadow sampling (`shadow_native`, `shadow_custom`) including weighted root-node sampling.
2. Multiple logger backends (CSV, TensorBoard, WandB, MLflow, Comet, Neptune, Aim).
3. Debug presets (`fdr`, `limit`, `overfit`, `profiler`) for rapid iteration.
4. Device/runtime presets (`cpu`, `gpu`, `mps`, `ddp`, `ddp_sim`).
5. Resume training from checkpoints via `ckpt_path`.
6. Notebook-based smoke tests for data loading, components, and regime behavior.
7. CI and pre-commit checks for quality and compatibility.

## 5.1) Train-vs-Validation Sampler Split

Validation can use a dedicated sampler config (`data.val_sampler`) while training
keeps `data.sampler`. This is useful when training should keep proportional root
sampling but validation should approximate tissue-level uniform sampling.

Example:

```bash
grass-mil-train \
  data.sampler.runtime.proportional_root_sampling=true \
  data.val_sampler.name=shadow_custom \
  data.val_sampler.kwargs="{}" \
  data.val_sampler.runtime.enabled=true \
  data.val_sampler.runtime.depth=2 \
  data.val_sampler.runtime.num_neighbors=8 \
  data.val_sampler.runtime.subgraph_batch_size=32 \
  data.val_sampler.runtime.proportional_root_sampling=false \
  data.val_sampler.runtime.subsample_fraction=0.5 \
  data.val_sampler.runtime.subsample_seed=7
```

When `data.val_sampler=null`, validation reuses `data.sampler`.

## 5.2) Validation Sampling And Epoch Metric Semantics

Validation behavior is split between step-level logging and epoch-end
aggregation:

- `validation_step(...)` computes region loss on each batch and logs
  `val/loss` with `on_step=true` and `on_epoch=true`.
- Validation predictions/metadata are buffered across the epoch.
- `on_validation_epoch_end(...)` rebuilds a full validation payload (across
  distributed ranks when DDP is active), then re-aggregates at
  `bag_scope='region'`.
- Aggregation mode at validation epoch-end is:
  - `attention_weighted` when `task.aggregation=mil_attention`
  - `mean` otherwise
- Validation epoch-end aggregation always uses full instance usage
  (`subsample_fraction=1.0`).
- Final task metrics are logged once per epoch from aggregated region-level
  outputs:
  - binary/categorical: `val/acc`
  - regression: `val/mae`, `val/rmse`, `val/r2`
  - survival: `val/c_index`

Practical implications:

- If you want validation sampling behavior different from training, configure
  `data.val_sampler.*` explicitly.
- If `data.val_sampler` is not set, validation inherits `data.sampler`.
- Validation metrics are region-aggregated epoch metrics; batch-level
  `val/loss` can diverge from the final epoch metric trend.

Example: keep training weighted root sampling, but validate with deterministic,
non-proportional sampling.

```bash
grass-mil-train \
  data.sampler.runtime.proportional_root_sampling=true \
  data.val_sampler.name=shadow_custom \
  data.val_sampler.kwargs="{}" \
  data.val_sampler.runtime.enabled=true \
  data.val_sampler.runtime.proportional_root_sampling=false \
  data.val_sampler.runtime.shuffle_subgraphs=false \
  data.val_sampler.runtime.subsample_fraction=1.0
```

## 6) Optional Smoke Validation Commands

Quick config and runtime checks:

```bash
pytest tests/test_configs.py -q
pytest tests/test_train_regimes.py -q
pytest tests/test_eval.py -q
```

Pre-commit quality checks:

```bash
pre-commit run -a
```

These are optional sanity checks and are not required for every training run.
