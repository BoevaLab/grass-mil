# grass-mil

Geometric deep learning framework for spatial omics data, built around a Hydra-configured pipeline from raw cell tables to graph-level training and evaluation.

## Project Purpose And Architecture

`grass-mil` provides a configurable end-to-end workflow for:

1. Loading spatial omics data (`csv`/`tsv`, `h5ad`, `SingleCellExperiment`/`rds`)
2. Building graph units (tiling/polygon patches + PyG graphs)
3. Running supervised training (mean aggregation or MIL attention)
4. Running self-supervised pretraining (BGRL)
5. Evaluating checkpoints with reproducible Hydra config composition

Core architecture:

- Data pipeline: `src/grass_mil/data/` + `configs/data/`
- Model/runtime pipeline: `src/grass_mil/models/` + `configs/model/` + `configs/task/`
- Interpretability pipeline: `src/grass_mil/interpretability/` + `configs/interpretability/`
- Entrypoints: `src/grass_mil/train.py`, `src/grass_mil/eval.py`
- Configuration system: `configs/` (Hydra groups)

## Interpretability

`grass-mil` includes a first-class interpretability suite:

- parameterized dimensionality reduction (`pca`, `umap`, `tsne`)
- parameterized clustering (`kmeans`, `agglomerative`, `hdbscan`, `spectral`, `optics`, `dbscan`)
- core (composition-based) and tier-2 (spatial) analyses via a plugin registry
- report generation to interactive Plotly HTML and static PDF (Playwright)

Quick run:

```bash
grass-mil-report \
  data.instance_table=/absolute/path/to/instance_table.csv \
  data.spatial_table=/absolute/path/to/spatial_table.csv
```

## Start Here

### 1) Install

```bash
git clone git@github.com:BoevaLab/grass-mil.git
cd grass-mil
python -m venv .venv
source .venv/bin/activate

# Editable install for development; drop -e for a plain install.
pip install -e .

# Optional extras: interpretability plots, report rendering, spatial statistics,
# and the h5ad/SingleCellExperiment readers.
pip install -e ".[interpretability,report,spatial-stats,io]"
```

Installing exposes the CLI entrypoints (`grass-mil-train`, `grass-mil-eval`,
`grass-mil-loocv`, `grass-mil-predict`, `grass-mil-report`). The Hydra config
tree ships inside the package, so it resolves from any working directory.

Hydra's `paths.root_dir` defaults to the `PROJECT_ROOT` environment variable
when set (`rootutils` exports it automatically from the `.project-root` marker
in a source checkout) and otherwise falls back to the current working
directory. Set it explicitly to place `data/` and `logs/` elsewhere:

```bash
export PROJECT_ROOT=/path/to/your/workspace
```

### 2) Prepare New Data

Create a raw manifest at `data/raw/manifest.csv` (or override `data.raw_manifest_path`) with at least:

- `sample_id`
- `input_path`
- `input_type`

Optional:

- `region_id`
- `polygons_path`

For the full onboarding checklist and dataset schema details, see
[`docs/dataset_preparation.md`](docs/dataset_preparation.md).

### 3) Run Default Supervised Pipeline

```bash
grass-mil-train task=finetune_mean model=supervised_module data=spatial_omics
```

### 4) Evaluate A Checkpoint

```bash
grass-mil-eval \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  task=finetune_mean \
  model=supervised_module \
  data=spatial_omics
```

## Documentation Index

- Full tracked-file map: [`docs/repository_map.md`](docs/repository_map.md)
- Dataset onboarding and preparation checklist: [`docs/dataset_preparation.md`](docs/dataset_preparation.md)
- End-to-end workflows (new data, SSL pretrain->finetune, model adjustments, inference): [`docs/workflows_train_infer.md`](docs/workflows_train_infer.md)
- Inference utility suite (prediction artifacts, metrics, embeddings): [`docs/inference_utility_suite.md`](docs/inference_utility_suite.md)
- Interpretability suite architecture and usage: [`docs/interpretability_suite.md`](docs/interpretability_suite.md)
- Interpretability extension guide (plugins/new analyses): [`docs/interpretability_extension_guide.md`](docs/interpretability_extension_guide.md)
- Exhaustive hyperparameter reference for repo configs: [`docs/hyperparameter_reference.md`](docs/hyperparameter_reference.md)
- Capability modes and operational options: [`docs/capabilities_and_modes.md`](docs/capabilities_and_modes.md)
- Existing model contracts: [`docs/model_component_contracts.md`](docs/model_component_contracts.md)
- Existing legacy gap audit: [`docs/legacy_gap_audit.md`](docs/legacy_gap_audit.md)

## Supported Training/Inference Modes

| Mode | Task Config | Model Config | Entrypoint | Primary Output |
|---|---|---|---|---|
| Supervised mean aggregation | `task=finetune_mean` | `model=supervised_module` | `src/grass_mil/train.py` | Region/sample-level logits via mean bag aggregation |
| Supervised MIL attention | `task=finetune_mil` | `model=supervised_module` | `src/grass_mil/train.py` | Attention-weighted bag logits (+ optional MIL auxiliary terms) |
| SSL pretraining (BGRL) | `task=pretrain_bgrl` | `model=bgrl_module` | `src/grass_mil/train.py` | BGRL-pretrained encoder checkpoint |
| Checkpoint evaluation | any compatible task/model | matching training stack | `src/grass_mil/eval.py` | Test metrics from selected checkpoint |
| Advanced prediction (Python API) | compatible with `SupervisedModule.predict_step` | `model=supervised_module` | `Trainer.predict(...)` | `bag_ids`, `bag_logits`, `row_region_ids`, `row_sample_ids`, optional `bag_targets`, optional `bag_attention`, optional `instance_*`, optional `embedding_*` |

## Required Inputs (Manifest Summary)

Default data config expects:

- Manifest path: `${paths.data_dir}/raw/manifest.csv`
- Processed output dir: `${paths.data_dir}/processed/spatial_omics`

Manifest columns (defaults from `configs/data/spatial_omics.yaml`):

| Column | Required | Meaning |
|---|---|---|
| `sample_id` | yes | Logical sample identifier |
| `input_path` | yes | Path to raw input table/file |
| `input_type` | yes | One of `csv`, `tsv`, `h5ad`, `sce`, `rds` |
| `region_id` | no | Region/group identifier for bagging/label scopes |
| `polygons_path` | no | Polygon definitions for patch extraction/filtering |

See full input contracts and file examples in [`docs/workflows_train_infer.md`](docs/workflows_train_infer.md).

## Common Command Patterns (Hydra Overrides)

### Switch aggregation regime

```bash
grass-mil-train task=finetune_mean model=supervised_module
grass-mil-train task=finetune_mil model=supervised_module
```

### Run SSL pretraining

```bash
grass-mil-train task=pretrain_bgrl model=bgrl_module
```

### Change backbone

```bash
grass-mil-train task=finetune_mil model=supervised_module \
  model.encoder.conv_type=gine \
  model.encoder.use_edge_attr=true \
  model.encoder.edge_attr_dim=2
```

### Force data re-precompute

```bash
grass-mil-train data.force_precompute=true
```

### Resume training

```bash
grass-mil-train ckpt_path=/absolute/path/to/last.ckpt
```

## Where Outputs Go

Hydra runtime outputs:

- Run logs and artifacts: `${paths.log_dir}/${task_name}/runs/<timestamp>/`
- Multiruns: `${paths.log_dir}/${task_name}/multiruns/<timestamp>/<job_num>/`
- Checkpoints (default callback): `${paths.output_dir}/checkpoints/`

Data precompute outputs:

- Graph files: `${data.processed_dir}/graphs/.../*.pt`
- Index: `${data.processed_dir}/processed_index.json`
- Metadata: `${data.processed_dir}/metadata.json`

## Known Gaps / Non-Goals

Current repository behavior intentionally excludes:

1. Curated first-class hyperparameter sweep recipes in `configs/hparams_search/`.
2. Turnkey dataset-specific benchmark packs beyond the generic Hydra/config contracts.

All currently supported capabilities are documented in [`docs/workflows_train_infer.md`](docs/workflows_train_infer.md), [`docs/hyperparameter_reference.md`](docs/hyperparameter_reference.md), and [`docs/capabilities_and_modes.md`](docs/capabilities_and_modes.md).
