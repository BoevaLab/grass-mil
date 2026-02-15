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

- Data pipeline: `src/data/` + `configs/data/`
- Model/runtime pipeline: `src/models/` + `configs/model/` + `configs/task/`
- Entrypoints: `src/train.py`, `src/eval.py`
- Configuration system: `configs/` (Hydra groups)

## Start Here

### 1) Install

```bash
cd /Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set project root for Hydra path resolution:

```bash
export PROJECT_ROOT=/Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil
```

### 2) Prepare New Data

Create a raw manifest at `data/raw/manifest.csv` (or override `data.raw_manifest_path`) with at least:

- `sample_id`
- `input_path`
- `input_type`

Optional:

- `region_id`
- `polygons_path`

### 3) Run Default Supervised Pipeline

```bash
python src/train.py task=finetune_mean model=supervised_module data=spatial_omics
```

### 4) Evaluate A Checkpoint

```bash
python src/eval.py \
  ckpt_path=/absolute/path/to/checkpoint.ckpt \
  task=finetune_mean \
  model=supervised_module \
  data=spatial_omics
```

## Documentation Index

- Full tracked-file map: [`docs/repository_map.md`](docs/repository_map.md)
- End-to-end workflows (new data, SSL pretrain->finetune, model adjustments, inference): [`docs/workflows_train_infer.md`](docs/workflows_train_infer.md)
- Exhaustive hyperparameter reference for repo configs: [`docs/hyperparameter_reference.md`](docs/hyperparameter_reference.md)
- Capability modes and operational options: [`docs/capabilities_and_modes.md`](docs/capabilities_and_modes.md)
- Existing model contracts: [`docs/model_component_contracts.md`](docs/model_component_contracts.md)
- Existing legacy gap audit: [`docs/legacy_gap_audit.md`](docs/legacy_gap_audit.md)

## Supported Training/Inference Modes

| Mode | Task Config | Model Config | Entrypoint | Primary Output |
|---|---|---|---|---|
| Supervised mean aggregation | `task=finetune_mean` | `model=supervised_module` | `src/train.py` | Region/sample-level logits via mean bag aggregation |
| Supervised MIL attention | `task=finetune_mil` | `model=supervised_module` | `src/train.py` | Attention-weighted bag logits (+ optional MIL auxiliary terms) |
| SSL pretraining (BGRL) | `task=pretrain_bgrl` | `model=bgrl_module` | `src/train.py` | BGRL-pretrained encoder checkpoint |
| Checkpoint evaluation | any compatible task/model | matching training stack | `src/eval.py` | Test metrics from selected checkpoint |
| Advanced prediction (Python API) | compatible with `SupervisedModule.predict_step` | `model=supervised_module` | `Trainer.predict(...)` | `bag_ids`, `bag_logits`, optional `bag_targets`, optional `bag_attention` |

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
python src/train.py task=finetune_mean model=supervised_module
python src/train.py task=finetune_mil model=supervised_module
```

### Run SSL pretraining

```bash
python src/train.py task=pretrain_bgrl model=bgrl_module
```

### Change backbone

```bash
python src/train.py task=finetune_mil model=supervised_module \
  model.encoder.conv_type=gine \
  model.encoder.use_edge_attr=true \
  model.encoder.edge_attr_dim=2
```

### Force data re-precompute

```bash
python src/train.py data.force_precompute=true
```

### Resume training

```bash
python src/train.py ckpt_path=/absolute/path/to/last.ckpt
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

1. A standalone `predict.py` CLI entrypoint for inference-only export.
2. Curated first-class hyperparameter sweep recipes in `configs/hparams_search/`.
3. Turnkey dataset-specific benchmark packs beyond the generic Hydra/config contracts.

All currently supported capabilities are documented in [`docs/workflows_train_infer.md`](docs/workflows_train_infer.md), [`docs/hyperparameter_reference.md`](docs/hyperparameter_reference.md), and [`docs/capabilities_and_modes.md`](docs/capabilities_and_modes.md).
