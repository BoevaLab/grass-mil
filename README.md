# grass-mil

A geometric deep learning framework for spatial omics data.

This repository contains:
- a configurable preprocessing pipeline from raw spatial omics tables to PyG graphs
- graph encoders (GIN/GCN/GAT/GraphSAGE/GINE)
- supervised training for mean pooling and MIL attention
- self-supervised pretraining with BGRL
- Hydra-based experiment configuration and Lightning training loops

## Implemented

- Data ingestion: CSV/TSV, h5ad, SCE/RDS
- Graph construction: Delaunay-based builder and edge features
- Feature reducers: identity and PCA
- Patching/sampling: tiling, polygon patching, identity and ShaDow samplers
- Tasks:
  - `pretrain_bgrl`
  - `finetune_mean`
  - `finetune_mil`
- Training runtime:
  - checkpoint initialization/remapping
  - optimizer + optional warmup scheduler wrapping
  - optional region accumulation for MIL

## Quick Start

```bash
cd grass-mil
pip install -r requirements.txt
```

Train:

```bash
python src/train.py task=finetune_mean model=supervised_module data=spatial_omics
```

Evaluate:

```bash
python src/eval.py ckpt_path=/absolute/path/to/checkpoint.ckpt task=finetune_mean model=supervised_module data=spatial_omics
```

Pretrain BGRL:

```bash
python src/train.py task=pretrain_bgrl model=bgrl_module data=spatial_omics
```

## Config Entry Points

- Main train config: `configs/train.yaml`
- Main eval config: `configs/eval.yaml`
- Data config: `configs/data/spatial_omics.yaml`
- Task configs: `configs/task/`
- Model configs: `configs/model/`

## Project Layout

- `src/data/`: preprocessing, datasets, samplers, datamodule
- `src/models/`: modules, components, training helpers
- `src/utils/`: logging, Hydra/Lightning utilities
- `configs/`: Hydra config groups
- `tests/`: unit and integration tests

## Known Gaps (Not Implemented Yet)

The following were present as template concepts but are not fully implemented as first-class project workflows yet:
- End-to-end hyperparameter search workflow presets (for example, curated Optuna search spaces and task-specific sweep recipes).
