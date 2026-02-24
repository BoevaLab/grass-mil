# Capabilities And Modes

This document catalogs operational modes and non-default capabilities in `grass-mil`.

## 1) Training Regimes

| Regime | Task | Model | Primary Use |
|---|---|---|---|
| Supervised mean | `finetune_mean` | `supervised_module` | Stable baseline graph-level classification/regression/survival |
| Supervised MIL | `finetune_mil` | `supervised_module` | Region-level attention aggregation + optional MIL auxiliaries |
| SSL pretrain | `pretrain_bgrl` | `bgrl_module` | Representation learning before supervised fine-tuning |

## 2) Aggregation And MIL Feature Modes

| Mode | Config | Behavior |
|---|---|---|
| Mean bag aggregation | `task.aggregation=mean` | Averages instance logits per bag |
| MIL attention aggregation | `task.aggregation=mil_attention` | Learns attention weights over instance embeddings |
| Region accumulation | `task.region_accumulation.enabled=true` | Manual optimizer steps over region hyperbatches |
| Node auxiliary loss | `task.node_aux.enabled=true` | Adds node-level auxiliary objective in binary MIL training |
| Entropy regularization | `task.entropy_reg.enabled=true` | Adds entropy-based regularization term |

## 3) Backbone/Encoder Modes

Supported `model.encoder.conv_type` values:

- `gin`
- `gcn`
- `gat`
- `graphsage`
- `gine`

Notable behavior:

1. `gcn` and `gine` support edge-aware message passing.
2. `gine` requires edge feature dimensionality (`model.encoder.edge_attr_dim`) and runtime `edge_attr`.
3. `gat` requires `hidden_dim` divisible by `gat_heads`.
4. JK modes: `last`, `concat`, `max`, `sum`.
5. Pooling modes: `sum`, `mean`, `max`, `attention`, `set2set`.

## 4) Data Sampling Modes

Sampler strategies (`data.sampler.name`):

- `identity`: standard graph batching.
- `shadow_native`: native PyG ShaDow sampler.
- `shadow_custom`: custom ShaDow implementation supporting per-subgraph transforms.

Runtime controls (`data.sampler.runtime.*`):

- depth and neighbor count
- subgraph batch size
- replacement and shuffling controls
- proportional root sampling by categorical property
- class-weight mode (`inverse`, `sqrt_inverse`, `proportional`)

## 5) Loss/Task Objective Modes

Supported `task.target_type`:

- `binary`
- `regression`
- `survival`

Supported `task.loss` values (via model loss presets):

- `categorical_bce`
- `categorical_ce`
- `regression_mse`
- `regression_huber`
- `survival_coxsgd`

## 6) Runtime Device/Distribution Modes

Trainer presets:

- `trainer=cpu`
- `trainer=gpu`
- `trainer=mps`
- `trainer=ddp`
- `trainer=ddp_sim`

Additional runtime toggles can be overridden directly via Hydra:

- precision/mixed precision
- deterministic mode
- max/min epochs
- validation frequency

## 7) Logging/Experiment Tracking Modes

Logger presets:

- `logger=csv`
- `logger=tensorboard`
- `logger=wandb`
- `logger=mlflow`
- `logger=comet`
- `logger=neptune`
- `logger=aim`
- `logger=many_loggers` (composition)

Callbacks can be selected via `callbacks=...` group or disabled with `callbacks=none`.

## 8) Debug/Development Modes

Debug presets:

- `debug=default`
- `debug=fdr`
- `debug=limit`
- `debug=overfit`
- `debug=profiler`

Common usage:

```bash
python src/train.py debug=fdr
python src/train.py debug=limit
```

## 9) Checkpoint Initialization And Resume Modes

| Mode | Config | Behavior |
|---|---|---|
| Resume training | `ckpt_path=/abs/last.ckpt` | Resume optimizer/trainer state through Lightning fit |
| Initialize weights from checkpoint | `model.init_from_ckpt=/abs/ckpt` | Load state dict before training |
| BGRL-to-supervised key remap | `model.encoder_init_map=auto_bgrl_or_identity` | Remaps known encoder key prefixes |
| Strict load | `model.init_strict=true` | Fail on missing/unexpected keys |
| Freeze encoder | `model.freeze_encoder=true` | Train head/attention while encoder weights stay fixed |

## 10) Inference Modes

1. CLI evaluation mode: `src/eval.py` with required `ckpt_path`.
2. Advanced prediction mode: `Trainer.predict(...)` using `SupervisedModule.predict_step`.
3. Inference utility CLI mode: `src/inference/predict.py` for prediction export, metrics, and embeddings.

Predict outputs include:

- `bag_ids`
- `bag_logits`
- `row_region_ids`
- `row_sample_ids`
- optional `bag_targets`
- optional `bag_attention`
- optional `instance_*` fields
- optional embedding fields (`embedding_bag_ids`, `graph_embeddings`, `embedding_bag_counts`)
- optional node embedding fields (`node_embeddings`, `node_bag_ids`)

Metrics available through the inference utility suite:

- categorical/binary: accuracy, precision, recall, F1, ROC-AUC
- regression: R2, MAE, RMSE
- survival: concordance index (c-index)

Validation (during `src/train.py`) includes:

- step-level `val/loss` logging
- epoch-end region-level aggregation of buffered validation outputs
- task-specific epoch metrics:
  - `val/acc` for binary/categorical
  - `val/mae`, `val/rmse`, `val/r2` for regression
  - `val/c_index` for survival
- optional dedicated validation sampler via `data.val_sampler.*`

## 11) Notebook Validation Modes

Notebooks support interactive smoke validation of:

- data loading and graph construction (`notebooks/data_loading_testing.ipynb`)
- component compatibility matrix (`notebooks/model_components_testing.ipynb`)
- training regime smoke behavior (`notebooks/training_logic_testing.ipynb`)

## 12) CI And Quality Modes

Quality and validation tooling:

- `pre-commit run -a`
- `pytest`
- GitHub Actions workflows for tests and code quality

These operational modes are versioned in `.github/workflows/*.yaml` and `.pre-commit-config.yaml`.
