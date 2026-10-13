# Capabilities And Modes

This document catalogs operational modes and non-default capabilities in `grass-mil`.

## 1) Training Regimes

| Regime | Task | Model | Primary Use |
|---|---|---|---|
| Supervised mean | `finetune_mean` | `supervised_module` | Stable baseline graph-level classification/regression/survival |
| Supervised MIL | `finetune_mil` | `supervised_module` | Per-class attention aggregation; region cross-entropy is the sole objective |
| Cox survival MIL | `finetune_survival` | `supervised_module` | Scalar log-hazard per bag, scored by c-index |
| SSL pretrain | `pretrain_bgrl` | `bgrl_module` | Representation learning before supervised fine-tuning |
| GRASS-MIL pretrain (A3) | `experiment=grass_mil_pretrain_a3` | `bgrl_a3` | The preprint's pretraining recipe |
| GRASS-MIL fine-tuning | `regime=mil` (`mil_frozen`, `linear_probehead`, `linear_ftall`, `linear_ftall_reg`) | `grass_mil` | One region = one bag, scaled learning rates, probe pre-fit, region-level selection |

## 1b) Analysis Entry Points

| CLI | Purpose | Doc |
|---|---|---|
| `grass-mil-probe` | logistic probe on the frozen encoder, folded into a linear head | `nested_cv.md` |
| `grass-mil-nested-cv` | nested patient-grouped cross-validation with out-of-fold predictions | `nested_cv.md` |
| `grass-mil-extract` | per-cell cache (embedding, instance logits, attention) of trained models | `niche_discovery.md` |
| `grass-mil-niches` | niche discovery and attention-weighted attribution on a model's cells | `niche_discovery.md` |
| `grass-mil-metaclusters` | cross-fold niche replication, transfer, shared basis | `niche_discovery.md` |
| `grass-mil-scores` | AUROC intervals, DeLong, operating points | `evaluation_stats.md` |
| `grass-mil-spatial-stats` | neighbourhood enrichment, cellular neighbourhoods, contacts, cross-K | `spatial_statistics.md` |
| `grass-mil-predict` | predictions, region/patient aggregation (mean or merged bag), embeddings, instance tables | `inference_utility_suite.md` |
| `grass-mil-report` | HTML/PDF interpretability report on instance tables | `interpretability_suite.md` |

## 2) Aggregation And MIL Feature Modes

| Mode | Config | Behavior |
|---|---|---|
| Mean bag aggregation | `task.aggregation=mean` | Averages instance logits per bag |
| MIL attention aggregation | `task.aggregation=mil_attention` | Learns attention weights over instance embeddings |
| Region accumulation | `task.region_accumulation.enabled=true` | Manual optimizer steps over region hyperbatches |
| One region per bag | `data.sampler.runtime.unit_as_bag=true` (`data/sampler=ego_graph_mil`) | All sampled ego-graphs of a region in one batch: one attention softmax, one loss |
| Patient aggregation | `aggregation.bag_scope=patient` with `mode=mean_probability` or `attention_weighted` | Mean of region probabilities, or one merged bag per patient |

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
5. Pooling modes: `sum`, `mean`, `max`. Learned pooling was removed: instance
   selection is the MIL head's job, pooling across ego-graphs.

## 4) Data Sampling Modes

Sampler strategies (`data.sampler.name`):

- `identity`: standard graph batching.
- `shadow_native`: native PyG ShaDow sampler.
- `shadow_custom`: custom ShaDow implementation supporting per-subgraph transforms.

Runtime controls (`data.sampler.runtime.*`):

- depth and neighbor count (`num_neighbors=-1` = unlimited)
- subgraph batch size
- replacement and shuffling controls
- proportional root sampling by categorical property
- class-weight mode (`inverse`, `sqrt_inverse`, `proportional`)
- roots per unit per epoch (`subsample_count` / `subsample_fraction`), fixed per-unit seeds
- one batch per unit (`unit_as_bag`), roots from the union of units (`merge_units`)
- separate validation and test samplers (`data.val_sampler`, `data.test_sampler`)

## 5) Loss/Task Objective Modes

Supported `task.target_type`:

- `binary`
- `categorical` (per-class attention; the GRASS-MIL default for binary tasks)
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
grass-mil-train debug=fdr
grass-mil-train debug=limit
```

## 9) Checkpoint Initialization And Resume Modes

| Mode | Config | Behavior |
|---|---|---|
| Resume training | `ckpt_path=/abs/last.ckpt` | Resume optimizer/trainer state through Lightning fit |
| Initialize weights from checkpoint | `model.init_from_ckpt=/abs/ckpt` | Load state dict before training |
| BGRL-to-supervised key remap | `model.encoder_init_map=auto_bgrl_or_identity` (`spacegm_bgrl` for legacy checkpoints) | Remaps known encoder key prefixes |
| Probe head init | `task.head_init_path=<probe_head.pt>` | Linear head from `grass-mil-probe`; `freeze_head` keeps it fixed |
| Strict load | `model.init_strict=true` | Fail on missing/unexpected keys |
| Freeze encoder | `model.freeze_encoder=true` | Train head/attention while encoder weights stay fixed |

## 10) Inference Modes

1. CLI evaluation mode: `src/grass_mil/eval.py` with required `ckpt_path`.
2. Advanced prediction mode: `Trainer.predict(...)` using `SupervisedModule.predict_step`.
3. Inference utility CLI mode: `src/grass_mil/inference/predict.py` for prediction export, metrics, and embeddings.

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

Validation (during `src/grass_mil/train.py`) includes:

- step-level `val/loss` logging
- epoch-end region-level aggregation of buffered validation outputs
- task-specific epoch metrics:
  - `val/acc`, `val/loss_region`, `val/auroc`, `val/bal_acc` for binary/categorical (region level)
  - `val/mae`, `val/rmse`, `val/r2` for regression
  - `val/c_index` for survival
- optional dedicated validation sampler via `data.val_sampler.*`

## 11) Notebook Validation Modes

Notebooks support interactive smoke validation of:

- ingesting a public AnnData cohort and building cellular graphs
  (`vignettes/01_anndata_to_spatial_graphs.ipynb`)
- self-supervised pretraining and spatial-niche discovery
  (`vignettes/02_pretraining_and_niches.ipynb`)
- nested cross-validation, niche attribution, cross-fold metaclusters and scores
  (`vignettes/03_nested_cv_niches_and_metaclusters.ipynb`)

## 12) CI And Quality Modes

Quality and validation tooling:

- `pre-commit run -a`
- `pytest`
- GitHub Actions workflows for tests and code quality

These operational modes are versioned in `.github/workflows/*.yaml` and `.pre-commit-config.yaml`.
