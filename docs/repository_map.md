# Repository Map

This document maps tracked files in `/Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil` to their responsibilities.

Scope rules used here:

1. Source of truth is `git ls-files` (tracked files only).
2. Generated/runtime artifacts are documented separately at the end.
3. Every tracked file is listed with one-line purpose and usage context.

## Root Metadata And Project Tooling

- `.project-root`: Root marker used by `rootutils.setup_root(...)`.
- `.gitignore`: Ignore rules for generated artifacts, envs, logs, and data outputs.
- `.env.example`: Example environment variable file loaded by Hydra/OmegaConf env references.
- `.pre-commit-config.yaml`: Code quality hooks (format/lint/notebook cleanup/docs checks).
- `LICENSE`: Project license text.
- `README.md`: Primary entrypoint documentation and navigation index.
- `Makefile`: Convenience targets for clean, format, tests, and default training.
- `pyproject.toml`: Package metadata, dependencies and optional extras.
- `environment.yaml`: Conda environment alternative with pinned major versions.
- `setup.py`: Package metadata and console entry points (`train_command`, `eval_command`).
- `pyproject.toml`: Pytest and coverage tool configuration.

## CI/DevOps And Contribution Files

- `.github/PULL_REQUEST_TEMPLATE.md`: Pull-request submission template.
- `.github/codecov.yml`: Codecov upload/report behavior.
- `.github/dependabot.yml`: Dependabot update policy.
- `.github/release-drafter.yml`: Release drafter configuration.
- `.github/workflows/code-quality-main.yaml`: Pre-commit quality checks on pushes to `main`.
- `.github/workflows/code-quality-pr.yaml`: Pre-commit quality checks on changed PR files.
- `.github/workflows/release-drafter.yml`: Automated draft release workflow.
- `.github/workflows/test.yml`: Multi-OS, multi-Python automated test workflow.

## Rules And Contracts

- `.cursor/rules/model_component_contracts.mdc`: Contract rules for model/loss/sampler architecture constraints.

## Config System (`configs/`)

### Config Package Marker

- `configs/__init__.py`: Marks config directory as package context.

### Top-Level Compositions

- `configs/train.yaml`: Primary training composition and defaults chain.
- `configs/eval.yaml`: Primary evaluation composition and defaults chain.
- `configs/loocv.yaml`: LOOCV composition for single-fold and full-loop orchestration.

### Callback Config Group

- `configs/callbacks/default.yaml`: Default callback bundle composition.
- `configs/callbacks/model_checkpoint.yaml`: Checkpoint callback schema/default fields.
- `configs/callbacks/early_stopping.yaml`: Early stopping callback schema/default fields.
- `configs/callbacks/model_summary.yaml`: Rich model summary callback config.
- `configs/callbacks/rich_progress_bar.yaml`: Rich progress bar callback config.
- `configs/callbacks/none.yaml`: Empty callback preset to disable callbacks.

### Data Config Group

- `configs/data/spatial_omics.yaml`: Main datamodule and preprocessing/data-loading parameters.

Sampler subgroup:

- `configs/data/sampler/identity.yaml`: Identity batching strategy preset.
- `configs/data/sampler/shadow_native.yaml`: Native PyG ShaDow sampling preset.
- `configs/data/sampler/shadow_custom.yaml`: Custom ShaDow sampling preset with transform support.

### Debug Config Group

- `configs/debug/default.yaml`: Debug baseline (CPU, no logger/callbacks, anomaly detection).
- `configs/debug/fdr.yaml`: Fast-dev-run debug preset.
- `configs/debug/limit.yaml`: Subset-batch debug preset.
- `configs/debug/overfit.yaml`: Overfit-a-few-batches debug preset.
- `configs/debug/profiler.yaml`: Profiling-enabled debug preset.

### Experiment Config Group

- `configs/experiment/debug.yaml`: Dummy-data composition for notebook/debug experiments.
- `configs/experiment/notebook_data_shadow_custom.yaml`: Notebook data-loading scenario with shadow sampler.
- `configs/experiment/notebook_model_components.yaml`: Notebook scenario for component-combination checks.
- `configs/experiment/notebook_training_logic.yaml`: Notebook scenario for training-regime smoke tests.

### Extras/Hydra/Paths Groups

- `configs/extras/default.yaml`: Run-time UX toggles (warnings, tags, config printing).
- `configs/hydra/default.yaml`: Hydra logging and output directory templates.
- `configs/paths/default.yaml`: Path roots, data dir, log dir, output dir, work dir.
- `configs/local/.gitkeep`: Placeholder for untracked machine-local overrides.

### Logger Config Group

- `configs/logger/csv.yaml`: CSV logger preset.
- `configs/logger/tensorboard.yaml`: TensorBoard logger preset.
- `configs/logger/wandb.yaml`: Weights & Biases logger preset.
- `configs/logger/mlflow.yaml`: MLflow logger preset.
- `configs/logger/comet.yaml`: Comet logger preset.
- `configs/logger/neptune.yaml`: Neptune logger preset.
- `configs/logger/aim.yaml`: Aim logger preset.
- `configs/logger/many_loggers.yaml`: Multi-logger composition preset.

### Model Config Group

Root model module configs:

- `configs/model/supervised_module.yaml`: Unified supervised module defaults (mean/MIL via task config).
- `configs/model/bgrl_module.yaml`: BGRL pretraining module defaults.
- `configs/model/components.yaml`: Component-instantiation playground config used in notebooks/tests.

Encoder presets:

- `configs/model/encoder/gin.yaml`: GIN encoder preset.
- `configs/model/encoder/gcn.yaml`: GCN encoder preset.
- `configs/model/encoder/gat.yaml`: GAT encoder preset.
- `configs/model/encoder/graphsage.yaml`: GraphSAGE encoder preset.
- `configs/model/encoder/gine.yaml`: GINE encoder preset with edge feature dimension.

Attention presets:

- `configs/model/attention/gated.yaml`: Gated MIL attention preset.
- `configs/model/attention/gated_projected.yaml`: Projected gated MIL attention preset.

Head presets:

- `configs/model/heads/graph.yaml`: Graph-level prediction MLP head preset.

Loss presets:

- `configs/model/loss/categorical_bce.yaml`: Binary BCE-with-logits loss preset.
- `configs/model/loss/categorical_ce.yaml`: Multiclass cross-entropy loss preset.
- `configs/model/loss/regression_mse.yaml`: MSE regression loss preset.
- `configs/model/loss/regression_huber.yaml`: Huber regression loss preset.
- `configs/model/loss/survival_coxsgd.yaml`: Cox-style survival loss preset.

SSL preset:

- `configs/model/ssl/bgrl.yaml`: BGRL SSL predictor config preset.

### Optimization And Scheduling Groups

- `configs/optim/adamw.yaml`: Default AdamW optimizer preset.
- `configs/scheduler/cosine_epoch.yaml`: Epoch-interval cosine scheduler preset.
- `configs/scheduler/cosine_step.yaml`: Step-interval cosine scheduler preset.

### Task Group

- `configs/task/finetune_mean.yaml`: Supervised mean-aggregation task preset.
- `configs/task/finetune_mil.yaml`: Supervised MIL-attention task preset with optional MIL extras.
- `configs/task/pretrain_bgrl.yaml`: SSL BGRL pretraining task preset.

### Trainer Group

- `configs/trainer/default.yaml`: Base Lightning Trainer defaults.
- `configs/trainer/cpu.yaml`: CPU runtime preset.
- `configs/trainer/gpu.yaml`: Single-GPU preset.
- `configs/trainer/mps.yaml`: Apple MPS runtime preset.
- `configs/trainer/ddp.yaml`: Multi-GPU DDP preset.
- `configs/trainer/ddp_sim.yaml`: CPU `ddp_spawn` simulation preset.

## Source Code (`src/grass_mil/`)

### Package Marker

- `src/grass_mil/__init__.py`: Package marker.
- `src/grass_mil/_paths.py`: Best-effort `PROJECT_ROOT` resolution for the CLI entrypoints.
- `src/grass_mil/configs/`: Hydra config tree, shipped as package data.

### Entrypoints

- `src/grass_mil/train.py`: Hydra training entrypoint (train + optional test).
- `src/grass_mil/eval.py`: Hydra evaluation entrypoint (test from checkpoint).
- `src/grass_mil/loocv.py`: Hydra LOOCV entrypoint with fold discovery, per-fold runs, and aggregate summaries.

### Data Pipeline (`src/grass_mil/data/`)

- `src/grass_mil/data/__init__.py`: Data package marker.
- `src/grass_mil/data/spatial_omics_datamodule.py`: LightningDataModule wiring preprocessing, splitting, and loaders.

Data components:

- `src/grass_mil/data/components/__init__.py`: Data components package marker.
- `src/grass_mil/data/components/spatial_types.py`: Canonical in-memory spatial table dataclass.
- `src/grass_mil/data/components/loaders.py`: CSV/H5AD/SCE loaders and polygon parsing utilities.
- `src/grass_mil/data/components/patching.py`: Polygon/tile patch extraction helpers.
- `src/grass_mil/data/components/graph_builders.py`: Graph builder interfaces and Delaunay implementation.
- `src/grass_mil/data/components/feature_reducers.py`: Feature reducer interfaces and implementations (`identity`, `pca`).
- `src/grass_mil/data/components/precompute.py`: Precompute orchestration into persisted PyG graph artifacts/indexes.
- `src/grass_mil/data/components/datasets.py`: Dataset wrappers for persisted graph indices and transform application.
- `src/grass_mil/data/components/samplers.py`: Batch/sampler strategies (identity, shadow_native, shadow_custom).
- `src/grass_mil/data/components/transforms.py`: Transform instantiation and composition-vector transform.

### Model Runtime (`src/grass_mil/models/`)

- `src/grass_mil/models/__init__.py`: Model package marker.
- `src/grass_mil/models/supervised_module.py`: Unified supervised training module (mean + MIL behavior).
- `src/grass_mil/models/bgrl_module.py`: BGRL self-supervised training module.

Model components:

- `src/grass_mil/models/components/__init__.py`: Component package exports.
- `src/grass_mil/models/components/backbones.py`: GNN encoder definitions and configuration contract.
- `src/grass_mil/models/components/pooling.py`: Graph pooling wrapper.
- `src/grass_mil/models/components/attention.py`: MIL attention blocks.
- `src/grass_mil/models/components/heads.py`: Graph/node MLP heads.
- `src/grass_mil/models/components/losses.py`: Weighted loss implementations (classification/regression/survival).
- `src/grass_mil/models/components/ssl.py`: BGRL and predictor modules.
- `src/grass_mil/models/components/factory.py`: Factory builders for encoder/attention/loss/head/SSL modules.

Training helpers:

- `src/grass_mil/models/training/__init__.py`: Training-helper exports.
- `src/grass_mil/models/training/builders.py`: Component build/inference helpers and task validation.
- `src/grass_mil/models/training/bagging.py`: Bag ID resolution, instance grouping/sampling, bag aggregation logic.
- `src/grass_mil/models/training/loss_utils.py`: Loss routing and MIL auxiliary/regularization utilities.
- `src/grass_mil/models/training/optimization.py`: Optimizer/scheduler instantiation and warmup wrapper.
- `src/grass_mil/models/training/checkpoint_init.py`: Checkpoint loading and encoder-key remapping.
- `src/grass_mil/models/training/ssl_runtime.py`: SSL augmentations and momentum schedule helpers.

### Utilities (`src/grass_mil/utils/`)

- `src/grass_mil/utils/__init__.py`: Utility export surface.
- `src/grass_mil/utils/pylogger.py`: Rank-aware logger adapter.
- `src/grass_mil/utils/instantiators.py`: Hydra callback/logger instantiation helpers.
- `src/grass_mil/utils/logging_utils.py`: Hyperparameter logging packaging for logger backends.
- `src/grass_mil/utils/rich_utils.py`: Rich config-tree and tag prompting utilities.
- `src/grass_mil/utils/utils.py`: Misc execution helpers (`extras`, `task_wrapper`, metric extraction).

## Tests (`tests/`)

### Package/Fixture Layer

- `tests/__init__.py`: Tests package marker.
- `tests/conftest.py`: Shared Hydra config fixtures for train/eval tests.

### Top-Level Integration/Config Tests

- `tests/test_configs.py`: Compose/instantiate sanity tests for train/eval configs.
- `tests/test_train.py`: Core training loop behavior tests (fast-dev, resume, DDP sim, etc.).
- `tests/test_eval.py`: Train-then-eval integration test.
- `tests/test_train_regimes.py`: Regime matrix tests (BGRL, mean, MIL, SSL->finetune).
- `tests/test_spatial_omics_datamodule.py`: Data preprocessing and split persistence behavior tests.

### Config Contract Tests

- `tests/configs/test_component_instantiation.py`: YAML component and sampler resolution tests.

### Data Tests

- `tests/data/test_samplers.py`: Identity and shadow sampler behavior/edge-case tests.
- `tests/data/test_datamodule_sampler_exposure.py`: Datamodule sampler strategy exposure tests.
- `tests/data/test_precompute_reducer_modes.py`: Reducer fit-mode semantics (`global`, `train_only`) tests.

### Model Tests

- `tests/models/test_backbones.py`: Backbone forward and JK mode tests.
- `tests/models/test_attention.py`: Attention shape and factory optionality tests.
- `tests/models/test_losses.py`: Loss finite-output and edge-case tests.
- `tests/models/test_ssl.py`: SSL/BGRL forward/update and factory validation tests.
- `tests/models/test_optimization.py`: Warmup/scheduler and multi-LR optimizer-group tests.
- `tests/models/test_runtime.py`: Runtime helper behavior and MIL buffering/predict-step tests.

### Contract Tests

- `tests/contracts/test_model_component_contracts.py`: Presence/consistency checks for documented component contracts.

### Test Helper Utilities

- `tests/helpers/__init__.py`: Helper package marker.
- `tests/helpers/package_available.py`: Dependency availability probes for conditional tests.
- `tests/helpers/run_if.py`: Conditional marker helper for environment-gated tests.
- `tests/helpers/run_sh_command.py`: Shell-command helper for tests.
- `tests/helpers/synthetic_datamodule.py`: Synthetic datamodule fixture for fast isolated training tests.

## Scripts, Docs, Notebooks, And Placeholders

### Scripts

- `scripts/schedule.sh`: Example sequential training-run script.

### Documentation

- `docs/legacy_gap_audit.md`: Audit of resolved legacy implementation risks.
- `docs/model_component_contracts.md`: Canonical component contracts for model/sampler/loss behavior.
- `docs/repository_map.md`: This tracked-file responsibility map.
- `docs/workflows_train_infer.md`: End-to-end training/inference runbooks.
- `docs/hyperparameter_reference.md`: Exhaustive reference of repo-defined configuration parameters.
- `docs/capabilities_and_modes.md`: Operational capability matrix and mode catalog.

### Notebooks

- `notebooks/.gitkeep`: Placeholder to keep notebook directory tracked.
- `notebooks/data_loading_testing.ipynb`: Interactive data-loading and graph-construction smoke notebook.
- `notebooks/model_components_testing.ipynb`: Interactive model-component combinatorial smoke notebook.
- `notebooks/training_logic_testing.ipynb`: Interactive training-regime smoke notebook.

### Data/Logs Placeholders

- `data/.gitkeep`: Placeholder to keep data directory tracked.
- `logs/.gitkeep`: Placeholder to keep logs directory tracked.

## Generated/Runtime Artifacts (Not Part Of Tracked Functional Map)

These are intentionally excluded from the tracked functional map above and are typically ignored or environment-specific:

- `.git/` internals (VCS metadata and object store)
- `__pycache__/` and `*.pyc`
- `.pytest_cache/`
- Hydra run directories under `logs/<task>/runs/*` and `logs/<task>/multiruns/*`
- Precomputed runtime outputs under `data/processed/*` (unless explicitly committed by the repository owner)
- Local virtual environments (`.venv`, `venv`, etc.)

Use `git ls-files` to regenerate the tracked-only baseline when the repository changes.
