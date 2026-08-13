# Repository Map

This document maps tracked files in the repository root to their responsibilities.

Scope rules used here:

1. Source of truth is `git ls-files` (tracked files only).
2. Generated/runtime artifacts are documented separately at the end.
3. Every tracked file is listed with one-line purpose and usage context.

## Root Metadata And Project Tooling

- `.project-root`: Root marker used by `rootutils.setup_root(...)`.
- `.gitignore`: Ignore rules for generated artifacts, envs, logs, and data outputs.
- `.env.example`: Example environment variable file loaded by Hydra/OmegaConf env references.
- `.pre-commit-config.yaml`: Code quality hooks (ruff lint + format, plus basic file hygiene checks).
- `LICENSE`: Project license text.
- `README.md`: Primary entrypoint documentation and navigation index.
- `Makefile`: Convenience targets for clean, format, tests, and default training.
- `pyproject.toml`: Package metadata, dependencies, optional extras, console
  entry points (`grass-mil-train`, `-eval`, `-loocv`, `-predict`, `-report`),
  and pytest/coverage/ruff configuration.
- `environment.yaml`: Conda environment alternative for local development, pinned to major versions. CI does not use it; the CI stack is pinned inline in `.github/workflows/test.yml`.

## CI/DevOps And Contribution Files

- `.github/PULL_REQUEST_TEMPLATE.md`: Pull-request submission template.
- `.github/codecov.yml`: Codecov upload/report behavior.
- `.github/dependabot.yml`: Dependabot update policy.
- `.github/release-drafter.yml`: Release drafter configuration.
- `.github/workflows/code-quality-main.yaml`: Pre-commit quality checks on pushes to `main`.
- `.github/workflows/code-quality-pr.yaml`: Pre-commit quality checks on changed PR files.
- `.github/workflows/release-drafter.yml`: Automated draft release workflow.
- `.github/workflows/test.yml`: Multi-OS (Linux/macOS) test workflow, plus coverage and a from-source packaging job.

## Rules And Contracts

- `.cursor/rules/model_component_contracts.mdc`: Contract rules for model/loss/sampler architecture constraints.

## Config System (`configs/`)

> Throughout this document `configs/...` is shorthand for
> `src/grass_mil/configs/...`. The config tree lives inside the package so it
> ships with the wheel; Hydra resolves it relative to the entrypoint module.

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
- `configs/model/components.yaml`: Component-instantiation config used by the component tests.

Encoder presets:

- `configs/model/encoder/gin.yaml`: GIN encoder preset.
- `configs/model/encoder/gcn.yaml`: GCN encoder preset.
- `configs/model/encoder/gat.yaml`: GAT encoder preset.
- `configs/model/encoder/graphsage.yaml`: GraphSAGE encoder preset.
- `configs/model/encoder/gine.yaml`: GINE encoder preset with edge feature dimension.
- `configs/model/encoder/gine_length.yaml`: Production encoder — GINE on the scalar edge length, LayerNorm, cell-type embedding.

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
- `configs/task/finetune_survival.yaml`: Cox survival task preset (scalar log-hazard, c-index).
- `configs/task/pretrain_bgrl.yaml`: SSL BGRL pretraining task preset.

### Augmentation Group

- `configs/augmentation/bgrl_paper.yaml`: Degree-importance view generator (production).
- `configs/augmentation/uniform.yaml`: Structure-independent uniform drop baseline.

### Inference Group

- `configs/inference/predict.yaml`: Prediction entrypoint composition.
- `configs/inference/aggregation/default.yaml`: Bag aggregation mode, scope, and subsampling.
- `configs/inference/embeddings/default.yaml`: Embedding export toggles.
- `configs/inference/interpretability/default.yaml`: Instance/spatial table export toggles.
- `configs/inference/metrics/default.yaml`: Task-aware metric selection.

### Interpretability Group

- `configs/interpretability/report.yaml`: Report entrypoint composition.
- `configs/interpretability/reduction/{pca,tsne,umap}.yaml`: Dimensionality-reduction presets.
- `configs/interpretability/clustering/{agglomerative,dbscan,hdbscan,kmeans,optics,spectral}.yaml`: Clustering presets.
- `configs/interpretability/plugins/default.yaml`: Minimal plugin set (`niche_profiles`).
- `configs/interpretability/plugins/full.yaml`: All inference-only plugins enabled.

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
- `src/grass_mil/contracts.py`: Package-level contracts — attention-width validation and interpretability dataclasses.

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
- `src/grass_mil/data/components/seed_sampling.py`: Convex-hull interior seed masks with on-disk cache.
- `src/grass_mil/data/components/ego_radius.py`: Physical radius cutoff for sampled ego-graphs.

### Model Runtime (`src/grass_mil/models/`)

- `src/grass_mil/models/__init__.py`: Model package marker.
- `src/grass_mil/models/supervised_module.py`: Unified supervised training module (mean + MIL behavior).
- `src/grass_mil/models/bgrl_module.py`: BGRL self-supervised training module.

Model components:

- `src/grass_mil/models/components/__init__.py`: Component package exports.
- `src/grass_mil/models/components/backbones.py`: GNN encoder definitions and configuration contract.
- `src/grass_mil/models/components/pooling.py`: Graph pooling wrapper.
- `src/grass_mil/models/components/attention.py`: MIL attention blocks.
- `src/grass_mil/models/components/heads.py`: Graph-level MLP prediction head. (The node head was removed with the instance-pull auxiliary loss.)
- `src/grass_mil/models/components/losses.py`: Weighted loss implementations (classification/regression/survival).
- `src/grass_mil/models/components/ssl.py`: BGRL and predictor modules.
- `src/grass_mil/models/components/factory.py`: Factory builders for encoder/attention/loss/head/SSL modules.
- `src/grass_mil/models/components/embeddings.py`: `NodeInputEmbedding` — cell-type embedding summed with the projected features.

Training helpers:

- `src/grass_mil/models/training/__init__.py`: Training-helper exports.
- `src/grass_mil/models/training/builders.py`: Component build/inference helpers and task validation.
- `src/grass_mil/models/training/bagging.py`: Bag ID resolution, instance grouping/sampling, bag aggregation logic.
- `src/grass_mil/models/training/loss_utils.py`: Loss routing and accuracy helpers. (Auxiliary/regularisation terms were removed; region cross-entropy is the sole objective.)
- `src/grass_mil/models/training/optimization.py`: Optimizer/scheduler instantiation and warmup wrapper.
- `src/grass_mil/models/training/checkpoint_init.py`: Checkpoint loading and encoder-key remapping.
- `src/grass_mil/models/training/augmentations.py`: BGRL view generation (degree-importance drop, edge drop, feature noise).
- `src/grass_mil/models/training/ssl_runtime.py`: `CosineWarmup` momentum/LR schedule helper. (View augmentations live in `augmentations.py`.)

### Inference (`src/grass_mil/inference/`)

- `src/grass_mil/inference/__init__.py`: Inference package exports.
- `src/grass_mil/inference/predict.py`: Hydra prediction entrypoint; writes predictions, metrics, embeddings and interpretability tables.
- `src/grass_mil/inference/collectors.py`: Normalises `predict_step` outputs into a single payload.
- `src/grass_mil/inference/aggregation.py`: Bag/region aggregation of instance logits (mean, max, attention-weighted).
- `src/grass_mil/inference/metrics.py`: Task-aware metrics, including the chunked concordance index.
- `src/grass_mil/inference/embeddings.py`: Graph/node embedding tables from the collected payload.
- `src/grass_mil/inference/interpretability_export.py`: Builds `instance_table` and `spatial_table` for the report tool.
- `src/grass_mil/inference/schemas.py`: Output table schemas and column contracts.
- `src/grass_mil/inference/io.py`: JSON/CSV writers with non-finite sanitisation.

### Interpretability (`src/grass_mil/interpretability/`)

- `src/grass_mil/interpretability/__init__.py`: Public interpretability API surface.
- `src/grass_mil/interpretability/pipeline.py`: Reduction → clustering → niche ordering → plugin execution.
- `src/grass_mil/interpretability/report_cli.py`: Hydra report entrypoint (`grass-mil-report`).

Core analysis:

- `src/grass_mil/interpretability/core/__init__.py`: Core exports.
- `src/grass_mil/interpretability/core/data.py`: Table loading and embedding-set extraction.
- `src/grass_mil/interpretability/core/reduction.py`: PCA/UMAP/t-SNE reduction.
- `src/grass_mil/interpretability/core/clustering.py`: Clustering backends.
- `src/grass_mil/interpretability/core/niches.py`: Background labelling, dendrogram ordering, niche renumbering.
- `src/grass_mil/interpretability/core/biomarkers.py`: Per-niche composition summaries.
- `src/grass_mil/interpretability/core/attribution.py`: Attention-weighted logit margins, OvR contrasts, bootstrap CIs, identity residual.
- `src/grass_mil/interpretability/core/agreement.py`: Cross-space agreement (ARI/AMI/NMI) and cluster Jaccard.
- `src/grass_mil/interpretability/core/transfer.py`: Niche label transfer to new cohorts via PCA+kNN.

Tier-2 spatial statistics:

- `src/grass_mil/interpretability/tier2/__init__.py`: Tier-2 exports.
- `src/grass_mil/interpretability/tier2/neighborhood.py`: Neighbourhood enrichment and differential enrichment.
- `src/grass_mil/interpretability/tier2/filtration.py`: Filtration curves over distance thresholds.
- `src/grass_mil/interpretability/tier2/autocorrelation.py`: Moran's I with permutation null and BH-FDR.
- `src/grass_mil/interpretability/tier2/ripley.py`: Ripley cross-L with self-pair correction.
- `src/grass_mil/interpretability/tier2/tissue_graph.py`: Tissue-graph view preparation and figure building.

Plugins:

- `src/grass_mil/interpretability/plugins/__init__.py`: Plugin exports.
- `src/grass_mil/interpretability/plugins/base.py`: Plugin protocol, context and result types.
- `src/grass_mil/interpretability/plugins/registry.py`: Name-keyed plugin registry.
- `src/grass_mil/interpretability/plugins/builtin.py`: The ten built-in plugins.

Reporting:

- `src/grass_mil/interpretability/reporting/__init__.py`: Reporting exports.
- `src/grass_mil/interpretability/reporting/plotly_builders.py`: Figure builders.
- `src/grass_mil/interpretability/reporting/sections.py`: Section assembly from plugin results.
- `src/grass_mil/interpretability/reporting/html.py`: HTML document rendering.
- `src/grass_mil/interpretability/reporting/snapshot.py`: Static image export for PDF.
- `src/grass_mil/interpretability/reporting/pdf.py`: PDF rendering backend.
- `src/grass_mil/interpretability/reporting/render.py`: Render orchestration and graceful degradation.

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
- `tests/helpers/config_paths.py`: Resolves the packaged config tree for tests.

### Top-Level Integration/Config Tests

- `tests/test_configs.py`: Compose/instantiate sanity tests for train/eval configs.
- `tests/test_train.py`: Core training loop behavior tests (fast-dev, resume, DDP sim, etc.).
- `tests/test_eval.py`: Train-then-eval integration test.
- `tests/test_train_regimes.py`: Regime matrix tests (BGRL, mean, MIL, SSL->finetune).
- `tests/test_spatial_omics_datamodule.py`: Data preprocessing and split persistence behavior tests.
- `tests/test_loocv.py`: Fold discovery, isolation-flag safety, and per-fold run tests.
- `tests/test_packaging.py`: Installed-package import, shipped configs, and console-script contract tests.

### Config Contract Tests

- `tests/configs/test_component_instantiation.py`: YAML component and sampler resolution tests.

### Data Tests

- `tests/data/test_samplers.py`: Identity and shadow sampler behavior/edge-case tests.
- `tests/data/test_datamodule_sampler_exposure.py`: Datamodule sampler strategy exposure tests.
- `tests/data/test_precompute_reducer_modes.py`: Reducer fit-mode semantics (`global`, `train_only`) tests.
- `tests/data/test_precompute_manifest_regions.py`: Manifest region-id uniqueness validation tests.
- `tests/data/test_loader_flags.py`: Loader flag propagation tests.
- `tests/data/test_seed_sampling.py`: Interior seed masks and ego-radius cutoff tests.

### Model Tests

- `tests/models/test_backbones.py`: Backbone forward and JK mode tests.
- `tests/models/test_attention.py`: Attention shape and factory optionality tests.
- `tests/models/test_losses.py`: Loss finite-output and edge-case tests.
- `tests/models/test_ssl.py`: SSL/BGRL forward/update and factory validation tests.
- `tests/models/test_optimization.py`: Warmup/scheduler and multi-LR optimizer-group tests.
- `tests/models/test_runtime.py`: Runtime helper behavior and MIL buffering/predict-step tests.
- `tests/models/test_bagging.py`: Per-class attention normalisation and the bag-logit identity.
- `tests/models/test_node_embeddings.py`: Cell-type embedding, zero-`input_dim`, and edge-column selection tests.
- `tests/models/test_augmentations.py`: BGRL view generation determinism and hyperparameter validation.

### Inference Tests

- `tests/eval/test_predict_cli.py`: Prediction entrypoint smoke and subsampling-plan tests.
- `tests/eval/test_aggregation.py`: Bag/region aggregation across modes and scopes.
- `tests/eval/test_collectors.py`: Payload normalisation and required-field errors.
- `tests/eval/test_metrics.py`: Task metrics, including concordance-index correctness and speed.
- `tests/eval/test_embeddings.py`: Embedding payload to DataFrame conversion.
- `tests/eval/test_interpretability_export.py`: Instance/spatial table construction and attention columns.
- `tests/eval/test_io.py`: JSON writer non-finite sanitisation.

### Interpretability Tests

- `tests/interpretability/test_core.py`: Reduction, clustering and composition-summary determinism.
- `tests/interpretability/test_niches.py`: Background labelling, dendrogram ordering and renumbering.
- `tests/interpretability/test_attribution.py`: Margin attribution, OvR contrasts, identity residual, bootstrap CIs.
- `tests/interpretability/test_spatial_statistics.py`: Moran's I, Ripley cross-L and cluster agreement.
- `tests/interpretability/test_plugins_and_tier2.py`: Plugin registry and tier-2 analysis behaviour.
- `tests/interpretability/test_pipeline_and_report.py`: End-to-end pipeline and report rendering.
- `tests/interpretability/test_tissue_graph.py`: Tissue-graph view preparation and figure tests.
- `tests/interpretability/test_transfer.py`: Niche transfer fit/apply and bundle round-trip tests.

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

- `docs/method_fidelity.md`: Method → code → test map, known divergences from `methods.tex`, and known defects.
- `docs/legacy_gap_audit.md`: Status record of closed fidelity gaps and the bugs found closing them.
- `docs/model_component_contracts.md`: Canonical component contracts for model/sampler/loss behavior.
- `docs/repository_map.md`: This tracked-file responsibility map.
- `docs/dataset_preparation.md`: Manifest and input-schema preparation guide.
- `docs/workflows_train_infer.md`: End-to-end training/inference runbooks.
- `docs/hyperparameter_reference.md`: Exhaustive reference of repo-defined configuration parameters.
- `docs/capabilities_and_modes.md`: Operational capability matrix and mode catalog.
- `docs/inference_utility_suite.md`: Prediction outputs, aggregation modes and exported tables.
- `docs/interpretability_suite.md`: Interpretability architecture, contracts and plugin catalog.
- `docs/interpretability_extension_guide.md`: How to add a new analysis plugin.

### Vignettes

- `vignettes/README.md`: Index, dataset description and honest scope notes.
- `vignettes/01_anndata_to_spatial_graphs.ipynb`: squidpy AnnData to manifest to cellular graphs.
- `vignettes/02_pretraining_and_niches.ipynb`: BGRL pretraining, embedding export and niche discovery.

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
