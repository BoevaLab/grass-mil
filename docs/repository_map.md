# Repository Map

Every tracked file (`git ls-files`) with a one-line purpose. Regenerate with the
script in the git history of this file after adding files.

## Root metadata and tooling

- `.gitignore`: Ignore rules for generated artefacts, environments, logs, data and vignette outputs.
- `.pre-commit-config.yaml`: Code-quality hooks (ruff lint + format, file hygiene checks).
- `.project-root`: Root marker used by `rootutils.setup_root(...)`.
- `LICENSE`: MIT license.
- `Makefile`: Convenience targets: `test`, `format`, `clean`, `clean-logs`.
- `README.md`: Landing page: method summary, install, quick start, workflow table, citation.
- `environment.yaml`: Conda environment alternative for local development.
- `pyproject.toml`: Package metadata, dependencies, extras, console entry points (`grass-mil-*`), pytest/ruff configuration.

## CI and contribution files

- `.github/PULL_REQUEST_TEMPLATE.md`: Pull-request template.
- `.github/codecov.yml`: Codecov configuration.
- `.github/dependabot.yml`: Dependabot update policy.
- `.github/release-drafter.yml`: Release drafter configuration.
- `.github/workflows/code-quality-main.yaml`: CI: pre-commit on every push to main.
- `.github/workflows/code-quality-pr.yaml`: CI: pre-commit on changed files of a pull request.
- `.github/workflows/release-drafter.yml`: CI: draft release notes from merged pull requests.
- `.github/workflows/test.yml`: CI: pytest on Linux and macOS, coverage, install-from-source smoke test.
- `.vscode/settings.json`: Editor settings.

## Package: entry points and shared modules

- `src/grass_mil/__init__.py`: GRASS-MIL: graph-based representation and discovery of phenotype-associated.
- `src/grass_mil/_compat.py`: Cross-version compatibility shims applied at package import.
- `src/grass_mil/_paths.py`: Project-root resolution shared by the CLI entrypoints.
- `src/grass_mil/batching.py`: Splitting batched graphs back into their per-subgraph parts.
- `src/grass_mil/callbacks.py`: Lightweight run-artefact callbacks.
- `src/grass_mil/contracts.py`: Cross-cutting contracts.
- `src/grass_mil/eval.py`: Module.
- `src/grass_mil/extract.py`: ``grass-mil-extract``: per-cell cache of trained models (see inference/cell_cache.py).
- `src/grass_mil/loocv.py`: Module.
- `src/grass_mil/nested_cv.py`: ``grass-mil-nested-cv``: nested, patient-grouped cross-validation (preprint protocol).
- `src/grass_mil/probe.py`: ``grass-mil-probe``: fit the linear probe head on a frozen encoder.
- `src/grass_mil/train.py`: Module.

## Package: data pipeline

- `src/grass_mil/data/__init__.py`: Package marker / re-exports.
- `src/grass_mil/data/components/__init__.py`: Package marker / re-exports.
- `src/grass_mil/data/components/datasets.py`: Module.
- `src/grass_mil/data/components/ego_radius.py`: Euclidean radius cutoff for k-hop ego-graphs.
- `src/grass_mil/data/components/feature_reducers.py`: Module.
- `src/grass_mil/data/components/graph_builders.py`: Module.
- `src/grass_mil/data/components/loaders.py`: Module.
- `src/grass_mil/data/components/merge.py`: Merge several graph units into one disconnected graph.
- `src/grass_mil/data/components/patching.py`: Module.
- `src/grass_mil/data/components/precompute.py`: Module.
- `src/grass_mil/data/components/samplers.py`: Module.
- `src/grass_mil/data/components/seed_sampling.py`: Interior-only seed sampling for ego-graph extraction.
- `src/grass_mil/data/components/spatial_types.py`: Module.
- `src/grass_mil/data/components/splits.py`: Explicit splits, fold files and patient-grouped inner folds.
- `src/grass_mil/data/components/transforms.py`: Module.
- `src/grass_mil/data/spatial_omics_datamodule.py`: Module.

## Package: models

- `src/grass_mil/models/__init__.py`: Package marker / re-exports.
- `src/grass_mil/models/bgrl_module.py`: Module.
- `src/grass_mil/models/components/__init__.py`: Package marker / re-exports.
- `src/grass_mil/models/components/attention.py`: Module.
- `src/grass_mil/models/components/backbones.py`: Module.
- `src/grass_mil/models/components/embeddings.py`: Input embeddings for node features.
- `src/grass_mil/models/components/factory.py`: Module.
- `src/grass_mil/models/components/heads.py`: Module.
- `src/grass_mil/models/components/losses.py`: Module.
- `src/grass_mil/models/components/pooling.py`: Module.
- `src/grass_mil/models/components/ssl.py`: Module.
- `src/grass_mil/models/supervised_module.py`: Module.
- `src/grass_mil/models/training/__init__.py`: Package marker / re-exports.
- `src/grass_mil/models/training/augmentations.py`: Graph augmentations for self-supervised pretraining.
- `src/grass_mil/models/training/bagging.py`: Module.
- `src/grass_mil/models/training/builders.py`: Module.
- `src/grass_mil/models/training/checkpoint_init.py`: Module.
- `src/grass_mil/models/training/class_weights.py`: Class weights for the region cross-entropy, computed from the TRAINING split only.
- `src/grass_mil/models/training/loss_utils.py`: Module.
- `src/grass_mil/models/training/optimization.py`: Module.
- `src/grass_mil/models/training/probe.py`: Linear probe pre-fit on a frozen encoder (preprint "probe head").
- `src/grass_mil/models/training/ssl_runtime.py`: Module.

## Package: inference

- `src/grass_mil/inference/__init__.py`: Package marker / re-exports.
- `src/grass_mil/inference/aggregation.py`: Module.
- `src/grass_mil/inference/cell_cache.py`: Per-cell cache of a trained GRASS-MIL model: the input of niche discovery.
- `src/grass_mil/inference/collectors.py`: Module.
- `src/grass_mil/inference/interpretability_export.py`: Module.
- `src/grass_mil/inference/io.py`: Module.
- `src/grass_mil/inference/metrics.py`: Module.
- `src/grass_mil/inference/model_loading.py`: Build a supervised module from config and load trained weights outside ``Trainer``.
- `src/grass_mil/inference/predict.py`: Module.
- `src/grass_mil/inference/schemas.py`: Module.

## Package: evaluation statistics

- `src/grass_mil/evaluation/__init__.py`: Evaluation statistics for held-out scores (AUROC CIs, DeLong, operating points).
- `src/grass_mil/evaluation/cli.py`: ``grass-mil-scores``: AUROC CIs, DeLong comparisons and operating points.
- `src/grass_mil/evaluation/operating_points.py`: Sensitivity / specificity / PPV / NPV at pre-specified operating rules.
- `src/grass_mil/evaluation/stats.py`: AUROC statistics for held-out (out-of-fold) scores.

## Package: interpretability (niches, metaclusters, report)

- `src/grass_mil/interpretability/__init__.py`: Package marker / re-exports.
- `src/grass_mil/interpretability/core/__init__.py`: Package marker / re-exports.
- `src/grass_mil/interpretability/core/agreement.py`: Agreement between partitions of the same instances.
- `src/grass_mil/interpretability/core/attribution.py`: Exact attribution of a bag prediction to its instances.
- `src/grass_mil/interpretability/core/biomarkers.py`: Module.
- `src/grass_mil/interpretability/core/clustering.py`: Module.
- `src/grass_mil/interpretability/core/data.py`: Module.
- `src/grass_mil/interpretability/core/niches.py`: Niche identity: naming, the background label, and composition ordering.
- `src/grass_mil/interpretability/core/reduction.py`: Module.
- `src/grass_mil/interpretability/core/transfer.py`: Module.
- `src/grass_mil/interpretability/metaclusters/__init__.py`: Cross-fold niche replication: per-fold clustering, matching into metaclusters,.
- `src/grass_mil/interpretability/metaclusters/basis.py`: Stage 4: a shared, label-free k-means basis with per-model attribution.
- `src/grass_mil/interpretability/metaclusters/match.py`: Stage 2: match fold-clusters across folds into metaclusters.
- `src/grass_mil/interpretability/metaclusters/records.py`: Stage 1: cluster every fold model's held-out cells in its own space.
- `src/grass_mil/interpretability/metaclusters/transfer.py`: Stage 3: transfer every fold model's clustering to ALL regions in its own space.
- `src/grass_mil/interpretability/metaclusters_cli.py`: ``grass-mil-metaclusters``: cross-fold niche replication (cluster / match / transfer / basis).
- `src/grass_mil/interpretability/niches/__init__.py`: Niche discovery on a trained GRASS-MIL model's per-cell cache.
- `src/grass_mil/interpretability/niches/colocalisation.py`: Niche-niche co-localisation on the cell graph: log2(observed / expected) edge counts.
- `src/grass_mil/interpretability/niches/figures.py`: Figures of the niche report (matplotlib; UMAP optional).
- `src/grass_mil/interpretability/niches/margins.py`: Per-cell attention-weighted margins and the attention feature space.
- `src/grass_mil/interpretability/niches/pipeline.py`: Niche discovery on one model's cell cache.
- `src/grass_mil/interpretability/niches/tables.py`: Per-niche summary table (the preprint's cluster table).
- `src/grass_mil/interpretability/niches_cli.py`: ``grass-mil-niches``: niche discovery on one model's cell cache.
- `src/grass_mil/interpretability/pipeline.py`: Module.
- `src/grass_mil/interpretability/plugins/__init__.py`: Package marker / re-exports.
- `src/grass_mil/interpretability/plugins/base.py`: Module.
- `src/grass_mil/interpretability/plugins/builtin.py`: Module.
- `src/grass_mil/interpretability/plugins/registry.py`: Module.
- `src/grass_mil/interpretability/report_cli.py`: Module.
- `src/grass_mil/interpretability/reporting/__init__.py`: Package marker / re-exports.
- `src/grass_mil/interpretability/reporting/html.py`: Module.
- `src/grass_mil/interpretability/reporting/pdf.py`: Module.
- `src/grass_mil/interpretability/reporting/plotly_builders.py`: Module.
- `src/grass_mil/interpretability/reporting/render.py`: Module.
- `src/grass_mil/interpretability/reporting/sections.py`: Module.
- `src/grass_mil/interpretability/reporting/snapshot.py`: Module.
- `src/grass_mil/interpretability/tier2/__init__.py`: Package marker / re-exports.
- `src/grass_mil/interpretability/tier2/autocorrelation.py`: Global spatial autocorrelation (Moran's I).
- `src/grass_mil/interpretability/tier2/cell_level.py`: Cell-level tier-2 analyses, run within each niche.
- `src/grass_mil/interpretability/tier2/filtration.py`: Module.
- `src/grass_mil/interpretability/tier2/neighborhood.py`: Module.
- `src/grass_mil/interpretability/tier2/ripley.py`: Ripley's K/L and the bivariate cross-K.
- `src/grass_mil/interpretability/tier2/tissue_graph.py`: Module.

## Package: classical spatial statistics

- `src/grass_mil/spatial_stats/__init__.py`: Classical spatial statistics on the cell graph, per region, compared between groups.
- `src/grass_mil/spatial_stats/cellular_neighbourhoods.py`: Cellular neighbourhoods: k-NN window composition clustered with k-means.
- `src/grass_mil/spatial_stats/cli.py`: ``grass-mil-spatial-stats``: classical spatial statistics on a cell cache.
- `src/grass_mil/spatial_stats/contacts.py`: Contact fractions on the cell graph.
- `src/grass_mil/spatial_stats/cross_k.py`: Ripley cross-K between a reference group and partner groups, normalised by pi r^2.
- `src/grass_mil/spatial_stats/nhood_enrichment.py`: Neighbourhood enrichment z-scores on a cell graph (label permutation null).
- `src/grass_mil/spatial_stats/run.py`: Run the classical spatial statistics over a cell cache and compare groups.
- `src/grass_mil/spatial_stats/tests.py`: Per-region -> per-patient summaries and two-group tests with BH correction.

## Package: utilities

- `src/grass_mil/utils/__init__.py`: Package marker / re-exports.
- `src/grass_mil/utils/instantiators.py`: Module.
- `src/grass_mil/utils/logging_utils.py`: Module.
- `src/grass_mil/utils/pylogger.py`: Module.
- `src/grass_mil/utils/rich_utils.py`: Module.
- `src/grass_mil/utils/utils.py`: Module.

## Hydra configs

- `src/grass_mil/configs/__init__.py`: Package marker / re-exports.
- `src/grass_mil/configs/augmentation/bgrl_paper.yaml`: Degree-importance-weighted node/edge dropping plus Gaussian noise on the.
- `src/grass_mil/configs/augmentation/uniform.yaml`: Uniform edge dropping and feature masking, independent of graph structure.
- `src/grass_mil/configs/callbacks/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/callbacks/early_stopping.yaml`: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html.
- `src/grass_mil/configs/callbacks/mil_selection.yaml`: Model selection for MIL fine-tuning (preprint): keep the epoch with the best.
- `src/grass_mil/configs/callbacks/model_checkpoint.yaml`: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html.
- `src/grass_mil/configs/callbacks/model_summary.yaml`: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html.
- `src/grass_mil/configs/callbacks/none.yaml`: Hydra config: `none`.
- `src/grass_mil/configs/callbacks/pretrain_checkpoint.yaml`: Pretraining: a checkpoint every N optimizer steps (preprint: 5000), all kept,.
- `src/grass_mil/configs/callbacks/refit.yaml`: Refit on all development folds for a fixed number of epochs: no early.
- `src/grass_mil/configs/callbacks/rich_progress_bar.yaml`: https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.RichProgressBar.html.
- `src/grass_mil/configs/data/sampler/ego_graph_mil.yaml`: Ego-graph bags for MIL fine-tuning (preprint defaults).
- `src/grass_mil/configs/data/sampler/ego_graph_pretrain.yaml`: Ego-graph batches for BGRL pretraining (preprint "A3" defaults): roots drawn.
- `src/grass_mil/configs/data/sampler/identity.yaml`: Hydra config: `identity`.
- `src/grass_mil/configs/data/sampler/shadow_custom.yaml`: Hydra config: `shadow_custom`.
- `src/grass_mil/configs/data/sampler/shadow_native.yaml`: Hydra config: `shadow_native`.
- `src/grass_mil/configs/data/spatial_omics.yaml`: Hydra config: `spatial_omics`.
- `src/grass_mil/configs/debug/default.yaml`: default debugging setup, runs 1 full epoch.
- `src/grass_mil/configs/debug/fdr.yaml`: runs 1 train, 1 validation and 1 test step.
- `src/grass_mil/configs/debug/limit.yaml`: uses only 1% of the training data and 5% of validation/test data.
- `src/grass_mil/configs/debug/overfit.yaml`: overfits to 3 batches.
- `src/grass_mil/configs/debug/profiler.yaml`: runs with execution time profiling.
- `src/grass_mil/configs/eval.yaml`: Hydra config: `eval`.
- `src/grass_mil/configs/experiment/grass_mil_pretrain_a3.yaml`: GRASS-MIL encoder pretraining (preprint "A3"):
- `src/grass_mil/configs/extract.yaml`: grass-mil-extract: per-cell cache (embeddings, instance logits, attention) of.
- `src/grass_mil/configs/extras/default.yaml`: disable python warnings if they annoy you.
- `src/grass_mil/configs/hydra/default.yaml`: https://hydra.cc/docs/configure_hydra/intro/.
- `src/grass_mil/configs/inference/aggregation/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/inference/embeddings/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/inference/interpretability/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/inference/metrics/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/inference/predict.yaml`: Hydra config: `predict`.
- `src/grass_mil/configs/interpretability/clustering/agglomerative.yaml`: Hydra config: `agglomerative`.
- `src/grass_mil/configs/interpretability/clustering/dbscan.yaml`: Hydra config: `dbscan`.
- `src/grass_mil/configs/interpretability/clustering/hdbscan.yaml`: Hydra config: `hdbscan`.
- `src/grass_mil/configs/interpretability/clustering/kmeans.yaml`: Hydra config: `kmeans`.
- `src/grass_mil/configs/interpretability/clustering/optics.yaml`: Hydra config: `optics`.
- `src/grass_mil/configs/interpretability/clustering/spectral.yaml`: Hydra config: `spectral`.
- `src/grass_mil/configs/interpretability/plugins/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/interpretability/plugins/full.yaml`: Every builtin analysis enabled.
- `src/grass_mil/configs/interpretability/reduction/pca.yaml`: Hydra config: `pca`.
- `src/grass_mil/configs/interpretability/reduction/tsne.yaml`: Hydra config: `tsne`.
- `src/grass_mil/configs/interpretability/reduction/umap.yaml`: Hydra config: `umap`.
- `src/grass_mil/configs/interpretability/report.yaml`: Hydra config: `report`.
- `src/grass_mil/configs/local/.gitkeep`: Placeholder for untracked machine-local configs.
- `src/grass_mil/configs/logger/aim.yaml`: https://aimstack.io/.
- `src/grass_mil/configs/logger/comet.yaml`: https://www.comet.ml.
- `src/grass_mil/configs/logger/csv.yaml`: csv logger built in lightning.
- `src/grass_mil/configs/logger/many_loggers.yaml`: train with many loggers at once.
- `src/grass_mil/configs/logger/mlflow.yaml`: https://mlflow.org.
- `src/grass_mil/configs/logger/neptune.yaml`: https://neptune.ai.
- `src/grass_mil/configs/logger/tensorboard.yaml`: https://www.tensorflow.org/tensorboard/.
- `src/grass_mil/configs/logger/wandb.yaml`: https://wandb.ai.
- `src/grass_mil/configs/loocv.yaml`: Hydra config: `loocv`.
- `src/grass_mil/configs/metaclusters.yaml`: grass-mil-metaclusters: cross-fold niche replication.
- `src/grass_mil/configs/model/attention/gated.yaml`: Plain gated attention (Ilse et al. 2018), one attention channel per class.
- `src/grass_mil/configs/model/attention/gated_projected.yaml`: Projected gated attention (GRASS-MIL default): Linear + LeakyReLU projection,.
- `src/grass_mil/configs/model/bgrl_a3.yaml`: Hydra config: `bgrl_a3`.
- `src/grass_mil/configs/model/bgrl_module.yaml`: Hydra config: `bgrl_module`.
- `src/grass_mil/configs/model/encoder/gat.yaml`: Hydra config: `gat`.
- `src/grass_mil/configs/model/encoder/gcn.yaml`: Hydra config: `gcn`.
- `src/grass_mil/configs/model/encoder/gin.yaml`: Small GIN encoder without edge features: a lightweight default for smoke runs.
- `src/grass_mil/configs/model/encoder/gine.yaml`: Hydra config: `gine`.
- `src/grass_mil/configs/model/encoder/gine_length.yaml`: GRASS-MIL encoder (preprint defaults): 4 GINE layers of width 256 conditioned.
- `src/grass_mil/configs/model/encoder/graphsage.yaml`: Hydra config: `graphsage`.
- `src/grass_mil/configs/model/grass_mil.yaml`: Hydra config: `grass_mil`.
- `src/grass_mil/configs/model/supervised_module.yaml`: Hydra config: `supervised_module`.
- `src/grass_mil/configs/nested_cv.yaml`: grass-mil-nested-cv: the preprint's nested, patient-grouped cross-validation.
- `src/grass_mil/configs/niches.yaml`: grass-mil-niches: niche discovery on a trained model's cell cache.
- `src/grass_mil/configs/optim/adamw.yaml`: Hydra config: `adamw`.
- `src/grass_mil/configs/paths/default.yaml`: path to root directory.
- `src/grass_mil/configs/probe.yaml`: grass-mil-probe: fit the linear probe head on a frozen encoder.
- `src/grass_mil/configs/regime/linear_ftall.yaml`: Preprint "ftALL": encoder fine-tuned end to end (lr x1.0), linear head.
- `src/grass_mil/configs/regime/linear_ftall_reg.yaml`: Preprint "ftALL, lr x0.3 + regularisation" (the STAS model of record):
- `src/grass_mil/configs/regime/linear_probehead.yaml`: Preprint "probehead": frozen encoder, LINEAR head pre-fitted by grass-mil-probe.
- `src/grass_mil/configs/regime/mil.yaml`: GRASS-MIL default regime: BGRL-initialised encoder fine-tuned end to end,.
- `src/grass_mil/configs/regime/mil_frozen.yaml`: Frozen encoder (learning-rate scale 0, kept in eval mode); MLP head and.
- `src/grass_mil/configs/scheduler/cosine_epoch.yaml`: Hydra config: `cosine_epoch`.
- `src/grass_mil/configs/scheduler/cosine_mil_step.yaml`: Cosine annealing per optimizer step over the whole run (preprint): with.
- `src/grass_mil/configs/scheduler/cosine_step.yaml`: Hydra config: `cosine_step`.
- `src/grass_mil/configs/scores.yaml`: grass-mil-scores: evaluation statistics for held-out scores.
- `src/grass_mil/configs/spatial_stats.yaml`: grass-mil-spatial-stats: classical spatial statistics per region, compared.
- `src/grass_mil/configs/task/finetune_mean.yaml`: Hydra config: `finetune_mean`.
- `src/grass_mil/configs/task/finetune_mil.yaml`: Hydra config: `finetune_mil`.
- `src/grass_mil/configs/task/finetune_survival.yaml`: Cox survival with MIL attention pooling. The head emits a single scalar.
- `src/grass_mil/configs/task/pretrain_bgrl.yaml`: Hydra config: `pretrain_bgrl`.
- `src/grass_mil/configs/task/pretrain_bgrl_a3.yaml`: BGRL pretraining, preprint "A3" recipe: AdamW 5e-5 / weight decay 1e-5,.
- `src/grass_mil/configs/train.yaml`: specify here default configuration.
- `src/grass_mil/configs/trainer/cpu.yaml`: Hydra config: `cpu`.
- `src/grass_mil/configs/trainer/ddp.yaml`: Hydra config: `ddp`.
- `src/grass_mil/configs/trainer/ddp_sim.yaml`: Hydra config: `ddp_sim`.
- `src/grass_mil/configs/trainer/default.yaml`: Hydra config: `default`.
- `src/grass_mil/configs/trainer/gpu.yaml`: Hydra config: `gpu`.
- `src/grass_mil/configs/trainer/gpu_amp.yaml`: Hydra config: `gpu_amp`.
- `src/grass_mil/configs/trainer/mps.yaml`: Hydra config: `mps`.

## Documentation

- `docs/capabilities_and_modes.md`: Capabilities And Modes.
- `docs/dataset_preparation.md`: Dataset Preparation Guide.
- `docs/hyperparameter_reference.md`: Hyperparameter Reference.
- `docs/inference_utility_suite.md`: Inference Utility Suite.
- `docs/interpretability_extension_guide.md`: Interpretability Extension Guide.
- `docs/interpretability_suite.md`: Interpretability Suite.
- `docs/model_component_contracts.md`: Model Component Contracts.
- `docs/repository_map.md`: Repository Map.
- `docs/workflows_train_infer.md`: Training And Inference Workflows.

## Vignettes

- `vignettes/01_anndata_to_spatial_graphs.ipynb`: Vignette 1: AnnData -> manifest -> cellular graphs.
- `vignettes/02_pretraining_and_niches.ipynb`: Vignette 2: BGRL pretraining, fine-tuning, instance export, interpretability report.
- `vignettes/README.md`: Index of the worked examples on the public MIBI-TOF dataset.

## Tests

- `tests/__init__.py`: Package marker / re-exports.
- `tests/configs/test_component_instantiation.py`: Compose the model/encoder and model/attention groups through Hydra and build them.
- `tests/configs/test_task_preset_composition.py`: Compose task presets through Hydra the way the CLI does.
- `tests/conftest.py`: Shared pytest fixtures (Hydra configs on a synthetic datamodule).
- `tests/contracts/test_model_component_contracts.py`: Tests for model component contracts.
- `tests/data/test_datamodule_sampler_exposure.py`: Tests for datamodule sampler exposure.
- `tests/data/test_loader_flags.py`: Tests for loader flags.
- `tests/data/test_precompute_manifest_regions.py`: Tests for precompute manifest regions.
- `tests/data/test_precompute_reducer_modes.py`: Tests for precompute reducer modes.
- `tests/data/test_samplers.py`: Tests for samplers.
- `tests/data/test_seed_sampling.py`: Contracts for interior seed restriction and the ego-radius cutoff.
- `tests/eval/test_aggregation.py`: Tests for aggregation.
- `tests/eval/test_collectors.py`: Tests for collectors.
- `tests/eval/test_embeddings.py`: Tests for embeddings.
- `tests/eval/test_interpretability_export.py`: Tests for interpretability export.
- `tests/eval/test_io.py`: Tests for io.
- `tests/eval/test_metrics.py`: Tests for metrics.
- `tests/eval/test_predict_cli.py`: Tests for predict cli.
- `tests/helpers/__init__.py`: Package marker / re-exports.
- `tests/helpers/config_paths.py`: Locate the packaged Hydra config tree.
- `tests/helpers/package_available.py`: Tests for package available.
- `tests/helpers/run_if.py`: Adapted from:
- `tests/helpers/run_sh_command.py`: Tests for run sh command.
- `tests/helpers/synthetic_cohort.py`: Synthetic cohort on disk (regions, labels, folds) with a planted niche, used by the phase tests.
- `tests/helpers/synthetic_datamodule.py`: In-memory bag datamodule for fast training tests.
- `tests/interpretability/test_attribution.py`: Contracts for the exact additive attribution.
- `tests/interpretability/test_cell_level.py`: Cell-level tier-2 analyses.
- `tests/interpretability/test_core.py`: Tests for core.
- `tests/interpretability/test_fidelity_phase6.py`: Phase 6: niche discovery (cosine HDBSCAN, relattn margins, tables, co-localisation, pipeline).
- `tests/interpretability/test_fidelity_phase7.py`: Phase 7: cross-fold metaclusters, transfer and shared basis.
- `tests/interpretability/test_niches.py`: Niche naming, the Background label, and composition ordering.
- `tests/interpretability/test_pipeline_and_report.py`: Tests for pipeline and report.
- `tests/interpretability/test_plugins_and_tier2.py`: Tests for plugins and tier2.
- `tests/interpretability/test_spatial_statistics.py`: Contracts for Moran's I, Ripley's cross-L, and cross-space agreement.
- `tests/interpretability/test_tissue_graph.py`: Tests for tissue graph.
- `tests/interpretability/test_transfer.py`: Tests for transfer.
- `tests/models/test_attention.py`: Tests for attention.
- `tests/models/test_augmentations.py`: Contracts for the BGRL view generators.
- `tests/models/test_backbones.py`: Tests for backbones.
- `tests/models/test_bagging.py`: Bag-pooling contracts, including the attribution identity.
- `tests/models/test_fidelity_phase1.py`: Phase 1 fidelity: graph edge features, encoder epsilon/init, attention init and space.
- `tests/models/test_fidelity_phase2.py`: Phase 2 fidelity: region bags, root sampling, parameter groups, schedules, class weights,.
- `tests/models/test_fidelity_phase3.py`: Phase 3: explicit splits, probe pre-fit, nested cross-validation.
- `tests/models/test_fidelity_phase4.py`: Phase 4: BGRL "A3" pretraining fidelity (merged-unit sampling, momentum schedule,.
- `tests/models/test_fidelity_phase5.py`: Phase 5: cell cache extraction, patient aggregation, evaluation statistics.
- `tests/models/test_losses.py`: Tests for losses.
- `tests/models/test_node_embeddings.py`: Contracts for the encoder's input embeddings.
- `tests/models/test_optimization.py`: Tests for optimization.
- `tests/models/test_runtime.py`: Tests for runtime.
- `tests/models/test_ssl.py`: Tests for ssl.
- `tests/spatial_stats/__init__.py`: Package marker / re-exports.
- `tests/spatial_stats/test_fidelity_phase8.py`: Phase 8: classical spatial statistics.
- `tests/test_configs.py`: Tests for configs.
- `tests/test_eval.py`: Tests for eval.
- `tests/test_loocv.py`: Tests for loocv.
- `tests/test_packaging.py`: Guards on the installable-package contract.
- `tests/test_spatial_omics_datamodule.py`: Tests for spatial omics datamodule.
- `tests/test_train.py`: Tests for train.
- `tests/test_train_regimes.py`: Tests for train regimes.

## Other

- `data/.gitkeep`: Tracked file.
- `logs/.gitkeep`: Tracked file.

## Generated at runtime (untracked)

- `logs/`: Hydra run directories (checkpoints, `metrics_per_epoch.jsonl`, predictions, caches, reports).
- `data/`: raw manifests and processed graphs (`processed_index.json`, `metadata.json`, `graphs/`).
- `vignettes/vignette_work/`: data, graphs and runs produced by the vignettes.
