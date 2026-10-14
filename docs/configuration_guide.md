# Configuration Guide

Everything in `grass-mil` is configured through Hydra: a command composes a
tree of YAML files from `src/grass_mil/configs/`, and every leaf can be
overridden on the command line as `key.path=value`. This page explains how the
tree is composed, what the groups are, and how to change the things people
usually change. The exhaustive key-by-key tables are in the
[hyperparameter reference](hyperparameter_reference.md).

## 1. How a command is composed

Each CLI has a root config (`train.yaml`, `nested_cv.yaml`, `niches.yaml`, ...).
A root config lists *groups* in its `defaults`, in order; later entries override
earlier ones, and command-line overrides win over everything:

```
train.yaml defaults:  data -> model -> optim -> scheduler -> task -> callbacks -> trainer -> paths
                      -> experiment (null) -> regime (null) -> debug (null)
```

A *preset* is a file in a group. `regime=mil` is the file
`configs/regime/mil.yaml`; it is "`@package _global_`", so it may set keys
anywhere in the tree and may swap other groups (`override /model: grass_mil`,
`override /task: finetune_mil`, `override /scheduler: cosine_mil_step`,
`override /callbacks: mil_selection`, `override /data/sampler: ego_graph_mil`).
That is how one word selects a whole training recipe. An `experiment=` file is
the same mechanism, meant for *your* settings (the cohort template in
`configs/experiment/example_cohort.yaml` is one).

Precedence, lowest to highest: group file defaults -> `experiment=` -> `regime=`
-> command-line `key=value`. Two consequences:

- a regime's values (e.g. `data.sampler.runtime.subsample_count=256`) beat the
  same key in your experiment file; override it on the command line if needed;
- a key that does not exist yet needs `+key=value` (e.g.
  `+trainer.fast_dev_run=true`); an existing key is set with `key=value`.

Useful Hydra flags: `--cfg job` prints the fully composed config and exits;
`--info defaults` shows which files were merged; `-m key=a,b` runs a sweep.

The composed config is also written to the run directory
(`<output_dir>/.hydra/config.yaml`) so every run is reproducible from its
folder.

## 2. The groups and their presets

| Group | Presets (default first) | What it decides |
|---|---|---|
| `data` | `spatial_omics` | inputs, graph building, labels, splits, tiling (section 3) |
| `data/sampler` | `shadow_custom`, `ego_graph_mil` (regimes), `ego_graph_pretrain` (pretraining), `identity`, `shadow_native` | how ego-graphs are drawn (section 4) |
| `model` | `supervised_module`, `grass_mil` (regimes), `bgrl_module`, `bgrl_a3` (pretraining) | the module: encoder + head + attention, or BGRL |
| `model/encoder` | `gin`, `gine_length` (GRASS-MIL), `gine`, `gcn`, `gat`, `graphsage` | encoder architecture (section 5) |
| `model/attention` | `gated_projected`, `gated` | attention net (section 6) |
| `task` | `finetune_mean`, `finetune_mil` (regimes), `finetune_survival`, `pretrain_bgrl`, `pretrain_bgrl_a3` | objective, bag definition, optimisation knobs, augmentation (section 7) |
| `regime` | `mil`, `mil_frozen`, `linear_probehead`, `linear_ftall`, `linear_ftall_reg`, `pretrain_a3` | the preprint recipes: fine-tuning (section 7) and pretraining (section 9) |
| `optim` | `adamw` | optimiser |
| `scheduler` | `cosine_epoch`, `cosine_mil_step` (regimes), `cosine_step` (pretraining) | learning-rate schedule |
| `callbacks` | `default`, `mil_selection` (regimes), `refit`, `pretrain_checkpoint`, `none` | checkpointing, early stopping (section 8) |
| `trainer` | `default` (cpu), `gpu`, `gpu_amp`, `mps`, `ddp`, `ddp_sim` | device, epochs, precision (section 11) |
| `experiment` | `example_cohort` (template), `grass_mil_pretrain_a3` (alias of `regime=pretrain_a3`) | your cohort file goes here; one experiment file per command |
| `logger` | `csv`, `tensorboard`, `wandb`, `mlflow`, ... | experiment tracking (off by default) |
| `debug` | `fdr`, `limit`, `overfit`, `profiler` | quick runs |

The analysis CLIs have flat root configs rather than groups: `probe.yaml`,
`nested_cv.yaml`, `extract.yaml`, `scores.yaml`, `niches.yaml`,
`metaclusters.yaml`, `spatial_stats.yaml`, `inference/predict.yaml`. The ones
that build a model (`probe`, `nested_cv`, `extract`, `predict`, `eval`)
include the `train.yaml` tree, so `experiment=`, `regime=` and every `data.*`
/ `model.*` key work there too.

## 3. Inputs, features and graph (`data.*`)

| What | Key | Notes |
|---|---|---|
| where the regions are | `data.raw_manifest_path`, `data.processed_dir` | manifest columns: `sample_id,input_path,input_type[,region_id,polygons_path]`; relative input paths resolve against the manifest |
| what a unit is | `data.sample_unit=full|tile` (+ `data.tiling.{tile_size_um,stride_um,min_cells}`) | `full` = one graph per region |
| coordinate unit | `data.coord_scale_um` | multiplies the input coordinates; all distances below are in the resulting unit |
| cell-type column | `data.csv.categorical_label_columns=[cell_type]`, `data.categorical_features.include_labels=[cell_type]` | the vocabulary is learnt from the data and stored in `processed/metadata.json` |
| continuous node features | `data.csv.molecular_columns=[SIZE]` (`[]` for none; `null` = every non-coordinate numeric column) | h5ad: `data.h5ad.molecular_features`, `data.h5ad.molecular_layer` |
| feature reduction | `data.feature_reducer.name=identity|pca|...`, `fit_mode` | identity by default |
| edges | `data.graph_builder.name=delaunay|knn`, `kwargs.edge_features=[log_distance,neighbor]`, `kwargs.neighbor_cutoff_um` | `log_distance` = `clip((ln d - 2)/3, 0, 1)` (`log_lower`, `log_upper` adjustable); `distance` = raw length |
| labels | `data.graph_labels.{label_file,id_column,tasks,scope,patient_column}` | `scope` = `region|patch|sample`; `patient_column` groups, never a target |
| splits | `data.split.explicit.{train,val,test,id_kind}` (lists or files) or the persisted `data.split.{split_by,train_val_test_split}` | nested CV sets explicit splits per fold |
| which cells may be roots | `data.sampler.runtime.interior_seeds.*` | exclude the tissue rim |

Which node-feature layout reaches the encoder: `x` = continuous columns
(after the reducer), `categorical_codes` = the cell-type code; the encoder sums
a learnt cell-type embedding and a linear projection of `x`. A cohort with cell
types only has `x` of width 0 and works unchanged.

## 4. Ego-graphs (`data.sampler.runtime.*`)

| What | Key (default in `ego_graph_mil`) |
|---|---|
| hops | `depth` (4) |
| neighbours kept per hop | `num_neighbors` (-1 = all) |
| radius cap | `ego_radius.enabled` (false), `radius_per_hop`, `radius_offset` |
| roots per region per epoch (bag size) | `subsample_count` (256; `subsample_fraction` as an alternative) |
| uniform or cell-type-weighted roots | `proportional_root_sampling` (false), `property_name`, `weight_mode` |
| one bag per region | `unit_as_bag` (true) |
| validation subset | `data.val_sampler.runtime.subsample_count` (1024) with `fixed_per_unit_seed` (0) |
| test / prediction | `data.test_sampler` (null = every cell once) |
| pretraining batches | `ego_graph_pretrain`: `merge_units=true`, `subgraph_batch_size=64` |

Example, 3-hop ego-graphs capped at 150 units of distance, 128 roots per bag:
`data.sampler.runtime.depth=3 data.sampler.runtime.ego_radius.enabled=true data.sampler.runtime.ego_radius.radius_per_hop=50 data.sampler.runtime.subsample_count=128`.

## 5. Encoder (`model.encoder.*`)

Selected by `model/encoder=gine_length` (the GRASS-MIL default inside
`model=grass_mil` and `model=bgrl_a3`).

| What | Key (default) |
|---|---|
| depth and width | `num_layers` (4), `hidden_dim` (256), `out_dim` (256) |
| convolution | `conv_type` (`gine`; `gin`, `gcn`, `gat`, `graphsage`), `train_eps` (true) |
| edge conditioning | `use_edge_attr` (true), `edge_feature_index` (0 = the `log_distance` column), `edge_attr_dim` (1) |
| normalisation, activation, dropout | `norm` (`layernorm`), `act` (`relu`), `dropout` (0 when fine-tuning, 0.1 in `bgrl_a3`) |
| readout | `pooling` (`max`), `jk` (`last`) |
| cell-type embedding | `categorical_embedding.{label,reserve_unassigned,unassigned_label}`; `input_init` (`xavier`) |

`model.encoder.num_layers=3 model.encoder.hidden_dim=128 model.encoder.out_dim=128`
gives a smaller encoder; the head and attention widths follow `out_dim`
automatically. A pretrained checkpoint must have been trained with the same
encoder settings.

## 6. Attention and instance head

| What | Key (default) |
|---|---|
| attention net | `model/attention=gated_projected` (`gated` = no projection); `model.attention.projection_dim` (32), `hidden_dim` (8), `dropout` (true), `init` (`kaiming`) |
| channels | `model.attention.n_classes` (2, one per class; set with `model.graph_head.output_dim` for more classes) |
| instance head | `model.graph_head.num_layers` (2 = MLP; 1 = linear, required by the `linear_*` regimes), `hidden_dim`, `dropout` |
| pooling | always logit-space: `L_c = sum_i A[i,c] * l[i,c]` (`task.aggregation=mil_attention`); `task.aggregation=mean` averages instance logits instead |

## 7. Training regimes and optimisation (`regime=`, `task.*`, `optim.*`)

Pick a regime, then adjust:

| What | Key (where the regimes set it) |
|---|---|
| learning rate, weight decay | `optim.lr` (1e-4), `optim.weight_decay` (1e-3; 1e-2 in `linear_ftall_reg`) |
| per-component rates | `task.optimization.encoder_lr_scale` (1 / 0.3 / 0 = frozen), `head_lr_scale`, `attention_lr_scale` (2), `head_weight_decay` |
| head pre-fit | `task.head_init_path` (`auto` in `linear_*`: fitted by nested CV; or a `grass-mil-probe` file), `task.freeze_head` |
| class weights | `task.class_weights` (`auto_train`, a list, or null) |
| augmentation while fine-tuning | `task.finetune_augmentation.{drop_edge_p,drop_feat_p}` (0.05 in `linear_ftall_reg`) |
| accumulation and clipping | `task.region_accumulation.hyperbatch_size` (12), `task.grad_clip_norm` (1.0) |
| schedule | `scheduler=cosine_mil_step` with `scheduler.T_max=auto` (whole run) and `task.scheduler_interval=step`; `scheduler.eta_min` |
| epochs | `trainer.max_epochs` (25), `trainer.min_epochs` (10, before early stopping may act) |
| what a bag is | `task.bag_key` (`region_id`; `patch_id` for tiles) |
| probe settings | `probe.{roots_per_unit,seed,C,class_weight,max_iter}` and `nested_cv.probe.*` |

To define your own recipe, copy `configs/regime/mil.yaml` to
`configs/regime/my_regime.yaml`, edit, and run `regime=my_regime`.

## 8. Selection and checkpoints (`callbacks=`)

`mil_selection` (regimes): checkpoint on `val/auroc`, early stopping on
`val/loss_region` with `callbacks.early_stopping.patience` (8). `refit`: no
selection, last weights. `pretrain_checkpoint`: every
`callbacks.model_checkpoint.every_n_train_steps` (5000). Any Lightning
callback argument is reachable, e.g.
`callbacks.model_checkpoint.save_top_k=3`.

Logged per epoch and written to `metrics_per_epoch.jsonl`: `val/loss_region`,
`val/auroc`, `val/bal_acc`, `val/acc`, `train/loss`.

## 9. Pretraining (`regime=pretrain_a3`)

| What | Key (default) |
|---|---|
| length | `task.total_steps` (200000), `task.warmup_steps` and `task.lr_warmup.warmup_steps` (1000) |
| optimiser | `optim.lr` (5e-5), `optim.weight_decay` (1e-5) |
| target network | `task.momentum` (0.99) -> `task.momentum_final` (1.0) |
| views | `task.augmentation.{mode,mu,p_lambda,feature_noise_std,feature_noise_columns}`; `augmentation=uniform` for plain edge/feature dropout |
| encoder dropout | `model.encoder.dropout` (0.1) |
| batches | `data.sampler.runtime.subgraph_batch_size` (64), `depth` (4) |
| data | the regime sets `data.split.train_val_test_split=[1,0,0]`: every region, no labels needed |

Any `model.encoder.*` change here must be repeated when fine-tuning.

## 10. Nested cross-validation (`nested_cv.*`)

| What | Key (default) |
|---|---|
| folds and labels | `nested_cv.folds_file`, `nested_cv.{label_file,id_column,label_column,patient_column}` (taken from `data.graph_labels` when unset), `nested_cv.id_kind=region|sample|patch` |
| inner selection | `nested_cv.inner_k` (3), `inner_seed` (0), `selection_metric` (`val/auroc`), `selection_mode`, `epoch_reduce` (`median`), `min_refit_epochs` |
| which outer folds | `nested_cv.outer_folds=[2]` (null = all), `nested_cv.resume` (true) |
| probes | `nested_cv.probe.{enabled,cache_dir,roots_per_unit,...}` |
| patient score | `nested_cv.patient_aggregation` (`mean`) |

## 11. Devices, precision, runtime

`trainer=gpu` (one GPU), `trainer.devices=2 trainer.strategy=ddp`,
`trainer.precision=16-mixed` (the regimes' default; `32` on CPU),
`data.num_workers` (0 is usually fastest for ego-graph sampling, which creates a
loader per region), `seed=0` for reproducible runs, `paths.output_dir=...` to
choose where a run writes (default `logs/<task_name>/runs/<timestamp>`).

## 12. Extraction, niches, metaclusters, statistics

These read a cell cache, not the training config, so their keys are flat:

| CLI | Keys people change |
|---|---|
| `grass-mil-extract` | `extract.nested_cv_summary` or `+extract.models.<name>=<ckpt>`, `extract.unit_id`, `extract.regions`, `extract.batch_size`, `extract.output_dir` (model composed from the same `experiment=`/`regime=` as training) |
| `grass-mil-niches` | `regions=held_out|all|<list>`, `space=gnn|attention`, `reduction.{method,n_components}`, `clustering.{method,n_clusters,params.min_cluster_size,params.min_samples,params.cluster_selection_method}`, `sample_size`, `l2_normalise`, `assign_k`, `present_threshold`, `n_boot`, `umap.*`, `figures` |
| `grass-mil-metaclusters` | the same clustering keys, `match.{distance,threshold,scan,min_folds}`, `transfer.{min_cells,present_threshold}`, `basis.{k,space_model,sample_size}`, `stage` |
| `grass-mil-scores` | `scores`, `fold_column`, `patient_column`, `patient_level`, `standardise`, `strata`, `operating_rules`, `n_boot` |
| `grass-mil-spatial-stats` | `group_map`, `reference`, `partners`, `infiltrating`, `analyses`, `nhood.{n_perms,backend}`, `cn.{window_k,n_cn}`, `crossk.radii`, `min_per_group` |

Lists on the command line are written `key=[a,b]`; dictionaries
`key='{a: b, c: d}'`; `null` clears a value.

## 13. Reading the defaults

`src/grass_mil/configs/` is the source of truth; every file is commented.
Start from `regime/mil.yaml` (what a fine-tuning run composes),
`task/finetune_mil.yaml` (every MIL knob), `data/sampler/ego_graph_mil.yaml`
(ego-graphs), `model/encoder/gine_length.yaml` (encoder) and
`regime/pretrain_a3.yaml` (pretraining). The
[hyperparameter reference](hyperparameter_reference.md) lists every key with
its type, valid values and an example override.
