# Configuration

Every command is a [Hydra](https://hydra.cc) application whose configuration
ships with the package (`src/grass_mil/configs`). Any key can be set on the
command line (`key=value`, lists as `[a,b]`), and `--cfg job` prints the
composed configuration without running anything:

```bash
grass-mil-train experiment=my_cohort regime=finetune_reg --cfg job
```

## Groups

`grass-mil-train` (and `nested-cv`, `extract`, `predict`, which start from it)
composes these groups:

| Group | Options | Selects |
|---|---|---|
| `data` | `spatial_omics` | inputs, units, graphs, labels, splits, sampling |
| `data/sampler` | `ego_graph`, `ego_graph_pretrain` | ego-graph sampling (set by the regime) |
| `model` | `grass_mil`, `bgrl` | the module (`bgrl` is set by `regime=pretrain`) |
| `task` | `classification`, `regression`, `survival` | targets, objective, selection metric |
| `regime` | `finetune`, `finetune_reg`, `attention_only`, `mlp_head`, `mlp_head_frozen`, `pretrain` | the training procedure |
| `experiment` | your cohort file | paths and columns of one cohort |
| `trainer` | `default` (CPU), `cpu`, `gpu` (fp16 mixed precision) | the Lightning trainer |
| `logger` | `csv`, `tensorboard` | optional metric loggers |
| `optim`, `scheduler`, `callbacks` | one option each | AdamW, cosine schedule, selection callbacks |

Later entries win: a cohort file sets data and labels, the regime sets the
procedure, and command-line keys override both.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `seed` | `0` | seeds every random number generator |
| `train`, `test` | `true` | fit; score the test split with the selected checkpoint |
| `ckpt_path` | `null` | resume an interrupted run (prediction: the model to score) |
| `print_config` | `false` | log the composed configuration |
| `paths.data_dir`, `paths.log_dir` | `./data`, `./logs` | relative to the working directory |
| `paths.output_dir` | per run | `logs/<task_name>/runs/<date>` |

## `data`

| Key | Default | Meaning |
|---|---|---|
| `manifest_path` | `${paths.data_dir}/manifest.csv` | see [data preparation](data_preparation.md) |
| `processed_dir` | `${paths.data_dir}/processed` | built graphs |
| `force_precompute` | `false` | rebuild the graphs |
| `unit` | `region` | `region`, `tile` or `polygon` |
| `coord_scale_um` | `1.0` | coordinate multiplier |
| `min_cells` | `10` | smaller units are dropped |
| `tiling.size`, `tiling.stride` | `200`, `200` | tiles, coordinate units |
| `input.coord_columns` | `[x, y]` | |
| `input.obsm_key` | `spatial` | h5ad coordinates (`null`: `obs` columns) |
| `input.reduced_dim` | `null` | SCE coordinates (`null`: `colData` columns) |
| `input.cell_id_column` | `null` | |
| `input.cell_type_column` | `cell_type` | `null`: no cell types |
| `input.features` | `all` | `all` (numeric columns), a list of names, or `null` |
| `input.h5ad_layer`, `input.sce_assay`, `input.sep` | `null`, `null`, `","` | |
| `graph.edge_features` | `[log_distance, neighbor]` | also `distance` |
| `graph.neighbor_cutoff` | `20.0` | contact distance |
| `graph.log_distance_bounds` | `[2.0, 5.0]` | ln-length range mapped to [0, 1] |
| `feature_reducer.name` | `identity` | `pca` |
| `feature_reducer.n_components`, `.fit_on`, `.state_from` | `32`, `all`, `null` | `fit_on`: `all` or `train`; see data preparation |
| `labels.file`, `.id_column`, `.patient_column` | `null`, `region_id`, `null` | |
| `labels.columns` | `task.target_columns` | |
| `labelled_only` | `true` | skip unlabelled units (pretraining: `false`) |
| `split.ratios`, `split.seed` | `[0.8, 0.1, 0.1]`, `0` | random split, grouped by patient |
| `split.train`, `.val`, `.test` | `null` | explicit ids (lists or files) |
| `val_roots_per_unit` | `1024` | cells scored per validation unit |
| `num_workers`, `pin_memory` | `0`, `false` | |

### `data.sampler`

| Key | `ego_graph` | Meaning |
|---|---|---|
| `depth` | `4` | hops (keep equal to `model.encoder.num_layers`) |
| `num_neighbors` | `-1` | neighbours per node and hop (`-1`: all) |
| `roots_per_unit` | `256` | cells per bag and epoch (`null`: all) |
| `batch_size` | `64` | ego-graphs per batch when scoring and pretraining |
| `merge_units` | `false` | pool the cells of all units (pretraining) |
| `root_weighting` | `null` | `inverse` / `sqrt_inverse` cell-type frequency |
| `interior_roots.enabled` | `false` | only cells at least `n_hops` hops from the convex hull |
| `ego_radius.enabled` | `false` | drop nodes farther than `radius_per_hop * depth + radius_offset` |

## `model`

| Key | Default | Meaning |
|---|---|---|
| `encoder.hidden_dim`, `.out_dim` | `256`, `256` | |
| `encoder.num_layers` | `4` | GINE layers |
| `encoder.dropout` | `0.0` (`0.1` in `bgrl`) | |
| `encoder.norm`, `.jk`, `.act`, `.pooling` | `layernorm`, `last`, `relu`, `max` | |
| `encoder.mlp_ratio`, `.train_eps` | `2`, `true` | GINE MLP width, learnable epsilon |
| `encoder.edge_feature` | `0` | edge-feature column the convolutions see (`null`: all) |
| `encoder.input_init` | `xavier` | |
| `encoder.unassigned_cell_type` | `null` | masked-node token (`null`: an extra row) |
| `head.num_layers` | `1` | `1` linear, `2` MLP |
| `head.hidden_dim`, `.dropout`, `.act` | `null`, `0.0`, `leaky_relu` | |
| `attention.projection_dim`, `.hidden_dim` | `32`, `8` | |
| `attention.dropout`, `.init` | `0.25`, `kaiming` | |
| `init_from_ckpt` | `null` | encoder weights (a pretraining checkpoint) |

## `task`

| Key | Default | Meaning |
|---|---|---|
| `target_type` | `classification` | `regression`, `survival` |
| `target_columns` | `null` | label columns (`null`: the first one) |
| `num_classes` | `2` | classification |
| `class_weights` | `auto` | inverse training frequency, a list, or `null` |
| `loss`, `huber_delta` | `mse`, `1.0` | regression |
| `cox_regularization` | `0.0` | survival |
| `selection.metric`, `.mode`, `.patience` | `val/auroc`, `max`, `8` | checkpoint selection and early stopping |

## `finetune` (fine-tuning regimes)

| Key | `finetune` | Meaning |
|---|---|---|
| `lr_scales.encoder/head/attention` | `1.0`, `0.01`, `2.0` | multipliers of `optim.lr`; `0` freezes |
| `head_weight_decay` | `1e-3` | (`null`: `optim.weight_decay`) |
| `head_init` | `probe` | or `random` |
| `hyperbatch_size` | `12` | units per optimizer step |
| `grad_clip_norm` | `1.0` | |
| `warmup_steps` | `0` | linear learning-rate warmup |
| `schedule_epochs` | `null` | epochs the learning-rate schedule spans (`null`: `trainer.max_epochs`; cross-validation refits set it) |
| `augmentation.drop_edge_p`, `.mask_node_p` | `0.0`, `0.0` | edge drop per direction; node masking |
| `probe.roots_per_unit`, `.C`, `.class_weight` | `96`, `1.0`, `balanced` | `C`: inverse L2 penalty of the probe |
| `optim.lr`, `optim.weight_decay` | `1e-4`, `1e-3` | |
| `trainer.max_epochs`, `.min_epochs` | `25`, `10` | |

## `pretrain` (`regime=pretrain`)

| Key | Default | Meaning |
|---|---|---|
| `total_steps`, `warmup_steps` | `200000`, `1000` | optimizer steps |
| `momentum_start`, `momentum_end` | `0.99`, `1.0` | target-encoder EMA, cosine schedule |
| `predictor_hidden_dim` | `512` | |
| `augmentation.mu`, `.p_lambda` | `0.25`, `0.45` | node/edge drop scale and ceiling |
| `augmentation.noise_std`, `.noise_features` | `0.2`, `[]` | Gaussian noise on named features |
| `optim.lr`, `optim.weight_decay` | `5e-5`, `1e-5` | |

## Other commands

* `grass-mil-nested-cv`: `nested_cv.folds_file`, `inner_k` (3), `inner_seed`,
  `epoch_reduce` (`median`), `min_refit_epochs`, `outer_folds`, `resume`.
* `grass-mil-extract`: `extract.nested_cv_summary`, `models`, `output_dir`,
  `units`, `batch_size` (512), `device` (`auto`), `resume`.
* `grass-mil-predict`: `ckpt_path`, `predict.units`, `patient_aggregation`
  (`mean`, `merged_bag`), `batch_size`, `device`.
* `grass-mil-niches`, `grass-mil-metaclusters`, `grass-mil-spatial-stats`: see
  [niche analysis](niche_analysis.md); their configs
  (`configs/niches.yaml`, `metaclusters.yaml`, `spatial_stats.yaml`) document
  every key.
