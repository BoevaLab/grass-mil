# Hyperparameter Reference

This reference covers repo-defined configuration keys and behavior for:

- `configs/train.yaml`
- `configs/eval.yaml`
- `configs/loocv.yaml`
- `configs/data/spatial_omics.yaml`
- `configs/task/*.yaml`
- `configs/model/**/*.yaml`
- `configs/optim/*.yaml`
- `configs/scheduler/*.yaml`
- `configs/trainer/*.yaml`
- `configs/callbacks/*.yaml`
- `configs/logger/*.yaml`
- `configs/debug/*.yaml`
- `configs/hydra/default.yaml`
- `configs/extras/default.yaml`

Override syntax examples use:

```bash
python src/train.py key.path=value
```

## 1) Top-Level Train/Eval Composition Keys

## 1.1 `configs/train.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | composition list | list | Hydra config groups | always | `python src/train.py trainer=gpu` |
| `task_name` | `train` | str | any string | always | `python src/train.py task_name=my_run` |
| `tags` | `[dev]` | list[str] | any list | always | `python src/train.py tags="[paper,ablation]"` |
| `train` | `True` | bool | `true/false` | always | `python src/train.py train=false` |
| `test` | `True` | bool | `true/false` | after fit | `python src/train.py test=false` |
| `ckpt_path` | `null` | str/null | path or null | resume training when set | `python src/train.py ckpt_path=/abs/last.ckpt` |
| `seed` | `null` | int/null | integer or null | random seeding | `python src/train.py seed=42` |

## 1.2 `configs/eval.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | composition list | list | Hydra config groups | always | `python src/eval.py trainer=gpu` |
| `task_name` | `eval` | str | any string | always | `python src/eval.py task_name=holdout_eval` |
| `tags` | `[dev]` | list[str] | any list | always | `python src/eval.py tags="[final_eval]"` |
| `ckpt_path` | `???` | str | required absolute/relative checkpoint path | always | `python src/eval.py ckpt_path=/abs/model.ckpt` |

## 1.3 `configs/loocv.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task_name` | `loocv` | str | any string | LOOCV entrypoint | `python src/loocv.py task_name=loocv_run` |
| `loocv.mode` | `single` | str | `single`, `all` | LOOCV entrypoint | `python src/loocv.py loocv.mode=all` |
| `loocv.fold_index` | `null` | int/null | `>=0` or null | single mode fold selection | `python src/loocv.py loocv.mode=single loocv.fold_index=3` |
| `loocv.per_fold_processed_dir` | `true` | bool | `true/false` | fold isolation | `python src/loocv.py loocv.per_fold_processed_dir=false` |
| `loocv.force_precompute_per_fold` | `true` | bool | `true/false` | leakage-safe fold preprocessing | `python src/loocv.py loocv.force_precompute_per_fold=true` |
| `loocv.summary_filename` | `loocv_summary.json` | str | filename | fold summary artifact naming | `python src/loocv.py loocv.summary_filename=my_summary.json` |

## 2) Data Config (`configs/data/spatial_omics.yaml`)

## 2.1 Datamodule Core

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data._target_` | `src.data.spatial_omics_datamodule.SpatialOmicsDataModule` | str | import path | always | `python src/train.py data._target_=...` |
| `data.data_dir` | `${paths.data_dir}` | str | path | always | `python src/train.py data.data_dir=/tmp/data` |
| `data.raw_manifest_path` | `${paths.data_dir}/raw/manifest.csv` | str | path | always | `python src/train.py data.raw_manifest_path=/abs/manifest.csv` |
| `data.processed_dir` | `${paths.data_dir}/processed/spatial_omics` | str | path | always | `python src/train.py data.processed_dir=/abs/processed` |
| `data.batch_size` | `1` | int | `>=1` | loader batching | `python src/train.py data.batch_size=4` |
| `data.num_workers` | `0` | int | `>=0` | dataloader workers | `python src/train.py data.num_workers=8` |
| `data.pin_memory` | `False` | bool | `true/false` | dataloader | `python src/train.py data.pin_memory=true` |
| `data.coord_scale_um` | `1.0` | float | `>0` | preprocessing | `python src/train.py data.coord_scale_um=0.5` |
| `data.sample_unit` | `tile` | str | `tile`/`full` | patch generation | `python src/train.py data.sample_unit=full` |
| `data.reducer_scope` | `sample` | str | `sample`/`dataset` | reducer fitting | `python src/train.py data.reducer_scope=dataset` |
| `data.keep_raw_molecular` | `false` | bool | `true/false` | precompute outputs | `python src/train.py data.keep_raw_molecular=true` |
| `data.force_precompute` | `false` | bool | `true/false` | precompute trigger | `python src/train.py data.force_precompute=true` |
| `data.min_cells` | `10` | int | `>=0` | patch filtering | `python src/train.py data.min_cells=25` |
| `data.use_molecular_features` | `true` | bool | `true/false` | loader/reducer behavior | `python src/train.py data.use_molecular_features=false` |

## 2.2 Categorical Feature Encoding

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.categorical_features.include_labels` | `[cell_type]` | list[str] | label names present in loaded data | attach `categorical_index` | `python src/train.py data.categorical_features.include_labels="[cell_type,state]"` |

## 2.3 Split Config

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.split.train_val_test_split` | `[0.8,0.1,0.1]` | list[float] len=3 | must sum to `1.0` | precompute split assignment | `python src/train.py data.split.train_val_test_split="[0.7,0.15,0.15]"` |
| `data.split.split_by` | `sample` | str | `sample`/`region`/`patch` | split semantics | `python src/train.py data.split.split_by=region` |
| `data.split.loocv.enabled` | `false` | bool | `true/false` | LOOCV override path | `python src/train.py data.split.loocv.enabled=true` |
| `data.split.loocv.fold_unit` | `region` | str | `region`/`sample` | LOOCV override path | `python src/train.py data.split.loocv.fold_unit=sample` |
| `data.split.loocv.holdout_id` | `null` | str/null | canonical fold id (`sample` or `sample::region`) | LOOCV override path | `python src/train.py data.split.loocv.holdout_id=s1::r3` |
| `data.split.loocv.validation_strategy` | `heldout_fold_items` | str | `heldout_fold_items`/`patches_from_train_items` | LOOCV override path | `python src/train.py data.split.loocv.validation_strategy=patches_from_train_items` |
| `data.split.loocv.val_ratio` | `0.1` | float | `[0.0,1.0)` | LOOCV validation split sizing | `python src/train.py data.split.loocv.val_ratio=0.2` |
| `data.split.loocv.seed` | `42` | int | integer | LOOCV validation split determinism | `python src/train.py data.split.loocv.seed=7` |

Notes:

- When LOOCV is disabled, behavior is identical to current non-LOOCV splitting.
- Split labels are persisted at precompute time in `processed_index.json`.
  Changing `data.split.*` requires regenerating processed artifacts
  (`data.force_precompute=true` or a new `data.processed_dir`).
- For `reducer_scope=dataset` + `feature_reducer.fit_mode=train_only`, LOOCV requires fold-specific precompute (`force_precompute=true` or fold-specific `processed_dir`).
- For safe fold isolation, at least one of `loocv.force_precompute_per_fold` or `loocv.per_fold_processed_dir` must be `true`.
- In `loocv.mode=single`, fold selection can use either
  `data.split.loocv.holdout_id` or `loocv.fold_index`. If both are set,
  `loocv.fold_index` takes priority and a warning is emitted.

## 2.4 Manifest Field Mapping

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.manifest.sample_id` | `sample_id` | str | manifest column name | manifest parse | `python src/train.py data.manifest.sample_id=sid` |
| `data.manifest.input_path` | `input_path` | str | manifest column name | manifest parse | `python src/train.py data.manifest.input_path=file_path` |
| `data.manifest.input_type` | `input_type` | str | manifest column name | manifest parse | `python src/train.py data.manifest.input_type=source_type` |
| `data.manifest.region_id` | `region_id` | str | manifest column name | optional region grouping | `python src/train.py data.manifest.region_id=roi` |
| `data.manifest.polygons_path` | `polygons_path` | str | manifest column name | optional polygons | `python src/train.py data.manifest.polygons_path=poly_path` |

## 2.5 CSV Loader Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.csv.sep` | `,` | str | delimiter string | `input_type=csv/tsv` | `python src/train.py data.csv.sep=';'` |
| `data.csv.coord_columns` | `[x,y]` | list[str] len=2 | existing columns | `input_type=csv/tsv` | `python src/train.py data.csv.coord_columns="[x_um,y_um]"` |
| `data.csv.cell_id_column` | `cell_id` | str/null | existing column or null | `input_type=csv/tsv` | `python src/train.py data.csv.cell_id_column=id` |
| `data.csv.categorical_label_columns` | `[]` | list[str] | existing columns | `input_type=csv/tsv` | `python src/train.py data.csv.categorical_label_columns="[cell_type]"` |
| `data.csv.molecular_columns` | `null` | list[str]/null | existing columns or null infer | `input_type=csv/tsv` and `use_molecular_features=true` | `python src/train.py data.csv.molecular_columns="[gene1,gene2]"` |

## 2.6 H5AD Loader Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.h5ad.coord_source` | `obsm` | str | `obs`/`obsm` | `input_type=h5ad` | `python src/train.py data.h5ad.coord_source=obs` |
| `data.h5ad.coord_key` | `spatial` | str/null | obsm key | `coord_source=obsm` | `python src/train.py data.h5ad.coord_key=X_spatial` |
| `data.h5ad.coord_columns` | `[x,y]` | list[str] len=2 | obs columns | `coord_source=obs` | `python src/train.py data.h5ad.coord_columns="[x_um,y_um]"` |
| `data.h5ad.cell_id_column` | `null` | str/null | obs column or null | `input_type=h5ad` | `python src/train.py data.h5ad.cell_id_column=cell_id` |
| `data.h5ad.categorical_label_columns` | `[]` | list[str] | obs columns | `input_type=h5ad` | `python src/train.py data.h5ad.categorical_label_columns="[cell_type]"` |
| `data.h5ad.molecular_layer` | `null` | str/null | layer key or null (`X`) | `use_molecular_features=true` | `python src/train.py data.h5ad.molecular_layer=counts` |
| `data.h5ad.molecular_features` | `null` | list[str]/null | var names subset | `use_molecular_features=true` | `python src/train.py data.h5ad.molecular_features="[CD3,CD8]"` |

## 2.7 SCE Loader Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.sce.assay_name` | `null` | str/null | assay name or default assay | `input_type=sce/rds` | `python src/train.py data.sce.assay_name=logcounts` |
| `data.sce.coord_source` | `colData` | str | `colData`/`reducedDims` | `input_type=sce/rds` | `python src/train.py data.sce.coord_source=reducedDims` |
| `data.sce.coord_key` | `null` | str/null | reducedDims key | `coord_source=reducedDims` | `python src/train.py data.sce.coord_key=spatial` |
| `data.sce.coord_columns` | `[x,y]` | list[str]/null | colData columns | `coord_source=colData` | `python src/train.py data.sce.coord_columns="[x_um,y_um]"` |
| `data.sce.cell_id_column` | `null` | str/null | colData column or colnames fallback | `input_type=sce/rds` | `python src/train.py data.sce.cell_id_column=cell_id` |
| `data.sce.categorical_label_columns` | `[]` | list[str] | colData columns | `input_type=sce/rds` | `python src/train.py data.sce.categorical_label_columns="[cell_type]"` |
| `data.sce.transpose_assay` | `true` | bool | `true/false` | molecular matrix orientation | `python src/train.py data.sce.transpose_assay=false` |
| `data.sce.molecular_features` | `null` | list[str]/null | rowname subset | `use_molecular_features=true` | `python src/train.py data.sce.molecular_features="[geneA,geneB]"` |

## 2.8 Graph Builder Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.graph_builder.name` | `delaunay` | str | registered builder name/import path | graph construction | `python src/train.py data.graph_builder.name=delaunay` |
| `data.graph_builder.kwargs.edge_features` | `[distance,neighbor]` | list[str] | `distance`, `neighbor` | `delaunay` builder | `python src/train.py data.graph_builder.kwargs.edge_features="[distance]"` |
| `data.graph_builder.kwargs.neighbor_cutoff_um` | `20.0` | float | `>0` | when `neighbor` edge feature enabled | `python src/train.py data.graph_builder.kwargs.neighbor_cutoff_um=50.0` |

## 2.9 Feature Reducer Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.feature_reducer.name` | `identity` | str | `identity`, `pca`, custom import path | molecular reduction | `python src/train.py data.feature_reducer.name=pca` |
| `data.feature_reducer.fit_mode` | `train_only` | str | `train_only`/`global` | dataset-scope reducer | `python src/train.py data.feature_reducer.fit_mode=global` |
| `data.feature_reducer.kwargs` | `{}` | dict | reducer kwargs | reducer-specific | `python src/train.py data.feature_reducer.kwargs.n_components=64` |

## 2.10 Sampler Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.sampler.name` | `shadow_custom` | str | `identity`, `shadow_native`, `shadow_custom` | dataloader build | `python src/train.py data.sampler.name=identity` |
| `data.sampler.kwargs` | `{}` | dict | strategy kwargs | strategy-specific | `python src/train.py data.sampler.kwargs='{}'` |
| `data.sampler.runtime.enabled` | `true` | bool | `true/false` | shadow strategies | `python src/train.py data.sampler.runtime.enabled=false` |
| `data.sampler.runtime.depth` | `2` | int | `>=1` | shadow runtime | `python src/train.py data.sampler.runtime.depth=3` |
| `data.sampler.runtime.num_neighbors` | `8` | int | `>=1` | shadow runtime | `python src/train.py data.sampler.runtime.num_neighbors=16` |
| `data.sampler.runtime.subgraph_batch_size` | `32` | int | `>=1` | runtime subgraph batching | `python src/train.py data.sampler.runtime.subgraph_batch_size=64` |
| `data.sampler.runtime.replace` | `false` | bool | `true/false` | root sampling | `python src/train.py data.sampler.runtime.replace=true` |
| `data.sampler.runtime.shuffle_subgraphs` | `true` | bool | `true/false` | training shuffle behavior | `python src/train.py data.sampler.runtime.shuffle_subgraphs=false` |
| `data.sampler.runtime.proportional_root_sampling` | `true` | bool | `true/false` | weighted root sampling | `python src/train.py data.sampler.runtime.proportional_root_sampling=false` |
| `data.sampler.runtime.property_name` | `cell_type` | str | categorical label present in `categorical_slices` | weighted root sampling | `python src/train.py data.sampler.runtime.property_name=cell_type` |
| `data.sampler.runtime.weight_mode` | `inverse` | str | `inverse`, `sqrt_inverse`, `proportional` | weighted root sampling | `python src/train.py data.sampler.runtime.weight_mode=sqrt_inverse` |
| `data.sampler.runtime.min_weight` | `1.0e-6` | float | `>0` | weighted root sampling | `python src/train.py data.sampler.runtime.min_weight=1.0e-4` |

## 2.11 Tiling And Graph Label Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.tiling.tile_size_um` | `200.0` | float | `>0` | `sample_unit=tile` | `python src/train.py data.tiling.tile_size_um=300.0` |
| `data.tiling.stride_um` | `200.0` | float | `>0` | `sample_unit=tile` | `python src/train.py data.tiling.stride_um=150.0` |
| `data.tiling.min_cells` | `10` | int | `>=0` | tile retention | `python src/train.py data.tiling.min_cells=20` |
| `data.graph_labels.label_file` | `null` | str/null | CSV path or null | optional graph labels | `python src/train.py data.graph_labels.label_file=/abs/labels.csv` |
| `data.graph_labels.id_column` | `id` | str | column in label file | graph labels enabled | `python src/train.py data.graph_labels.id_column=region_id` |
| `data.graph_labels.tasks` | `null` | list[str]/null | task columns or null for all | graph labels enabled | `python src/train.py data.graph_labels.tasks="[outcome]"` |
| `data.graph_labels.scope` | `region` | str | `region`, `patch`, `sample` | graph labels enabled | `python src/train.py data.graph_labels.scope=sample` |
| `data.transforms` | `[]` | list[dict] | importable transform specs | transform pipeline | `python src/train.py data.transforms='[{name: src.data.components.transforms.CompositionVector, kwargs: {label_attr: label_cell_type}}]'` |

## 2.12 Sampler Preset Files (`configs/data/sampler/*.yaml`)

| File | Keys In File | Defaults |
|---|---|---|
| `configs/data/sampler/identity.yaml` | `name`, `kwargs` | `name=identity`, `kwargs={}` |
| `configs/data/sampler/shadow_native.yaml` | `name`, `kwargs`, `runtime.*` | `name=shadow_native`, runtime defaults mirror `data.sampler.runtime.*` |
| `configs/data/sampler/shadow_custom.yaml` | `name`, `kwargs`, `runtime.*` | `name=shadow_custom`, runtime defaults mirror `data.sampler.runtime.*` |

## 3) Task Config Group (`configs/task/*.yaml`)

## 3.1 `configs/task/finetune_mean.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task.name` | `finetune_mean` | str | label | always | `python src/train.py task.name=custom_mean` |
| `task.aggregation` | `mean` | str | `mean` | supervised module | `python src/train.py task.aggregation=mean` |
| `task.bag_key` | `region_id` | str | batch attribute name | supervised bagging | `python src/train.py task.bag_key=sample_id` |
| `task.bag_fallback_key` | `sample_id` | str | batch attribute name | bag key missing fallback | `python src/train.py task.bag_fallback_key=patch_id` |
| `task.max_instances_per_bag` | `0` | int | `0` means all, otherwise `>=1` | bag aggregation | `python src/train.py task.max_instances_per_bag=64` |
| `task.instance_sampling` | `all` | str | `all`, `random` | if max instances > 0 | `python src/train.py task.instance_sampling=random` |
| `task.target_type` | `binary` | str | `binary`, `regression`, `survival` | loss routing | `python src/train.py task.target_type=regression` |
| `task.target_columns` | `null` | list[str]/null | label names | target selection | `python src/train.py task.target_columns="[outcome]"` |
| `task.loss` | `categorical_bce` | str | model loss presets | loss build | `python src/train.py task.loss=regression_mse` |
| `task.lr_warmup.enabled` | `false` | bool | `true/false` | scheduler wrapping | `python src/train.py task.lr_warmup.enabled=true` |
| `task.lr_warmup.warmup_steps` | `0` | int | `>=0` | warmup enabled | `python src/train.py task.lr_warmup.warmup_steps=100` |
| `task.lr_warmup.start_factor` | `0.1` | float | `(0,1]` (clamped) | warmup enabled | `python src/train.py task.lr_warmup.start_factor=0.2` |

## 3.2 `configs/task/finetune_mil.yaml`

Includes all keys above plus MIL-specific controls:

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task.name` | `finetune_mil` | str | label | always | `python src/train.py task.name=mil_exp` |
| `task.aggregation` | `mil_attention` | str | `mil_attention` | supervised module | `python src/train.py task.aggregation=mil_attention` |
| `task.optimization.backbone_lr` | `null` | float/null | lr value or null | MIL optimizer param groups | `python src/train.py task.optimization.backbone_lr=1e-3` |
| `task.optimization.attention_lr` | `null` | float/null | lr value or null | MIL optimizer param groups | `python src/train.py task.optimization.attention_lr=4e-3` |
| `task.region_accumulation.enabled` | `false` | bool | `true/false` | MIL training only | `python src/train.py task.region_accumulation.enabled=true` |
| `task.region_accumulation.mode` | `manual_region_buffer` | str | currently manual mode in code | MIL accumulation metadata | `python src/train.py task.region_accumulation.mode=manual_region_buffer` |
| `task.region_accumulation.hyperbatch_size` | `8` | int | `>=1` when enabled | MIL accumulation | `python src/train.py task.region_accumulation.hyperbatch_size=4` |
| `task.region_accumulation.flush_on_epoch_end` | `true` | bool | `true/false` | MIL accumulation | `python src/train.py task.region_accumulation.flush_on_epoch_end=true` |
| `task.node_aux.enabled` | `false` | bool | `true/false` | MIL + binary training stage | `python src/train.py task.node_aux.enabled=true` |
| `task.node_aux.target_mode` | `attention_shaped_ti` | str | `graph_label`, `attention_shaped_ti` | node auxiliary targets | `python src/train.py task.node_aux.target_mode=graph_label` |
| `task.node_aux.loss_mode` | `bce` | str | `bce`, `weighted_bce` | node auxiliary loss | `python src/train.py task.node_aux.loss_mode=weighted_bce` |
| `task.node_aux.weight` | `1.0` | float | any float | node auxiliary weighting | `python src/train.py task.node_aux.weight=0.2` |
| `task.entropy_reg.enabled` | `false` | bool | `true/false` | MIL + binary training stage | `python src/train.py task.entropy_reg.enabled=true` |
| `task.entropy_reg.mode` | `attention_shaped_target` | str | `attention`, `attention_shaped_target` | entropy regularization | `python src/train.py task.entropy_reg.mode=attention` |
| `task.entropy_reg.weight` | `0.01` | float | any float | entropy weighting | `python src/train.py task.entropy_reg.weight=0.005` |
| `task.loss_weights.region` | `1.0` | float | any float | MIL total loss composition | `python src/train.py task.loss_weights.region=1.0` |
| `task.loss_weights.node_aux` | `1.0` | float | any float | MIL total loss composition | `python src/train.py task.loss_weights.node_aux=0.5` |
| `task.loss_weights.entropy` | `0.0` | float | any float | MIL total loss composition | `python src/train.py task.loss_weights.entropy=0.1` |

## 3.3 `configs/task/pretrain_bgrl.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | `override /scheduler: cosine_step` | Hydra defaults entry | scheduler group override | pretrain task composition | `python src/train.py task=pretrain_bgrl` |
| `task.name` | `pretrain_bgrl` | str | label | always | `python src/train.py task.name=ssl_run` |
| `task.aggregation` | `mean` | str | `mean` | BGRL module ignores bag loss semantics but keeps task schema | `python src/train.py task.aggregation=mean` |
| `task.bag_key` | `region_id` | str | batch attr | task metadata | `python src/train.py task.bag_key=sample_id` |
| `task.bag_fallback_key` | `sample_id` | str | batch attr | task metadata | `python src/train.py task.bag_fallback_key=patch_id` |
| `task.max_instances_per_bag` | `0` | int | `0`/`>=1` | task metadata | `python src/train.py task.max_instances_per_bag=0` |
| `task.instance_sampling` | `all` | str | `all`/`random` | task metadata | `python src/train.py task.instance_sampling=all` |
| `task.target_type` | `binary` | str | task labels | task metadata | `python src/train.py task.target_type=binary` |
| `task.target_columns` | `null` | list/null | optional | task metadata | `python src/train.py task.target_columns='[label]'` |
| `task.loss` | `categorical_bce` | str | loss label | task metadata | `python src/train.py task.loss=categorical_bce` |
| `task.lr_warmup.enabled` | `true` | bool | `true/false` | optimizer scheduler warmup | `python src/train.py task.lr_warmup.enabled=true` |
| `task.lr_warmup.warmup_steps` | `1000` | int | `>=0` | warmup enabled | `python src/train.py task.lr_warmup.warmup_steps=500` |
| `task.lr_warmup.start_factor` | `0.1` | float | `(0,1]` | warmup enabled | `python src/train.py task.lr_warmup.start_factor=0.2` |
| `task.warmup_steps` | `1000` | int | `>=0` | BGRL momentum scheduler | `python src/train.py task.warmup_steps=500` |
| `task.total_steps` | `10000` | int | `>0` preferred | BGRL momentum scheduler | `python src/train.py task.total_steps=20000` |
| `task.momentum` | `0.99` | float | `[0,1]` practical range | target network EMA | `python src/train.py task.momentum=0.996` |
| `task.momentum_min` | `0.99` | float | `[0,1]` practical range | cosine momentum floor | `python src/train.py task.momentum_min=0.9` |
| `task.drop_edge_p1` | `0.2` | float | `[0,1]` | view-1 edge dropout | `python src/train.py task.drop_edge_p1=0.1` |
| `task.drop_edge_p2` | `0.2` | float | `[0,1]` | view-2 edge dropout | `python src/train.py task.drop_edge_p2=0.3` |
| `task.drop_feat_p1` | `0.2` | float | `[0,1]` | view-1 feature dropout | `python src/train.py task.drop_feat_p1=0.1` |
| `task.drop_feat_p2` | `0.2` | float | `[0,1]` | view-2 feature dropout | `python src/train.py task.drop_feat_p2=0.3` |

## 4) Model Config Group (`configs/model/**/*.yaml`)

## 4.1 `configs/model/supervised_module.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model._target_` | `src.models.supervised_module.SupervisedModule` | str | import path | supervised runs | `python src/train.py model=supervised_module` |
| `model._recursive_` | `false` | bool | `true/false` | Hydra instantiate semantics | `python src/train.py model._recursive_=false` |
| `model.flags.use_ssl` | `false` | bool | `true/false` | optional SSL wrapper in supervised module | `python src/train.py model.flags.use_ssl=true` |
| `model.encoder.input_dim` | `0` | int | `>0` or `0` for infer | encoder build | `python src/train.py model.encoder.input_dim=128` |
| `model.encoder.hidden_dim` | `128` | int | `>=1` | encoder | `python src/train.py model.encoder.hidden_dim=256` |
| `model.encoder.out_dim` | `128` | int/null | `>=1` or null | encoder projection | `python src/train.py model.encoder.out_dim=256` |
| `model.encoder.num_layers` | `3` | int | `>=1` | encoder | `python src/train.py model.encoder.num_layers=4` |
| `model.encoder.dropout` | `0.1` | float | `[0,1]` | encoder | `python src/train.py model.encoder.dropout=0.2` |
| `model.encoder.conv_type` | `gin` | str | `gin`, `gcn`, `gat`, `graphsage`, `gine` | encoder | `python src/train.py model.encoder.conv_type=gine` |
| `model.encoder.norm` | `batchnorm` | str | `batchnorm`, `layernorm`, `none` | encoder | `python src/train.py model.encoder.norm=layernorm` |
| `model.encoder.jk` | `last` | str | `last`, `concat`, `max`, `sum` | encoder | `python src/train.py model.encoder.jk=concat` |
| `model.encoder.act` | `relu` | str | `relu`, `gelu`, `leaky_relu` | encoder | `python src/train.py model.encoder.act=gelu` |
| `model.encoder.pooling` | `mean` | str | `sum`,`mean`,`max`,`attention`,`set2set` | graph embedding | `python src/train.py model.encoder.pooling=set2set` |
| `model.encoder.set2set_steps` | `3` | int | `>=1` | `pooling=set2set` | `python src/train.py model.encoder.set2set_steps=4` |
| `model.encoder.gat_heads` | `4` | int | `>=1` and divides hidden dim | `conv_type=gat` | `python src/train.py model.encoder.gat_heads=8` |
| `model.encoder.use_edge_attr` | `false` | bool | `true/false` | `gcn`/`gine` meaningful | `python src/train.py model.encoder.use_edge_attr=true` |
| `model.encoder.edge_weight_index` | `0` | int | column index | `gcn` with edge attrs | `python src/train.py model.encoder.edge_weight_index=1` |
| `model.encoder.edge_attr_dim` | `null` | int/null | required for `gine` | `conv_type=gine` | `python src/train.py model.encoder.edge_attr_dim=2` |
| `model.attention.attention_type` | `gated_projected` | str | `gated`, `gated_projected` | `task.aggregation=mil_attention` | `python src/train.py model.attention.attention_type=gated` |
| `model.attention.input_dim` | `128` | int | `>=1` | MIL attention | `python src/train.py model.attention.input_dim=256` |
| `model.attention.projection_dim` | `64` | int | `>=1` | projected attention | `python src/train.py model.attention.projection_dim=128` |
| `model.attention.hidden_dim` | `16` | int | `>=1` | MIL attention | `python src/train.py model.attention.hidden_dim=32` |
| `model.attention.dropout` | `false` | bool | `true/false` | MIL attention | `python src/train.py model.attention.dropout=true` |
| `model.attention.n_classes` | `1` | int | `>=1` | MIL attention output channels | `python src/train.py model.attention.n_classes=1` |
| `model.graph_head.input_dim` | `128` | int | `>=1` | graph head | `python src/train.py model.graph_head.input_dim=256` |
| `model.graph_head.output_dim` | `1` | int | `>=1` | graph head | `python src/train.py model.graph_head.output_dim=2` |
| `model.graph_head.hidden_dim` | `128` | int | `>=1` | graph head | `python src/train.py model.graph_head.hidden_dim=256` |
| `model.graph_head.num_layers` | `2` | int | `>=1` | graph head | `python src/train.py model.graph_head.num_layers=3` |
| `model.graph_head.dropout` | `0.0` | float | `[0,1]` | graph head | `python src/train.py model.graph_head.dropout=0.2` |
| `model.ssl.method` | `bgrl` | str | `bgrl` | SSL factory | `python src/train.py model.ssl.method=bgrl` |
| `model.ssl.predictor.hidden_size` | `512` | int | `>=1` | SSL enabled | `python src/train.py model.ssl.predictor.hidden_size=256` |
| `model.loss.loss_type` | `${task.loss}` | str | loss name | loss factory | `python src/train.py task.loss=regression_mse` |
| `model.optim` | `${optim}` | DictConfig | optimizer config object | always | `python src/train.py optim.lr=5e-4` |
| `model.scheduler` | `${scheduler}` | DictConfig | scheduler config object | always | `python src/train.py scheduler.T_max=200` |
| `model.task` | `${task}` | DictConfig | task config object | always | `python src/train.py task=finetune_mil` |
| `model.init_from_ckpt` | `null` | str/null | checkpoint path | optional init | `python src/train.py model.init_from_ckpt=/abs/pretrain.ckpt` |
| `model.init_strict` | `false` | bool | `true/false` | checkpoint init | `python src/train.py model.init_strict=true` |
| `model.encoder_init_map` | `auto_bgrl_or_identity` | str | `identity`, `auto_bgrl_or_identity` | checkpoint init | `python src/train.py model.encoder_init_map=identity` |
| `model.freeze_encoder` | `false` | bool | `true/false` | checkpoint transfer/finetuning | `python src/train.py model.freeze_encoder=true` |

## 4.2 `configs/model/bgrl_module.yaml`

Same encoder block as supervised module plus SSL/pretrain init keys:

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model._target_` | `src.models.bgrl_module.BGRLModule` | str | import path | pretrain task | `python src/train.py model=bgrl_module` |
| `model._recursive_` | `false` | bool | `true/false` | instantiate behavior | `python src/train.py model._recursive_=false` |
| `model.encoder.*` | same as supervised defaults | mixed | see section 4.1 | BGRL encoder | `python src/train.py model.encoder.conv_type=gin` |
| `model.ssl.method` | `bgrl` | str | `bgrl` | SSL module | `python src/train.py model.ssl.method=bgrl` |
| `model.ssl.predictor.hidden_size` | `512` | int | `>=1` | predictor width | `python src/train.py model.ssl.predictor.hidden_size=1024` |
| `model.optim` | `${optim}` | DictConfig | optimizer object | always | `python src/train.py optim.lr=1e-3` |
| `model.scheduler` | `${scheduler}` | DictConfig | scheduler object | always | `python src/train.py scheduler.T_max=20000` |
| `model.task` | `${task}` | DictConfig | pretrain task object | always | `python src/train.py task=pretrain_bgrl` |
| `model.init_from_ckpt` | `null` | str/null | checkpoint path | optional init | `python src/train.py model.init_from_ckpt=/abs/ssl.ckpt` |
| `model.init_strict` | `false` | bool | `true/false` | checkpoint init | `python src/train.py model.init_strict=true` |
| `model.encoder_init_map` | `auto_bgrl_or_identity` | str | `identity`, `auto_bgrl_or_identity` | checkpoint init | `python src/train.py model.encoder_init_map=identity` |

## 4.3 Component Preset Files

### `configs/model/components.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model.flags.use_attention` | `true` | bool | `true/false` | component notebook/testing | `python src/train.py model.flags.use_attention=false` |
| `model.flags.use_ssl` | `false` | bool | `true/false` | component notebook/testing | `python src/train.py model.flags.use_ssl=true` |
| `model.encoder._target_` | `src.models.components.backbones.EncoderConfig` | str | import path | hydra instantiate | `python src/train.py model.encoder._target_=...` |
| `model.encoder.*` | same fields as section 4.1 | mixed | see section 4.1 | component testing | `python src/train.py model.encoder.conv_type=gcn` |
| `model.attention._target_` | `src.models.components.attention.AttnNetGatedProjected` | str | import path | component testing | `python src/train.py model.attention._target_=...AttnNetGated` |
| `model.attention.*` | defaults per file | mixed | see attention files below | component testing | `python src/train.py model.attention.hidden_dim=32` |
| `model.ssl.method` | `bgrl` | str | `bgrl` | component testing | `python src/train.py model.ssl.method=bgrl` |
| `model.ssl.predictor._target_` | `src.models.components.ssl.MLPPredictor` | str | import path | component testing | `python src/train.py model.ssl.predictor._target_=...` |
| `model.ssl.predictor.input_size` | `128` | int | `>=1` | component testing | `python src/train.py model.ssl.predictor.input_size=256` |
| `model.ssl.predictor.output_size` | `128` | int | `>=1` | component testing | `python src/train.py model.ssl.predictor.output_size=256` |
| `model.ssl.predictor.hidden_size` | `512` | int | `>=1` | component testing | `python src/train.py model.ssl.predictor.hidden_size=256` |
| `model.graph_head._target_` | `src.models.components.heads.GraphPredictionHead` | str | import path | component testing | `python src/train.py model.graph_head._target_=...` |
| `model.graph_head.*` | see section 4.1 | mixed | MLP head args | component testing | `python src/train.py model.graph_head.num_layers=3` |
| `model.node_head._target_` | `src.models.components.heads.NodePredictionHead` | str | import path | component testing | `python src/train.py model.node_head._target_=...` |
| `model.node_head.*` | graph-head-like args | mixed | MLP head args | component testing | `python src/train.py model.node_head.dropout=0.1` |

### Encoder preset files (`configs/model/encoder/*.yaml`)

Covered files:

- `configs/model/encoder/gin.yaml`
- `configs/model/encoder/gcn.yaml`
- `configs/model/encoder/gat.yaml`
- `configs/model/encoder/graphsage.yaml`
- `configs/model/encoder/gine.yaml`

All five files expose the same keys:

- `_target_`, `input_dim`, `hidden_dim`, `out_dim`, `num_layers`, `dropout`, `conv_type`, `norm`, `jk`, `act`, `pooling`, `set2set_steps`, `gat_heads`, `use_edge_attr`, `edge_weight_index`, `edge_attr_dim`.

Per-file defaults differ mainly in `conv_type`, `use_edge_attr`, and `edge_attr_dim`:

- `gin.yaml`: `conv_type=gin`, `use_edge_attr=false`, `edge_attr_dim=null`
- `gcn.yaml`: `conv_type=gcn`, `use_edge_attr=true`, `edge_attr_dim=null`
- `gat.yaml`: `conv_type=gat`, `use_edge_attr=false`, `edge_attr_dim=null`
- `graphsage.yaml`: `conv_type=graphsage`, `use_edge_attr=false`, `edge_attr_dim=null`
- `gine.yaml`: `conv_type=gine`, `use_edge_attr=true`, `edge_attr_dim=2`

### Attention preset files (`configs/model/attention/*.yaml`)

Covered files:

- `configs/model/attention/gated.yaml`
- `configs/model/attention/gated_projected.yaml`

| File | Keys | Defaults |
|---|---|---|
| `gated.yaml` | `_target_`, `input_dim`, `hidden_dim`, `dropout`, `n_classes` | `AttnNetGated`, `128`, `64`, `false`, `1` |
| `gated_projected.yaml` | `_target_`, `input_dim`, `projection_dim`, `hidden_dim`, `dropout`, `n_classes` | `AttnNetGatedProjected`, `128`, `64`, `16`, `false`, `1` |

### Head preset files (`configs/model/heads/*.yaml`)

Covered files:

- `configs/model/heads/graph.yaml`
- `configs/model/heads/node.yaml`

Both `graph.yaml` and `node.yaml` expose:

- `_target_`, `input_dim`, `output_dim`, `hidden_dim`, `num_layers`, `dropout`

Defaults in both files:

- `input_dim=128`, `output_dim=1`, `hidden_dim=128`, `num_layers=2`, `dropout=0.0`

### Loss preset files (`configs/model/loss/*.yaml`)

Covered files:

- `configs/model/loss/categorical_bce.yaml`
- `configs/model/loss/categorical_ce.yaml`
- `configs/model/loss/regression_mse.yaml`
- `configs/model/loss/regression_huber.yaml`
- `configs/model/loss/survival_coxsgd.yaml`

| File | Keys | Defaults |
|---|---|---|
| `categorical_bce.yaml` | `_target_`, `pos_weight` | `WeightedBCEWithLogitsLoss`, `null` |
| `categorical_ce.yaml` | `_target_`, `class_weight` | `WeightedCrossEntropyLoss`, `null` |
| `regression_mse.yaml` | `_target_` | `WeightedMSELoss` |
| `regression_huber.yaml` | `_target_`, `delta` | `WeightedHuberLoss`, `1.0` |
| `survival_coxsgd.yaml` | `_target_`, `top_n`, `regularizer_weight` | `CoxSGDLoss`, `5`, `0.05` |

### SSL preset file (`configs/model/ssl/bgrl.yaml`)

Keys:

- `method` (`bgrl`)
- `predictor._target_`
- `predictor.input_size`
- `predictor.output_size`
- `predictor.hidden_size`

## 5) Optimizer And Scheduler Configs

## 5.1 `configs/optim/adamw.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `optim._target_` | `torch.optim.AdamW` | str | optimizer class path | always | `python src/train.py optim._target_=torch.optim.SGD` |
| `optim.lr` | `0.001` | float | `>0` | optimizer step | `python src/train.py optim.lr=5e-4` |
| `optim.weight_decay` | `0.01` | float | `>=0` | optimizer step | `python src/train.py optim.weight_decay=1e-4` |

## 5.2 Scheduler files

| File | Key | Default | Type | Example Override |
|---|---|---|---|---|
| `configs/scheduler/cosine_epoch.yaml` | `_target_` | `torch.optim.lr_scheduler.CosineAnnealingLR` | str | `python src/train.py scheduler._target_=...` |
| `configs/scheduler/cosine_epoch.yaml` | `T_max` | `100` | int | `python src/train.py scheduler.T_max=200` |
| `configs/scheduler/cosine_epoch.yaml` | `eta_min` | `1.0e-6` | float | `python src/train.py scheduler.eta_min=1.0e-7` |
| `configs/scheduler/cosine_step.yaml` | `_target_` | `torch.optim.lr_scheduler.CosineAnnealingLR` | str | `python src/train.py scheduler._target_=...` |
| `configs/scheduler/cosine_step.yaml` | `T_max` | `10000` | int | `python src/train.py scheduler.T_max=20000` |
| `configs/scheduler/cosine_step.yaml` | `eta_min` | `1.0e-7` | float | `python src/train.py scheduler.eta_min=1.0e-8` |

## 6) Trainer Config Group (`configs/trainer/*.yaml`)

## 6.1 `configs/trainer/default.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `trainer._target_` | `lightning.pytorch.trainer.Trainer` | str | trainer import path | always | `python src/train.py trainer._target_=...` |
| `trainer.default_root_dir` | `${paths.output_dir}` | str | path | trainer init | `python src/train.py trainer.default_root_dir=/tmp/out` |
| `trainer.min_epochs` | `1` | int | `>=0` | fit loop | `python src/train.py trainer.min_epochs=0` |
| `trainer.max_epochs` | `10` | int | `>=1` | fit loop | `python src/train.py trainer.max_epochs=100` |
| `trainer.accelerator` | `cpu` | str | lightning-supported accelerators | runtime | `python src/train.py trainer.accelerator=gpu` |
| `trainer.devices` | `1` | int/list/str | lightning-supported device spec | runtime | `python src/train.py trainer.devices=2` |
| `trainer.check_val_every_n_epoch` | `1` | int | `>=1` | fit loop | `python src/train.py trainer.check_val_every_n_epoch=2` |
| `trainer.deterministic` | `False` | bool | `true/false` | reproducibility | `python src/train.py trainer.deterministic=true` |

## 6.2 Preset overrides

| File | Keys In File | Meaning |
|---|---|---|
| `configs/trainer/cpu.yaml` | `defaults: [default]`, `accelerator=cpu`, `devices=1` | explicit CPU runtime |
| `configs/trainer/gpu.yaml` | `defaults: [default]`, `accelerator=gpu`, `devices=1` | single-GPU runtime |
| `configs/trainer/mps.yaml` | `defaults: [default]`, `accelerator=mps`, `devices=1` | Apple MPS runtime |
| `configs/trainer/ddp.yaml` | `defaults: [default]`, `strategy=ddp`, `accelerator=gpu`, `devices=4`, `num_nodes=1`, `sync_batchnorm=True` | multi-GPU distributed preset |
| `configs/trainer/ddp_sim.yaml` | `defaults: [default]`, `accelerator=cpu`, `devices=2`, `strategy=ddp_spawn` | CPU DDP simulation for debugging |

## 7) Callback Config Group (`configs/callbacks/*.yaml`)

## 7.1 `configs/callbacks/default.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | `[model_checkpoint, early_stopping, model_summary, rich_progress_bar, _self_]` | list | callback group names | always | `python src/train.py callbacks=none` |
| `callbacks.model_checkpoint.dirpath` | `${paths.output_dir}/checkpoints` | str | path | checkpoint callback enabled | `python src/train.py callbacks.model_checkpoint.dirpath=/tmp/ckpt` |
| `callbacks.model_checkpoint.filename` | `epoch_{epoch:03d}` | str | format string | checkpoint callback enabled | `python src/train.py callbacks.model_checkpoint.filename=best` |
| `callbacks.model_checkpoint.monitor` | `val/loss` | str | logged metric name | checkpoint callback enabled | `python src/train.py callbacks.model_checkpoint.monitor=val/acc` |
| `callbacks.model_checkpoint.mode` | `min` | str | `min`, `max` | checkpoint callback enabled | `python src/train.py callbacks.model_checkpoint.mode=max` |
| `callbacks.model_checkpoint.save_last` | `True` | bool | `true/false` | checkpoint callback enabled | `python src/train.py callbacks.model_checkpoint.save_last=false` |
| `callbacks.model_checkpoint.auto_insert_metric_name` | `False` | bool | `true/false` | checkpoint callback enabled | `python src/train.py callbacks.model_checkpoint.auto_insert_metric_name=true` |
| `callbacks.early_stopping.monitor` | `val/loss` | str | logged metric name | early stopping enabled | `python src/train.py callbacks.early_stopping.monitor=val/acc` |
| `callbacks.early_stopping.patience` | `100` | int | `>=0` | early stopping enabled | `python src/train.py callbacks.early_stopping.patience=20` |
| `callbacks.early_stopping.mode` | `min` | str | `min`, `max` | early stopping enabled | `python src/train.py callbacks.early_stopping.mode=max` |
| `callbacks.model_summary.max_depth` | `-1` | int | integer depth | model summary enabled | `python src/train.py callbacks.model_summary.max_depth=2` |

## 7.2 `configs/callbacks/model_checkpoint.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model_checkpoint._target_` | `lightning.pytorch.callbacks.ModelCheckpoint` | str | import path | always | `python src/train.py callbacks.model_checkpoint._target_=...` |
| `model_checkpoint.dirpath` | `null` | str/null | path or null | callback active | `python src/train.py callbacks.model_checkpoint.dirpath=/tmp/ckpt` |
| `model_checkpoint.filename` | `null` | str/null | filename template or null | callback active | `python src/train.py callbacks.model_checkpoint.filename=epoch_{epoch}` |
| `model_checkpoint.monitor` | `null` | str/null | metric name or null | callback active | `python src/train.py callbacks.model_checkpoint.monitor=val/loss` |
| `model_checkpoint.verbose` | `False` | bool | `true/false` | callback active | `python src/train.py callbacks.model_checkpoint.verbose=true` |
| `model_checkpoint.save_last` | `null` | bool/null | bool or null | callback active | `python src/train.py callbacks.model_checkpoint.save_last=true` |
| `model_checkpoint.save_top_k` | `1` | int | integer | callback active | `python src/train.py callbacks.model_checkpoint.save_top_k=3` |
| `model_checkpoint.mode` | `min` | str | `min`, `max` | callback active | `python src/train.py callbacks.model_checkpoint.mode=max` |
| `model_checkpoint.auto_insert_metric_name` | `True` | bool | `true/false` | callback active | `python src/train.py callbacks.model_checkpoint.auto_insert_metric_name=false` |
| `model_checkpoint.save_weights_only` | `False` | bool | `true/false` | callback active | `python src/train.py callbacks.model_checkpoint.save_weights_only=true` |
| `model_checkpoint.every_n_train_steps` | `null` | int/null | integer or null | callback active | `python src/train.py callbacks.model_checkpoint.every_n_train_steps=100` |
| `model_checkpoint.train_time_interval` | `null` | duration/null | Lightning duration format or null | callback active | `python src/train.py callbacks.model_checkpoint.train_time_interval='00:10:00'` |
| `model_checkpoint.every_n_epochs` | `null` | int/null | integer or null | callback active | `python src/train.py callbacks.model_checkpoint.every_n_epochs=1` |
| `model_checkpoint.save_on_train_epoch_end` | `null` | bool/null | bool or null | callback active | `python src/train.py callbacks.model_checkpoint.save_on_train_epoch_end=true` |

## 7.3 `configs/callbacks/early_stopping.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `early_stopping._target_` | `lightning.pytorch.callbacks.EarlyStopping` | str | import path | always | `python src/train.py callbacks.early_stopping._target_=...` |
| `early_stopping.monitor` | `???` | str | metric name | early stopping enabled | `python src/train.py callbacks.early_stopping.monitor=val/loss` |
| `early_stopping.min_delta` | `0.` | float | any float | early stopping enabled | `python src/train.py callbacks.early_stopping.min_delta=0.001` |
| `early_stopping.patience` | `3` | int | `>=0` | early stopping enabled | `python src/train.py callbacks.early_stopping.patience=10` |
| `early_stopping.verbose` | `False` | bool | `true/false` | early stopping enabled | `python src/train.py callbacks.early_stopping.verbose=true` |
| `early_stopping.mode` | `min` | str | `min`, `max` | early stopping enabled | `python src/train.py callbacks.early_stopping.mode=max` |
| `early_stopping.strict` | `True` | bool | `true/false` | early stopping enabled | `python src/train.py callbacks.early_stopping.strict=false` |
| `early_stopping.check_finite` | `True` | bool | `true/false` | early stopping enabled | `python src/train.py callbacks.early_stopping.check_finite=true` |
| `early_stopping.stopping_threshold` | `null` | float/null | float or null | early stopping enabled | `python src/train.py callbacks.early_stopping.stopping_threshold=0.1` |
| `early_stopping.divergence_threshold` | `null` | float/null | float or null | early stopping enabled | `python src/train.py callbacks.early_stopping.divergence_threshold=10.0` |
| `early_stopping.check_on_train_epoch_end` | `null` | bool/null | bool or null | early stopping enabled | `python src/train.py callbacks.early_stopping.check_on_train_epoch_end=true` |

## 7.4 Other callback files

Covered files:

- `configs/callbacks/model_summary.yaml`
- `configs/callbacks/rich_progress_bar.yaml`
- `configs/callbacks/none.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model_summary._target_` | `lightning.pytorch.callbacks.RichModelSummary` | str | import path | `callbacks=model_summary` | `python src/train.py callbacks.model_summary._target_=...` |
| `model_summary.max_depth` | `1` | int | integer depth | `callbacks=model_summary` | `python src/train.py callbacks.model_summary.max_depth=2` |
| `rich_progress_bar._target_` | `lightning.pytorch.callbacks.RichProgressBar` | str | import path | `callbacks=rich_progress_bar` | `python src/train.py callbacks.rich_progress_bar._target_=...` |
| `configs/callbacks/none.yaml` | empty file | n/a | no fields | `callbacks=none` | `python src/train.py callbacks=none` |

## 8) Logger Config Group (`configs/logger/*.yaml`)

## 8.1 `configs/logger/csv.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `csv._target_` | `lightning.pytorch.loggers.csv_logs.CSVLogger` | str | import path | `logger=csv` | `python src/train.py logger.csv._target_=...` |
| `csv.save_dir` | `${paths.output_dir}` | str | path | `logger=csv` | `python src/train.py logger.csv.save_dir=/tmp/logs` |
| `csv.name` | `csv/` | str | name | `logger=csv` | `python src/train.py logger.csv.name=metrics` |
| `csv.prefix` | `""` | str | prefix string | `logger=csv` | `python src/train.py logger.csv.prefix=exp1` |

## 8.2 `configs/logger/tensorboard.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `tensorboard._target_` | `lightning.pytorch.loggers.tensorboard.TensorBoardLogger` | str | import path | `logger=tensorboard` | `python src/train.py logger.tensorboard._target_=...` |
| `tensorboard.save_dir` | `${paths.output_dir}/tensorboard/` | str | path | `logger=tensorboard` | `python src/train.py logger.tensorboard.save_dir=/tmp/tb` |
| `tensorboard.name` | `null` | str/null | name or null | `logger=tensorboard` | `python src/train.py logger.tensorboard.name=runA` |
| `tensorboard.log_graph` | `False` | bool | `true/false` | `logger=tensorboard` | `python src/train.py logger.tensorboard.log_graph=true` |
| `tensorboard.default_hp_metric` | `True` | bool | `true/false` | `logger=tensorboard` | `python src/train.py logger.tensorboard.default_hp_metric=false` |
| `tensorboard.prefix` | `""` | str | prefix string | `logger=tensorboard` | `python src/train.py logger.tensorboard.prefix=train` |

## 8.3 `configs/logger/wandb.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `wandb._target_` | `lightning.pytorch.loggers.wandb.WandbLogger` | str | import path | `logger=wandb` | `python src/train.py logger.wandb._target_=...` |
| `wandb.save_dir` | `${paths.output_dir}` | str | path | `logger=wandb` | `python src/train.py logger.wandb.save_dir=/tmp/wb` |
| `wandb.offline` | `False` | bool | `true/false` | `logger=wandb` | `python src/train.py logger.wandb.offline=true` |
| `wandb.id` | `null` | str/null | run ID or null | `logger=wandb` | `python src/train.py logger.wandb.id=my_id` |
| `wandb.anonymous` | `null` | str/null | WandB anonymous mode or null | `logger=wandb` | `python src/train.py logger.wandb.anonymous=allow` |
| `wandb.project` | `grass-mil` | str | project name | `logger=wandb` | `python src/train.py logger.wandb.project=grass-mil` |
| `wandb.log_model` | `False` | bool | `true/false` | `logger=wandb` | `python src/train.py logger.wandb.log_model=true` |
| `wandb.prefix` | `""` | str | prefix string | `logger=wandb` | `python src/train.py logger.wandb.prefix=exp` |
| `wandb.group` | `""` | str | group name | `logger=wandb` | `python src/train.py logger.wandb.group=ablation` |
| `wandb.tags` | `[]` | list[str] | any list | `logger=wandb` | `python src/train.py logger.wandb.tags='[ssl,mil]'` |
| `wandb.job_type` | `""` | str | job type | `logger=wandb` | `python src/train.py logger.wandb.job_type=train` |

## 8.4 `configs/logger/mlflow.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `mlflow._target_` | `lightning.pytorch.loggers.mlflow.MLFlowLogger` | str | import path | `logger=mlflow` | `python src/train.py logger.mlflow._target_=...` |
| `mlflow.tracking_uri` | `${paths.log_dir}/mlflow/mlruns` | str | URI/path | `logger=mlflow` | `python src/train.py logger.mlflow.tracking_uri=/tmp/mlruns` |
| `mlflow.tags` | `null` | dict/null | tags dict or null | `logger=mlflow` | `python src/train.py logger.mlflow.tags='{stage:dev}'` |
| `mlflow.prefix` | `""` | str | prefix | `logger=mlflow` | `python src/train.py logger.mlflow.prefix=exp` |
| `mlflow.artifact_location` | `null` | str/null | URI/path or null | `logger=mlflow` | `python src/train.py logger.mlflow.artifact_location=/tmp/artifacts` |

## 8.5 `configs/logger/comet.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `comet._target_` | `lightning.pytorch.loggers.comet.CometLogger` | str | import path | `logger=comet` | `python src/train.py logger.comet._target_=...` |
| `comet.api_key` | `${oc.env:COMET_API_TOKEN}` | str | API key/env ref | `logger=comet` | `python src/train.py logger.comet.api_key=...` |
| `comet.save_dir` | `${paths.output_dir}` | str | path | `logger=comet` | `python src/train.py logger.comet.save_dir=/tmp/comet` |
| `comet.project_name` | `grass-mil` | str | project name | `logger=comet` | `python src/train.py logger.comet.project_name=grass-mil` |
| `comet.rest_api_key` | `null` | str/null | key or null | `logger=comet` | `python src/train.py logger.comet.rest_api_key=...` |
| `comet.experiment_key` | `null` | str/null | key or null | `logger=comet` | `python src/train.py logger.comet.experiment_key=...` |
| `comet.offline` | `False` | bool | `true/false` | `logger=comet` | `python src/train.py logger.comet.offline=true` |
| `comet.prefix` | `""` | str | prefix | `logger=comet` | `python src/train.py logger.comet.prefix=train` |

## 8.6 `configs/logger/neptune.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `neptune._target_` | `lightning.pytorch.loggers.neptune.NeptuneLogger` | str | import path | `logger=neptune` | `python src/train.py logger.neptune._target_=...` |
| `neptune.api_key` | `${oc.env:NEPTUNE_API_TOKEN}` | str | API key/env ref | `logger=neptune` | `python src/train.py logger.neptune.api_key=...` |
| `neptune.project` | `org/grass-mil` | str | project slug | `logger=neptune` | `python src/train.py logger.neptune.project=org/project` |
| `neptune.log_model_checkpoints` | `True` | bool | `true/false` | `logger=neptune` | `python src/train.py logger.neptune.log_model_checkpoints=false` |
| `neptune.prefix` | `""` | str | prefix | `logger=neptune` | `python src/train.py logger.neptune.prefix=exp` |

## 8.7 `configs/logger/aim.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `aim._target_` | `aim.pytorch_lightning.AimLogger` | str | import path | `logger=aim` | `python src/train.py logger.aim._target_=...` |
| `aim.repo` | `${paths.root_dir}` | str | path/remote URI | `logger=aim` | `python src/train.py logger.aim.repo=/tmp/.aim` |
| `aim.experiment` | `null` | str/null | experiment name or null | `logger=aim` | `python src/train.py logger.aim.experiment=baseline` |
| `aim.train_metric_prefix` | `train/` | str | prefix | `logger=aim` | `python src/train.py logger.aim.train_metric_prefix=train` |
| `aim.val_metric_prefix` | `val/` | str | prefix | `logger=aim` | `python src/train.py logger.aim.val_metric_prefix=val` |
| `aim.test_metric_prefix` | `test/` | str | prefix | `logger=aim` | `python src/train.py logger.aim.test_metric_prefix=test` |
| `aim.system_tracking_interval` | `10` | int/null | seconds or null | `logger=aim` | `python src/train.py logger.aim.system_tracking_interval=5` |
| `aim.log_system_params` | `true` | bool | `true/false` | `logger=aim` | `python src/train.py logger.aim.log_system_params=false` |
| `aim.capture_terminal_logs` | `false` | bool | `true/false` | `logger=aim` | `python src/train.py logger.aim.capture_terminal_logs=true` |

## 8.8 `configs/logger/many_loggers.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | `[csv, tensorboard, wandb]` | list | logger group names | `logger=many_loggers` | `python src/train.py logger=many_loggers logger.defaults='[csv,tensorboard]'` |

## 9) Debug Config Group (`configs/debug/*.yaml`)

## 9.1 `configs/debug/default.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task_name` | `debug` | str | any string | `debug=default` | `python src/train.py debug=default task_name=dbg` |
| `callbacks` | `null` | null/config | null or callback config | `debug=default` | `python src/train.py debug=default callbacks=default` |
| `logger` | `null` | null/config | null or logger config | `debug=default` | `python src/train.py debug=default logger=csv` |
| `extras.ignore_warnings` | `False` | bool | `true/false` | `debug=default` | `python src/train.py debug=default extras.ignore_warnings=true` |
| `extras.enforce_tags` | `False` | bool | `true/false` | `debug=default` | `python src/train.py debug=default extras.enforce_tags=false` |
| `hydra.job_logging.root.level` | `DEBUG` | str | logging level | `debug=default` | `python src/train.py debug=default hydra.job_logging.root.level=INFO` |
| `trainer.max_epochs` | `1` | int | `>=1` | `debug=default` | `python src/train.py debug=default trainer.max_epochs=2` |
| `trainer.accelerator` | `cpu` | str | accelerator value | `debug=default` | `python src/train.py debug=default trainer.accelerator=cpu` |
| `trainer.devices` | `1` | int | `>=1` | `debug=default` | `python src/train.py debug=default trainer.devices=1` |
| `trainer.detect_anomaly` | `true` | bool | `true/false` | `debug=default` | `python src/train.py debug=default trainer.detect_anomaly=true` |
| `data.num_workers` | `0` | int | `>=0` | `debug=default` | `python src/train.py debug=default data.num_workers=0` |
| `data.pin_memory` | `False` | bool | `true/false` | `debug=default` | `python src/train.py debug=default data.pin_memory=false` |

## 9.2 Other debug presets

Covered files:

- `configs/debug/fdr.yaml`
- `configs/debug/limit.yaml`
- `configs/debug/overfit.yaml`
- `configs/debug/profiler.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `debug/fdr.defaults` | `[default]` | list | debug group list | `debug=fdr` | `python src/train.py debug=fdr` |
| `debug/fdr.trainer.fast_dev_run` | `true` | bool | `true/false` | `debug=fdr` | `python src/train.py debug=fdr trainer.fast_dev_run=true` |
| `debug/limit.defaults` | `[default]` | list | debug group list | `debug=limit` | `python src/train.py debug=limit` |
| `debug/limit.trainer.max_epochs` | `3` | int | `>=1` | `debug=limit` | `python src/train.py debug=limit trainer.max_epochs=5` |
| `debug/limit.trainer.limit_train_batches` | `0.01` | float | `(0,1]` or int | `debug=limit` | `python src/train.py debug=limit trainer.limit_train_batches=0.1` |
| `debug/limit.trainer.limit_val_batches` | `0.05` | float | `(0,1]` or int | `debug=limit` | `python src/train.py debug=limit trainer.limit_val_batches=0.2` |
| `debug/limit.trainer.limit_test_batches` | `0.05` | float | `(0,1]` or int | `debug=limit` | `python src/train.py debug=limit trainer.limit_test_batches=0.2` |
| `debug/overfit.defaults` | `[default]` | list | debug group list | `debug=overfit` | `python src/train.py debug=overfit` |
| `debug/overfit.trainer.max_epochs` | `20` | int | `>=1` | `debug=overfit` | `python src/train.py debug=overfit trainer.max_epochs=10` |
| `debug/overfit.trainer.overfit_batches` | `3` | int | `>=1` | `debug=overfit` | `python src/train.py debug=overfit trainer.overfit_batches=1` |
| `debug/overfit.callbacks` | `null` | null/config | null or callback config | `debug=overfit` | `python src/train.py debug=overfit callbacks=none` |
| `debug/profiler.defaults` | `[default]` | list | debug group list | `debug=profiler` | `python src/train.py debug=profiler` |
| `debug/profiler.trainer.max_epochs` | `1` | int | `>=1` | `debug=profiler` | `python src/train.py debug=profiler trainer.max_epochs=2` |
| `debug/profiler.trainer.profiler` | `simple` | str | `simple`, `advanced`, `pytorch`, etc. | `debug=profiler` | `python src/train.py debug=profiler trainer.profiler=advanced` |

## 10) Hydra Runtime Config (`configs/hydra/default.yaml`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | overrides for `hydra_logging=colorlog`, `job_logging=colorlog` | list | hydra logging groups | always | `python src/train.py hydra/job_logging=stdout` |
| `hydra.run.dir` | `${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}` | str | directory template | single run | `python src/train.py hydra.run.dir=/tmp/single_run` |
| `hydra.sweep.dir` | `${paths.log_dir}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}` | str | directory template | multirun | `python src/train.py -m hydra.sweep.dir=/tmp/multi` |
| `hydra.sweep.subdir` | `${hydra.job.num}` | str | template string | multirun | `python src/train.py -m hydra.sweep.subdir=job_${hydra.job.num}` |
| `hydra.job_logging.handlers.file.filename` | `${hydra.runtime.output_dir}/${task_name}.log` | str | file path template | Hydra file logging | `python src/train.py hydra.job_logging.handlers.file.filename=/tmp/train.log` |

## 11) Paths Config (`configs/paths/default.yaml`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `paths.root_dir` | `${oc.env:PROJECT_ROOT}` | str | existing path | always | `python src/train.py paths.root_dir=/abs/project` |
| `paths.data_dir` | `${paths.root_dir}/data/` | str | path | always | `python src/train.py paths.data_dir=/abs/data` |
| `paths.log_dir` | `${paths.root_dir}/logs/` | str | path | always | `python src/train.py paths.log_dir=/abs/logs` |
| `paths.output_dir` | `${hydra:runtime.output_dir}` | str | hydra runtime path | always | `python src/train.py paths.output_dir=/tmp/out` |
| `paths.work_dir` | `${hydra:runtime.cwd}` | str | working directory path | always | `python src/train.py paths.work_dir=/tmp` |

## 12) Extras Config (`configs/extras/default.yaml`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `extras.ignore_warnings` | `False` | bool | `true/false` | `extras` enabled | `python src/train.py extras.ignore_warnings=true` |
| `extras.enforce_tags` | `True` | bool | `true/false` | `extras` enabled | `python src/train.py extras.enforce_tags=false` |
| `extras.print_config` | `True` | bool | `true/false` | `extras` enabled | `python src/train.py extras.print_config=false` |

## 13) Repo-Enforced Behavioral Constraints (From Code)

These constraints are not just documentation conventions; they are enforced in runtime code.

| Constraint | Enforced In | Behavior |
|---|---|---|
| `task.target_type` must be one of `binary`, `regression`, `survival` | `src/models/training/builders.py` | raises `ValueError` for unsupported target types |
| `task.instance_sampling` must be `all` or `random` | `src/models/training/builders.py` | raises `ValueError` for invalid sampling mode |
| `task.region_accumulation.hyperbatch_size >= 1` when enabled | `src/models/supervised_module.py` | raises `ValueError` |
| `task.node_aux.target_mode` must be `graph_label` or `attention_shaped_ti` | `src/models/training/loss_utils.py` | raises `ValueError` |
| `task.node_aux.loss_mode` must be `bce` or `weighted_bce` | `src/models/training/loss_utils.py` | raises `ValueError` |
| `task.entropy_reg.mode` must be `attention` or `attention_shaped_target` | `src/models/supervised_module.py` | raises `ValueError` |
| `conv_type=gine` requires `model.encoder.edge_attr_dim` and runtime `edge_attr` | `src/models/components/backbones.py` | raises `ValueError` |
| `conv_type=gat` requires `hidden_dim % gat_heads == 0` | `src/models/components/backbones.py` | raises `ValueError` |
| `model.encoder.use_edge_attr=true` is meaningful only for `gcn`/`gine` | `src/models/components/backbones.py` | warning/ignored for unsupported convs |
| `data.split.train_val_test_split` must sum to `1.0` | `src/data/components/precompute.py` split helpers | raises `ValueError` |
| `data.feature_reducer.fit_mode` must be `global` or `train_only` | `src/data/components/precompute.py` | raises `ValueError` |
| `data.sampler.runtime.weight_mode` must be `inverse`, `sqrt_inverse`, or `proportional` | `src/data/components/samplers.py` | raises `ValueError` |
| `model.encoder_init_map` must be `identity` or `auto_bgrl_or_identity` | `src/models/training/checkpoint_init.py` | raises `ValueError` |
| `model.init_from_ckpt` path must exist | checkpoint init helper | raises `FileNotFoundError` |

## 14) Trainer/Callback/Logger Exhaustiveness Note

This file is exhaustive for keys defined in repository config presets.

Hydra still allows passing additional Lightning kwargs that are not pre-declared in these YAML files. Those extra keys are valid if accepted by the target Lightning class.

Example:

```bash
python src/train.py trainer.log_every_n_steps=1 trainer.enable_progress_bar=true
```

Use this flexibility with caution and keep custom overrides version-controlled in experiment config files when possible.
