# Hyperparameter Reference

This reference covers repo-defined configuration keys and behavior for:

- `configs/train.yaml`
- `configs/eval.yaml`
- `configs/loocv.yaml`
- `configs/probe.yaml`, `configs/nested_cv.yaml`, `configs/extract.yaml`, `configs/scores.yaml`, `configs/niches.yaml`, `configs/metaclusters.yaml`, `configs/spatial_stats.yaml` (analysis CLIs, section 12b)
- `configs/regime/*.yaml` (training regime presets, section 12a)
- `configs/experiment/grass_mil_pretrain_a3.yaml`
- `configs/data/spatial_omics.yaml`
- `configs/task/*.yaml`
- `configs/model/**/*.yaml` (presets `grass_mil`, `bgrl_a3`, generic modules, `encoder/`, `attention/` groups)
- `configs/augmentation/*.yaml`
- `configs/data/sampler/*.yaml`
- `configs/optim/*.yaml`
- `configs/scheduler/*.yaml`
- `configs/trainer/*.yaml`
- `configs/callbacks/*.yaml`
- `configs/logger/*.yaml`
- `configs/debug/*.yaml`
- `configs/hydra/default.yaml`
- `configs/extras/default.yaml`

> Throughout this document `configs/...` is shorthand for
> `src/grass_mil/configs/...`. The config tree lives inside the package so it
> ships with the wheel; Hydra resolves it relative to the entrypoint module.

Override syntax examples use:

```bash
grass-mil-train key.path=value
```

## 1) Top-Level Train/Eval Composition Keys

## 1.1 `configs/train.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | composition list | list | Hydra config groups | always | `grass-mil-train trainer=gpu` |
| `task_name` | `train` | str | any string | always | `grass-mil-train task_name=my_run` |
| `tags` | `[dev]` | list[str] | any list | always | `grass-mil-train tags="[paper,ablation]"` |
| `train` | `True` | bool | `true/false` | always | `grass-mil-train train=false` |
| `test` | `True` | bool | `true/false` | after fit | `grass-mil-train test=false` |
| `ckpt_path` | `null` | str/null | path or null | resume training when set | `grass-mil-train ckpt_path=/abs/last.ckpt` |
| `seed` | `null` | int/null | integer or null | random seeding | `grass-mil-train seed=42` |

## 1.2 `configs/eval.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | composition list | list | Hydra config groups | always | `grass-mil-eval trainer=gpu` |
| `task_name` | `eval` | str | any string | always | `grass-mil-eval task_name=holdout_eval` |
| `tags` | `[dev]` | list[str] | any list | always | `grass-mil-eval tags="[final_eval]"` |
| `ckpt_path` | `???` | str | required absolute/relative checkpoint path | always | `grass-mil-eval ckpt_path=/abs/model.ckpt` |

## 1.3 `configs/loocv.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task_name` | `loocv` | str | any string | LOOCV entrypoint | `grass-mil-loocv task_name=loocv_run` |
| `loocv.mode` | `single` | str | `single`, `all` | LOOCV entrypoint | `grass-mil-loocv loocv.mode=all` |
| `loocv.fold_index` | `null` | int/null | `>=0` or null | single mode fold selection | `grass-mil-loocv loocv.mode=single loocv.fold_index=3` |
| `loocv.per_fold_processed_dir` | `true` | bool | `true/false` | fold isolation | `grass-mil-loocv loocv.per_fold_processed_dir=false` |
| `loocv.force_precompute_per_fold` | `true` | bool | `true/false` | leakage-safe fold preprocessing | `grass-mil-loocv loocv.force_precompute_per_fold=true` |
| `loocv.summary_filename` | `loocv_summary.json` | str | filename | fold summary artifact naming | `grass-mil-loocv loocv.summary_filename=my_summary.json` |

## 2) Data Config (`configs/data/spatial_omics.yaml`)

## 2.1 Datamodule Core

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data._target_` | `grass_mil.data.spatial_omics_datamodule.SpatialOmicsDataModule` | str | import path | always | `grass-mil-train data._target_=...` |
| `data.data_dir` | `${paths.data_dir}` | str | path | always | `grass-mil-train data.data_dir=/tmp/data` |
| `data.raw_manifest_path` | `${paths.data_dir}/raw/manifest.csv` | str | path | always | `grass-mil-train data.raw_manifest_path=/abs/manifest.csv` |
| `data.processed_dir` | `${paths.data_dir}/processed/spatial_omics` | str | path | always | `grass-mil-train data.processed_dir=/abs/processed` |
| `data.batch_size` | `1` | int | `>=1` | loader batching | `grass-mil-train data.batch_size=4` |
| `data.num_workers` | `0` | int | `>=0` | dataloader workers | `grass-mil-train data.num_workers=8` |
| `data.pin_memory` | `False` | bool | `true/false` | dataloader | `grass-mil-train data.pin_memory=true` |
| `data.coord_scale_um` | `1.0` | float | `>0` | preprocessing (`1.0` means input coords already in `um`) | `grass-mil-train data.coord_scale_um=0.5` |
| `data.sample_unit` | `tile` | str | `tile`/`full` | patch generation | `grass-mil-train data.sample_unit=full` |
| `data.reducer_scope` | `sample` | str | `sample`/`dataset` | reducer fitting | `grass-mil-train data.reducer_scope=dataset` |
| `data.keep_raw_molecular` | `false` | bool | `true/false` | precompute outputs | `grass-mil-train data.keep_raw_molecular=true` |
| `data.force_precompute` | `false` | bool | `true/false` | precompute trigger | `grass-mil-train data.force_precompute=true` |
| `data.min_cells` | `10` | int | `>=0` | patch filtering | `grass-mil-train data.min_cells=25` |
| `data.use_molecular_features` | `true` | bool | `true/false` | loader/reducer behavior | `grass-mil-train data.use_molecular_features=false` |

## 2.2 Categorical Feature Encoding

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.categorical_features.include_labels` | `[cell_type]` | list[str] | label names present in loaded data | attach `categorical_codes` | `grass-mil-train data.categorical_features.include_labels="[cell_type,state]"` |

## 2.3 Split Config

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.split.train_val_test_split` | `[0.8,0.1,0.1]` | list[float] len=3 | must sum to `1.0` | precompute split assignment | `grass-mil-train data.split.train_val_test_split="[0.7,0.15,0.15]"` |
| `data.split.split_by` | `sample` | str | `sample`/`region`/`patch` | split semantics | `grass-mil-train data.split.split_by=region` |
| `data.split.loocv.enabled` | `false` | bool | `true/false` | LOOCV override path | `grass-mil-train data.split.loocv.enabled=true` |
| `data.split.loocv.fold_unit` | `region` | str | `region`/`sample` | LOOCV override path | `grass-mil-train data.split.loocv.fold_unit=sample` |
| `data.split.loocv.holdout_id` | `null` | str/null | canonical fold id (`sample` or `sample::region`) | LOOCV override path | `grass-mil-train data.split.loocv.holdout_id=s1::r3` |
| `data.split.loocv.validation_strategy` | `heldout_fold_items` | str | `heldout_fold_items`/`patches_from_train_items` | LOOCV override path | `grass-mil-train data.split.loocv.validation_strategy=patches_from_train_items` |
| `data.split.loocv.val_ratio` | `0.1` | float | `[0.0,1.0)` | LOOCV validation split sizing | `grass-mil-train data.split.loocv.val_ratio=0.2` |
| `data.split.loocv.seed` | `42` | int | integer | LOOCV validation split determinism | `grass-mil-train data.split.loocv.seed=7` |

Notes:

- When LOOCV is disabled, behavior is identical to current non-LOOCV splitting.
- `data.coord_scale_um` is applied at preprocessing time; if changed, processed artifacts must be regenerated (`data.force_precompute=true` or a fresh `data.processed_dir`).
- Split labels are persisted at precompute time in `processed_index.json`.
  Changing `data.split.*` requires regenerating processed artifacts
  (`data.force_precompute=true` or a new `data.processed_dir`).
- For `reducer_scope=dataset` + `feature_reducer.fit_mode=train_only`, LOOCV requires fold-specific precompute (`force_precompute=true` or fold-specific `processed_dir`).
- For safe fold isolation, at least one of `loocv.force_precompute_per_fold` or `loocv.per_fold_processed_dir` must be `true`.
- In `loocv.mode=single`, fold selection can use either
  `data.split.loocv.holdout_id` or `loocv.fold_index`. If both are set,
  `loocv.fold_index` takes priority and a warning is emitted.

### Explicit splits (`data.split.explicit.*`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.split.explicit.train` | `null` | list/str | unit ids, or a `.json` / `.txt` file of ids | re-partition an existing precompute | `grass-mil-train data.split.explicit.train=train.json` |
| `data.split.explicit.val` | `null` | list/str | as above (may equal `test` for refits) | | `data.split.explicit.val=[r1,r2]` |
| `data.split.explicit.test` | `null` | list/str | as above | | `data.split.explicit.test=test.json` |
| `data.split.explicit.id_kind` | `region` | str | `region`, `sample`, `patch` | explicit split set | `data.split.explicit.id_kind=patch` |

Explicit lists override the persisted splits without re-processing the graphs
(ids not listed are dropped); they need `feature_reducer.name=identity` or
`fit_mode=global`. `grass-mil-nested-cv` and `grass-mil-probe` set them per fold.

## 2.4 Manifest Field Mapping

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.manifest.sample_id` | `sample_id` | str | manifest column name | manifest parse | `grass-mil-train data.manifest.sample_id=sid` |
| `data.manifest.input_path` | `input_path` | str | manifest column name | manifest parse | `grass-mil-train data.manifest.input_path=file_path` |
| `data.manifest.input_type` | `input_type` | str | manifest column name | manifest parse | `grass-mil-train data.manifest.input_type=source_type` |
| `data.manifest.region_id` | `region_id` | str | manifest column name | optional region grouping | `grass-mil-train data.manifest.region_id=roi` |
| `data.manifest.polygons_path` | `polygons_path` | str | manifest column name | optional polygons | `grass-mil-train data.manifest.polygons_path=poly_path` |

## 2.5 CSV Loader Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.csv.sep` | `,` | str | delimiter string | `input_type=csv/tsv` | `grass-mil-train data.csv.sep=';'` |
| `data.csv.coord_columns` | `[x,y]` | list[str] len=2 | existing columns | `input_type=csv/tsv` | `grass-mil-train data.csv.coord_columns="[x_um,y_um]"` |
| `data.csv.cell_id_column` | `cell_id` | str/null | existing column or null | `input_type=csv/tsv` | `grass-mil-train data.csv.cell_id_column=id` |
| `data.csv.categorical_label_columns` | `[]` | list[str] | existing columns | `input_type=csv/tsv` | `grass-mil-train data.csv.categorical_label_columns="[cell_type]"` |
| `data.csv.molecular_columns` | `null` | list[str]/null | existing columns or null infer | `input_type=csv/tsv` and `use_molecular_features=true` | `grass-mil-train data.csv.molecular_columns="[gene1,gene2]"` |

## 2.6 H5AD Loader Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.h5ad.coord_source` | `obsm` | str | `obs`/`obsm` | `input_type=h5ad` | `grass-mil-train data.h5ad.coord_source=obs` |
| `data.h5ad.coord_key` | `spatial` | str/null | obsm key | `coord_source=obsm` | `grass-mil-train data.h5ad.coord_key=X_spatial` |
| `data.h5ad.coord_columns` | `[x,y]` | list[str] len=2 | obs columns | `coord_source=obs` | `grass-mil-train data.h5ad.coord_columns="[x_um,y_um]"` |
| `data.h5ad.cell_id_column` | `null` | str/null | obs column or null | `input_type=h5ad` | `grass-mil-train data.h5ad.cell_id_column=cell_id` |
| `data.h5ad.categorical_label_columns` | `[]` | list[str] | obs columns | `input_type=h5ad` | `grass-mil-train data.h5ad.categorical_label_columns="[cell_type]"` |
| `data.h5ad.molecular_layer` | `null` | str/null | layer key or null (`X`) | `use_molecular_features=true` | `grass-mil-train data.h5ad.molecular_layer=counts` |
| `data.h5ad.molecular_features` | `null` | list[str]/null | var names subset | `use_molecular_features=true` | `grass-mil-train data.h5ad.molecular_features="[CD3,CD8]"` |

## 2.7 SCE Loader Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.sce.assay_name` | `null` | str/null | assay name or default assay | `input_type=sce/rds` | `grass-mil-train data.sce.assay_name=logcounts` |
| `data.sce.coord_source` | `colData` | str | `colData`/`reducedDims` | `input_type=sce/rds` | `grass-mil-train data.sce.coord_source=reducedDims` |
| `data.sce.coord_key` | `null` | str/null | reducedDims key | `coord_source=reducedDims` | `grass-mil-train data.sce.coord_key=spatial` |
| `data.sce.coord_columns` | `[x,y]` | list[str]/null | colData columns | `coord_source=colData` | `grass-mil-train data.sce.coord_columns="[x_um,y_um]"` |
| `data.sce.cell_id_column` | `null` | str/null | colData column or colnames fallback | `input_type=sce/rds` | `grass-mil-train data.sce.cell_id_column=cell_id` |
| `data.sce.categorical_label_columns` | `[]` | list[str] | colData columns | `input_type=sce/rds` | `grass-mil-train data.sce.categorical_label_columns="[cell_type]"` |
| `data.sce.transpose_assay` | `true` | bool | `true/false` | molecular matrix orientation | `grass-mil-train data.sce.transpose_assay=false` |
| `data.sce.molecular_features` | `null` | list[str]/null | rowname subset | `use_molecular_features=true` | `grass-mil-train data.sce.molecular_features="[geneA,geneB]"` |

## 2.8 Graph Builder Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.graph_builder.name` | `delaunay` | str | registered builder name/import path | graph construction | `grass-mil-train data.graph_builder.name=delaunay` |
| `data.graph_builder.kwargs.edge_features` | `[distance,neighbor]` | list[str] | `distance`, `neighbor` | `delaunay` builder | `grass-mil-train data.graph_builder.kwargs.edge_features="[distance]"` |
| `data.graph_builder.kwargs.neighbor_cutoff_um` | `20.0` | float | `>0` | when `neighbor` edge feature enabled | `grass-mil-train data.graph_builder.kwargs.neighbor_cutoff_um=50.0` |

## 2.9 Feature Reducer Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.feature_reducer.name` | `identity` | str | `identity`, `pca`, custom import path | molecular reduction | `grass-mil-train data.feature_reducer.name=pca` |
| `data.feature_reducer.fit_mode` | `train_only` | str | `train_only`/`global` | dataset-scope reducer | `grass-mil-train data.feature_reducer.fit_mode=global` |
| `data.feature_reducer.kwargs` | `{}` | dict | reducer kwargs | reducer-specific | `grass-mil-train data.feature_reducer.kwargs.n_components=64` |

## 2.10 Sampler Keys

`data.sampler` is a config group (`data/sampler=shadow_custom` by default,
`ego_graph_mil` for MIL fine-tuning, `ego_graph_pretrain` for BGRL). Single
keys are overridden as `data.sampler.runtime.<key>=...`.

| Key | Default (`shadow_custom`) | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.sampler.name` | `shadow_custom` | str | `identity`, `shadow_native`, `shadow_custom` | dataloader build | `grass-mil-train data.sampler.name=identity` |
| `data.sampler.kwargs` | `{}` | dict | strategy kwargs | strategy-specific | `grass-mil-train data.sampler.kwargs='{}'` |
| `data.sampler.runtime.enabled` | `true` | bool | `true/false` | shadow strategies | `grass-mil-train data.sampler.runtime.enabled=false` |
| `data.sampler.runtime.depth` | `2` (`4` in `ego_graph_*`) | int | `>=1` | ego-graph hops | `grass-mil-train data.sampler.runtime.depth=3` |
| `data.sampler.runtime.num_neighbors` | `8` (`-1` in `ego_graph_*`) | int | `>=1`, or `-1` = every neighbour at every hop | shadow runtime | `grass-mil-train data.sampler.runtime.num_neighbors=-1` |
| `data.sampler.runtime.subgraph_batch_size` | `32` | int | `>=1` | ego-graphs per chunk (upper bound when `unit_as_bag`) | `grass-mil-train data.sampler.runtime.subgraph_batch_size=64` |
| `data.sampler.runtime.replace` | `false` | bool | `true/false` | root sampling | `grass-mil-train data.sampler.runtime.replace=true` |
| `data.sampler.runtime.shuffle_subgraphs` | `true` | bool | `true/false` | training shuffle behavior | `grass-mil-train data.sampler.runtime.shuffle_subgraphs=false` |
| `data.sampler.runtime.proportional_root_sampling` | `true` (`false` in `ego_graph_*`) | bool | `true/false` | cell-type weighted roots | `grass-mil-train data.sampler.runtime.proportional_root_sampling=false` |
| `data.sampler.runtime.property_name` | `cell_type` | str | categorical label present in `categorical_slices` | weighted root sampling | `grass-mil-train data.sampler.runtime.property_name=cell_type` |
| `data.sampler.runtime.weight_mode` | `inverse` | str | `inverse`, `sqrt_inverse`, `proportional` | weighted root sampling | `grass-mil-train data.sampler.runtime.weight_mode=sqrt_inverse` |
| `data.sampler.runtime.min_weight` | `1.0e-6` | float | `>0` | weighted root sampling | `grass-mil-train data.sampler.runtime.min_weight=1.0e-4` |
| `data.sampler.runtime.subsample_fraction` | `1.0` | float | `(0, 1]` | fraction of candidate roots per unit per epoch | `grass-mil-train data.sampler.runtime.subsample_fraction=0.5` |
| `data.sampler.runtime.subsample_count` | `null` (`256` in `ego_graph_mil`) | int/null | `>=1` or null | absolute roots per unit per epoch (uniform, without replacement; overrides the fraction) | `grass-mil-train data.sampler.runtime.subsample_count=512` |
| `data.sampler.runtime.subsample_seed` | `null` | int/null | | seed of the root subsampling generator | `grass-mil-train data.sampler.runtime.subsample_seed=0` |
| `data.sampler.runtime.fixed_per_unit_seed` | `null` | int/null | | identical root subset per unit on every pass (validation) | `grass-mil-train data.val_sampler.runtime.fixed_per_unit_seed=0` |
| `data.sampler.runtime.unit_as_bag` | `false` (`true` in `ego_graph_mil`) | bool | `true/false` | one batch per unit holding all its selected roots (one bag, one softmax, one loss) | `grass-mil-train data.sampler.runtime.unit_as_bag=true` |
| `data.sampler.runtime.merge_units` | `false` (`true` in `ego_graph_pretrain`) | bool | `true/false` | pool all units into one graph and draw roots from the union (BGRL) | `grass-mil-train data.sampler.runtime.merge_units=true` |
| `data.sampler.runtime.interior_seeds.enabled` | `false` | bool | `true/false` | shadow runtime | `grass-mil-train data.sampler.runtime.interior_seeds.enabled=true` |
| `data.sampler.runtime.interior_seeds.n_hops` | `null` | int/null | `>=1`, `null` uses `depth` | `interior_seeds.enabled=true` | `grass-mil-train data.sampler.runtime.interior_seeds.n_hops=3` |
| `data.sampler.runtime.interior_seeds.seed_tolerance_multiplier` | `1.0` | float | `>=0` | `interior_seeds.enabled=true` | `grass-mil-train data.sampler.runtime.interior_seeds.seed_tolerance_multiplier=1.5` |
| `data.sampler.runtime.interior_seeds.keep_largest_component` | `true` | bool | `true/false` | `interior_seeds.enabled=true` | `grass-mil-train data.sampler.runtime.interior_seeds.keep_largest_component=false` |
| `data.sampler.runtime.interior_seeds.cache_dir` | `null` | str/null | path, `null` disables caching | `interior_seeds.enabled=true` | `grass-mil-train data.sampler.runtime.interior_seeds.cache_dir=.cache/seeds` |
| `data.sampler.runtime.ego_radius.enabled` | `false` | bool | `true/false` | shadow runtime | `grass-mil-train data.sampler.runtime.ego_radius.enabled=true` |
| `data.sampler.runtime.ego_radius.radius_per_hop` | `75.0` | float | `>0`, datamodule units | `ego_radius.enabled=true` | `grass-mil-train data.sampler.runtime.ego_radius.radius_per_hop=30.0` |
| `data.sampler.runtime.ego_radius.radius_offset` | `55.0` | float | `>=0`, datamodule units | `ego_radius.enabled=true` | `grass-mil-train data.sampler.runtime.ego_radius.radius_offset=22.0` |
| `data.sampler.runtime.ego_radius.keep_root_component` | `true` | bool | `true/false` | `ego_radius.enabled=true` | `grass-mil-train data.sampler.runtime.ego_radius.keep_root_component=false` |
| `data.val_sampler` | `null` (fixed 1024-root subset in `regime=mil`) | dict/null | full sampler config | validation loader | see `configs/regime/mil.yaml` |
| `data.test_sampler` | `null` | dict/null | full sampler config; `null` derives an all-roots, unweighted, no-bagging variant of `sampler` | test / predict loaders | `grass-mil-train data.test_sampler.runtime.subsample_count=1024` |

Two notes on these, both of which cause silent misconfiguration rather than errors:

- **`interior_seeds`** restricts ego-graph roots to the tissue interior.
  Ego-graphs rooted at the section edge are truncated by the slide boundary
  rather than by biology, so they are not comparable to interior ones. Only the
  `n_hops` rim around the convex hull is excluded; internal holes are not
  treated as boundary. The interior mask is intersected into the candidate pool
  *before* class-proportional weighting.
- **`ego_radius` distances are in datamodule coordinate units** (µm after
  `coord_scale_um`), **not pixels**. The `75.0`/`55.0` defaults come from the
  NSCLC cohort at its native pixel scale; convert your cohort's values before
  setting them, or the cap will be wrong by whatever the scale factor is.

## 2.11 Tiling And Graph Label Keys

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `data.tiling.tile_size_um` | `200.0` | float | `>0` | `sample_unit=tile` | `grass-mil-train data.tiling.tile_size_um=300.0` |
| `data.tiling.stride_um` | `200.0` | float | `>0` | `sample_unit=tile` | `grass-mil-train data.tiling.stride_um=150.0` |
| `data.tiling.min_cells` | `10` | int | `>=0` | tile retention | `grass-mil-train data.tiling.min_cells=20` |
| `data.graph_labels.label_file` | `null` | str/null | CSV path or null | optional graph labels | `grass-mil-train data.graph_labels.label_file=/abs/labels.csv` |
| `data.graph_labels.id_column` | `id` | str | column in label file | graph labels enabled | `grass-mil-train data.graph_labels.id_column=region_id` |
| `data.graph_labels.tasks` | `null` | list[str]/null | task columns or null for all | graph labels enabled | `grass-mil-train data.graph_labels.tasks="[outcome]"` |
| `data.graph_labels.scope` | `region` | str | `region`, `patch`, `sample` | graph labels enabled | `grass-mil-train data.graph_labels.scope=sample` |
| `data.graph_labels.patient_column` | `null` | str/null | column of the label file | graph labels enabled | `grass-mil-train data.graph_labels.patient_column=PATIENT` (attached as `data.patient_id`, recorded in the index; never a target) |
| `data.transforms` | `[]` | list[dict] | importable transform specs | transform pipeline | `grass-mil-train data.transforms='[{name: grass_mil.data.components.transforms.CompositionVector, kwargs: {label_attr: label_cell_type}}]'` |

## 2.12 Sampler Preset Files (`configs/data/sampler/*.yaml`)

| File | Keys In File | Defaults |
|---|---|---|
| `configs/data/sampler/identity.yaml` | `name`, `kwargs` | `name=identity`, `kwargs={}` |
| `configs/data/sampler/shadow_native.yaml` | `name`, `kwargs`, `runtime.*` | `name=shadow_native`, runtime defaults mirror `data.sampler.runtime.*` |
| `configs/data/sampler/shadow_custom.yaml` | `name`, `kwargs`, `runtime.*` | `name=shadow_custom`, runtime defaults mirror `data.sampler.runtime.*` |
| `configs/data/sampler/ego_graph_mil.yaml` | `name`, `kwargs`, `runtime.*` | GRASS-MIL fine-tuning: depth 4, unlimited neighbours, uniform roots, `subsample_count=256`, `unit_as_bag=true` |
| `configs/data/sampler/ego_graph_pretrain.yaml` | `name`, `kwargs`, `runtime.*` | BGRL: depth 4, unlimited neighbours, 64 roots per step from the union of all units (`merge_units=true`) |

## 3) Task Config Group (`configs/task/*.yaml`)

## 3.1 `configs/task/finetune_mean.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task.name` | `finetune_mean` | str | label | always | `grass-mil-train task.name=custom_mean` |
| `task.aggregation` | `mean` | str | `mean` | supervised module | `grass-mil-train task.aggregation=mean` |
| `task.bag_key` | `region_id` | str | batch attribute name | supervised bagging | `grass-mil-train task.bag_key=sample_id` |
| `task.bag_fallback_key` | `sample_id` | str | batch attribute name | bag key missing fallback | `grass-mil-train task.bag_fallback_key=patch_id` |
| `task.max_instances_per_bag` | `0` | int | `0` means all, otherwise `>=1` | bag aggregation | `grass-mil-train task.max_instances_per_bag=64` |
| `task.instance_sampling` | `all` | str | `all`, `random` | if max instances > 0 | `grass-mil-train task.instance_sampling=random` |
| `task.target_type` | `binary` | str | `binary`, `regression`, `survival` | loss routing | `grass-mil-train task.target_type=regression` |
| `task.target_columns` | `null` | list[str]/null | label names | target selection | `grass-mil-train task.target_columns="[outcome]"` |
| `task.loss` | `categorical_bce` | str | model loss presets | loss build | `grass-mil-train task.loss=regression_mse` |
| `task.lr_warmup.enabled` | `false` | bool | `true/false` | scheduler wrapping | `grass-mil-train task.lr_warmup.enabled=true` |
| `task.lr_warmup.warmup_steps` | `0` | int | `>=0` | warmup enabled | `grass-mil-train task.lr_warmup.warmup_steps=100` |
| `task.lr_warmup.start_factor` | `0.1` | float | `(0,1]` (clamped) | warmup enabled | `grass-mil-train task.lr_warmup.start_factor=0.2` |

## 3.2 `configs/task/finetune_mil.yaml`

Includes the keys of 3.1 (with `aggregation=mil_attention`, `target_type=categorical`,
`loss=categorical_ce`, and per-class widths `model.graph_head.output_dim=2`,
`model.attention.n_classes=2`) plus the MIL controls below. The `regime=`
presets (section 12a) set them to the preprint values.

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task.bag_key` | `region_id` | str | `region_id`, `patch_id`, `sample_id` | what one bag is | `grass-mil-train task.bag_key=patch_id` |
| `task.class_weights` | `auto_train` | str/list/null | `auto_train` = inverse frequency on the training units (mean 1); a list; `null` | categorical loss | `grass-mil-train task.class_weights=[1.0,2.0]` |
| `task.optimization.encoder_lr_scale` | `1.0` | float | `>=0`; `0` freezes the encoder (no gradients, eval mode) | optimizer groups | `grass-mil-train task.optimization.encoder_lr_scale=0.3` |
| `task.optimization.head_lr_scale` | `1.0` | float | `>=0` | optimizer groups | `grass-mil-train task.optimization.head_lr_scale=0.01` |
| `task.optimization.attention_lr_scale` | `2.0` | float | `>=0` | optimizer groups | `grass-mil-train task.optimization.attention_lr_scale=1.0` |
| `task.optimization.head_weight_decay` | `null` | float/null | overrides `optim.weight_decay` for the head group | optimizer groups | `grass-mil-train task.optimization.head_weight_decay=1e-3` |
| `task.optimization.backbone_lr` / `attention_lr` | `null` | float/null | legacy absolute pair (both or neither); ignored when scales are used | | |
| `task.finetune_augmentation.drop_edge_p` | `0.0` | float | `[0,1]` | training batches only; edges dropped with both directions | `grass-mil-train task.finetune_augmentation.drop_edge_p=0.05` |
| `task.finetune_augmentation.drop_feat_p` | `0.0` | float | `[0,1]` | elementwise feature drop, writes the unassigned cell-type token | `grass-mil-train task.finetune_augmentation.drop_feat_p=0.05` |
| `task.finetune_augmentation.seed` | `0` | int | | augmentation generator | |
| `task.head_init_path` | `null` | str/null | probe head file (`graph_head.net.0.*`), or `auto` (resolved per split by `grass-mil-nested-cv`) | linear head (`model.graph_head.num_layers=1`) | `grass-mil-train task.head_init_path=probe_head.pt` |
| `task.freeze_head` | `false` | bool | | head excluded from the optimizer | `grass-mil-train task.freeze_head=true` |
| `task.grad_clip_norm` | `1.0` | float/null | `>0` or null | gradient-norm clipping in the region flush | `grass-mil-train task.grad_clip_norm=null` |
| `task.scheduler_interval` | `step` | str | `step`, `epoch` | with `scheduler=cosine_mil_step` (`T_max: auto`) the cosine spans every optimizer step of the run; skipped AMP steps do not advance it | `grass-mil-train task.scheduler_interval=epoch` |
| `task.region_accumulation.enabled` | `true` | bool | `true/false` | MIL training | `grass-mil-train task.region_accumulation.enabled=false` |
| `task.region_accumulation.hyperbatch_size` | `12` | int | `>=1` | regions per optimizer step | `grass-mil-train task.region_accumulation.hyperbatch_size=8` |
| `task.region_accumulation.flush_on_epoch_end` | `true` | bool | | | |
| `task.max_instances_per_bag` / `instance_sampling` | `0` / `all` | | as in 3.1 | | |

Region-level metrics are logged per epoch (`val/loss_region`, `val/auroc`,
`val/bal_acc`, `val/acc`, `val/pred_pos_frac`; the same under `test/`) and written
to `metrics_per_epoch.jsonl`; `test_predictions.csv` holds one row per bag.

## 3.3 `configs/task/pretrain_bgrl.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | `override /scheduler: cosine_step` | Hydra defaults entry | scheduler group override | pretrain task composition | `grass-mil-train task=pretrain_bgrl model=bgrl_module` |
| `task.name` | `pretrain_bgrl` | str | label | always | `grass-mil-train task.name=ssl_run` |
| `task.aggregation` | `mean` | str | `mean` | BGRL module ignores bag loss semantics but keeps task schema | `grass-mil-train task.aggregation=mean` |
| `task.bag_key` | `region_id` | str | batch attr | task metadata | `grass-mil-train task.bag_key=sample_id` |
| `task.bag_fallback_key` | `sample_id` | str | batch attr | task metadata | `grass-mil-train task.bag_fallback_key=patch_id` |
| `task.max_instances_per_bag` | `0` | int | `0`/`>=1` | task metadata | `grass-mil-train task.max_instances_per_bag=0` |
| `task.instance_sampling` | `all` | str | `all`/`random` | task metadata | `grass-mil-train task.instance_sampling=all` |
| `task.target_type` | `binary` | str | task labels | task metadata | `grass-mil-train task.target_type=binary` |
| `task.target_columns` | `null` | list/null | optional | task metadata | `grass-mil-train task.target_columns='[label]'` |
| `task.loss` | `categorical_bce` | str | loss label | task metadata | `grass-mil-train task.loss=categorical_bce` |
| `task.lr_warmup.enabled` | `true` | bool | `true/false` | optimizer scheduler warmup | `grass-mil-train task.lr_warmup.enabled=true` |
| `task.lr_warmup.warmup_steps` | `1000` | int | `>=0` | warmup enabled | `grass-mil-train task.lr_warmup.warmup_steps=500` |
| `task.lr_warmup.start_factor` | `0.1` | float | `(0,1]` | warmup enabled | `grass-mil-train task.lr_warmup.start_factor=0.2` |
| `task.warmup_steps` | `1000` | int | `>=0` | BGRL momentum scheduler | `grass-mil-train task.warmup_steps=500` |
| `task.total_steps` | `10000` | int | `>0` preferred | BGRL momentum scheduler | `grass-mil-train task.total_steps=20000` |
| `task.momentum` | `0.99` | float | `[0,1]` practical range | target network EMA | `grass-mil-train task.momentum=0.996` |
| `task.momentum_min` | `0.99` | float | `[0,1]` practical range | cosine momentum floor | `grass-mil-train task.momentum_min=0.9` |
| `task.drop_edge_p1` | `0.2` | float | `[0,1]` | view-1 edge dropout | `grass-mil-train task.drop_edge_p1=0.1` |
| `task.drop_edge_p2` | `0.2` | float | `[0,1]` | view-2 edge dropout | `grass-mil-train task.drop_edge_p2=0.3` |
| `task.drop_feat_p1` | `0.2` | float | `[0,1]` | view-1 feature dropout | `grass-mil-train task.drop_feat_p1=0.1` |
| `task.drop_feat_p2` | `0.2` | float | `[0,1]` | view-2 feature dropout | `grass-mil-train task.drop_feat_p2=0.3` |

`task.momentum_min` is the legacy name of `task.momentum_final`.

## 3.4 `configs/task/pretrain_bgrl_a3.yaml` (GRASS-MIL pretraining)

Selected by `experiment=grass_mil_pretrain_a3` (with `model=bgrl_a3`,
`callbacks=pretrain_checkpoint`, `data/sampler=ego_graph_pretrain`).

| Key | Default | Meaning |
|---|---|---|
| `optim.lr` / `optim.weight_decay` | `5e-5` / `1e-5` | AdamW |
| `task.lr_warmup.*` | enabled, `1000` steps, `start_factor=0.0` | linear warmup, then cosine |
| `scheduler.T_max` | `auto` | `= total_steps - warmup` so the LR reaches `eta_min=1e-7` at the end |
| `task.total_steps` | `200000` | optimizer steps (`trainer.max_steps` follows it) |
| `task.momentum` / `task.momentum_final` | `0.99` / `1.0` | target-encoder EMA, cosine, applied after every optimizer step |
| `task.momentum_warmup_steps` | `0` | no momentum warmup (`null` falls back to `task.warmup_steps`) |
| `task.augmentation` | `mode=importance`, `mu=0.25`, `p_lambda=0.45`, `feature_noise_std=0.2`, `feature_noise_columns=[SIZE]` | symmetric views: degree-importance drop with the unassigned token + Gaussian noise on the named continuous feature(s); set `[]` without a size feature |
| `model.encoder.dropout` | `0.1` (in `model=bgrl_a3`) | encoder dropout during pretraining only |
| `callbacks.model_checkpoint.every_n_train_steps` | `5000` | all step checkpoints kept, plus `last.ckpt` |
| `data.split.train_val_test_split` | `[1,0,0]` | every region used label-free |

## 4) Model Config Group (`configs/model/**/*.yaml`)

## 4.1 `configs/model/supervised_module.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model._target_` | `grass_mil.models.supervised_module.SupervisedModule` | str | import path | supervised runs | `grass-mil-train model=supervised_module` |
| `model._recursive_` | `false` | bool | `true/false` | Hydra instantiate semantics | `grass-mil-train model._recursive_=false` |
| `model.flags.use_ssl` | `false` | bool | `true/false` | optional SSL wrapper in supervised module | `grass-mil-train model.flags.use_ssl=true` |
| `model.encoder.input_dim` | `0` | int | `>0` or `0` for infer | encoder build | `grass-mil-train model.encoder.input_dim=128` |
| `model.encoder.hidden_dim` | `128` | int | `>=1` | encoder | `grass-mil-train model.encoder.hidden_dim=256` |
| `model.encoder.out_dim` | `128` | int/null | `>=1` or null | encoder projection | `grass-mil-train model.encoder.out_dim=256` |
| `model.encoder.num_layers` | `3` | int | `>=1` | encoder | `grass-mil-train model.encoder.num_layers=4` |
| `model.encoder.dropout` | `0.1` | float | `[0,1]` | encoder | `grass-mil-train model.encoder.dropout=0.2` |
| `model.encoder.conv_type` | `gin` | str | `gin`, `gcn`, `gat`, `graphsage`, `gine` | encoder | `grass-mil-train model.encoder.conv_type=gine` |
| `model.encoder.norm` | `batchnorm` | str | `batchnorm`, `layernorm`, `none` | encoder | `grass-mil-train model.encoder.norm=layernorm` |
| `model.encoder.jk` | `last` | str | `last`, `concat`, `max`, `sum` | encoder | `grass-mil-train model.encoder.jk=concat` |
| `model.encoder.act` | `relu` | str | `relu`, `gelu`, `leaky_relu` | encoder | `grass-mil-train model.encoder.act=gelu` |
| `model.encoder.pooling` | `mean` | str | `sum`,`mean`,`max` | graph embedding | `grass-mil-train model.encoder.pooling=max` |
| `model.encoder.gat_heads` | `4` | int | `>=1` and divides hidden dim | `conv_type=gat` | `grass-mil-train model.encoder.gat_heads=8` |
| `model.encoder.use_edge_attr` | `false` | bool | `true/false` | `gcn`/`gine` meaningful | `grass-mil-train model.encoder.use_edge_attr=true` |
| `model.encoder.edge_weight_index` | `0` | int | column index | `gcn` with edge attrs | `grass-mil-train model.encoder.edge_weight_index=1` |
| `model.encoder.edge_attr_dim` | `null` | int/null | required for `gine` | `conv_type=gine` | `grass-mil-train model.encoder.edge_attr_dim=2` |
| `model.encoder.edge_feature_index` | `null` | int/null | column of `edge_attr` fed to the convolution (`gine_length`: `0`) | `conv_type=gine` | `grass-mil-train model.encoder.edge_feature_index=0` |
| `model.encoder.train_eps` | `true` | bool | learnable GIN/GINE epsilon | `gin`/`gine` | `grass-mil-train model.encoder.train_eps=false` |
| `model.encoder.input_init` | `xavier` | str | `xavier`, `default` | input embedding + linear init | `grass-mil-train model.encoder.input_init=default` |
| `model.encoder.categorical_embedding.label` | `cell_type` | str | categorical label name | summed cell-type embedding | |
| `model.encoder.categorical_embedding.num_embeddings` / `column_index` | `null` | int/null | resolved from the processed dataset | | |
| `model.encoder.categorical_embedding.reserve_unassigned` | `true` | bool | reserve a trailing row as the unassigned token | augmentations | |
| `model.encoder.categorical_embedding.unassigned_label` / `unassigned_index` | `null` | str/int/null | reuse an existing category (e.g. `other`) as the unassigned token | | `grass-mil-train model.encoder.categorical_embedding.unassigned_label=other` |
| `model.attention.attention_type` | `gated_projected` | str | `gated`, `gated_projected` | `task.aggregation=mil_attention` | `grass-mil-train model.attention.attention_type=gated` |
| `model.attention.input_dim` | `128` | int | `>=1` | MIL attention | `grass-mil-train model.attention.input_dim=256` |
| `model.attention.projection_dim` | `32` | int | `>=1` | projected attention | `grass-mil-train model.attention.projection_dim=64` |
| `model.attention.hidden_dim` | `8` | int | `>=1` | MIL attention (the "attention space" width) | `grass-mil-train model.attention.hidden_dim=16` |
| `model.attention.dropout` | `true` | bool | `true/false` | dropout 0.25 in the attention net while training | `grass-mil-train model.attention.dropout=false` |
| `model.attention.init` | `kaiming` | str | `kaiming`, `default` | attention linears init | `grass-mil-train model.attention.init=default` |
| `model.attention.n_classes` | `1` (`2` with `task=finetune_mil`) | int | `>=1` | one attention channel per class | `grass-mil-train model.attention.n_classes=3` |
| `model.graph_head.input_dim` | `128` | int | `>=1` | graph head | `grass-mil-train model.graph_head.input_dim=256` |
| `model.graph_head.output_dim` | `1` | int | `>=1` | graph head | `grass-mil-train model.graph_head.output_dim=2` |
| `model.graph_head.hidden_dim` | `128` | int | `>=1` | graph head | `grass-mil-train model.graph_head.hidden_dim=256` |
| `model.graph_head.num_layers` | `2` | int | `>=1`; `1` = linear instance head (required by the probe regimes) | graph head | `grass-mil-train model.graph_head.num_layers=1` |
| `model.graph_head.dropout` | `0.0` | float | `[0,1]` | graph head | `grass-mil-train model.graph_head.dropout=0.2` |
| `model.ssl.method` | `bgrl` | str | `bgrl` | SSL factory | `grass-mil-train model.ssl.method=bgrl` |
| `model.ssl.predictor.hidden_size` | `512` | int | `>=1` | SSL enabled | `grass-mil-train model.ssl.predictor.hidden_size=256` |
| `model.loss.loss_type` | `${task.loss}` | str | loss name | loss factory | `grass-mil-train task.loss=regression_mse` |
| `model.optim` | `${optim}` | DictConfig | optimizer config object | always | `grass-mil-train optim.lr=5e-4` |
| `model.scheduler` | `${scheduler}` | DictConfig | scheduler config object | always | `grass-mil-train scheduler.T_max=200` |
| `model.task` | `${task}` | DictConfig | task config object | always | `grass-mil-train task=finetune_mil` |
| `model.init_from_ckpt` | `null` | str/null | checkpoint path | optional init | `grass-mil-train model.init_from_ckpt=/abs/pretrain.ckpt` |
| `model.init_strict` | `false` | bool | `true/false` | checkpoint init | `grass-mil-train model.init_strict=true` |
| `model.encoder_init_map` | `auto_bgrl_or_identity` | str | `identity`, `auto_bgrl_or_identity`, `spacegm_bgrl` (legacy SPACE-GM BGRL checkpoints) | checkpoint init | `grass-mil-train model.encoder_init_map=spacegm_bgrl` |
| `model.freeze_encoder` | `false` | bool | `true/false` | checkpoint transfer/finetuning | `grass-mil-train model.freeze_encoder=true` |

## 4.2 `configs/model/bgrl_module.yaml`

Same encoder block as supervised module plus SSL/pretrain init keys:

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model._target_` | `grass_mil.models.bgrl_module.BGRLModule` | str | import path | pretrain task | `grass-mil-train model=bgrl_module` |
| `model._recursive_` | `false` | bool | `true/false` | instantiate behavior | `grass-mil-train model._recursive_=false` |
| `model.encoder.*` | same as supervised defaults | mixed | see section 4.1 | BGRL encoder | `grass-mil-train model.encoder.conv_type=gin` |
| `model.ssl.method` | `bgrl` | str | `bgrl` | SSL module | `grass-mil-train model.ssl.method=bgrl` |
| `model.ssl.predictor.hidden_size` | `512` | int | `>=1` | predictor width | `grass-mil-train model.ssl.predictor.hidden_size=1024` |
| `model.optim` | `${optim}` | DictConfig | optimizer object | always | `grass-mil-train optim.lr=1e-3` |
| `model.scheduler` | `${scheduler}` | DictConfig | scheduler object | always | `grass-mil-train scheduler.T_max=20000` |
| `model.task` | `${task}` | DictConfig | pretrain task object | always | `grass-mil-train task=pretrain_bgrl model=bgrl_module` |
| `model.init_from_ckpt` | `null` | str/null | checkpoint path | optional init | `grass-mil-train model.init_from_ckpt=/abs/ssl.ckpt` |
| `model.init_strict` | `false` | bool | `true/false` | checkpoint init | `grass-mil-train model.init_strict=true` |
| `model.encoder_init_map` | `auto_bgrl_or_identity` | str | `identity`, `auto_bgrl_or_identity` | checkpoint init | `grass-mil-train model.encoder_init_map=identity` |

## 4.3 Model Presets And Component Groups

### `configs/model/grass_mil.yaml` (GRASS-MIL supervised model)

`defaults: [encoder: gine_length, attention: gated_projected]`, MLP instance head
(`graph_head.num_layers=2`, hidden = `encoder.out_dim`), bag logit pooled in logit
space (`L_c = sum_i A[i,c] f_c(h_i)`). Use `model=grass_mil task=finetune_mil` or a
`regime=` preset. `model.graph_head.num_layers=1` gives the linear head of the
probe regimes.

### `configs/model/bgrl_a3.yaml` (GRASS-MIL pretraining model)

`defaults: [encoder: gine_length]`, `encoder.dropout=0.1`, predictor hidden 512.

### `configs/model/supervised_module.yaml`, `configs/model/bgrl_module.yaml`

Generic modules: `defaults: [encoder: gin, attention: gated_projected]`. Switch
the encoder with `model/encoder=gine_length` and the attention with
`model/attention=gated`.

Attention is **not** toggled by a model flag. `SupervisedModule` derives it from
the task: `use_attention = (task.aggregation == "mil_attention")`. Use
`task=finetune_mil` to enable it and `task=finetune_mean` to disable it. The
component sub-configs (`model.encoder`, `model.attention`, `model.graph_head`,
`model.ssl.predictor`) are plain option dicts consumed by
`build_supervised_components`, not `_target_` instantiations: select another
attention implementation with `model.attention.attention_type=gated`.

`task=finetune_mil` sets the per-class widths itself
(`model.graph_head.output_dim=2`, `model.attention.n_classes=2`); `task` is merged
after `model` in the root defaults list (see `configs/train.yaml`).

### Encoder group (`configs/model/encoder/*.yaml`)

Files: `gin`, `gcn`, `gat`, `graphsage`, `gine`, `gine_length`. All expose the
keys of section 4.1; per-file defaults differ in `conv_type`, `norm`,
`use_edge_attr`, `edge_attr_dim`:

- `gin.yaml`: `conv_type=gin`, `norm=batchnorm`, `use_edge_attr=false`
- `gcn.yaml`: `conv_type=gcn`, `norm=batchnorm`, `use_edge_attr=true`
- `gat.yaml`: `conv_type=gat`, `norm=batchnorm`
- `graphsage.yaml`: `conv_type=graphsage`, `norm=batchnorm`
- `gine.yaml`: `conv_type=gine`, `norm=batchnorm`, `use_edge_attr=true`, `edge_attr_dim=2`
- `gine_length.yaml` (GRASS-MIL): `conv_type=gine`, 4 x 256, `norm=layernorm`,
  `pooling=max`, `edge_attr_dim=1`, `edge_feature_index=0` (the bounded log
  edge length), `train_eps=true`, `input_init=xavier`, `categorical_embedding`
  with a reserved unassigned row. Supports cell-type-only cohorts (`input_dim=0`).

### Attention group (`configs/model/attention/*.yaml`)

| File | Keys | Defaults |
|---|---|---|
| `gated.yaml` | `attention_type`, `input_dim`, `hidden_dim`, `dropout`, `n_classes`, `init` | `gated`, `${model.encoder.out_dim}`, `8`, `true`, `1`, `kaiming` |
| `gated_projected.yaml` | `attention_type`, `input_dim`, `projection_dim`, `hidden_dim`, `dropout`, `n_classes`, `init` | `gated_projected`, `${model.encoder.out_dim}`, `32`, `8`, `true`, `1`, `kaiming` |

### Losses

`task.loss` selects the loss: `categorical_bce` (`pos_weight`), `categorical_ce`
(`class_weight`, filled by `task.class_weights=auto_train`), `regression_mse`,
`regression_huber` (`delta`), `survival_coxsgd` (`top_n`, `regularizer_weight`).
Extra loss kwargs go under `model.loss.*`.

### Augmentation preset files (`configs/augmentation/*.yaml`)

Selects how BGRL builds its two views. Applies to `task=pretrain_bgrl`.

Covered files:

- `configs/augmentation/bgrl_paper.yaml` (production view generator)
- `configs/augmentation/uniform.yaml` (structure-independent baseline)

**This group is not in any defaults list, so it must be attached explicitly.**
`bgrl_module` reads `task.augmentation`; attach a preset there with:

```bash
grass-mil-train task=pretrain_bgrl model=bgrl_module \
  +augmentation@task.augmentation=bgrl_paper
```

The leading `+` and the `@task.augmentation` package are both required. A bare
`augmentation.mode=uniform` fails with `Could not override 'augmentation.mode'`,
because no `augmentation` key exists in the composed config until the preset is
attached.

If `task.augmentation` is left unset, the module falls back to per-view uniform
drop rates read from `task.drop_edge_p1/p2` and `task.drop_feat_p1/p2` (see the
pretrain task table above). That fallback, not `bgrl_paper`, is what you get by
default.

Once attached, individual keys are overridden under `task.augmentation.*`:

| Key | Default (`bgrl_paper`) | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task.augmentation.mode` | `importance` | str | `importance`, `uniform` | always | `grass-mil-train task.augmentation.mode=uniform` |
| `task.augmentation.mu` | `0.25` | float | `>0` | `mode=importance` | `grass-mil-train task.augmentation.mu=0.3` |
| `task.augmentation.p_lambda` | `0.45` | float | `(0, 1]` | `mode=importance` | `grass-mil-train task.augmentation.p_lambda=0.5` |
| `task.augmentation.feature_noise_std` | `0.2` | float | `>=0` | `mode=importance` | `grass-mil-train task.augmentation.feature_noise_std=0.1` |
| `task.augmentation.feature_noise_columns` | `["SIZE"]` | list[str] | column names | `mode=importance` | `grass-mil-train 'task.augmentation.feature_noise_columns=[AREA]'` |
| `task.augmentation.drop_edge_p` | `0.1` (`uniform`) | float | `[0, 1)` | `mode=uniform` | `grass-mil-train task.augmentation.drop_edge_p=0.2` |
| `task.augmentation.drop_feat_p` | `0.1` (`uniform`) | float | `[0, 1)` | `mode=uniform` | `grass-mil-train task.augmentation.drop_feat_p=0.2` |

Two further constraints are easy to get wrong:

- **Noise columns are resolved by name, never by index.** A hardcoded index that
  means cell size in one cohort means something else in another, which silently
  feeds the encoder pure noise. Names are matched against the dataset's
  molecular feature names.
- **`mode=importance` needs the reserved embedding slot.** Dropped nodes have
  their features zeroed and their cell type set to an "unassigned" code, so it
  requires `model.encoder.categorical_embedding.reserve_unassigned=true`.
  Cohorts with no continuous node features cannot use the Gaussian-noise step at
  all; use `uniform` there.

Node drop probability under `importance` is
`min((1 − importance) · mu, p_lambda)`, where importance is the mean-max
normalised log-degree — so hubs are preserved and low-degree nodes are masked.

## 5) Optimizer And Scheduler Configs

## 5.1 `configs/optim/adamw.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `optim._target_` | `torch.optim.AdamW` | str | optimizer class path | always | `grass-mil-train optim._target_=torch.optim.SGD` |
| `optim.lr` | `0.001` | float | `>0` | optimizer step | `grass-mil-train optim.lr=5e-4` |
| `optim.weight_decay` | `0.01` | float | `>=0` | optimizer step | `grass-mil-train optim.weight_decay=1e-4` |

## 5.2 Scheduler files

| File | Key | Default | Type | Example Override |
|---|---|---|---|---|
| `configs/scheduler/cosine_epoch.yaml` | `_target_` | `torch.optim.lr_scheduler.CosineAnnealingLR` | str | `grass-mil-train scheduler._target_=...` |
| `configs/scheduler/cosine_epoch.yaml` | `T_max` | `100` | int | `grass-mil-train scheduler.T_max=200` |
| `configs/scheduler/cosine_epoch.yaml` | `eta_min` | `1.0e-6` | float | `grass-mil-train scheduler.eta_min=1.0e-7` |
| `configs/scheduler/cosine_step.yaml` | `_target_` | `torch.optim.lr_scheduler.CosineAnnealingLR` | str | `grass-mil-train scheduler._target_=...` |
| `configs/scheduler/cosine_step.yaml` | `T_max` | `10000` | int | `grass-mil-train scheduler.T_max=20000` |
| `configs/scheduler/cosine_step.yaml` | `eta_min` | `1.0e-7` | float | `grass-mil-train scheduler.eta_min=1.0e-8` |
| `configs/scheduler/cosine_mil_step.yaml` | `T_max` | `auto` (`= max_epochs * ceil(n_train_units / hyperbatch)`, per optimizer step) | str/int | `grass-mil-train scheduler.T_max=5000` |
| `configs/scheduler/cosine_mil_step.yaml` | `eta_min` | `1.0e-7` | float | `grass-mil-train scheduler.eta_min=0` |

## 6) Trainer Config Group (`configs/trainer/*.yaml`)

## 6.1 `configs/trainer/default.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `trainer._target_` | `lightning.pytorch.trainer.Trainer` | str | trainer import path | always | `grass-mil-train trainer._target_=...` |
| `trainer.default_root_dir` | `${paths.output_dir}` | str | path | trainer init | `grass-mil-train trainer.default_root_dir=/tmp/out` |
| `trainer.min_epochs` | `1` | int | `>=0` | fit loop | `grass-mil-train trainer.min_epochs=0` |
| `trainer.max_epochs` | `10` | int | `>=1` | fit loop | `grass-mil-train trainer.max_epochs=100` |
| `trainer.accelerator` | `cpu` | str | lightning-supported accelerators | runtime | `grass-mil-train trainer.accelerator=gpu` |
| `trainer.devices` | `1` | int/list/str | lightning-supported device spec | runtime | `grass-mil-train trainer.devices=2` |
| `trainer.check_val_every_n_epoch` | `1` | int | `>=1` | fit loop | `grass-mil-train trainer.check_val_every_n_epoch=2` |
| `trainer.deterministic` | `False` | bool | `true/false` | reproducibility | `grass-mil-train trainer.deterministic=true` |

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
| `defaults` | `[model_checkpoint, early_stopping, model_summary, rich_progress_bar, _self_]` | list | callback group names | always | `grass-mil-train callbacks=none` |
| `callbacks.model_checkpoint.dirpath` | `${paths.output_dir}/checkpoints` | str | path | checkpoint callback enabled | `grass-mil-train callbacks.model_checkpoint.dirpath=/tmp/ckpt` |
| `callbacks.model_checkpoint.filename` | `epoch_{epoch:03d}` | str | format string | checkpoint callback enabled | `grass-mil-train callbacks.model_checkpoint.filename=best` |
| `callbacks.model_checkpoint.monitor` | `val/loss` | str | logged metric name | checkpoint callback enabled | `grass-mil-train callbacks.model_checkpoint.monitor=val/acc` |
| `callbacks.model_checkpoint.mode` | `min` | str | `min`, `max` | checkpoint callback enabled | `grass-mil-train callbacks.model_checkpoint.mode=max` |
| `callbacks.model_checkpoint.save_last` | `True` | bool | `true/false` | checkpoint callback enabled | `grass-mil-train callbacks.model_checkpoint.save_last=false` |
| `callbacks.model_checkpoint.auto_insert_metric_name` | `False` | bool | `true/false` | checkpoint callback enabled | `grass-mil-train callbacks.model_checkpoint.auto_insert_metric_name=true` |
| `callbacks.early_stopping.monitor` | `val/loss` | str | logged metric name | early stopping enabled | `grass-mil-train callbacks.early_stopping.monitor=val/acc` |
| `callbacks.early_stopping.patience` | `100` | int | `>=0` | early stopping enabled | `grass-mil-train callbacks.early_stopping.patience=20` |
| `callbacks.early_stopping.mode` | `min` | str | `min`, `max` | early stopping enabled | `grass-mil-train callbacks.early_stopping.mode=max` |
| `callbacks.model_summary.max_depth` | `-1` | int | integer depth | model summary enabled | `grass-mil-train callbacks.model_summary.max_depth=2` |

## 7.2 `configs/callbacks/model_checkpoint.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model_checkpoint._target_` | `lightning.pytorch.callbacks.ModelCheckpoint` | str | import path | always | `grass-mil-train callbacks.model_checkpoint._target_=...` |
| `model_checkpoint.dirpath` | `null` | str/null | path or null | callback active | `grass-mil-train callbacks.model_checkpoint.dirpath=/tmp/ckpt` |
| `model_checkpoint.filename` | `null` | str/null | filename template or null | callback active | `grass-mil-train callbacks.model_checkpoint.filename=epoch_{epoch}` |
| `model_checkpoint.monitor` | `null` | str/null | metric name or null | callback active | `grass-mil-train callbacks.model_checkpoint.monitor=val/loss` |
| `model_checkpoint.verbose` | `False` | bool | `true/false` | callback active | `grass-mil-train callbacks.model_checkpoint.verbose=true` |
| `model_checkpoint.save_last` | `null` | bool/null | bool or null | callback active | `grass-mil-train callbacks.model_checkpoint.save_last=true` |
| `model_checkpoint.save_top_k` | `1` | int | integer | callback active | `grass-mil-train callbacks.model_checkpoint.save_top_k=3` |
| `model_checkpoint.mode` | `min` | str | `min`, `max` | callback active | `grass-mil-train callbacks.model_checkpoint.mode=max` |
| `model_checkpoint.auto_insert_metric_name` | `True` | bool | `true/false` | callback active | `grass-mil-train callbacks.model_checkpoint.auto_insert_metric_name=false` |
| `model_checkpoint.save_weights_only` | `False` | bool | `true/false` | callback active | `grass-mil-train callbacks.model_checkpoint.save_weights_only=true` |
| `model_checkpoint.every_n_train_steps` | `null` | int/null | integer or null | callback active | `grass-mil-train callbacks.model_checkpoint.every_n_train_steps=100` |
| `model_checkpoint.train_time_interval` | `null` | duration/null | Lightning duration format or null | callback active | `grass-mil-train callbacks.model_checkpoint.train_time_interval='00:10:00'` |
| `model_checkpoint.every_n_epochs` | `null` | int/null | integer or null | callback active | `grass-mil-train callbacks.model_checkpoint.every_n_epochs=1` |
| `model_checkpoint.save_on_train_epoch_end` | `null` | bool/null | bool or null | callback active | `grass-mil-train callbacks.model_checkpoint.save_on_train_epoch_end=true` |

## 7.3 `configs/callbacks/early_stopping.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `early_stopping._target_` | `lightning.pytorch.callbacks.EarlyStopping` | str | import path | always | `grass-mil-train callbacks.early_stopping._target_=...` |
| `early_stopping.monitor` | `???` | str | metric name | early stopping enabled | `grass-mil-train callbacks.early_stopping.monitor=val/loss` |
| `early_stopping.min_delta` | `0.` | float | any float | early stopping enabled | `grass-mil-train callbacks.early_stopping.min_delta=0.001` |
| `early_stopping.patience` | `3` | int | `>=0` | early stopping enabled | `grass-mil-train callbacks.early_stopping.patience=10` |
| `early_stopping.verbose` | `False` | bool | `true/false` | early stopping enabled | `grass-mil-train callbacks.early_stopping.verbose=true` |
| `early_stopping.mode` | `min` | str | `min`, `max` | early stopping enabled | `grass-mil-train callbacks.early_stopping.mode=max` |
| `early_stopping.strict` | `True` | bool | `true/false` | early stopping enabled | `grass-mil-train callbacks.early_stopping.strict=false` |
| `early_stopping.check_finite` | `True` | bool | `true/false` | early stopping enabled | `grass-mil-train callbacks.early_stopping.check_finite=true` |
| `early_stopping.stopping_threshold` | `null` | float/null | float or null | early stopping enabled | `grass-mil-train callbacks.early_stopping.stopping_threshold=0.1` |
| `early_stopping.divergence_threshold` | `null` | float/null | float or null | early stopping enabled | `grass-mil-train callbacks.early_stopping.divergence_threshold=10.0` |
| `early_stopping.check_on_train_epoch_end` | `null` | bool/null | bool or null | early stopping enabled | `grass-mil-train callbacks.early_stopping.check_on_train_epoch_end=true` |

## 7.4 Other callback files

Covered files:

- `configs/callbacks/model_summary.yaml`
- `configs/callbacks/rich_progress_bar.yaml`
- `configs/callbacks/none.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `model_summary._target_` | `lightning.pytorch.callbacks.RichModelSummary` | str | import path | `callbacks=model_summary` | `grass-mil-train callbacks.model_summary._target_=...` |
| `model_summary.max_depth` | `1` | int | integer depth | `callbacks=model_summary` | `grass-mil-train callbacks.model_summary.max_depth=2` |
| `rich_progress_bar._target_` | `lightning.pytorch.callbacks.RichProgressBar` | str | import path | `callbacks=rich_progress_bar` | `grass-mil-train callbacks.rich_progress_bar._target_=...` |
| `configs/callbacks/none.yaml` | empty file | n/a | no fields | `callbacks=none` | `grass-mil-train callbacks=none` |

## 7.5 Selection presets

| File | Content |
|---|---|
| `configs/callbacks/mil_selection.yaml` | checkpoint on `val/auroc` (max, top 1 + last), early stopping on `val/loss_region` (patience 8, after `trainer.min_epochs`) |
| `configs/callbacks/refit.yaml` | fixed-length refit: no early stopping, no metric selection, `last.ckpt` only |
| `configs/callbacks/pretrain_checkpoint.yaml` | checkpoint every `every_n_train_steps=5000`, all kept |

## 8) Logger Config Group (`configs/logger/*.yaml`)

## 8.1 `configs/logger/csv.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `csv._target_` | `lightning.pytorch.loggers.csv_logs.CSVLogger` | str | import path | `logger=csv` | `grass-mil-train logger.csv._target_=...` |
| `csv.save_dir` | `${paths.output_dir}` | str | path | `logger=csv` | `grass-mil-train logger.csv.save_dir=/tmp/logs` |
| `csv.name` | `csv/` | str | name | `logger=csv` | `grass-mil-train logger.csv.name=metrics` |
| `csv.prefix` | `""` | str | prefix string | `logger=csv` | `grass-mil-train logger.csv.prefix=exp1` |

## 8.2 `configs/logger/tensorboard.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `tensorboard._target_` | `lightning.pytorch.loggers.tensorboard.TensorBoardLogger` | str | import path | `logger=tensorboard` | `grass-mil-train logger.tensorboard._target_=...` |
| `tensorboard.save_dir` | `${paths.output_dir}/tensorboard/` | str | path | `logger=tensorboard` | `grass-mil-train logger.tensorboard.save_dir=/tmp/tb` |
| `tensorboard.name` | `null` | str/null | name or null | `logger=tensorboard` | `grass-mil-train logger.tensorboard.name=runA` |
| `tensorboard.log_graph` | `False` | bool | `true/false` | `logger=tensorboard` | `grass-mil-train logger.tensorboard.log_graph=true` |
| `tensorboard.default_hp_metric` | `True` | bool | `true/false` | `logger=tensorboard` | `grass-mil-train logger.tensorboard.default_hp_metric=false` |
| `tensorboard.prefix` | `""` | str | prefix string | `logger=tensorboard` | `grass-mil-train logger.tensorboard.prefix=train` |

## 8.3 `configs/logger/wandb.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `wandb._target_` | `lightning.pytorch.loggers.wandb.WandbLogger` | str | import path | `logger=wandb` | `grass-mil-train logger.wandb._target_=...` |
| `wandb.save_dir` | `${paths.output_dir}` | str | path | `logger=wandb` | `grass-mil-train logger.wandb.save_dir=/tmp/wb` |
| `wandb.offline` | `False` | bool | `true/false` | `logger=wandb` | `grass-mil-train logger.wandb.offline=true` |
| `wandb.id` | `null` | str/null | run ID or null | `logger=wandb` | `grass-mil-train logger.wandb.id=my_id` |
| `wandb.anonymous` | `null` | str/null | WandB anonymous mode or null | `logger=wandb` | `grass-mil-train logger.wandb.anonymous=allow` |
| `wandb.project` | `grass-mil` | str | project name | `logger=wandb` | `grass-mil-train logger.wandb.project=grass-mil` |
| `wandb.log_model` | `False` | bool | `true/false` | `logger=wandb` | `grass-mil-train logger.wandb.log_model=true` |
| `wandb.prefix` | `""` | str | prefix string | `logger=wandb` | `grass-mil-train logger.wandb.prefix=exp` |
| `wandb.group` | `""` | str | group name | `logger=wandb` | `grass-mil-train logger.wandb.group=ablation` |
| `wandb.tags` | `[]` | list[str] | any list | `logger=wandb` | `grass-mil-train logger.wandb.tags='[ssl,mil]'` |
| `wandb.job_type` | `""` | str | job type | `logger=wandb` | `grass-mil-train logger.wandb.job_type=train` |

## 8.4 `configs/logger/mlflow.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `mlflow._target_` | `lightning.pytorch.loggers.mlflow.MLFlowLogger` | str | import path | `logger=mlflow` | `grass-mil-train logger.mlflow._target_=...` |
| `mlflow.tracking_uri` | `${paths.log_dir}/mlflow/mlruns` | str | URI/path | `logger=mlflow` | `grass-mil-train logger.mlflow.tracking_uri=/tmp/mlruns` |
| `mlflow.tags` | `null` | dict/null | tags dict or null | `logger=mlflow` | `grass-mil-train logger.mlflow.tags='{stage:dev}'` |
| `mlflow.prefix` | `""` | str | prefix | `logger=mlflow` | `grass-mil-train logger.mlflow.prefix=exp` |
| `mlflow.artifact_location` | `null` | str/null | URI/path or null | `logger=mlflow` | `grass-mil-train logger.mlflow.artifact_location=/tmp/artifacts` |

## 8.5 `configs/logger/comet.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `comet._target_` | `lightning.pytorch.loggers.comet.CometLogger` | str | import path | `logger=comet` | `grass-mil-train logger.comet._target_=...` |
| `comet.api_key` | `${oc.env:COMET_API_TOKEN}` | str | API key/env ref | `logger=comet` | `grass-mil-train logger.comet.api_key=...` |
| `comet.save_dir` | `${paths.output_dir}` | str | path | `logger=comet` | `grass-mil-train logger.comet.save_dir=/tmp/comet` |
| `comet.project_name` | `grass-mil` | str | project name | `logger=comet` | `grass-mil-train logger.comet.project_name=grass-mil` |
| `comet.rest_api_key` | `null` | str/null | key or null | `logger=comet` | `grass-mil-train logger.comet.rest_api_key=...` |
| `comet.experiment_key` | `null` | str/null | key or null | `logger=comet` | `grass-mil-train logger.comet.experiment_key=...` |
| `comet.offline` | `False` | bool | `true/false` | `logger=comet` | `grass-mil-train logger.comet.offline=true` |
| `comet.prefix` | `""` | str | prefix | `logger=comet` | `grass-mil-train logger.comet.prefix=train` |

## 8.6 `configs/logger/neptune.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `neptune._target_` | `lightning.pytorch.loggers.neptune.NeptuneLogger` | str | import path | `logger=neptune` | `grass-mil-train logger.neptune._target_=...` |
| `neptune.api_key` | `${oc.env:NEPTUNE_API_TOKEN}` | str | API key/env ref | `logger=neptune` | `grass-mil-train logger.neptune.api_key=...` |
| `neptune.project` | `org/grass-mil` | str | project slug | `logger=neptune` | `grass-mil-train logger.neptune.project=org/project` |
| `neptune.log_model_checkpoints` | `True` | bool | `true/false` | `logger=neptune` | `grass-mil-train logger.neptune.log_model_checkpoints=false` |
| `neptune.prefix` | `""` | str | prefix | `logger=neptune` | `grass-mil-train logger.neptune.prefix=exp` |

## 8.7 `configs/logger/aim.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `aim._target_` | `aim.pytorch_lightning.AimLogger` | str | import path | `logger=aim` | `grass-mil-train logger.aim._target_=...` |
| `aim.repo` | `${paths.root_dir}` | str | path/remote URI | `logger=aim` | `grass-mil-train logger.aim.repo=/tmp/.aim` |
| `aim.experiment` | `null` | str/null | experiment name or null | `logger=aim` | `grass-mil-train logger.aim.experiment=baseline` |
| `aim.train_metric_prefix` | `train/` | str | prefix | `logger=aim` | `grass-mil-train logger.aim.train_metric_prefix=train` |
| `aim.val_metric_prefix` | `val/` | str | prefix | `logger=aim` | `grass-mil-train logger.aim.val_metric_prefix=val` |
| `aim.test_metric_prefix` | `test/` | str | prefix | `logger=aim` | `grass-mil-train logger.aim.test_metric_prefix=test` |
| `aim.system_tracking_interval` | `10` | int/null | seconds or null | `logger=aim` | `grass-mil-train logger.aim.system_tracking_interval=5` |
| `aim.log_system_params` | `true` | bool | `true/false` | `logger=aim` | `grass-mil-train logger.aim.log_system_params=false` |
| `aim.capture_terminal_logs` | `false` | bool | `true/false` | `logger=aim` | `grass-mil-train logger.aim.capture_terminal_logs=true` |

## 8.8 `configs/logger/many_loggers.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | `[csv, tensorboard, wandb]` | list | logger group names | `logger=many_loggers` | `grass-mil-train logger=many_loggers logger.defaults='[csv,tensorboard]'` |

## 9) Debug Config Group (`configs/debug/*.yaml`)

## 9.1 `configs/debug/default.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `task_name` | `debug` | str | any string | `debug=default` | `grass-mil-train debug=default task_name=dbg` |
| `callbacks` | `null` | null/config | null or callback config | `debug=default` | `grass-mil-train debug=default callbacks=default` |
| `logger` | `null` | null/config | null or logger config | `debug=default` | `grass-mil-train debug=default logger=csv` |
| `extras.ignore_warnings` | `False` | bool | `true/false` | `debug=default` | `grass-mil-train debug=default extras.ignore_warnings=true` |
| `extras.enforce_tags` | `False` | bool | `true/false` | `debug=default` | `grass-mil-train debug=default extras.enforce_tags=false` |
| `hydra.job_logging.root.level` | `DEBUG` | str | logging level | `debug=default` | `grass-mil-train debug=default hydra.job_logging.root.level=INFO` |
| `trainer.max_epochs` | `1` | int | `>=1` | `debug=default` | `grass-mil-train debug=default trainer.max_epochs=2` |
| `trainer.accelerator` | `cpu` | str | accelerator value | `debug=default` | `grass-mil-train debug=default trainer.accelerator=cpu` |
| `trainer.devices` | `1` | int | `>=1` | `debug=default` | `grass-mil-train debug=default trainer.devices=1` |
| `trainer.detect_anomaly` | `true` | bool | `true/false` | `debug=default` | `grass-mil-train debug=default trainer.detect_anomaly=true` |
| `data.num_workers` | `0` | int | `>=0` | `debug=default` | `grass-mil-train debug=default data.num_workers=0` |
| `data.pin_memory` | `False` | bool | `true/false` | `debug=default` | `grass-mil-train debug=default data.pin_memory=false` |

## 9.2 Other debug presets

Covered files:

- `configs/debug/fdr.yaml`
- `configs/debug/limit.yaml`
- `configs/debug/overfit.yaml`
- `configs/debug/profiler.yaml`

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `debug/fdr.defaults` | `[default]` | list | debug group list | `debug=fdr` | `grass-mil-train debug=fdr` |
| `debug/fdr.trainer.fast_dev_run` | `true` | bool | `true/false` | `debug=fdr` | `grass-mil-train debug=fdr trainer.fast_dev_run=true` |
| `debug/limit.defaults` | `[default]` | list | debug group list | `debug=limit` | `grass-mil-train debug=limit` |
| `debug/limit.trainer.max_epochs` | `3` | int | `>=1` | `debug=limit` | `grass-mil-train debug=limit trainer.max_epochs=5` |
| `debug/limit.trainer.limit_train_batches` | `0.01` | float | `(0,1]` or int | `debug=limit` | `grass-mil-train debug=limit trainer.limit_train_batches=0.1` |
| `debug/limit.trainer.limit_val_batches` | `0.05` | float | `(0,1]` or int | `debug=limit` | `grass-mil-train debug=limit trainer.limit_val_batches=0.2` |
| `debug/limit.trainer.limit_test_batches` | `0.05` | float | `(0,1]` or int | `debug=limit` | `grass-mil-train debug=limit trainer.limit_test_batches=0.2` |
| `debug/overfit.defaults` | `[default]` | list | debug group list | `debug=overfit` | `grass-mil-train debug=overfit` |
| `debug/overfit.trainer.max_epochs` | `20` | int | `>=1` | `debug=overfit` | `grass-mil-train debug=overfit trainer.max_epochs=10` |
| `debug/overfit.trainer.overfit_batches` | `3` | int | `>=1` | `debug=overfit` | `grass-mil-train debug=overfit trainer.overfit_batches=1` |
| `debug/overfit.callbacks` | `null` | null/config | null or callback config | `debug=overfit` | `grass-mil-train debug=overfit callbacks=none` |
| `debug/profiler.defaults` | `[default]` | list | debug group list | `debug=profiler` | `grass-mil-train debug=profiler` |
| `debug/profiler.trainer.max_epochs` | `1` | int | `>=1` | `debug=profiler` | `grass-mil-train debug=profiler trainer.max_epochs=2` |
| `debug/profiler.trainer.profiler` | `simple` | str | `simple`, `advanced`, `pytorch`, etc. | `debug=profiler` | `grass-mil-train debug=profiler trainer.profiler=advanced` |

## 10) Hydra Runtime Config (`configs/hydra/default.yaml`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `defaults` | overrides for `hydra_logging=colorlog`, `job_logging=colorlog` | list | hydra logging groups | always | `grass-mil-train hydra/job_logging=stdout` |
| `hydra.run.dir` | `${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}` | str | directory template | single run | `grass-mil-train hydra.run.dir=/tmp/single_run` |
| `hydra.sweep.dir` | `${paths.log_dir}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}` | str | directory template | multirun | `grass-mil-train -m hydra.sweep.dir=/tmp/multi` |
| `hydra.sweep.subdir` | `${hydra.job.num}` | str | template string | multirun | `grass-mil-train -m hydra.sweep.subdir=job_${hydra.job.num}` |
| `hydra.job_logging.handlers.file.filename` | `${hydra.runtime.output_dir}/${task_name}.log` | str | file path template | Hydra file logging | `grass-mil-train hydra.job_logging.handlers.file.filename=/tmp/train.log` |

## 11) Paths Config (`configs/paths/default.yaml`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `paths.root_dir` | `${oc.env:PROJECT_ROOT}` | str | existing path | always | `grass-mil-train paths.root_dir=/abs/project` |
| `paths.data_dir` | `${paths.root_dir}/data/` | str | path | always | `grass-mil-train paths.data_dir=/abs/data` |
| `paths.log_dir` | `${paths.root_dir}/logs/` | str | path | always | `grass-mil-train paths.log_dir=/abs/logs` |
| `paths.output_dir` | `${hydra:runtime.output_dir}` | str | hydra runtime path | always | `grass-mil-train paths.output_dir=/tmp/out` |
| `paths.work_dir` | `${hydra:runtime.cwd}` | str | working directory path | always | `grass-mil-train paths.work_dir=/tmp` |

## 12) Extras Config (`configs/extras/default.yaml`)

| Key | Default | Type | Valid Values | Active When | Example Override |
|---|---|---|---|---|---|
| `extras.ignore_warnings` | `False` | bool | `true/false` | `extras` enabled | `grass-mil-train extras.ignore_warnings=true` |
| `extras.enforce_tags` | `True` | bool | `true/false` | `extras` enabled | `grass-mil-train extras.enforce_tags=false` |
| `extras.print_config` | `True` | bool | `true/false` | `extras` enabled | `grass-mil-train extras.print_config=false` |

## 12a) Regime Presets (`configs/regime/*.yaml`)

`regime=<name>` composes `model=grass_mil`, `task=finetune_mil`,
`scheduler=cosine_mil_step`, `callbacks=mil_selection`, `data/sampler=ego_graph_mil`
and sets the keys below; each is an ordinary override afterwards.

| Regime | `optim.lr` / `weight_decay` | `encoder_lr_scale` | head | `head_lr_scale` / `head_weight_decay` | `attention_lr_scale` | roots per region | `encoder.dropout` | drop edge / feat | epochs (min) |
|---|---|---|---|---|---|---|---|---|---|
| `mil` | 1e-4 / 1e-3 | 1.0 | MLP (2 layers) | 1.0 / null | 2.0 | 256 | 0.0 | 0 / 0 | 25 (10) |
| `mil_frozen` | 1e-4 / 1e-3 | 0.0 | MLP | 1.0 / null | 2.0 | 512 | 0.0 | 0 / 0 | 25 (10) |
| `linear_probehead` | 1e-4 / 1e-3 | 0.0 | linear, probe pre-fit, frozen | 0.0 / null | 2.0 | 512 | 0.0 | 0 / 0 | 25 (10) |
| `linear_ftall` | 1e-4 / 1e-3 | 1.0 | linear, probe pre-fit | 0.01 / 1e-3 | 2.0 | 256 | 0.0 | 0 / 0 | 25 (10) |
| `linear_ftall_reg` | 1e-4 / 1e-2 | 0.3 | linear, probe pre-fit | 0.01 / 1e-3 | 2.0 | 256 | 0.2 | 0.05 / 0.05 | 25 (10) |

Shared: `task.region_accumulation.hyperbatch_size=12`, `task.grad_clip_norm=1.0`,
`trainer.precision=16-mixed`, `data.val_sampler` = fixed 1024-root subset per
region (`fixed_per_unit_seed=0`), `data.test_sampler` = all roots.

## 12b) Analysis CLIs

| CLI | Config | Main keys |
|---|---|---|
| `grass-mil-probe` | `configs/probe.yaml` | `probe.roots_per_unit=96`, `probe.seed=0`, `probe.batch_size=256`, `probe.C=1.0`, `probe.class_weight=balanced`, `probe.max_iter=3000`, `probe.eval_splits=[val]`, `probe.output_path`; splits from `data.split.explicit` |
| `grass-mil-nested-cv` | `configs/nested_cv.yaml` | `nested_cv.folds_file`, `nested_cv.id_kind=region|sample|patch`, `nested_cv.{label_file,id_column,label_column,patient_column}` (default from `data.graph_labels`), `nested_cv.inner_k=3`, `inner_seed=0`, `selection_metric=val/auroc`, `epoch_reduce=median`, `min_refit_epochs=1`, `outer_folds`, `resume`, `nested_cv.probe.*` (as the probe keys, `enabled=auto`, `cache_dir`), `patient_aggregation=mean` |
| `grass-mil-extract` | `configs/extract.yaml` | `extract.nested_cv_summary` or `+extract.models.<name>=<ckpt>` (+ `+extract.held_out.<name>=[...]`), `extract.unit_id=auto|region|sample|patch`, `extract.regions`, `extract.batch_size=512`, `extract.autocast`, `extract.device`, `extract.resume`, `extract.output_dir` |
| `grass-mil-scores` | `configs/scores.yaml` | `input`, `label_column`, `scores`, `fold_column`, `patient_column`, `patient_level`, `standardise=z|rank|none`, `strata`, `n_boot=2000`, `seed=0`, `operating_rules`, `prevalences` |
| `grass-mil-niches` | `configs/niches.yaml` | `cells_dir`, `regions=held_out|all|list|file`, `labels.*`, `space=gnn|attention`, `attention_ckpt`, `reduction.{method,n_components}`, `clustering.{method,n_clusters,params.*}`, `sample_size`, `l2_normalise`, `assign_k`, `seed`, `present_threshold`, `n_boot`, `figures`, `umap.*`, `positive_class`, `positive_name` |
| `grass-mil-metaclusters` | `configs/metaclusters.yaml` | `cells_dir`, `models`, `stage=all|cluster|match|transfer|basis`, `labels.*`, the clustering keys of `niches.yaml`, `match.{distance,threshold,scan.{min,max,n},min_folds}`, `transfer.{min_cells,present_threshold}`, `basis.{enabled,space_model,k,sample_size,n_init,seed}` |
| `grass-mil-spatial-stats` | `configs/spatial_stats.yaml` | `cells_dir`, `regions`, `labels.*`, `analyses`, `group_map`, `reference`, `partners`, `infiltrating`, `nhood.{n_perms,backend}`, `cn.{window_k,n_cn,sample_size}`, `crossk.radii`, `seed`, `min_per_group` |
| `grass-mil-predict` | `configs/inference/predict.yaml` | `aggregation.mode=mean|max|none|attention_weighted|mean_probability`, `aggregation.bag_scope=patch|region|sample|patient`, `aggregation.patient_map.{label_file,id_column,patient_column}` |

## 13) Repo-Enforced Behavioral Constraints (From Code)

These constraints are not just documentation conventions; they are enforced in runtime code.

| Constraint | Enforced In | Behavior |
|---|---|---|
| `task.target_type` must be one of `binary`, `regression`, `survival` | `src/grass_mil/models/training/builders.py` | raises `ValueError` for unsupported target types |
| `task.instance_sampling` must be `all` or `random` | `src/grass_mil/models/training/builders.py` | raises `ValueError` for invalid sampling mode |
| `task.region_accumulation.hyperbatch_size >= 1` when enabled | `src/grass_mil/models/supervised_module.py` | raises `ValueError` |
| `conv_type=gine` requires `model.encoder.edge_attr_dim` and runtime `edge_attr` | `src/grass_mil/models/components/backbones.py` | raises `ValueError` |
| `conv_type=gat` requires `hidden_dim % gat_heads == 0` | `src/grass_mil/models/components/backbones.py` | raises `ValueError` |
| `model.encoder.use_edge_attr=true` is meaningful only for `gcn`/`gine` | `src/grass_mil/models/components/backbones.py` | warning/ignored for unsupported convs |
| `data.split.train_val_test_split` must sum to `1.0` | `src/grass_mil/data/components/precompute.py` split helpers | raises `ValueError` |
| `data.feature_reducer.fit_mode` must be `global` or `train_only` | `src/grass_mil/data/components/precompute.py` | raises `ValueError` |
| `data.sampler.runtime.weight_mode` must be `inverse`, `sqrt_inverse`, or `proportional` | `src/grass_mil/data/components/samplers.py` | raises `ValueError` |
| `model.encoder_init_map` must be `identity` or `auto_bgrl_or_identity` | `src/grass_mil/models/training/checkpoint_init.py` | raises `ValueError` |
| `model.init_from_ckpt` path must exist | checkpoint init helper | raises `FileNotFoundError` |

## 14) Trainer/Callback/Logger Exhaustiveness Note

This file is exhaustive for keys defined in repository config presets.

Hydra still allows passing additional Lightning kwargs that are not pre-declared in these YAML files. Those extra keys are valid if accepted by the target Lightning class.

Example:

```bash
grass-mil-train trainer.log_every_n_steps=1 trainer.enable_progress_bar=true
```

Use this flexibility with caution and keep custom overrides version-controlled in experiment config files when possible.
