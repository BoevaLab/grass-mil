# Getting Started: Typical Usage On Your Own Cohort

This page is the end-to-end path from cell tables to niches. Each step is one
command; the data description is written once in a cohort config and reused.
For the meaning of every setting see the
[configuration guide](configuration_guide.md); for a runnable demo on public
data see the [vignettes](../vignettes/README.md).

## 0. What you need

| Input | Form | Example |
|---|---|---|
| one cell table per region | CSV (or h5ad / SingleCellExperiment): one row per cell with coordinates, a cell-type column and optional continuous features such as cell size | `raw/P001_core1.csv` with `cell_id,x,y,cell_type,SIZE` |
| manifest | CSV listing the regions: `sample_id,input_path,input_type,region_id` | `raw/manifest.csv` |
| labels | CSV with a region id, the target and a patient column | `raw/labels.csv` with `REGION_ID,STAS,PATIENT` |
| folds | JSON list of lists of region ids, patient-grouped | `raw/folds.json`: `[["P001_core1","P001_core2",...], [...], ...]` |

Coordinates can stay in pixels: set `coord_scale_um` to the factor that brings
them to the unit you want to express distances in, and give `neighbor_cutoff_um`
in that unit. Regions are whole cores, ROIs or slides (`sample_unit: full`); a
cohort of a few large images is cut into tiles instead (see the end of this page).

Write the folds once and keep them with the data:

```python
from grass_mil.data.components.splits import write_fold_json
# folds = [[...region ids...], ...] built from your patient ids, e.g. with
# sklearn.model_selection.StratifiedGroupKFold over regions grouped by patient
write_fold_json(folds, "raw/folds.json")
```

Install (`pip install -e ".[interpretability,io,stats,plots]"`), then check the
entry points: `grass-mil-train --help`.

## 1. Describe the cohort once

Copy the template and fill in paths and column names:

```bash
cp src/grass_mil/configs/experiment/example_cohort.yaml src/grass_mil/configs/experiment/my_cohort.yaml
```

(With an installed wheel, put the file in any directory `my_configs/experiment/`
and add `--config-dir my_configs` to every command.) From here on every command
starts with `experiment=my_cohort`. Confirm the composed configuration before
running anything:

```bash
grass-mil-train experiment=my_cohort --cfg job | less
```

## 2. Build the graphs

The first run precomputes the Delaunay graphs of every region into
`processed_dir` (`processed_index.json` + one `.pt` per region) and reuses them
afterwards. A dry run does only that:

```bash
grass-mil-train experiment=my_cohort +trainer.fast_dev_run=true
```

Check `processed/metadata.json`: number of regions, cell-type vocabulary,
continuous feature names. Re-run with `data.force_precompute=true` whenever an
input, a graph setting or `coord_scale_um` changes.

## 3. Pretrain the encoder (label-free)

```bash
grass-mil-train experiment=my_cohort regime=pretrain_a3 paths.output_dir=runs/pretrain_a3 trainer=gpu
```

Every region is used, no labels. Defaults: 200k steps, 64 ego-graphs per step,
4-hop ego-graphs, DropImportance + noise on `SIZE`. Shorten with
`task.total_steps=50000`; without a cell-size feature set
`task.augmentation.feature_noise_columns=[]`. Output:
`runs/pretrain_a3/checkpoints/last.ckpt` (plus one checkpoint every 5000 steps).

## 4. Nested cross-validation

```bash
grass-mil-nested-cv experiment=my_cohort regime=linear_ftall_reg \
  model.init_from_ckpt=runs/pretrain_a3/checkpoints/last.ckpt \
  paths.output_dir=runs/nested_ftall_reg trainer=gpu
```

Per outer fold this fits the probe head on the inner training regions, trains
three inner runs with region-level selection, refits on the whole development
set for the median best epoch and scores the held-out fold on every cell.
Expect about one minute per epoch for 300 regions of 3,000 cells on one GPU.
Resumable: re-run the same command after an interruption.

Outputs in `runs/nested_ftall_reg/`:

| file | content |
|---|---|
| `nested_cv_summary.json` | `auc_mean`, `auc_sd`, `auc_folds`, refit epochs, run directories |
| `oof_predictions.csv` | one held-out score per region (`outer, region_id, patient_id, label, prob_class_1`) |
| `oof_patient_predictions.csv` | mean of the region scores per patient |
| `outer_<i>/refit/checkpoints/last.ckpt` | the fold models |

Run the other regimes the same way (`regime=mil`, `linear_probehead`,
`linear_ftall`) into their own output directories; the probes are cached and
shared through `nested_cv.probe.cache_dir`.

## 5. Evaluation statistics

```bash
grass-mil-scores input=runs/nested_ftall_reg/oof_predictions.csv output_dir=runs/scores_ftall_reg
```

Patient-level (mean of regions) bootstrap AUROC interval, per-fold mean,
operating points at "flag the top q" rules; add a second table's scores with
`scores=[...]` to get DeLong comparisons. See
[evaluation_stats.md](evaluation_stats.md).

## 6. Extract every cell through the fold models

```bash
grass-mil-extract experiment=my_cohort regime=linear_ftall_reg \
  extract.nested_cv_summary=runs/nested_ftall_reg/nested_cv_summary.json \
  extract.output_dir=runs/cells trainer=gpu
```

Writes `runs/cells/fold<i>/<region>.npz` with the embedding, instance logits
and attention of every cell, plus the cell types, positions and graph. The
held-out regions of every fold model are recorded.

## 7. Niches of one model, and replication across models

```bash
grass-mil-niches cells_dir=runs/cells/fold2 output_dir=runs/niches_fold2 positive_name=STAS
grass-mil-metaclusters cells_dir=runs/cells output_dir=runs/meta positive_name=STAS \
  labels.label_file=/path/to/my_cohort/raw/labels.csv labels.label_column=STAS labels.patient_column=PATIENT
```

The first clusters the held-out cells of one fold model (PCA-10, L2, HDBSCAN,
15-NN assignment) and writes `clusters.csv`, `cells.csv`, `report.md` and the
figures. The second does it for every fold model, matches the clusters across
folds into metaclusters, tests their patient-level shares, transfers every
model to every region and fits a shared label-free basis. Both accept any
clustering backend (`clustering.method=kmeans clustering.n_clusters=12`). See
[niche_discovery.md](niche_discovery.md).

## 8. Model-free spatial statistics

```bash
grass-mil-spatial-stats cells_dir=runs/cells/fold0 output_dir=runs/spatial positive_name=STAS \
  labels.label_file=/path/to/my_cohort/raw/labels.csv labels.label_column=STAS labels.patient_column=PATIENT \
  reference=[tumor] partners=[myeloid,CD8,Treg] infiltrating=[myeloid,CD8] \
  group_map='{normal: tumor, ki67_cd8: CD8, cd8: CD8}'
```

Neighbourhood enrichment, cellular neighbourhoods, contact fractions and
cross-K per region, compared between classes at the patient level. See
[spatial_statistics.md](spatial_statistics.md).

## Variants

**One split instead of nested CV.** Give explicit id lists:

```bash
grass-mil-train experiment=my_cohort regime=mil model.init_from_ckpt=runs/pretrain_a3/checkpoints/last.ckpt \
  data.split.explicit.train=raw/train.json data.split.explicit.val=raw/val.json data.split.explicit.test=raw/test.json
```

For a `linear_*` regime, first `grass-mil-probe` with the same splits and pass
`task.head_init_path=<probe_head.pt>`.

**Patient-level prediction from one model** (`mean_probability` = mean of the
patient's regions, `attention_weighted` = one merged bag):

```bash
grass-mil-predict experiment=my_cohort regime=linear_ftall_reg ckpt_path=runs/nested_ftall_reg/outer_2/refit/checkpoints/last.ckpt \
  aggregation.bag_scope=patient aggregation.mode=mean_probability \
  aggregation.patient_map.label_file=/path/to/my_cohort/raw/labels.csv
```

**Tiles as units** (a few large images): `data.sample_unit=tile`,
`data.tiling.tile_size_um=...`, labels keyed by `patch_id` with
`data.graph_labels.scope=patch`, folds of patch ids, `nested_cv.id_kind=patch`.
Vignette 3 does exactly this.

**h5ad inputs**: replace the `csv:` block by `h5ad: {coord_source: obsm,
coord_key: spatial, categorical_label_columns: [cell_type]}` and use
`input_type: h5ad` in the manifest; the marker matrix becomes the continuous
features (`molecular_features` selects a subset).

**Cell type only** (no continuous feature): `data.csv.molecular_columns=[]`
and `task.augmentation.feature_noise_columns=[]` for pretraining.
