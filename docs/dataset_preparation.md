# Dataset Preparation Guide

This guide describes how to prepare a brand-new dataset so it can run through the `grass-mil` training pipeline.

## 1) Prepare Raw Input Files

`grass-mil` supports per-sample inputs in:

- `csv`
- `tsv`
- `h5ad`
- `sce` / `rds` (SingleCellExperiment via `rpy2` + R)

Each manifest row points to one input file.

## 2) Create The Manifest CSV

Default location: `data/raw/manifest.csv` (override with `data.raw_manifest_path=...`).

Required columns:

- `sample_id`
- `input_path`
- `input_type` (`csv`, `tsv`, `h5ad`, `sce`, `rds`)

Optional columns:

- `region_id`
- `polygons_path`

Example:

```csv
sample_id,input_path,input_type,region_id,polygons_path
sample_001,data/raw/sample_001.csv,csv,region_001,
sample_002,data/raw/sample_002.h5ad,h5ad,region_002,data/raw/sample_002_polygons.geojson
```

Important manifest rules:

- `input_path` and `polygons_path` can be absolute or relative.
- Relative paths are resolved relative to the manifest file directory.
- If `region_id` is used, values must be globally unique.
- If a `sample_id` appears multiple times, each row for that sample must have a non-empty, unique `region_id`.

## 3) Match Input Schema To Config

Default parsing config lives in [`configs/data/spatial_omics.yaml`](/Users/lovrorabuzin/Projects/grass-mil_unification/grass-mil/configs/data/spatial_omics.yaml).

### CSV / TSV

Default expectation:

- coordinate columns: `x`, `y` (`data.csv.coord_columns`)
- optional cell id column: `data.csv.cell_id_column`
- optional categorical columns: `data.csv.categorical_label_columns`

If `data.use_molecular_features=true` (default), molecular columns are required:

- if `data.csv.molecular_columns=null`, all non-coordinate/non-id/non-categorical columns are treated as molecular features
- if that inferred set is empty, preprocessing fails

### H5AD

Configure under `data.h5ad.*`:

- `coord_source=obsm` with `coord_key` (default `spatial`), or
- `coord_source=obs` with `coord_columns`
- optional `cell_id_column`, `categorical_label_columns`, `molecular_layer`, `molecular_features`

### SCE / RDS

Configure under `data.sce.*`:

- coordinate source from `colData` (`coord_columns`) or `reducedDims` (`coord_key`)
- optional assay/feature selection controls

## 4) Set Coordinate Units

Spatial distances are normalized to micrometers during preprocessing.

- `data.coord_scale_um=1.0` means input coordinates are already in micrometers
- set another scale if your inputs are in a different unit

If you change `data.coord_scale_um`, regenerate processed artifacts (`data.force_precompute=true` or a new `data.processed_dir`).

## 5) Optional Polygon And Graph Labels

### Polygon files (`polygons_path`)

Supported formats:

- `.pkl` / `.pickle`
- `.wkt` / `.txt` (line-delimited WKT)
- `.json` / `.geojson`

Behavior:

- coordinates are filtered by polygon union
- when `data.sample_unit != full`, patches can be built from polygons

### Graph labels (`data.graph_labels.*`)

Use a CSV file to attach supervision targets at `sample`, `region`, or `patch` scope.

## 6) Run First Precompute/Training Pass

Use a fresh processed directory when onboarding new data:

```bash
python src/train.py \
  task=finetune_mean \
  model=supervised_module \
  data=spatial_omics \
  data.raw_manifest_path=/absolute/path/to/manifest.csv \
  data.processed_dir=/absolute/path/to/processed/spatial_omics \
  data.force_precompute=true
```

Generated artifacts:

- `${data.processed_dir}/processed_index.json`
- `${data.processed_dir}/metadata.json`
- `${data.processed_dir}/graphs/.../*.pt`

## 7) When You Must Re-Precompute

Rebuild processed artifacts if you change:

- `data.split.*` (splits are persisted in `processed_index.json`)
- `data.coord_scale_um`
- feature reducer settings that affect embedding generation
- tiling / polygon behavior (`data.sample_unit`, `data.tiling.*`, `data.min_cells`)

Use either:

- `data.force_precompute=true`, or
- a new `data.processed_dir`
