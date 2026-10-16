# Data preparation

GRASS-MIL needs the cell tables of a cohort and three small files describing
it:

| File | Content |
|---|---|
| manifest (CSV) | one row per region: where its cell table is |
| labels (CSV) | one row per labelled region (or tile, or sample): targets and patient |
| folds (JSON) | outer cross-validation folds, for `grass-mil-nested-cv` |

The words used throughout:

* **sample**: one cell table (a file), for example one slide or one image;
* **region**: a part of a sample with its own row in the manifest, for example
  one core of a tissue microarray or one region of interest; by default a
  region is its whole sample;
* **unit**: what one prediction is made for, one bag of the model, one label
  and one member of a fold: a region, or a tile or polygon of it (`data.unit`);
* **patient**: units of one patient are kept together in every split and fold.

The paths and column names go in a cohort file, `configs/experiment/my_cohort.yaml`
in your project folder (start from
[the template](../src/grass_mil/configs/experiment/example_cohort.yaml); the
README shows how to copy it), and every command that reads the cohort gets
`--config-dir configs experiment=my_cohort`.

## Cell tables

One file per region with, for every cell, its coordinates and (normally) its
cell type, plus any continuous features:

| Format | Coordinates | Cell type and features |
|---|---|---|
| CSV / TSV | `data.input.coord_columns` (default `x`, `y`) | columns |
| h5ad | `obsm[data.input.obsm_key]` (default `spatial`), or `obs` columns with `obsm_key: null` | `obs` column; features from `X` or `data.input.h5ad_layer` |
| SingleCellExperiment (`.rds`, needs `pip install 'grass-mil[sce]'` and R) | `colData` columns, or `reducedDim(data.input.reduced_dim)` | `colData` column; features from the assay |

`data.input.cell_type_column` names the cell-type column (`null` if there is
none; the model then uses only the features). `data.input.features` selects the
continuous node features: `all` (every remaining numeric CSV column, every
h5ad/SCE feature), a list of names (for example `[area]`), or `null` for cell
type only. With `all`, set `data.input.cell_id_column` so that a numeric id
column is not taken for a feature. Missing or infinite values are refused.

Coordinates are multiplied by `data.coord_scale_um`; every distance setting
(contact distance, tile size, radii) is in the resulting unit.

## Manifest

```csv
sample_id,input_path,region_id
S01,cells/S01_roi1.csv,S01_roi1
S01,cells/S01_roi2.csv,S01_roi2
S02,cells/S02.h5ad,
```

* `sample_id` and `input_path` are required; relative paths are resolved
  against the manifest's directory.
* `region_id` defaults to `sample_id` and must be unique.
* `input_type` (`csv`, `tsv`, `h5ad`, `sce`) is optional when the file suffix
  says it.
* `polygons_path` (optional) points to GeoJSON or WKT polygons of the region:
  cells outside them are dropped, or with `data.unit=polygon` every polygon
  becomes its own unit.

With one CSV per region in `data/cells/`, this writes the manifest:

```python
import pathlib

import pandas as pd

tables = sorted(pathlib.Path("data/cells").glob("*.csv"))
pd.DataFrame(
    {"sample_id": [t.stem for t in tables], "input_path": [f"cells/{t.name}" for t in tables]}
).to_csv("data/manifest.csv", index=False)
```

## Units

A unit is what one bag, one label and one fold member is:

| `data.unit` | Unit | Id |
|---|---|---|
| `region` (default) | the whole region | `region_id` |
| `tile` | squares of `data.tiling.size`, every `data.tiling.stride` | `<region_id>_tile<k>` |
| `polygon` | every polygon of the region's `polygons_path` | `<region_id>_polygon<k>` |

Units with fewer than `data.min_cells` cells are dropped. Tiles and polygons
inherit the label of their region unless the label file lists them.

## Labels

```csv
region_id,label,patient_id
S01_roi1,1,P01
S01_roi2,1,P01
S02,0,P02
```

* `data.labels.id_column` holds unit, region or sample ids (a unit takes the
  label of its own id, else of its region, else of its sample).
* `task.target_columns` selects the target column(s): one class column for
  classification (numbers, or text encoded in sorted order), one or more
  numeric columns for regression, `[time, event]` for survival.
* `data.labels.patient_column` groups units: random splits, cross-validation
  folds and patient-level scores never separate a patient's units.

Units without a label are skipped for supervised training and used for
pretraining.

## Folds

A JSON list of lists of unit (or region, or sample) ids, one list per outer
fold, grouped by patient (cross-validation refuses a file that puts a unit or a
patient in two folds):

```json
[["S01_roi1", "S01_roi2", "S07"], ["S02", "S05_roi1"], ["S03", "S04"]]
```

This writes five folds that keep every patient in one fold and balance the
classes over patients (for survival, stratify on the event column):

```python
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

labels = pd.read_csv("data/labels.csv")  # region_id, label, patient_id
patients = labels.groupby("patient_id")["label"].first()  # one label per patient
splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(patients.index, patients)
folds = [
    labels.loc[labels["patient_id"].isin(patients.index[test]), "region_id"].tolist()
    for _, test in splits
]
json.dump(folds, open("data/folds.json", "w"))
```

Every fold should hold every class (the command warns otherwise: a fold's AUROC
is undefined with a single class), so use fewer folds than the smaller class
has patients. Keep the file with the data so that every model is evaluated on
the same folds.

## Cell graphs

The first command that needs the graphs builds them into
`data.processed_dir`: a Delaunay triangulation of every unit, with edge
features `log_distance` (the bounded log length
`clip((ln d - 2) / 3, 0, 1)`, which the encoder conditions on) and `neighbor`
(length at most `data.graph.neighbor_cutoff`). The directory is reused while
the settings that shaped it are unchanged; after a change, a command refuses
to reuse it and asks for `data.force_precompute=true` or a new
`data.processed_dir`. Labels and splits are not part of the graphs, so one
directory serves pretraining, every fold and every task.

### Reducing many features

For high-dimensional features (for example gene expression in `X`),
`data.feature_reducer.name=pca` projects them onto `n_components` principal
components fitted on all cells (`fit_on: all`, label-free) or on the training
split (`fit_on: train`, single random split only).
To score a new cohort in the basis of a trained one, set
`data.feature_reducer.state_from=<trained cohort>/processed/processed_index.json`.
