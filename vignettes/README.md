# Vignettes

The whole GRASS-MIL workflow on a public dataset, run on a laptop.

| Vignette | Content |
|---|---|
| [`01_anndata_to_cell_graphs.ipynb`](01_anndata_to_cell_graphs.ipynb) | cell tables, manifest, units (100 µm tiles), cell graphs, labels and folds |
| [`02_pretraining_training_and_niches.ipynb`](02_pretraining_training_and_niches.ipynb) | label-free pretraining, fine-tuning with the default regime, per-cell outputs, niche discovery and description |
| [`03_cross_validation_and_replication.ipynb`](03_cross_validation_and_replication.ipynb) | nested cross-validation, niches of a fold model on its held-out tiles, replication of niches across fold models, model-free spatial statistics |

Run them in order: each one uses the outputs of the previous one. Vignette 2
pretrains for 3,000 steps (a few minutes on a GPU, about ten on a CPU); the
rest takes a few minutes.

## Data and phenotype

`squidpy.datasets.mibitof()`: MIBI-TOF images of colorectal carcinoma, 3
images, about 3,300 cells, 8 annotated cell types (Hartmann et al.). Only the
cell types enter the model. The images are cut into 100 µm tiles, and each tile
is labelled by a phenotype derived from the data, CD8 T-cell infiltration (a
CD8 fraction at or above the median), so the predictions and niches can be
checked against what is known to drive them. Three images make small folds:
the numbers illustrate the procedure, not the performance of the method.

## Setup

```bash
pip install -e ".[io]" squidpy jupyter
jupyter lab vignettes/
```

The notebooks write to `vignettes/vignette_work/` (ignored by git); delete it
to start again.
