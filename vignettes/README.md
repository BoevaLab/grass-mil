# Vignettes

Worked examples that run `grass-mil` end to end on a public dataset, so you can
see the whole path from a published AnnData object to a spatial-niche report
before pointing the framework at your own cohort.

| Vignette | What it covers |
|---|---|
| [`01_anndata_to_spatial_graphs.ipynb`](01_anndata_to_spatial_graphs.ipynb) | Turning a `squidpy` dataset into the manifest + per-sample files `grass-mil` expects, building the cellular graphs, and inspecting them |
| [`02_pretraining_and_niches.ipynb`](02_pretraining_and_niches.ipynb) | BGRL self-supervised pretraining, exporting per-instance embeddings, and discovering spatial niches with the interpretability report |

Run them in order: the second reads the graphs the first builds.

## Dataset

Both use `squidpy.datasets.mibitof()` — MIBI-TOF imaging of colorectal
carcinoma: 3 images, ~3,300 cells, 36 protein markers, 8 annotated cell types
(Hartmann et al.). It is a natural fit because `grass-mil` is built for
cell-resolved spatial proteomics: every cell has coordinates, a type, and a
marker profile. It downloads in a few seconds and everything here runs on CPU in
a few minutes.

## Scope, honestly

Three images from two donors is enough to demonstrate the mechanics, and enough
for unsupervised niche discovery. It is **not** enough for supervised
evaluation: a correct sample-level split leaves one image per fold, and the only
genuine bag-level variable — donor identity — ends up single-class in training.

The vignettes therefore treat the supervised step as plumbing (it exists because
prediction needs a supervised checkpoint) and put the weight on the parts that
stand on their own: graph construction, self-supervised pretraining, and niche
discovery. Vignette 2 says so at the top rather than burying it.

## Setup

```bash
pip install "grass-mil[interpretability,io]" squidpy matplotlib jupyter
jupyter lab vignettes/
```

`squidpy` is used only to fetch the public dataset; it is not a runtime
dependency of `grass-mil`.

Both notebooks write to `vignettes/vignette_work/`, which is git-ignored. Delete
it to start clean.
