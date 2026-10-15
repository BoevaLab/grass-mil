# GRASS-MIL: graph-based representation and discovery of phenotype-associated spatial structures with multiple instance learning

[![Preprint](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.08.757722-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.10.08.757722v1)
[![Tests](https://github.com/BoevaLab/grass-mil/actions/workflows/test.yml/badge.svg)](https://github.com/BoevaLab/grass-mil/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)

GRASS-MIL predicts a patient- or region-level phenotype from cell-resolved
spatial omics data (multiplexed imaging, spatial transcriptomics with cell
types) and shows *which spatial structures* carry the prediction. It
represents every cell by its spatial neighbourhood, learns those
representations without labels, pools them with per-class attention into a
prediction whose every logit is an exact sum of per-cell contributions, and
turns the contributions into **niches**: recurrent cellular neighbourhoods,
each with its composition, its association with the phenotype, the model's
attribution to it and its spatial organisation.

## The method in one paragraph

Cells of a region form a Delaunay graph whose edges carry the bounded log edge
length. A GINE graph network encodes the *k*-hop ego-graph around every cell
(4 hops, width 256) and is pretrained with BGRL on all regions, label-free. For
prediction, the ego-graphs of a region are the instances of one bag: an
instance head gives per-class logits, a projected gated attention gives
per-class weights, and the bag logit is `L_c = sum_i A[i,c] * l[i,c]`, so every
cell's contribution `A[i,c] (l[i,c] - r_c)` relative to the model's average
cell `r` is exact rather than approximated.
By default the pretrained encoder is fine-tuned end to end with a linear
instance head initialised by a probe on its frozen embeddings. Evaluation is a
nested, patient-grouped cross-validation. Interpretation clusters the held-out
cells of a model in its attention space (cosine HDBSCAN) and describes every
niche by its composition, empirical association, attention-weighted margin and
classical spatial statistics within the niche. Every setting is a key of the
[configuration](docs/configuration.md) and can be changed from the command line.

## Install

```bash
git clone https://github.com/BoevaLab/grass-mil.git
cd grass-mil
python -m venv .venv && source .venv/bin/activate
pip install -e ".[io]"
# torch-sparse, matching your PyTorch build (ego-graph sampling):
pip install torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html
```

or `conda env create -f environment.yaml`. Extras: `io` (h5ad inputs, polygon
units), `sce` (SingleCellExperiment inputs; also needs R with Bioconductor),
`test`.

## Quick start

The [vignettes](vignettes/) run the whole workflow on a public MIBI-TOF dataset
on a CPU in a few minutes:

```bash
pip install squidpy jupyter
jupyter lab vignettes/
```

## Workflow

Describe the cohort once (copy
[`configs/experiment/example_cohort.yaml`](src/grass_mil/configs/experiment/example_cohort.yaml)
to `my_cohort.yaml` and fill in paths and column names; see
[data preparation](docs/data_preparation.md)), then:

| Step | Command | Output |
|---|---|---|
| 1. Pretrain the encoder, label-free (builds the cell graphs on first use) | `grass-mil-train experiment=my_cohort regime=pretrain` | `checkpoints/last.ckpt` |
| 2. Evaluate by nested cross-validation | `grass-mil-nested-cv experiment=my_cohort model.init_from_ckpt=<step 1>` | per-fold metrics (mean ± s.d.), out-of-fold unit and patient predictions, one model per fold |
| 3. Score every cell with the fold models | `grass-mil-extract experiment=my_cohort extract.nested_cv_summary=<step 2>/nested_cv_summary.json` | per-cell embeddings, attention and logits |
| 4. Discover and describe niches | `grass-mil-niches cells_dir=<step 3>/cells/fold0` | niche table, spatial descriptors, figures, report |
| Optional: replication across folds | `grass-mil-metaclusters cells_dir=<step 3>/cells` | metaclusters of niches found by several models |
| Optional: model-free statistics | `grass-mil-spatial-stats cells_dir=<step 3>/cells/fold0` | classical spatial statistics compared between classes |
| Score new data | `grass-mil-predict experiment=new_cohort ckpt_path=<checkpoint>` | unit and patient predictions |

A single model on all labelled data (`grass-mil-train experiment=my_cohort
model.init_from_ckpt=<step 1>`) uses a random patient-grouped split for
validation and testing. Add `trainer=gpu` to train on a GPU.

## Changing the defaults

```bash
# another fine-tuning regime: slower encoder with regularisation
grass-mil-nested-cv experiment=my_cohort regime=finetune_reg model.init_from_ckpt=...

# a shallower encoder on 3-hop ego-graphs, cell type only (no continuous feature)
grass-mil-train experiment=my_cohort regime=pretrain model.encoder.num_layers=3 \
  data.sampler.depth=3 model.encoder.hidden_dim=128 model.encoder.out_dim=128 data.input.features=null

# radius-capped ego-graphs and another contact distance
grass-mil-train experiment=my_cohort data.sampler.ego_radius.enabled=true \
  data.sampler.ego_radius.radius_per_hop=30 data.graph.neighbor_cutoff=25 ...

# more cells per bag, a longer schedule, five inner folds
grass-mil-nested-cv experiment=my_cohort data.sampler.roots_per_unit=512 \
  trainer.max_epochs=40 nested_cv.inner_k=5 ...

# survival instead of classification
grass-mil-nested-cv experiment=my_cohort task=survival task.target_columns=[time,event] ...

# niches in the encoder embedding with k-means
grass-mil-niches cells_dir=... clustering.space=embedding clustering.pca_components=10 \
  clustering.method=kmeans clustering.n_clusters=12
```

## Documentation

- [Data preparation](docs/data_preparation.md): cell tables, manifest, units, labels, folds, graphs
- [Training and evaluation](docs/training_and_evaluation.md): the model, pretraining, regimes, tasks, nested cross-validation, prediction
- [Niche analysis](docs/niche_analysis.md): per-cell outputs, niches, descriptors, survival, replication, model-free statistics
- [Configuration](docs/configuration.md): groups, precedence and every key

## Citation

If you use GRASS-MIL, please cite:

> Rabuzin L, Yates J, Xu M, Boeva V. *GRASS-MIL: graph-based representation
> and discovery of phenotype-associated spatial structures with multiple
> instance learning.* bioRxiv (2026).
> doi: [10.64898/2026.10.08.757722](https://www.biorxiv.org/content/10.64898/2026.10.08.757722v1)

```bibtex
@article{rabuzin2026grassmil,
  title   = {GRASS-MIL: graph-based representation and discovery of phenotype-associated spatial structures with multiple instance learning},
  author  = {Rabuzin, Lovro and Yates, Josephine and Xu, Mina and Boeva, Valentina},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.08.757722},
  url     = {https://www.biorxiv.org/content/10.64898/2026.10.08.757722v1}
}
```

## License and contact

MIT license. Questions and issues: open a GitHub issue, or contact the Boeva
lab (Department of Computer Science, ETH Zurich).
