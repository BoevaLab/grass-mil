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
cell's contribution `A[i,c] (l[i,c] - b_c)` is exact rather than approximated.
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
# torch-sparse for your PyTorch version (ego-graph sampling runs on the CPU, so the
# CPU wheel serves CUDA builds of PyTorch as well):
pip install torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
```

or `conda env create -f environment.yaml`. Extras: `io` (h5ad inputs, polygon
units), `sce` (SingleCellExperiment inputs; also needs R with Bioconductor),
`test`.

## Quick start

The [vignettes](vignettes/) run the whole workflow on a public MIBI-TOF dataset
in a few minutes (about ten on a CPU):

```bash
pip install squidpy jupyter
jupyter lab vignettes/
```

## Your first cohort

Work in a project folder of your own and run every command from it:

```text
my_project/
  data/                          cell tables, manifest.csv, labels.csv, folds.json
  configs/experiment/my_cohort.yaml
  runs/                          one output folder per step
```

**1. Prepare the data**: one cell table per region (CSV, h5ad or
SingleCellExperiment), a manifest listing them, a label file and the
cross-validation folds. [Data preparation](docs/data_preparation.md) describes
the formats and has short scripts for the manifest and the folds.

**2. Describe the cohort**: copy the template into `configs/experiment/` and
fill in the column names (the data are looked up in `./data`):

```bash
mkdir -p configs/experiment
python -c "import grass_mil.configs as c, pathlib, shutil; shutil.copy(pathlib.Path(c.__file__).parent / 'experiment/example_cohort.yaml', 'configs/experiment/my_cohort.yaml')"
```

Every command that reads the cohort gets `--config-dir configs
experiment=my_cohort`, and `hydra.run.dir` names the folder a step writes to.

**3. Run the steps** (`trainer=gpu` uses a GPU; drop it to run on the CPU):

```bash
# Pretrain the encoder on every region, without labels; the first command
# also builds the cell graphs (200 000 steps, about an hour on a GPU).
grass-mil-train --config-dir configs experiment=my_cohort regime=pretrain trainer=gpu \
  hydra.run.dir=runs/pretrain

# Evaluate by nested cross-validation (one model per outer fold).
grass-mil-nested-cv --config-dir configs experiment=my_cohort trainer=gpu \
  model.init_from_ckpt=runs/pretrain/checkpoints/last.ckpt hydra.run.dir=runs/nested_cv

# Score every cell with the fold models.
grass-mil-extract --config-dir configs experiment=my_cohort \
  extract.nested_cv_summary=runs/nested_cv/nested_cv_summary.json hydra.run.dir=runs/extract

# Niches of fold model 0, on the regions it did not train on.
grass-mil-niches cells_dir=runs/extract/cells/fold0 hydra.run.dir=runs/niches_fold0
```

| Folder | What to look at |
|---|---|
| `runs/pretrain` | `checkpoints/last.ckpt`, the pretrained encoder |
| `runs/nested_cv` | `nested_cv_summary.json`: the metric of every outer fold and their mean ± s.d., the number to report; `oof_predictions.csv`: every region scored by the model that held it out |
| `runs/extract/cells/fold<i>` | per-cell outputs of the model of outer fold `i` |
| `runs/niches_fold0` | `report.md`, `niches.csv`, `descriptors/`, `figures/` |

Each command ends by printing where it wrote. With a few dozen regions, see
[small cohorts](docs/training_and_evaluation.md#small-cohorts) first.

**Optional steps**, with the same conventions:

```bash
# Niches that recur across the fold models
grass-mil-metaclusters cells_dir=runs/extract/cells hydra.run.dir=runs/metaclusters
# Classical spatial statistics compared between classes, without a model
grass-mil-spatial-stats cells_dir=runs/extract/cells/fold0 hydra.run.dir=runs/spatial_stats
# Score a new cohort (its own cohort file) with one trained model
grass-mil-predict --config-dir configs experiment=new_cohort \
  ckpt_path=runs/nested_cv/outer_0/refit/checkpoints/last.ckpt hydra.run.dir=runs/predict
```

A single model on all labelled regions (`grass-mil-train --config-dir configs
experiment=my_cohort model.init_from_ckpt=runs/pretrain/checkpoints/last.ckpt`)
uses a random patient-grouped split for validation and testing.

## Changing the defaults

Any setting can be changed on the command line (`key=value`); the examples
leave out `--config-dir configs` and `hydra.run.dir`.

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
- [Training and evaluation](docs/training_and_evaluation.md): the model, pretraining, regimes, tasks, nested cross-validation, small cohorts, prediction
- [Niche analysis](docs/niche_analysis.md): per-cell outputs, niches, descriptors, survival, replication, model-free statistics
- [Configuration](docs/configuration.md): command-line basics, groups and every key

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
