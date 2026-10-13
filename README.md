# GRASS-MIL: graph-based representation and discovery of phenotype-associated spatial structures with multiple instance learning

[![Preprint](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.08.757722-b31b1b)](https://www.biorxiv.org/content/10.64898/2026.10.08.757722v1)
[![Tests](https://github.com/BoevaLab/grass-mil/actions/workflows/test.yml/badge.svg)](https://github.com/BoevaLab/grass-mil/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)

GRASS-MIL predicts a patient- or region-level phenotype from cell-resolved
spatial omics data (multiplexed imaging, spatial transcriptomics with cell
types) and tells you *which spatial structures* carry the prediction. It
represents every cell by its spatial neighbourhood, learns those
representations without labels, pools them with per-class attention into a
bag prediction whose every logit is an exact sum of per-cell contributions,
and turns the contributions into **niches**: recurrent cellular
neighbourhoods with an association to the phenotype and an attribution by
the model, checked for replication across cross-validation folds.

This repository is the public implementation accompanying the preprint:

> Rabuzin L, Yates J, Xu M, Boeva V. *GRASS-MIL: graph-based representation
> and discovery of phenotype-associated spatial structures with multiple
> instance learning.* bioRxiv (2026).
> doi: [10.64898/2026.10.08.757722](https://www.biorxiv.org/content/10.64898/2026.10.08.757722v1)

## The method in one paragraph

Cells of a region form a Delaunay graph whose edges carry the bounded log
edge length. A GINE graph network encodes the *k*-hop ego-graph around every
cell (4 hops, width 256 by default) and is pretrained with BGRL on all regions,
label-free. For prediction, the ego-graphs of a region are the instances of
one bag: an instance head gives per-class logits, a projected gated attention
net gives per-class weights, and the bag logit is `L_c = sum_i A[i,c] * l[i,c]`
(logit-space pooling), so every cell's contribution `A[i,c] (l[i,c] - b_c)` is
exact rather than approximated. Fine-tuning regimes range from attention-only
on a frozen encoder with a probe-fitted linear head to end-to-end fine-tuning.
Evaluation is a nested, patient-grouped cross-validation. Interpretation
clusters the held-out cells of each fold model in its own embedding space
(cosine HDBSCAN by default), scores every cluster by its empirical association
and its attention-weighted margin, and matches clusters across folds into
metaclusters that replicate. Every setting above is a default of a Hydra
config, not a constant: depth, width, ego-graph hops, node and edge features,
attention sizes, learning rates, bag sizes, schedules, augmentations, fold
counts and clustering settings are all overridable from the command line.

## Install

```bash
git clone https://github.com/BoevaLab/grass-mil.git
cd grass-mil
python -m venv .venv && source .venv/bin/activate
pip install -e ".[interpretability,io,stats,plots]"
```

`torch-geometric` needs the compiled `torch-sparse` / `torch-scatter` wheels
matching your PyTorch build for the ego-graph sampler (see
[`.github/workflows/test.yml`](.github/workflows/test.yml) for the exact pip
lines used in CI, or use [`environment.yaml`](environment.yaml) with conda).
Extras: `interpretability` (UMAP, plotly, matplotlib), `io` (h5ad, polygons),
`stats` (statsmodels), `plots`, `report` (PDF rendering), `spatial-stats`
(squidpy backend), `sce` (R inputs).

Installing exposes the CLI entry points below; the Hydra config tree ships
inside the package, so every command works from any directory.

## Quick start (public demo data, CPU, a few minutes)

The vignettes run the whole chain on `squidpy.datasets.mibitof()` (three
MIBI-TOF images, ~3,300 cells):

```bash
pip install squidpy jupyter
jupyter lab vignettes/
```

1. [`01_anndata_to_spatial_graphs.ipynb`](vignettes/01_anndata_to_spatial_graphs.ipynb): AnnData -> manifest -> cellular graphs.
2. [`02_pretraining_and_niches.ipynb`](vignettes/02_pretraining_and_niches.ipynb): BGRL pretraining (`experiment=grass_mil_pretrain_a3`), fine-tuning (`regime=mil`), instance export, interpretability report.
3. [`03_nested_cv_niches_and_metaclusters.ipynb`](vignettes/03_nested_cv_niches_and_metaclusters.ipynb): nested CV, cell extraction, niches, metaclusters, scores, spatial statistics.

## Workflows

Inputs are a manifest CSV (`sample_id,input_path,input_type[,region_id,polygons_path]`),
per-sample cell tables (CSV/TSV, h5ad, SingleCellExperiment), a label CSV with a
unit-id column, a target column and a patient column, and a fold file (JSON list
of lists of unit ids). See [`docs/dataset_preparation.md`](docs/dataset_preparation.md).

| Step | Command | Preset / keys | Outputs | Doc |
|---|---|---|---|---|
| Build graphs | `grass-mil-train data=... +trainer.fast_dev_run=true` (or any run) | `data.graph_builder.kwargs.edge_features=[log_distance,neighbor]`, `data.sample_unit=full` | `processed_index.json`, graph `.pt` files | [dataset_preparation](docs/dataset_preparation.md) |
| Pretrain the encoder | `grass-mil-train experiment=grass_mil_pretrain_a3 data=...` | `model=bgrl_a3`, `task=pretrain_bgrl_a3`, `data/sampler=ego_graph_pretrain` | `checkpoints/last.ckpt` | [workflows](docs/workflows_train_infer.md) |
| Fine-tune one split | `grass-mil-train regime=mil data=... model.init_from_ckpt=...` | `regime=mil|mil_frozen|linear_probehead|linear_ftall|linear_ftall_reg`, `data.split.explicit.*` | checkpoints, `metrics_per_epoch.jsonl`, `test_predictions.csv` | [nested_cv](docs/nested_cv.md) |
| Probe head | `grass-mil-probe regime=linear_probehead data=...` | `probe.*` | `probe_head.pt` | [nested_cv](docs/nested_cv.md) |
| Nested cross-validation | `grass-mil-nested-cv regime=... nested_cv.folds_file=folds.json` | `nested_cv.*` | `nested_cv_summary.json`, `oof_predictions.csv`, `oof_patient_predictions.csv` | [nested_cv](docs/nested_cv.md) |
| Predict / aggregate | `grass-mil-predict ckpt_path=... regime=...` | `aggregation.bag_scope=patient aggregation.mode=mean_probability|attention_weighted` | `predictions.csv`, `metrics.json`, instance tables | [inference](docs/inference_utility_suite.md) |
| Extract cells | `grass-mil-extract regime=... extract.nested_cv_summary=...` | `extract.*` | `cells/<model>/<region>.npz` | [niche_discovery](docs/niche_discovery.md) |
| Niches of one model | `grass-mil-niches cells_dir=cells/fold2` | `reduction.*`, `clustering.*`, `sample_size`, `assign_k` | `clusters.csv`, `cells.csv`, `report.md`, figures | [niche_discovery](docs/niche_discovery.md) |
| Cross-fold replication | `grass-mil-metaclusters cells_dir=cells labels.label_file=...` | `stage`, `match.*`, `transfer.*`, `basis.*` | `metaclusters.csv`, patient shares and tests, transfer and basis reports | [niche_discovery](docs/niche_discovery.md) |
| Evaluation statistics | `grass-mil-scores input=oof_predictions.csv` | `scores`, `strata`, `operating_rules` | AUROC CIs, DeLong, operating points | [evaluation_stats](docs/evaluation_stats.md) |
| Classical spatial statistics | `grass-mil-spatial-stats cells_dir=... reference=[tumor]` | `analyses`, `group_map`, `nhood.*`, `cn.*`, `crossk.*` | neighbourhood enrichment, cellular neighbourhoods, contacts, cross-K tests | [spatial_statistics](docs/spatial_statistics.md) |
| Interpretability report | `grass-mil-report data.instance_table=...` | `clustering.*`, plugins | HTML / PDF report | [interpretability_suite](docs/interpretability_suite.md) |

## Running the analyses of the preprint

The preprint's numbers come from this chain with the default settings:

1. graphs with `edge_features=[log_distance,neighbor]` and the cohort's native
   coordinate units (`neighbor_cutoff_um` is the contact distance in those units);
2. `experiment=grass_mil_pretrain_a3` on every region, label-free;
3. `grass-mil-nested-cv` with patient-grouped outer folds, inner `k=3`, the
   regimes `linear_probehead` ("probehead"), `linear_ftall` ("ftALL"),
   `linear_ftall_reg` ("ftALL, lr x0.3 + regularisation") and `mil`;
   out-of-fold region scores, patient scores as the mean of a patient's regions
   (merged bags via `grass-mil-predict`);
4. `grass-mil-scores` for bootstrap intervals, per-fold means, DeLong
   comparisons and operating points;
5. `grass-mil-extract` of the fold refits, `grass-mil-niches` on each model's
   held-out cells (PCA-10, L2, HDBSCAN mcs 300 / min_samples 1 / eom, 15-NN
   assignment), `grass-mil-metaclusters` (Hellinger matching with cannot-link
   within fold, transfer to all regions, shared k-means basis);
6. `grass-mil-spatial-stats` for the model-free comparison.

[`docs/method_correspondence.md`](docs/method_correspondence.md) maps every
section of the preprint to the preset, module and test that implements it and
records the validation against the original runs.

### Changing the defaults

```bash
# shallower encoder, 3-hop ego-graphs, cell type only (no size feature)
grass-mil-nested-cv regime=mil model.encoder.num_layers=3 model.encoder.hidden_dim=128 \
  model.encoder.out_dim=128 data.sampler.runtime.depth=3 data.csv.molecular_columns=[] ...

# radius-capped ego-graphs and a different contact distance
grass-mil-train regime=mil data.sampler.runtime.ego_radius.enabled=true \
  data.sampler.runtime.ego_radius.radius_per_hop=30 data.graph_builder.kwargs.neighbor_cutoff_um=25 ...

# more roots per bag, longer schedule, five inner folds
grass-mil-nested-cv regime=linear_ftall data.sampler.runtime.subsample_count=512 \
  trainer.max_epochs=40 nested_cv.inner_k=5 ...

# niches with k-means instead of HDBSCAN, 20 PCs, no sampling
grass-mil-niches cells_dir=... clustering.method=kmeans clustering.n_clusters=12 \
  reduction.n_components=20 sample_size=null
```

Every key is documented in [`docs/hyperparameter_reference.md`](docs/hyperparameter_reference.md).

## Documentation

- [Dataset preparation](docs/dataset_preparation.md): manifest, inputs, labels, patient column, fold files
- [Training and inference workflows](docs/workflows_train_infer.md)
- [Nested cross-validation and regimes](docs/nested_cv.md)
- [Niche discovery and cross-fold replication](docs/niche_discovery.md)
- [Evaluation statistics](docs/evaluation_stats.md)
- [Classical spatial statistics](docs/spatial_statistics.md)
- [Inference utility suite](docs/inference_utility_suite.md)
- [Interpretability report suite](docs/interpretability_suite.md) and [extension guide](docs/interpretability_extension_guide.md)
- [Hyperparameter reference](docs/hyperparameter_reference.md)
- [Capabilities and modes](docs/capabilities_and_modes.md)
- [Model component contracts](docs/model_component_contracts.md)
- [Method correspondence with the preprint](docs/method_correspondence.md)
- [Repository map](docs/repository_map.md)

## Citation

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

A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

## License and contact

MIT license. Questions and issues: open a GitHub issue, or contact the Boeva
lab (Department of Computer Science, ETH Zurich).
