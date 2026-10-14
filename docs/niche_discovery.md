# Niche Discovery And Cross-Fold Replication

Everything downstream of a trained model works on its **cell cache**: for every
cell of every region, the encoder embedding of the ego-graph rooted at that
cell (`E`, fp16), the per-class instance logits (`LG`) and the raw per-class
attention logits (`AR`), plus the head bias, cell types, positions and the cell
graph once per region.

## 1. Extract the cells

```bash
# every refit model of a nested CV (held-out regions recorded per model)
grass-mil-extract regime=linear_ftall_reg data=... \
  extract.nested_cv_summary=runs/nested_ftall_reg/nested_cv_summary.json \
  paths.output_dir=runs/cells

# one explicit checkpoint
grass-mil-extract regime=mil data=... +extract.models.final=runs/train/checkpoints/last.ckpt
```

Output: `<output_dir>/cells/<model>/index.json` + one `<region>.npz` per region.
The model is composed from the same `regime`/`model` keys as the training run so
the checkpoint fits; ego-graph settings come from `data.sampler.runtime`.

## 2. Niches of one model (`grass-mil-niches`)

```bash
grass-mil-niches cells_dir=runs/cells/fold2 output_dir=runs/niches_fold2 \
  positive_name=STAS held_out_auroc=0.664
```

Default settings (the preprint recipe; every key is overridable, no sweep):

| key | default | meaning |
|---|---|---|
| `regions` | `held_out` | cells of the regions the model never trained on (`all`, a list or a file are accepted) |
| `space` | `gnn` | encoder embedding; `attention` = the 8-d pre-tanh attention activation (`attention_ckpt`) |
| `reduction` | PCA, 10 components | fitted on a random sample of `sample_size` cells |
| `l2_normalise` | true | cluster on L2-normalised PCs, i.e. the cosine geometry |
| `clustering` | HDBSCAN, `min_cluster_size` 300, `min_samples` 1, `eom` | any backend of the clustering registry (`kmeans` with `n_clusters`, `agglomerative`, `dbscan`, `optics`, `spectral`) |
| `assign_k` | 15 | k-NN assignment of unsampled cells on the PCs (backends with `predict`, e.g. k-means, assign directly) |
| `sample_size` | 100000 | cells used to fit reducer and clustering (`null` = all) |

Per cell, the **attention-weighted margin** is
`relattn_i = N_r [A_{i,1}(l_{i,1} - b_1) - A_{i,0}(l_{i,0} - b_0)]`
with the attention normalised over the region's cells per class, so that the
margins of a region sum to `N_r` times its bias-free bag margin.

Outputs: `labels.npy`, `cluster_model.pkl` (joblib; reducer + assigner, reusable
on other cells), `cells.csv` (region, cell index, cluster, cell type, relattn,
x, y), `clusters.csv`, `coloc_log2.npy`, `report.md`, `pca_triptych.png`,
`umap_triptych.png` (if `umap-learn` is installed), `clusters_overview.png`,
`colocalisation.png`.

`clusters.csv` columns: share of cells, regions present (> `present_threshold`),
positive / negative regions present, composition (`comp_<type>`), top types,
enriched types (> 1.5x the cohort frequency), empirical fraction of cells from
positive regions, AUROC of the per-region share, `relattn` with a bootstrap CI
over regions (`n_boot`), signed share of the region margin, sign consistency,
attention lift. The report adds the Spearman concordance between empirical
association and margin and the co-localisation `log2(obs/exp)` of cell-cell edges
between clusters.

## 3. Cross-fold replication (`grass-mil-metaclusters`)

A clustering of one model lives in that model's space. To ask which niches
replicate, cluster every fold model's held-out cells with the same fixed
settings and match the clusters across folds by composition:

```bash
grass-mil-metaclusters cells_dir=runs/cells output_dir=runs/meta \
  labels.label_file=data/raw/labels.csv labels.label_column=STAS labels.patient_column=PATIENT \
  positive_name=STAS
```

Stages (`stage=all|cluster|match|transfer|basis`):

1. **cluster**: per fold model, `grass-mil-niches` settings on its held-out cells -> `cluster/fold_clusters.csv`, `labels_<model>.npy`, `cluster_model_<model>.pkl`.
2. **match**: Hellinger distance between cluster compositions (`match.distance=aitchison` for CLR euclidean), constrained average linkage that never merges two clusters of the same fold, threshold from a scan maximising the number of metaclusters replicated in `>= match.min_folds` folds (ties -> fewer metaclusters) or fixed (`match.threshold=0.4`). Outputs `match/metaclusters.csv` (members, within distance, per-fold empirical association and margin, sign agreement, CI exclusion count), `metacluster_patient_shares.csv` (each region labelled once, by the model that held it out) and `metacluster_patient_tests.csv` (Mann-Whitney + BH, AUROC of the share), `report.md`, `metaclusters_overview.png`.
3. **transfer**: every model labels every region in its own space with its saved cluster model; `transfer/transfer_meta_by_model.csv` (share, association, margin on train vs held-out regions per metacluster and model), `transfer_model_ari.csv` (pairwise ARI on cells both models assign), consensus patient shares and tests.
4. **basis**: a shared label-free k-means basis (`basis.k`, standardised embeddings of `basis.space_model`, `basis.sample_size` cells) with the attribution of every fold model split into held-out / train / all (model, region) pairs: `basis/basis_attribution.csv`, `report_basis.md`, `basis_attribution.png`.

## 4. The interpretability report (`grass-mil-report`)

The HTML/PDF report of `docs/interpretability_suite.md` works on instance
tables exported by `grass-mil-predict`. Its clustering step accepts the same
sample / L2 / assignment recipe (`clustering.sample_size`,
`clustering.l2_normalise`, `clustering.assign_k`) and its margin attribution
checks the additive identity against the bag table (`data.bag_table`).
