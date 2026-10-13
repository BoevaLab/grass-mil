# Method Correspondence With The Preprint

Each item of the preprint's method maps to a preset, a module and a test. The
settings are the defaults of the presets; all remain overridable.

| Preprint element | Default | Config | Code | Test |
|---|---|---|---|---|
| Delaunay cell graph, contact edges, bounded log edge length `clip((ln d - 2)/3, 0, 1)` | `edge_features=[log_distance, neighbor]`, `neighbor_cutoff_um` in the data's units | `data/spatial_omics.yaml` (`graph_builder`) | `data/components/graph_builders.py::log_distance_feature` | `tests/models/test_fidelity_phase1.py` |
| Node input: cell-type embedding + linear projection of continuous features, xavier init | `input_init=xavier`, `categorical_embedding` | `model/encoder/gine_length.yaml` | `models/components/embeddings.py` | phase 1 |
| GINE encoder 4 x 256, learnable epsilon, layer norm, max pooling over the ego-graph | `num_layers=4`, `hidden_dim=out_dim=256`, `train_eps=true`, `norm=layernorm`, `pooling=max` | `model/encoder/gine_length.yaml` | `models/components/backbones.py` | phase 1 |
| Ego-graphs: k = 4 hops, all neighbours, no radius cap | `depth=4`, `num_neighbors=-1`, `ego_radius.enabled=false` | `data/sampler/ego_graph_mil.yaml` | `data/components/samplers.py` | `tests/models/test_fidelity_phase2.py` |
| BGRL pretraining "A3": roots from all regions, 64 per step, AdamW 5e-5 / 1e-5, warmup 1000, 200k steps, momentum 0.99 -> 1, encoder dropout 0.1, DropImportance (mu 0.25, p_lambda 0.45) + feature noise 0.2 on size, checkpoint every 5000 steps | `experiment/grass_mil_pretrain_a3.yaml` | `model/bgrl_a3.yaml`, `task/pretrain_bgrl_a3.yaml`, `data/sampler/ego_graph_pretrain.yaml`, `callbacks/pretrain_checkpoint.yaml` | `models/bgrl_module.py`, `data/components/merge.py`, `models/training/augmentations.py` | `tests/models/test_fidelity_phase4.py` |
| Projected gated attention (32 / 8, dropout 0.25, one channel per class, kaiming) | `model/attention/gated_projected.yaml` | | `models/components/attention.py` | phase 1 |
| Logit-space pooling `L_c = sum_i A[i,c] l[i,c]`, exact per-cell contributions | `task.aggregation=mil_attention` | `task/finetune_mil.yaml` | `models/training/bagging.py`, `interpretability/core/attribution.py` | `tests/models/test_bagging.py`, phase 6 |
| One region = one bag, one softmax, one loss; class-weighted CE (train split, mean 1); fixed 1024-root validation; all-cells test | `unit_as_bag=true`, `subsample_count`, `class_weights=auto_train`, `val_sampler`, `test_sampler=null` | `regime/mil.yaml` | `models/supervised_module.py`, `models/training/class_weights.py` | phase 2 |
| Probe pre-fit: 96 roots per region, StandardScaler + LogisticRegression(C 1, balanced), folded into a linear head | `probe.yaml` | `regime/linear_*.yaml` (`task.head_init_path=auto`) | `models/training/probe.py` | `tests/models/test_fidelity_phase3.py` |
| Regimes probehead / ftALL / ftALL-reg: learning-rate scales (encoder 0 / 1 / 0.3, head 0 / 0.01, attention 2), weight decay 1e-3 / 1e-2, dropout 0.2 and edge/feature drop 0.05 in the regularised regime, bag 512 / 256, per-step cosine over the run, hyperbatch 12, clip 1, AMP, early stopping on region loss (patience 8, min 10 epochs), checkpoint on region AUROC | `regime/*.yaml`, `scheduler/cosine_mil_step.yaml`, `callbacks/mil_selection.yaml` | | `models/training/optimization.py`, `models/supervised_module.py` | phase 2 |
| Nested CV: patient-grouped outer folds, inner `StratifiedGroupKFold(3, seed 0)`, best epoch = argmax val AUROC, median refit, test once | `nested_cv.yaml` | | `nested_cv.py`, `data/components/splits.py` | phase 3 |
| Patient score: mean of core probabilities (merged bag as alternative) | `aggregation.bag_scope=patient` | `inference/aggregation/default.yaml` | `inference/aggregation.py` | `tests/models/test_fidelity_phase5.py` |
| Cell cache of every fold model (E fp16, instance logits, raw attention, bias) | `extract.yaml` | | `inference/cell_cache.py` | phase 5 |
| relattn margin `N_r [A_1 (l_1 - b_1) - A_0 (l_0 - b_0)]` | | | `interpretability/niches/margins.py` | `tests/interpretability/test_fidelity_phase6.py` |
| Cosine HDBSCAN: PCA-10 on 100k cells, L2, mcs 300, min_samples 1, eom, 15-NN assignment; niche table, co-localisation, figures | `niches.yaml` | | `interpretability/core/clustering.py::cosine_hdbscan`, `interpretability/niches/*` | phase 6 |
| Metaclusters: per-fold clustering, Hellinger distance, constrained average linkage, threshold scan or 0.40, patient shares (each core labelled once), transfer, pairwise ARI, shared k-means basis | `metaclusters.yaml` | | `interpretability/metaclusters/*` | `tests/interpretability/test_fidelity_phase7.py` |
| Bootstrap AUROC CI (2000, seed 0), DeLong, within-fold z / rank, operating points at within-fold quantile rules | `scores.yaml` | | `evaluation/*` | phase 5 |
| Neighbourhood enrichment (200 permutations), cellular neighbourhoods (10-NN windows, k-means 10 on 200k windows, CLR), contacts, cross-K (r 20/50/100, pi r^2) | `spatial_stats.yaml` | | `spatial_stats/*` | `tests/spatial_stats/test_fidelity_phase8.py` |

## Known deviations

- The LR warmup starts at `start_factor=0.0` through `LinearLR`, which clamps to
  1e-4 of the base rate at step 0 (the reference starts at exactly 0); the
  difference is one step.
- `grass-mil-nested-cv` rejects inner-fold seeds that leave a fold without one
  class and tries the next seed; on cohorts with more than a handful of patients
  per class the seed is never changed.
- The niche pipeline has no clustering sweep; it reports the settings it was
  given (the preprint chose its settings by a sweep once and then fixed them).
- Validation is computed on a fixed 1024-root subset per region and test on all
  cells, as in the reference; both are configurable.

## Validation against the original runs

Recorded after the fidelity check of the public package against the preprint's
NSCLC runs (private data; the converter lives outside the repository):

| Check | Reference | Public package | Status |
|---|---|---|---|
| Frozen embeddings of three cores (per-cell cosine with the reference fp16 cache; 875-5150 cells each) | 1.0 | median 0.993-0.995, minimum 0.97, 91-95 % of cells >= 0.99 | passed |
| Probe region AUROC, outer fold 2: inner folds 0 / 1 / 2 and refit (scored on the held-out fold) | 0.482 / 0.304 / 0.564 / 0.661 | 0.471 / 0.354 / 0.555 / 0.685 | passed (max deviation 0.05 on inner fold 1, 0.024 on the refit) |
| Held-out AUROC, `linear_ftall_reg`, outer fold 2 (inner best epochs, refit length E) | 0.664 (1 / 1 / 1, E = 1) | 0.660 (3 / 1 / 3, E = 3) | passed (deviation 0.005) |
| Held-out AUROC, `linear_probehead`, outer fold 2 (inner best epochs, refit length E) | 0.627 (10 / 14 / 1, E = 10) | 0.665 (16 / 15 / 1, E = 15) | deviation 0.038, slightly above the 0.03 target; single-seed runs on 86 held-out regions (bootstrap CI about +/- 0.12) |
| Niches of fold 2: cluster count, noise, Spearman(emp, relattn) | reference report | pending | |
| Neighbourhood-enrichment z-scores on all 426 cores (numpy backend, 200 permutations) vs the reference squidpy table: Spearman of the patient-mean z per group pair, 225 pairs | > 0.9 | 0.984 (STAS+) and 0.977 (STAS-); the same six pairs lead the ranked test table | passed |
