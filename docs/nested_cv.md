# Nested Cross-Validation (the preprint protocol)

`grass-mil-nested-cv` runs the evaluation protocol of the preprint: fixed,
patient-grouped outer folds; stratified, patient-grouped inner folds for epoch
selection; a refit on the whole development set; one held-out prediction per
region and a mean per patient. `grass-mil-loocv` (leave-one-region-out) is
still available for very small cohorts, but nested CV is the protocol all
preprint numbers come from.

## Inputs

| Input | Form |
|---|---|
| processed graphs | any `data.processed_dir` built by `grass-mil-train` / `prepare_data` (whole regions, `data.sample_unit=full`, one graph per region) |
| labels | CSV with a region-id column, a label column (0/1 or category names) and, for patient grouping, a patient column: `data.graph_labels.{label_file,id_column,tasks,patient_column}` |
| outer folds | JSON list of lists of region ids: `nested_cv.folds_file` (patients never split across folds; build it once and version it) |
| pretrained encoder | `model.init_from_ckpt=<bgrl>.ckpt` (optional but recommended) |
| regime | `regime=mil` (default MLP head), `mil_frozen`, `linear_probehead`, `linear_ftall`, `linear_ftall_reg` |

```bash
grass-mil-nested-cv regime=linear_ftall_reg \
  data.raw_manifest_path=data/raw/manifest.csv data.processed_dir=data/processed/cohort \
  data.sample_unit=full data.csv.categorical_label_columns=[cell_type] data.csv.molecular_columns=[SIZE] \
  data.graph_builder.kwargs.neighbor_cutoff_um=55 \
  data.graph_labels.label_file=data/raw/labels.csv data.graph_labels.id_column=REGION_ID \
  data.graph_labels.tasks=[STAS] data.graph_labels.patient_column=PATIENT \
  nested_cv.folds_file=data/raw/folds.json \
  model.init_from_ckpt=runs/pretrain_a3/checkpoints/last.ckpt \
  paths.output_dir=runs/nested_ftall_reg
```

## What happens per outer fold `i`

1. Development set = every other fold, sorted by region id.
2. `StratifiedGroupKFold(nested_cv.inner_k, shuffle=True, random_state=nested_cv.inner_seed)` over the development regions, grouped by patient. A seed that leaves a fold without one class is rejected and the next seed tried (only matters for tiny cohorts).
3. For every inner fold: if the regime uses a probe head (`task.head_init_path: auto`, the `linear_*` regimes), `grass-mil-probe` is run on the inner training regions (cached under `nested_cv.probe.cache_dir`); then `grass-mil-train` with the regime, `callbacks=mil_selection` (checkpoint on region-level `val/auroc`, early stopping on `val/loss_region`, `trainer.min_epochs`), writing `metrics_per_epoch.jsonl`. The best epoch is `argmax val/auroc` (earliest on ties; `val/loss_region` is the fallback when AUROC is undefined).
4. `E = median` (`nested_cv.epoch_reduce`) of the inner best epochs.
5. Refit on all development regions for exactly `E` epochs (no early stopping, last weights; the probe is refitted on the development set), then test on fold `i` with every cell of every region (`test_sampler` = all roots).
6. Outputs: `outer_<i>/result.json`, `outer_<i>/refit/test_predictions.csv`, `outer_<i>/oof_predictions.csv`.

The run resumes: finished inner runs and outer folds are skipped (`nested_cv.resume=true`).

## Outputs

| File | Content |
|---|---|
| `nested_cv_summary.json` | `auc_mean`, `auc_sd`, `auc_folds`, per-fold `E`, inner epochs, run directories, pooled region AUROC, patient AUROC |
| `oof_predictions.csv` | one row per region: `outer`, `region_id`, `patient_id`, `label`, `logit_c`, `prob_class_c` |
| `oof_patient_predictions.csv` | mean of the region probabilities per patient |
| `probes/*.pt` | folded probe heads (`graph_head.net.0.{weight,bias}`) per split |

Score the table with `grass-mil-scores input=<dir>/oof_predictions.csv` (bootstrap
CI, per-fold mean, operating points) and extract cells of the five refit models
with `grass-mil-extract extract.nested_cv_summary=<dir>/nested_cv_summary.json`.

## Regimes

| `regime=` | encoder | instance head | attention | bag | notes |
|---|---|---|---|---|---|
| `mil` (default) | fine-tuned (lr x1) | MLP (2 layers), trained | lr x2 | 256 roots | GRASS-MIL default |
| `mil_frozen` | frozen | MLP | lr x2 | 512 | encoder in eval mode, no gradients |
| `linear_probehead` | frozen | linear, pre-fitted by the probe, frozen | lr x2 | 512 | attention is the only trainable part |
| `linear_ftall` | fine-tuned (lr x1) | linear, pre-fitted, lr x0.01, wd 1e-3 | lr x2 | 256 | |
| `linear_ftall_reg` | fine-tuned (lr x0.3) | linear, pre-fitted, lr x0.01 | lr x2 | 256 | wd 1e-2, encoder dropout 0.2, edge/feature drop 0.05 |

All regimes share AdamW (lr 1e-4), a per-step cosine schedule over the whole run
(`scheduler=cosine_mil_step`, `T_max: auto`), inverse-frequency class weights from
the training regions, gradient accumulation over 12 regions, gradient clipping
at 1, mixed precision, a fixed 1024-root validation subset per region and
all-cells testing. Every value is a Hydra key, e.g. `regime=mil
task.optimization.encoder_lr_scale=0.3 data.sampler.runtime.subsample_count=128
trainer.max_epochs=40 model.encoder.num_layers=3 data.sampler.runtime.depth=3`.

## Patient-level scores

`oof_patient_predictions.csv` averages the region probabilities (the preprint
default). The merged-bag alternative, where all cells of a patient's regions
form one bag with a joint attention softmax, is available at prediction time:

```bash
grass-mil-predict ckpt_path=<refit>/checkpoints/last.ckpt regime=linear_ftall_reg data=... \
  aggregation.mode=attention_weighted aggregation.bag_scope=patient \
  aggregation.patient_map.label_file=labels.csv aggregation.patient_map.id_column=REGION_ID \
  aggregation.patient_map.patient_column=PATIENT
```

`aggregation.mode=mean_probability aggregation.bag_scope=patient` reproduces the
mean-of-regions score from the same run.
