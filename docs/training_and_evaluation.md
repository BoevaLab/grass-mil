# Training and evaluation

## The model

Every cell is represented by its *k*-hop ego-graph in the cell graph
(`data.sampler.depth`, default 4). The encoder embeds each ego-graph: a learned
cell-type embedding summed with a projection of the continuous features, then
`model.encoder.num_layers` GINE layers that condition every message on the
edge length, and max pooling. A unit is a bag of the ego-graphs of
`data.sampler.roots_per_unit` cells (256 by default, redrawn every epoch).

For each instance `i`, the instance head gives one logit `l[i, c]` per class
and the projected gated attention gives one score per class; the attention is
a softmax over the bag's instances `A[i, c]`, and the bag logit is

```
L_c = sum_i A[i, c] * l[i, c]
```

so every unit prediction is an exact sum of per-cell contributions, which the
niche analysis uses.

## Pretraining

```bash
grass-mil-train --config-dir configs experiment=my_cohort regime=pretrain trainer=gpu \
  hydra.run.dir=runs/pretrain
```

The encoder first learns from the tissue alone, without labels, by BGRL (a
self-supervised method: two randomly perturbed copies of the same neighbourhood
should get similar embeddings). In detail: two augmented views
(importance-weighted node and edge dropping, Gaussian noise on
`pretrain.augmentation.noise_features`), an online encoder with a predictor and
an exponential-moving-average target encoder. Edges are dropped per direction
and kept only when both directions survive, so an edge with drop probability
`p` per direction disappears with probability `1 - (1 - p)^2`. Pretraining sees
every unit, including the ones cross-validation later holds out; it uses no
label. The run writes `checkpoints/last.ckpt` and a checkpoint every 5000
steps (`pretrain.total_steps`, default 200 000, about an hour on a recent GPU;
lower `pretrain.warmup_steps` with it for short runs).

## Fine-tuning regimes

A regime is the training procedure; select it with `regime=<name>`.

| Regime | Encoder | Instance head | Trained |
|---|---|---|---|
| `finetune` (default) | fine-tuned (lr x1) | linear, initialised by a probe | encoder, head (lr x0.01), attention (lr x2) |
| `finetune_reg` | fine-tuned (lr x0.3), dropout 0.2 | linear, probe | as `finetune`, with weight decay 1e-2, edge drop and node masking 0.05 |
| `attention_only` | frozen | linear, probe, frozen | attention only (512 roots per unit) |
| `mlp_head` | fine-tuned | two-layer MLP, random | everything |
| `mlp_head_frozen` | frozen | two-layer MLP, random | head and attention (512 roots per unit) |

The probe gives the instance head a sensible start: before training, the
frozen encoder embeds
`finetune.probe.roots_per_unit` cells of every training unit, each ego-graph
inherits its unit's label, and a standardised, L2-penalised logistic regression
(ridge for regression, a Cox model for survival; inverse penalty
`finetune.probe.C`) is folded into the linear head. The model then starts from
an attention-weighted average of the probe's per-cell logits. The probe is
fitted on standardised embeddings, so an encoder whose embeddings barely vary
(too little pretraining) gives large head weights. The log reports their norm:
a few units is typical, and above 20 the command warns, because fine-tuning the
encoder then tends to diverge (see [small cohorts](#small-cohorts)).

Every regime trains one bag (one unit) per step of the dataloader and takes an
optimizer step every `finetune.hyperbatch_size` units (the "hyperbatch", 12;
the last step of an epoch takes the remaining units), with gradient-norm clipping (`finetune.grad_clip_norm`),
a cosine learning-rate schedule over the whole run and AdamW (`optim.lr` 1e-4).
Class labels must be the integers `0..num_classes-1`. Validation units are scored on a fixed set of
`data.val_roots_per_unit` cells; the checkpoint with the best validation metric
is kept and training stops early after `task.selection.patience` epochs without
improvement (after `trainer.min_epochs`).

```bash
grass-mil-train --config-dir configs experiment=my_cohort \
  model.init_from_ckpt=runs/pretrain/checkpoints/last.ckpt hydra.run.dir=runs/train
```

A single run splits the units at random (`data.split.ratios`, grouped by
patient and stratified by class) unless `data.split.train/val/test` list ids,
and writes `test_predictions.csv` for the test units.

## Tasks

| `task=` | Targets | Loss | Selection |
|---|---|---|---|
| `classification` (default) | one class column, `task.num_classes` classes | cross-entropy, classes weighted by inverse training frequency | `val/auroc` |
| `regression` | one or more numeric columns | mean squared error (`task.loss=huber` for Huber) | `val/rmse` |
| `survival` | `[time, event]` | Cox partial likelihood over the units of each optimizer step | `val/c_index` |

## Nested cross-validation

```bash
grass-mil-nested-cv --config-dir configs experiment=my_cohort \
  model.init_from_ckpt=runs/pretrain/checkpoints/last.ckpt hydra.run.dir=runs/nested_cv
```

The folds file must keep every patient (or, without a patient column, every
region) in one fold; the command checks this before training. For every outer
fold:

1. the other folds form the development set, split into `nested_cv.inner_k`
   (3) inner folds that keep patients together and every class in every fold;
2. every inner run trains on its training part and records the epoch with the
   best `task.selection.metric` on its validation part;
3. the model is refitted on the whole development set with the same
   learning-rate schedule as the inner runs, stopped at the median of those
   epochs, without validation, and scored once on the held-out fold.

| Output | Content |
|---|---|
| `nested_cv_summary.json` | metric of every outer fold, mean and standard deviation, pooled out-of-fold metrics, patient-level metrics |
| `oof_predictions.csv` | one row per unit, scored by the model that held it out |
| `oof_patient_predictions.csv` | the mean of each patient's unit scores (when every unit of a patient has the same label) |
| `outer_<i>/refit/checkpoints/last.ckpt` | the model of outer fold `i` |

**The result to report** is the mean ± s.d. of the task's metric over the outer
folds (`mean` and `sd` in `nested_cv_summary.json`: AUROC for classification,
RMSE for regression, the concordance index for survival). The pooled metrics
rank units scored by different fold models, whose scores need not be on the
same scale, so they can be lower than every fold's own value.

Runs that finished are reused when the command is repeated
(`nested_cv.resume`); `nested_cv.outer_folds=[2]` runs a single fold.

## Small cohorts

The defaults suit cohorts of hundreds of units. With a few dozen:

* **Freeze the encoder**: `regime=attention_only`. Fine-tuning a million
  encoder weights on a few dozen units is unstable (validation loss jumps after
  the first epoch and predictions saturate at 0 or 1); a frozen encoder with a
  trained attention gives stable, calibrated predictions.
* **Penalise the probe more**: `finetune.probe.C=0.01`, and fewer units per
  step, `finetune.hyperbatch_size=4`.
* **Select epochs on the loss**: `task.selection.metric=val/loss
  task.selection.mode=min`. On a handful of validation units the AUROC often
  ties at 1.0 from the first epoch, and ties pick the earliest epoch.
* **Pretrain long enough**: too short a pretraining leaves embeddings that
  barely vary (the probe's weight norm warns about it); the vignettes use 3,000
  steps on 44 tiles.
* **Use fewer folds** than the smaller class has patients, and see the
  [niche analysis](niche_analysis.md#small-cohorts) for clustering settings.

## Prediction

```bash
grass-mil-predict --config-dir configs experiment=new_cohort \
  ckpt_path=runs/nested_cv/outer_0/refit/checkpoints/last.ckpt hydra.run.dir=runs/predict
```

The model and its ego-graph settings are rebuilt from the checkpoint; `data`
only needs to describe the cohort. Its cell types, features, edge features,
graph settings and feature basis must be the model's (for PCA features, reuse
the basis with `data.feature_reducer.state_from`); the command refuses other
inputs. Every unit is scored on all the cells training drew roots from, as
for the test split (`predictions.csv`). With a patient column,
patients are scored as the mean of their units (`predict.patient_aggregation=mean`)
or as one bag holding every cell of their units (`merged_bag`). With labels,
`metrics.json` reports unit- and patient-level metrics.
