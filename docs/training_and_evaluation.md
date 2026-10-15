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
grass-mil-train experiment=my_cohort regime=pretrain trainer=gpu
```

BGRL on the ego-graphs of every unit, label-free: two augmented views
(importance-weighted node and edge dropping, Gaussian noise on
`pretrain.augmentation.noise_features`), an online encoder with a predictor and
an exponential-moving-average target encoder. The run writes
`checkpoints/last.ckpt` and a checkpoint every 5000 steps
(`pretrain.total_steps`, default 200 000).

## Fine-tuning regimes

A regime is the training procedure; select it with `regime=<name>`.

| Regime | Encoder | Instance head | Trained |
|---|---|---|---|
| `finetune` (default) | fine-tuned (lr x1) | linear, initialised by a probe | encoder, head (lr x0.01), attention (lr x2) |
| `finetune_reg` | fine-tuned (lr x0.3), dropout 0.2 | linear, probe | as `finetune`, with weight decay 1e-2, edge drop and node masking 0.05 |
| `attention_only` | frozen | linear, probe, frozen | attention only (512 roots per unit) |
| `mlp_head` | fine-tuned | two-layer MLP, random | everything |
| `mlp_head_frozen` | frozen | two-layer MLP, random | head and attention (512 roots per unit) |

The probe: before training, the frozen encoder embeds
`finetune.probe.roots_per_unit` cells of every training unit, each ego-graph
inherits its unit's label, and a standardised logistic regression (ridge for
regression) is folded into the linear head. With uniform attention, the model
then starts from the mean-pooled probe. Survival models start from a random
head.

Every regime trains one bag per step of the dataloader and takes an optimizer
step every `finetune.hyperbatch_size` units (12), with gradient-norm clipping
(`finetune.grad_clip_norm`), a cosine learning-rate schedule over the whole run
and AdamW (`optim.lr` 1e-4). Validation units are scored on a fixed set of
`data.val_roots_per_unit` cells; the checkpoint with the best validation metric
is kept and training stops early after `task.selection.patience` epochs without
improvement (after `trainer.min_epochs`).

```bash
grass-mil-train experiment=my_cohort model.init_from_ckpt=<pretrain>/checkpoints/last.ckpt
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
grass-mil-nested-cv experiment=my_cohort model.init_from_ckpt=<pretrain>/checkpoints/last.ckpt
```

For every outer fold of `nested_cv.folds_file`:

1. the other folds form the development set, split into `nested_cv.inner_k`
   (3) inner folds that keep patients together and every class in every fold;
2. every inner run trains on its training part and records the epoch with the
   best `task.selection.metric` on its validation part;
3. the model is refitted on the whole development set for the median of those
   epochs, without validation, and scored once on the held-out fold, on every
   cell.

| Output | Content |
|---|---|
| `nested_cv_summary.json` | metric of every outer fold, mean and standard deviation, pooled out-of-fold metrics, patient-level metrics |
| `oof_predictions.csv` | one row per unit, scored by the model that held it out |
| `oof_patient_predictions.csv` | the mean of each patient's unit scores |
| `outer_<i>/refit/checkpoints/last.ckpt` | the model of outer fold `i` |

Runs that finished are reused when the command is repeated
(`nested_cv.resume`); `nested_cv.outer_folds=[2]` runs a single fold.

## Prediction

```bash
grass-mil-predict experiment=new_cohort ckpt_path=<run>/checkpoints/last.ckpt
```

The model and its ego-graph settings are rebuilt from the checkpoint; `data`
only needs to describe the cohort (its cell-type vocabulary must match). Every
unit is scored on all of its cells (`predictions.csv`). With a patient column,
patients are scored as the mean of their units (`predict.patient_aggregation=mean`)
or as one bag holding every cell of their units (`merged_bag`). With labels,
`metrics.json` reports unit- and patient-level metrics.
