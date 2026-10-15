# Changelog

## 1.1.0 (2026-10)

A package-level cleanup. Commands, configuration keys and file formats changed;
graphs and checkpoints of 1.0.0 need to be rebuilt.

- **Regimes renamed**; `finetune` is the default: `linear_ftall` → `finetune`,
  `linear_ftall_reg` → `finetune_reg`, `linear_probehead` → `attention_only`,
  `mil` → `mlp_head`, `mil_frozen` → `mlp_head_frozen`, `pretrain_a3` →
  `pretrain`. The probe head is fitted automatically; `grass-mil-probe` is gone.
- **Niche analysis** (`grass-mil-niches`) reports the primary results directly:
  niches in the attention space, composition, association, margin with
  intervals, per-niche Moran's I, neighbourhood enrichment, cross-L and contact
  enrichment, and optional survival. `grass-mil-metaclusters` is optional.
- **Removed**: `grass-mil-report` and its plugin stack, `grass-mil-eval`,
  `grass-mil-loocv`, `grass-mil-scores`, mean pooling, the GCN/GAT/GraphSAGE/GIN
  encoders, per-sample PCA, `finetune.freeze_head`, the 1.0.0 checkpoint-import
  options and unused logger, debug and multi-GPU presets.
- **Configuration**: one `data.input` block for all formats, `data.unit`
  (`region`, `tile`, `polygon`), labels attached at load time (one set of
  graphs for every task and fold), `task` (`classification`, `regression`,
  `survival`) with its own selection metric.
- **Fixes**: gradient clipping under mixed precision now acts on unscaled
  gradients; cross-validation refits no longer evaluate on the held-out fold;
  `grass-mil-predict` reports AUROC and works without `regime=`; the PCA
  feature reducer works; survival training optimises over several units per
  step; numeric sample ids; stale processed graphs are detected.
- **Statistics**: the Cox loss handles tied times (Breslow) and survival heads
  start from a Cox probe; outer folds must keep patients together; refits keep
  the inner runs' learning-rate schedule; margins are relative to the model's
  average cell rather than the head bias, and their intervals resample patients;
  per-niche cross-L is relative to random labelling and Moran's I is tested
  with exact randomisation moments; prediction checks that the cohort was
  preprocessed like the training data.

## 1.0.0 (2026-10)

Public release accompanying the preprint.
