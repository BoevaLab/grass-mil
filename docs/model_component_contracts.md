# Model Component Contracts

This document defines the canonical contracts for model components, losses, and samplers.
These contracts are mirrored in `.cursor/rules/model_component_contracts.mdc`.

## Goals

- Keep interfaces minimal and composable.
- Keep behavior deterministic under fixed seeds.
- Avoid hidden mutable global state.
- Keep modules data-layer agnostic.

## Graph Unit Contract

A graph unit is any `torch_geometric.data.Data` object emitted by the data layer.
It may represent a patch, a region-level graph, or a full-sample graph.

Required fields:

- `x`: `torch.Tensor` with shape `[num_nodes, num_node_features]`, float dtype.
- `edge_index`: `torch.LongTensor` with shape `[2, num_edges]`.

Optional fields:

- `edge_attr`: `torch.Tensor` with shape `[num_edges, num_edge_features]`.
- `batch`: `torch.LongTensor` with shape `[num_nodes]` for batched data.
- `pos`: `torch.Tensor` with shape `[num_nodes, 2]` or `[num_nodes, d]`.
- `graph_y`: `torch.Tensor` with shape `[1, num_tasks]` or `[batch_size, num_tasks]`.
- `graph_w`: `torch.Tensor` with same shape as `graph_y` for weighted objectives.
- `categorical_codes`: `torch.LongTensor` for categorical annotations.

Non-requirements:

- Samplers must not assume graph units are patches.
- Model components must not depend on `sample_id`, `region_id`, or `patch_id`.

## Encoder Contract

Encoders accept:

- `x`, `edge_index`, optional `edge_attr`, optional `batch`.

Encoders return:

- Node embeddings: shape `[num_nodes, hidden_dim_or_projected_dim]`.
- Optional graph embedding via pooling: shape `[num_graphs, pooled_dim]`.

Supported jump-knowledge modes:

- `last`, `concat`, `max`, `sum`.

## Attention Contract

Attention modules accept node/subgraph embeddings:

- Input shape `[n_items, emb_dim]`.

Attention modules return:

- Unnormalized attention logits: shape `[n_items, n_classes]` (default `n_classes=1`).
- Pass-through embeddings (for compatibility): shape `[n_items, emb_dim]`.

Normalization (softmax/sigmoid) is handled by caller to keep modules reusable.

## Loss Contracts

Categorical losses:

- Predictions: logits with shape `[N, C]` or `[N, 1]` for binary.
- Targets: class ids `[N]` or binary labels `[N, 1]`.
- Optional weights:
  - class weights for CE-style loss.
  - per-sample weights shape `[N]` or `[N, 1]`.

Regression losses:

- Predictions and targets shape-compatible (`[N, D]` or `[N]`).
- Optional per-sample weights shape `[N]` or broadcast-compatible.

Survival loss (`CoxSGD`):

- `y_pred`: shape `[N]` or `[N, 1]`.
- `length`: observed times shape `[N]` or `[N, 1]`.
- `event`: event indicator shape `[N]` or `[N, 1]`, expected binary values.

## Sampler Contracts

Sampler strategies operate on graph units produced by the data layer.

Identity strategy:

- Batches graph units without neighborhood expansion.

Neighborhood strategies:

- Operate per graph unit.
- Must preserve graph-unit metadata unless explicitly documented otherwise.
- Must support optional post-hoc transform hooks.

Native-first policy:

- Prefer native PyG samplers when contracts can be satisfied.
- Use custom sampler only when required behavior is unavailable in native path.

## Anti-Patterns

- No script-specific hardcoding in components.
- No mutable global constants that alter runtime behavior.
- No implicit side effects in constructors.
- No reliance on deprecated APIs when modern equivalent exists.

## Config-First Constants

- Keep hyperparameters and architecture constants in configs with explicit names.
- Avoid undocumented inline numeric defaults in implementation modules.
- Structural literals are acceptable only for clear invariants (for example, rank checks).

## Encoder Contract (updated)

- `forward(x, edge_index, edge_attr=None, batch=None, categorical_codes=None,
  return_graph_embedding=False)`.
- `categorical_codes` is `[num_nodes, num_labels]` long. The encoder embeds one
  configured column and **sums** it with the projection of `x`; it is not
  concatenated, so the embedding width is `hidden_dim`.
- `input_dim == 0` is legal and means the cohort has no continuous node
  features. A categorical embedding is then required.
- Out-of-range categorical codes raise. They are never clamped: clamping trains
  on a wrong cell type silently.

## Attention Contract

- Attention returns `[n_items, n_classes]`. Callers apply softmax over
  `dim=0` (instances) **per class column**.
- Only two widths are legal: 1 (one shared channel) or one per class. Enforced
  by `grass_mil.contracts.validate_attention_width`, at build time and at use.
- `head_spaces()` exposes intermediate activations (`a_pre_tanh`,
  `gate_product`) so analysis code does not reach into module internals.

## Attribution Contract

The bag logit is an attention-weighted sum of instance logits with no added
bias, so for every bag and class:

    sum_i A[i,c] * l[i,c] == L_c

This is an invariant, not an approximation. Any change to bag pooling must
preserve it, and `cluster_attribution_summary` reports `identity_residual` so a
violation surfaces rather than propagating into the cluster summaries.
