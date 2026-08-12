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
