# Published-Method Fidelity Status

This document used to assert that a list of "legacy risks" had been resolved by
rebuilding from concepts rather than porting code. That framing hid a real
problem: several load-bearing pieces of the published method were not
implemented here at all, and the audit could not have caught it, because it
checked engineering hygiene rather than fidelity to the method.

It is now a status record. For the authoritative mapping of method → code →
test, see [`method_fidelity.md`](method_fidelity.md).

## Gaps that were open and are now closed

1. **The encoder could not express the published input scheme.** There was no
   cell-type embedding, so cell identity — the only node input for some cohorts
   — had no path into the model. Closed by `NodeInputEmbedding`.
2. **Edge conditioning was wrong.** `gine` fed the whole edge-attribute vector
   into the convolution. The production models condition on the scalar edge
   length. Closed by `EncoderConfig.edge_feature_index`.
3. **Attention was single-channel.** Bag pooling squeezed the head to one
   channel broadcast across all class logits, so the per-class formulation —
   and therefore the exact attribution identity — could not be expressed.
   Closed; the identity is now asserted at `atol=1e-12`.
4. **Sampling controls were missing.** No ego-graph radius cutoff and no
   interior seeding, so ego-graphs rooted at a section edge were truncated by
   the boundary rather than by biology.
5. **SSL views were uniform.** Elementwise masking instead of the
   degree-importance drop plus cell-size noise.
6. **No attribution.** Attention lift existed; margins, the identity check and
   uncertainty intervals did not.
7. **No spatial statistics.** Moran's I, Ripley's cross-L and cross-space
   agreement had no implementation.

## Bugs found while closing them

These produced silently wrong results rather than failures, which is why they
survived:

1. **Sampling discarded cell type.** `_apply_transform_to_each_subgraph` dropped
   every node-length tensor as "already represented by `x`" — false for
   `categorical_index` and `pos`.
2. **Node codes were batched along the wrong axis.** PyTorch Geometric treats
   any attribute whose name contains `index` as an edge-index tensor, so
   `categorical_index` was concatenated along the feature dimension. Renamed to
   `categorical_codes`; graphs from older runs are migrated on load.
3. **Attention flattening in two directions.** `inference/aggregation.py`
   silently reshaped multi-column attention to 1-D while
   `interpretability_export.py` raised on it. One gave wrong numbers, the other
   crashed.
4. **Asymmetric edge dropping.** The finetune-time transform left
   `force_undirected=False` while pretraining used `True`, so the two regimes
   augmented differently and message passing became direction-dependent.
5. **Order-dependent Ripley radii.** The radius grid came from whichever region
   was iterated first.
6. **A concordance index that synchronised per pair.** O(n²) Python with two
   `.item()` calls per pair.

## Engineering invariants (the original content, still enforced)

- Learnable submodules live in `nn.ModuleList`/`nn.ModuleDict`.
- Attention shape contracts are explicit; normalisation is caller-controlled and
  validated by `grass_mil.contracts.validate_attention_width`.
- Samplers do not mutate datamodule state.
- Behaviour is config-driven, not hardcoded per project. The legacy
  `os.environ` control surface is deliberately not ported.
- Native PyG operators are preferred; adapters stay thin.
- Optional components instantiate only when their config is set.

## Deliberately retained

- Region accumulation. It is a throughput mechanism rather than an objective,
  and the final NSCLC runs use it (`hyperbatch_size: 16`), so it stays.
- The clustering algorithm's own names (`run_clustering`, `n_clusters`,
  `min_cluster_size`). Clustering is the method; niches are its result.
