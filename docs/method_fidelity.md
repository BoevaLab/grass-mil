# Method Fidelity

Where each piece of the GRASS-MIL method lives in this package, and the test
that holds it in place. This is the map to check before trusting a number.

**The run code is the reference, not the manuscript.** Where `methods.tex` and
the final NSCLC/LUAD run code disagree, the code wins; the known divergences are
listed at the bottom.

## Core method

| Component | Definition | Implementation | Test |
|---|---|---|---|
| Cellular graph | Voronoi/Delaunay adjacency, edge distance + neighbour flag | `data/components/graph_builders.py` | `tests/data/` |
| Node inputs | `h⁰ = E[t] + Wₓx` — cell-type embedding **summed** with a projection of continuous features | `models/components/embeddings.py::NodeInputEmbedding` | `test_node_embeddings.py::test_node_embedding_sums_rather_than_concatenates` |
| Cell-type-only cohorts | `x` has zero columns; the embedding is the whole input | same | `test_node_embedding_supports_cell_type_only_cohorts`, `test_encoder_trains_on_cell_type_only_cohort` |
| Message passing | GINE conditioned on the **scalar edge length** only | `models/components/backbones.py::_select_edge_features` | `test_gine_reads_only_the_selected_edge_column` |
| Ego-graph extent | `r_k = radius_per_hop·k + radius_offset` | `data/components/ego_radius.py` | `test_seed_sampling.py::test_resolve_ego_radius_is_linear_in_depth` |
| Interior seeding | exclude the `n_hops` rim around the convex hull | `data/components/seed_sampling.py::convex_hull_interior_mask` | `test_interior_mask_excludes_the_hull_rim` |
| MIL attention | gated attention, softmax over instances **per class** | `models/components/attention.py`, `models/training/bagging.py` | `test_bagging.py::test_attention_is_normalised_per_class_over_instances` |
| Bag logit | `L_c = Σᵢ A[i,c]·ℓ[i,c]` | `models/training/bagging.py::aggregate_bag_logits_attention` | `test_bag_logit_is_the_attention_weighted_sum_of_instance_logits` |
| Objective | region cross-entropy, **sole** loss | `models/supervised_module.py::_compute_losses` | `tests/test_train_regimes.py` |
| Survival | Cox partial likelihood on a scalar log-hazard | `models/components/losses.py::CoxSGDLoss` | `test_survival_regime_fast_dev_run` |
| SSL pretraining | BGRL, symmetric cosine loss, EMA target | `models/bgrl_module.py` | `tests/test_train_regimes.py` |
| SSL views | degree-importance node/edge drop + Gaussian noise on cell size | `models/training/augmentations.py` | `tests/models/test_augmentations.py` |

## Attribution

The interpretability claim rests on one identity. If `identity_residual` is not
~0, every cluster-level attribution derived from it is invalid.

| Quantity | Definition | Implementation | Test |
|---|---|---|---|
| Instance contribution | `M[i,c] = A[i,c]·(ℓ[i,c] − βc)` — bias-free attention-weighted logits | `interpretability/core/attribution.py` | `test_bias_removal_shifts_the_margin_by_exactly_the_bias_term` |
| Margin (any C) | one-vs-rest: `m[i,c] = M[i,c] − mean(M[i,c'≠c])`; reduces to `M[i,1] − M[i,0]` when C=2 | `instance_ovr_margins` | `test_ovr_margin_reduces_to_the_binary_margin_when_c_is_two`, `test_ovr_margin_contrasts_each_class_against_the_mean_of_the_rest` |
| Additive identity | `Σᵢ M[i,c] = L_c` | same, `identity_residual` | `test_contributions_sum_exactly_to_the_bag_logit` |
| Attention lift | `λ = (Σ A) / prevalence`, 1 = neutral | same | `test_attention_lift_is_neutral_when_attention_matches_abundance` |
| Uncertainty | 200-resample percentile bootstrap **over regions** | `percentile_bootstrap_ci` | `test_bootstrap_ci_brackets_the_point_estimate` |

## Spatial statistics

| Statistic | Implementation | Test |
|---|---|---|
| Moran's I + permutation null + BH-FDR | `interpretability/tier2/autocorrelation.py` | `test_spatial_statistics.py::test_morans_i_*` |
| Ripley cross-L, centred `L(r) − r` | `interpretability/tier2/ripley.py` | `test_ripley_l_is_near_zero_for_a_poisson_pattern` |
| Neighbourhood enrichment | `interpretability/tier2/neighborhood.py` | `test_plugins_and_tier2.py` |
| Cross-space agreement (ARI/AMI/NMI) | `interpretability/core/agreement.py` | `test_cluster_agreement_*` |

## Known divergences from `methods.tex`

The manuscript is stale on these points. Each is deliberate.

1. **Edge features.** `methods.tex` describes a learned edge-**type** embedding
   propagated through the conv, with the distance channel supplied but unread.
   The implemented models do the opposite: GINE on the scalar edge **length**,
   no edge-type notion. *The manuscript needs updating.*
2. **Degree normalisation.** `methods.tex` says min-max; the run code uses
   mean-max (`(v − mean)/(max − mean)`, clamped at 0), which maps every
   below-average-degree node to zero importance. Implemented as in the code.
3. **Signed margin share bounds.** `methods.tex` states this is bounded in
   `[-1, 1]`. It is not: it is a fraction of the bag margin, so one cluster can
   exceed 1 when another opposes it. The real invariant is that a full
   partition sums to 1.
4. **Moran's I scope.** The legacy report computes Moran's I over *cells* inside
   pooled ego-graphs. Here the spatial table is root-instance level, so this
   measures autocorrelation over the *instance* graph — a defensible and
   arguably cleaner object, but not numerically the same statistic.

## Not implemented

Claimed in `methods.tex` but absent here, and absent from the legacy code too.
Implement or remove before submission:

- Archetypal analysis and the three simplex tests that depend on it.
- The 10×10 ensemble, ICC(2,1)/ICC(2,k), and the two-way ANOVA variance
  decomposition. All training uses a single seed; there is no ensemble harness.
- Graphlet/motif analysis and its permutation null.
- Paired bootstrap of the cross-L condition difference (the contrast is two
  separately aggregated mean curves).
