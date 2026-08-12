# Interpretability Suite

This document describes the interpretability subsystem in `grass-mil`.

Scope:

- architecture and design principles
- normalized analysis inputs
- reduction/clustering APIs and config controls
- plugin execution model (core + tier-2)
- report generation (interactive HTML + static PDF)
- operational runbook and troubleshooting

## 1) Purpose And Design Principles

The interpretability suite is designed as a first-class system, not a notebook helper.

Key principles:

1. Extensibility first.
2. Parameterized algorithms (no hardcoded DR/clustering assumptions).
3. Reproducible artifacts and reports.
4. Separation of concerns:
- analysis numerics
- plugin orchestration
- report rendering
5. Stable Python API for both core and tier-2 analyses.

Package roots:

- `src/grass_mil/interpretability/`
- `configs/interpretability/`
- `tests/interpretability/`

## 2) High-Level Architecture

Core layers:

1. Data contracts and loaders
- `src/grass_mil/interpretability/contracts.py`
- `src/grass_mil/interpretability/core/data.py`

2. Core numerical analysis
- `src/grass_mil/interpretability/core/reduction.py`
- `src/grass_mil/interpretability/core/clustering.py`
- `src/grass_mil/interpretability/core/biomarkers.py`

3. Tier-2 analyses
- `src/grass_mil/interpretability/tier2/neighborhood.py`
- `src/grass_mil/interpretability/tier2/filtration.py`

4. Plugin orchestration
- `src/grass_mil/interpretability/plugins/base.py`
- `src/grass_mil/interpretability/plugins/registry.py`
- `src/grass_mil/interpretability/plugins/builtin.py`
- `src/grass_mil/interpretability/pipeline.py`

5. Reporting
- `src/grass_mil/interpretability/reporting/plotly_builders.py`
- `src/grass_mil/interpretability/reporting/sections.py`
- `src/grass_mil/interpretability/reporting/html.py`
- `src/grass_mil/interpretability/reporting/snapshot.py`
- `src/grass_mil/interpretability/reporting/pdf.py`
- `src/grass_mil/interpretability/reporting/render.py`

6. CLI
- `src/grass_mil/interpretability/report_cli.py`

## 3) Normalized Input Contracts

The suite consumes normalized tables, independent of model internals.

### Required table: `instance_table` (`.csv` or `.parquet`)

Required minimal columns:

- `instance_id`
- `bag_id`
- embedding columns with one of supported prefixes:
  - `inst_emb_*`
  - `emb_*`
  - `graph_emb_*`

Common optional columns:

- `region_id`
- `sample_id`
- `cell_type`
- `score`
- `attention`
- `hazard`
- composition columns `comp_*`

`score` convention:

- exported inference tables use configurable score projection from `instance_logits`:
  - `score_mode=sigmoid` (default): `score = sigmoid(logit[score_logit_index])`
  - `score_mode=identity`: `score = logit[score_logit_index]`
- `score_mode=softmax`: `score = softmax(logits)[score_logit_index]`
- defaults (`sigmoid`, index `0`) keep notebook-era binary classification parity.
- for survival/hazard-style interpretation, prefer `score_mode=identity`.
- `instance_attention_logits` export requires shape `(N, 1)`; multi-column attention logits are
  rejected as ambiguous.

Note: `cluster_profiles` and `attention_attribution` require `comp_*` columns.
There is no fallback to one-hot `cell_type` composition.

`variance_estimator` convention (biomarker z-scoring):

- `unbiased` (default): sample variance/std (`ddof=1`)
- `biased`: population variance/std (`ddof=0`, notebook-style scaling)

### Optional table: `bag_table`

Used for downstream joins and metadata context. Not required by default pipeline math.

### Optional table: `spatial_table`

Required for tier-2 spatial plugins.

Required columns for current tier-2 methods:

- `source_id`
- `target_id`
- `distance` (required by filtration curves)
- `weight` (optional; currently ignored by neighborhood enrichment, which is unweighted/count-based)

Unit contract:

- During preprocessing, spatial coordinates are converted to micrometers (`um`) via `data.coord_scale_um`.
- `data.coord_scale_um=1.0` means input coordinates are already in `um`.
- `spatial_table.distance` is treated as micrometers by tier-2 analyses.
- If `data.coord_scale_um` changes, regenerate processed artifacts (`data.force_precompute=true` or a new `data.processed_dir`) before export/report.

## 4) Python API

Public exports:

- `run_reduction(...)`
- `run_clustering(...)`
- `cluster_biomarker_summary(...)`
- `cluster_attention_summary(...)`
- `cluster_survival_attention_summary(...)`
- `run_neighborhood_enrichment(...)`
- `run_diff_neighborhood_enrichment(...)`
- `compute_filtration_curves(...)`
- `prepare_tissue_graph_view(...)`
- `build_tissue_graph_figure(...)`
- `fit_cluster_transfer_from_report_bundle(...)`
- `apply_cluster_transfer(...)`
- `save_cluster_transfer_bundle(...)`
- `load_cluster_transfer_bundle(...)`
- `run_interpretability_pipeline(...)`
- `render_interpretability_report(...)`

Import path:

```python
from src.interpretability import (
    run_reduction,
    run_clustering,
    prepare_tissue_graph_view,
    build_tissue_graph_figure,
    fit_cluster_transfer_from_report_bundle,
    apply_cluster_transfer,
    run_interpretability_pipeline,
    render_interpretability_report,
)
```

Biomarker summary API:

- `cluster_biomarker_summary`, `cluster_attention_summary`, and
  `cluster_survival_attention_summary` operate on explicit composition columns (`comp_*`).
- `cell_type_column` is not part of these helper signatures.

### Neighborhood Enrichment Behavior

`run_neighborhood_enrichment(...)`:

- Uses unweighted edge counts (edge `weight` is ignored).
- `undirected=true` mirrors each edge (`u->v` and `v->u`).
- In `undirected=true`, unordered node pairs are deduplicated before mirroring so input tables
  that already contain both directions are not double-counted.
- `enrichment_mode` controls the returned enrichment matrix:
  - `zscore`: `(observed - expected) / std`
  - `obs-exp`: `observed - expected`
  - `log2fc`: `log2(observed / expected)` with epsilon clipping for stability
- `n_perms > 0`:
  - expected and std are estimated from label permutations
  - enrichment uses the selected `enrichment_mode`
  - p-values are two-sided empirical permutation p-values
- `n_perms = 0`:
  - uses analytical expected/std approximation
  - emits a warning that analytical mode is active

`run_diff_neighborhood_enrichment(...)`:

- Computes pairwise condition differences:
  `metric(cond_a) - metric(cond_b)` where `metric` is selected by `enrichment_mode`.
- Per-condition baselines are analytical.
- If `n_perms > 0`, only the differential p-values are permutation-based.
- Differential permutations are performed at the library/sample level via `permutation_group_column` (defaults to `sample_id`), matching notebook behavior.
- Each permutation group id must map to exactly one condition.
- Emits a warning that analytical per-condition baselines are used.

## 5) Reduction And Clustering APIs

### `run_reduction(method, X, **params)`

Supported methods:

- `pca`
- `umap`
- `tsne`

Common parameters:

- `n_components`
- `random_state`
- method-specific options (`n_neighbors`, `min_dist`, `metric`, `perplexity`, etc.)

Returns a `ReductionResult` dataclass with:

- reduced embedding matrix
- resolved parameters
- fitted object

### `run_clustering(method, X, **params)`

Supported methods:

- `kmeans`
- `agglomerative`
- `hdbscan`
- `spectral`
- `optics`
- `dbscan`

Common parameters:

- `n_clusters` (where required)
- `random_state`
- method-specific options (`min_cluster_size`, `metric`, `linkage`, `eps`, etc.)

Returns a `ClusteringResult` dataclass with:

- cluster labels
- resolved parameters
- fitted object

## 6) Plugin Execution Model

The pipeline runs plugin list from config after reduction/clustering.

Plugin protocol (`InterpretabilityPlugin`):

- `name`
- `required_inputs()`
- `run(dataset, context, **params) -> PluginResult`

`required_inputs()` is enforced by the pipeline before plugin execution; missing inputs
fail fast with a `ValueError` naming the plugin and missing keys.

Plugin registration is per pipeline run (fresh `PluginRegistry`).

`PluginResult` includes:

- `payload`: machine-readable outputs
- `sections`: report sections (tables/metadata, optional visuals)

Built-in plugins:

- `cluster_profiles`
- `attention_attribution`
- `neighborhood_enrichment`
- `diff_neighborhood_enrichment`
- `filtration_curves`
- `tissue_graph`

`filtration_curves` default threshold grid:

- notebook-parity default is `np.linspace(0.0, 55.0, 500)`
- thresholds are interpreted in micrometers

`tissue_graph` behavior:

- disabled by default in `plugins.enabled`
- requires explicit `sample_value` selection (no auto-pick)
- requires `center_x` and `center_y` coordinates in `instance_table`
- colors nodes by pipeline `cluster_labels` context
- uses `spatial_table` edges filtered to selected sample nodes

## 7) Pipeline Orchestration

`run_interpretability_pipeline(...)` is the single coordinator.

Workflow:

1. Load embedding matrix from `instance_table`.
2. Run configured reduction.
3. Run configured clustering:
- `cluster_on_pca=true`: cluster on PCA projection used for clustering.
- `cluster_on_pca=false`: cluster on raw embedding columns.
4. Build core cluster summaries.
5. Execute enabled plugins in order.
6. Persist canonical artifacts:
- `artifacts/instance_with_clusters.csv`
- `artifacts/reduction.csv` (if enabled)
- `artifacts/cluster_labels.csv` (if enabled)
- `artifacts/pipeline_summary.json`
7. Return `ReportBundle`.

`ReportBundle` includes `cluster_feature_reduction`:

- populated with clustering PCA result when `cluster_on_pca=true`
- `None` when clustering runs on raw embeddings

## 8) Cluster Transfer Workflow (Library API)

Notebook-parity cluster transfer is available as a library workflow:

1. Fit transfer bundle on a source interpretability run (cluster labels + clustering feature space).
2. Persist bundle (`.joblib`) for deterministic reuse.
3. Apply bundle to a query `instance_table` to obtain transferred labels.

Design notes:

- Uses kNN over the exact clustering feature space used in the source run.
- If source clustering used PCA (`cluster_on_pca=true`), transfer uses that same PCA projector.
- HDBSCAN noise label `-1` is retained as a transferable class.
- This replaces fragile notebook assumptions like `clusterer._embedding_`.

Example:

```python
from pathlib import Path

import pandas as pd

from grass_mil.interpretability.core.data import load_interpretability_dataset
from grass_mil.interpretability.core.transfer import (
    apply_cluster_transfer,
    fit_cluster_transfer_from_report_bundle,
    load_cluster_transfer_bundle,
    save_cluster_transfer_bundle,
)
from grass_mil.interpretability.pipeline import run_interpretability_pipeline

dataset = load_interpretability_dataset(instance_table_path=Path("/abs/source_instance_table.csv"))
bundle = run_interpretability_pipeline(
    dataset,
    {
        "reduction": {"enabled": True, "method": "pca", "params": {"n_components": 2}},
        "clustering": {
            "enabled": True,
            "method": "hdbscan",
            "cluster_on_pca": True,
            "pca_components": 10,
            "params": {"min_cluster_size": 100, "min_samples": 1},
        },
        "plugins": {"enabled": [], "params": {}},
    },
    artifacts_dir=Path("/abs/source_artifacts"),
)
transfer = fit_cluster_transfer_from_report_bundle(dataset, bundle, n_neighbors=15)
save_cluster_transfer_bundle(transfer, "/abs/cluster_transfer_bundle.joblib")

loaded = load_cluster_transfer_bundle("/abs/cluster_transfer_bundle.joblib")
query_table = pd.read_csv("/abs/query_instance_table.csv")
transferred = apply_cluster_transfer(loaded, query_table)
```

Notebook helper example for single tissue graph:

```python
from grass_mil.interpretability.tier2.tissue_graph import (
    build_tissue_graph_figure,
    prepare_tissue_graph_view,
)

view = prepare_tissue_graph_view(
    node_table=instance_with_clusters_df,
    spatial_table=spatial_table_df,
    sample_column="sample_id",
    sample_value="sx",
    id_column="instance_id",
    x_column="center_x",
    y_column="center_y",
    label_column="cluster_label",
)
fig = build_tissue_graph_figure(view, show_edges=True)
fig.show()
```

## 9) Reporting System

`render_interpretability_report(...)` generates:

1. Interactive report: `report.html`
2. Static snapshots: `artifacts/snapshots/*.png` (Plotly + kaleido)
3. Static PDF: `report.pdf` (Playwright print pipeline)

### HTML report

- Plotly interactive sections.
- Tabular summaries from core and plugin outputs.
- Structured section breakdown with metadata.

### PDF report

Design intent:

- publication-stable, deterministic layout
- static figure rendering
- explicit print CSS and margins

Engine:

- Playwright Chromium print-to-PDF.

## 10) CLI Usage

Generate interpretability-ready tables directly from inference:

```bash
grass-mil-predict \
  ckpt_path=/abs/path/model.ckpt \
  data=spatial_omics \
  interpretability.enabled=true \
  interpretability.spatial.enabled=true
```

This writes `instance_table.csv` (and optionally `spatial_table.csv`) into
`${paths.output_dir}/${predict.output_subdir}` and records paths in
`inference_summary.json` under `interpretability.*`.

Entrypoint:

```bash
grass-mil-report \
  data.instance_table=/abs/path/instance_table.csv
```

With spatial plugins:

```bash
grass-mil-report \
  data.instance_table=/abs/path/instance_table.csv \
  data.spatial_table=/abs/path/spatial_table.csv \
  plugins.enabled="[cluster_profiles,attention_attribution,neighborhood_enrichment,diff_neighborhood_enrichment,filtration_curves]"
```

Render a single selected tissue graph colored by cluster labels:

```bash
grass-mil-report \
  data.instance_table=/abs/path/instance_table.csv \
  data.spatial_table=/abs/path/spatial_table.csv \
  plugins.enabled="[tissue_graph]" \
  plugins.params.tissue_graph.sample_column=sample_id \
  plugins.params.tissue_graph.sample_value=sx
```

Override algorithms:

```bash
grass-mil-report \
  data.instance_table=/abs/path/instance_table.csv \
  reduction=umap \
  clustering=dbscan \
  clustering.params.eps=0.25 \
  clustering.params.min_samples=10
```

Tune HDBSCAN selection behavior explicitly:

```bash
grass-mil-report \
  data.instance_table=/abs/path/instance_table.csv \
  clustering=hdbscan \
  clustering.params.min_cluster_size=100 \
  clustering.params.min_samples=5 \
  clustering.params.cluster_selection_method=leaf \
  clustering.params.cluster_selection_epsilon=0.0
```

Enable/disable report outputs:

```bash
grass-mil-report \
  data.instance_table=/abs/path/instance_table.csv \
  report.pdf.enabled=true \
  report.html.enabled=true
```

## 11) Config Groups

Main config:

- `configs/interpretability/report.yaml`

Subgroups:

- `configs/interpretability/reduction/*.yaml`
- `configs/interpretability/clustering/*.yaml`
- `configs/interpretability/plugins/default.yaml`

Important knobs:

- `reduction.method`
- `reduction.params.*`
- `clustering.method`
- `clustering.params.*`
- `clustering.cluster_on_pca`
- `clustering.pca_components`
- `plugins.enabled`
- `plugins.params.<plugin_name>.*`
- `variance_estimator`
- `embedding_prefixes`
- `report.html.enabled`
- `report.pdf.enabled`
- `report.pdf.snapshot_dpi_scale`

Notebook-aligned defaults:

- `reduction=umap` with `n_neighbors=50`, `min_dist=0.1`, `random_state=null`
- `clustering=hdbscan` with `min_cluster_size=100`, `min_samples=1`, `cluster_selection_method=eom`
- `clustering.cluster_on_pca=true`, `clustering.pca_components=10`
- effect: clustering runs on PCA(10) while report scatter uses UMAP output
- note: PCA components are capped to `min(n_samples, n_features)` on small datasets

Parameter contract:

- `clustering.params.*` is forwarded directly to the underlying sklearn clustering estimator.
- No method-specific parameter filtering is applied beyond common `n_clusters`/`random_state` handling.

## 12) Environment Prerequisites

From project root:

```bash
pip install -r requirements.txt
```

For PDF export, install Playwright browser binaries:

```bash
playwright install chromium
```

If PDF export fails, check:

1. `playwright` package installed.
2. Chromium binaries installed.
3. sandbox constraints for headless browser runtime.

If snapshot export fails, check:

1. `kaleido` package installed.
2. Plotly-kaleido compatibility in your environment.

## 13) Typical Operational Flow

1. Generate/prepare normalized instance/spatial tables.
2. Run report CLI with selected reduction/clustering presets.
3. Inspect `report.html` interactively.
4. Share `report.pdf` for static dissemination.
5. Consume `artifacts/*.csv` and JSON for downstream analysis scripts.

## 14) Troubleshooting

### "No embedding columns found"

Cause:

- `instance_table` has no columns matching configured prefixes.

Fix:

- rename columns to `inst_emb_*` (or set `embedding_prefixes` override).

### "Spatial table is required for ... plugin"

Cause:

- tier-2 plugin enabled without `data.spatial_table`.

Fix:

- provide `spatial_table`, or remove plugin from `plugins.enabled`.

### "Playwright is unavailable"

Cause:

- Python package missing or browser binaries not installed.

Fix:

- `pip install playwright`
- `playwright install chromium`

### "Failed to export plot snapshot"

Cause:

- `kaleido` missing or incompatible runtime.

Fix:

- `pip install kaleido`

## 15) Relationship To Other Docs

- prediction/inference payload generation: `docs/inference_utility_suite.md`
- training/eval workflows: `docs/workflows_train_infer.md`
- extension instructions: `docs/interpretability_extension_guide.md`
