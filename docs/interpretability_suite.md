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

- `src/interpretability/`
- `configs/interpretability/`
- `tests/interpretability/`

## 2) High-Level Architecture

Core layers:

1. Data contracts and loaders
- `src/interpretability/contracts.py`
- `src/interpretability/core/data.py`

2. Core numerical analysis
- `src/interpretability/core/reduction.py`
- `src/interpretability/core/clustering.py`
- `src/interpretability/core/biomarkers.py`

3. Tier-2 analyses
- `src/interpretability/tier2/neighborhood.py`
- `src/interpretability/tier2/filtration.py`

4. Plugin orchestration
- `src/interpretability/plugins/base.py`
- `src/interpretability/plugins/registry.py`
- `src/interpretability/plugins/builtin.py`
- `src/interpretability/pipeline.py`

5. Reporting
- `src/interpretability/reporting/plotly_builders.py`
- `src/interpretability/reporting/sections.py`
- `src/interpretability/reporting/html.py`
- `src/interpretability/reporting/snapshot.py`
- `src/interpretability/reporting/pdf.py`
- `src/interpretability/reporting/render.py`

6. CLI
- `src/interpretability/report_cli.py`

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

- exported inference tables use `score = sigmoid(logit_0)` to match notebook-era interpretability calculations.

Note: `cluster_profiles` and `attention_attribution` require `comp_*` columns.
There is no fallback to one-hot `cell_type` composition.

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

- During preprocessing, spatial coordinates/distances must be normalized to micrometers (`um`).
- `spatial_table.distance` is treated as micrometers by tier-2 analyses.

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
- `run_interpretability_pipeline(...)`
- `render_interpretability_report(...)`

Import path:

```python
from src.interpretability import (
    run_reduction,
    run_clustering,
    run_interpretability_pipeline,
    render_interpretability_report,
)
```

### Neighborhood Enrichment Behavior

`run_neighborhood_enrichment(...)`:

- Uses unweighted edge counts (edge `weight` is ignored).
- `undirected=true` mirrors each edge (`u->v` and `v->u`).
- `n_perms > 0`:
  - expected and std are estimated from label permutations
  - enrichment is z-score `(observed - expected) / std`
  - p-values are two-sided empirical permutation p-values
- `n_perms = 0`:
  - uses analytical expected/std approximation
  - emits a warning that analytical mode is active

`run_diff_neighborhood_enrichment(...)`:

- Computes pairwise condition differences: `zscore(cond_a) - zscore(cond_b)`.
- Per-condition baselines are analytical.
- If `n_perms > 0`, only the differential p-values are permutation-based (condition-label permutations).
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

`PluginResult` includes:

- `payload`: machine-readable outputs
- `sections`: report sections (tables/metadata, optional visuals)

Built-in plugins:

- `cluster_profiles`
- `attention_attribution`
- `neighborhood_enrichment`
- `filtration_curves`

`filtration_curves` default threshold grid:

- notebook-parity default is `np.linspace(0.0, 55.0, 500)`
- thresholds are interpreted in micrometers

## 7) Pipeline Orchestration

`run_interpretability_pipeline(...)` is the single coordinator.

Workflow:

1. Load embedding matrix from `instance_table`.
2. Run configured reduction.
3. Run configured clustering.
4. Build core cluster summaries.
5. Execute enabled plugins in order.
6. Persist canonical artifacts:
- `artifacts/instance_with_clusters.csv`
- `artifacts/reduction.csv` (if enabled)
- `artifacts/cluster_labels.csv` (if enabled)
- `artifacts/pipeline_summary.json`
7. Return `ReportBundle`.

## 8) Reporting System

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

## 9) CLI Usage

Generate interpretability-ready tables directly from inference:

```bash
python src/inference/predict.py \
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
python src/interpretability/report_cli.py \
  data.instance_table=/abs/path/instance_table.csv
```

With spatial plugins:

```bash
python src/interpretability/report_cli.py \
  data.instance_table=/abs/path/instance_table.csv \
  data.spatial_table=/abs/path/spatial_table.csv \
  plugins.enabled="[cluster_profiles,attention_attribution,neighborhood_enrichment,filtration_curves]"
```

Override algorithms:

```bash
python src/interpretability/report_cli.py \
  data.instance_table=/abs/path/instance_table.csv \
  reduction=umap \
  clustering=dbscan \
  clustering.params.eps=0.25 \
  clustering.params.min_samples=10
```

Enable/disable report outputs:

```bash
python src/interpretability/report_cli.py \
  data.instance_table=/abs/path/instance_table.csv \
  report.pdf.enabled=true \
  report.html.enabled=true
```

## 10) Config Groups

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
- `embedding_prefixes`
- `report.html.enabled`
- `report.pdf.enabled`
- `report.pdf.snapshot_dpi_scale`

Notebook-aligned defaults:

- `reduction=umap` with `n_neighbors=50`, `min_dist=0.1`, `random_state=null`
- `clustering=hdbscan` with `min_cluster_size=100`
- `clustering.cluster_on_pca=true`, `clustering.pca_components=10`
- effect: clustering runs on PCA(10) while report scatter uses UMAP output
- note: PCA components are capped to `min(n_samples, n_features)` on small datasets

## 11) Environment Prerequisites

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

## 12) Typical Operational Flow

1. Generate/prepare normalized instance/spatial tables.
2. Run report CLI with selected reduction/clustering presets.
3. Inspect `report.html` interactively.
4. Share `report.pdf` for static dissemination.
5. Consume `artifacts/*.csv` and JSON for downstream analysis scripts.

## 13) Troubleshooting

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

## 14) Relationship To Other Docs

- prediction/inference payload generation: `docs/inference_utility_suite.md`
- training/eval workflows: `docs/workflows_train_infer.md`
- extension instructions: `docs/interpretability_extension_guide.md`
