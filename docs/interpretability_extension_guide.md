# Interpretability Extension Guide

This guide describes the recommended workflow for adding new interpretability analyses.

Core rule:

- add new analysis logic in `core/` or `tier2/`
- expose it through a plugin
- enable it via config
- validate with unit + pipeline tests

The pipeline should not need structural edits for each new method.

## 1) Decide Where Logic Belongs

Use `core/` when:

- analysis is generally applicable and does not require spatial graph topology.

Use `tier2/` when:

- analysis uses spatial neighborhood/connectivity structures or higher-order post-hoc behavior.

Examples:

- spatial entropy: `tier2/`
- boundary rugosity: `tier2/`
- non-spatial embedding reliability score: `core/`

## 2) Implement Pure Analysis Functions

Create a module:

- `src/interpretability/core/<your_method>.py`
- or `src/interpretability/tier2/<your_method>.py`

Guidelines:

1. Keep function signatures explicit.
2. Validate required columns early and fail with clear errors.
3. Return typed outputs/dataclasses (or pandas frames) without side effects.
4. Avoid direct plotting in compute functions.

Example skeleton:

```python
from __future__ import annotations

import pandas as pd

def compute_spatial_entropy(
    node_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    *,
    label_column: str = "cluster_label",
) -> pd.DataFrame:
    if label_column not in node_table.columns:
        raise ValueError(f"Missing label column {label_column!r}.")
    # compute ...
    return pd.DataFrame(...)
```

## 3) Create A Plugin Wrapper

Add plugin class in:

- `src/interpretability/plugins/builtin.py`
- or a new plugin module imported at registration time.

Plugin responsibilities:

1. Read context state (e.g., cluster labels).
2. Call your pure analysis function.
3. Return `PluginResult` with:
- `payload` for machine consumption
- `sections` for report composition

Example skeleton:

```python
@dataclass
class SpatialEntropyPlugin(InterpretabilityPlugin):
    name: str = "spatial_entropy"

    def required_inputs(self) -> list[str]:
        return ["spatial_table", "cluster_labels"]

    def run(self, dataset, context: PluginContext, **params):
        table = dataset.instance_table.copy()
        table["cluster_label"] = context.state["cluster_labels"]
        entropy_df = compute_spatial_entropy(
            table,
            dataset.spatial_table,
            label_column=str(params.get("label_column", "cluster_label")),
        )
        return PluginResult(
            name=self.name,
            payload={"entropy": entropy_df},
            sections=[
                ReportSection(
                    title="Spatial Entropy",
                    description="Entropy by cluster.",
                    tables={"entropy": entropy_df},
                )
            ],
        )
```

`required_inputs()` entries are checked by `run_interpretability_pipeline(...)` before
`run(...)` is called. If any required input is missing from dataset fields/context state,
the pipeline raises `ValueError` immediately.

## 4) Register The Plugin

Register in `register_builtin_plugins()`:

```python
register_plugin(SpatialEntropyPlugin())
```

Or create a dedicated registration function and invoke it from pipeline startup.

## 5) Add Config Wiring

Update:

- `configs/interpretability/plugins/default.yaml`

Add:

1. plugin name to `enabled` list (if default-on behavior desired)
2. plugin parameter block under `params.<plugin_name>`

Example:

```yaml
enabled:
  - cluster_profiles
  - spatial_entropy
params:
  spatial_entropy:
    label_column: cluster_label
    radius: 50.0
```

## 6) Optional: Add Custom Plotly Visuals

If table-only output is insufficient:

1. Add figure builder in `src/interpretability/reporting/plotly_builders.py`.
2. Detect plugin payload and create a figure.
3. Ensure figure naming is stable for snapshot paths.

Guideline:

- keep plotting dependent on payload shape contracts, not ad hoc column discovery.

## 7) Add Tests (Required)

Minimum:

1. Unit test for compute function.
2. Plugin test validating payload/section structure.
3. Pipeline integration test with plugin enabled.

Suggested files:

- `tests/interpretability/test_<method>.py`
- extend `tests/interpretability/test_plugins_and_tier2.py`
- extend `tests/interpretability/test_pipeline_and_report.py`

## 8) Extension Checklist

Before merging, ensure:

1. Compute function is side-effect free.
2. Plugin handles missing required inputs with explicit errors.
3. Config defaults exist and are documented.
4. Report section renders without runtime-only assumptions.
5. Unit and integration tests pass.
6. Doc updates added to `interpretability_suite.md` if behavior is user-facing.

## 9) Recommended Pattern For Future Methods

For methods like spatial entropy or boundary rugosity:

1. Add feature calculator in `tier2/`.
2. Add plugin + default params.
3. Add one table section and one plot section.
4. Keep pipeline unchanged.

This preserves the architecture promise:

- new analyses are plug-ins, not pipeline rewrites.
