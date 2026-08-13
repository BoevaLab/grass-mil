from __future__ import annotations

from dataclasses import replace

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from grass_mil.contracts import (
    NicheSummary,
    InterpretabilityDataset,
    PluginResult,
    ReportBundle,
)
from grass_mil.interpretability.core.biomarkers import niche_composition_summary
from grass_mil.interpretability.core.niches import relabel_niches_by_composition
from grass_mil.interpretability.core.clustering import run_clustering
from grass_mil.interpretability.core.data import extract_embedding_set
from grass_mil.interpretability.core.reduction import run_reduction
from grass_mil.interpretability.plugins.base import PluginContext
from grass_mil.interpretability.plugins.builtin import create_builtin_registry
from grass_mil.interpretability.plugins.registry import PluginRegistry


def _is_plugin_input_available(
    dataset: InterpretabilityDataset,
    context: PluginContext,
    input_name: str,
) -> bool:
    key = str(input_name).strip()
    if not key:
        return False
    if key in context.state:
        return context.state[key] is not None
    if hasattr(dataset, key):
        return getattr(dataset, key) is not None
    return False


def _validate_plugin_required_inputs(
    plugin_name: str,
    required_inputs: list[str],
    dataset: InterpretabilityDataset,
    context: PluginContext,
) -> None:
    required = [str(name).strip() for name in required_inputs if str(name).strip()]
    missing = [
        name
        for name in required
        if not _is_plugin_input_available(dataset=dataset, context=context, input_name=name)
    ]
    if missing:
        missing_text = ", ".join(sorted(set(missing)))
        raise ValueError(
            f"Plugin {plugin_name!r} is missing required inputs: {missing_text}. "
            "Provide these inputs in dataset fields or plugin context state."
        )


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def run_interpretability_pipeline(
    dataset: InterpretabilityDataset,
    config: Dict[str, Any],
    *,
    artifacts_dir: Path,
    plugin_registry: Optional[PluginRegistry] = None,
) -> ReportBundle:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    resolved_registry = (
        plugin_registry if plugin_registry is not None else create_builtin_registry()
    )

    emb_set = extract_embedding_set(
        dataset,
        embedding_prefixes=tuple(
            config.get("embedding_prefixes", ("inst_emb_", "emb_", "graph_emb_"))
        ),
    )

    clustering_cfg = dict(config.get("clustering", {}))
    clustering_enabled = bool(clustering_cfg.get("enabled", True))
    cluster_on_pca = bool(clustering_cfg.get("cluster_on_pca", False))
    cluster_pca_components_raw = clustering_cfg.get("pca_components", None)
    cluster_pca_components = (
        int(cluster_pca_components_raw) if cluster_pca_components_raw is not None else None
    )
    effective_cluster_pca_components = cluster_pca_components
    if clustering_enabled and cluster_on_pca:
        if cluster_pca_components is None:
            raise ValueError("clustering.cluster_on_pca=true requires clustering.pca_components.")
        max_pca_components = int(min(emb_set.matrix.shape[0], emb_set.matrix.shape[1]))
        if max_pca_components < 1:
            raise ValueError("Cannot run PCA-based clustering with empty embedding matrix.")
        effective_cluster_pca_components = min(cluster_pca_components, max_pca_components)

    reduction_cfg = dict(config.get("reduction", {}))
    reduction_enabled = bool(reduction_cfg.get("enabled", True))
    reduction_result = None
    combo_pca_result = None
    if reduction_enabled:
        method = str(reduction_cfg.get("method", "pca"))
        params = dict(reduction_cfg.get("params", {}))
        reduction_input = emb_set.matrix
        if method.strip().lower() == "umap" and cluster_on_pca:
            combo_pca_result = run_reduction(
                "pca",
                emb_set.matrix,
                n_components=effective_cluster_pca_components,
                random_state=None,
            )
            reduction_input = combo_pca_result.embedding
        reduction_result = run_reduction(method, reduction_input, **params)

    clustering_result = None
    labels: Optional[np.ndarray] = None
    if clustering_enabled:
        method = str(clustering_cfg.get("method", "kmeans"))
        params = dict(clustering_cfg.get("params", {}))
        # Contract: when cluster_on_pca is disabled, niche on raw embeddings.
        # Reduction remains available for visualization/reporting, but does not
        # change the clustering feature space.
        clustering_input = emb_set.matrix
        if cluster_on_pca:
            if combo_pca_result is None:
                combo_pca_result = run_reduction(
                    "pca",
                    emb_set.matrix,
                    n_components=effective_cluster_pca_components,
                    random_state=None,
                )
            clustering_input = combo_pca_result.embedding
        clustering_result = run_clustering(method, clustering_input, **params)
        labels = clustering_result.labels

    niche_relabel_map: Dict[Any, Any] = {}
    if labels is not None and bool(config.get("relabel_niches_by_composition", True)):
        # Renumber once so the niche id carries the dendrogram order. Everything
        # downstream then gets composition order by sorting on the id.
        labels, niche_relabel_map = relabel_niches_by_composition(
            dataset.instance_table,
            labels,
            composition_prefix=str(config.get("composition_prefix", "comp_")),
        )
        if clustering_result is not None:
            clustering_result = replace(clustering_result, labels=labels)

    niche_summary: Optional[NicheSummary] = None
    if labels is not None:
        niche_summary = niche_composition_summary(
            dataset.instance_table,
            labels,
            composition_prefix=str(config.get("composition_prefix", "comp_")),
            variance_estimator=str(config.get("variance_estimator", "unbiased")),
        )

    context = PluginContext(
        state={
            "reduction": reduction_result,
            "clustering": clustering_result,
            "niche_labels": labels,
            "niche_summary": niche_summary,
        }
    )
    plugin_results: Dict[str, PluginResult] = {}
    plugin_cfg = dict(config.get("plugins", {}))
    enabled_plugins = list(plugin_cfg.get("enabled", []))
    plugin_params = dict(plugin_cfg.get("params", {}))

    for plugin_name in enabled_plugins:
        plugin = resolved_registry.get(str(plugin_name))
        _validate_plugin_required_inputs(
            plugin_name=str(plugin_name),
            required_inputs=list(plugin.required_inputs()),
            dataset=dataset,
            context=context,
        )
        result = plugin.run(dataset, context, **dict(plugin_params.get(plugin_name, {})))
        plugin_results[str(plugin_name)] = result

    # Persist core artifacts
    base_instance = dataset.instance_table.copy()
    if labels is not None:
        base_instance["niche_label"] = labels
    _write_table(base_instance, artifacts_dir / "instance_with_niches.csv")
    if reduction_result is not None:
        red_df = pd.DataFrame(reduction_result.embedding)
        red_df.columns = [f"reduced_{i}" for i in range(red_df.shape[1])]
        red_df.insert(
            0, dataset.id_column, dataset.instance_table[dataset.id_column].astype(str).tolist()
        )
        _write_table(red_df, artifacts_dir / "reduction.csv")
    if clustering_result is not None:
        niche_df = pd.DataFrame(
            {
                dataset.id_column: dataset.instance_table[dataset.id_column].astype(str).tolist(),
                "niche_label": clustering_result.labels,
            }
        )
        _write_table(niche_df, artifacts_dir / "niche_labels.csv")

    summary_payload: Dict[str, Any] = {
        "rows": int(len(dataset.instance_table)),
        "reduction_method": reduction_result.method if reduction_result else None,
        "clustering_method": clustering_result.method if clustering_result else None,
        "cluster_on_pca": cluster_on_pca,
        "cluster_pca_components": effective_cluster_pca_components,
        "enabled_plugins": enabled_plugins,
        "niche_relabel_map": {str(k): int(v) for k, v in niche_relabel_map.items()},
    }
    _write_json(summary_payload, artifacts_dir / "pipeline_summary.json")

    return ReportBundle(
        dataset=dataset,
        reduction=reduction_result,
        niche_feature_reduction=combo_pca_result if cluster_on_pca else None,
        clustering=clustering_result,
        niche_summary=niche_summary,
        plugin_results=plugin_results,
        artifacts_dir=artifacts_dir,
        metadata=summary_payload,
    )
