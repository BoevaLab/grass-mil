"""Merge several graph units into one disconnected graph.

Used by the pretraining sampler (``runtime.merge_units``: ego-graph roots are
drawn from the union of all regions, as the preprint's BGRL does) and by the
patient-level "merged bag" scoring, where the cores of one patient are pooled
into a single bag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch_geometric.data import Data

__all__ = ["merge_graph_units"]

_NODE_KEYS = ("x", "pos", "x_raw", "categorical_codes")
_EDGE_KEYS = ("edge_attr",)
_COPY_FIRST = ("edge_attr_names", "categorical_labels", "categorical_slices", "molecular_feature_names")


def merge_graph_units(
    units: Sequence[Data],
    *,
    keep_graph_attrs: Optional[Sequence[str]] = ("graph_y", "graph_w"),
) -> Data:
    """Concatenate node and edge tensors with offset ``edge_index``.

    The merged graph carries ``unit_index`` (node -> position in ``units``),
    ``unit_region_ids`` / ``unit_sample_ids`` / ``unit_patch_ids`` lists and
    ``unit_ptr`` (node offsets). Graph-level attributes listed in
    ``keep_graph_attrs`` are stacked per unit when present on every unit.
    """
    if len(units) == 0:
        raise ValueError("merge_graph_units needs at least one unit.")
    node_parts: Dict[str, List[torch.Tensor]] = {}
    edge_parts: Dict[str, List[torch.Tensor]] = {}
    edge_index_parts: List[torch.Tensor] = []
    unit_index: List[torch.Tensor] = []
    offsets = [0]
    for i, unit in enumerate(units):
        n = int(unit.num_nodes)
        for key in _NODE_KEYS:
            val = getattr(unit, key, None)
            if isinstance(val, torch.Tensor):
                node_parts.setdefault(key, []).append(val)
        for key in _EDGE_KEYS:
            val = getattr(unit, key, None)
            if isinstance(val, torch.Tensor):
                edge_parts.setdefault(key, []).append(val)
        ei = unit.edge_index
        edge_index_parts.append(ei + offsets[-1] if ei is not None else torch.zeros(2, 0, dtype=torch.long))
        unit_index.append(torch.full((n,), i, dtype=torch.long))
        offsets.append(offsets[-1] + n)
    merged = Data(edge_index=torch.cat(edge_index_parts, dim=1))
    for key, parts in node_parts.items():
        if len(parts) != len(units):
            raise ValueError(f"Node attribute '{key}' is missing on some units; cannot merge.")
        setattr(merged, key, torch.cat(parts, dim=0))
    for key, parts in edge_parts.items():
        if len(parts) != len(units):
            raise ValueError(f"Edge attribute '{key}' is missing on some units; cannot merge.")
        setattr(merged, key, torch.cat(parts, dim=0))
    merged.num_nodes = offsets[-1]
    merged.unit_index = torch.cat(unit_index)
    merged.unit_ptr = torch.tensor(offsets, dtype=torch.long)
    merged.unit_region_ids = [None if getattr(u, "region_id", None) is None else str(u.region_id) for u in units]
    merged.unit_sample_ids = [None if getattr(u, "sample_id", None) is None else str(u.sample_id) for u in units]
    merged.unit_patch_ids = [None if getattr(u, "patch_id", None) is None else str(u.patch_id) for u in units]
    for key in _COPY_FIRST:
        val = getattr(units[0], key, None)
        if val is not None:
            setattr(merged, key, val)
    for key in keep_graph_attrs or ():
        vals = [getattr(u, key, None) for u in units]
        if all(isinstance(v, torch.Tensor) for v in vals):
            setattr(merged, f"unit_{key}", torch.cat([v.view(1, -1) for v in vals], dim=0))
    return merged
