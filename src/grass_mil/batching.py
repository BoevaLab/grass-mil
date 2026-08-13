"""Splitting batched graphs back into their per-subgraph parts.

A leaf module: it depends only on torch and PyG, so both the data layer (which
builds these batches) and the model layer (which un-builds them at inference)
can use it without importing each other.

`Batch.to_data_list()` only works for batches assembled by
`Batch.from_data_list()`, which records the `_slice_dict`/`_inc_dict`
bookkeeping it needs. The ShaDow samplers construct their `Batch` directly and
delimit subgraphs with a `ptr` tensor instead, so `to_data_list()` raises on
them. `split_batch_by_ptr` reconstructs the pieces from `ptr`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import torch
from torch_geometric.data import Batch, Data

# Handled explicitly when rebuilding a subgraph, so they must not be copied
# across by the generic attribute loop below.
_STRUCTURAL_KEYS = {
    "x",
    "edge_index",
    "edge_attr",
    "batch",
    "ptr",
    "root_n_id",
    "n_id",
    "adj_t",
    "num_nodes",
}


def _select_sequence_value(
    *, key: str, value: Sequence[Any], subgraph_index: int, num_subgraphs: int
) -> Any:
    """Pick this subgraph's entry from a per-subgraph sequence.

    Sequences that are not per-subgraph (schema name lists, for instance) are
    passed through whole rather than sliced by position.
    """
    if len(value) == num_subgraphs:
        return [value[subgraph_index]]
    return value


def split_batch_by_ptr(batch: Batch) -> List[Data]:
    """Reconstruct the per-subgraph `Data` objects delimited by `batch.ptr`.

    Node-level tensors are sliced to the subgraph's node range, edges are
    restricted to it and relabelled, and `root_n_id` is rebased onto the
    subgraph's local indexing.
    """
    from torch_geometric.utils import subgraph

    if getattr(batch, "ptr", None) is None:
        raise ValueError("split_batch_by_ptr requires a batch carrying `ptr`.")

    num_subgraphs = len(batch.ptr) - 1
    num_nodes = int(batch.x.size(0))
    has_edge_attr = "edge_attr" in batch
    num_edges = int(batch.edge_index.size(1))

    out: List[Data] = []
    for i in range(num_subgraphs):
        node_start = int(batch.ptr[i])
        node_end = int(batch.ptr[i + 1])
        node_ids = torch.arange(node_start, node_end, device=batch.ptr.device)

        sub_edge_index, sub_edge_attr = subgraph(
            subset=node_ids,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr if has_edge_attr else None,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

        root_n_id = batch.root_n_id[i : i + 1] if "root_n_id" in batch else None
        # Rebase batch-space root indices onto the subgraph. Global ids (node
        # ids in the source graph, flagged by `root_n_id_is_global`) must be
        # left alone: subtracting a node offset from one silently turns it into
        # an unrelated node, which is how the root stopped being findable in its
        # own subgraph.
        root_is_global = False
        flags = getattr(batch, "root_n_id_is_global", None)
        if flags is not None:
            flags = torch.as_tensor(flags).view(-1)
            root_is_global = bool(flags[0]) if flags.numel() else False
        if root_n_id is not None and root_n_id.numel() == 1 and not root_is_global:
            root_value = int(root_n_id.item())
            if node_start <= root_value < node_end:
                root_n_id = root_n_id - node_start

        sub_data = Data(
            x=batch.x[node_ids],
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            root_n_id=root_n_id,
        )
        if hasattr(batch, "n_id") and isinstance(batch.n_id, torch.Tensor):
            sub_data.n_id = batch.n_id[node_ids]

        for key, val in batch:
            if key in _STRUCTURAL_KEYS:
                continue
            if isinstance(val, torch.Tensor):
                if val.dim() > 0 and val.size(0) == num_subgraphs:
                    setattr(sub_data, key, val[i : i + 1])
                elif val.dim() == 0:
                    setattr(sub_data, key, val)
                elif val.dim() > 0 and val.size(0) == num_nodes:
                    # Node-level tensors must be sliced down to this subgraph.
                    # They are NOT redundant with `x`: `categorical_codes`
                    # carries the cell-type codes the encoder embeds, and `pos`
                    # carries coordinates. Dropping them here silently removed
                    # cell type from every sampled subgraph.
                    setattr(sub_data, key, val[node_ids])
                elif val.dim() > 0 and has_edge_attr and val.size(0) == num_edges:
                    continue
                else:
                    setattr(sub_data, key, val)
            elif isinstance(val, (list, tuple)):
                setattr(
                    sub_data,
                    key,
                    _select_sequence_value(
                        key=key,
                        value=val,
                        subgraph_index=i,
                        num_subgraphs=num_subgraphs,
                    ),
                )
            else:
                setattr(sub_data, key, val)
        out.append(sub_data)
    return out


def extract_subgraphs(batch: Any) -> Optional[List[Data]]:
    """Best available split of `batch` into per-subgraph `Data` objects.

    Prefers PyG's own `to_data_list()`, and falls back to `ptr` for batches the
    ShaDow samplers built directly. Returns None only when the object carries
    neither route, which means it is not a batched graph at all.

    Deliberately does not swallow errors: a genuine failure to split raises,
    rather than surfacing several steps downstream as a missing payload field.
    """
    if hasattr(batch, "to_data_list"):
        try:
            return list(batch.to_data_list())
        except (RuntimeError, KeyError, AttributeError):
            # Batch was not assembled by `Batch.from_data_list()`. The ShaDow
            # samplers construct it directly and track subgraph boundaries in
            # `ptr`, so fall through to that.
            pass
    if getattr(batch, "ptr", None) is not None:
        return split_batch_by_ptr(batch)
    return None


def apply_per_subgraph(batch: Batch, transform: Callable[[Data], Data]) -> Batch:
    """Apply `transform` to each subgraph and re-collate."""
    return Batch.from_data_list([transform(d) for d in split_batch_by_ptr(batch)])
