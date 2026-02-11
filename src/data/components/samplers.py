from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.loader import ShaDowKHopSampler
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for sampler strategies") from exc

try:
    from torch_geometric.typing import WITH_TORCH_SPARSE, SparseTensor
except Exception:  # pragma: no cover
    WITH_TORCH_SPARSE = False
    SparseTensor = None  # type: ignore


@dataclass
class SamplerConfig:
    name: str
    kwargs: Dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "SamplerConfig":
        if cfg is None:
            return cls(name="identity", kwargs={})
        return cls(name=cfg.get("name", "identity"), kwargs=cfg.get("kwargs", {}))


class BaseSamplerStrategy:
    """Data-layer agnostic sampler strategy.

    A strategy consumes graph units emitted by the data layer and returns iterables/loaders.
    """

    def build_dataset_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        raise NotImplementedError

    def build_unit_loader(self, data: Data, **kwargs: Any):
        raise NotImplementedError


class IdentityBatchStrategy(BaseSamplerStrategy):
    def build_dataset_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        loader_cls = PyGDataLoader or TorchDataLoader
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "shuffle": shuffle,
            "persistent_workers": kwargs.get("persistent_workers", False)
            if num_workers > 0
            else False,
        }
        if num_workers > 0 and "prefetch_factor" in kwargs:
            loader_kwargs["prefetch_factor"] = kwargs["prefetch_factor"]
        return loader_cls(
            **loader_kwargs,
        )

    def build_unit_loader(self, data: Data, **kwargs: Any):
        # Identity strategy on a single unit is just one-element iteration.
        return [data]


class ShadowNativeStrategy(BaseSamplerStrategy):
    """Native PyG ShaDow sampler, to be used per graph unit."""

    def build_dataset_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        identity = IdentityBatchStrategy()
        return identity.build_dataset_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            **kwargs,
        )

    def build_unit_loader(
        self,
        data: Data,
        depth: int,
        num_neighbors: int,
        batch_size: int,
        node_idx: Optional[torch.Tensor] = None,
        replace: bool = False,
        shuffle: bool = True,
        transform: Optional[Callable[[Data], Data]] = None,
        **kwargs: Any,
    ):
        if transform is not None:
            # Native loader does not guarantee per-subgraph post-collate transform behavior.
            # Caller should choose ShadowCustomStrategy when strict semantics are required.
            raise ValueError(
                "ShadowNativeStrategy does not support strict post-hoc per-subgraph transforms. "
                "Use ShadowCustomStrategy instead."
            )
        return ShaDowKHopSampler(
            data=data,
            depth=depth,
            num_neighbors=num_neighbors,
            node_idx=node_idx,
            replace=replace,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", False),
            persistent_workers=kwargs.get("persistent_workers", False),
            prefetch_factor=kwargs.get("prefetch_factor", 2),
        )


class ShadowCustomStrategy(BaseSamplerStrategy):
    """Custom ShaDow sampler with optional per-subgraph transform support."""

    def build_dataset_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        identity = IdentityBatchStrategy()
        return identity.build_dataset_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            **kwargs,
        )

    def build_unit_loader(
        self,
        data: Data,
        depth: int,
        num_neighbors: int,
        batch_size: int,
        node_idx: Optional[torch.Tensor] = None,
        replace: bool = False,
        shuffle: bool = True,
        transform: Optional[Callable[[Data], Data]] = None,
        **kwargs: Any,
    ):
        return _ShaDowKHopSamplerWithTransform(
            data=data,
            depth=depth,
            num_neighbors=num_neighbors,
            node_idx=node_idx,
            replace=replace,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=transform,
            num_workers=kwargs.get("num_workers", 0),
            pin_memory=kwargs.get("pin_memory", False),
            persistent_workers=kwargs.get("persistent_workers", False),
            prefetch_factor=kwargs.get("prefetch_factor", 2),
        )


_SAMPLER_REGISTRY: Dict[str, Callable[..., BaseSamplerStrategy]] = {
    "identity": IdentityBatchStrategy,
    "shadow_native": ShadowNativeStrategy,
    "shadow_custom": ShadowCustomStrategy,
}


def register_sampler_strategy(name: str):
    def _decorator(cls):
        _SAMPLER_REGISTRY[name] = cls
        return cls

    return _decorator


def get_sampler_strategy(
    config: SamplerConfig | Dict[str, Any] | None,
) -> BaseSamplerStrategy:
    cfg = (
        config if isinstance(config, SamplerConfig) else SamplerConfig.from_dict(config)
    )
    name = cfg.name.lower()
    if name not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler strategy '{cfg.name}'. Available: {sorted(_SAMPLER_REGISTRY)}"
        )
    return _SAMPLER_REGISTRY[name](**cfg.kwargs)


class _ShaDowKHopSamplerWithTransform(torch.utils.data.DataLoader):
    def __init__(
        self,
        data: Data,
        depth: int,
        num_neighbors: int,
        node_idx: Optional[torch.Tensor] = None,
        replace: bool = False,
        transform: Optional[Callable[[Data], Data]] = None,
        **kwargs: Any,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(
                "Custom ShaDow sampler requires torch-sparse; install torch-sparse or use shadow_native."
            )
        self.data = copy.copy(data)
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.transform = transform

        if data.edge_index is not None:
            self.is_sparse_tensor = False
            row, col = data.edge_index.cpu()
            self.adj_t = SparseTensor(
                row=row,
                col=col,
                value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()
        else:
            self.is_sparse_tensor = True
            self.adj_t = data.adj_t.cpu()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        self.node_idx = node_idx

        super().__init__(node_idx.tolist(), collate_fn=self._collate, **kwargs)

    def _collate(self, n_id: Iterable[int]):
        n_id = torch.tensor(list(n_id))
        rowptr, col, value = self.adj_t.csr()
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr, col, n_id, self.depth, self.num_neighbors, self.replace
        )
        rowptr, col, n_id, e_id, ptr, root_n_id = out

        adj_t = SparseTensor(
            rowptr=rowptr,
            col=col,
            value=value[e_id] if value is not None else None,
            sparse_sizes=(n_id.numel(), n_id.numel()),
            is_sorted=True,
            trust_data=True,
        )

        batch = Batch(batch=torch.ops.torch_sparse.ptr2ind(ptr, n_id.numel()), ptr=ptr)
        batch.root_n_id = root_n_id

        if self.is_sparse_tensor:
            batch.adj_t = adj_t
        else:
            row, col, e_id = adj_t.t().coo()
            batch.edge_index = torch.stack([row, col], dim=0)

        for key, val in self.data:
            if key in ["edge_index", "adj_t", "num_nodes", "batch", "ptr"]:
                continue
            if (
                key == "y"
                and isinstance(val, torch.Tensor)
                and val.size(0) == self.data.num_nodes
            ):
                batch[key] = val[n_id][root_n_id]
            elif isinstance(val, torch.Tensor) and val.size(0) == self.data.num_nodes:
                batch[key] = val[n_id]
            elif isinstance(val, torch.Tensor) and val.size(0) == self.data.num_edges:
                batch[key] = val[e_id]
            else:
                batch[key] = val

        if self.transform is not None:
            batch = self._apply_transform_to_each_subgraph(batch)
        return batch

    def _apply_transform_to_each_subgraph(self, batch: Batch) -> Batch:
        from torch_geometric.utils import subgraph

        processed = []
        for i in range(len(batch.ptr) - 1):
            node_start = int(batch.ptr[i])
            node_end = int(batch.ptr[i + 1])
            node_ids = torch.arange(node_start, node_end, device=batch.ptr.device)
            sub_edge_index, sub_edge_attr = subgraph(
                subset=node_ids,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if "edge_attr" in batch else None,
                relabel_nodes=True,
            )
            sub_data = Data(
                x=batch.x[node_ids],
                edge_index=sub_edge_index,
                edge_attr=sub_edge_attr,
                root_n_id=batch.root_n_id[i : i + 1],
            )
            for key in ["graph_y", "graph_w", "sample_id", "region_id", "patch_id"]:
                if hasattr(batch, key):
                    val = getattr(batch, key)
                    setattr(sub_data, key, val)
            processed.append(self.transform(sub_data))
        return Batch.from_data_list(processed)
