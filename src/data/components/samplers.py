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
    runtime: Dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "SamplerConfig":
        if cfg is None:
            return cls(name="identity", kwargs={}, runtime={})
        return cls(
            name=cfg.get("name", "identity"),
            kwargs=cfg.get("kwargs", {}),
            runtime=cfg.get("runtime", {}),
        )


@dataclass
class RuntimeShadowConfig:
    enabled: bool = False
    depth: int = 2
    num_neighbors: int = 8
    subgraph_batch_size: int = 32
    replace: bool = False
    shuffle_subgraphs: bool = True
    node_idx: Optional[torch.Tensor] = None
    proportional_root_sampling: bool = True
    property_name: str = "cell_type"
    weight_mode: str = "inverse"
    min_weight: float = 1e-6
    subsample_fraction: float = 1.0
    subsample_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "RuntimeShadowConfig":
        cfg = cfg or {}
        weight_mode = str(cfg.get("weight_mode", "inverse")).strip().lower()
        allowed_weight_modes = {"inverse", "sqrt_inverse", "proportional"}
        if weight_mode not in allowed_weight_modes:
            raise ValueError(
                f"Unsupported weight_mode '{weight_mode}'. "
                f"Expected one of {sorted(allowed_weight_modes)}."
            )
        min_weight = float(cfg.get("min_weight", 1e-6))
        if min_weight <= 0:
            min_weight = 1e-6
        subsample_fraction = float(cfg.get("subsample_fraction", 1.0))
        if subsample_fraction == 0.0:
            raise ValueError(
                "runtime.subsample_fraction=0.0 selects no roots and is invalid."
            )
        if not (0.0 < subsample_fraction <= 1.0):
            raise ValueError(
                "runtime.subsample_fraction must be within (0, 1]. "
                f"Got {subsample_fraction}."
            )
        subsample_seed = cfg.get("subsample_seed")
        if subsample_seed is not None:
            subsample_seed = int(subsample_seed)
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            depth=int(cfg.get("depth", 2)),
            num_neighbors=int(cfg.get("num_neighbors", 8)),
            subgraph_batch_size=int(cfg.get("subgraph_batch_size", 32)),
            replace=bool(cfg.get("replace", False)),
            shuffle_subgraphs=bool(cfg.get("shuffle_subgraphs", True)),
            node_idx=cfg.get("node_idx"),
            proportional_root_sampling=bool(
                cfg.get("proportional_root_sampling", True)
            ),
            property_name=str(cfg.get("property_name", "cell_type")),
            weight_mode=weight_mode,
            min_weight=min_weight,
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
        )


def _build_weighted_node_idx(
    data: Data, *, property_name: str, weight_mode: str, min_weight: float
) -> Optional[torch.Tensor]:
    if not hasattr(data, "categorical_index") or not hasattr(
        data, "categorical_slices"
    ):
        return None
    categorical_index = getattr(data, "categorical_index")
    categorical_slices = getattr(data, "categorical_slices")
    if not isinstance(categorical_slices, dict):
        return None
    if property_name not in categorical_slices:
        return None
    if (
        not isinstance(categorical_index, torch.Tensor)
        or categorical_index.numel() == 0
    ):
        return None
    if categorical_index.dim() != 2:
        return None

    prop_col = int(categorical_slices[property_name])
    if prop_col < 0 or prop_col >= int(categorical_index.size(1)):
        return None

    node_labels = categorical_index[:, prop_col].long().view(-1)
    if node_labels.numel() == 0:
        return None

    unique_labels, inverse, counts = torch.unique(
        node_labels, sorted=False, return_inverse=True, return_counts=True
    )
    if unique_labels.numel() == 0:
        return None

    counts = counts.float()
    if weight_mode == "inverse":
        class_weights = 1.0 / counts
    elif weight_mode == "sqrt_inverse":
        class_weights = 1.0 / torch.sqrt(counts)
    elif weight_mode == "proportional":
        class_weights = counts / counts.sum()
    else:
        return None

    class_weights = torch.clamp(class_weights, min=float(min_weight))
    node_weights = class_weights[inverse]
    weight_sum = node_weights.sum()
    if not torch.isfinite(weight_sum) or float(weight_sum) <= 0.0:
        return None
    probs = node_weights / weight_sum
    if not torch.isfinite(probs).all():
        return None

    num_samples = int(getattr(data, "num_nodes", node_labels.numel()))
    if num_samples <= 0:
        num_samples = int(node_labels.numel())
    if num_samples <= 0:
        return None
    return torch.multinomial(probs, num_samples=num_samples, replacement=True).long()


def _resolve_root_candidates(data: Data, node_idx: Optional[torch.Tensor]) -> torch.Tensor:
    if node_idx is None:
        num_nodes = int(getattr(data, "num_nodes", 0))
        if num_nodes <= 0:
            return torch.empty(0, dtype=torch.long)
        return torch.arange(num_nodes, dtype=torch.long)
    if isinstance(node_idx, torch.Tensor):
        if node_idx.dtype == torch.bool:
            return node_idx.nonzero(as_tuple=False).view(-1).long()
        return node_idx.view(-1).long()
    if hasattr(node_idx, "__len__"):
        return torch.as_tensor(list(node_idx), dtype=torch.long)
    return torch.empty(0, dtype=torch.long)


def _subsample_count(num_candidates: int, subsample_fraction: float) -> int:
    if num_candidates <= 0:
        return 0
    if subsample_fraction >= 1.0:
        return num_candidates
    n_keep = int(num_candidates * subsample_fraction)
    if n_keep <= 0:
        n_keep = 1
    return min(n_keep, num_candidates)


def _subsample_root_candidates(
    candidates: torch.Tensor,
    *,
    subsample_fraction: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    n_candidates = int(candidates.numel())
    n_keep = _subsample_count(n_candidates, subsample_fraction)
    if n_keep >= n_candidates:
        return candidates
    perm = torch.randperm(n_candidates, generator=generator)
    selected = perm[:n_keep].sort().values
    return candidates.index_select(0, selected)


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


class _RuntimeUnitShadowDatasetLoader:
    def __init__(
        self,
        *,
        dataset: Dataset,
        strategy: BaseSamplerStrategy,
        batch_size: int,
        runtime: RuntimeShadowConfig,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        self.dataset = dataset
        self.strategy = strategy
        self.dataset_batch_size = batch_size
        self.runtime = runtime
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.kwargs = kwargs

    def __len__(self) -> int:
        if len(self.dataset) == 0:
            return 0
        subgraph_batch_size = int(self.runtime.subgraph_batch_size)
        if subgraph_batch_size <= 0:
            subgraph_batch_size = max(1, int(self.dataset_batch_size))

        total_batches = 0
        for idx in range(len(self.dataset)):
            unit_data = self.dataset[idx]
            effective_node_idx = self.runtime.node_idx
            if self.runtime.proportional_root_sampling:
                computed_node_idx = _build_weighted_node_idx(
                    unit_data,
                    property_name=self.runtime.property_name,
                    weight_mode=self.runtime.weight_mode,
                    min_weight=self.runtime.min_weight,
                )
                if computed_node_idx is not None:
                    effective_node_idx = computed_node_idx
            num_candidate_roots = _resolve_num_roots(unit_data, effective_node_idx)
            num_roots = _subsample_count(
                num_candidate_roots, float(self.runtime.subsample_fraction)
            )
            total_batches += int((num_roots + subgraph_batch_size - 1) // subgraph_batch_size)
        return total_batches

    def __iter__(self):
        if len(self.dataset) == 0:
            return

        indices = torch.arange(len(self.dataset), dtype=torch.long)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]

        subgraph_batch_size = int(self.runtime.subgraph_batch_size)
        if subgraph_batch_size <= 0:
            subgraph_batch_size = max(1, int(self.dataset_batch_size))
        subsample_generator = None
        if self.runtime.subsample_seed is not None:
            subsample_generator = torch.Generator()
            subsample_generator.manual_seed(int(self.runtime.subsample_seed))

        for idx in indices.tolist():
            unit_data = self.dataset[idx]
            effective_node_idx = self.runtime.node_idx
            if self.runtime.proportional_root_sampling:
                computed_node_idx = _build_weighted_node_idx(
                    unit_data,
                    property_name=self.runtime.property_name,
                    weight_mode=self.runtime.weight_mode,
                    min_weight=self.runtime.min_weight,
                )
                if computed_node_idx is not None:
                    effective_node_idx = computed_node_idx
            effective_root_candidates = _resolve_root_candidates(
                unit_data, effective_node_idx
            )
            effective_root_candidates = _subsample_root_candidates(
                effective_root_candidates,
                subsample_fraction=float(self.runtime.subsample_fraction),
                generator=subsample_generator,
            )
            unit_loader_kwargs: Dict[str, Any] = {
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.kwargs.get("persistent_workers", False)
                if self.num_workers > 0
                else False,
            }
            if self.num_workers > 0 and "prefetch_factor" in self.kwargs:
                unit_loader_kwargs["prefetch_factor"] = self.kwargs["prefetch_factor"]
            loader = self.strategy.build_unit_loader(
                data=unit_data,
                depth=int(self.runtime.depth),
                num_neighbors=int(self.runtime.num_neighbors),
                batch_size=subgraph_batch_size,
                node_idx=effective_root_candidates,
                replace=bool(self.runtime.replace),
                shuffle=bool(self.runtime.shuffle_subgraphs and self.shuffle),
                transform=self.kwargs.get("transform"),
                **unit_loader_kwargs,
            )
            for sub_batch in loader:
                yield _normalize_shadow_batch(sub_batch, unit_data)


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

    def __init__(self, runtime: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.runtime = RuntimeShadowConfig.from_dict(runtime)
        self.kwargs = kwargs

    def build_dataset_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        if self.runtime.enabled:
            return _RuntimeUnitShadowDatasetLoader(
                dataset=dataset,
                strategy=self,
                batch_size=batch_size,
                runtime=self.runtime,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,
                **kwargs,
            )
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
        num_workers = kwargs.get("num_workers", 0)
        loader_kwargs: Dict[str, Any] = {
            "data": data,
            "depth": depth,
            "num_neighbors": num_neighbors,
            "node_idx": node_idx,
            "replace": replace,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": kwargs.get("pin_memory", False),
            "persistent_workers": kwargs.get("persistent_workers", False)
            if num_workers > 0
            else False,
        }
        if num_workers > 0 and "prefetch_factor" in kwargs:
            loader_kwargs["prefetch_factor"] = kwargs["prefetch_factor"]
        return ShaDowKHopSampler(**loader_kwargs)


class ShadowCustomStrategy(BaseSamplerStrategy):
    """Custom ShaDow sampler with optional per-subgraph transform support."""

    def __init__(self, runtime: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.runtime = RuntimeShadowConfig.from_dict(runtime)
        self.kwargs = kwargs

    def build_dataset_loader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        **kwargs: Any,
    ):
        if self.runtime.enabled:
            return _RuntimeUnitShadowDatasetLoader(
                dataset=dataset,
                strategy=self,
                batch_size=batch_size,
                runtime=self.runtime,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=shuffle,
                **kwargs,
            )
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
        if not WITH_TORCH_SPARSE:
            if transform is not None:
                raise ValueError(
                    "ShadowCustomStrategy transform support requires torch-sparse. "
                    "Install torch-sparse or disable transforms."
                )
            native = ShadowNativeStrategy(runtime={"enabled": False})
            return native.build_unit_loader(
                data=data,
                depth=depth,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                node_idx=node_idx,
                replace=replace,
                shuffle=shuffle,
                transform=None,
                **kwargs,
            )
        num_workers = kwargs.get("num_workers", 0)
        loader_kwargs: Dict[str, Any] = {
            "data": data,
            "depth": depth,
            "num_neighbors": num_neighbors,
            "node_idx": node_idx,
            "replace": replace,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "transform": transform,
            "num_workers": num_workers,
            "pin_memory": kwargs.get("pin_memory", False),
            "persistent_workers": kwargs.get("persistent_workers", False)
            if num_workers > 0
            else False,
        }
        if num_workers > 0 and "prefetch_factor" in kwargs:
            loader_kwargs["prefetch_factor"] = kwargs["prefetch_factor"]
        return _ShaDowKHopSamplerWithTransform(**loader_kwargs)


def _expand_graph_level_value(value: Any, count: int) -> Any:
    if count <= 0:
        return value
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.repeat(count)
        if value.size(0) == count:
            return value
        if value.size(0) == 1:
            repeats = [count] + [1] * (value.dim() - 1)
            return value.repeat(*repeats)
        return value
    if isinstance(value, (str, bytes)):
        return [value] * count
    if isinstance(value, (list, tuple)):
        if len(value) == count:
            return list(value)
        if len(value) == 1:
            return [value[0]] * count
        return list(value)
    return [value] * count


def _ensure_edge_index(batch: Batch) -> Batch:
    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        return batch
    if not hasattr(batch, "adj_t"):
        return batch
    row, col, edge_val = batch.adj_t.t().coo()
    batch.edge_index = torch.stack([row, col], dim=0)
    if edge_val is not None and not hasattr(batch, "edge_attr"):
        batch.edge_attr = edge_val
    return batch


def _resolve_batch_root_node_ids(batch: Batch) -> Optional[torch.Tensor]:
    root_n_id = getattr(batch, "root_n_id", None)
    if not isinstance(root_n_id, torch.Tensor):
        return None
    roots = root_n_id.detach().cpu().long().view(-1)
    if roots.numel() == 0:
        return roots

    roots_abs = roots
    if hasattr(batch, "ptr") and isinstance(batch.ptr, torch.Tensor):
        ptr = batch.ptr.detach().cpu().long().view(-1)
        if ptr.numel() == roots.numel() + 1:
            counts = ptr[1:] - ptr[:-1]
            if bool(torch.all((roots >= 0) & (roots < counts))):
                roots_abs = ptr[:-1] + roots
            elif bool(torch.all((roots >= ptr[:-1]) & (roots < ptr[1:]))):
                roots_abs = roots

    n_id = getattr(batch, "n_id", None)
    if isinstance(n_id, torch.Tensor):
        n_id = n_id.detach().cpu().long().view(-1)
        if n_id.numel() > 0 and bool(torch.all((roots_abs >= 0) & (roots_abs < n_id.numel()))):
            return n_id.index_select(0, roots_abs)
    return roots_abs


def _normalize_shadow_batch(batch: Batch, source_data: Optional[Data] = None) -> Batch:
    batch = _ensure_edge_index(batch)
    num_subgraphs = int(len(batch.ptr) - 1) if hasattr(batch, "ptr") else 1

    for key in ["graph_y", "graph_w", "sample_id", "region_id", "patch_id"]:
        if hasattr(batch, key):
            setattr(
                batch,
                key,
                _expand_graph_level_value(getattr(batch, key), num_subgraphs),
            )
        elif source_data is not None and hasattr(source_data, key):
            setattr(
                batch,
                key,
                _expand_graph_level_value(getattr(source_data, key), num_subgraphs),
            )
    root_node_ids = _resolve_batch_root_node_ids(batch)
    if isinstance(root_node_ids, torch.Tensor):
        batch.root_n_id = root_node_ids
        batch.root_n_id_is_global = torch.ones(
            int(root_node_ids.numel()), dtype=torch.bool
        )
    return batch


def _is_schema_name_key(key: str) -> bool:
    return str(key).endswith("_names")


def _select_sequence_value_for_subgraph(
    *,
    key: str,
    value: list[Any] | tuple[Any, ...],
    subgraph_index: int,
    num_subgraphs: int,
) -> list[Any] | tuple[Any, ...]:
    if _is_schema_name_key(key):
        if (
            len(value) == num_subgraphs
            and len(value) > subgraph_index
            and isinstance(value[subgraph_index], (list, tuple))
        ):
            item = value[subgraph_index]
            return list(item) if isinstance(value, list) else tuple(item)
        return list(value) if isinstance(value, list) else tuple(value)

    if len(value) == num_subgraphs and len(value) > subgraph_index:
        item = value[subgraph_index]
        return [item] if isinstance(value, list) else (item,)
    return value


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
    if name == "identity":
        return _SAMPLER_REGISTRY[name](**cfg.kwargs)
    return _SAMPLER_REGISTRY[name](runtime=cfg.runtime, **cfg.kwargs)


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
        batch.n_id = n_id

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

        batch = _normalize_shadow_batch(batch, self.data)
        if self.transform is not None:
            batch = self._apply_transform_to_each_subgraph(batch)
        return batch

    def _apply_transform_to_each_subgraph(self, batch: Batch) -> Batch:
        from torch_geometric.utils import subgraph

        processed = []
        num_subgraphs = len(batch.ptr) - 1
        skip_keys = {
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
        for i in range(len(batch.ptr) - 1):
            node_start = int(batch.ptr[i])
            node_end = int(batch.ptr[i + 1])
            node_ids = torch.arange(node_start, node_end, device=batch.ptr.device)
            sub_edge_index, sub_edge_attr = subgraph(
                subset=node_ids,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if "edge_attr" in batch else None,
                relabel_nodes=True,
                num_nodes=int(batch.x.size(0)),
            )
            root_n_id = batch.root_n_id[i : i + 1]
            if root_n_id.numel() == 1:
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
                if key in skip_keys:
                    continue
                if isinstance(val, torch.Tensor):
                    if val.dim() > 0 and val.size(0) == num_subgraphs:
                        setattr(sub_data, key, val[i : i + 1])
                    elif val.dim() == 0:
                        setattr(sub_data, key, val)
                    elif val.dim() > 0 and val.size(0) == batch.x.size(0):
                        # Node-level feature tensors are already represented by `x`.
                        continue
                    elif val.dim() > 0 and "edge_attr" in batch and val.size(0) == batch.edge_index.size(1):
                        continue
                    else:
                        setattr(sub_data, key, val)
                elif isinstance(val, list):
                    setattr(
                        sub_data,
                        key,
                        _select_sequence_value_for_subgraph(
                            key=key,
                            value=val,
                            subgraph_index=i,
                            num_subgraphs=num_subgraphs,
                        ),
                    )
                elif isinstance(val, tuple):
                    setattr(
                        sub_data,
                        key,
                        _select_sequence_value_for_subgraph(
                            key=key,
                            value=val,
                            subgraph_index=i,
                            num_subgraphs=num_subgraphs,
                        ),
                    )
                else:
                    setattr(sub_data, key, val)
            processed.append(self.transform(sub_data))
        return Batch.from_data_list(processed)


def _resolve_num_roots(data: Data, node_idx: Optional[torch.Tensor]) -> int:
    if node_idx is None:
        return int(getattr(data, "num_nodes", 0))
    if isinstance(node_idx, torch.Tensor):
        if node_idx.dtype == torch.bool:
            return int(node_idx.sum().item())
        return int(node_idx.numel())
    if hasattr(node_idx, "__len__"):
        return int(len(node_idx))
    return int(getattr(data, "num_nodes", 0))
