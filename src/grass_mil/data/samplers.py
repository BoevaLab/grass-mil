"""Ego-graph sampling: every instance of a bag is the k-hop ego-graph of one cell.

:class:`EgoGraphLoader` turns a dataset of graph units into batches of
ego-graphs (one subgraph per root cell, built by PyG's ``ShaDowKHopSampler``):

* ``bags``: one batch per unit holding the ego-graphs of all of its selected
  roots, so the attention softmax and the loss of a bag see the whole unit
  (MIL training);
* ``chunks``: the selected roots of each unit in batches of ``batch_size``
  (evaluation, embedding extraction);
* ``merged``: the roots of all units pooled and shuffled into batches of
  ``batch_size`` (label-free pretraining).

Every batch carries, per ego-graph, ``root_cell`` (the root's node index in
its unit) and the unit's ids and ``graph_y`` target row.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Mapping

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.typing import WITH_TORCH_SPARSE

from .ego_radius import EgoRadiusConfig, EgoRadiusCutoff, radius_for_depth
from .interior import InteriorCache, InteriorConfig

__all__ = ["SamplerConfig", "EgoGraphLoader"]

MODES = ("bags", "chunks", "merged")
ID_KEYS = ("unit_id", "region_id", "sample_id")


@dataclass(frozen=True)
class SamplerConfig:
    """Ego-graph sampling settings.

    Attributes:
        depth: Hops of every ego-graph.
        num_neighbors: Neighbours kept per node and hop (``-1``: all, i.e. the
            full k-hop neighbourhood).
        roots_per_unit: Root cells drawn per unit and epoch (``None``: all).
        batch_size: Ego-graphs per batch in ``chunks`` and ``merged`` mode.
        merge_units: Pool the roots of all units (pretraining) instead of one
            bag per unit.
        root_weighting: ``None`` draws roots uniformly; ``inverse`` /
            ``sqrt_inverse`` weight them by the inverse (square-root) frequency
            of their cell type within the unit.
        interior_roots: Restrict roots to the tissue interior.
        ego_radius: Cap the physical extent of every ego-graph.
    """

    depth: int = 4
    num_neighbors: int = -1
    roots_per_unit: int | None = 256
    batch_size: int = 64
    merge_units: bool = False
    root_weighting: str | None = None
    interior_roots: InteriorConfig = field(default_factory=InteriorConfig)
    ego_radius: EgoRadiusConfig = field(default_factory=EgoRadiusConfig)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> SamplerConfig:
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown data.sampler keys: {sorted(unknown)}")
        cfg["interior_roots"] = InteriorConfig.from_dict(cfg.get("interior_roots"))
        cfg["ego_radius"] = EgoRadiusConfig.from_dict(cfg.get("ego_radius"))
        out = cls(**cfg)
        if out.depth < 1:
            raise ValueError("data.sampler.depth must be >= 1.")
        if out.roots_per_unit is not None and out.roots_per_unit < 1:
            raise ValueError("data.sampler.roots_per_unit must be positive or null.")
        if out.root_weighting not in (None, "inverse", "sqrt_inverse"):
            raise ValueError("data.sampler.root_weighting must be null, inverse or sqrt_inverse.")
        return out

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class _EgoGraphBatches(ShaDowKHopSampler):
    """``ShaDowKHopSampler`` that also records each ego-graph's root cell and
    applies the optional radius cap."""

    def __init__(self, data: Data, roots: torch.Tensor, *, depth: int, num_neighbors: int,
                 cutoff: EgoRadiusCutoff | None, **loader_kwargs: Any) -> None:
        data = data.clone()
        data.node_position = torch.arange(int(data.num_nodes))
        self.cutoff = cutoff
        neighbours = int(num_neighbors) if num_neighbors >= 0 else int(data.num_nodes)
        super().__init__(data, depth=depth, num_neighbors=neighbours, node_idx=roots, **loader_kwargs)

    def __collate__(self, n_id):
        batch = super().__collate__(n_id)
        n = int(batch.ptr.numel()) - 1
        batch.root_cell = batch.node_position[batch.root_n_id]
        del batch.node_position
        if self.cutoff is not None:
            batch = _cut_each(batch, self.cutoff)
        y = getattr(batch, "graph_y", None)
        if isinstance(y, torch.Tensor):
            batch.graph_y = y.view(1, -1).expand(n, -1).contiguous()
        for key in ID_KEYS:
            if isinstance(getattr(batch, key, None), str):
                batch[key] = [batch[key]] * n
        return batch


def _cut_each(batch: Batch, cutoff: EgoRadiusCutoff) -> Batch:
    """Apply the radius cap to every ego-graph of a ShaDow batch and re-batch."""
    ptr = batch.ptr
    n_nodes = int(batch.num_nodes)
    n_edges = int(batch.edge_index.size(1))
    node_keys = [k for k, v in batch if isinstance(v, torch.Tensor) and v.dim() and v.size(0) == n_nodes and k not in ("batch",)]
    edge_owner = batch.batch[batch.edge_index[0]]
    order = torch.argsort(edge_owner, stable=True)
    edge_counts = torch.bincount(edge_owner, minlength=int(ptr.numel()) - 1).tolist()
    edge_index = batch.edge_index[:, order]
    edge_attr = batch.edge_attr[order] if getattr(batch, "edge_attr", None) is not None else None
    parts, roots = [], []
    start_e = 0
    for i, count in enumerate(edge_counts):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        sub = Data(edge_index=edge_index[:, start_e : start_e + count] - lo, num_nodes=hi - lo)
        if edge_attr is not None and edge_attr.size(0) == n_edges:
            sub.edge_attr = edge_attr[start_e : start_e + count]
        for key in node_keys:
            sub[key] = batch[key][lo:hi]
        start_e += count
        sub, root = cutoff(sub, int(batch.root_n_id[i]) - lo)
        parts.append(sub)
        roots.append(root)
    out = Batch.from_data_list(parts)
    out.root_n_id = out.ptr[:-1] + torch.tensor(roots, dtype=torch.long)
    out.root_cell = batch.root_cell
    for key, value in batch:
        if key not in out and key not in ("batch", "ptr", "edge_index", "edge_attr"):
            out[key] = value
    return out


class EgoGraphLoader:
    """Iterable of ego-graph batches over a dataset of graph units.

    Args:
        dataset: Units with ``entries`` (``unit_id``, ``n_cells``) and
            ``__getitem__`` returning the unit graph.
        cfg: Sampling settings.
        mode: ``bags`` | ``chunks`` | ``merged``.
        roots_per_unit: Overrides ``cfg.roots_per_unit`` (``None``: all roots).
        seed: Fixes each unit's root subset (from the seed and the unit id), so
            it is identical on every pass; ``None`` redraws every pass.
        shuffle: Shuffle units (and roots in ``merged`` mode).
        num_workers: DataLoader workers; used in ``merged`` mode, where one
            loader spans the epoch. Per-unit modes collate in the main process.
    """

    def __init__(
        self,
        dataset,
        cfg: SamplerConfig,
        *,
        mode: str,
        roots_per_unit: int | None,
        seed: int | None = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not WITH_TORCH_SPARSE:
            raise ImportError(
                "Ego-graph sampling needs torch-sparse; install the wheel matching your PyTorch "
                "build, e.g. pip install torch-sparse -f https://data.pyg.org/whl/torch-<version>.html"
            )
        self.dataset = dataset
        self.cfg = cfg
        self.mode = mode
        self.roots_per_unit = roots_per_unit
        self.seed = seed
        self.shuffle = shuffle
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self._interior = InteriorCache(cfg.interior_roots.cache_dir)
        self._cutoff = (
            EgoRadiusCutoff(
                radius=radius_for_depth(cfg.depth, cfg.ego_radius),
                keep_root_component=cfg.ego_radius.keep_root_component,
            )
            if cfg.ego_radius.enabled
            else None
        )

    # ------------------------------------------------------------------ roots
    def _pool(self, data: Data) -> torch.Tensor:
        if not self.cfg.interior_roots.enabled:
            return torch.arange(int(data.num_nodes))
        idx = self._interior.get(data, self.cfg.interior_roots, n_hops=self.cfg.depth)
        return torch.as_tensor(idx, dtype=torch.long)

    def _n_roots(self, pool_size: int) -> int:
        k = self.roots_per_unit
        return pool_size if k is None else min(int(k), pool_size)

    def _generator(self, unit_id: str) -> torch.Generator | None:
        if self.seed is None:
            return None
        return torch.Generator().manual_seed(int(self.seed) * 7919 + zlib.crc32(unit_id.encode()))

    def roots(self, data: Data) -> torch.Tensor:
        """Root cells of one unit for this pass (sorted node indices)."""
        pool = self._pool(data)
        k = self._n_roots(int(pool.numel()))
        if k >= pool.numel():
            return pool
        gen = self._generator(str(data.unit_id))
        if self.cfg.root_weighting and getattr(data, "cell_type", None) is not None:
            types = data.cell_type[pool]
            counts = torch.bincount(types).float()
            weights = 1.0 / counts[types]
            if self.cfg.root_weighting == "sqrt_inverse":
                weights = weights.sqrt()
            chosen = torch.multinomial(weights, k, replacement=False, generator=gen)
        else:
            chosen = torch.randperm(int(pool.numel()), generator=gen)[:k]
        return pool[chosen.sort().values]

    # ---------------------------------------------------------------- batches
    def _batches(self, data: Data, roots: torch.Tensor, batch_size: int, shuffle: bool, **kw):
        return _EgoGraphBatches(
            data,
            roots,
            depth=self.cfg.depth,
            num_neighbors=self.cfg.num_neighbors,
            cutoff=self._cutoff,
            batch_size=batch_size,
            shuffle=shuffle,
            **kw,
        )

    def __len__(self) -> int:
        if self.mode == "bags":
            return len(self.dataset)
        if self.cfg.interior_roots.enabled:
            sizes = [int(self._pool(self.dataset[i]).numel()) for i in range(len(self.dataset))]
        else:
            sizes = [int(e.n_cells) for e in self.dataset.entries]
        n_roots = [self._n_roots(s) for s in sizes]
        if self.mode == "merged":
            return math.ceil(sum(n_roots) / self.cfg.batch_size)
        return sum(math.ceil(n / self.cfg.batch_size) for n in n_roots)

    def __iter__(self) -> Iterator[Batch]:
        if self.mode == "merged":
            yield from self._iter_merged()
            return
        order = torch.randperm(len(self.dataset)) if self.shuffle else torch.arange(len(self.dataset))
        for i in order.tolist():
            data = self.dataset[i]
            roots = self.roots(data)
            if roots.numel() == 0:
                continue
            size = int(roots.numel()) if self.mode == "bags" else self.cfg.batch_size
            yield from self._batches(data, roots, size, shuffle=False)

    def _iter_merged(self) -> Iterator[Batch]:
        units = [self.dataset[i] for i in range(len(self.dataset))]
        offsets = [0]
        for unit in units:
            offsets.append(offsets[-1] + int(unit.num_nodes))
        roots = torch.cat([self.roots(u) + offsets[k] for k, u in enumerate(units)])
        merged = Data(
            edge_index=torch.cat([u.edge_index + offsets[k] for k, u in enumerate(units)], dim=1),
            num_nodes=offsets[-1],
        )
        for key in ("x", "pos", "cell_type", "edge_attr"):
            values = [getattr(u, key, None) for u in units]
            if all(isinstance(v, torch.Tensor) for v in values):
                merged[key] = torch.cat(values)
        kw = {"num_workers": self.num_workers, "pin_memory": self.pin_memory}
        yield from self._batches(merged, roots, self.cfg.batch_size, shuffle=self.shuffle, **kw)
