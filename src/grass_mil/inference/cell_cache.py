"""Per-cell cache of a trained GRASS-MIL model: the input of niche discovery.

For every region and every cell (the ego-graph rooted at that cell) the cache
stores the encoder embedding ``E`` (fp16), the per-class instance logits
``LG``, the raw per-class attention logits ``AR`` and, once per region, the
head bias, cell types, positions, the cell graph and the region label. One
directory per model::

    <cache_dir>/<model>/index.json
    <cache_dir>/<model>/<region>.npz

``index.json`` records the model checkpoint, the region list with cell counts,
the head bias, the cell-type vocabulary and, when known, which regions were
held out for that model (nested CV). ``CellCache`` reads it back.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["CellCache", "extract_cell_cache", "region_npz_path"]

_INDEX = "index.json"


def region_npz_path(model_dir: Path, region_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(region_id))
    return model_dir / f"{safe}.npz"


@dataclass
class CellCache:
    """Reader for one model's cache directory."""

    model_dir: Path
    index: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, model_dir: str | Path) -> "CellCache":
        model_dir = Path(model_dir)
        index_path = model_dir / _INDEX
        if not index_path.exists():
            raise FileNotFoundError(f"No cell cache index at {index_path}")
        return cls(model_dir=model_dir, index=json.loads(index_path.read_text()))

    @property
    def regions(self) -> List[str]:
        return [r["region_id"] for r in self.index["regions"]]

    @property
    def held_out(self) -> List[str]:
        return list(self.index.get("held_out_regions") or [])

    @property
    def bias(self) -> np.ndarray:
        return np.asarray(self.index["head_bias"], dtype=np.float64)

    @property
    def cell_type_names(self) -> Optional[List[str]]:
        names = self.index.get("cell_type_names")
        return list(names) if names else None

    def label_of(self, region_id: str) -> Optional[float]:
        for r in self.index["regions"]:
            if r["region_id"] == region_id:
                return r.get("label")
        return None

    def n_cells(self, region_id: str) -> int:
        for r in self.index["regions"]:
            if r["region_id"] == region_id:
                return int(r["n_cells"])
        raise KeyError(region_id)

    def load(self, region_id: str, keys: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
        path = region_npz_path(self.model_dir, region_id)
        with np.load(path, allow_pickle=False) as z:
            if keys is None:
                return {k: z[k] for k in z.files}
            return {k: z[k] for k in keys if k in z.files}

    def iter_regions(self, regions: Optional[Sequence[str]] = None, keys=None):
        for rid in regions if regions is not None else self.regions:
            yield rid, self.load(rid, keys)


def _cell_type_codes(data) -> Optional[np.ndarray]:
    codes = getattr(data, "categorical_codes", None)
    if codes is None:
        return None
    codes = codes.detach().cpu()
    if codes.dim() == 2:
        codes = codes[:, 0]
    return codes.numpy().astype(np.int32)


def _roots_of(batch) -> np.ndarray:
    roots = getattr(batch, "root_n_id", None)
    if roots is None:
        raise ValueError("ShaDow batch without root_n_id; cannot align cells.")
    roots = roots.detach().cpu().long().view(-1)
    n_id = getattr(batch, "n_id", None)
    flag = getattr(batch, "root_n_id_is_global", None)
    is_global = False
    if flag is not None:
        f = torch.as_tensor(flag).view(-1)
        is_global = bool(f[0]) if f.numel() else False
    if not is_global and n_id is not None:
        roots = n_id.detach().cpu().long().view(-1)[roots]
    return roots.numpy()


@torch.no_grad()
def extract_cell_cache(
    model,
    dataset,
    *,
    out_dir: str | Path,
    model_name: str,
    sampler_config,
    ckpt_path: Optional[str] = None,
    held_out_regions: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    cell_type_names: Optional[Sequence[str]] = None,
    autocast: bool = True,
    device: Optional[torch.device | str] = None,
    resume: bool = True,
    log=None,
) -> Path:
    """Run the model over every cell of every region and write the cache."""
    from grass_mil.data.components.samplers import get_sampler_strategy

    model_dir = Path(out_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if device is not None else next(model.parameters()).device
    strategy = get_sampler_strategy(sampler_config)
    head_bias = _head_bias(model)
    wanted = set(str(r) for r in regions) if regions is not None else None
    index_regions: List[Dict[str, Any]] = []
    use_autocast = bool(autocast) and device.type == "cuda"
    t0 = time.time()
    for idx in range(len(dataset)):
        entry = dataset.entries[idx]
        rid = str(entry.region_id if entry.region_id is not None else entry.sample_id)
        if wanted is not None and rid not in wanted:
            continue
        npz = region_npz_path(model_dir, rid)
        data = dataset[idx]
        n = int(data.num_nodes)
        label = None
        if getattr(data, "graph_y", None) is not None:
            label = float(data.graph_y.view(-1)[0])
        if resume and npz.exists():
            index_regions.append({"region_id": rid, "n_cells": n, "label": label})
            continue
        E = np.zeros((n, int(model.encoder.output_dim)), dtype=np.float16)
        LG = None
        AR = None
        seen = np.zeros(n, dtype=bool)
        loader = strategy.build_dataset_loader(
            dataset=[data], batch_size=1, num_workers=0, pin_memory=False, shuffle=False
        )
        for batch in loader:
            roots = _roots_of(batch)
            batch = batch.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_autocast):
                logits, emb, _ = model._forward_instances(batch)
                attn = model.attention(emb)[0] if model.attention is not None else None
            emb = emb.float()
            if LG is None:
                LG = np.zeros((n, int(logits.shape[1])), dtype=np.float32)
                AR = (
                    np.zeros((n, int(attn.shape[1])), dtype=np.float32)
                    if attn is not None
                    else None
                )
            E[roots] = emb.half().cpu().numpy()
            LG[roots] = logits.float().cpu().numpy()
            if AR is not None:
                AR[roots] = attn.float().cpu().numpy()
            seen[roots] = True
        if not seen.all():
            raise RuntimeError(
                f"Region {rid}: {int((~seen).sum())} of {n} cells were not visited; "
                "the extraction sampler must keep every root (all-roots mode)."
            )
        payload: Dict[str, Any] = {
            "E": E,
            "LG": LG,
            "pos": data.pos.detach().cpu().numpy().astype(np.float32),
            "edge_index": data.edge_index.detach().cpu().numpy().astype(np.int64),
        }
        if AR is not None:
            payload["AR"] = AR
        ct = _cell_type_codes(data)
        if ct is not None:
            payload["ct"] = ct
        if getattr(data, "edge_attr", None) is not None:
            payload["edge_attr"] = data.edge_attr.detach().cpu().numpy().astype(np.float32)
        np.savez(npz, **payload)
        index_regions.append({"region_id": rid, "n_cells": n, "label": label})
        if log is not None and len(index_regions) % 25 == 0:
            log.info(
                f"[{model_name}] {len(index_regions)} regions ({(time.time() - t0) / 60:.1f} min)"
            )
    index = {
        "model_name": model_name,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "regions": index_regions,
        "held_out_regions": [str(r) for r in held_out_regions] if held_out_regions else [],
        "head_bias": head_bias,
        "cell_type_names": list(cell_type_names) if cell_type_names else None,
        "embedding_dim": int(model.encoder.output_dim),
        "sampler_runtime": _plain(getattr(sampler_config, "runtime", {}) or {}),
    }
    (model_dir / _INDEX).write_text(json.dumps(index, indent=1, default=str))
    return model_dir


def _plain(obj: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _head_bias(model) -> List[float]:
    head = getattr(model, "graph_head", None)
    if head is None:
        return []
    last = None
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    if last is None or last.bias is None:
        return []
    return [float(v) for v in last.bias.detach().cpu().view(-1)]
