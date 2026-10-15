"""Per-cell outputs of a trained model: the input of niche discovery.

For every unit and every cell (the ego-graph rooted at it) the cache stores

* ``embedding``: the encoder embedding (fp16),
* ``attention_space``: the attention head's low-dimensional pre-tanh activation,
* ``instance_logits`` and ``attention_logits``: per-class logits and raw
  attention scores,

and once per unit the cell types, positions, cell ids and the cell graph::

    <cache_dir>/<model>/index.json
    <cache_dir>/<model>/<unit>.npz

``index.json`` records the checkpoint, the units with their labels and
patients, the units held out from the model's training (cross-validation), the
head bias and the cell-type names.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from grass_mil.data.samplers import EgoGraphLoader, SamplerConfig

__all__ = ["CellCache", "extract_cells", "sampler_of_checkpoint"]

log = logging.getLogger(__name__)

INDEX_NAME = "index.json"
ARRAYS = ("embedding", "attention_space", "instance_logits", "attention_logits")


def _npz_path(model_dir: Path, unit_id: str) -> Path:
    return model_dir / ("".join(c if c.isalnum() or c in "._-" else "_" for c in unit_id) + ".npz")


@dataclass
class CellCache:
    """Reader of one model's cache directory."""

    model_dir: Path
    index: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, model_dir: str | Path) -> CellCache:
        model_dir = Path(model_dir)
        path = model_dir / INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(f"No cell cache at {model_dir} (missing {INDEX_NAME}).")
        return cls(model_dir=model_dir, index=json.loads(path.read_text()))

    @property
    def name(self) -> str:
        return str(self.index["model_name"])

    @property
    def units(self) -> list[str]:
        return [u["unit_id"] for u in self.index["units"]]

    @property
    def held_out(self) -> list[str]:
        return list(self.index.get("held_out_units") or [])

    @property
    def cell_types(self) -> list[str]:
        return list(self.index["cell_types"])

    @property
    def head_bias(self) -> np.ndarray:
        return np.asarray(self.index["head_bias"], dtype=np.float64)

    def unit_info(self, unit_id: str) -> dict[str, Any]:
        for u in self.index["units"]:
            if u["unit_id"] == unit_id:
                return u
        raise KeyError(unit_id)

    def load(self, unit_id: str) -> dict[str, np.ndarray]:
        with np.load(_npz_path(self.model_dir, unit_id), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def __iter__(self) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
        for unit in self.units:
            yield unit, self.load(unit)


def sampler_of_checkpoint(ckpt: dict[str, Any], batch_size: int) -> SamplerConfig:
    """Ego-graph settings the model was trained with, scoring every cell."""
    cfg = dict(ckpt["datamodule_hyper_parameters"]["sampler"])
    cfg.update(
        roots_per_unit=None, batch_size=int(batch_size), merge_units=False, root_weighting=None
    )
    cfg["interior_roots"] = {**dict(cfg.get("interior_roots") or {}), "enabled": False}
    return SamplerConfig.from_dict(cfg)


class _OneUnit:
    def __init__(self, data, entry) -> None:
        self.data, self.entries = data, [entry]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return self.data


@torch.no_grad()
def extract_cells(
    model,
    dataset,
    *,
    sampler: SamplerConfig,
    out_dir: str | Path,
    model_name: str,
    ckpt_path: str,
    held_out: Sequence[str] = (),
    unit_meta: dict[str, dict[str, Any]] | None = None,
    units: Sequence[str] | None = None,
    device: torch.device,
    resume: bool = True,
) -> Path:
    """Score every cell of every unit of ``dataset`` and write the cache."""
    model_dir = Path(out_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    wanted = set(units) if units is not None else None
    index_units = []
    autocast = device.type == "cuda"
    for i, entry in enumerate(dataset.entries):
        if wanted is not None and entry.unit_id not in wanted:
            continue
        info = {
            "unit_id": entry.unit_id,
            "region_id": entry.region_id,
            "sample_id": entry.sample_id,
            "n_cells": entry.n_cells,
        }
        info.update((unit_meta or {}).get(entry.unit_id, {}))
        index_units.append(info)
        path = _npz_path(model_dir, entry.unit_id)
        if resume and path.exists():
            continue
        data = dataset[i]
        n = int(data.num_nodes)
        arrays: dict[str, np.ndarray] = {}
        seen = np.zeros(n, dtype=bool)
        for batch in EgoGraphLoader(
            _OneUnit(data, entry), sampler, mode="chunks", roots_per_unit=None
        ):
            roots = batch.root_cell.numpy()
            batch = batch.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=autocast):
                emb = model.encoder(batch)
                outputs = {
                    "embedding": emb,
                    "attention_space": model.attention.space(emb),
                    "instance_logits": model.head(emb),
                    "attention_logits": model.attention(emb),
                }
            for key, value in outputs.items():
                if key not in arrays:
                    dtype = np.float16 if key == "embedding" else np.float32
                    arrays[key] = np.zeros((n, int(value.shape[1])), dtype=dtype)
                arrays[key][roots] = value.float().cpu().numpy()
            seen[roots] = True
        if not seen.all():
            raise RuntimeError(
                f"Unit {entry.unit_id}: {int((~seen).sum())} cells were not scored."
            )
        arrays["pos"] = data.pos.numpy().astype(np.float32)
        arrays["edge_index"] = data.edge_index.numpy().astype(np.int64)
        arrays["cell_ids"] = np.asarray(dataset.load(i)["cell_ids"], dtype=str)
        if getattr(data, "cell_type", None) is not None:
            arrays["cell_type"] = data.cell_type.numpy().astype(np.int32)
        np.savez(path, **arrays)
        if len(index_units) % 25 == 0:
            log.info(f"[{model_name}] {len(index_units)} units")
    index = {
        "model_name": model_name,
        "ckpt_path": str(ckpt_path),
        "target_type": model.task["target_type"],
        "units": index_units,
        "held_out_units": sorted(map(str, held_out)),
        "head_bias": model.head.bias.detach().cpu().tolist(),
        "cell_types": list(model.input_spec.cell_types),
        "sampler": sampler.as_dict(),
    }
    (model_dir / INDEX_NAME).write_text(json.dumps(index, indent=1, default=str))
    return model_dir
