from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class ProcessedIndexEntry:
    path: str
    sample_id: str
    region_id: Optional[str]
    patch_id: str
    split: Optional[str] = None


def _migrate_legacy_attributes(data):
    """Upgrade graphs written by an older precompute.

    ``categorical_index`` was renamed to ``categorical_codes`` because
    PyTorch Geometric treats *any* attribute whose name contains ``index`` as an
    edge-index tensor and concatenates it along the last dimension, which
    corrupts node-level codes as soon as more than one graph is batched.
    Migrating on load means existing processed caches stay usable without a
    forced re-precompute.
    """
    legacy = getattr(data, "categorical_index", None)
    if legacy is not None and getattr(data, "categorical_codes", None) is None:
        data.categorical_codes = legacy
        del data.categorical_index
    return data


class SpatialOmicsGraphDataset(Dataset):
    def __init__(self, index_path: str | Path) -> None:
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Processed index not found: {index_path}")
        self.index_path = index_path
        data = json.loads(index_path.read_text())
        self.entries: List[ProcessedIndexEntry] = [
            ProcessedIndexEntry(**entry) for entry in data["entries"]
        ]
        self.label_maps: Dict[str, Dict[str, int]] = data.get("label_maps", {})
        self.label_maps_inv: Dict[str, Dict[int, str]] = {
            label: {idx: name for name, idx in mapping.items()}
            for label, mapping in self.label_maps.items()
        }
        self.graph_label_maps: Dict[str, Dict[str, int]] = data.get(
            "graph_label_maps", {}
        )
        self.reducer_state: Dict[str, Any] = data.get("reducer_state", {})

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> torch.nn.Module:
        entry = self.entries[idx]
        # weights_only=False is required: these are PyG `Data` objects written
        # by our own precompute step, not bare tensors. torch>=2.6 defaults the
        # flag to True, which cannot unpickle `DataEdgeAttr`/`GlobalStorage`
        # without allowlisting a set of PyG internals that changes between
        # releases. The file is first-party, produced by this pipeline.
        data = torch.load(entry.path, weights_only=False)
        return _migrate_legacy_attributes(data)


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transforms) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        for transform in self.transforms:
            data = transform(data)
        return data
