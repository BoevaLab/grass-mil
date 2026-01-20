from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch


def instantiate_transforms(configs: Optional[Iterable[Dict[str, Any]]]) -> List[Callable]:
    if not configs:
        return []
    transforms: List[Callable] = []
    for cfg in configs:
        name = cfg.get("name")
        kwargs = cfg.get("kwargs", {})
        if not name:
            continue
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        transforms.append(cls(**kwargs))
    return transforms


class CompositionVector:
    """Convert node labels into a composition vector (graph-level)."""

    def __init__(
        self,
        label_attr: str = "label_cell_type",
        num_classes: Optional[int] = None,
        output_attr: str = "x",
        normalize: bool = True,
        keep_original: bool = False,
    ) -> None:
        self.label_attr = label_attr
        self.num_classes = num_classes
        self.output_attr = output_attr
        self.normalize = normalize
        self.keep_original = keep_original

    def __call__(self, data):
        if not hasattr(data, self.label_attr):
            raise AttributeError(f"Missing label attribute '{self.label_attr}' for composition vector")
        labels = getattr(data, self.label_attr)
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)

        labels_np = labels_np.reshape(-1)
        num_classes = self.num_classes or (int(labels_np.max()) + 1 if labels_np.size > 0 else 0)
        counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        if self.normalize and counts.sum() > 0:
            counts = counts / counts.sum()

        vec = torch.from_numpy(counts).view(1, -1)
        if self.keep_original:
            data.x_original = data.x
        setattr(data, self.output_attr, vec)
        return data
