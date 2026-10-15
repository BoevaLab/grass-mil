"""The cells of selected units of a cell cache, concatenated, with their labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from grass_mil.inference.cell_cache import CellCache

__all__ = ["Cells", "load_cells", "resolve_units"]

SPACES = {"attention": "attention_space", "embedding": "embedding"}


@dataclass
class Cells:
    units: list[str]
    unit_of_cell: np.ndarray  # (N,) unit id of every cell
    space: dict[str, np.ndarray]  # the loaded spaces: "attention" / "embedding" -> (N, d)
    instance_logits: np.ndarray  # (N, C)
    attention_logits: np.ndarray  # (N, C)
    reference_logits: np.ndarray  # (C,) the model's mean instance logit
    cell_type: np.ndarray  # (N,) int (-1 without cell types)
    cell_type_names: list[str]
    pos: np.ndarray  # (N, 2)
    cell_ids: np.ndarray  # (N,)
    edges: np.ndarray  # (2, M) undirected, global cell indices
    target_type: str
    targets: dict[str, list[float]] = field(default_factory=dict)  # unit -> targets
    patients: dict[str, str] = field(default_factory=dict)  # unit -> patient
    slices: dict[str, slice] = field(default_factory=dict)  # unit -> its block of cells
    edge_slices: dict[str, slice] = field(default_factory=dict)  # unit -> its block of edges

    @property
    def n(self) -> int:
        return int(self.unit_of_cell.size)

    def unit_graph(self, unit: str) -> tuple[slice, np.ndarray]:
        """The unit's block of cells and its edges in local cell indices."""
        block = self.slices[unit]
        return block, self.edges[:, self.edge_slices[unit]] - block.start

    def patient_of(self, unit: str) -> str:
        """The unit's patient (the unit itself without patient information)."""
        return self.patients.get(unit, unit)

    def unit_class(self, unit: str) -> int:
        return int(self.targets[unit][0])

    def positive(self, positive_class: int) -> dict[str, bool]:
        """Which units count as positive (class ``positive_class``; event for survival)."""
        if self.target_type == "classification":
            return {u: self.unit_class(u) == positive_class for u in self.targets}
        if self.target_type == "survival":
            return {u: bool(self.targets[u][1] > 0) for u in self.targets}
        median = float(np.median([t[0] for t in self.targets.values()]))
        return {u: t[0] > median for u, t in self.targets.items()}


def resolve_units(selection, cache: CellCache) -> list[str]:
    """``held_out`` (the units the model did not train on), ``all``, or explicit ids."""
    from grass_mil.data.splits import read_ids

    if selection in (None, "held_out"):
        if not cache.held_out:
            raise ValueError(
                f"{cache.model_dir} records no held-out units; pass units=all or a list of unit ids."
            )
        return sorted(cache.held_out)
    if selection == "all":
        return sorted(cache.units)
    return sorted(read_ids(selection))


def load_cells(cache: CellCache, units: Sequence[str], spaces: Sequence[str] = ()) -> Cells:
    """The cells of ``units`` (in that order) with the requested ``spaces``."""
    unknown = set(spaces) - set(SPACES)
    if unknown:
        raise ValueError(f"Unknown cell spaces {sorted(unknown)}; expected {list(SPACES)}.")
    keys = {
        "logits": "instance_logits",
        "scores": "attention_logits",
        "pos": "pos",
        "ids": "cell_ids",
        **{s: SPACES[s] for s in spaces},
    }
    parts: dict[str, list] = {k: [] for k in [*keys, "ct", "edges", "unit"]}
    offset = 0
    targets, patients, slices, edge_slices = {}, {}, {}, {}
    n_edges = 0
    for unit in units:
        info = cache.unit_info(unit)
        d = cache.load(unit)
        n = int(d["pos"].shape[0])
        for key, name in keys.items():
            parts[key].append(d[name])
        parts["ct"].append(d["cell_type"] if "cell_type" in d else np.full(n, -1, dtype=np.int32))
        e = d["edge_index"]
        e = e[:, e[0] < e[1]]
        parts["edges"].append(e + offset)
        edge_slices[unit] = slice(n_edges, n_edges + e.shape[1])
        n_edges += e.shape[1]
        parts["unit"].append(np.full(n, unit, dtype=object))
        if info.get("targets") is not None:
            targets[unit] = list(info["targets"])
        if info.get("patient_id") is not None:
            patients[unit] = str(info["patient_id"])
        slices[unit] = slice(offset, offset + n)
        offset += n
    return Cells(
        units=list(units),
        unit_of_cell=np.concatenate(parts["unit"]),
        space={s: np.concatenate(parts[s]).astype(np.float32) for s in spaces},
        instance_logits=np.concatenate(parts["logits"]),
        attention_logits=np.concatenate(parts["scores"]),
        reference_logits=cache.reference_logits,
        cell_type=np.concatenate(parts["ct"]).astype(int),
        cell_type_names=cache.cell_types,
        pos=np.concatenate(parts["pos"]),
        cell_ids=np.concatenate(parts["ids"]),
        edges=np.concatenate(parts["edges"], axis=1),
        target_type=str(cache.index.get("target_type", "classification")),
        targets=targets,
        patients=patients,
        slices=slices,
        edge_slices=edge_slices,
    )
