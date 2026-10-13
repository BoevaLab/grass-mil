"""A small synthetic cohort on disk, in the public data contract.

Regions are CSV cell tables (``cell_id, x, y, cell_type, area``) listed in a
manifest; labels live in a CSV with a region id, a target and a patient column;
outer folds are a JSON list of region-id lists grouped by patient. Positive
regions carry a planted niche: cell type ``D`` forms a compact cluster next to
``C`` cells, while in negatives ``D`` is scattered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

CELL_TYPES = ["A", "B", "C", "D"]


@dataclass
class SyntheticCohort:
    root: Path
    manifest_path: Path
    label_path: Path
    folds_path: Path
    processed_dir: Path
    region_ids: List[str]
    patient_of: Dict[str, str]
    label_of: Dict[str, int]
    folds: List[List[str]] = field(default_factory=list)


def _region_cells(rng: np.random.Generator, n_cells: int, positive: bool, extent: float):
    xy = rng.uniform(0.0, extent, size=(n_cells, 2))
    types = rng.choice(CELL_TYPES[:3], size=n_cells, p=[0.45, 0.35, 0.20])
    n_d = max(8, n_cells // 10)
    d_idx = rng.choice(n_cells, size=n_d, replace=False)
    types[d_idx] = "D"
    if positive:
        # Planted niche: D cells packed around a C cell.
        c_idx = np.where(types == "C")[0]
        centre = xy[rng.choice(c_idx)] if c_idx.size else xy[0]
        xy[d_idx] = centre + rng.normal(0.0, extent * 0.04, size=(n_d, 2))
        xy = np.clip(xy, 0.0, extent)
    area = rng.lognormal(mean=4.0, sigma=0.3, size=n_cells)
    return pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(n_cells)],
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cell_type": types,
            "area": area,
        }
    )


def make_synthetic_cohort(
    root: str | Path,
    *,
    n_patients: int = 6,
    regions_per_patient: int = 2,
    n_cells: int = 150,
    n_outer_folds: int = 3,
    extent: float = 600.0,
    seed: int = 0,
) -> SyntheticCohort:
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    label_rows = []
    region_ids: List[str] = []
    patient_of: Dict[str, str] = {}
    label_of: Dict[str, int] = {}
    for p in range(n_patients):
        patient = f"P{p:02d}"
        label = int(p % 2 == 1)
        for r in range(regions_per_patient):
            rid = f"{patient}_core{r}"
            df = _region_cells(rng, n_cells, positive=bool(label), extent=extent)
            path = raw / f"{rid}.csv"
            df.to_csv(path, index=False)
            rows.append(
                {"sample_id": rid, "input_path": str(path), "input_type": "csv", "region_id": rid}
            )
            label_rows.append({"REGION_ID": rid, "STAS": label, "PATIENT": patient})
            region_ids.append(rid)
            patient_of[rid] = patient
            label_of[rid] = label
    manifest_path = raw / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    label_path = raw / "labels.csv"
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    # Patient-grouped outer folds: consecutive (negative, positive) patient pairs
    # are dealt round-robin, so every fold holds both classes.
    patients = sorted(set(patient_of.values()))
    folds: List[List[str]] = [[] for _ in range(n_outer_folds)]
    for i, patient in enumerate(patients):
        folds[(i // 2) % n_outer_folds].extend(r for r in region_ids if patient_of[r] == patient)
    folds_path = raw / "folds.json"
    folds_path.write_text(json.dumps(folds))

    return SyntheticCohort(
        root=root,
        manifest_path=manifest_path,
        label_path=label_path,
        folds_path=folds_path,
        processed_dir=root / "processed",
        region_ids=region_ids,
        patient_of=patient_of,
        label_of=label_of,
        folds=folds,
    )


def cohort_overrides(cohort: SyntheticCohort, output_dir: str | Path) -> List[str]:
    """Hydra overrides that point the public configs at the synthetic cohort."""
    return [
        "regime=mil",
        f"paths.output_dir={output_dir}",
        f"paths.root_dir={cohort.root}",
        f"data.raw_manifest_path={cohort.manifest_path}",
        f"data.processed_dir={cohort.processed_dir}",
        "data.sample_unit=full",
        "data.min_cells=5",
        "data.csv.categorical_label_columns=[cell_type]",
        "data.csv.molecular_columns=[area]",
        "data.graph_builder.kwargs.edge_features=[log_distance,neighbor]",
        "data.graph_builder.kwargs.neighbor_cutoff_um=55.0",
        "data.split.split_by=region",
        "data.split.train_val_test_split=[0.6,0.2,0.2]",
        f"data.graph_labels.label_file={cohort.label_path}",
        "data.graph_labels.id_column=REGION_ID",
        "data.graph_labels.tasks=[STAS]",
        "data.graph_labels.patient_column=PATIENT",
        "data.sampler.runtime.depth=2",
        "data.sampler.runtime.subsample_count=24",
        "data.val_sampler.runtime.subsample_count=32",
        "model.encoder.hidden_dim=16",
        "model.encoder.out_dim=16",
        "model.encoder.num_layers=2",
        "model.attention.projection_dim=8",
        "model.attention.hidden_dim=4",
        "task.region_accumulation.hyperbatch_size=4",
        "trainer.max_epochs=2",
        "trainer.min_epochs=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.precision=32",
        "+trainer.enable_progress_bar=false",
        "~callbacks.rich_progress_bar",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        "seed=0",
    ]


def compose_cfg(config_name: str, overrides: Sequence[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="grass_mil.configs"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    GlobalHydra.instance().clear()
    if "trainer" in cfg:
        with open_dict(cfg):
            cfg.trainer.default_root_dir = cfg.paths.output_dir
    return cfg
