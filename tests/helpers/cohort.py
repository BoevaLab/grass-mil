"""A small synthetic cohort in the public data contract, and config helpers.

Regions are CSV cell tables (``cell_id, x, y, cell_type, area``) listed in a
manifest; labels live in a CSV with ``region_id``, ``label``, ``patient_id``,
``time`` and ``event``; outer folds are a JSON list of region-id lists grouped
by patient. Positive regions carry a planted niche: cell type ``D`` packed
around a ``C`` cell; in negative regions ``D`` is scattered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

CELL_TYPES = ["A", "B", "C", "D"]


@dataclass
class Cohort:
    root: Path
    manifest: Path
    labels: Path
    folds: Path
    regions: list[str]
    label_of: dict[str, int]
    patient_of: dict[str, str]


def _cells(rng: np.random.Generator, n: int, positive: bool, extent: float) -> pd.DataFrame:
    xy = rng.uniform(0.0, extent, size=(n, 2))
    types = rng.choice(CELL_TYPES[:3], size=n, p=[0.45, 0.35, 0.20])
    d = rng.choice(n, size=max(8, n // 10), replace=False)
    types[d] = "D"
    if positive:
        centre = xy[rng.choice(np.flatnonzero(types == "C"))]
        xy[d] = np.clip(centre + rng.normal(0.0, extent * 0.04, size=(d.size, 2)), 0.0, extent)
    return pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(n)],
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cell_type": types,
            "area": rng.lognormal(4.0, 0.3, size=n),
        }
    )


def make_cohort(
    root: Path,
    *,
    n_patients: int = 6,
    regions_per_patient: int = 2,
    n_cells: int = 120,
    n_folds: int = 3,
    seed: int = 0,
) -> Cohort:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest, labels = [], []
    label_of, patient_of = {}, {}
    for p in range(n_patients):
        patient, label = f"P{p:02d}", p % 2
        for r in range(regions_per_patient):
            region = f"{patient}_core{r}"
            _cells(rng, n_cells, bool(label), 600.0).to_csv(raw / f"{region}.csv", index=False)
            manifest.append(
                {"sample_id": region, "input_path": f"{region}.csv", "region_id": region}
            )
            labels.append(
                {
                    "region_id": region,
                    "label": label,
                    "patient_id": patient,
                    "time": float(rng.exponential(10.0 if label else 30.0)),
                    "event": int(rng.random() < 0.8),
                    "score": float(label + rng.normal(0, 0.3)),
                }
            )
            label_of[region], patient_of[region] = label, patient
    pd.DataFrame(manifest).to_csv(raw / "manifest.csv", index=False)
    pd.DataFrame(labels).to_csv(raw / "labels.csv", index=False)
    patients = sorted(set(patient_of.values()))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, patient in enumerate(patients):
        folds[(i // 2) % n_folds].extend(r for r in label_of if patient_of[r] == patient)
    (raw / "folds.json").write_text(json.dumps(folds))
    return Cohort(
        root,
        raw / "manifest.csv",
        raw / "labels.csv",
        raw / "folds.json",
        list(label_of),
        label_of,
        patient_of,
    )


def cohort_overrides(
    cohort: Cohort, processed: str = "processed", *, finetune: bool = True
) -> list[str]:
    """Overrides pointing the packaged configs at the cohort, with a tiny model."""
    small = ["finetune.hyperbatch_size=3", "finetune.probe.roots_per_unit=16"] if finetune else []
    return small + [
        f"data.manifest_path={cohort.manifest}",
        f"data.processed_dir={cohort.root / processed}",
        "data.input.cell_id_column=cell_id",
        "data.input.features=[area]",
        "data.graph.neighbor_cutoff=55",
        f"data.labels.file={cohort.labels}",
        "data.labels.patient_column=patient_id",
        "task.target_columns=[label]",
        "data.sampler.depth=2",
        "data.sampler.roots_per_unit=24",
        "data.sampler.batch_size=16",
        "data.val_roots_per_unit=32",
        "model.encoder.num_layers=2",
        "model.encoder.hidden_dim=16",
        "model.encoder.out_dim=16",
        "trainer.max_epochs=2",
        "trainer.min_epochs=1",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "~callbacks.rich_progress_bar",
        "~callbacks.model_summary",
    ]


def compose_cfg(name: str, overrides: Sequence[str], output_dir: Path | None = None) -> DictConfig:
    """Compose a packaged config as its entry point would (Hydra runtime set)."""
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="grass_mil.configs"):
        cfg = compose(config_name=name, overrides=list(overrides), return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    if output_dir is not None:
        with open_dict(cfg):
            cfg.paths.output_dir = str(output_dir)
            if "trainer" in cfg:
                cfg.trainer.default_root_dir = str(output_dir)
    return cfg
