from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SpatialOmicsTable:
    """Normalized in-memory representation of spatial omics inputs."""

    coords: np.ndarray  # (n_cells, 2) float, in micrometers
    molecular_features: Optional[np.ndarray]  # (n_cells, n_features) float
    categorical_labels: Dict[str, np.ndarray]  # {label_name: (n_cells,)}
    cell_ids: np.ndarray  # (n_cells,)
    sample_id: Optional[str] = None
    region_id: Optional[str] = None
