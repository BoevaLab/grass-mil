"""Niche discovery and description on the per-cell outputs of a trained model."""

from .analysis import NicheAnalysis, analyse_niches, write_niche_outputs
from .attribution import cell_margins
from .cells import Cells, load_cells
from .discovery import ClusteringConfig, NicheModel, fit_niches

__all__ = [
    "Cells",
    "ClusteringConfig",
    "NicheAnalysis",
    "NicheModel",
    "analyse_niches",
    "cell_margins",
    "fit_niches",
    "load_cells",
    "write_niche_outputs",
]
