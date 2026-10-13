"""Niche discovery on a trained GRASS-MIL model's per-cell cache."""

from .colocalisation import cluster_colocalisation
from .margins import attention_space_from_checkpoint, relattn_margins
from .pipeline import NicheResult, run_niche_discovery
from .tables import niche_table, spearman_concordance

__all__ = [
    "NicheResult",
    "attention_space_from_checkpoint",
    "cluster_colocalisation",
    "niche_table",
    "relattn_margins",
    "run_niche_discovery",
    "spearman_concordance",
]
