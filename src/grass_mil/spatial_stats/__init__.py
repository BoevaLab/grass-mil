"""Classical spatial statistics on the cell graph, per region, compared between groups.

Neighbourhood enrichment z-scores (label permutations on the Delaunay graph),
cellular neighbourhoods (k-NN window composition clustered with k-means),
cell-cell contact fractions and Ripley cross-K, each summarised per region,
averaged per patient and compared between classes with Mann-Whitney + BH.
"""

from .cellular_neighbourhoods import cellular_neighbourhoods, window_composition
from .contacts import contact_fractions
from .cross_k import cross_k
from .nhood_enrichment import nhood_enrichment_zscores
from .tests import bh_adjust, mann_whitney_table, patient_mean

__all__ = [
    "bh_adjust",
    "cellular_neighbourhoods",
    "contact_fractions",
    "cross_k",
    "mann_whitney_table",
    "nhood_enrichment_zscores",
    "patient_mean",
    "window_composition",
]
