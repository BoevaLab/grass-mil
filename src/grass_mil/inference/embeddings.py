from __future__ import annotations

import pandas as pd

from grass_mil.inference.io import (
    embeddings_to_dataframe as _embeddings_to_dataframe,
    node_embeddings_to_dataframe as _node_embeddings_to_dataframe,
)
from grass_mil.inference.schemas import EmbeddingPayload


def embedding_payload_to_table(payload: EmbeddingPayload) -> pd.DataFrame:
    """Public wrapper used by downstream analysis and tests."""
    return _embeddings_to_dataframe(payload)


def embeddings_to_dataframe(payload: EmbeddingPayload) -> pd.DataFrame:
    """Backwards-friendly alias for table conversion."""
    return _embeddings_to_dataframe(payload)


def node_embeddings_to_dataframe(payload: EmbeddingPayload) -> pd.DataFrame:
    return _node_embeddings_to_dataframe(payload)
