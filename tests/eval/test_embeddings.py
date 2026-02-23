from __future__ import annotations

import torch

from src.inference.embeddings import embeddings_to_dataframe, node_embeddings_to_dataframe
from src.inference.schemas import EmbeddingPayload


def test_embedding_payload_to_dataframes() -> None:
    payload = EmbeddingPayload(
        bag_ids=["r2", "r1"],
        graph_embeddings=torch.tensor([[2.0, 3.0], [1.0, 1.5]]),
        node_embeddings=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        node_bag_ids=["r1", "r2"],
    )
    graph_df = embeddings_to_dataframe(payload)
    node_df = node_embeddings_to_dataframe(payload)

    assert list(graph_df.columns) == ["bag_id", "graph_emb_0", "graph_emb_1"]
    assert graph_df.iloc[0]["bag_id"] == "r1"
    assert list(node_df.columns) == ["bag_id", "node_emb_0", "node_emb_1"]
