from __future__ import annotations

from torch import nn


class _MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden = hidden_dim or input_dim
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(curr_dim, hidden), nn.LeakyReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GraphPredictionHead(_MLPHead):
    pass
