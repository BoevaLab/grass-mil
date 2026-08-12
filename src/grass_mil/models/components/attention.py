from __future__ import annotations

from torch import nn


class AttnNetGated(nn.Module):
    """Gated MIL attention network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: bool = False,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        a_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        b_layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            a_layers.append(nn.Dropout(0.25))
            b_layers.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        logits = self.attention_c(a * b)
        return logits, x


class AttnNetGatedProjected(nn.Module):
    """Projected variant of gated MIL attention network."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 64,
        hidden_dim: int = 16,
        dropout: bool = False,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        projection = [nn.Linear(input_dim, projection_dim), nn.LeakyReLU()]
        attn_a = [nn.Linear(projection_dim, hidden_dim), nn.Tanh()]
        attn_b = [nn.Linear(projection_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            projection.append(nn.Dropout(0.25))
            attn_a.append(nn.Dropout(0.25))
            attn_b.append(nn.Dropout(0.25))
        self.projection = nn.Sequential(*projection)
        self.attention_a = nn.Sequential(*attn_a)
        self.attention_b = nn.Sequential(*attn_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = self.projection(x)
        logits = self.attention_c(self.attention_a(h) * self.attention_b(h))
        return logits, x
