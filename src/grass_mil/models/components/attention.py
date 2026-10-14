from __future__ import annotations

from typing import Literal

import torch
from torch import nn

InitType = Literal["default", "kaiming"]


def _init_linears(module: nn.Module, init: InitType) -> None:
    """Apply the requested initialisation to every ``nn.Linear`` in ``module``.

    ``kaiming`` is the reference GRASS-MIL initialisation (``kaiming_uniform_`` on
    the weights, biases left at the torch default); ``default`` keeps torch's own.
    """
    if init == "default":
        return
    if init != "kaiming":
        raise ValueError(f"init must be 'default' or 'kaiming', got {init!r}.")
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight)


class AttnNetGated(nn.Module):
    """Gated MIL attention network (Ilse et al. 2018), one attention channel per class.

    ``forward`` returns the raw (pre-softmax) attention logits of shape
    ``[num_instances, n_classes]``; the softmax over instances is applied by the
    bag aggregation so that it can be taken within each bag.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: bool = False,
        n_classes: int = 1,
        init: InitType = "kaiming",
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
        self.init = init
        _init_linears(self, init)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        logits = self.attention_c(a * b)
        return logits, x

    def attention_space(self, x: torch.Tensor) -> torch.Tensor:
        """The ``hidden_dim``-d pre-tanh activation, the attention head's own feature space."""
        return self.attention_a[0](x)

    def head_spaces(self, x):
        """Intermediate activations of the attention head.

        ``a_pre_tanh`` is the argument of the tanh gate, before saturation. It
        is a low-dimensional space directly upstream of the attention weights;
        clustering the saturated output collapses the extremes together.
        """
        pre_tanh = self.attention_space(x)
        a = self.attention_a(x)
        b = self.attention_b(x)
        return {"a_pre_tanh": pre_tanh, "gate_product": a * b}


class AttnNetGatedProjected(nn.Module):
    """Projected gated MIL attention: ``Linear + LeakyReLU`` projection, then the gate.

    GRASS-MIL default: ``projection_dim=32``, ``hidden_dim=8``, dropout 0.25 in
    training, one channel per class, kaiming-initialised linears.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 32,
        hidden_dim: int = 8,
        dropout: bool = True,
        n_classes: int = 1,
        init: InitType = "kaiming",
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
        self.init = init
        _init_linears(self, init)

    def forward(self, x):
        h = self.projection(x)
        logits = self.attention_c(self.attention_a(h) * self.attention_b(h))
        return logits, x

    def attention_space(self, x: torch.Tensor) -> torch.Tensor:
        """The ``hidden_dim``-d pre-tanh activation ``V · LeakyReLU(W_proj x)``.

        Computed without dropout (``projection[0]`` and ``attention_a[0]`` only), so
        it is deterministic in eval and train mode alike.
        """
        h = self.projection[1](self.projection[0](x))
        return self.attention_a[0](h)

    def head_spaces(self, x):
        """Intermediate activations of the attention head.

        ``a_pre_tanh`` is ``V * LeakyReLU(W_proj x)``, the argument of the tanh
        gate: low-dimensional, directly upstream of the attention weights, and
        not yet collapsed by tanh saturation.
        """
        h = self.projection(x)
        pre_tanh = self.attention_a[0](h)
        return {
            "a_pre_tanh": pre_tanh,
            "gate_product": self.attention_a(h) * self.attention_b(h),
        }
