from __future__ import annotations

import copy

import torch
from torch import nn


class MLPPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class BGRL(nn.Module):
    """Bootstrap Graph Representation Learning wrapper."""

    def __init__(self, encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.online_encoder = encoder
        self.predictor = predictor
        self.target_encoder = copy.deepcopy(encoder)

        if hasattr(self.target_encoder, "reset_parameters"):
            self.target_encoder.reset_parameters()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, momentum: float):
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Momentum must be in [0,1], got {momentum}")
        for online_p, target_p in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)

    def forward(self, online_x, target_x):
        online_y = (
            self.online_encoder(*online_x)
            if isinstance(online_x, tuple)
            else self.online_encoder(online_x)
        )
        online_q = self.predictor(online_y)
        with torch.no_grad():
            target_y = (
                self.target_encoder(*target_x)
                if isinstance(target_x, tuple)
                else self.target_encoder(target_x)
            ).detach()
        return online_q, target_y
