"""Loading encoder weights from a pretraining (or fine-tuning) checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["load_encoder_weights"]


def load_encoder_weights(encoder: torch.nn.Module, path: str | Path) -> None:
    """Copy the ``encoder.*`` weights of a checkpoint into ``encoder``.

    Both module types store their (online) encoder under ``encoder``, so a BGRL
    checkpoint and a fine-tuned checkpoint initialise an encoder the same way.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    weights = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    if not weights:
        raise ValueError(f"{path} holds no encoder weights (no 'encoder.*' keys).")
    encoder.load_state_dict(weights, strict=True)
