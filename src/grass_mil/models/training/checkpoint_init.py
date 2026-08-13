from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch


def load_state_dict_with_optional_mapping(
    module: torch.nn.Module,
    *,
    init_from_ckpt: Optional[str],
    init_strict: bool,
    encoder_init_map: str,
) -> Dict[str, List[str]]:
    if not init_from_ckpt:
        return {"missing_keys": [], "unexpected_keys": []}
    ckpt_path = Path(init_from_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {init_from_ckpt}")
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must resolve to a state_dict dictionary.")
    remapped = remap_encoder_keys(state_dict, encoder_init_map=encoder_init_map)
    missing, unexpected = module.load_state_dict(remapped, strict=init_strict)
    return {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def remap_encoder_keys(
    state_dict: Dict[str, torch.Tensor], *, encoder_init_map: str
) -> Dict[str, torch.Tensor]:
    if encoder_init_map == "identity":
        return dict(state_dict)
    if encoder_init_map != "auto_bgrl_or_identity":
        raise ValueError("encoder_init_map must be one of ['identity', 'auto_bgrl_or_identity']")
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("online_encoder."):
            new_key = f"encoder.{key[len('online_encoder.') :]}"
        elif key.startswith("ssl_model.online_encoder."):
            new_key = f"encoder.{key[len('ssl_model.online_encoder.') :]}"
        elif key.startswith("model.online_encoder."):
            new_key = f"encoder.{key[len('model.online_encoder.') :]}"
        remapped[new_key] = value
    return remapped


def read_graph_head_bias(
    checkpoint_path: str,
    *,
    head_prefix: str = "graph_head",
) -> Optional[List[float]]:
    """Read the graph head's output bias from a training checkpoint.

    Attribution subtracts this bias so the reported margin reflects the
    instance-driven part of the decision rather than the head's prior. The bias
    lives on the final ``Linear`` of the head MLP, which is the highest-numbered
    ``net.<i>.bias`` entry under ``head_prefix``.

    Returns:
        One value per class, or ``None`` when the checkpoint carries no head
        (an encoder-only checkpoint, for instance).
    """
    import re

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path!r} holds no state dict.")

    pattern = re.compile(rf"^{re.escape(head_prefix)}\.net\.(\d+)\.bias$")
    candidates = {}
    for key, value in state.items():
        match = pattern.match(str(key))
        if match is not None:
            candidates[int(match.group(1))] = value
    if not candidates:
        return None

    bias = candidates[max(candidates)]
    return [float(v) for v in bias.detach().cpu().reshape(-1)]
