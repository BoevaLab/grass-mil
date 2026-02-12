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
    state_dict = (
        raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    )
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
        raise ValueError(
            "encoder_init_map must be one of ['identity', 'auto_bgrl_or_identity']"
        )
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
