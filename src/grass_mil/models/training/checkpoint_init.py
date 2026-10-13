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
    # First-party Lightning checkpoint; see the note in data/components/datasets.py
    # on why weights_only must be set explicitly under torch>=2.6.
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
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
    if encoder_init_map == "spacegm_bgrl":
        return remap_spacegm_bgrl_keys(state_dict)
    if encoder_init_map != "auto_bgrl_or_identity":
        raise ValueError(
            "encoder_init_map must be one of ['identity', 'auto_bgrl_or_identity', 'spacegm_bgrl']"
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


def remap_spacegm_bgrl_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Import an encoder pretrained with the original SPACE-GM style BGRL code.

    Maps ``online_encoder.x_embedding`` -> ``encoder.input_proj.embedding``,
    ``online_encoder.feat_embedding`` -> ``encoder.input_proj.linear``,
    ``online_encoder.gnns.{i}.gine.{nn.0,nn.2,lin,eps}`` -> ``encoder.layers.{i}.*``
    and ``online_encoder.batch_norms.{i}`` -> ``encoder.norms.{i}``. The unused
    GIN MLP copies, edge-type embeddings, the target encoder and the predictor
    are dropped. The continuous input width must match the public model's
    (``input_proj.linear.weight`` has one column per continuous node feature);
    select columns beforehand if the legacy run carried masked extra columns.
    """
    import re

    out: Dict[str, torch.Tensor] = {}
    layer_re = re.compile(
        r"^online_encoder\.gnns\.(\d+)\.gine\.(nn\.0|nn\.2|lin|eps)(\.(weight|bias))?$"
    )
    norm_re = re.compile(
        r"^online_encoder\.batch_norms\.(\d+)\.(weight|bias|running_mean|running_var|num_batches_tracked)$"
    )
    for key, value in state_dict.items():
        if key == "online_encoder.x_embedding.weight":
            out["encoder.input_proj.embedding.weight"] = value
            continue
        if key in ("online_encoder.feat_embedding.weight", "online_encoder.feat_embedding.bias"):
            out[key.replace("online_encoder.feat_embedding", "encoder.input_proj.linear")] = value
            continue
        m = layer_re.match(key)
        if m:
            suffix = m.group(2) + (m.group(3) or "")
            out[f"encoder.layers.{m.group(1)}.{suffix}"] = value
            continue
        m = norm_re.match(key)
        if m:
            out[f"encoder.norms.{m.group(1)}.{m.group(2)}"] = value
            continue
        # gnns.{i}.mlp.* (unused duplicate), edge_embedding, target_encoder, predictor: dropped
    if not out:
        raise ValueError(
            "No SPACE-GM BGRL encoder keys (online_encoder.*) found in the checkpoint."
        )
    return out


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

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
