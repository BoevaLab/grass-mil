"""Small helpers shared by the configuration dataclasses and modules."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

__all__ = ["checked", "plain"]


def checked(cls: type, cfg: Mapping[str, Any] | None, section: str) -> dict[str, Any]:
    """``cfg`` as a dict, after checking that every key is a field of the dataclass ``cls``."""
    out = dict(cfg or {})
    unknown = set(out) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")
    return out


def plain(value: Any) -> Any:
    """OmegaConf containers as plain Python containers (anything else unchanged)."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
