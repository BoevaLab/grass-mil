from __future__ import annotations

from typing import Dict, List

from src.interpretability.plugins.base import InterpretabilityPlugin

_REGISTRY: Dict[str, InterpretabilityPlugin] = {}


def register_plugin(plugin: InterpretabilityPlugin) -> None:
    name = str(plugin.name).strip()
    if not name:
        raise ValueError("Plugin name must be non-empty.")
    _REGISTRY[name] = plugin


def get_plugin(name: str) -> InterpretabilityPlugin:
    key = str(name).strip()
    if key not in _REGISTRY:
        raise KeyError(f"Plugin {name!r} is not registered.")
    return _REGISTRY[key]


def list_plugins() -> List[str]:
    return sorted(_REGISTRY.keys())
