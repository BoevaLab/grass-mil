from __future__ import annotations

from typing import Dict, Iterable, List

from grass_mil.interpretability.plugins.base import InterpretabilityPlugin


class PluginRegistry:
    def __init__(self, plugins: Iterable[InterpretabilityPlugin] | None = None) -> None:
        self._plugins: Dict[str, InterpretabilityPlugin] = {}
        if plugins is not None:
            for plugin in plugins:
                self.register(plugin)

    def register(self, plugin: InterpretabilityPlugin, *, replace: bool = False) -> None:
        name = str(plugin.name).strip()
        if not name:
            raise ValueError("Plugin name must be non-empty.")
        existing = self._plugins.get(name)
        if existing is not None and not replace:
            raise ValueError(
                f"Plugin {name!r} is already registered. "
                "Use replace=True to intentionally overwrite it."
            )
        self._plugins[name] = plugin

    def get(self, name: str) -> InterpretabilityPlugin:
        key = str(name).strip()
        if key not in self._plugins:
            raise KeyError(f"Plugin {name!r} is not registered.")
        return self._plugins[key]

    def list(self) -> List[str]:
        return sorted(self._plugins.keys())


def create_plugin_registry(
    plugins: Iterable[InterpretabilityPlugin] | None = None,
) -> PluginRegistry:
    return PluginRegistry(plugins=plugins)
