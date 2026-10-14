from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from grass_mil.contracts import InterpretabilityDataset, PluginResult


@dataclass
class PluginContext:
    state: Dict[str, Any] = field(default_factory=dict)


class InterpretabilityPlugin(Protocol):
    name: str

    def required_inputs(self) -> List[str]: ...

    def run(
        self, dataset: InterpretabilityDataset, context: PluginContext, **params: Any
    ) -> PluginResult: ...
