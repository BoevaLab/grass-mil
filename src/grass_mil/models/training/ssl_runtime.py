from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CosineWarmup:
    base_value: float
    warmup_steps: int
    total_steps: int
    min_value: float = 0.0

    def value(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.base_value
        step = max(0, min(step, self.total_steps))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_value * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_value + (self.base_value - self.min_value) * cosine
