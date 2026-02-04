from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import importlib

import numpy as np


class FeatureReducer:
    def fit(self, features: np.ndarray) -> "FeatureReducer":
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return features

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def state_dict(self) -> Dict[str, object]:
        return {}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        _ = state


@dataclass
class FeatureReducerConfig:
    name: str
    kwargs: Dict[str, object]

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FeatureReducerConfig":
        return cls(
            name=data["name"],
            kwargs=data.get("kwargs", {}),
        )


_FEATURE_REDUCERS: Dict[str, Callable[..., FeatureReducer]] = {}


def register_feature_reducer(name: str):
    def decorator(cls: Callable[..., FeatureReducer]):
        _FEATURE_REDUCERS[name] = cls
        return cls

    return decorator


def get_feature_reducer(config: FeatureReducerConfig) -> FeatureReducer:
    if config.name in _FEATURE_REDUCERS:
        return _FEATURE_REDUCERS[config.name](**(config.kwargs or {}))
    if "." in config.name:
        module_name, class_name = config.name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**(config.kwargs or {}))
    raise ValueError(
        f"Unknown feature reducer '{config.name}'. Available: {list(_FEATURE_REDUCERS)}"
    )


@register_feature_reducer("identity")
class IdentityReducer(FeatureReducer):
    pass


@register_feature_reducer("pca")
class PcaReducer(FeatureReducer):
    def __init__(self, n_components: int = 64) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> "PcaReducer":
        mean = features.mean(axis=0, keepdims=True)
        centered = features - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        self.mean_ = mean
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA reducer has not been fitted.")
        centered = features - self.mean_
        return centered @ self.components_.T

    def state_dict(self) -> Dict[str, object]:
        return {
            "n_components": self.n_components,
            "components": self.components_,
            "mean": self.mean_,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.n_components = int(state["n_components"])
        self.components_ = np.asarray(state["components"])
        self.mean_ = np.asarray(state["mean"])
