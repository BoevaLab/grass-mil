"""Guards on the installable-package contract.

Configs ship inside ``grass_mil`` so Hydra resolves them identically from a
source checkout and from an installed wheel. These tests fail loudly if a move
or a rename breaks that, which is otherwise only discovered when someone
installs the package.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

ROOT_CONFIGS = [
    "train.yaml",
    "eval.yaml",
    "loocv.yaml",
    "inference/predict.yaml",
    "interpretability/report.yaml",
]


def _pyproject() -> dict:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_package_is_importable_under_its_distribution_name() -> None:
    module = importlib.import_module("grass_mil")
    assert Path(module.__file__).parent.name == "grass_mil"


def test_configs_ship_inside_the_package() -> None:
    configs = importlib.import_module("grass_mil.configs")
    configs_dir = Path(configs.__file__).parent
    for name in ROOT_CONFIGS:
        assert (configs_dir / name).is_file(), f"missing packaged config: {name}"


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_root_configs_compose_from_the_config_module(config_name: str) -> None:
    """Catches a broken ``config_path`` after a layout change."""
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="grass_mil.configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=[])
        assert cfg is not None
    GlobalHydra.instance().clear()


def test_console_script_targets_are_importable_and_callable() -> None:
    scripts = _pyproject()["project"]["scripts"]
    assert scripts, "no console scripts declared"
    for name, target in scripts.items():
        module_path, _, attribute = target.partition(":")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, attribute)), f"{name} -> {target} is not callable"


def test_wheel_packages_the_source_tree() -> None:
    config = _pyproject()
    assert config["build-system"]["build-backend"] == "hatchling.build"
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/grass_mil"]
