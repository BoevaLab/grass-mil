"""Configuration tree and packaging."""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from tests.helpers.cohort import compose_cfg

CONFIG_DIR = Path(__file__).parents[1] / "src" / "grass_mil" / "configs"
REGIMES = ["finetune", "finetune_reg", "attention_only", "mlp_head", "mlp_head_frozen", "pretrain"]
COMMANDS = {
    "grass-mil-train": "grass_mil.train:main",
    "grass-mil-nested-cv": "grass_mil.nested_cv:main",
    "grass-mil-predict": "grass_mil.predict:main",
    "grass-mil-extract": "grass_mil.extract:main",
    "grass-mil-niches": "grass_mil.niches.cli:main",
    "grass-mil-metaclusters": "grass_mil.niches.replication.cli:main",
    "grass-mil-spatial-stats": "grass_mil.spatial.cli:main",
}


def test_finetune_is_the_default_regime():
    cfg = compose_cfg("train", [])
    assert cfg.regime_name == "finetune"
    assert cfg.model._target_ == "grass_mil.models.SupervisedModule"
    assert cfg.finetune.head_init == "probe" and cfg.model.head.num_layers == 1
    assert cfg.finetune.lr_scales == {"encoder": 1.0, "head": 0.01, "attention": 2.0}


@pytest.mark.parametrize("regime", REGIMES)
def test_every_regime_composes(regime):
    cfg = compose_cfg("train", [f"regime={regime}"])
    assert cfg.regime_name == regime
    if regime == "pretrain":
        assert cfg.model._target_ == "grass_mil.models.BGRLModule"
        assert cfg.data.sampler.merge_units and not cfg.data.labelled_only
    else:
        assert cfg.callbacks.model_checkpoint.monitor == "val/auroc"


@pytest.mark.parametrize(
    "task,metric",
    [("classification", "val/auroc"), ("regression", "val/rmse"), ("survival", "val/c_index")],
)
def test_selection_follows_the_task(task, metric):
    cfg = compose_cfg("train", [f"task={task}", "task.target_columns=[a,b]"])
    assert cfg.callbacks.early_stopping.monitor == metric


@pytest.mark.parametrize(
    "name", ["train", "nested_cv", "predict", "extract", "niches", "metaclusters", "spatial_stats"]
)
def test_every_root_config_composes(name):
    compose_cfg(name, [])


def test_entry_points_match_the_packaged_commands():
    found = {
        ep.name: ep.value
        for ep in entry_points(group="console_scripts")
        if ep.name.startswith("grass-mil")
    }
    assert found == COMMANDS
    for target in COMMANDS.values():
        module, func = target.split(":")
        assert callable(getattr(importlib.import_module(module), func))


def test_cli_runs_from_another_directory(tmp_path):
    command = shutil.which("grass-mil-train")
    if command is None:
        pytest.skip("the package is not installed")
    # mkl-service sets MKL_THREADING_LAYER in this process; a fresh one picks its own.
    env = {k: v for k, v in os.environ.items() if k != "MKL_THREADING_LAYER"}
    result = subprocess.run(
        [command, "--help"], cwd=tmp_path, capture_output=True, text=True, env=env
    )
    assert result.returncode == 0 and "regime" in result.stdout


def test_configs_describe_the_method_not_its_history():
    forbidden = re.compile(
        r"preprint|space-?gm|\ba3\b|probehead|ftall|fidelity|legacy", re.IGNORECASE
    )
    for path in CONFIG_DIR.rglob("*.yaml"):
        assert not forbidden.search(path.read_text()), path
