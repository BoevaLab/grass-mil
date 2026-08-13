"""Compose task presets through Hydra the way the CLI does.

The rest of the training tests build their config by loading YAML files directly
and assigning the nodes onto a base config. That is convenient, but it bypasses
Hydra's defaults-list resolution entirely, so a task preset can declare an
override that never takes effect and every test still passes.

Two real bugs reached the docs that way: ``task=finetune_mil`` silently composed
a single-logit head with a single attention channel, and ``task=pretrain_bgrl``
failed to compose at all. Both are invisible unless the config is composed the
way the entrypoints compose it.
"""

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

CONFIG_MODULE = "grass_mil.configs"
ROOT_CONFIGS = ["train", "eval"]
SUPERVISED_TASKS = ["finetune_mean", "finetune_mil", "finetune_survival"]


def _compose(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module=CONFIG_MODULE):
        cfg = compose(config_name=config_name, overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
@pytest.mark.parametrize("task_name", SUPERVISED_TASKS)
def test_supervised_task_presets_compose(config_name: str, task_name: str) -> None:
    cfg = _compose(config_name, [f"task={task_name}"])
    assert cfg.task.name == task_name


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
@pytest.mark.parametrize("task_name", SUPERVISED_TASKS)
def test_attention_width_matches_head_width_as_composed(config_name: str, task_name: str) -> None:
    """The contract must hold on the *composed* config, not just in code.

    ``validate_attention_width`` runs at build time, but it can only see what
    composition produced. A preset whose widths are silently discarded still
    passes it, because one channel for one class is internally consistent.
    """
    # Imported lazily on purpose. This module is collected early, and importing
    # grass_mil at module scope pulls in numpy/MKL before torch spawns the
    # ddp_sim subprocesses, which then die with
    # "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1".
    from grass_mil.contracts import validate_attention_width

    cfg = _compose(config_name, [f"task={task_name}"])
    validate_attention_width(
        attention_width=int(cfg.model.attention.n_classes),
        num_classes=int(cfg.model.graph_head.output_dim),
    )


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_finetune_mil_composes_per_class_widths(config_name: str) -> None:
    """Regression: the task preset's ``model`` block must survive composition.

    ``task/finetune_mil.yaml`` declares a two-class head and two attention
    channels. If ``model`` is merged after ``task`` in the defaults list, the
    module preset overwrites both and the published MIL head is not what runs.
    """
    cfg = _compose(config_name, ["task=finetune_mil"])
    assert int(cfg.model.graph_head.output_dim) == 2
    assert int(cfg.model.attention.n_classes) == 2
    assert cfg.task.target_type == "categorical"
    assert cfg.task.loss == "categorical_ce"


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_pretrain_bgrl_composes_and_overrides_the_scheduler(config_name: str) -> None:
    """Regression: ``defaults: - override /scheduler`` must actually resolve.

    This raised ``Could not override 'scheduler'`` from the CLI while every
    training test passed, because the tests never composed this preset.
    """
    cfg = _compose(config_name, ["task=pretrain_bgrl", "model=bgrl_module"])
    assert cfg.task.name == "pretrain_bgrl"
    assert cfg.model._target_.endswith("BGRLModule")
    # cosine_step, not the cosine_epoch default.
    assert int(cfg.scheduler.T_max) == 10000


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_default_task_is_unchanged_by_defaults_ordering(config_name: str) -> None:
    """Guards the reorder itself: the no-override baseline must not move."""
    cfg = _compose(config_name, [])
    assert cfg.task.name == "finetune_mean"
    assert int(cfg.model.graph_head.output_dim) == 1
    assert int(cfg.model.attention.n_classes) == 1
    assert int(cfg.scheduler.T_max) == 100
