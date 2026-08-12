from pathlib import Path

from tests.helpers.config_paths import CONFIGS_DIR

import hydra
from omegaconf import OmegaConf


def _instantiate_yaml(path: Path):
    cfg = OmegaConf.load(path)
    return hydra.utils.instantiate(cfg)


def test_model_component_yaml_instantiation():
    base = CONFIGS_DIR / "model"
    targets = [
        base / "encoder" / "gin.yaml",
        base / "encoder" / "gcn.yaml",
        base / "encoder" / "gat.yaml",
        base / "encoder" / "graphsage.yaml",
        base / "encoder" / "gine.yaml",
        base / "attention" / "gated.yaml",
        base / "attention" / "gated_projected.yaml",
        base / "heads" / "graph.yaml",
        base / "heads" / "node.yaml",
        base / "loss" / "categorical_ce.yaml",
        base / "loss" / "categorical_bce.yaml",
        base / "loss" / "regression_mse.yaml",
        base / "loss" / "regression_huber.yaml",
        base / "loss" / "survival_coxsgd.yaml",
    ]
    for path in targets:
        obj = _instantiate_yaml(path)
        assert obj is not None


def test_training_module_yaml_instantiation():
    base = CONFIGS_DIR / "model"
    cases = [
        ("bgrl_module.yaml", "pretrain_bgrl.yaml", "cosine_step.yaml"),
        ("supervised_module.yaml", "finetune_mean.yaml", "cosine_epoch.yaml"),
        ("supervised_module.yaml", "finetune_mil.yaml", "cosine_epoch.yaml"),
    ]
    for model_name, task_name, scheduler_name in cases:
        path = base / model_name
        cfg = OmegaConf.load(path)
        cfg.optim = OmegaConf.load(CONFIGS_DIR / "optim" / "adamw.yaml")
        cfg.scheduler = OmegaConf.load(CONFIGS_DIR / "scheduler" / scheduler_name)
        cfg.task = OmegaConf.load(CONFIGS_DIR / "task" / task_name).task
        obj = hydra.utils.instantiate(cfg)
        assert obj is not None


def test_sampler_strategy_resolution():
    from grass_mil.data.components.samplers import get_sampler_strategy

    base = CONFIGS_DIR / "data" / "sampler"
    for name in ["identity", "shadow_native", "shadow_custom"]:
        cfg = OmegaConf.load(base / f"{name}.yaml")
        strategy = get_sampler_strategy(OmegaConf.to_container(cfg, resolve=True))
        assert strategy is not None


def test_default_component_and_data_sampler_configs():
    model_cfg = OmegaConf.load(CONFIGS_DIR / "model" / "components.yaml")
    assert model_cfg.model.attention._target_ == (
        "grass_mil.models.components.attention.AttnNetGatedProjected"
    )

    data_cfg = OmegaConf.load(CONFIGS_DIR / "data" / "spatial_omics.yaml")
    assert data_cfg.sampler.name == "shadow_custom"
