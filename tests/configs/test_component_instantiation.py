from pathlib import Path

import hydra
from omegaconf import OmegaConf


def _instantiate_yaml(path: Path):
    cfg = OmegaConf.load(path)
    return hydra.utils.instantiate(cfg)


def test_model_component_yaml_instantiation():
    base = Path("configs/model")
    targets = [
        base / "encoder" / "gin.yaml",
        base / "encoder" / "gcn.yaml",
        base / "encoder" / "gat.yaml",
        base / "encoder" / "graphsage.yaml",
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


def test_sampler_strategy_resolution():
    from src.data.components.samplers import get_sampler_strategy

    base = Path("configs/data/sampler")
    for name in ["identity", "shadow_native", "shadow_custom"]:
        cfg = OmegaConf.load(base / f"{name}.yaml")
        strategy = get_sampler_strategy(OmegaConf.to_container(cfg, resolve=True))
        assert strategy is not None
