from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from grass_mil.train import train
from tests.helpers.config_paths import CONFIGS_DIR


def _load_model_cfg(name: str):
    # Model presets carry a defaults list (model/encoder, model/attention groups),
    # so they must be composed through Hydra rather than loaded as plain YAML.
    from hydra import compose, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="grass_mil.configs"):
        cfg = compose(config_name="train", overrides=[f"model={name}"])
    GlobalHydra.instance().clear()
    return cfg.model


def _load_task_cfg(name: str):
    return OmegaConf.load(CONFIGS_DIR / "task" / f"{name}.yaml").task


def _load_task_model_overrides(name: str):
    """Model overrides a task preset injects into the global package.

    Hydra applies these during composition; this test composes manually, so it
    must apply them too or the head width and the task objective disagree.
    """
    cfg = OmegaConf.load(CONFIGS_DIR / "task" / f"{name}.yaml")
    return cfg.get("model", None)


def _make_pretrain_ckpt(path: Path) -> str:
    from grass_mil.models.bgrl_module import BGRLModule

    module = BGRLModule(
        encoder={
            "input_dim": 8,
            "hidden_dim": 32,
            "out_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "conv_type": "gin",
            "norm": "batchnorm",
            "jk": "last",
            "act": "relu",
            "pooling": "mean",
        },
        ssl={"method": "bgrl", "predictor": {"hidden_size": 64}},
        optim={"_target_": "torch.optim.AdamW", "lr": 1e-3, "weight_decay": 0.0},
        scheduler={
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 10,
        },
        task={"total_steps": 10},
    )
    module.setup("fit")
    ckpt_path = path / "pretrain_mock.ckpt"
    import torch

    torch.save({"state_dict": module.ssl_model.state_dict()}, ckpt_path)
    return str(ckpt_path)


@pytest.mark.parametrize(
    "task_name,model_name,use_pretrained,advanced_mil",
    [
        ("pretrain_bgrl", "bgrl_module", False, False),
        ("finetune_mean", "supervised_module", False, False),
        ("finetune_mil", "supervised_module", False, False),
        ("finetune_mil", "supervised_module", False, True),
        ("finetune_mean", "supervised_module", True, False),
        ("finetune_mil", "supervised_module", True, False),
    ],
)
def test_training_regimes_fast_dev_run(
    cfg_train, tmp_path, task_name, model_name, use_pretrained, advanced_mil
):
    with open_dict(cfg_train):
        cfg_train.data = OmegaConf.create(
            {
                "_target_": "tests.helpers.synthetic_datamodule.SyntheticBagDataModule",
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "input_dim": 8,
            }
        )
        cfg_train.task = _load_task_cfg(task_name)
        cfg_train.model = _load_model_cfg(model_name)
        cfg_train.optim = OmegaConf.load(CONFIGS_DIR / "optim" / "adamw.yaml")
        scheduler_name = (
            "cosine_step.yaml" if task_name == "pretrain_bgrl" else "cosine_epoch.yaml"
        )
        cfg_train.scheduler = OmegaConf.load(CONFIGS_DIR / "scheduler" / scheduler_name)
        task_model_overrides = _load_task_model_overrides(task_name)
        if task_model_overrides is not None:
            cfg_train.model = OmegaConf.merge(cfg_train.model, task_model_overrides)
        cfg_train.model.encoder.input_dim = 8
        cfg_train.model.task = cfg_train.task
        cfg_train.model.optim = cfg_train.optim
        cfg_train.model.scheduler = cfg_train.scheduler
        if advanced_mil:
            # Region accumulation only; the auxiliary instance and entropy
            # terms were removed with the region-CE-only objective.
            cfg_train.model.task.region_accumulation.enabled = True
            cfg_train.model.task.region_accumulation.hyperbatch_size = 2
            cfg_train.model.task.region_accumulation.flush_on_epoch_end = True
        if use_pretrained:
            cfg_train.model.init_from_ckpt = _make_pretrain_ckpt(tmp_path)
            cfg_train.model.init_strict = False
            # Align encoder dims with the mock checkpoint (hidden_dim=32, out_dim=32).
            cfg_train.model.encoder.hidden_dim = 32
            cfg_train.model.encoder.out_dim = 32
        if "graph_head" in cfg_train.model:
            cfg_train.model.graph_head.input_dim = cfg_train.model.encoder.out_dim
            cfg_train.model.graph_head.hidden_dim = cfg_train.model.encoder.out_dim
        if "attention" in cfg_train.model:
            cfg_train.model.attention.input_dim = cfg_train.model.encoder.out_dim
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.train = True
        cfg_train.test = False

    HydraConfig().set_config(cfg_train)
    metric_dict, _ = train(cfg_train)
    assert "train/loss" in metric_dict
