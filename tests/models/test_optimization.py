import torch
import pytest


def _encoder_cfg():
    return {
        "input_dim": 8,
        "hidden_dim": 16,
        "out_dim": 16,
        "num_layers": 2,
        "dropout": 0.1,
        "conv_type": "gin",
        "norm": "batchnorm",
        "jk": "last",
        "act": "relu",
        "pooling": "mean",
        "gat_heads": 2,
        "use_edge_attr": False,
        "edge_weight_index": 0,
        "edge_attr_dim": None,
    }


def _optim_cfg():
    return {"_target_": "torch.optim.AdamW", "lr": 1e-3, "weight_decay": 1e-2}


def _cosine_epoch_cfg():
    return {
        "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 10,
        "eta_min": 1e-6,
    }


def _cosine_step_cfg():
    return {
        "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 1e-7,
    }


def test_supervised_mil_uses_explicit_backbone_and_attention_lrs():
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder=_encoder_cfg(),
        graph_head={
            "input_dim": 16,
            "output_dim": 1,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
        },
        attention={
            "attention_type": "gated_projected",
            "input_dim": 16,
            "projection_dim": 8,
            "hidden_dim": 4,
            "dropout": False,
            "n_classes": 1,
        },
        loss={"loss_type": "categorical_bce"},
        optim=_optim_cfg(),
        scheduler=_cosine_epoch_cfg(),
        task={
            "aggregation": "mil_attention",
            "target_type": "binary",
            "loss": "categorical_bce",
            "optimization": {"backbone_lr": 1e-3, "attention_lr": 4e-3},
            "lr_warmup": {"enabled": False},
            "region_accumulation": {"enabled": False},
        },
    )
    module.setup("fit")
    out = module.configure_optimizers()
    optimizer = out["optimizer"] if isinstance(out, dict) else out

    lrs = sorted([group["lr"] for group in optimizer.param_groups])
    assert len(optimizer.param_groups) == 2
    assert lrs == [1e-3, 4e-3]


def test_supervised_mean_warmup_is_opt_in():
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

    base_kwargs = dict(
        encoder=_encoder_cfg(),
        graph_head={
            "input_dim": 16,
            "output_dim": 1,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
        },
        loss={"loss_type": "categorical_bce"},
        optim=_optim_cfg(),
        scheduler=_cosine_epoch_cfg(),
    )
    base_task = {"target_type": "binary", "loss": "categorical_bce"}

    no_warmup = SupervisedModule(
        **base_kwargs,
        task={
            **base_task,
            "lr_warmup": {"enabled": False, "warmup_steps": 0, "start_factor": 0.1},
        },
    )
    no_warmup.setup("fit")
    no_warmup_out = no_warmup.configure_optimizers()
    no_warmup_scheduler = no_warmup_out["lr_scheduler"]["scheduler"]
    assert isinstance(no_warmup_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    with_warmup = SupervisedModule(
        **base_kwargs,
        task={
            **base_task,
            "lr_warmup": {"enabled": True, "warmup_steps": 2, "start_factor": 0.1},
        },
    )
    with_warmup.setup("fit")
    with_warmup_out = with_warmup.configure_optimizers()
    with_warmup_scheduler = with_warmup_out["lr_scheduler"]["scheduler"]
    assert isinstance(with_warmup_scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_bgrl_uses_warmup_then_base_scheduler():
    pytest.importorskip("torch_geometric")
    from grass_mil.models.bgrl_module import BGRLModule

    module = BGRLModule(
        encoder=_encoder_cfg(),
        ssl={"method": "bgrl", "predictor": {"hidden_size": 32}},
        optim=_optim_cfg(),
        scheduler=_cosine_step_cfg(),
        task={
            "total_steps": 100,
            "warmup_steps": 10,
            "momentum": 0.99,
            "momentum_min": 0.99,
            "lr_warmup": {"enabled": True, "warmup_steps": 5, "start_factor": 0.1},
        },
    )
    module.setup("fit")
    out = module.configure_optimizers()
    scheduler = out["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert len(scheduler._schedulers) == 2
    assert isinstance(scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR)
