"""Phase 2 fidelity: region bags, root sampling, parameter groups, schedules, class weights,
fine-tune augmentation, regime presets and region-level metrics."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from grass_mil.data.components.samplers import (
    UNLIMITED_NEIGHBORS,
    RuntimeShadowConfig,
    ShadowCustomStrategy,
)
from grass_mil.models.training import (
    build_finetune_augmentation,
    build_param_groups,
    drop_features,
    inverse_frequency_class_weights,
    resolve_t_max_auto,
)

CONFIG_MODULE = "grass_mil.configs"


def _compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module=CONFIG_MODULE):
        cfg = compose(config_name="train", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


def _units(n_units: int = 3, n_nodes: int = 40):
    from torch_geometric.data import Data

    out = []
    g = torch.Generator().manual_seed(0)
    for i in range(n_units):
        edge_index = torch.randint(0, n_nodes, (2, 160), generator=g)
        data = Data(
            x=torch.randn(n_nodes, 4, generator=g),
            edge_index=edge_index,
            edge_attr=torch.rand(160, 2, generator=g),
            graph_y=torch.tensor([[float(i % 2)]]),
            graph_w=torch.ones(1, 1),
            num_nodes=n_nodes,
        )
        data.region_id = f"region_{i}"
        data.sample_id = f"sample_{i}"
        data.patch_id = f"patch_{i}"
        out.append(data)
    return out


def _runtime(**kw):
    base = dict(
        enabled=True,
        depth=2,
        num_neighbors=-1,
        subgraph_batch_size=8,
        proportional_root_sampling=False,
    )
    base.update(kw)
    return base


def test_num_neighbors_minus_one_means_unlimited():
    cfg = RuntimeShadowConfig.from_dict(_runtime())
    assert cfg.num_neighbors == UNLIMITED_NEIGHBORS


def test_unit_as_bag_yields_one_batch_per_unit_with_subsample_count_roots():
    strategy = ShadowCustomStrategy(runtime=_runtime(subsample_count=12, unit_as_bag=True))
    loader = strategy.build_dataset_loader(dataset=_units(), batch_size=1, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3 == len(loader)
    for batch in batches:
        assert int(batch.ptr.numel()) - 1 == 12
        assert len(set(batch.region_id)) == 1


def test_fixed_per_unit_seed_gives_identical_roots_every_pass():
    strategy = ShadowCustomStrategy(
        runtime=_runtime(subsample_count=10, unit_as_bag=True, fixed_per_unit_seed=7)
    )
    loader = strategy.build_dataset_loader(dataset=_units(), batch_size=1, shuffle=False)
    first = [b.root_n_id.sort().values.tolist() for b in loader]
    second = [b.root_n_id.sort().values.tolist() for b in loader]
    assert first == second
    unseeded = ShadowCustomStrategy(runtime=_runtime(subsample_count=10, unit_as_bag=True))
    loader2 = unseeded.build_dataset_loader(dataset=_units(), batch_size=1, shuffle=False)
    a = [b.root_n_id.sort().values.tolist() for b in loader2]
    b = [b.root_n_id.sort().values.tolist() for b in loader2]
    assert a != b


def test_all_roots_mode_covers_every_node_once():
    strategy = ShadowCustomStrategy(runtime=_runtime(subsample_count=None, subgraph_batch_size=16))
    loader = strategy.build_dataset_loader(
        dataset=_units(n_units=1, n_nodes=40), batch_size=1, shuffle=False
    )
    roots = torch.cat([b.root_n_id for b in loader]).sort().values.tolist()
    assert roots == list(range(40))


def test_build_param_groups_scales_and_freezes():
    enc = torch.nn.Linear(4, 4)
    head = torch.nn.Linear(4, 2)
    att = torch.nn.Linear(4, 2)
    groups = build_param_groups(
        base_lr=1e-4,
        base_weight_decay=1e-3,
        encoder=enc,
        head=head,
        attention=att,
        encoder_lr_scale=0.0,
        head_lr_scale=0.01,
        attention_lr_scale=2.0,
        head_weight_decay=5e-4,
    )
    names = [g["name"] for g in groups]
    assert names == ["head", "attention"]
    assert groups[0]["lr"] == pytest.approx(1e-6) and groups[0]["weight_decay"] == pytest.approx(
        5e-4
    )
    assert groups[1]["lr"] == pytest.approx(2e-4) and groups[1]["weight_decay"] == pytest.approx(
        1e-3
    )
    with pytest.raises(ValueError):
        build_param_groups(
            base_lr=1e-4,
            base_weight_decay=0.0,
            encoder=enc,
            head=head,
            attention=att,
            encoder_lr_scale=0.0,
            head_lr_scale=0.0,
            attention_lr_scale=0.0,
        )


def test_t_max_auto_counts_optimizer_steps():
    assert resolve_t_max_auto(num_train_units=340, max_epochs=25, hyperbatch_size=12) == 25 * 29
    assert resolve_t_max_auto(num_train_units=5, max_epochs=3, hyperbatch_size=12) == 3


def test_inverse_frequency_class_weights_have_mean_one():
    w = inverse_frequency_class_weights([0] * 70 + [1] * 30, 2)
    assert w.mean().item() == pytest.approx(1.0)
    assert w[1] > w[0]
    assert (w[1] / w[0]).item() == pytest.approx(70 / 30)


def test_elementwise_feature_drop_writes_unassigned_token():
    from torch_geometric.data import Data

    data = Data(
        x=torch.ones(200, 3), edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=200
    )
    data.categorical_codes = torch.zeros(200, 1, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    out = drop_features(
        data, p=0.3, generator=gen, mode="elementwise", categorical_column=0, unassigned_index=9
    )
    zero_frac = float((out.x == 0).float().mean())
    assert 0.2 < zero_frac < 0.4
    assert 0.2 < float((out.categorical_codes[:, 0] == 9).float().mean()) < 0.4
    assert torch.equal(data.x, torch.ones(200, 3))  # input untouched
    assert build_finetune_augmentation({"drop_edge_p": 0.0, "drop_feat_p": 0.0}) is None
    aug = build_finetune_augmentation({"drop_edge_p": 0.1, "drop_feat_p": 0.1}, unassigned_index=9)
    assert aug(data).x.shape == data.x.shape


@pytest.mark.parametrize(
    ("regime", "extra"),
    [
        ("mil", []),
        ("mil_frozen", []),
        ("linear_probehead", ["task.head_init_path=/tmp/none.pt"]),
        ("linear_ftall", ["task.head_init_path=/tmp/none.pt"]),
        ("linear_ftall_reg", ["task.head_init_path=/tmp/none.pt"]),
    ],
)
def test_regime_presets_compose(regime: str, extra: list[str]):
    cfg = _compose([f"regime={regime}", *extra])
    assert cfg.regime_name == regime
    assert cfg.model.encoder.conv_type == "gine" and int(cfg.model.encoder.num_layers) == 4
    assert cfg.data.sampler.runtime.unit_as_bag is True
    assert int(cfg.data.val_sampler.runtime.subsample_count) == 1024
    assert str(cfg.scheduler.T_max) == "auto" and cfg.task.scheduler_interval == "step"
    if regime.startswith("linear"):
        assert int(cfg.model.graph_head.num_layers) == 1
    else:
        assert int(cfg.model.graph_head.num_layers) == 2
    if regime == "linear_ftall_reg":
        assert float(cfg.task.optimization.encoder_lr_scale) == pytest.approx(0.3)
        assert float(cfg.task.finetune_augmentation.drop_edge_p) == pytest.approx(0.05)
        assert float(cfg.model.encoder.dropout) == pytest.approx(0.2)
    if regime == "linear_probehead":
        assert (
            cfg.task.freeze_head is True and float(cfg.task.optimization.encoder_lr_scale) == 0.0
        )


def test_regime_defaults_can_be_overridden():
    cfg = _compose(
        [
            "regime=mil",
            "model.encoder.num_layers=3",
            "data.sampler.runtime.depth=3",
            "model.graph_head.num_layers=1",
            "data.sampler.runtime.subsample_count=64",
        ]
    )
    assert int(cfg.model.encoder.num_layers) == 3 and int(cfg.data.sampler.runtime.depth) == 3
    assert (
        int(cfg.model.graph_head.num_layers) == 1
        and int(cfg.data.sampler.runtime.subsample_count) == 64
    )


def test_region_level_metrics_binary_and_categorical():
    from grass_mil.models.supervised_module import SupervisedModule

    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [1.5, 0.0], [-2.0, 1.0]])
    targets = torch.tensor([0, 1, 0, 1])
    m = SupervisedModule._region_level_metrics("categorical", logits, targets)
    assert m["auroc"] == pytest.approx(1.0) and m["bal_acc"] == pytest.approx(1.0)
    assert m["loss_region"] > 0
    mb = SupervisedModule._region_level_metrics(
        "binary", torch.tensor([[2.0], [-2.0]]), torch.tensor([1.0, 0.0])
    )
    assert mb["auroc"] == pytest.approx(1.0)


def test_mil_training_with_scaled_groups_and_auto_weights(cfg_train, tmp_path):
    """End-to-end: region accumulation, scaled parameter groups, auto class weights,
    per-step cosine with T_max auto, region-level val metrics and test predictions."""
    from grass_mil.train import train

    cfg = _compose(["task=finetune_mil", "model=supervised_module", "scheduler=cosine_mil_step"])
    with open_dict(cfg):
        cfg.data = OmegaConf.create(
            {
                "_target_": "tests.helpers.synthetic_datamodule.SyntheticBagDataModule",
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "input_dim": 8,
            }
        )
        cfg.model.encoder.input_dim = 8
        cfg.task.optimization.encoder_lr_scale = 0.3
        cfg.task.optimization.head_lr_scale = 0.01
        cfg.task.optimization.head_weight_decay = 1.0e-3
        cfg.task.region_accumulation.hyperbatch_size = 2
        cfg.task.grad_clip_norm = 1.0
        cfg.callbacks = None
        cfg.logger = None
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.root_dir = str(tmp_path)
        cfg.trainer.max_epochs = 1
        cfg.trainer.min_epochs = 1
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.default_root_dir = str(tmp_path)
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.train = True
        cfg.test = True
    metrics, objects = train(cfg)
    module = objects["model"]
    assert module.loss_fn.class_weight is not None
    assert module.loss_fn.class_weight.mean().item() == pytest.approx(1.0)
    opt = objects["trainer"].optimizers[0]
    group_lrs = sorted(g["initial_lr"] if "initial_lr" in g else g["lr"] for g in opt.param_groups)
    assert len(opt.param_groups) == 3
    assert "val/auroc" in metrics and "test/auroc" in metrics
    assert (Path(tmp_path) / "metrics_per_epoch.jsonl").exists()
    assert (Path(tmp_path) / "test_predictions.csv").exists()
    assert group_lrs[0] < group_lrs[-1]
