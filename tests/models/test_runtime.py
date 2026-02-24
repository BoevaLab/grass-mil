import pytest
import torch


def test_remap_encoder_keys_auto_bgrl_or_identity():
    from src.models.training.checkpoint_init import remap_encoder_keys

    sd = {
        "online_encoder.layers.0.weight": torch.randn(4, 4),
        "ssl_model.online_encoder.layers.0.bias": torch.randn(4),
        "graph_head.net.0.weight": torch.randn(4, 4),
    }
    out = remap_encoder_keys(sd, encoder_init_map="auto_bgrl_or_identity")
    assert "encoder.layers.0.weight" in out
    assert "encoder.layers.0.bias" in out
    assert "graph_head.net.0.weight" in out


def test_compute_supervised_loss_routes():
    from src.models.components.losses import (
        CoxSGDLoss,
        WeightedBCEWithLogitsLoss,
        WeightedCrossEntropyLoss,
        WeightedMSELoss,
    )
    from src.models.training.loss_utils import compute_supervised_loss

    bce = compute_supervised_loss(
        loss_fn=WeightedBCEWithLogitsLoss(),
        task_cfg={"target_type": "binary"},
        bag_logits=torch.randn(4, 1),
        bag_targets=torch.randint(0, 2, (4, 1)).float(),
        bag_weights=torch.ones(4, 1),
    )
    assert bce.ndim == 0

    mse = compute_supervised_loss(
        loss_fn=WeightedMSELoss(),
        task_cfg={"target_type": "regression"},
        bag_logits=torch.randn(4, 1),
        bag_targets=torch.randn(4, 1),
        bag_weights=torch.ones(4, 1),
    )
    assert mse.ndim == 0

    ce = compute_supervised_loss(
        loss_fn=WeightedCrossEntropyLoss(),
        task_cfg={"target_type": "categorical"},
        bag_logits=torch.randn(5, 3),
        bag_targets=torch.tensor([[0], [1], [2], [1], [0]], dtype=torch.long),
        bag_weights=torch.ones(5),
    )
    assert ce.ndim == 0

    cox = compute_supervised_loss(
        loss_fn=CoxSGDLoss(),
        task_cfg={"target_type": "survival"},
        bag_logits=torch.randn(6, 1),
        bag_targets=torch.stack(
            [
                torch.linspace(1, 6, 6),
                torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.float32),
            ],
            dim=1,
        ),
        bag_weights=None,
    )
    assert cox.ndim == 0


def test_validate_task_config_accepts_categorical():
    from src.models.training.builders import validate_task_config

    validate_task_config({"target_type": "categorical", "instance_sampling": "all"})


def test_supervised_module_validation_logs_step_loss_not_acc():
    pytest.importorskip("lightning")
    from src.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "categorical"},
    )

    logged = {}

    def _capture_log(name, value, **kwargs):
        logged[name] = (value, kwargs)

    module.log = _capture_log  # type: ignore[method-assign]
    module._log_stage_metrics(
        stage="val",
        loss_value=torch.tensor(0.2),
        bag_logits=torch.tensor([[5.0, 1.0, -1.0], [-2.0, 0.2, 3.0], [0.1, 0.2, 0.3]]),
        bag_targets=torch.tensor([0, 2, 1]),
    )
    assert "val/acc" not in logged
    assert "val/loss" in logged
    _, kwargs = logged["val/loss"]
    assert kwargs["on_step"] is True
    assert kwargs["on_epoch"] is True


def test_supervised_module_logs_epoch_validation_accuracy_from_aggregated_regions():
    pytest.importorskip("lightning")
    from src.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "binary"},
    )
    logged = {}

    def _capture_log(name, value, **kwargs):
        logged[name] = (value, kwargs)

    module.log = _capture_log  # type: ignore[method-assign]
    module.on_validation_epoch_start()
    module._val_bag_ids = ["p0", "p1", "p2"]
    module._val_bag_logits_chunks = [torch.tensor([[0.0], [0.0], [0.0]])]
    module._val_bag_targets_chunks = [torch.tensor([[1.0], [1.0], [0.0]])]
    module._val_row_region_ids = ["rA", "rA", "rB"]
    module._val_row_sample_ids = ["s0", "s0", "s1"]
    module._val_instance_logits_chunks = [torch.tensor([[2.0], [-2.0], [-2.0]])]
    module._val_instance_patch_ids = ["p0", "p1", "p2"]
    module._val_instance_region_ids = ["rA", "rA", "rB"]
    module._val_instance_sample_ids = ["s0", "s0", "s1"]

    module.on_validation_epoch_end()
    assert "val/acc" in logged
    acc_value, kwargs = logged["val/acc"]
    assert torch.isclose(acc_value, torch.tensor(1.0))
    assert kwargs["on_step"] is False
    assert kwargs["on_epoch"] is True


def test_supervised_module_logs_epoch_validation_regression_metrics():
    pytest.importorskip("lightning")
    from src.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "regression"},
    )
    logged = {}

    def _capture_log(name, value, **kwargs):
        logged[name] = (value, kwargs)

    module.log = _capture_log  # type: ignore[method-assign]
    module.on_validation_epoch_start()
    module._val_bag_ids = ["p0", "p1"]
    module._val_bag_logits_chunks = [torch.tensor([[1.0], [3.0]])]
    module._val_bag_targets_chunks = [torch.tensor([[1.5], [2.5]])]
    module._val_row_region_ids = ["rA", "rB"]
    module._val_row_sample_ids = ["s0", "s1"]
    module._val_instance_logits_chunks = [torch.tensor([[1.0], [3.0]])]
    module._val_instance_patch_ids = ["p0", "p1"]
    module._val_instance_region_ids = ["rA", "rB"]
    module._val_instance_sample_ids = ["s0", "s1"]

    module.on_validation_epoch_end()
    assert "val/mae" in logged
    assert "val/rmse" in logged
    assert "val/r2" in logged
    assert logged["val/mae"][1]["on_epoch"] is True
    assert logged["val/rmse"][1]["on_epoch"] is True
    assert logged["val/r2"][1]["on_epoch"] is True


def test_supervised_module_logs_epoch_validation_survival_c_index():
    pytest.importorskip("lightning")
    from src.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "survival"},
    )
    logged = {}

    def _capture_log(name, value, **kwargs):
        logged[name] = (value, kwargs)

    module.log = _capture_log  # type: ignore[method-assign]
    module.on_validation_epoch_start()
    module._val_bag_ids = ["p0", "p1", "p2"]
    module._val_bag_logits_chunks = [torch.tensor([[3.0], [2.0], [1.0]])]
    module._val_bag_targets_chunks = [
        torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    ]
    module._val_row_region_ids = ["rA", "rB", "rC"]
    module._val_row_sample_ids = ["s0", "s1", "s2"]
    module._val_instance_logits_chunks = [torch.tensor([[3.0], [2.0], [1.0]])]
    module._val_instance_patch_ids = ["p0", "p1", "p2"]
    module._val_instance_region_ids = ["rA", "rB", "rC"]
    module._val_instance_sample_ids = ["s0", "s1", "s2"]

    module.on_validation_epoch_end()
    assert "val/c_index" in logged
    c_index_value, kwargs = logged["val/c_index"]
    assert torch.isclose(c_index_value, torch.tensor(1.0))
    assert kwargs["on_step"] is False
    assert kwargs["on_epoch"] is True


def test_attention_bag_aggregation_shapes():
    import torch
    from src.models.components.attention import AttnNetGatedProjected
    from src.models.training.bagging import aggregate_bag_logits_attention

    logits = torch.randn(5, 1)
    emb = torch.randn(5, 8)
    att = AttnNetGatedProjected(
        input_dim=8, projection_dim=4, hidden_dim=2, n_classes=1
    )
    grouped = {"r0": [0, 1, 2], "r1": [3, 4]}
    bag_logits, bag_ids, bag_attn, bag_indices = aggregate_bag_logits_attention(
        logits=logits,
        embeddings=emb,
        attention=att,
        bag_groups=grouped,
        max_instances_per_bag=0,
        instance_sampling="all",
    )
    assert bag_logits.shape == (2, 1)
    assert bag_ids == ["r0", "r1"]
    assert len(bag_attn["r0"]) == 3
    assert len(bag_indices) == 2


def test_build_mil_aux_targets_attention_shaped():
    from src.models.training.loss_utils import build_mil_aux_targets

    bag_targets = torch.tensor([[1.0], [0.0]])
    bag_indices = [[0, 1], [2, 3, 4]]
    bag_ids = ["r0", "r1"]
    bag_attention = {
        "r0": torch.tensor([0.25, 0.75]),
        "r1": torch.tensor([0.2, 0.3, 0.5]),
    }
    targets, weights = build_mil_aux_targets(
        bag_targets=bag_targets,
        bag_indices=bag_indices,
        bag_ids=bag_ids,
        bag_attention=bag_attention,
        target_mode="attention_shaped_ti",
    )
    assert targets.shape == (5, 1)
    assert weights.shape == (5, 1)
    assert torch.all(targets[:2] >= 0.5)
    assert torch.all(targets[2:] <= 0.5)


def test_compute_aux_and_entropy_terms():
    from src.models.training.loss_utils import (
        compute_aux_node_loss,
        compute_entropy_regularization,
    )

    logits = torch.tensor([[0.2], [-0.3], [0.9]])
    targets = torch.tensor([[1.0], [0.0], [1.0]])
    weights = torch.tensor([[0.5], [0.25], [0.25]])
    bce = compute_aux_node_loss(
        aux_logits=logits, aux_targets=targets, aux_weights=None, loss_mode="bce"
    )
    weighted = compute_aux_node_loss(
        aux_logits=logits,
        aux_targets=targets,
        aux_weights=weights,
        loss_mode="weighted_bce",
    )
    ent = compute_entropy_regularization(targets)
    assert bce.ndim == 0
    assert weighted.ndim == 0
    assert ent.ndim == 0


def test_select_target_columns_handles_single_label_list():
    from types import SimpleNamespace

    from src.models.training.bagging import select_target_columns

    graph_y = torch.tensor([[1.0]])
    batch = SimpleNamespace(graph_label_names=["label"])
    selected = select_target_columns(graph_y, batch, target_columns=["label"])
    assert selected.shape == (1, 1)
    assert torch.equal(selected, graph_y)


def test_supervised_predict_step_allows_missing_targets():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from src.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "binary"},
    )

    def _forward_bags(_batch, *, allow_missing_targets=False):
        assert allow_missing_targets is True
        return {
            "patch_logits": torch.randn(4, 1),
            "bag_logits": torch.randn(2, 1),
            "ordered_bag_ids": ["b0", "b1"],
            "bag_attention": None,
            "bag_indices": [[0, 1], [2, 3]],
            "bag_targets": None,
            "bag_weights": None,
        }

    module._forward_bags = _forward_bags  # type: ignore[method-assign]
    out = module.predict_step(batch=None, batch_idx=0)
    assert out["bag_ids"] == ["b0", "b1"]
    assert out["bag_logits"].shape == (2, 1)
    assert "bag_targets" not in out


def test_extract_bag_ids_falls_back_per_item_when_primary_missing():
    from types import SimpleNamespace

    from src.models.training.bagging import extract_bag_ids

    batch = SimpleNamespace(region_id=[None, "", "r2"], sample_id=["s0", "s1", "s2"])
    bag_ids = extract_bag_ids(batch, bag_key="region_id", bag_fallback_key="sample_id")
    assert bag_ids == ["s0", "s1", "r2"]


def test_gather_bag_targets_requires_consistent_labels_within_bag():
    from types import SimpleNamespace

    from src.models.training.bagging import gather_bag_targets

    batch = SimpleNamespace(
        graph_y=torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32),
        graph_w=torch.ones(3, 1),
    )
    with pytest.raises(ValueError, match="Inconsistent graph_y"):
        gather_bag_targets(
            batch=batch,
            bag_indices=[[0, 1], [2]],
            target_columns=None,
        )


def test_attention_shaped_aux_targets_are_detached():
    from src.models.training.loss_utils import build_mil_aux_targets

    bag_targets = torch.tensor([[1.0]], dtype=torch.float32)
    bag_attention = {"b0": torch.tensor([0.2, 0.8], requires_grad=True)}
    targets, weights = build_mil_aux_targets(
        bag_targets=bag_targets,
        bag_indices=[[0, 1]],
        bag_ids=["b0"],
        bag_attention=bag_attention,
        target_mode="attention_shaped_ti",
    )
    assert targets.requires_grad is False
    assert weights.requires_grad is False


def test_region_buffer_flushes_in_hyperbatch_chunks():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from src.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        attention={"input_dim": 1},
        loss={},
        optim={},
        task={
            "aggregation": "mil_attention",
            "target_type": "binary",
            "region_accumulation": {"enabled": True, "hyperbatch_size": 2},
        },
    )

    class _Opt:
        def __init__(self):
            self.steps = 0
            self.zeroes = 0

        def step(self):
            self.steps += 1

        def zero_grad(self, set_to_none=True):
            self.zeroes += 1

    class _Sched:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    opt = _Opt()
    sch = _Sched()
    module.optimizers = lambda: opt  # type: ignore[method-assign]
    module.lr_schedulers = lambda: sch  # type: ignore[method-assign]
    module.manual_backward = lambda loss: None  # type: ignore[method-assign]
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    module._region_total_loss_buffer = [
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(4.0),
        torch.tensor(5.0),
    ]
    flushes = module._flush_region_buffer_if_needed(force=False)
    assert len(flushes) == 2
    assert len(module._region_total_loss_buffer) == 1
    assert opt.steps == 2
    assert sch.steps == 2

    flushes = module._flush_region_buffer_if_needed(force=True)
    assert len(flushes) == 1
    assert len(module._region_total_loss_buffer) == 0
    assert opt.steps == 3
    assert sch.steps == 3
