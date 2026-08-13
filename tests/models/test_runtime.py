import pytest
import torch


def test_remap_encoder_keys_auto_bgrl_or_identity():
    from grass_mil.models.training.checkpoint_init import remap_encoder_keys

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
    from grass_mil.models.components.losses import (
        CoxSGDLoss,
        WeightedBCEWithLogitsLoss,
        WeightedCrossEntropyLoss,
        WeightedMSELoss,
    )
    from grass_mil.models.training.loss_utils import compute_supervised_loss

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
    from grass_mil.models.training.builders import validate_task_config

    validate_task_config({"target_type": "categorical", "instance_sampling": "all"})


def test_supervised_module_validation_logs_step_loss_not_acc():
    pytest.importorskip("lightning")
    from grass_mil.models.supervised_module import SupervisedModule

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
    from grass_mil.models.supervised_module import SupervisedModule

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
    from grass_mil.models.supervised_module import SupervisedModule

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
    from grass_mil.models.supervised_module import SupervisedModule

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
    module._val_bag_targets_chunks = [torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])]
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
    from grass_mil.models.components.attention import AttnNetGatedProjected
    from grass_mil.models.training.bagging import aggregate_bag_logits_attention

    logits = torch.randn(5, 1)
    emb = torch.randn(5, 8)
    att = AttnNetGatedProjected(input_dim=8, projection_dim=4, hidden_dim=2, n_classes=1)
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


def test_select_target_columns_handles_single_label_list():
    from types import SimpleNamespace

    from grass_mil.models.training.bagging import select_target_columns

    graph_y = torch.tensor([[1.0]])
    batch = SimpleNamespace(graph_label_names=["label"])
    selected = select_target_columns(graph_y, batch, target_columns=["label"])
    assert selected.shape == (1, 1)
    assert torch.equal(selected, graph_y)


def test_supervised_predict_step_allows_missing_targets():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

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


def test_supervised_predict_step_can_disable_instance_payload():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "binary"},
    )

    def _forward_bags(_batch, *, allow_missing_targets=False):
        return {
            "patch_logits": torch.randn(4, 1),
            "bag_logits": torch.randn(2, 1),
            "ordered_bag_ids": ["b0", "b1"],
            "bag_attention": None,
            "bag_indices": [[0, 1], [2, 3]],
            "bag_targets": None,
            "bag_weights": None,
        }

    captured = {"include_instance_payload": None}

    def _build_predict_group_metadata(
        _batch,
        *,
        patch_logits,
        graph_emb,
        ordered_bag_ids,
        bag_indices,
        include_instance_payload=True,
    ):
        captured["include_instance_payload"] = include_instance_payload
        payload = {
            "row_region_ids": ["r0", "r1"],
            "row_sample_ids": ["s0", "s1"],
        }
        if include_instance_payload:
            payload.update(
                {
                    "instance_logits": torch.randn(4, 1),
                    "instance_patch_ids": ["p0", "p1", "p2", "p3"],
                    "instance_region_ids": ["r0", "r0", "r1", "r1"],
                    "instance_sample_ids": ["s0", "s0", "s1", "s1"],
                }
            )
        return payload

    module._forward_bags = _forward_bags  # type: ignore[method-assign]
    module._build_predict_group_metadata = _build_predict_group_metadata  # type: ignore[method-assign]
    module._predict_emit_instance_payload = False

    out = module.predict_step(batch=None, batch_idx=0)
    assert captured["include_instance_payload"] is False
    assert "instance_logits" not in out
    assert "instance_patch_ids" not in out
    assert "instance_region_ids" not in out
    assert "instance_sample_ids" not in out


def test_supervised_predict_step_can_emit_embedding_payload():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "binary"},
    )

    def _forward_bags(_batch, *, allow_missing_targets=False):
        return {
            "patch_logits": torch.randn(4, 2),
            "graph_emb": torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [14.0, 14.0]]),
            "node_emb": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            "bag_logits": torch.randn(2, 1),
            "ordered_bag_ids": ["b0", "b1"],
            "bag_attention": None,
            "bag_indices": [[0, 1], [2, 3]],
            "bag_targets": None,
            "bag_weights": None,
        }

    def _build_predict_group_metadata(
        _batch,
        *,
        patch_logits,
        graph_emb,
        ordered_bag_ids,
        bag_indices,
        include_instance_payload=True,
    ):
        return {
            "row_region_ids": ["r0", "r1"],
            "row_sample_ids": ["s0", "s1"],
        }

    module._forward_bags = _forward_bags  # type: ignore[method-assign]
    module._build_predict_group_metadata = _build_predict_group_metadata  # type: ignore[method-assign]
    module._predict_emit_instance_payload = False
    module._predict_emit_embeddings_payload = True
    module._predict_emit_node_embeddings = True

    class _Batch:
        region_id = ["r0", "r0", "r1", "r1"]
        sample_id = ["s0", "s0", "s1", "s1"]
        batch = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    out = module.predict_step(batch=_Batch(), batch_idx=0)
    assert out["embedding_bag_ids"] == ["b0", "b1"]
    assert torch.allclose(out["graph_embeddings"][0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(out["graph_embeddings"][1], torch.tensor([12.0, 12.0]))
    assert out["embedding_bag_counts"] == [2, 2]
    assert "node_embeddings" in out
    assert out["node_bag_ids"] == ["s0::r0", "s0::r0", "s1::r1", "s1::r1"]


def test_supervised_predict_step_omits_embedding_payload_by_default():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "binary"},
    )

    def _forward_bags(_batch, *, allow_missing_targets=False):
        return {
            "patch_logits": torch.randn(2, 1),
            "graph_emb": torch.randn(2, 2),
            "node_emb": torch.randn(2, 2),
            "bag_logits": torch.randn(1, 1),
            "ordered_bag_ids": ["b0"],
            "bag_attention": None,
            "bag_indices": [[0, 1]],
            "bag_targets": None,
            "bag_weights": None,
        }

    def _build_predict_group_metadata(
        _batch,
        *,
        patch_logits,
        graph_emb,
        ordered_bag_ids,
        bag_indices,
        include_instance_payload=True,
    ):
        return {
            "row_region_ids": ["r0"],
            "row_sample_ids": ["s0"],
        }

    module._forward_bags = _forward_bags  # type: ignore[method-assign]
    module._build_predict_group_metadata = _build_predict_group_metadata  # type: ignore[method-assign]
    module._predict_emit_instance_payload = False

    class _Batch:
        region_id = ["r0", "r0"]
        sample_id = ["s0", "s0"]

    out = module.predict_step(batch=_Batch(), batch_idx=0)
    assert "embedding_bag_ids" not in out
    assert "graph_embeddings" not in out
    assert "embedding_bag_counts" not in out


def test_extract_bag_ids_falls_back_per_item_when_primary_missing():
    from types import SimpleNamespace

    from grass_mil.models.training.bagging import extract_bag_ids

    batch = SimpleNamespace(region_id=[None, "", "r2"], sample_id=["s0", "s1", "s2"])
    bag_ids = extract_bag_ids(batch, bag_key="region_id", bag_fallback_key="sample_id")
    assert bag_ids == ["s0", "s1", "s2::r2"]


def test_extract_bag_ids_namespaces_primary_by_fallback_for_uniqueness():
    from types import SimpleNamespace

    from grass_mil.models.training.bagging import extract_bag_ids

    batch = SimpleNamespace(region_id=["r0", "r0"], sample_id=["s0", "s1"])
    bag_ids = extract_bag_ids(batch, bag_key="region_id", bag_fallback_key="sample_id")
    assert bag_ids == ["s0::r0", "s1::r0"]


def test_extract_bag_ids_patch_key_uses_sample_region_patch_hierarchy():
    from types import SimpleNamespace

    from grass_mil.models.training.bagging import extract_bag_ids

    batch = SimpleNamespace(
        patch_id=["p0", "p1"],
        region_id=["r0", "r1"],
        sample_id=["s0", "s0"],
    )
    bag_ids = extract_bag_ids(batch, bag_key="patch_id", bag_fallback_key="sample_id")
    assert bag_ids == ["s0::r0::p0", "s0::r1::p1"]


def test_extract_bag_ids_patch_key_fallback_uses_sample_region_when_patch_missing():
    from types import SimpleNamespace

    from grass_mil.models.training.bagging import extract_bag_ids

    batch = SimpleNamespace(
        patch_id=[None, ""],
        region_id=["r0", "r1"],
        sample_id=["s0", "s1"],
    )
    bag_ids = extract_bag_ids(batch, bag_key="patch_id", bag_fallback_key="sample_id")
    assert bag_ids == ["s0::r0", "s1::r1"]


def test_predict_group_metadata_uses_batch_patch_ids_for_instance_patch_ids():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from types import SimpleNamespace

    from grass_mil.models.supervised_module import SupervisedModule

    module = SupervisedModule(
        encoder={},
        graph_head={},
        loss={},
        optim={},
        task={"aggregation": "mean", "target_type": "binary"},
    )
    batch = SimpleNamespace(
        region_id=["r0", "r1", "r0", "r1"],
        sample_id=["s0", "s0", "s0", "s0"],
        patch_id=["p0", "p1", "p2", None],
    )
    metadata = module._build_predict_group_metadata(
        batch,
        patch_logits=torch.randn(4, 1),
        graph_emb=torch.randn(4, 8),
        ordered_bag_ids=["bag_a", "bag_b"],
        bag_indices=[[0, 2], [1, 3]],
    )
    assert metadata["instance_patch_ids"] == ["p0", "p2", "p1", "bag_b"]


def test_gather_bag_targets_requires_consistent_labels_within_bag():
    from types import SimpleNamespace

    from grass_mil.models.training.bagging import gather_bag_targets

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


def test_region_buffer_flushes_in_hyperbatch_chunks():
    pytest.importorskip("lightning")
    pytest.importorskip("torch_geometric")
    from grass_mil.models.supervised_module import SupervisedModule

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
