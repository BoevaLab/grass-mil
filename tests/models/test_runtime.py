import torch


def test_remap_encoder_keys_auto_bgrl_or_identity():
    from src.models.runtime import remap_encoder_keys

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
        WeightedMSELoss,
    )
    from src.models.runtime import compute_supervised_loss

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


def test_attention_bag_aggregation_shapes():
    import torch
    from src.models.components.attention import AttnNetGatedProjected
    from src.models.runtime import aggregate_bag_logits_attention

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
    from src.models.runtime import build_mil_aux_targets

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
    from src.models.runtime import (
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
