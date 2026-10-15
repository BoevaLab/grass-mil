"""Encoder, attention pooling, objectives, probe, augmentations and optimisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from grass_mil.data.datamodule import InputSpec
from grass_mil.models import SupervisedModule, attention_pool, pool_by_bag
from grass_mil.models.augmentations import drop_edges, drop_importance
from grass_mil.models.encoder import EncoderConfig, GINEEncoder
from grass_mil.models.objectives import (
    Objective,
    class_weights_from_labels,
    concordance_index,
    cox_loss,
)
from grass_mil.models.probe import ProbeConfig, fit_linear_head

SPEC = InputSpec(feature_names=("f",), cell_types=("a", "b", "c"), edge_features=("d", "n"))


def _batch(n_graphs: int = 3, n: int = 6) -> Batch:
    graphs = []
    for _ in range(n_graphs):
        edge_index = torch.tensor(
            [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
        ).t()
        graphs.append(
            Data(
                x=torch.randn(n, 1),
                cell_type=torch.randint(0, 3, (n,)),
                edge_index=edge_index,
                edge_attr=torch.rand(edge_index.size(1), 2),
            )
        )
    return Batch.from_data_list(graphs)


def _module(**finetune) -> SupervisedModule:
    module = SupervisedModule(
        encoder={"hidden_dim": 8, "out_dim": 8, "num_layers": 2},
        head={"num_layers": 1},
        attention={"projection_dim": 4, "hidden_dim": 4},
        task={"target_type": "classification", "num_classes": 2, "class_weights": None},
        finetune={"lr_scales": {"encoder": 1.0, "head": 1.0, "attention": 1.0}, **finetune},
        optim={"_target_": "torch.optim.AdamW", "lr": 1e-3},
        scheduler=None,
    )
    module.build(SPEC)
    return module


def test_encoder_needs_some_input_and_embeds_cell_types_only():
    with pytest.raises(ValueError):
        GINEEncoder(EncoderConfig(), feature_dim=0, cell_types=(), edge_dim=1)
    enc = GINEEncoder(
        EncoderConfig(hidden_dim=8, out_dim=4, num_layers=2),
        feature_dim=0,
        cell_types=("a", "b"),
        edge_dim=1,
    )
    batch = _batch()
    batch.x = torch.zeros(batch.num_nodes, 0)
    batch.cell_type = batch.cell_type % 2
    batch.edge_attr = batch.edge_attr[:, :1]
    assert enc(batch).shape == (3, 4)
    assert enc.unassigned_index == 2  # reserved extra row


def test_attention_pooling_is_an_exact_sum_of_contributions():
    logits, scores = torch.randn(10, 2), torch.randn(10, 2)
    bag, weights = attention_pool(logits, scores)
    assert torch.allclose(weights.sum(0), torch.ones(2))
    assert torch.allclose((weights * logits).sum(0), bag)
    names, pooled = pool_by_bag(
        torch.cat([logits, logits]), torch.cat([scores, scores]), ["x"] * 10 + ["y"] * 10
    )
    assert (
        names == ["x", "y"] and torch.allclose(pooled[0], bag) and torch.allclose(pooled[1], bag)
    )


def test_attention_needs_one_channel_per_class():
    with pytest.raises(ValueError, match="one channel per class"):
        attention_pool(torch.randn(5, 2), torch.randn(5, 1))


def test_class_weights_inverse_frequency_mean_one():
    w = class_weights_from_labels(np.array([0, 0, 0, 1]), 2)
    assert torch.allclose(w.mean(), torch.tensor(1.0)) and w[1] > w[0]


def test_cox_loss_is_the_breslow_partial_likelihood_with_ties():
    risk = torch.tensor([0.5, -0.2, 1.0, 0.0, 0.3, -1.0], requires_grad=True)
    time = torch.tensor([3.0, 5.0, 1.0, 3.0, 3.0, 1.0])
    event = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    loss = cox_loss(risk, time, event)
    expected = 0.0
    for i in range(6):
        if event[i]:
            at_risk = time >= time[i]  # every subject tied with i is at risk
            expected -= float(risk[i] - torch.logsumexp(risk[at_risk], 0))
    assert loss.item() == pytest.approx(expected / 4, rel=1e-5)
    loss.backward()
    assert risk.grad is not None and torch.isfinite(risk.grad).all()


def test_concordance_counts_censoring_at_an_event_time_as_at_risk():
    times, events = np.array([1.0, 2, 3]), np.array([1, 1, 1])
    assert concordance_index(np.array([3.0, 2, 1]), times, events) == 1.0
    # Subject 1 is censored at subject 0's event time: the pair is comparable.
    risk, times, events = np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([1, 0])
    assert concordance_index(risk, times, events) == 0.0


def test_survival_objective_needs_time_and_event():
    with pytest.raises(ValueError, match="time, event"):
        Objective({"target_type": "survival", "target_columns": ["t"]})


def test_probe_head_reproduces_the_logistic_decision():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(float)
    weight, bias = fit_linear_head(
        x, y[:, None], kind="classification", num_logits=2, cfg=ProbeConfig(class_weight=None)
    )
    probs = torch.softmax(torch.from_numpy(x) @ weight.T + bias, 1)[:, 1].numpy()
    reference = LogisticRegression(max_iter=3000).fit(StandardScaler().fit_transform(x), y)
    expected = reference.predict_proba(StandardScaler().fit(x).transform(x))[:, 1]
    assert np.allclose(probs, expected, atol=1e-4)


def test_cox_probe_ranks_by_the_risk_feature():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4)).astype(np.float32)
    time = rng.exponential(np.exp(-1.5 * x[:, 0]))
    event = (rng.random(300) < 0.8).astype(float)
    weight, bias = fit_linear_head(
        x, np.stack([time, event], 1), kind="survival", num_logits=1, cfg=ProbeConfig()
    )
    risk = (torch.from_numpy(x) @ weight.T + bias)[:, 0].numpy()
    assert weight[0, 0] > 3 * weight[0, 1:].abs().max()
    assert concordance_index(risk, time, event) > 0.7


def test_augmentations_keep_graphs_undirected_and_inputs_intact():
    batch = _batch()
    original = batch.edge_index.clone()
    gen = torch.Generator().manual_seed(0)
    out = drop_edges(batch, p=0.5, generator=gen)
    pairs = set(map(tuple, out.edge_index.t().tolist()))
    assert all((b, a) in pairs for a, b in pairs)
    assert torch.equal(batch.edge_index, original)
    masked = drop_importance(batch, mu=1.0, p_lambda=1.0, unassigned_index=3, generator=gen)
    changed = masked.cell_type != batch.cell_type
    assert bool((masked.cell_type[changed] == 3).all())
    assert bool((masked.x[changed] == 0).all())


def test_gradient_clipping_happens_in_the_optimizer_hook():
    module = _module(grad_clip_norm=1.0)
    for p in module.parameters():
        p.grad = torch.full_like(p, 10.0)
    module.on_before_optimizer_step(None)
    total = torch.sqrt(sum((p.grad**2).sum() for p in module.parameters()))
    assert total.item() == pytest.approx(1.0, rel=1e-4)


def test_lr_scales_must_be_numbers():
    with pytest.raises(ValueError, match="lr_scales"):
        SupervisedModule(
            encoder={},
            head={},
            attention={},
            task={"target_type": "classification"},
            finetune={"lr_scales": {"encoder": None, "head": 1.0, "attention": 1.0}},
        )


def test_zero_scale_freezes_the_encoder_in_eval_mode():
    module = _module(lr_scales={"encoder": 0.0, "head": 1.0, "attention": 2.0})
    module._apply_freezing()
    module.train()
    assert not any(p.requires_grad for p in module.encoder.parameters())
    assert not module.encoder.training and module.attention.training
