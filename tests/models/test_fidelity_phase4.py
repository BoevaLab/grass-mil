"""Phase 4: BGRL "A3" pretraining fidelity (merged-unit sampling, momentum schedule,
legacy checkpoint import, preset end to end)."""

from __future__ import annotations


import numpy as np
import pytest
import torch

from tests.helpers.synthetic_cohort import cohort_overrides, compose_cfg, make_synthetic_cohort
from tests.models.test_fidelity_phase2 import _runtime, _units

pytest.importorskip("torch_geometric")
pytest.importorskip("torch_sparse")


def test_merge_graph_units_offsets_edges_and_tracks_units():
    from grass_mil.data.components.merge import merge_graph_units

    units = _units(3, 40)
    merged = merge_graph_units(units)
    assert merged.num_nodes == 120
    assert merged.x.shape == (120, 4)
    assert merged.edge_index.shape[1] == 3 * 160
    assert merged.edge_index[:, 160:320].min() >= 40 and merged.edge_index[:, 160:320].max() < 80
    assert merged.unit_index.tolist() == [0] * 40 + [1] * 40 + [2] * 40
    assert merged.unit_region_ids == ["region_0", "region_1", "region_2"]
    assert merged.unit_graph_y.shape == (3, 1)


def test_merge_units_sampler_mixes_regions_and_covers_all_roots():
    from grass_mil.data.components.samplers import get_sampler_strategy

    units = _units(3, 40)
    strategy = get_sampler_strategy(
        {
            "name": "shadow_custom",
            "kwargs": {},
            "runtime": _runtime(
                merge_units=True, subgraph_batch_size=32, depth=1, num_neighbors=-1
            ),
        }
    )
    loader = strategy.build_dataset_loader(dataset=units, batch_size=1, shuffle=True)
    assert len(loader) == int(np.ceil(120 / 32))
    batches = list(loader)
    assert len(batches) == len(loader)
    n_graphs = sum(int(b.ptr.numel() - 1) for b in batches)
    assert n_graphs == 120  # every root of every unit exactly once
    # A full-size batch spans more than one unit: units are pooled, not iterated.
    first = batches[0]
    unit_of_graph = first.unit_index[first.ptr[:-1]]
    assert len(set(unit_of_graph.tolist())) > 1


def test_momentum_schedule_cosine_to_final_without_warmup():
    from grass_mil.models.training.ssl_runtime import CosineWarmup

    sched = CosineWarmup(base_value=0.99, warmup_steps=0, total_steps=1000, min_value=1.0)
    assert sched.value(0) == pytest.approx(0.99)
    assert sched.value(500) == pytest.approx(0.995)
    assert sched.value(1000) == pytest.approx(1.0)
    assert all(sched.value(s) <= sched.value(s + 1) + 1e-12 for s in range(999))


def test_bgrl_module_reads_momentum_final_and_zero_momentum_warmup():
    from grass_mil.models.bgrl_module import BGRLModule

    module = BGRLModule(
        encoder={"input_dim": 4, "hidden_dim": 8, "out_dim": 8, "num_layers": 1},
        ssl={"method": "bgrl", "predictor": {"hidden_size": 8}},
        task={
            "warmup_steps": 100,
            "total_steps": 1000,
            "momentum": 0.99,
            "momentum_final": 1.0,
            "momentum_warmup_steps": 0,
        },
    )
    assert module._momentum_min == 1.0
    assert module._momentum_warmup_steps == 0
    legacy = BGRLModule(
        encoder={"input_dim": 4, "hidden_dim": 8, "out_dim": 8, "num_layers": 1},
        ssl={"method": "bgrl", "predictor": {"hidden_size": 8}},
        task={"warmup_steps": 100, "total_steps": 1000, "momentum": 0.99, "momentum_min": 0.99},
    )
    assert legacy._momentum_min == 0.99 and legacy._momentum_warmup_steps == 100


def test_spacegm_bgrl_remap_loads_into_public_encoder():
    from grass_mil.models.components.backbones import EncoderConfig
    from grass_mil.models.components.factory import build_encoder
    from grass_mil.models.training.checkpoint_init import remap_encoder_keys

    cfg = EncoderConfig(
        input_dim=1,
        hidden_dim=16,
        out_dim=16,
        num_layers=2,
        conv_type="gine",
        norm="layernorm",
        pooling="max",
        use_edge_attr=True,
        edge_attr_dim=1,
        edge_feature_index=0,
        train_eps=True,
        categorical_embedding={
            "label": "cell_type",
            "num_embeddings": 5,
            "column_index": 0,
            "reserve_unassigned": True,
        },
    )
    enc = build_encoder(cfg)
    ref = {k: v.clone() for k, v in enc.state_dict().items()}
    legacy = {}
    for k, v in ref.items():
        if k == "input_proj.embedding.weight":
            legacy["online_encoder.x_embedding.weight"] = v + 1
        elif k.startswith("input_proj.linear."):
            legacy[k.replace("input_proj.linear.", "online_encoder.feat_embedding.")] = v + 1
        elif k.startswith("layers."):
            i, rest = k.split(".", 2)[1], k.split(".", 2)[2]
            legacy[f"online_encoder.gnns.{i}.gine.{rest}"] = v + 1
            legacy[f"online_encoder.gnns.{i}.mlp.0.weight"] = torch.zeros(1)
            legacy[f"online_encoder.gnns.{i}.edge_embedding.weight"] = torch.zeros(4, 16)
        elif k.startswith("norms."):
            legacy[k.replace("norms.", "online_encoder.batch_norms.")] = v + 1
    legacy["target_encoder.x_embedding.weight"] = torch.zeros(1)
    legacy["predictor.net.0.weight"] = torch.zeros(1)
    remapped = remap_encoder_keys(legacy, encoder_init_map="spacegm_bgrl")
    assert set(remapped) == {f"encoder.{k}" for k in ref}
    stripped = {k[len("encoder.") :]: v for k, v in remapped.items()}
    enc.load_state_dict(stripped, strict=True)
    assert torch.allclose(enc.state_dict()["layers.0.eps"], ref["layers.0.eps"] + 1)
    with pytest.raises(ValueError):
        remap_encoder_keys({"foo": torch.zeros(1)}, encoder_init_map="spacegm_bgrl")


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort4")
    c = make_synthetic_cohort(root, n_patients=4, regions_per_patient=2, n_cells=100)
    import hydra

    cfg = compose_cfg("train", cohort_overrides(c, root / "out_precompute"))
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    return c


def test_pretrain_a3_preset_then_finetune_from_checkpoint(cohort, tmp_path):
    from grass_mil.train import train

    pre_out = tmp_path / "pretrain"
    base = [
        o
        for o in cohort_overrides(cohort, pre_out)
        if not o.startswith(("regime=", "data.val_sampler", "model.attention", "task."))
    ]
    cfg = compose_cfg(
        "train",
        base
        + [
            "experiment=grass_mil_pretrain_a3",
            "task.total_steps=4",
            "task.warmup_steps=1",
            "task.lr_warmup.warmup_steps=1",
            "task.augmentation.feature_noise_columns=[area]",
            "callbacks.model_checkpoint.every_n_train_steps=2",
            "trainer.precision=32",
            "data.sampler.runtime.depth=2",
            "data.sampler.runtime.subgraph_batch_size=16",
        ],
    )
    assert cfg.task.momentum_final == 1.0 and cfg.data.sampler.runtime.merge_units is True
    assert cfg.trainer.max_steps == 4 and cfg.scheduler.T_max == "auto"
    metrics, objects = train(cfg)
    module = objects["model"]
    assert module.encoder.dropout.p == pytest.approx(0.1)
    assert "train/loss" in metrics
    ckpts = sorted((pre_out / "checkpoints").glob("*.ckpt"))
    names = {p.name for p in ckpts}
    assert "last.ckpt" in names and any(n.startswith("step_") for n in names)
    # the scheduler covered total_steps - warmup and momentum reached its final value
    # last update used the momentum of zero-based step 3 of 4 (cosine 0.99 -> 1.0)
    import math

    expected = 1.0 + (0.99 - 1.0) * 0.5 * (1.0 + math.cos(math.pi * 3 / 4))
    assert float(metrics["train/momentum"]) == pytest.approx(expected, abs=1e-5)
    pre_state = torch.load(pre_out / "checkpoints" / "last.ckpt", weights_only=False)["state_dict"]
    enc_keys = [
        k for k in pre_state if k.startswith("ssl_model.online_encoder.layers.0.nn.0.weight")
    ]
    assert enc_keys

    ft_out = tmp_path / "finetune"
    ft_cfg = compose_cfg(
        "train",
        cohort_overrides(cohort, ft_out)
        + [
            f"model.init_from_ckpt={pre_out / 'checkpoints' / 'last.ckpt'}",
            "trainer.max_epochs=1",
        ],
    )
    _, ft_objects = train(ft_cfg)
    ft_module = ft_objects["model"]
    # the fine-tuned model started from the pretrained encoder weights
    loaded = ft_module.encoder.state_dict()["norms.0.weight"].detach().cpu()
    pretrained = pre_state["ssl_model.online_encoder.norms.0.weight"].detach().cpu()
    assert torch.allclose(loaded, pretrained, atol=5e-3)
    assert ft_module.encoder.dropout.p == 0.0


def test_zero_validation_callback_filter_keeps_interpolations_resolvable():
    from omegaconf import OmegaConf

    from grass_mil.train import _filter_val_monitor_callbacks

    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": "/tmp/run"},
            "callbacks": {
                "model_checkpoint": {
                    "_target_": "x",
                    "dirpath": "${paths.output_dir}/checkpoints",
                    "monitor": None,
                },
                "early_stopping": {"_target_": "y", "monitor": "val/loss"},
            },
        }
    )
    filtered, removed = _filter_val_monitor_callbacks(cfg.callbacks)
    assert removed == ["early_stopping"]
    assert filtered.model_checkpoint.dirpath == "/tmp/run/checkpoints"
