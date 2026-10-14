"""Compose the model/encoder and model/attention groups through Hydra and build them.

The groups ship as plain parameter dictionaries (no ``_target_``) that the
module factories turn into components, so the only faithful test is to compose
them the way the CLI does and instantiate the resulting module.
"""

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from tests.helpers.config_paths import CONFIGS_DIR

CONFIG_MODULE = "grass_mil.configs"
ENCODERS = ["gin", "gcn", "gat", "graphsage", "gine", "gine_length"]
ATTENTIONS = ["gated", "gated_projected"]


def _compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module=CONFIG_MODULE):
        cfg = compose(config_name="train", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


@pytest.mark.parametrize("encoder", ENCODERS)
def test_encoder_group_builds(encoder: str) -> None:
    from grass_mil.models.components.backbones import EncoderConfig, GNNEncoder

    cfg = _compose([f"model/encoder={encoder}", "model.encoder.input_dim=8"])
    enc_cfg = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    if enc_cfg.get("categorical_embedding") is not None:
        enc_cfg["categorical_embedding"] = {
            **enc_cfg["categorical_embedding"],
            "num_embeddings": 5,
            "column_index": 0,
        }
    encoder_module = GNNEncoder(EncoderConfig(**enc_cfg))
    assert encoder_module.output_dim == int(cfg.model.encoder.out_dim)


@pytest.mark.parametrize("attention", ATTENTIONS)
def test_attention_group_builds(attention: str) -> None:
    from grass_mil.models.components.factory import build_attention

    cfg = _compose([f"model/attention={attention}", "task=finetune_mil"])
    attn_cfg = OmegaConf.to_container(cfg.model.attention, resolve=True)
    module = build_attention(use_attention=True, **attn_cfg)
    assert module is not None
    assert module.attention_c.out_features == int(cfg.model.graph_head.output_dim)


@pytest.mark.parametrize(
    ("model_name", "task_name"),
    [
        ("bgrl_module", "pretrain_bgrl"),
        ("supervised_module", "finetune_mean"),
        ("supervised_module", "finetune_mil"),
        ("grass_mil", "finetune_mil"),
    ],
)
def test_training_module_presets_instantiate(model_name: str, task_name: str) -> None:
    import hydra

    cfg = _compose([f"model={model_name}", f"task={task_name}"])
    module = hydra.utils.instantiate(cfg.model)
    assert module is not None


def test_grass_mil_preset_defaults_match_preprint() -> None:
    cfg = _compose(["model=grass_mil", "task=finetune_mil"])
    enc = cfg.model.encoder
    assert enc.conv_type == "gine" and int(enc.num_layers) == 4 and int(enc.hidden_dim) == 256
    assert enc.norm == "layernorm" and enc.pooling == "max" and bool(enc.train_eps)
    assert int(enc.edge_feature_index) == 0 and int(enc.edge_attr_dim) == 1
    att = cfg.model.attention
    assert int(att.projection_dim) == 32 and int(att.hidden_dim) == 8 and bool(att.dropout)
    assert int(att.n_classes) == 2 and int(cfg.model.graph_head.output_dim) == 2
    assert int(cfg.model.graph_head.num_layers) == 2


def test_sampler_strategy_resolution():
    from grass_mil.data.components.samplers import get_sampler_strategy

    base = CONFIGS_DIR / "data" / "sampler"
    for name in ["identity", "shadow_native", "shadow_custom"]:
        cfg = OmegaConf.load(base / f"{name}.yaml")
        strategy = get_sampler_strategy(OmegaConf.to_container(cfg, resolve=True))
        assert strategy is not None


def test_default_data_sampler_config():
    cfg = _compose([])
    assert cfg.data.sampler.name == "shadow_custom"
    cfg_mil = _compose(["data/sampler=ego_graph_mil"])
    assert cfg_mil.data.sampler.runtime.unit_as_bag is True
