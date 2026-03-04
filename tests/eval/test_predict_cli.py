from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from omegaconf import OmegaConf

from src.inference.schemas import (
    AggregatedPredictionPayload,
    BatchPredictionPayload,
    CollectedInferencePayload,
    EmbeddingPayload,
)
from src.inference.predict import (
    _needs_instance_payload,
    _plan_preforward_subsampling,
    predict,
)
from src.train import train


@pytest.mark.slow
def test_predict_cli_smoke(cfg_train: DictConfig, cfg_predict: DictConfig, tmp_path: Path) -> None:
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True
    HydraConfig().set_config(cfg_train)
    _, _ = train(cfg_train)

    ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
    assert ckpt_path.exists()

    with open_dict(cfg_predict):
        cfg_predict.ckpt_path = str(ckpt_path)
        cfg_predict.embeddings.extract_node = False
        cfg_predict.predict.output_subdir = "predict_artifacts"
    HydraConfig().set_config(cfg_predict)

    summary, _ = predict(cfg_predict)
    assert summary["row_count"] > 0

    output_dir = tmp_path / "predict_artifacts"
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "graph_embeddings.csv").exists()


class _SamplerCfg:
    def __init__(self, *, name: str, runtime: dict):
        self.name = name
        self.runtime = runtime


class _DataModule:
    def __init__(self, *, sampler_name: str, runtime: dict):
        self.sampler_config = _SamplerCfg(name=sampler_name, runtime=runtime)
        self.sampler_strategy = object()
        self.val_sampler_strategy = object()


def test_plan_preforward_subsampling_applies_for_enabled_runtime_sampler() -> None:
    datamodule = _DataModule(sampler_name="shadow_native", runtime={"enabled": True})
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "subsample_fraction": 0.5,
                "subsample_seed": 123,
            }
        }
    )
    decision = _plan_preforward_subsampling(datamodule, cfg)
    assert decision.requested is True
    assert decision.effective is True
    assert decision.reason == "applied"
    assert decision.strategy_name == "shadow_native"
    assert decision.runtime_enabled is True
    assert datamodule.sampler_config.runtime["subsample_fraction"] == 0.5
    assert datamodule.sampler_config.runtime["subsample_seed"] == 123
    assert datamodule.sampler_strategy is None
    assert datamodule.val_sampler_strategy is None


def test_plan_preforward_subsampling_not_requested_for_full_fraction() -> None:
    datamodule = _DataModule(sampler_name="shadow_native", runtime={"enabled": True})
    datamodule.sampler_strategy = "keep"
    datamodule.val_sampler_strategy = "keep_val"
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "max",
                "subsample_fraction": 1.0,
                "subsample_seed": 7,
            }
        }
    )
    decision = _plan_preforward_subsampling(datamodule, cfg)
    assert decision.requested is False
    assert decision.effective is False
    assert decision.reason == "not_requested"
    assert "subsample_fraction" not in datamodule.sampler_config.runtime
    assert datamodule.sampler_strategy == "keep"
    assert datamodule.val_sampler_strategy == "keep_val"


def test_plan_preforward_subsampling_identity_sampler_is_ineffective() -> None:
    datamodule = _DataModule(sampler_name="identity", runtime={"enabled": True})
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "subsample_fraction": 0.5,
                "subsample_seed": 7,
            }
        }
    )
    decision = _plan_preforward_subsampling(datamodule, cfg)
    assert decision.requested is True
    assert decision.effective is False
    assert decision.reason == "identity_sampler"
    assert decision.strategy_name == "identity"
    assert decision.runtime_enabled is None


def test_plan_preforward_subsampling_runtime_disabled_is_ineffective() -> None:
    datamodule = _DataModule(sampler_name="shadow_custom", runtime={"enabled": False})
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "max",
                "subsample_fraction": 0.5,
                "subsample_seed": 11,
            }
        }
    )
    decision = _plan_preforward_subsampling(datamodule, cfg)
    assert decision.requested is True
    assert decision.effective is False
    assert decision.reason == "runtime_disabled"
    assert decision.strategy_name == "shadow_custom"
    assert decision.runtime_enabled is False


def test_plan_preforward_subsampling_unsupported_mode_is_ineffective() -> None:
    datamodule = _DataModule(sampler_name="shadow_native", runtime={"enabled": True})
    cfg = OmegaConf.create(
        {
            "aggregation": {
                "enabled": True,
                "mode": "none",
                "subsample_fraction": 0.5,
                "subsample_seed": 5,
            }
        }
    )
    decision = _plan_preforward_subsampling(datamodule, cfg)
    assert decision.requested is True
    assert decision.effective is False
    assert decision.reason == "unsupported_mode"


@pytest.mark.parametrize(
    ("cfg_dict", "expected"),
    [
        ({"aggregation": {"enabled": False, "mode": "mean"}}, False),
        ({"aggregation": {"enabled": True, "mode": "none"}}, False),
        ({"aggregation": {"enabled": True, "mode": "mean"}}, True),
        ({"aggregation": {"enabled": True, "mode": "max"}}, True),
        ({"aggregation": {"enabled": True, "mode": "attention_weighted"}}, True),
    ],
)
def test_needs_instance_payload(cfg_dict, expected) -> None:
    cfg = OmegaConf.create(cfg_dict)
    assert _needs_instance_payload(cfg) is expected


def test_predict_raises_when_preforward_subsampling_is_requested_but_ineffective(
    monkeypatch, tmp_path: Path
) -> None:
    import src.inference.predict as predict_module

    datamodule = _DataModule(sampler_name="identity", runtime={"enabled": True})

    def _fake_instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "fake.DataModule":
            return datamodule
        if target == "fake.Model":
            return SimpleNamespace()
        if target == "fake.Trainer":
            return SimpleNamespace()
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(predict_module, "instantiate_loggers", lambda cfg: [])
    monkeypatch.setattr(predict_module, "log_hyperparameters", lambda obj: None)

    cfg = OmegaConf.create(
        {
            "ckpt_path": "dummy.ckpt",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "bag_scope": "region",
                "subsample_fraction": 0.5,
                "subsample_seed": 17,
            },
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {"enabled": False, "save": False, "extract_node": False},
            "task": {"target_type": "binary"},
        }
    )
    with pytest.raises(ValueError, match="Preforward subsampling was requested"):
        predict(cfg)


def test_predict_disables_post_subsampling_only_when_preforward_is_effective(
    monkeypatch, tmp_path: Path
) -> None:
    import src.inference.predict as predict_module

    datamodule = _DataModule(sampler_name="shadow_native", runtime={"enabled": True})
    captured = {}

    def _fake_instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "fake.DataModule":
            return datamodule
        if target == "fake.Model":
            return SimpleNamespace()
        if target == "fake.Trainer":
            return SimpleNamespace()
        raise AssertionError(f"Unexpected instantiate target: {target}")

    def _fake_aggregate(payload, **kwargs):
        captured.update(kwargs)
        return AggregatedPredictionPayload(
            bag_ids=["r0"],
            bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
            bag_targets=None,
            bag_attention=None,
            metadata={"mode": "mean", "bag_scope": "region"},
            row_region_ids=["r0"],
            row_sample_ids=["s0"],
        )

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(predict_module, "instantiate_loggers", lambda cfg: [])
    monkeypatch.setattr(predict_module, "log_hyperparameters", lambda obj: None)
    monkeypatch.setattr(
        predict_module,
        "collect_inference_payload",
        lambda **kwargs: CollectedInferencePayload(
            prediction_payload=BatchPredictionPayload(
                bag_ids=["r0"],
                bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
                bag_targets=None,
                bag_attention=None,
                row_region_ids=["r0"],
                row_sample_ids=["s0"],
                instance_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
                instance_attention_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
                instance_patch_ids=["p0"],
                instance_region_ids=["r0"],
                instance_sample_ids=["s0"],
            ),
            embedding_payload=None,
        ),
    )
    monkeypatch.setattr(predict_module, "aggregate_group_logits", _fake_aggregate)
    monkeypatch.setattr(
        predict_module,
        "predictions_to_dataframe",
        lambda payload, include_targets, include_attention: pd.DataFrame(
            {"bag_id": ["r0"], "logit_0": [1.0]}
        ),
    )

    cfg = OmegaConf.create(
        {
            "ckpt_path": "dummy.ckpt",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "bag_scope": "region",
                "subsample_fraction": 0.5,
                "subsample_seed": 17,
            },
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {"enabled": False, "save": False, "extract_node": False},
            "task": {"target_type": "binary"},
        }
    )
    summary, _ = predict(cfg)

    assert captured["subsample_fraction"] == 1.0
    assert captured["subsample_seed"] is None
    assert summary["aggregation"]["preforward_subsampling_applied"] is True
    assert summary["aggregation"]["preforward_subsampling"]["effective"] is True


def test_predict_writes_embeddings_from_single_pass_collected_payload(
    monkeypatch, tmp_path: Path
) -> None:
    import src.inference.predict as predict_module

    datamodule = _DataModule(sampler_name="shadow_native", runtime={"enabled": True})
    wrote_paths = []

    def _fake_instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "fake.DataModule":
            return datamodule
        if target == "fake.Model":
            return SimpleNamespace()
        if target == "fake.Trainer":
            return SimpleNamespace()
        raise AssertionError(f"Unexpected instantiate target: {target}")

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(predict_module, "instantiate_loggers", lambda cfg: [])
    monkeypatch.setattr(predict_module, "log_hyperparameters", lambda obj: None)
    monkeypatch.setattr(
        predict_module,
        "collect_inference_payload",
        lambda **kwargs: CollectedInferencePayload(
            prediction_payload=BatchPredictionPayload(
                bag_ids=["r0"],
                bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
                bag_targets=None,
                bag_attention=None,
                row_region_ids=["r0"],
                row_sample_ids=["s0"],
            ),
            embedding_payload=EmbeddingPayload(
                bag_ids=["r0"],
                graph_embeddings=OmegaConf.create([[0.1, 0.2]]),  # type: ignore[arg-type]
                node_embeddings=OmegaConf.create([[0.3, 0.4]]),  # type: ignore[arg-type]
                node_bag_ids=["r0"],
            ),
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "aggregate_group_logits",
        lambda *args, **kwargs: AggregatedPredictionPayload(
            bag_ids=["r0"],
            bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
            bag_targets=None,
            bag_attention=None,
            metadata={"mode": "mean", "bag_scope": "region"},
            row_region_ids=["r0"],
            row_sample_ids=["s0"],
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "predictions_to_dataframe",
        lambda payload, include_targets, include_attention: pd.DataFrame(
            {"bag_id": ["r0"], "logit_0": [1.0]}
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "embeddings_to_dataframe",
        lambda payload: pd.DataFrame({"bag_id": payload.bag_ids, "graph_emb_0": [0.1]}),
    )
    monkeypatch.setattr(
        predict_module,
        "node_embeddings_to_dataframe",
        lambda payload: pd.DataFrame({"bag_id": payload.node_bag_ids, "node_emb_0": [0.3]}),
    )

    def _fake_write_dataframe(frame, path):
        wrote_paths.append(str(path))

    monkeypatch.setattr(predict_module, "write_dataframe", _fake_write_dataframe)

    cfg = OmegaConf.create(
        {
            "ckpt_path": "dummy.ckpt",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "bag_scope": "region",
                "subsample_fraction": 1.0,
                "subsample_seed": None,
            },
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {
                "enabled": True,
                "save": True,
                "extract_node": True,
                "filename": "graph_embeddings.csv",
                "node_filename": "node_embeddings.csv",
            },
            "task": {"target_type": "binary"},
        }
    )

    summary, _ = predict(cfg)
    assert summary["embeddings_path"] is not None
    assert any(path.endswith("graph_embeddings.csv") for path in wrote_paths)
    assert any(path.endswith("node_embeddings.csv") for path in wrote_paths)


def test_predict_summary_includes_aggregation_metadata(monkeypatch, tmp_path: Path) -> None:
    import src.inference.predict as predict_module

    class _FakeDataModule:
        pass

    class _FakeModel:
        pass

    class _FakeTrainer:
        pass

    fake_datamodule = _FakeDataModule()
    fake_model = _FakeModel()
    fake_trainer = _FakeTrainer()

    def _fake_instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "fake.DataModule":
            return fake_datamodule
        if target == "fake.Model":
            return fake_model
        if target == "fake.Trainer":
            return fake_trainer
        raise AssertionError(f"Unexpected instantiate target: {target}")

    captured_summary_payload = {}

    def _fake_write_json(payload, path):
        captured_summary_payload.clear()
        captured_summary_payload.update(payload)

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(predict_module, "instantiate_loggers", lambda cfg: [])
    monkeypatch.setattr(predict_module, "log_hyperparameters", lambda obj: None)
    monkeypatch.setattr(
        predict_module,
        "collect_inference_payload",
        lambda **kwargs: CollectedInferencePayload(
            prediction_payload=BatchPredictionPayload(
                bag_ids=["r0"],
                bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
                bag_targets=None,
                bag_attention=None,
                row_region_ids=["r0"],
                row_sample_ids=["s0"],
            ),
            embedding_payload=None,
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "aggregate_group_logits",
        lambda *args, **kwargs: AggregatedPredictionPayload(
            bag_ids=["r0"],
            bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
            bag_targets=None,
            bag_attention=None,
            metadata={"mode": "mean", "bag_scope": "region"},
            row_region_ids=["r0"],
            row_sample_ids=["s0"],
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "predictions_to_dataframe",
        lambda payload, include_targets, include_attention: pd.DataFrame(
            {"bag_id": ["r0"], "logit_0": [1.0]}
        ),
    )
    monkeypatch.setattr(predict_module, "write_json", _fake_write_json)

    cfg = OmegaConf.create(
        {
            "ckpt_path": "dummy.ckpt",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {
                "enabled": True,
                "mode": "mean",
                "bag_scope": "region",
                "subsample_fraction": 1.0,
                "subsample_seed": None,
            },
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {"enabled": False, "save": False, "extract_node": False},
            "task": {"target_type": "binary"},
        }
    )

    summary, _ = predict(cfg)
    assert "aggregation" in summary
    assert summary["aggregation"]["mode"] == "mean"
    assert summary["aggregation"]["bag_scope"] == "region"
    assert "preforward_subsampling_applied" in summary["aggregation"]
    assert "preforward_subsampling" in summary["aggregation"]
    assert summary["aggregation"]["preforward_subsampling"]["reason"] == "not_requested"
    assert captured_summary_payload["aggregation"]["mode"] == "mean"


def test_predict_raises_when_ckpt_path_missing(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "ckpt_path": "",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {"enabled": False, "mode": "none"},
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {"enabled": False, "save": False, "extract_node": False},
            "task": {"target_type": "binary"},
        }
    )
    with pytest.raises(ValueError, match="Missing required `ckpt_path` for inference."):
        predict(cfg)


def test_predict_can_export_interpretability_tables(monkeypatch, tmp_path: Path) -> None:
    import src.inference.predict as predict_module

    class _FakeDataModule:
        class _Dataset:
            label_maps = {"cell_type": {"A": 0, "B": 1}}

        dataset_train = _Dataset()

    class _FakeModel:
        pass

    class _FakeTrainer:
        pass

    fake_datamodule = _FakeDataModule()
    fake_model = _FakeModel()
    fake_trainer = _FakeTrainer()
    captured = {"collector_kwargs": None, "frames": {}}

    def _fake_instantiate(cfg, *args, **kwargs):
        target = cfg.get("_target_")
        if target == "fake.DataModule":
            return fake_datamodule
        if target == "fake.Model":
            return fake_model
        if target == "fake.Trainer":
            return fake_trainer
        raise AssertionError(f"Unexpected instantiate target: {target}")

    def _fake_collect(**kwargs):
        captured["collector_kwargs"] = kwargs
        return CollectedInferencePayload(
            prediction_payload=BatchPredictionPayload(
                bag_ids=["r0"],
                bag_logits=OmegaConf.create([[1.0]]),  # type: ignore[arg-type]
                bag_targets=None,
                bag_attention=None,
                row_region_ids=["r0"],
                row_sample_ids=["s0"],
                instance_logits=torch.tensor([[0.2], [0.4]]),
                instance_attention_logits=torch.tensor([[0.1], [0.2]]),
                instance_patch_ids=["p0", "p1"],
                instance_bag_ids=["s0::r0", "s0::r0"],
                instance_region_ids=["r0", "r0"],
                instance_sample_ids=["s0", "s0"],
                instance_embeddings=torch.tensor([[0.11, 0.12], [0.21, 0.22]]),
                instance_composition=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                instance_centroids=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            ),
            embedding_payload=None,
        )

    def _fake_write_dataframe(frame, path):
        captured["frames"][str(path)] = frame.copy()

    monkeypatch.setattr(predict_module.hydra.utils, "instantiate", _fake_instantiate)
    monkeypatch.setattr(predict_module, "instantiate_loggers", lambda cfg: [])
    monkeypatch.setattr(predict_module, "log_hyperparameters", lambda obj: None)
    monkeypatch.setattr(predict_module, "collect_inference_payload", _fake_collect)
    monkeypatch.setattr(
        predict_module,
        "predictions_to_dataframe",
        lambda payload, include_targets, include_attention: pd.DataFrame(
            {"bag_id": ["r0"], "logit_0": [1.0]}
        ),
    )
    monkeypatch.setattr(predict_module, "write_dataframe", _fake_write_dataframe)

    cfg = OmegaConf.create(
        {
            "ckpt_path": "dummy.ckpt",
            "data": {"_target_": "fake.DataModule"},
            "model": {"_target_": "fake.Model"},
            "trainer": {"_target_": "fake.Trainer"},
            "logger": None,
            "paths": {"output_dir": str(tmp_path)},
            "predict": {
                "save_predictions": False,
                "save_metrics": False,
                "include_targets": False,
                "include_attention": False,
                "output_subdir": "predict_artifacts",
            },
            "output": {
                "predictions_filename": "predictions.csv",
                "metrics_filename": "metrics.json",
                "summary_filename": "inference_summary.json",
            },
            "aggregation": {"enabled": False, "mode": "none", "subsample_fraction": 1.0},
            "metrics": {"enabled": False, "per_group": False, "categorical": {"threshold": 0.5}},
            "embeddings": {"enabled": False, "save": False, "extract_node": False},
            "interpretability": {
                "enabled": True,
                "instance_filename": "instance_table.csv",
                "spatial_filename": "spatial_table.csv",
                "id_column": "instance_id",
                "bag_id_column": "bag_id",
                "composition_label": "cell_type",
                "composition_prefix": "comp_",
                "embedding_prefix": "inst_emb",
                "score_column": "score",
                "attention_column": "attention",
                "require_composition": True,
                "spatial": {
                    "enabled": True,
                    "x_column": "center_x",
                    "y_column": "center_y",
                    "n_neighbors": 1,
                    "undirected": True,
                },
            },
            "task": {"target_type": "binary"},
        }
    )

    summary, _ = predict(cfg)
    assert captured["collector_kwargs"]["include_instance_payload"] is True
    assert captured["collector_kwargs"]["include_instance_embeddings"] is True
    assert "interpretability" in summary
    assert summary["interpretability"]["instance_table_path"].endswith("instance_table.csv")
    assert summary["interpretability"]["spatial_table_path"].endswith("spatial_table.csv")
    assert any(path.endswith("instance_table.csv") for path in captured["frames"].keys())
    assert any(path.endswith("spatial_table.csv") for path in captured["frames"].keys())
