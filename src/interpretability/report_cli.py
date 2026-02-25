from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.interpretability.core.data import load_interpretability_dataset  # noqa: E402
from src.interpretability.pipeline import run_interpretability_pipeline  # noqa: E402
from src.interpretability.reporting.render import render_interpretability_report  # noqa: E402

log = logging.getLogger(__name__)


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg


def run_report(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset = load_interpretability_dataset(
        instance_table_path=Path(str(cfg.data.instance_table)),
        bag_table_path=Path(str(cfg.data.bag_table)) if cfg.data.get("bag_table") else None,
        spatial_table_path=Path(str(cfg.data.spatial_table))
        if cfg.data.get("spatial_table")
        else None,
        id_column=str(cfg.data.get("id_column", "instance_id")),
        bag_id_column=str(cfg.data.get("bag_id_column", "bag_id")),
        cell_type_column=cfg.data.get("cell_type_column"),
        condition_column=cfg.data.get("condition_column"),
    )
    output_dir = Path(str(cfg.output_dir))
    artifacts_dir = output_dir / "artifacts"
    pipeline_cfg = {
        "embedding_prefixes": list(cfg.embedding_prefixes),
        "composition_prefix": str(cfg.get("composition_prefix", "comp_")),
        "reduction": _cfg_to_dict(cfg.reduction),
        "clustering": _cfg_to_dict(cfg.clustering),
        "plugins": _cfg_to_dict(cfg.plugins),
    }
    bundle = run_interpretability_pipeline(dataset, pipeline_cfg, artifacts_dir=artifacts_dir)
    report_out = render_interpretability_report(
        bundle,
        output_dir=output_dir,
        html_enabled=bool(cfg.report.html.enabled),
        pdf_enabled=bool(cfg.report.pdf.enabled),
        snapshot_dpi_scale=float(cfg.report.pdf.snapshot_dpi_scale),
    )
    summary = {
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "report": report_out,
        "pipeline": bundle.metadata,
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary, indent=2))
    object_dict: Dict[str, Any] = {"cfg": cfg, "bundle": bundle}
    return summary, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="interpretability/report.yaml",
)
def main(cfg: DictConfig) -> None:
    run_report(cfg)


if __name__ == "__main__":
    main()
