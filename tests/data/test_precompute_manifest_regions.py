from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("lightning")

from src.data.components.precompute import ManifestConfig, SpatialOmicsPreprocessor


def _validator() -> SimpleNamespace:
    return SimpleNamespace(manifest_config=ManifestConfig())


def test_validate_manifest_regions_accepts_globally_unique_region_ids() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "input_path": ["a.csv", "b.csv", "c.csv"],
            "input_type": ["csv", "csv", "csv"],
            "region_id": ["r0", "r1", "r2"],
        }
    )

    SpatialOmicsPreprocessor._validate_manifest_regions(_validator(), df)


def test_validate_manifest_regions_rejects_cross_sample_region_id_reuse() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "input_path": ["a.csv", "b.csv"],
            "input_type": ["csv", "csv"],
            "region_id": ["shared_region", "shared_region"],
        }
    )

    with pytest.raises(ValueError, match="region_id values must be globally unique"):
        SpatialOmicsPreprocessor._validate_manifest_regions(_validator(), df)
