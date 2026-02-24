from __future__ import annotations

import json
import math
from pathlib import Path

from src.inference.io import write_json


def test_write_json_sanitizes_non_finite_floats(tmp_path: Path) -> None:
    payload = {
        "nan_metric": float("nan"),
        "pos_inf_metric": float("inf"),
        "neg_inf_metric": float("-inf"),
        "nested": {"values": [1.0, float("nan"), 3.0]},
    }
    out_path = tmp_path / "metrics.json"

    write_json(payload, out_path)

    contents = out_path.read_text()
    assert "NaN" not in contents
    assert "Infinity" not in contents
    assert "-Infinity" not in contents

    parsed = json.loads(contents)
    assert parsed["nan_metric"] is None
    assert parsed["pos_inf_metric"] is None
    assert parsed["neg_inf_metric"] is None
    assert parsed["nested"]["values"] == [1.0, None, 3.0]


def test_write_json_preserves_finite_values(tmp_path: Path) -> None:
    payload = {
        "accuracy": 0.95,
        "count": 3,
        "items": [0.0, -2.5],
        "nested": {"ok": True},
    }
    out_path = tmp_path / "summary.json"

    write_json(payload, out_path)

    parsed = json.loads(out_path.read_text())
    assert parsed["accuracy"] == 0.95
    assert parsed["count"] == 3
    assert parsed["items"] == [0.0, -2.5]
    assert parsed["nested"]["ok"] is True
    assert math.isfinite(parsed["accuracy"])
