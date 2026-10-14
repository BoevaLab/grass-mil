from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def export_plotly_snapshots(
    figures: Iterable[tuple[str, Any]],
    output_dir: Path,
    *,
    width: int = 1600,
    height: int = 900,
    scale: float = 2.0,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, fig in figures:
        path = output_dir / f"{name}.png"
        try:
            fig.write_image(path, width=width, height=height, scale=scale)
        except Exception as exc:  # pragma: no cover - requires kaleido runtime
            raise RuntimeError(
                "Failed to export plot snapshot. Ensure `kaleido` is installed and working."
            ) from exc
        out[name] = path
    return out
