"""Project-root resolution shared by the CLI entrypoints.

In a source checkout, ``rootutils`` walks up from this file to the
``.project-root`` marker and exports ``PROJECT_ROOT``, which
``configs/paths/default.yaml`` interpolates. An installed wheel has no such
marker above it, so resolution is best-effort: when it fails, the paths config
falls back to the Hydra runtime working directory.
"""

from __future__ import annotations

from typing import Optional

import rootutils

__all__ = ["setup_project_root"]


def setup_project_root() -> Optional[str]:
    """Export ``PROJECT_ROOT`` when a ``.project-root`` marker is present.

    Returns:
        The resolved project root, or ``None`` when running from an installed
        package where no marker exists.
    """
    try:
        return str(rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True))
    except (FileNotFoundError, RuntimeError):
        return None
