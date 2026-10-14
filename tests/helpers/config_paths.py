"""Locate the packaged Hydra config tree.

Configs ship inside the package (``grass_mil/configs``) so that they resolve
identically from a source checkout and from an installed wheel. Tests must not
reach for them via a CWD-relative path.
"""

from pathlib import Path

from grass_mil import configs as _configs

CONFIGS_DIR = Path(_configs.__file__).parent
