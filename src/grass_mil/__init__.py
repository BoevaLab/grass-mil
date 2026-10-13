"""GRASS-MIL: graph-based representation and discovery of phenotype-associated
spatial structures with multiple instance learning."""

from importlib.metadata import PackageNotFoundError, version

from grass_mil._compat import allow_omegaconf_in_checkpoints

# Applied at import so every entrypoint and any user code that loads a
# checkpoint gets it. See _compat for why this is needed under torch>=2.6.
allow_omegaconf_in_checkpoints()

try:
    __version__ = version("grass-mil")
except PackageNotFoundError:  # source checkout without an installed distribution
    __version__ = "0.0.0"

__all__ = ["__version__"]
