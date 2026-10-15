"""GRASS-MIL: graph-based representation and discovery of phenotype-associated
spatial structures with multiple instance learning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("grass-mil")
except PackageNotFoundError:  # source checkout without an installed distribution
    __version__ = "0.0.0"

__all__ = ["__version__"]
