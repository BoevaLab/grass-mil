"""Cross-version compatibility shims applied at package import.

Currently only one: allowlisting the OmegaConf types that appear inside
Lightning checkpoints.

PyTorch 2.6 flipped the default of ``torch.load(weights_only=...)`` from False
to True. Our own loaders pass ``weights_only=False`` explicitly, because they
read first-party artifacts. Lightning's checkpoint loader
(``lightning.fabric.utilities.cloud_io._load``) is not ours to change, and it
fails on any checkpoint whose ``hyper_parameters`` hold an OmegaConf container
-- which is every checkpoint this package writes, since the modules save their
Hydra config.

Allowlisting is preferable to disabling the check: the entries below are inert
container and metadata types whose construction cannot execute arbitrary code,
so the protection still applies to everything else in the file.
"""

from __future__ import annotations

import collections
import typing


def allow_omegaconf_in_checkpoints() -> None:
    """Register OmegaConf container types as safe for ``torch.load``.

    A no-op on torch versions without ``add_safe_globals`` (< 2.4) and on any
    OmegaConf layout that no longer exposes these names, so an upstream rename
    degrades to the previous behaviour rather than breaking package import.
    """
    try:
        import torch.serialization as ts
    except Exception:  # torch missing or partially initialised
        return

    add = getattr(ts, "add_safe_globals", None)
    if add is None:
        return

    safe: list = [dict, list, int, typing.Any, collections.defaultdict]
    try:
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode

        safe += [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode]
    except Exception:
        return

    try:
        add(safe)
    except Exception:
        # Never let a compatibility shim break importing the package.
        return
