"""Invariants shared across the model and inference layers.

Kept dependency-free and outside both subpackages so either side can import it
without creating a cycle (``grass_mil.models`` and ``grass_mil.inference``
already import each other through the supervised module).
"""

from __future__ import annotations

__all__ = ["validate_attention_width"]


def validate_attention_width(*, attention_width: int, num_classes: int) -> None:
    """Reject attention widths that cannot be broadcast against class logits.

    Bag pooling computes ``L_c = sum_i A[i,c] * l[i,c]``. Only two widths are
    meaningful: one shared attention channel broadcast across all classes, or
    one channel per class (the published form, and the one that makes the
    attribution identity exact). Any other width would silently broadcast into
    the wrong shape.
    """
    if attention_width not in (1, num_classes):
        raise ValueError(
            f"Attention emits {attention_width} channel(s) but the head emits "
            f"{num_classes} class logit(s). Set model.attention.n_classes to 1 "
            f"(one shared channel) or to {num_classes} (one per class)."
        )
