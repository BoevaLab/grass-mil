"""Models: the ego-graph encoder, attention-based MIL module and BGRL pretraining."""

from .encoder import EncoderConfig, GINEEncoder
from .mil import GatedAttention, InstanceHead, attention_pool, pool_by_bag
from .pretraining import BGRLModule
from .supervised import SupervisedModule

__all__ = [
    "BGRLModule",
    "EncoderConfig",
    "GINEEncoder",
    "GatedAttention",
    "InstanceHead",
    "SupervisedModule",
    "attention_pool",
    "pool_by_bag",
]
