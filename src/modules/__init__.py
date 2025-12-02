"""
Higher-level modules for HOPE architecture.
"""

from src.modules.continuum_memory import ContinuumMemorySystem, FrequencyFFN, AdaptiveCMS
from src.modules.titans import (
    SelfModifyingTitans,
    SelfModifyingTitansChunk,
    SelfReferentialTitans,
    FullyNestedTitans,
    MemoryAsGate,
    MemoryAsLayer,
)
from src.modules.hope_block import HopeBlock, HopeBlockWithAttention, HopeBlockStack

__all__ = [
    # Continuum Memory System
    "ContinuumMemorySystem",
    "FrequencyFFN",
    "AdaptiveCMS",
    # Titans variants
    "SelfModifyingTitans",
    "SelfModifyingTitansChunk",
    "SelfReferentialTitans",
    "FullyNestedTitans",
    "MemoryAsGate",  # MAG variant
    "MemoryAsLayer",  # MAL variant
    # HOPE blocks
    "HopeBlock",
    "HopeBlockWithAttention",
    "HopeBlockStack",
]
