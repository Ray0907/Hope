"""
Higher-level modules for HOPE architecture.
"""

from hope.modules.continuum_memory import ContinuumMemorySystem, FrequencyFFN
from hope.modules.titans import SelfModifyingTitans
from hope.modules.hope_block import HopeBlock

__all__ = [
    "ContinuumMemorySystem",
    "FrequencyFFN",
    "SelfModifyingTitans",
    "HopeBlock",
]
