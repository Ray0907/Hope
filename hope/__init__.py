"""
HOPE: Hierarchical Optimization with Persistent Experience

A self-referential learning module based on Nested Learning paradigm.
Reference: "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025)
"""

from hope.config import HopeConfig
from hope.model import Hope

__version__ = "0.1.0"
__all__ = ["Hope", "HopeConfig"]
