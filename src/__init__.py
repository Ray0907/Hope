"""
HOPE: Hierarchical Optimization with Persistent Experience

A self-referential learning module based on Nested Learning paradigm.
Reference: "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025)
"""

from src.config import HopeConfig
from src.model import Hope

__version__ = "0.1.0"
__all__ = ["Hope", "HopeConfig"]
