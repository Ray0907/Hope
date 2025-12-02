"""
Core memory layers for HOPE architecture.
"""

from src.layers.associative_memory import (
    AssociativeMemory,
    LinearAttentionMemory,
    DeltaRuleMemory,
    GatedDeltaRuleMemory,
    PersistentMemory,
    DynamicDecayMemory,
    SurpriseGatedMemory,
    MemoryWithCapacityManagement,
)
from src.layers.neural_memory import NeuralMemory, DeepMemory

__all__ = [
    # Base memory classes
    "AssociativeMemory",
    "LinearAttentionMemory",
    "DeltaRuleMemory",
    "GatedDeltaRuleMemory",
    # Advanced memory mechanisms
    "PersistentMemory",
    "DynamicDecayMemory",
    "SurpriseGatedMemory",
    "MemoryWithCapacityManagement",
    # Neural memory
    "NeuralMemory",
    "DeepMemory",
]
