"""
Core memory layers for HOPE architecture.

Includes:
- Associative memory modules (delta rule, linear attention)
- Neural memory modules (MLP-based)
- MIRAS framework components (Moneta, Yaad, Memora)
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

# MIRAS framework components
from src.layers.attentional_bias import (
    AttentionalBias,
    L2AttentionalBias,
    LpAttentionalBias,
    HuberAttentionalBias,
    KLAttentionalBias,
    DotProductAttentionalBias,
    RobustAttentionalBias,
    createAttentionalBias,
)
from src.layers.retention_gates import (
    RetentionGate,
    L2RetentionGate,
    LqRetentionGate,
    KLRetentionGate,
    ElasticNetRetentionGate,
    BregmanRetentionGate,
    DeltaRuleRetentionGate,
    AdaptiveRetentionGate,
    createRetentionGate,
)
from src.layers.miras_memory import (
    MirasMemory,
    Moneta,
    Yaad,
    Memora,
    createMirasModel,
)

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
    # MIRAS Attentional Bias
    "AttentionalBias",
    "L2AttentionalBias",
    "LpAttentionalBias",
    "HuberAttentionalBias",
    "KLAttentionalBias",
    "DotProductAttentionalBias",
    "RobustAttentionalBias",
    "createAttentionalBias",
    # MIRAS Retention Gates
    "RetentionGate",
    "L2RetentionGate",
    "LqRetentionGate",
    "KLRetentionGate",
    "ElasticNetRetentionGate",
    "BregmanRetentionGate",
    "DeltaRuleRetentionGate",
    "AdaptiveRetentionGate",
    "createRetentionGate",
    # MIRAS Models
    "MirasMemory",
    "Moneta",
    "Yaad",
    "Memora",
    "createMirasModel",
]
