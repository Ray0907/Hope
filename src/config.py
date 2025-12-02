"""
Configuration for HOPE architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


# Type aliases for configuration options
TitansVariant = Literal["mac", "mag", "mal"]
MemoryType = Literal["delta", "dynamic_decay", "surprise_gated", "capacity_managed"]


@dataclass
class HopeConfig:
    """
    Configuration class for HOPE model.

    Attributes:
        vocab_size: Size of the vocabulary
        dim: Model dimension (hidden size)
        num_layers: Number of HOPE blocks
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: dim // num_heads)
        num_memory_levels: Number of levels in Continuum Memory System
        chunk_sizes: Chunk sizes for each CMS level (determines update frequency)
        memory_depth: Number of layers in neural memory MLP
        ffn_expansion: Expansion factor for FFN layers
        dropout: Dropout probability
        learning_rate_memory: Learning rate for internal memory updates
        momentum_decay: Decay factor for momentum in memory updates
        use_delta_rule: Whether to use delta rule for memory updates
        max_seq_len: Maximum sequence length
        eps: Small epsilon for numerical stability
    """

    # Model architecture
    vocab_size: int = 32000
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: Optional[int] = None

    # Continuum Memory System
    num_memory_levels: int = 4
    chunk_sizes: List[int] = field(
        default_factory=lambda: [16, 256, 4096, 65536]
    )

    # Neural Memory
    memory_depth: int = 2
    memory_dim: Optional[int] = None

    # FFN
    ffn_expansion: int = 4

    # Regularization
    dropout: float = 0.1

    # Memory learning parameters
    learning_rate_memory: float = 0.1
    momentum_decay: float = 0.9
    use_delta_rule: bool = True

    # Titans variant configuration
    titans_variant: str = "mac"  # "mac", "mag", or "mal"

    # Persistent Memory configuration
    use_persistent_memory: bool = False
    num_persistent_slots: int = 64
    persistent_combine_mode: str = "gate"  # "gate", "add", "concat"

    # Dynamic Decay configuration
    use_dynamic_decay: bool = False
    base_decay: float = 0.01
    decay_utilization_weight: float = 0.1
    decay_surprise_weight: float = 0.1
    max_decay: float = 0.5

    # Surprise-based Gating configuration
    use_surprise_gating: bool = False
    surprise_threshold: float = 0.1
    surprise_temperature: float = 0.1
    adaptive_threshold: bool = True

    # Memory Capacity Management
    use_capacity_management: bool = False
    memory_max_norm: float = 10.0
    memory_prune_threshold: float = 0.01

    # Sequence
    max_seq_len: int = 8192

    # Numerical stability
    eps: float = 1e-6

    def __post_init__(self):
        """Validate and set derived parameters."""
        if self.head_dim is None:
            self.head_dim = self.dim // self.num_heads

        if self.memory_dim is None:
            self.memory_dim = self.dim

        if len(self.chunk_sizes) != self.num_memory_levels:
            raise ValueError(
                f"chunk_sizes length ({len(self.chunk_sizes)}) must match "
                f"num_memory_levels ({self.num_memory_levels})"
            )

        # Ensure chunk sizes are in ascending order (lower -> higher frequency)
        if self.chunk_sizes != sorted(self.chunk_sizes):
            raise ValueError("chunk_sizes must be in ascending order")

        # Validate Titans variant
        valid_variants = ["mac", "mag", "mal"]
        if self.titans_variant not in valid_variants:
            raise ValueError(
                f"titans_variant must be one of {valid_variants}, "
                f"got '{self.titans_variant}'"
            )

        # Validate persistent memory combine mode
        valid_combine_modes = ["gate", "add", "concat"]
        if self.persistent_combine_mode not in valid_combine_modes:
            raise ValueError(
                f"persistent_combine_mode must be one of {valid_combine_modes}, "
                f"got '{self.persistent_combine_mode}'"
            )


@dataclass
class HopeSmallConfig(HopeConfig):
    """Small HOPE model (~125M parameters)."""

    dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_memory_levels: int = 3
    chunk_sizes: List[int] = field(default_factory=lambda: [16, 256, 4096])


@dataclass
class HopeBaseConfig(HopeConfig):
    """Base HOPE model (~350M parameters)."""

    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_memory_levels: int = 4
    chunk_sizes: List[int] = field(
        default_factory=lambda: [16, 256, 4096, 65536]
    )


@dataclass
class HopeLargeConfig(HopeConfig):
    """Large HOPE model (~760M parameters)."""

    dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    num_memory_levels: int = 4
    chunk_sizes: List[int] = field(
        default_factory=lambda: [16, 256, 4096, 65536]
    )


@dataclass
class HopeXLConfig(HopeConfig):
    """XL HOPE model (~1.3B parameters)."""

    dim: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    num_memory_levels: int = 5
    chunk_sizes: List[int] = field(
        default_factory=lambda: [16, 256, 4096, 65536, 1048576]
    )
