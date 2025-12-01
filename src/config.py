"""
Configuration for HOPE architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
