"""
HOPE Block module.

Combines Self-Modifying Titans with Continuum Memory System to create
the complete HOPE block. This is the core building block of the HOPE
architecture.

Reference: Nested Learning paper, Section 3, Figure 3
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

from hope.modules.titans import SelfModifyingTitans, SelfModifyingTitansChunk
from hope.modules.continuum_memory import ContinuumMemorySystem, AdaptiveCMS


class HopeBlock(nn.Module):
    """
    HOPE Block: Self-Modifying Titans + Continuum Memory System.

    Architecture (from Figure 3):
        Input -> Self-Modifying Titans -> Add & Norm -> CMS (Multi-Freq FFN) -> Output

    The block processes sequences through:
    1. Self-modifying memory attention (Titans)
    2. Residual connection and normalization
    3. Multi-frequency FFN system (CMS)
    4. Final residual connection

    This creates a hierarchy of memory timescales:
    - Titans: Token-level, fast-updating memory
    - CMS: Multi-scale persistent knowledge storage
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        num_memory_levels: int = 4,
        chunk_sizes: Optional[List[int]] = None,
        ffn_expansion: int = 4,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
        use_chunked_titans: bool = False,
        titans_chunk_size: int = 16,
        use_adaptive_cms: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_chunked_titans = use_chunked_titans

        # Default chunk sizes for CMS
        if chunk_sizes is None:
            chunk_sizes = [16, 256, 4096, 65536][:num_memory_levels]

        # Self-Modifying Titans (memory attention)
        if use_chunked_titans:
            self.titans = SelfModifyingTitansChunk(
                dim=dim,
                head_dim=head_dim,
                num_heads=num_heads,
                chunk_size=titans_chunk_size,
                learning_rate=learning_rate,
                use_delta_rule=use_delta_rule,
                dropout=dropout,
                eps=eps,
            )
        else:
            self.titans = SelfModifyingTitans(
                dim=dim,
                head_dim=head_dim,
                num_heads=num_heads,
                learning_rate=learning_rate,
                momentum=momentum,
                use_delta_rule=use_delta_rule,
                dropout=dropout,
                eps=eps,
            )

        # Layer norm after Titans
        self.norm_titans = nn.LayerNorm(dim)

        # Continuum Memory System (multi-frequency FFN)
        if use_adaptive_cms:
            self.cms = AdaptiveCMS(
                dim=dim,
                num_levels=num_memory_levels,
                chunk_sizes=chunk_sizes,
                expansion=ffn_expansion,
                dropout=dropout,
            )
        else:
            self.cms = ContinuumMemorySystem(
                dim=dim,
                num_levels=num_memory_levels,
                chunk_sizes=chunk_sizes,
                expansion=ffn_expansion,
                dropout=dropout,
            )

        # Final layer norm
        self.norm_final = nn.LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        return_memory: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through HOPE block.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_state: Previous Titans memory state
            step: Current global step (for CMS update scheduling)
            return_memory: Whether to return updated memory state

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_state: Updated memory state (if return_memory=True)
        """
        # Self-Modifying Titans
        titans_output, memory_state = self.titans(
            x, memory_state=memory_state, return_memory=return_memory
        )

        # Residual + norm
        x = x + titans_output
        x = self.norm_titans(x)

        # Continuum Memory System (multi-frequency FFN)
        cms_output = self.cms(x)

        # Final residual + norm
        x = x + cms_output
        x = self.norm_final(x)

        if return_memory:
            return x, memory_state
        return x, None


class HopeBlockWithAttention(nn.Module):
    """
    HOPE Block variant with additional sliding window attention.

    Adds a sliding window attention mechanism for local context,
    complementing the global memory from Titans.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        num_memory_levels: int = 4,
        chunk_sizes: Optional[List[int]] = None,
        window_size: int = 512,
        ffn_expansion: int = 4,
        learning_rate: float = 0.1,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        # Default chunk sizes
        if chunk_sizes is None:
            chunk_sizes = [16, 256, 4096, 65536][:num_memory_levels]

        # Sliding window attention for local context
        self.local_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_local = nn.LayerNorm(dim)

        # Self-Modifying Titans for global memory
        self.titans = SelfModifyingTitans(
            dim=dim,
            head_dim=head_dim,
            num_heads=num_heads,
            learning_rate=learning_rate,
            use_delta_rule=use_delta_rule,
            dropout=dropout,
            eps=eps,
        )
        self.norm_titans = nn.LayerNorm(dim)

        # CMS
        self.cms = ContinuumMemorySystem(
            dim=dim,
            num_levels=num_memory_levels,
            chunk_sizes=chunk_sizes,
            expansion=ffn_expansion,
            dropout=dropout,
        )
        self.norm_cms = nn.LayerNorm(dim)

        # Gate to combine local and global
        self.combine_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def createSlidingWindowMask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal sliding window attention mask."""
        mask = torch.ones(seq_len, seq_len, device=device).bool()

        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = False

        return mask

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with local attention + global memory."""
        batch_size, seq_len, _ = x.shape

        # Local sliding window attention
        attn_mask = self.createSlidingWindowMask(seq_len, x.device)
        local_out, _ = self.local_attention(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x_local = x + local_out
        x_local = self.norm_local(x_local)

        # Global memory (Titans)
        titans_out, memory_state = self.titans(x, memory_state=memory_state)
        x_global = x + titans_out
        x_global = self.norm_titans(x_global)

        # Combine local and global
        combined = torch.cat([x_local, x_global], dim=-1)
        gate = self.combine_gate(combined)
        x = gate * x_local + (1 - gate) * x_global

        # CMS
        cms_out = self.cms(x, step=step)
        x = x + cms_out
        x = self.norm_cms(x)

        return x, memory_state


class HopeBlockStack(nn.Module):
    """
    Stack of HOPE blocks with shared or separate memory states.

    Provides utilities for processing through multiple HOPE blocks
    with proper memory state management.
    """

    def __init__(
        self,
        num_blocks: int,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        num_memory_levels: int = 4,
        chunk_sizes: Optional[List[int]] = None,
        ffn_expansion: int = 4,
        learning_rate: float = 0.1,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
        share_memory: bool = False,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.share_memory = share_memory

        # Default chunk sizes
        if chunk_sizes is None:
            chunk_sizes = [16, 256, 4096, 65536][:num_memory_levels]

        # Create blocks
        self.blocks = nn.ModuleList([
            HopeBlock(
                dim=dim,
                head_dim=head_dim,
                num_heads=num_heads,
                num_memory_levels=num_memory_levels,
                chunk_sizes=chunk_sizes,
                ffn_expansion=ffn_expansion,
                learning_rate=learning_rate,
                use_delta_rule=use_delta_rule,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[List[torch.Tensor]] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_states: List of memory states for each block
            step: Current global step

        Returns:
            output: Output tensor
            memory_states: Updated memory states
        """
        if memory_states is None:
            memory_states = [None] * self.num_blocks

        new_memory_states = []

        for i, block in enumerate(self.blocks):
            if self.share_memory and i > 0:
                # Share memory across blocks
                memory_state = new_memory_states[0]
            else:
                memory_state = memory_states[i]

            x, memory_state = block(x, memory_state=memory_state, step=step)
            new_memory_states.append(memory_state)

        return x, new_memory_states

    def getMemorySize(self) -> int:
        """Get total memory state size in parameters."""
        block = self.blocks[0]
        # Memory shape: (batch, num_heads, head_dim, head_dim)
        single_memory_size = (
            block.titans.num_heads *
            block.titans.head_dim *
            block.titans.head_dim
        )

        if self.share_memory:
            return single_memory_size
        return single_memory_size * self.num_blocks
