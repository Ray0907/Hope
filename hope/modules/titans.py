"""
Self-Modifying Titans module for HOPE architecture.

Implements the core sequence model from Titans with self-referential
learning capabilities. The module learns to modify its own memory
through surprise-based updates using the delta rule.

Reference: Titans paper, Nested Learning Section 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

from hope.layers.associative_memory import DeltaRuleMemory, LinearAttentionMemory


class SelfModifyingTitans(nn.Module):
    """
    Self-Modifying Titans module.

    Key features:
    1. Data-dependent K, V, Q projections that adapt to context
    2. Neural memory with surprise-based updates
    3. Delta rule for efficient memory management
    4. Self-referential optimization (memory optimizes its own update)

    The module processes sequences by:
    1. Computing context-dependent projections
    2. Using memory to retrieve relevant information
    3. Computing surprise (prediction error)
    4. Updating memory based on surprise using delta rule
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        memory_dim: Optional[int] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.memory_dim = memory_dim or head_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_delta_rule = use_delta_rule
        self.eps = eps

        # Static projections (updated during pre-training)
        self.w_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_q = nn.Linear(dim, num_heads * head_dim, bias=False)

        # Dynamic projection modifiers (context-dependent)
        # These generate modifications to the static projections
        self.k_modifier = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_heads * head_dim),
        )
        self.v_modifier = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_heads * head_dim),
        )
        self.q_modifier = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_heads * head_dim),
        )

        # Output projection
        self.w_o = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Normalization layers
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_input = nn.LayerNorm(dim)

        # Memory module (per head)
        if use_delta_rule:
            self.memory = DeltaRuleMemory(
                dim_key=head_dim,
                dim_value=head_dim,
                learning_rate=learning_rate,
                eps=eps,
            )
        else:
            self.memory = LinearAttentionMemory(
                dim_key=head_dim,
                dim_value=head_dim,
                learning_rate=learning_rate,
                eps=eps,
            )

        # Learnable parameters for memory update
        self.lr_scale = nn.Parameter(torch.ones(num_heads))
        self.forget_scale = nn.Parameter(torch.ones(num_heads))

        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.applyInit()

    def applyInit(self):
        """Initialize weights for stable training."""
        # Standard initialization for projections
        for module in [self.w_k, self.w_v, self.w_q, self.w_o]:
            nn.init.xavier_uniform_(module.weight)

        # Small initialization for modifiers (start close to identity)
        for modifier in [self.k_modifier, self.v_modifier, self.q_modifier]:
            for layer in modifier:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def computeDataDependentProjections(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute data-dependent K, V, Q projections.

        The projections are modulated by context-dependent modifiers:
            k = W_k * x + modifier_k(context)
            v = W_v * x + modifier_v(context)
            q = W_q * x + modifier_q(context)

        Args:
            x: Input tensor (batch, seq_len, dim)
            context: Optional context for modifiers (defaults to x)

        Returns:
            k, v, q: Projected tensors (batch, seq_len, num_heads * head_dim)
        """
        if context is None:
            context = x

        # Static projections
        k_static = self.w_k(x)
        v_static = self.w_v(x)
        q_static = self.w_q(x)

        # Context-dependent modifications
        k_mod = self.k_modifier(context)
        v_mod = self.v_modifier(context)
        q_mod = self.q_modifier(context)

        # Combine: static + small modification
        k = k_static + 0.1 * k_mod
        v = v_static + 0.1 * v_mod
        q = q_static + 0.1 * q_mod

        return k, v, q

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_memory: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Self-Modifying Titans.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_state: Previous memory state (batch, num_heads, head_dim, head_dim)
            context: Optional context for data-dependent projections
            return_memory: Whether to return updated memory state

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_state: Updated memory state (if return_memory=True)
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x_normed = self.norm_input(x)

        # Initialize memory if needed
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # Compute data-dependent projections
        k, v, q = self.computeDataDependentProjections(x_normed, context)

        # Reshape to multi-head format: (batch, seq, heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Normalize
        k = self.norm_k(k)
        v = self.norm_v(v)
        q = self.norm_q(q)

        # Transpose: (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)

        # Process with memory
        outputs = []
        lr = torch.sigmoid(self.lr_scale).view(1, self.num_heads, 1, 1) * self.learning_rate * 2

        for t in range(seq_len):
            k_t = k[:, :, t:t+1, :]  # (batch, heads, 1, head_dim)
            v_t = v[:, :, t:t+1, :]
            q_t = q[:, :, t:t+1, :]

            # Retrieve from memory
            # memory_state: (batch, heads, head_dim, head_dim)
            # q_t: (batch, heads, 1, head_dim)
            output_t = torch.einsum("bhdk,bhsk->bhsd", memory_state, q_t)

            if self.use_delta_rule:
                # Delta rule update (Eq. 28-29 from Nested Learning paper):
                # M_{t+1} = M_t - M_t * k_t * k_t^T - eta * (M_t * k_t - v_t) * k_t^T

                # Normalize key for stable updates (required for delta rule)
                k_t_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

                # Compute prediction: M @ k
                predicted = torch.einsum("bhdk,bhsk->bhsd", memory_state, k_t_norm)

                # Compute surprise (prediction error): M*k - v
                surprise = predicted - v_t  # (batch, heads, 1, head_dim)

                # Forgetting term: (M @ k) @ k^T (coefficient = 1 per paper)
                forget_term = torch.einsum("bhsd,bhsk->bhdk", predicted, k_t_norm)

                # Learning term: eta * (M*k - v) @ k^T = eta * surprise @ k^T
                learn_term = torch.einsum("bhsd,bhsk->bhdk", surprise, k_t_norm)

                # Delta rule: M = M - forget_term - lr * learn_term
                memory_state = memory_state - forget_term - lr * learn_term
            else:
                # Standard outer product update (Hebbian)
                update = torch.einsum("bhsv,bhsk->bhvk", v_t, k_t)
                memory_state = memory_state + lr * update

            outputs.append(output_t)

        # Stack outputs: (batch, heads, seq, head_dim)
        output = torch.cat(outputs, dim=2)

        # Reshape: (batch, seq, heads, head_dim) -> (batch, seq, dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Apply output gate
        gate = self.output_gate(x)
        output = gate * output

        output = self.dropout(output)

        if return_memory:
            return output, memory_state
        return output, None


class SelfModifyingTitansChunk(nn.Module):
    """
    Chunked version of Self-Modifying Titans for efficient training.

    Processes the sequence in chunks, updating memory at chunk boundaries.
    This allows for efficient parallel processing within chunks.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        chunk_size: int = 16,
        learning_rate: float = 0.1,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.use_delta_rule = use_delta_rule
        self.eps = eps

        # Projections
        self.w_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Normalization
        self.norm_input = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_q = nn.LayerNorm(head_dim)

        # Learnable memory parameters
        self.lr_scale = nn.Parameter(torch.ones(num_heads))
        self.forget_scale = nn.Parameter(torch.ones(num_heads))

        # Output gate
        self.output_gate = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forwardChunk(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single chunk.

        Uses parallel attention within chunk, then updates memory.
        """
        batch_size, chunk_len, _ = x.shape

        # Project
        k = self.w_k(x).view(batch_size, chunk_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, chunk_len, self.num_heads, self.head_dim)
        q = self.w_q(x).view(batch_size, chunk_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)  # (batch, heads, chunk, head_dim)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)

        k = self.norm_k(k)
        q = self.norm_q(q)

        # Apply feature map for linear attention
        k = F.elu(k) + 1
        q = F.elu(q) + 1

        lr = torch.sigmoid(self.lr_scale).view(1, self.num_heads, 1, 1)
        forget = torch.sigmoid(self.forget_scale).view(1, self.num_heads, 1, 1)

        # Intra-chunk attention (causal)
        # Build cumulative sum for linear attention
        kv = torch.einsum("bhsk,bhsv->bhskv", k, v)  # (batch, heads, chunk, dim_k, dim_v)
        kv_cumsum = torch.cumsum(kv, dim=2)

        # Query against cumulative KV
        output_intra = torch.einsum("bhskv,bhsk->bhsv", kv_cumsum, q)

        # Inter-chunk: query against persistent memory
        output_inter = torch.einsum("bhdk,bhsk->bhsd", memory, q)

        # Combine
        output = output_intra + output_inter

        # Update memory with chunk summary using delta rule (Eq. 28-29)
        if self.use_delta_rule:
            # Summarize chunk for memory update
            k_sum = k.sum(dim=2, keepdim=True) / chunk_len
            v_sum = v.sum(dim=2, keepdim=True) / chunk_len

            # Normalize key for stable updates
            k_sum_norm = k_sum / (k_sum.norm(dim=-1, keepdim=True) + 1e-6)

            # Compute prediction: M @ k_avg
            predicted = torch.einsum("bhdk,bhsk->bhsd", memory, k_sum_norm)

            # Compute surprise (prediction error): M*k - v
            surprise = predicted - v_sum

            # Forgetting term: (M @ k) @ k^T
            forget_term = torch.einsum("bhsd,bhsk->bhdk", predicted, k_sum_norm)

            # Learning term: eta * surprise @ k^T
            learn_term = torch.einsum("bhsd,bhsk->bhdk", surprise, k_sum_norm)

            # Delta rule: M = M - forget_term - lr * learn_term
            memory = memory - forget_term - lr * learn_term
        else:
            k_sum = k.sum(dim=2, keepdim=True) / chunk_len
            v_sum = v.sum(dim=2, keepdim=True) / chunk_len
            update = torch.einsum("bhsv,bhsk->bhvk", v_sum, k_sum)
            memory = memory + lr * update

        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, chunk_len, -1)

        return output, memory

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with chunked processing.
        """
        batch_size, seq_len, _ = x.shape

        # Normalize
        x = self.norm_input(x)

        # Initialize memory
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # Process chunks
        outputs = []
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end, :]

            output_chunk, memory_state = self.forwardChunk(chunk, memory_state)
            outputs.append(output_chunk)

        # Concatenate
        output = torch.cat(outputs, dim=1)

        # Output projection
        output = self.w_o(output)

        # Gate
        gate = torch.sigmoid(self.output_gate(x[:, :output.shape[1], :]))
        output = gate * output

        output = self.dropout(output)

        return output, memory_state


class SelfReferentialTitans(nn.Module):
    """
    Self-Referential Titans with meta-learning update rule.

    This module implements the full self-referential mechanism where
    the memory can learn to modify its own update rule. The key insight
    is that the update rule itself is parameterized and learned.

    Architecture:
    1. Memory M stores key-value associations
    2. Meta-memory M' stores the update rule parameters
    3. M' is updated based on how well M performs
    4. M is updated using the rule specified by M'

    This creates a two-level nested optimization:
    - Outer level: Learn how to learn (update M')
    - Inner level: Learn associations (update M using rule from M')

    Reference: Nested Learning paper, Section 3 "HOPE"
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        learning_rate: float = 0.1,
        meta_learning_rate: float = 0.01,
        momentum: float = 0.9,
        use_delta_rule: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.base_learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.momentum = momentum
        self.use_delta_rule = use_delta_rule
        self.eps = eps

        # Projections
        self.w_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Normalization
        self.norm_input = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.norm_q = nn.LayerNorm(head_dim)

        # Meta-memory: learns the update rule
        # Maps (key, value, surprise) -> (learning_rate, forget_rate)
        self.meta_memory = nn.Sequential(
            nn.Linear(head_dim * 3, head_dim * 2),
            nn.GELU(),
            nn.Linear(head_dim * 2, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 2),  # Output: (lr_modifier, forget_modifier)
            nn.Sigmoid(),
        )

        # Learnable base parameters
        self.lr_scale = nn.Parameter(torch.zeros(num_heads))
        self.forget_scale = nn.Parameter(torch.zeros(num_heads))
        self.meta_lr_scale = nn.Parameter(torch.zeros(1))

        # Memory momentum buffers
        self.register_buffer("memory_momentum", None)
        self.register_buffer("meta_momentum", None)

        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def computeMetaUpdate(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        surprise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive learning rate and forget rate using meta-memory.

        Args:
            key: Key tensor (batch, heads, head_dim)
            value: Value tensor (batch, heads, head_dim)
            surprise: Surprise signal (batch, heads, head_dim)

        Returns:
            lr_modifier: Learning rate modifier (batch, heads, 1)
            forget_modifier: Forget rate modifier (batch, heads, 1)
        """
        # Concatenate inputs for meta-memory
        meta_input = torch.cat([key, value, surprise], dim=-1)  # (batch, heads, head_dim*3)

        # Get modifiers from meta-memory
        modifiers = self.meta_memory(meta_input)  # (batch, heads, 2)

        lr_modifier = modifiers[..., 0:1]  # (batch, heads, 1)
        forget_modifier = modifiers[..., 1:2]

        return lr_modifier, forget_modifier

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        memory_momentum: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with self-referential memory update.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_state: Previous memory state (batch, heads, head_dim, head_dim)
            memory_momentum: Previous momentum (batch, heads, head_dim, head_dim)

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_state: Updated memory state
            memory_momentum: Updated momentum
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x = self.norm_input(x)

        # Initialize memory
        if memory_state is None:
            memory_state = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype,
            )
        if memory_momentum is None:
            memory_momentum = torch.zeros_like(memory_state)

        # Project
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)

        # Normalize
        k = self.norm_k(k)
        v = self.norm_v(v)
        q = self.norm_q(q)

        # Base learning rates
        base_lr = torch.sigmoid(self.lr_scale).view(1, self.num_heads, 1, 1) * self.base_learning_rate * 2
        base_forget = torch.sigmoid(self.forget_scale).view(1, self.num_heads, 1, 1) * 0.5

        outputs = []

        for t in range(seq_len):
            k_t = k[:, :, t, :]  # (batch, heads, head_dim)
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # Retrieve from memory
            # memory_state: (batch, heads, head_dim, head_dim)
            output_t = torch.einsum("bhdk,bhk->bhd", memory_state, q_t)
            outputs.append(output_t)

            # Compute surprise (prediction error for key)
            predicted = torch.einsum("bhdk,bhk->bhd", memory_state, k_t)
            surprise = predicted - v_t  # (batch, heads, head_dim)

            # Get adaptive update parameters from meta-memory
            lr_mod, forget_mod = self.computeMetaUpdate(k_t, v_t, surprise)

            # Compute effective learning rate and forget rate
            effective_lr = base_lr * (0.5 + lr_mod.unsqueeze(-1))  # Scale around base
            effective_forget = base_forget * (0.5 + forget_mod.unsqueeze(-1))

            if self.use_delta_rule:
                # Delta rule with adaptive parameters (Eq. 28-29 with meta-learning)
                # M = M - (M @ k) @ k^T - eta * (M*k - v) @ k^T

                # Normalize key for stable updates
                k_t_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

                # Compute prediction: M @ k
                predicted = torch.einsum("bhdk,bhk->bhd", memory_state, k_t_norm)

                # Forgetting term: (M @ k) @ k^T
                forget_term = torch.einsum("bhd,bhk->bhdk", predicted, k_t_norm)

                # Learning term: surprise @ k^T
                learn_term = torch.einsum("bhd,bhk->bhdk", surprise, k_t_norm)

                # Update momentum with learning term
                memory_momentum = self.momentum * memory_momentum + learn_term

                # Delta rule: M = M - g_f * forget_term - g_l * lr * momentum
                memory_state = (
                    memory_state
                    - effective_forget * forget_term
                    - effective_lr * memory_momentum
                )
            else:
                # Standard update with adaptive learning rate
                learn_term = torch.einsum("bhd,bhk->bhdk", surprise, k_t)
                memory_momentum = self.momentum * memory_momentum + learn_term
                memory_state = memory_state - effective_lr * memory_momentum

            # Update meta-memory based on how well the prediction improved
            # (This happens through normal backprop during training)

        # Stack outputs: (batch, heads, seq, head_dim)
        output = torch.stack(outputs, dim=2)
        output = output.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Gate
        gate = self.output_gate(x)
        output = gate * output

        output = self.dropout(output)

        return output, memory_state, memory_momentum


class FullyNestedTitans(nn.Module):
    """
    Fully Nested Titans with multiple levels of self-reference.

    Implements the full nested learning hierarchy:
    - Level 0: Token-level memory (highest frequency)
    - Level 1: Meta-memory for update rules
    - Level 2: Meta-meta-memory for learning meta-rules
    - ...

    Each level learns to control the level below it.

    Reference: Nested Learning paper, Figure 2 "Neural Learning Module"
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 12,
        num_levels: int = 3,
        learning_rates: Optional[List[float]] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.eps = eps

        if learning_rates is None:
            # Decreasing learning rates for higher levels
            learning_rates = [0.1 / (2 ** i) for i in range(num_levels)]
        self.learning_rates = learning_rates

        # Projections
        self.w_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Normalization
        self.norm_input = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.norm_q = nn.LayerNorm(head_dim)

        # Multi-level meta-memories
        # Each level takes input from level below and outputs control for level below
        self.meta_memories = nn.ModuleList()
        for level in range(num_levels - 1):
            # Level i controls level i-1
            # Input: state from level i-1 (head_dim)
            # Output: control signal for level i-1 (learning_rate, forget_rate)
            self.meta_memories.append(nn.Sequential(
                nn.Linear(head_dim * 2, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 2),
                nn.Sigmoid(),
            ))

        # Learnable base parameters per level
        self.lr_scales = nn.ParameterList([
            nn.Parameter(torch.zeros(num_heads)) for _ in range(num_levels)
        ])
        self.forget_scales = nn.ParameterList([
            nn.Parameter(torch.zeros(num_heads)) for _ in range(num_levels)
        ])

        # Output gate
        self.output_gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through fully nested memory hierarchy.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_states: List of memory states, one per level

        Returns:
            output: Output tensor
            memory_states: Updated memory states for all levels
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory states for all levels
        if memory_states is None:
            memory_states = [
                torch.zeros(
                    batch_size, self.num_heads, self.head_dim, self.head_dim,
                    device=x.device, dtype=x.dtype,
                )
                for _ in range(self.num_levels)
            ]

        # Normalize input
        x = self.norm_input(x)

        # Project
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)  # (batch, heads, seq, head_dim)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)

        k = self.norm_k(k)
        v = self.norm_v(v)
        q = self.norm_q(q)

        outputs = []

        for t in range(seq_len):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # Retrieve from lowest level memory
            output_t = torch.einsum("bhdk,bhk->bhd", memory_states[0], q_t)
            outputs.append(output_t)

            # Compute surprise at lowest level
            predicted = torch.einsum("bhdk,bhk->bhd", memory_states[0], k_t)
            surprise = predicted - v_t

            # Update memories bottom-up, using control from level above
            for level in range(self.num_levels):
                base_lr = torch.sigmoid(self.lr_scales[level]).view(1, self.num_heads, 1, 1)
                base_lr = base_lr * self.learning_rates[level] * 2
                base_forget = torch.sigmoid(self.forget_scales[level]).view(1, self.num_heads, 1, 1) * 0.5

                if level < self.num_levels - 1:
                    # Get control signal from level above
                    # Use summary of current level state as input to meta-memory
                    state_summary = memory_states[level].mean(dim=(-2, -1))  # (batch, heads)
                    surprise_summary = surprise.mean(dim=-1, keepdim=True)  # (batch, heads, 1)
                    meta_input = torch.cat([state_summary.unsqueeze(-1), surprise_summary], dim=-1)
                    meta_input = meta_input.squeeze(-1)  # (batch, heads, 2) -> need (batch, heads, head_dim*2)

                    # Pad to expected size
                    meta_input_padded = F.pad(meta_input, (0, self.head_dim * 2 - 2))
                    control = self.meta_memories[level](meta_input_padded)

                    lr_mod = control[..., 0:1].unsqueeze(-1)
                    forget_mod = control[..., 1:2].unsqueeze(-1)

                    effective_lr = base_lr * (0.5 + lr_mod)
                    effective_forget = base_forget * (0.5 + forget_mod)
                else:
                    effective_lr = base_lr
                    effective_forget = base_forget

                # Update this level's memory using delta rule (Eq. 28-29)
                # M = M - (M @ k) @ k^T - eta * (M*k - v) @ k^T

                # Compute prediction: M @ k
                predicted = torch.einsum("bhdk,bhk->bhd", memory_states[level], k_t)

                # Forgetting term: (M @ k) @ k^T
                forget_term = torch.einsum("bhd,bhk->bhdk", predicted, k_t)

                # Learning term: surprise @ k^T
                learn_term = torch.einsum("bhd,bhk->bhdk", surprise, k_t)

                # Delta rule: M = M - g_f * forget_term - g_l * learn_term
                memory_states[level] = (
                    memory_states[level]
                    - effective_forget * forget_term
                    - effective_lr * learn_term
                )

        # Stack outputs
        output = torch.stack(outputs, dim=2)  # (batch, heads, seq, head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Gate
        gate = torch.sigmoid(self.output_gate(x))
        output = gate * output

        output = self.dropout(output)

        return output, memory_states
