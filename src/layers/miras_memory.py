"""
MIRAS Memory modules for HOPE architecture.

Implements the three novel models from the MIRAS paper:
- Moneta: Lp attentional bias + Lq retention (robust to key collisions)
- Yaad: Huber attentional bias + L2 retention (robust to outlier values)
- Memora: L2 attentional bias + KL retention (soft thresholding)

Also provides a unified MirasMemory class for arbitrary combinations.

Reference: MIRAS paper (2504.13173), "Memory Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import math

from src.layers.attentional_bias import (
    AttentionalBias,
    L2AttentionalBias,
    LpAttentionalBias,
    HuberAttentionalBias,
    KLAttentionalBias,
    createAttentionalBias,
)
from src.layers.retention_gates import (
    RetentionGate,
    L2RetentionGate,
    LqRetentionGate,
    KLRetentionGate,
    DeltaRuleRetentionGate,
    createRetentionGate,
)


class MirasMemory(nn.Module):
    """
    Unified MIRAS Memory module with configurable components.

    The MIRAS framework unifies sequence models through 4 design choices:
    1. Memory Architecture (vector, matrix, MLP) - here we use matrix
    2. Attentional Bias (L2, Lp, Huber, KL, dot-product)
    3. Retention Gate (L2, Lq, KL, Elastic Net, Bregman)
    4. Memory Learning Algorithm (GD, GD+momentum, Newton)

    Memory update rule:
        M_{t+1} = argmin_M [ L_attn(M; k_t, v_t) + lambda * R(M, M_t) ]

    Solved via gradient descent:
        M_{t+1} = M_t - eta * grad_attn - lambda * grad_retention - forget_term

    Reference: MIRAS paper Table 1, unified framework
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        attentional_bias: Union[str, AttentionalBias] = "l2",
        retention_gate: Union[str, RetentionGate] = "l2",
        use_delta_rule_forget: bool = True,
        learning_rate: float = 0.1,
        retention_strength: float = 0.1,
        momentum: float = 0.9,
        eps: float = 1e-6,
        **kwargs,
    ):
        """
        Args:
            dim_key: Dimension of keys
            dim_value: Dimension of values
            attentional_bias: Type of attentional bias or AttentionalBias instance
            retention_gate: Type of retention gate or RetentionGate instance
            use_delta_rule_forget: Whether to use delta rule forgetting term
            learning_rate: Learning rate for memory updates
            retention_strength: Strength of retention regularization
            momentum: Momentum for memory updates
            eps: Small constant for numerical stability
            **kwargs: Additional args passed to bias/gate creation
        """
        super().__init__()
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.learning_rate = learning_rate
        self.retention_strength = retention_strength
        self.momentum = momentum
        self.eps = eps
        self.use_delta_rule_forget = use_delta_rule_forget

        # Create attentional bias
        if isinstance(attentional_bias, str):
            bias_kwargs = {k.replace("bias_", ""): v for k, v in kwargs.items() if k.startswith("bias_")}
            self.attentional_bias = createAttentionalBias(attentional_bias, **bias_kwargs)
        else:
            self.attentional_bias = attentional_bias

        # Create retention gate
        if isinstance(retention_gate, str):
            gate_kwargs = {k.replace("gate_", ""): v for k, v in kwargs.items() if k.startswith("gate_")}
            gate_kwargs["strength"] = retention_strength
            self.retention_gate = createRetentionGate(retention_gate, **gate_kwargs)
        else:
            self.retention_gate = retention_gate

        # Delta rule forgetting (optional)
        if use_delta_rule_forget:
            self.delta_forget = DeltaRuleRetentionGate(forget_strength=1.0, eps=eps)

        # Learnable parameters
        self.lr_scale = nn.Parameter(torch.zeros(1))
        self.retention_scale = nn.Parameter(torch.zeros(1))

    def initMemory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize memory state.

        Returns:
            memory: Zero-initialized memory matrix (batch, dim_value, dim_key)
            momentum_buffer: Zero-initialized momentum (batch, dim_value, dim_key)
        """
        memory = torch.zeros(batch_size, self.dim_value, self.dim_key, device=device, dtype=dtype)
        momentum_buffer = torch.zeros_like(memory)
        return memory, momentum_buffer

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Update memory with new key-value association.

        Args:
            memory: Current memory (batch, dim_value, dim_key)
            key: Key tensor (batch, dim_key)
            value: Value tensor (batch, dim_value)
            momentum_buffer: Previous momentum (batch, dim_value, dim_key)

        Returns:
            memory: Updated memory
            momentum_buffer: Updated momentum
            metrics: Dictionary with surprise, loss, etc.
        """
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(memory)

        # Effective learning rate and retention strength
        lr = torch.sigmoid(self.lr_scale) * self.learning_rate * 2
        retention = torch.sigmoid(self.retention_scale) * self.retention_strength * 2

        # Normalize key for stability
        key_norm = key / (key.norm(dim=-1, keepdim=True) + self.eps)

        # Compute attentional bias gradient
        grad_attn, surprise = self.attentional_bias.computeMemoryGradient(memory, key_norm, value)

        # Compute retention gradient (if not pure delta rule)
        if not isinstance(self.retention_gate, DeltaRuleRetentionGate):
            grad_retention = self.retention_gate.computeGradient(memory, memory)
        else:
            grad_retention = torch.zeros_like(memory)

        # Compute delta rule forgetting term
        if self.use_delta_rule_forget:
            forget_term = self.delta_forget.computeForgetTerm(memory, key_norm)
        else:
            forget_term = torch.zeros_like(memory)

        # Momentum update
        gradient = grad_attn + retention * grad_retention
        momentum_buffer = self.momentum * momentum_buffer + gradient

        # Memory update: M = M - forget - lr * momentum
        memory = memory - forget_term - lr * momentum_buffer

        # Compute metrics
        loss = self.attentional_bias.computeLoss(
            torch.einsum("bvk,bk->bv", memory + forget_term + lr * momentum_buffer, key_norm),
            value
        )

        metrics = {
            "surprise": surprise.detach(),
            "loss": loss.detach(),
            "surprise_norm": surprise.norm(dim=-1).mean().detach(),
        }

        return memory, momentum_buffer, metrics

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve value from memory using query.

        Args:
            memory: Memory matrix (batch, dim_value, dim_key)
            query: Query tensor (batch, dim_key)

        Returns:
            Retrieved value (batch, dim_value)
        """
        return torch.einsum("bvk,bk->bv", memory, query)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        momentum_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process sequence through memory.

        Args:
            keys: Key sequence (batch, seq_len, dim_key)
            values: Value sequence (batch, seq_len, dim_value)
            queries: Query sequence (batch, seq_len, dim_key)
            memory: Initial memory state
            momentum_buffer: Initial momentum

        Returns:
            outputs: Retrieved values (batch, seq_len, dim_value)
            memory: Final memory state
            momentum_buffer: Final momentum
        """
        batch_size, seq_len, _ = keys.shape

        if memory is None:
            memory, momentum_buffer = self.initMemory(batch_size, keys.device, keys.dtype)

        outputs = []
        for t in range(seq_len):
            k_t = keys[:, t]
            v_t = values[:, t]
            q_t = queries[:, t]

            # Retrieve before update (causal)
            output_t = self.retrieve(memory, q_t)
            outputs.append(output_t)

            # Update memory
            memory, momentum_buffer, _ = self.update(memory, k_t, v_t, momentum_buffer)

        outputs = torch.stack(outputs, dim=1)
        return outputs, memory, momentum_buffer


class Moneta(nn.Module):
    """
    Moneta: Lp attentional bias + Lq retention gate.

    Named after the Roman goddess of memory and mother of the Muses.

    Key properties:
    - Robust to key collisions (Lp with p < 2 bounds gradient magnitude)
    - Stable memory changes (Lq with q < 2 allows necessary large updates)
    - Combination provides balance between robustness and adaptability

    Update rule:
        grad_attn = sign(Mk - v) * |Mk - v|^{p-1} * k^T
        grad_retention = sign(M_new - M_old) * |M_new - M_old|^{q-1}
        M_{t+1} = M_t - forget - lr * (grad_attn + lambda * grad_retention)

    Reference: MIRAS paper Table 1, Moneta model
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        p: float = 1.5,
        q: float = 1.5,
        learning_rate: float = 0.1,
        retention_strength: float = 0.1,
        momentum: float = 0.9,
        use_delta_rule_forget: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            dim: Model dimension
            head_dim: Dimension per head
            num_heads: Number of attention heads
            p: Lp norm power for attentional bias (1 < p <= 2)
            q: Lq norm power for retention gate (1 < q <= 2)
            learning_rate: Base learning rate
            retention_strength: Retention regularization strength
            momentum: Momentum factor
            use_delta_rule_forget: Whether to use delta rule forgetting
            dropout: Dropout probability
            eps: Numerical stability constant
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
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

        # Per-head MIRAS memory with Lp + Lq
        self.memories = nn.ModuleList([
            MirasMemory(
                dim_key=head_dim,
                dim_value=head_dim,
                attentional_bias="lp",
                retention_gate="lq",
                use_delta_rule_forget=use_delta_rule_forget,
                learning_rate=learning_rate,
                retention_strength=retention_strength,
                momentum=momentum,
                eps=eps,
                bias_p=p,
                gate_q=q,
            )
            for _ in range(num_heads)
        ])

        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.applyInit()

    def applyInit(self):
        """Initialize weights."""
        for module in [self.w_k, self.w_v, self.w_q, self.w_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through Moneta.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_states: List of (memory, momentum) tuples per head

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_states: Updated memory states
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x_norm = self.norm_input(x)

        # Project
        k = self.w_k(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.w_q(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Normalize
        k = self.norm_k(k)
        q = self.norm_q(q)

        # Initialize memory states
        if memory_states is None:
            memory_states = [None] * self.num_heads

        # Process each head
        head_outputs = []
        new_memory_states = []

        for h in range(self.num_heads):
            memory = self.memories[h]
            state = memory_states[h]

            if state is not None:
                mem, mom = state
            else:
                mem, mom = None, None

            # Process sequence
            k_h = k[:, :, h]  # (batch, seq, head_dim)
            v_h = v[:, :, h]
            q_h = q[:, :, h]

            outputs, mem, mom = memory(k_h, v_h, q_h, mem, mom)
            head_outputs.append(outputs)
            new_memory_states.append((mem, mom))

        # Combine heads
        output = torch.stack(head_outputs, dim=2)  # (batch, seq, heads, head_dim)
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Gate
        gate = self.output_gate(x)
        output = gate * output

        output = self.dropout(output)

        return output, new_memory_states


class Yaad(nn.Module):
    """
    Yaad: Huber attentional bias + L2 retention gate.

    Named from Hebrew "remembrance" or "memorial".

    Key properties:
    - Robust to outlier values (Huber loss bounds gradient for large errors)
    - Stable memory (L2 retention provides smooth regularization)
    - Good for noisy data where some values may be corrupted

    The Huber loss provides:
    - Quadratic behavior for small errors (efficient learning)
    - Linear behavior for large errors (robustness to outliers)

    Reference: MIRAS paper Table 1, Yaad model
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        huber_delta: float = 1.0,
        learning_rate: float = 0.1,
        retention_strength: float = 0.1,
        momentum: float = 0.9,
        use_delta_rule_forget: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            dim: Model dimension
            head_dim: Dimension per head
            num_heads: Number of attention heads
            huber_delta: Threshold for Huber loss transition
            learning_rate: Base learning rate
            retention_strength: Retention regularization strength
            momentum: Momentum factor
            use_delta_rule_forget: Whether to use delta rule forgetting
            dropout: Dropout probability
            eps: Numerical stability constant
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
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

        # Per-head MIRAS memory with Huber + L2
        self.memories = nn.ModuleList([
            MirasMemory(
                dim_key=head_dim,
                dim_value=head_dim,
                attentional_bias="huber",
                retention_gate="l2",
                use_delta_rule_forget=use_delta_rule_forget,
                learning_rate=learning_rate,
                retention_strength=retention_strength,
                momentum=momentum,
                eps=eps,
                bias_delta=huber_delta,
            )
            for _ in range(num_heads)
        ])

        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.applyInit()

    def applyInit(self):
        """Initialize weights."""
        for module in [self.w_k, self.w_v, self.w_q, self.w_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through Yaad.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_states: List of (memory, momentum) tuples per head

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_states: Updated memory states
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x_norm = self.norm_input(x)

        # Project
        k = self.w_k(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.w_q(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Normalize
        k = self.norm_k(k)
        q = self.norm_q(q)

        # Initialize memory states
        if memory_states is None:
            memory_states = [None] * self.num_heads

        # Process each head
        head_outputs = []
        new_memory_states = []

        for h in range(self.num_heads):
            memory = self.memories[h]
            state = memory_states[h]

            if state is not None:
                mem, mom = state
            else:
                mem, mom = None, None

            k_h = k[:, :, h]
            v_h = v[:, :, h]
            q_h = q[:, :, h]

            outputs, mem, mom = memory(k_h, v_h, q_h, mem, mom)
            head_outputs.append(outputs)
            new_memory_states.append((mem, mom))

        # Combine heads
        output = torch.stack(head_outputs, dim=2)
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Gate
        gate = self.output_gate(x)
        output = gate * output

        output = self.dropout(output)

        return output, new_memory_states


class Memora(nn.Module):
    """
    Memora: L2 attentional bias + KL retention gate.

    Named from Latin "memoria" (memory).

    Key properties:
    - Standard L2 learning (efficient for well-behaved data)
    - KL retention provides "soft thresholding" effect:
      - Small memory values are hard to increase (log penalty)
      - Large memory values are easier to modify
    - Prevents catastrophic forgetting of rarely-used associations

    This is useful when memory should preserve a "prior" distribution
    and only update when strongly justified by new evidence.

    Reference: MIRAS paper Table 1, Memora model
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
        kl_temperature: float = 1.0,
        learning_rate: float = 0.1,
        retention_strength: float = 0.1,
        momentum: float = 0.9,
        use_delta_rule_forget: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            dim: Model dimension
            head_dim: Dimension per head
            num_heads: Number of attention heads
            kl_temperature: Temperature for KL softmax normalization
            learning_rate: Base learning rate
            retention_strength: Retention regularization strength
            momentum: Momentum factor
            use_delta_rule_forget: Whether to use delta rule forgetting
            dropout: Dropout probability
            eps: Numerical stability constant
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
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

        # Per-head MIRAS memory with L2 + KL
        self.memories = nn.ModuleList([
            MirasMemory(
                dim_key=head_dim,
                dim_value=head_dim,
                attentional_bias="l2",
                retention_gate="kl",
                use_delta_rule_forget=use_delta_rule_forget,
                learning_rate=learning_rate,
                retention_strength=retention_strength,
                momentum=momentum,
                eps=eps,
                gate_temperature=kl_temperature,
            )
            for _ in range(num_heads)
        ])

        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.applyInit()

    def applyInit(self):
        """Initialize weights."""
        for module in [self.w_k, self.w_v, self.w_q, self.w_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through Memora.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_states: List of (memory, momentum) tuples per head

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_states: Updated memory states
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x_norm = self.norm_input(x)

        # Project
        k = self.w_k(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.w_v(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.w_q(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Normalize
        k = self.norm_k(k)
        q = self.norm_q(q)

        # Initialize memory states
        if memory_states is None:
            memory_states = [None] * self.num_heads

        # Process each head
        head_outputs = []
        new_memory_states = []

        for h in range(self.num_heads):
            memory = self.memories[h]
            state = memory_states[h]

            if state is not None:
                mem, mom = state
            else:
                mem, mom = None, None

            k_h = k[:, :, h]
            v_h = v[:, :, h]
            q_h = q[:, :, h]

            outputs, mem, mom = memory(k_h, v_h, q_h, mem, mom)
            head_outputs.append(outputs)
            new_memory_states.append((mem, mom))

        # Combine heads
        output = torch.stack(head_outputs, dim=2)
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Gate
        gate = self.output_gate(x)
        output = gate * output

        output = self.dropout(output)

        return output, new_memory_states


def createMirasModel(
    model_type: str,
    dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create MIRAS models.

    Args:
        model_type: One of 'moneta', 'yaad', 'memora', 'custom'
        dim: Model dimension
        **kwargs: Additional arguments for the specific model

    Returns:
        MIRAS model instance
    """
    model_map = {
        "moneta": Moneta,
        "yaad": Yaad,
        "memora": Memora,
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")

    return model_map[model_type](dim=dim, **kwargs)
