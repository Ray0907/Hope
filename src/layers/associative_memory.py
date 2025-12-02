"""
Associative Memory modules for HOPE architecture.

Implements key-value associative memory with various update rules:
- Standard outer product update (linear attention)
- Delta rule update for better capacity management

Reference: Nested Learning paper, Section 2.1, Equations 13-16, 28-29
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AssociativeMemory(nn.Module):
    """
    Base class for associative memory modules.

    Associative memory maps keys to values through an optimization process.
    M* = argmin_M L(M(K); V)
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 1.0,
        momentum: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.eps = eps

    def initMemory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize memory state."""
        raise NotImplementedError

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Update memory with new key-value pair."""
        raise NotImplementedError

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve value from memory using query."""
        raise NotImplementedError


class LinearAttentionMemory(AssociativeMemory):
    """
    Linear attention as associative memory.

    Update rule (Eq. 13):
        M_{t+1} = M_t + v_t * k_t^T

    This corresponds to optimizing:
        min_M <M*k, v> + ||M - M_t||^2

    with gradient descent (learning rate = 1).
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 1.0,
        momentum: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(dim_key, dim_value, learning_rate, momentum, eps)

        # Optional momentum buffer
        self.register_buffer("momentum_buffer", None)

    def initMemory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Initialize memory as zero matrix.

        Returns:
            Memory tensor of shape (batch_size, dim_value, dim_key)
        """
        return torch.zeros(
            batch_size, self.dim_value, self.dim_key, device=device, dtype=dtype
        )

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Update memory with outer product rule.

        Args:
            memory: Current memory state (batch, dim_value, dim_key)
            key: Key tensor (batch, seq_len, dim_key) or (batch, dim_key)
            value: Value tensor (batch, seq_len, dim_value) or (batch, dim_value)
            momentum_state: Optional momentum buffer

        Returns:
            Updated memory and momentum state
        """
        # Handle both batched and single token cases
        if key.dim() == 2:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        # Compute update: v * k^T for each position
        # key: (batch, seq, dim_key), value: (batch, seq, dim_value)
        # update: (batch, dim_value, dim_key)
        update = torch.einsum("bsk,bsv->bvk", key, value)

        if self.momentum > 0 and momentum_state is not None:
            momentum_state = self.momentum * momentum_state + update
            memory = memory + self.learning_rate * momentum_state
        else:
            memory = memory + self.learning_rate * update
            momentum_state = update if self.momentum > 0 else None

        return memory, momentum_state

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory using query.

        Args:
            memory: Memory state (batch, dim_value, dim_key)
            query: Query tensor (batch, seq_len, dim_key) or (batch, dim_key)

        Returns:
            Retrieved values (batch, seq_len, dim_value) or (batch, dim_value)
        """
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True

        # output = M * q: (batch, dim_value, dim_key) @ (batch, seq, dim_key)^T
        output = torch.einsum("bvk,bsk->bsv", memory, query)

        if squeeze_output:
            output = output.squeeze(1)

        return output


class DeltaRuleMemory(AssociativeMemory):
    """
    Delta rule based associative memory.

    Update rule (Eq. 28-29 from Nested Learning paper):
        W_{t+1} = W_t(I - x_t * x_t^T) - eta * grad_y L(W_t; x_t) tensor x_t

    Expanded form:
        W_{t+1} = W_t - W_t * x_t * x_t^T - eta * (W_t * x_t - v_t) * x_t^T

    Where:
        - First term (W_t * x_t * x_t^T): Forgetting - removes old association for key x_t
        - Second term (eta * (W_t * x_t - v_t) * x_t^T): Learning - stores new association

    This uses L2 regression objective instead of dot-product similarity,
    allowing the memory to better manage its limited capacity.
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(dim_key, dim_value, learning_rate, momentum, eps)

    def initMemory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize memory as zero matrix."""
        return torch.zeros(
            batch_size, self.dim_value, self.dim_key, device=device, dtype=dtype
        )

    def computeSurprise(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Local Surprise Signal (LSS).

        LSS = M * k - v (prediction error)

        Args:
            memory: Current memory (batch, dim_value, dim_key)
            key: Key tensor (batch, dim_key)
            value: Value tensor (batch, dim_value)

        Returns:
            Surprise signal (batch, dim_value)
        """
        # Predicted value: M @ k
        predicted = torch.einsum("bvk,bk->bv", memory, key)
        # Surprise: prediction error
        surprise = predicted - value
        return surprise

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Update memory with delta rule (Eq. 28-29 from Nested Learning paper).

        Update formula:
            M_{t+1} = M_t - M_t * k_t * k_t^T - eta * (M_t * k_t - v_t) * k_t^T

        Where:
            - M_t * k_t * k_t^T: Forgetting term (removes old association)
            - eta * (M_t * k_t - v_t) * k_t^T: Learning term (gradient descent on L2 loss)

        Args:
            memory: Current memory state (batch, dim_value, dim_key)
            key: Key tensor (batch, seq_len, dim_key) or (batch, dim_key)
            value: Value tensor (batch, seq_len, dim_value) or (batch, dim_value)
            momentum_state: Optional momentum buffer

        Returns:
            Updated memory and momentum state
        """
        # Handle both batched and single token cases
        single_token = key.dim() == 2
        if single_token:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        batch_size, seq_len, _ = key.shape

        # Process each token sequentially for proper recurrence
        for t in range(seq_len):
            key_t = key[:, t, :]  # (batch, dim_key)
            value_t = value[:, t, :]  # (batch, dim_value)

            # Normalize key for stable updates
            key_norm = key_t / (key_t.norm(dim=-1, keepdim=True) + self.eps)

            # Compute prediction: M @ k
            predicted = torch.einsum("bvk,bk->bv", memory, key_norm)

            # Compute surprise (prediction error): M*k - v
            surprise = predicted - value_t  # (batch, dim_value)

            # Forgetting term: M * k * k^T (coefficient = 1 per Eq. 28-29)
            # This projects out the current key direction from memory
            forget_term = torch.einsum("bv,bk->bvk", predicted, key_norm)

            # Learning term: eta * (M*k - v) * k^T = eta * surprise * k^T
            learn_term = torch.einsum("bv,bk->bvk", surprise, key_norm)

            # Delta rule update: M = M - M*k*k^T - eta * surprise * k^T
            memory = memory - forget_term - self.learning_rate * learn_term

        return memory, momentum_state

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve from memory using query."""
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True

        output = torch.einsum("bvk,bsk->bsv", memory, query)

        if squeeze_output:
            output = output.squeeze(1)

        return output


class GatedDeltaRuleMemory(DeltaRuleMemory):
    """
    Gated Delta Rule Memory with learnable gates.

    Extends the delta rule (Eq. 28-29) with input and forget gates
    similar to LSTM for better gradient flow and adaptive control.

    Update formula:
        M_{t+1} = M_t - g_f * (M_t * k_t) * k_t^T - g_i * eta * (M_t * k_t - v_t) * k_t^T

    Where g_f and g_i are learned forget and input gates.
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(dim_key, dim_value, learning_rate, momentum, eps)

        # Learnable gates
        self.input_gate = nn.Linear(dim_key + dim_value, 1, bias=True)
        self.forget_gate = nn.Linear(dim_key + dim_value, 1, bias=True)

        # Initialize gates (start with gates mostly open)
        nn.init.zeros_(self.input_gate.weight)
        nn.init.constant_(self.input_gate.bias, 1.0)
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.constant_(self.forget_gate.bias, 1.0)

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Update with gated delta rule (Eq. 28-29 with gates)."""
        single_token = key.dim() == 2
        if single_token:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        batch_size, seq_len, _ = key.shape

        for t in range(seq_len):
            key_t = key[:, t, :]
            value_t = value[:, t, :]

            # Compute gates
            gate_input = torch.cat([key_t, value_t], dim=-1)
            input_g = torch.sigmoid(self.input_gate(gate_input))
            forget_g = torch.sigmoid(self.forget_gate(gate_input))

            # Normalize key
            key_norm = key_t / (key_t.norm(dim=-1, keepdim=True) + self.eps)

            # Compute prediction: M @ k
            predicted = torch.einsum("bvk,bk->bv", memory, key_norm)

            # Compute surprise (prediction error): M*k - v
            surprise = predicted - value_t

            # Gated forgetting: g_f * (M*k) * k^T
            forget_term = torch.einsum("bv,bk->bvk", predicted, key_norm)

            # Gated learning: g_i * eta * surprise * k^T
            learn_term = torch.einsum("bv,bk->bvk", surprise, key_norm)

            # Delta rule with gates: M = M - g_f * forget_term - g_i * eta * learn_term
            memory = (
                memory
                - forget_g.unsqueeze(-1) * forget_term
                - input_g.unsqueeze(-1) * self.learning_rate * learn_term
            )

        return memory, momentum_state


class PersistentMemory(nn.Module):
    """
    Persistent Memory module.

    Data-independent learnable memory that encodes task-relevant general knowledge.
    Unlike short-term (per-token) and long-term (sequence) memory, persistent memory
    is shared across all inputs and learned during pre-training.

    Architecture:
        - Learnable memory bank: P in R^{num_slots x dim}
        - Query-based retrieval: output = softmax(Q @ P^T) @ P
        - Combined with dynamic memory for final output

    Reference: Nested Learning paper Section 3 "Persistent Memory"
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        num_heads: int = 1,
        dropout: float = 0.0,
        combine_mode: str = "gate",
    ):
        """
        Initialize Persistent Memory.

        Args:
            dim: Dimension of memory slots
            num_slots: Number of persistent memory slots
            num_heads: Number of attention heads for retrieval
            dropout: Dropout probability
            combine_mode: How to combine with dynamic memory ('gate', 'add', 'concat')
        """
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.combine_mode = combine_mode

        # Learnable persistent memory bank
        self.memory_bank = nn.Parameter(torch.randn(num_slots, dim) * 0.02)

        # Query projection for retrieval
        self.w_q = nn.Linear(dim, dim, bias=False)

        # Key projection for memory bank (optional, can use memory directly)
        self.w_k = nn.Linear(dim, dim, bias=False)

        # Value projection
        self.w_v = nn.Linear(dim, dim, bias=False)

        # Output projection
        self.w_o = nn.Linear(dim, dim, bias=False)

        # Combination gate (if using gate mode)
        if combine_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )
        elif combine_mode == "concat":
            self.combine_proj = nn.Linear(dim * 2, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        dynamic_memory_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retrieve from persistent memory and optionally combine with dynamic memory.

        Args:
            x: Input tensor (batch, seq_len, dim)
            dynamic_memory_output: Optional output from dynamic memory (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project query
        q = self.w_q(x)  # (batch, seq, dim)

        # Project memory bank to keys and values
        k = self.w_k(self.memory_bank)  # (num_slots, dim)
        v = self.w_v(self.memory_bank)  # (num_slots, dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(self.num_slots, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = v.view(self.num_slots, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Attention: (batch, heads, seq, head_dim) @ (heads, num_slots, head_dim)^T
        attn = torch.einsum("bhsd,hnd->bhsn", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Retrieve: (batch, heads, seq, num_slots) @ (heads, num_slots, head_dim)
        output = torch.einsum("bhsn,hnd->bhsd", attn, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.w_o(output)

        # Combine with dynamic memory if provided
        if dynamic_memory_output is not None:
            if self.combine_mode == "gate":
                gate = self.gate(torch.cat([output, dynamic_memory_output], dim=-1))
                output = gate * output + (1 - gate) * dynamic_memory_output
            elif self.combine_mode == "add":
                output = output + dynamic_memory_output
            elif self.combine_mode == "concat":
                output = self.combine_proj(torch.cat([output, dynamic_memory_output], dim=-1))

        return output


class DynamicDecayMemory(AssociativeMemory):
    """
    Memory with Dynamic Decay Mechanism.

    Implements adaptive forgetting where the decay rate is based on:
    1. Memory utilization (how full the memory is)
    2. Surprise magnitude (how unexpected the input is)

    This is equivalent to weight decay + momentum in optimization terms.

    Update rule:
        M_{t+1} = (1 - decay_t) * M_t - eta * grad
        decay_t = base_decay * (1 + alpha * utilization + beta * surprise_norm)

    Reference: Nested Learning paper Section 3 "Dynamic Decay"
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 0.1,
        base_decay: float = 0.01,
        utilization_weight: float = 0.1,
        surprise_weight: float = 0.1,
        max_decay: float = 0.5,
        eps: float = 1e-6,
    ):
        """
        Initialize Dynamic Decay Memory.

        Args:
            dim_key: Key dimension
            dim_value: Value dimension
            learning_rate: Base learning rate for updates
            base_decay: Base decay rate
            utilization_weight: Weight for utilization-based decay (alpha)
            surprise_weight: Weight for surprise-based decay (beta)
            max_decay: Maximum allowed decay rate
            eps: Small constant for numerical stability
        """
        super().__init__(dim_key, dim_value, learning_rate, 0.0, eps)
        self.base_decay = base_decay
        self.utilization_weight = utilization_weight
        self.surprise_weight = surprise_weight
        self.max_decay = max_decay

    def computeUtilization(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Compute memory utilization as normalized Frobenius norm.

        Returns utilization in [0, 1] range.
        """
        # Frobenius norm of memory matrix
        mem_norm = torch.norm(memory, dim=(-2, -1), keepdim=True)
        # Normalize by theoretical maximum (all entries = 1)
        max_norm = (self.dim_value * self.dim_key) ** 0.5
        utilization = mem_norm / (max_norm + self.eps)
        return utilization.clamp(0, 1)

    def computeDynamicDecay(
        self,
        memory: torch.Tensor,
        surprise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive decay rate based on utilization and surprise.

        Args:
            memory: Current memory state (batch, dim_value, dim_key)
            surprise: Surprise signal (batch, dim_value)

        Returns:
            Decay rate (batch, 1, 1)
        """
        # Memory utilization
        utilization = self.computeUtilization(memory)  # (batch, 1, 1)

        # Surprise magnitude (normalized)
        surprise_norm = torch.norm(surprise, dim=-1, keepdim=True)  # (batch, 1)
        surprise_norm = surprise_norm / (surprise_norm.max() + self.eps)
        surprise_norm = surprise_norm.unsqueeze(-1)  # (batch, 1, 1)

        # Dynamic decay: base + alpha * util + beta * surprise
        decay = self.base_decay * (
            1.0
            + self.utilization_weight * utilization
            + self.surprise_weight * surprise_norm
        )

        # Clamp to max decay
        decay = decay.clamp(0, self.max_decay)

        return decay

    def initMemory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize memory as zero matrix."""
        return torch.zeros(
            batch_size, self.dim_value, self.dim_key, device=device, dtype=dtype
        )

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Update memory with dynamic decay and delta rule.

        Args:
            memory: Current memory (batch, dim_value, dim_key)
            key: Key tensor (batch, dim_key) or (batch, seq, dim_key)
            value: Value tensor (batch, dim_value) or (batch, seq, dim_value)

        Returns:
            Updated memory and momentum state
        """
        single_token = key.dim() == 2
        if single_token:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        batch_size, seq_len, _ = key.shape

        for t in range(seq_len):
            key_t = key[:, t, :]
            value_t = value[:, t, :]

            # Normalize key
            key_norm = key_t / (key_t.norm(dim=-1, keepdim=True) + self.eps)

            # Compute prediction and surprise
            predicted = torch.einsum("bvk,bk->bv", memory, key_norm)
            surprise = predicted - value_t

            # Compute dynamic decay
            decay = self.computeDynamicDecay(memory, surprise)

            # Apply decay to memory
            memory = (1 - decay) * memory

            # Standard delta rule update
            forget_term = torch.einsum("bv,bk->bvk", predicted, key_norm)
            learn_term = torch.einsum("bv,bk->bvk", surprise, key_norm)

            memory = memory - forget_term - self.learning_rate * learn_term

        return memory, momentum_state

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve from memory using query."""
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True

        output = torch.einsum("bvk,bsk->bsv", memory, query)

        if squeeze_output:
            output = output.squeeze(1)

        return output


class SurpriseGatedMemory(AssociativeMemory):
    """
    Surprise-based Memory Gating.

    Only updates memory for surprising inputs (where prediction error exceeds threshold).
    This prevents redundant updates for predictable inputs and focuses memory capacity
    on novel information.

    Gating mechanism:
        gate = sigmoid((|surprise| - threshold) / temperature)
        M_{t+1} = M_t - gate * (forget_term + lr * learn_term)

    Reference: Titans paper "Surprise-based Updates"
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 0.1,
        surprise_threshold: float = 0.1,
        temperature: float = 0.1,
        adaptive_threshold: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initialize Surprise-Gated Memory.

        Args:
            dim_key: Key dimension
            dim_value: Value dimension
            learning_rate: Learning rate for updates
            surprise_threshold: Base threshold for gating
            temperature: Temperature for soft gating (lower = sharper)
            adaptive_threshold: Whether to adapt threshold based on running statistics
            eps: Small constant for numerical stability
        """
        super().__init__(dim_key, dim_value, learning_rate, 0.0, eps)
        self.base_threshold = surprise_threshold
        self.temperature = temperature
        self.adaptive_threshold = adaptive_threshold

        # Running statistics for adaptive threshold
        if adaptive_threshold:
            self.register_buffer("running_mean", torch.zeros(1))
            self.register_buffer("running_var", torch.ones(1))
            self.register_buffer("num_updates", torch.zeros(1))
            self.momentum_stat = 0.99

    def computeSurpriseGate(
        self,
        surprise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gating value based on surprise magnitude.

        Args:
            surprise: Surprise signal (batch, dim_value)

        Returns:
            Gate values (batch, 1) in [0, 1]
        """
        surprise_norm = torch.norm(surprise, dim=-1, keepdim=True)

        # Get threshold (adaptive or fixed)
        if self.adaptive_threshold and self.training:
            # Update running statistics
            batch_mean = surprise_norm.mean()
            batch_var = surprise_norm.var()

            self.running_mean = (
                self.momentum_stat * self.running_mean
                + (1 - self.momentum_stat) * batch_mean
            )
            self.running_var = (
                self.momentum_stat * self.running_var
                + (1 - self.momentum_stat) * batch_var
            )
            self.num_updates += 1

            # Adaptive threshold: mean + base_threshold * std
            threshold = self.running_mean + self.base_threshold * (self.running_var + self.eps).sqrt()
        else:
            threshold = self.base_threshold

        # Soft gating: sigmoid((|surprise| - threshold) / temperature)
        gate = torch.sigmoid((surprise_norm - threshold) / self.temperature)

        return gate

    def initMemory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize memory as zero matrix."""
        return torch.zeros(
            batch_size, self.dim_value, self.dim_key, device=device, dtype=dtype
        )

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Update memory with surprise-based gating.

        Args:
            memory: Current memory (batch, dim_value, dim_key)
            key: Key tensor (batch, dim_key) or (batch, seq, dim_key)
            value: Value tensor (batch, dim_value) or (batch, seq, dim_value)

        Returns:
            Updated memory, momentum state, and surprise gate values
        """
        single_token = key.dim() == 2
        if single_token:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        batch_size, seq_len, _ = key.shape
        gates = []

        for t in range(seq_len):
            key_t = key[:, t, :]
            value_t = value[:, t, :]

            # Normalize key
            key_norm = key_t / (key_t.norm(dim=-1, keepdim=True) + self.eps)

            # Compute prediction and surprise
            predicted = torch.einsum("bvk,bk->bv", memory, key_norm)
            surprise = predicted - value_t

            # Compute surprise gate
            gate = self.computeSurpriseGate(surprise)
            gates.append(gate)

            # Gated delta rule update
            forget_term = torch.einsum("bv,bk->bvk", predicted, key_norm)
            learn_term = torch.einsum("bv,bk->bvk", surprise, key_norm)

            # Only update if surprise exceeds threshold (via soft gate)
            update = forget_term + self.learning_rate * learn_term
            memory = memory - gate.unsqueeze(-1) * update

        # Stack gates for monitoring
        gate_values = torch.stack(gates, dim=1)  # (batch, seq, 1)

        return memory, momentum_state, gate_values

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve from memory using query."""
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True

        output = torch.einsum("bvk,bsk->bsv", memory, query)

        if squeeze_output:
            output = output.squeeze(1)

        return output


class MemoryWithCapacityManagement(AssociativeMemory):
    """
    Memory with Capacity Management.

    Implements bounds on memory to prevent saturation and tracks utilization.
    Features:
    1. Memory normalization to prevent unbounded growth
    2. Utilization tracking
    3. Automatic capacity adjustment via pruning

    Reference: Nested Learning paper "Memory Capacity"
    """

    def __init__(
        self,
        dim_key: int,
        dim_value: int,
        learning_rate: float = 0.1,
        max_norm: float = 10.0,
        target_utilization: float = 0.7,
        prune_threshold: float = 0.01,
        eps: float = 1e-6,
    ):
        """
        Initialize Memory with Capacity Management.

        Args:
            dim_key: Key dimension
            dim_value: Value dimension
            learning_rate: Learning rate for updates
            max_norm: Maximum allowed memory norm
            target_utilization: Target utilization level
            prune_threshold: Threshold for pruning small entries
            eps: Small constant for numerical stability
        """
        super().__init__(dim_key, dim_value, learning_rate, 0.0, eps)
        self.max_norm = max_norm
        self.target_utilization = target_utilization
        self.prune_threshold = prune_threshold

    def initMemory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize memory as zero matrix."""
        return torch.zeros(
            batch_size, self.dim_value, self.dim_key, device=device, dtype=dtype
        )

    def normalizeMemory(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Normalize memory to prevent unbounded growth.

        Uses spectral normalization style: M = M / max(1, ||M||_F / max_norm)
        """
        mem_norm = torch.norm(memory, dim=(-2, -1), keepdim=True)
        scale = torch.clamp(mem_norm / self.max_norm, min=1.0)
        return memory / scale

    def pruneMemory(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Prune small entries to maintain sparsity and capacity.

        Entries below threshold are set to zero.
        """
        mask = torch.abs(memory) > self.prune_threshold
        return memory * mask.float()

    def getUtilization(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Compute memory utilization metrics.

        Returns:
            Dictionary with utilization metrics
        """
        # Frobenius norm
        frob_norm = torch.norm(memory, dim=(-2, -1))

        # Sparsity (fraction of near-zero entries)
        sparsity = (torch.abs(memory) < self.prune_threshold).float().mean(dim=(-2, -1))

        # Effective rank approximation
        svd_norm = torch.linalg.svdvals(memory).sum(dim=-1)
        eff_rank = svd_norm / (memory.abs().max(dim=-1)[0].max(dim=-1)[0] + self.eps)

        return {
            "frobenius_norm": frob_norm,
            "sparsity": sparsity,
            "effective_rank": eff_rank,
        }

    def update(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        momentum_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Update memory with capacity management.

        Args:
            memory: Current memory (batch, dim_value, dim_key)
            key: Key tensor
            value: Value tensor

        Returns:
            Updated memory and momentum state
        """
        single_token = key.dim() == 2
        if single_token:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        batch_size, seq_len, _ = key.shape

        for t in range(seq_len):
            key_t = key[:, t, :]
            value_t = value[:, t, :]

            # Normalize key
            key_norm = key_t / (key_t.norm(dim=-1, keepdim=True) + self.eps)

            # Compute prediction and surprise
            predicted = torch.einsum("bvk,bk->bv", memory, key_norm)
            surprise = predicted - value_t

            # Delta rule update
            forget_term = torch.einsum("bv,bk->bvk", predicted, key_norm)
            learn_term = torch.einsum("bv,bk->bvk", surprise, key_norm)
            memory = memory - forget_term - self.learning_rate * learn_term

            # Apply capacity management
            memory = self.normalizeMemory(memory)

            # Periodic pruning (every few steps during training)
            if t % 10 == 0:
                memory = self.pruneMemory(memory)

        return memory, momentum_state

    def retrieve(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve from memory using query."""
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True

        output = torch.einsum("bvk,bsk->bsv", memory, query)

        if squeeze_output:
            output = output.squeeze(1)

        return output
