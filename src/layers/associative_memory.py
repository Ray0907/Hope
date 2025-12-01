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
