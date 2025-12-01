"""
Neural Memory modules for HOPE architecture.

Implements MLP-based memory with surprise-based updates (Titans-style).
The memory learns to store and retrieve information through an internal
optimization process.

Reference: Nested Learning paper, Section 2.1, 2.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from typing import Optional, Tuple, List, Dict
import math


class MLP(nn.Module):
    """Simple MLP with residual connection."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dim = dim * expansion

        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DeepMemory(nn.Module):
    """
    Deep Memory module using MLP.

    Instead of a linear matrix-valued memory, uses a multi-layer MLP
    for higher capacity key-value mappings. The MLP parameters serve
    as the memory state.

    Reference: Section 2.3 "More Expressive Memory"
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.expansion = expansion
        hidden_dim = dim * expansion

        # Build MLP layers explicitly for parameter access
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

        # Initialize with small weights for stability
        self.applyInit()

    def applyInit(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through memory MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def forwardWithParams(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with external parameters.

        This allows using modified parameters without changing the module's state.
        Supports batched parameters where each batch item has its own weights.

        Args:
            x: Input tensor (batch, dim) or (dim,)
            params: Dictionary with fc1.weight (batch, hidden, dim), etc.
        """
        # Check if params are batched (3D weights)
        if params["fc1.weight"].dim() == 3:
            # Batched forward: x is (batch, dim), weight is (batch, hidden, dim)
            # h = x @ W^T + b = einsum("bd,bhd->bh", x, W) + b
            h = torch.einsum("bd,bhd->bh", x, params["fc1.weight"]) + params["fc1.bias"]
            h = F.gelu(h)
            out = torch.einsum("bh,bdh->bd", h, params["fc2.weight"]) + params["fc2.bias"]
        else:
            # Standard forward
            h = F.linear(x, params["fc1.weight"], params["fc1.bias"])
            h = F.gelu(h)
            out = F.linear(h, params["fc2.weight"], params["fc2.bias"])
        return out

    def getParamsDict(self) -> Dict[str, torch.Tensor]:
        """Get parameters as a dictionary."""
        return {
            "fc1.weight": self.fc1.weight,
            "fc1.bias": self.fc1.bias,
            "fc2.weight": self.fc2.weight,
            "fc2.bias": self.fc2.bias,
        }


class NeuralMemoryState:
    """
    State container for Neural Memory.

    Holds per-batch memory parameters and momentum buffers.
    """

    def __init__(
        self,
        params: Dict[str, torch.Tensor],
        momentum: Dict[str, torch.Tensor],
        step: int = 0,
    ):
        self.params = params
        self.momentum = momentum
        self.step = step

    def clone(self) -> "NeuralMemoryState":
        """Create a deep copy of the state."""
        return NeuralMemoryState(
            params={k: v.clone() for k, v in self.params.items()},
            momentum={k: v.clone() for k, v in self.momentum.items()},
            step=self.step,
        )


class NeuralMemory(nn.Module):
    """
    Neural Memory module with surprise-based updates.

    This is the core memory component from Titans, which learns to:
    1. Project inputs to keys, values, queries
    2. Compute surprise (prediction error) based on current memory
    3. Update memory parameters based on surprise
    4. Retrieve relevant information using queries

    The key innovation is that memory updates happen at inference time,
    allowing the model to continually learn from context.

    Reference: Titans paper, Nested Learning Section 3
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 1,
        memory_depth: int = 2,
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
        self.memory_depth = memory_depth
        self.base_learning_rate = learning_rate
        self.base_momentum = momentum
        self.use_delta_rule = use_delta_rule
        self.eps = eps

        # Projection layers
        self.w_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_q = nn.Linear(dim, num_heads * head_dim, bias=False)

        # Output projection
        self.w_o = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Memory module (MLP-based) - one per head
        self.memories = nn.ModuleList([
            DeepMemory(head_dim, num_layers=memory_depth, expansion=4, dropout=dropout)
            for _ in range(num_heads)
        ])

        # Layer norm for stability
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)

        # Learnable learning rate and momentum (per-head)
        self.lr_scale = nn.Parameter(torch.zeros(num_heads))  # sigmoid -> 0.5
        self.momentum_scale = nn.Parameter(torch.zeros(num_heads))  # sigmoid -> 0.5
        self.forget_scale = nn.Parameter(torch.zeros(num_heads))  # For delta rule

        # Gate for output
        self.output_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def initMemoryState(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> List[NeuralMemoryState]:
        """
        Initialize memory state for a new sequence.

        Returns list of NeuralMemoryState (one per head).
        """
        states = []
        for head_idx, memory in enumerate(self.memories):
            # Get base parameters and expand for batch
            base_params = memory.getParamsDict()
            params = {}
            momentum = {}

            for name, param in base_params.items():
                # Clone params for each batch item: (batch, *param_shape)
                # For a 2D weight (hidden, dim), result is (batch, hidden, dim)
                expanded = param.unsqueeze(0).repeat(batch_size, *([1] * param.dim()))
                params[name] = expanded.to(device=device, dtype=dtype)
                momentum[name] = torch.zeros_like(expanded)

            states.append(NeuralMemoryState(params, momentum, step=0))

        return states

    def computeSurpriseLoss(
        self,
        memory: DeepMemory,
        params: Dict[str, torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute surprise loss: ||M(k) - v||^2

        Args:
            memory: Memory module
            params: Current memory parameters
            key: Key tensor (head_dim,)
            value: Value tensor (head_dim,)

        Returns:
            Scalar loss value
        """
        predicted = memory.forwardWithParams(key, params)
        loss = 0.5 * torch.sum((predicted - value) ** 2)
        return loss

    def computeParamGradients(
        self,
        memory: DeepMemory,
        params: Dict[str, torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients of surprise loss w.r.t. memory parameters.

        Uses analytical gradients for efficiency.

        Args:
            memory: Memory module
            params: Current memory parameters (batch, *param_shape)
            key: Key tensor (batch, head_dim)
            value: Value tensor (batch, head_dim)

        Returns:
            Dictionary of parameter gradients
        """
        batch_size = key.shape[0]
        grads = {}

        # Forward pass to get activations with batched params
        # h = W1 @ k + b1, a = gelu(h), out = W2 @ a + b2
        if params["fc1.weight"].dim() == 3:
            # Batched: params["fc1.weight"] is (batch, hidden, dim)
            h = torch.einsum("bd,bhd->bh", key, params["fc1.weight"]) + params["fc1.bias"]
            a = F.gelu(h)
            out = torch.einsum("bh,bdh->bd", a, params["fc2.weight"]) + params["fc2.bias"]

            # Backward pass: d_loss/d_out = out - value (for L2 loss)
            d_out = out - value  # (batch, head_dim)

            # Gradient for fc2: d_loss/d_W2 = d_out @ a^T
            grads["fc2.weight"] = torch.einsum("bi,bj->bij", d_out, a)  # (batch, head_dim, hidden)
            grads["fc2.bias"] = d_out  # (batch, head_dim)

            # Backprop through fc2: d_a = d_out @ W2
            d_a = torch.einsum("bd,bdh->bh", d_out, params["fc2.weight"])  # (batch, hidden)
        else:
            h = F.linear(key, params["fc1.weight"], params["fc1.bias"])
            a = F.gelu(h)
            out = F.linear(a, params["fc2.weight"], params["fc2.bias"])

            d_out = out - value
            grads["fc2.weight"] = torch.einsum("bi,bj->bij", d_out, a)
            grads["fc2.bias"] = d_out
            d_a = torch.einsum("bi,ij->bj", d_out, params["fc2.weight"])

        # Backprop through GELU
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        cdf = 0.5 * (1.0 + torch.erf(h / math.sqrt(2.0)))
        pdf = sqrt_2_over_pi * torch.exp(-0.5 * h ** 2)
        gelu_grad = cdf + h * pdf
        d_h = d_a * gelu_grad  # (batch, hidden)

        # Gradient for fc1: d_loss/d_W1 = d_h @ k^T
        grads["fc1.weight"] = torch.einsum("bi,bj->bij", d_h, key)  # (batch, hidden, head_dim)
        grads["fc1.bias"] = d_h  # (batch, hidden)

        return grads

    def updateMemoryGradient(
        self,
        state: NeuralMemoryState,
        grads: Dict[str, torch.Tensor],
        lr: float,
        momentum: float,
    ) -> NeuralMemoryState:
        """
        Update memory parameters using gradient descent with momentum.

        Update rule:
            m_{t+1} = momentum * m_t + grad
            theta_{t+1} = theta_t - lr * m_{t+1}
        """
        new_params = {}
        new_momentum = {}

        for name in state.params.keys():
            # Update momentum
            new_mom = momentum * state.momentum[name] + grads[name]
            new_momentum[name] = new_mom

            # Update parameters
            new_params[name] = state.params[name] - lr * new_mom

        return NeuralMemoryState(new_params, new_momentum, state.step + 1)

    def updateMemoryDelta(
        self,
        state: NeuralMemoryState,
        grads: Dict[str, torch.Tensor],
        key: torch.Tensor,
        lr: float,
        momentum: float,
        forget_factor: float,
    ) -> NeuralMemoryState:
        """
        Update memory parameters using delta rule (Eq. 28-29 from Nested Learning paper).

        The update follows:
            theta_{t+1} = theta_t - theta_t * k * k^T - lr * (theta_t * k - v) * k^T

        For MLP parameters, we approximate this as:
            m_{t+1} = momentum * m_t + grad
            theta_{t+1} = theta_t - forget_term - lr * m_{t+1}

        Where the forgetting term projects out the key direction from parameters.
        """
        new_params = {}
        new_momentum = {}

        # Normalize key for stable updates
        key_norm = key / (key.norm(dim=-1, keepdim=True) + self.eps)

        for name in state.params.keys():
            # Update momentum with gradient
            new_mom = momentum * state.momentum[name] + grads[name]
            new_momentum[name] = new_mom

            # For delta rule: compute forgetting term based on key projection
            # This removes old associations related to current key
            if "fc1" in name and state.params[name].dim() >= 2:
                # fc1.weight: (batch, hidden, dim) - project out key direction
                # forget_term = param @ k @ k^T
                if state.params[name].dim() == 3:  # batched
                    pk = torch.einsum("bhd,bd->bh", state.params[name], key_norm)
                    forget_term = torch.einsum("bh,bd->bhd", pk, key_norm)
                else:
                    pk = torch.einsum("hd,d->h", state.params[name], key_norm.squeeze(0))
                    forget_term = torch.einsum("h,d->hd", pk, key_norm.squeeze(0))
                new_params[name] = state.params[name] - forget_term - lr * new_mom
            else:
                # For biases and fc2, just apply gradient update
                new_params[name] = state.params[name] - lr * new_mom

        return NeuralMemoryState(new_params, new_momentum, state.step + 1)

    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[List[NeuralMemoryState]] = None,
    ) -> Tuple[torch.Tensor, List[NeuralMemoryState]]:
        """
        Forward pass with memory update.

        Args:
            x: Input tensor (batch, seq_len, dim)
            memory_states: Optional previous memory states (one per head)

        Returns:
            output: Output tensor (batch, seq_len, dim)
            memory_states: Updated memory states
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory states if needed
        if memory_states is None:
            memory_states = self.initMemoryState(batch_size, x.device, x.dtype)

        # Project to keys, values, queries
        k = self.w_k(x)  # (batch, seq, num_heads * head_dim)
        v = self.w_v(x)
        q = self.w_q(x)

        # Reshape to (batch, num_heads, seq, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Normalize
        k = self.norm_k(k)
        v = self.norm_v(v)
        q = self.norm_q(q)

        # Get learning rates and momentum per head
        lrs = torch.sigmoid(self.lr_scale) * self.base_learning_rate * 2
        moms = torch.sigmoid(self.momentum_scale) * self.base_momentum * 2
        forgets = torch.sigmoid(self.forget_scale) * 0.5

        # Process each head
        head_outputs = []
        new_states = []

        for head_idx in range(self.num_heads):
            memory = self.memories[head_idx]
            state = memory_states[head_idx]
            lr = lrs[head_idx].item()
            mom = moms[head_idx].item()
            forget = forgets[head_idx].item()

            k_head = k[:, head_idx]  # (batch, seq, head_dim)
            v_head = v[:, head_idx]
            q_head = q[:, head_idx]

            # Process sequence
            outputs = []
            for t in range(seq_len):
                k_t = k_head[:, t]  # (batch, head_dim)
                v_t = v_head[:, t]
                q_t = q_head[:, t]

                # Retrieve from memory using query
                output_t = memory.forwardWithParams(q_t, state.params)
                outputs.append(output_t)

                # Compute gradients for key-value association
                grads = self.computeParamGradients(memory, state.params, k_t, v_t)

                # Update memory
                if self.use_delta_rule:
                    state = self.updateMemoryDelta(state, grads, k_t, lr, mom, forget)
                else:
                    state = self.updateMemoryGradient(state, grads, lr, mom)

            # Stack outputs for this head: (batch, seq, head_dim)
            head_output = torch.stack(outputs, dim=1)
            head_outputs.append(head_output)
            new_states.append(state)

        # Combine heads: (batch, seq, num_heads, head_dim)
        output = torch.stack(head_outputs, dim=2)
        output = output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.w_o(output)

        # Apply output gate
        gate = self.output_gate(x)
        output = gate * output

        output = self.dropout(output)

        return output, new_states


class FastNeuralMemory(nn.Module):
    """
    Fast Neural Memory using linear attention approximation.

    For efficiency during training, approximates the deep memory
    with a linear outer-product memory that can be parallelized.
    This version properly implements the delta rule update.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 1,
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
        self.base_learning_rate = learning_rate
        self.base_momentum = momentum
        self.use_delta_rule = use_delta_rule
        self.eps = eps

        # Projections
        self.w_k = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_v = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * head_dim, dim, bias=False)

        # Normalization
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.norm_q = nn.LayerNorm(head_dim)

        # Learnable parameters
        self.lr_scale = nn.Parameter(torch.zeros(num_heads))
        self.momentum_scale = nn.Parameter(torch.zeros(num_heads))
        self.beta = nn.Parameter(torch.zeros(num_heads))  # Forgetting factor

        # Output gate
        self.output_gate = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        momentum_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with fast linear memory.

        Args:
            x: Input (batch, seq, dim)
            memory: Previous memory state (batch, heads, head_dim, head_dim)
            momentum_buffer: Previous momentum (batch, heads, head_dim, head_dim)

        Returns:
            output: (batch, seq, dim)
            memory: Updated memory state
            momentum_buffer: Updated momentum
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory and momentum
        if memory is None:
            memory = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype,
            )
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(memory)

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

        # Get per-head parameters
        lr = (torch.sigmoid(self.lr_scale) * self.base_learning_rate * 2).view(
            1, self.num_heads, 1, 1
        )
        mom = (torch.sigmoid(self.momentum_scale) * self.base_momentum * 2).view(
            1, self.num_heads, 1, 1
        )
        beta = (torch.sigmoid(self.beta) * 0.5).view(1, self.num_heads, 1, 1)

        outputs = []

        for t in range(seq_len):
            k_t = k[:, :, t:t+1, :]  # (batch, heads, 1, head_dim)
            v_t = v[:, :, t:t+1, :]
            q_t = q[:, :, t:t+1, :]

            # Retrieve: output = M @ q
            output = torch.einsum("bhdk,bhsk->bhsd", memory, q_t)
            outputs.append(output)

            # Delta rule update (Eq. 28-29 from Nested Learning paper):
            # M_{t+1} = M_t - M_t * k_t * k_t^T - eta * (M_t * k_t - v_t) * k_t^T

            # Normalize key for stable updates (required for delta rule)
            k_t_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + self.eps)

            # Compute prediction: M @ k
            predicted = torch.einsum("bhdk,bhsk->bhsd", memory, k_t_norm)

            # Compute surprise (prediction error): M*k - v
            surprise = predicted - v_t  # (batch, heads, 1, head_dim)

            if self.use_delta_rule:
                # Forgetting term: (M @ k) @ k^T (coefficient = 1 per paper)
                forget_term = torch.einsum("bhsd,bhsk->bhdk", predicted, k_t_norm)

                # Learning term: surprise @ k^T
                learn_term = torch.einsum("bhsd,bhsk->bhdk", surprise, k_t_norm)

                # Update momentum with learning term
                momentum_buffer = mom * momentum_buffer + learn_term

                # Delta rule: M = M - forget_term - lr * momentum
                memory = memory - forget_term - lr * momentum_buffer
            else:
                # Standard momentum update (Hebbian)
                update = torch.einsum("bhsv,bhsk->bhvk", v_t, k_t)
                momentum_buffer = mom * momentum_buffer + update
                memory = memory + lr * momentum_buffer

        # Concatenate outputs
        output = torch.cat(outputs, dim=2)  # (batch, heads, seq, head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.w_o(output)

        # Gate
        gate = torch.sigmoid(self.output_gate(x))
        output = gate * output

        output = self.dropout(output)

        return output, memory, momentum_buffer


class ChunkedNeuralMemory(nn.Module):
    """
    Chunked Neural Memory for efficient parallel processing.

    Processes sequence in chunks, with parallel attention within chunks
    and sequential memory updates between chunks.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        num_heads: int = 1,
        chunk_size: int = 16,
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
        self.chunk_size = chunk_size
        self.use_delta_rule = use_delta_rule

        # Use FastNeuralMemory as base
        self.memory_module = FastNeuralMemory(
            dim=dim,
            head_dim=head_dim,
            num_heads=num_heads,
            learning_rate=learning_rate,
            momentum=momentum,
            use_delta_rule=use_delta_rule,
            dropout=dropout,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        momentum_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with chunked processing.
        """
        batch_size, seq_len, _ = x.shape

        # Initialize
        if memory is None:
            memory = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=x.dtype,
            )
        if momentum_buffer is None:
            momentum_buffer = torch.zeros_like(memory)

        # Process in chunks
        outputs = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end, :]

            chunk_output, memory, momentum_buffer = self.memory_module(
                chunk, memory, momentum_buffer
            )
            outputs.append(chunk_output)

        output = torch.cat(outputs, dim=1)
        return output, memory, momentum_buffer
