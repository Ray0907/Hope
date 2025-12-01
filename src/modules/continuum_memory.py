"""
Continuum Memory System (CMS) for HOPE architecture.

Implements a multi-frequency FFN system where different layers
are updated at different rates, creating a hierarchy of memory
timescales from short-term to long-term.

Reference: Nested Learning paper, Section 3, Equations 30-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import math
from dataclasses import dataclass


@dataclass
class CMSState:
    """State container for CMS online learning."""
    step: int = 0
    accumulated_grads: Optional[List[Dict[str, torch.Tensor]]] = None
    accumulated_counts: Optional[List[int]] = None


class FrequencyFFN(nn.Module):
    """
    FFN layer with associated update frequency.

    This FFN is updated only every `chunk_size` steps, creating
    a memory that stores information at a specific timescale.

    Lower frequency (larger chunk_size) = more abstract, long-term knowledge
    Higher frequency (smaller chunk_size) = more specific, short-term knowledge
    """

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        chunk_size: int = 16,
        dropout: float = 0.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        hidden_dim = dim * expansion

        # FFN layers
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Layer norm before FFN
        self.norm = nn.LayerNorm(dim)

        # Momentum buffers for online updates
        self.register_buffer(
            "momentum_fc1_weight",
            torch.zeros_like(self.fc1.weight),
        )
        self.register_buffer(
            "momentum_fc1_bias",
            torch.zeros_like(self.fc1.bias),
        )
        self.register_buffer(
            "momentum_fc2_weight",
            torch.zeros_like(self.fc2.weight),
        )
        self.register_buffer(
            "momentum_fc2_bias",
            torch.zeros_like(self.fc2.bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through FFN.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x

    def forwardWithLocalGrad(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass that also computes local gradients for online update.

        Uses a local reconstruction loss to compute gradients without
        backpropagating through the entire network.

        Args:
            x: Input tensor (batch, seq_len, dim)
            target: Optional target for supervised loss (defaults to reconstruction)

        Returns:
            output: Output tensor
            grads: Dictionary of parameter gradients
        """
        residual = x
        x_norm = self.norm(x)

        # Forward through FFN
        h = self.fc1(x_norm)
        a = self.act(h)
        a_drop = self.dropout(a)
        out = self.fc2(a_drop)
        out_drop = self.dropout(out)

        output = residual + out_drop

        # Compute local loss and gradients
        # Default: reconstruction loss on the residual stream
        if target is None:
            # Use input as target (autoencoder-style)
            target = x_norm

        # Local loss: ||output - target||^2
        loss = 0.5 * ((out - target) ** 2).mean()

        # Compute gradients analytically
        batch_size, seq_len, _ = x.shape

        # d_loss/d_out = (out - target) / (batch_size * seq_len)
        d_out = (out - target) / (batch_size * seq_len)

        # Gradient for fc2: d_loss/d_W2 = d_out^T @ a
        d_out_flat = d_out.view(-1, d_out.shape[-1])  # (B*S, dim)
        a_flat = a_drop.view(-1, a_drop.shape[-1])  # (B*S, hidden)

        grad_fc2_weight = d_out_flat.T @ a_flat  # (dim, hidden)
        grad_fc2_bias = d_out_flat.sum(dim=0)  # (dim,)

        # Backprop through fc2
        d_a = d_out_flat @ self.fc2.weight  # (B*S, hidden)

        # Backprop through GELU
        h_flat = h.view(-1, h.shape[-1])
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        cdf = 0.5 * (1.0 + torch.erf(h_flat / math.sqrt(2.0)))
        pdf = sqrt_2_over_pi * torch.exp(-0.5 * h_flat ** 2)
        gelu_grad = cdf + h_flat * pdf
        d_h = d_a * gelu_grad

        # Gradient for fc1: d_loss/d_W1 = d_h^T @ x
        x_norm_flat = x_norm.view(-1, x_norm.shape[-1])  # (B*S, dim)
        grad_fc1_weight = d_h.T @ x_norm_flat  # (hidden, dim)
        grad_fc1_bias = d_h.sum(dim=0)  # (hidden,)

        grads = {
            "fc1.weight": grad_fc1_weight,
            "fc1.bias": grad_fc1_bias,
            "fc2.weight": grad_fc2_weight,
            "fc2.bias": grad_fc2_bias,
        }

        return output, grads

    def applyOnlineUpdate(
        self,
        grads: Dict[str, torch.Tensor],
        scale: float = 1.0,
    ):
        """
        Apply online parameter update using accumulated gradients.

        Args:
            grads: Dictionary of gradients
            scale: Scale factor for the update (e.g., 1/chunk_size)
        """
        lr = self.learning_rate * scale

        with torch.no_grad():
            # Update momentum and apply
            self.momentum_fc1_weight.mul_(self.momentum).add_(grads["fc1.weight"])
            self.momentum_fc1_bias.mul_(self.momentum).add_(grads["fc1.bias"])
            self.momentum_fc2_weight.mul_(self.momentum).add_(grads["fc2.weight"])
            self.momentum_fc2_bias.mul_(self.momentum).add_(grads["fc2.bias"])

            # Apply updates
            self.fc1.weight.sub_(lr * self.momentum_fc1_weight)
            self.fc1.bias.sub_(lr * self.momentum_fc1_bias)
            self.fc2.weight.sub_(lr * self.momentum_fc2_weight)
            self.fc2.bias.sub_(lr * self.momentum_fc2_bias)


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System (CMS).

    A chain of FFN blocks with different update frequencies, creating
    a spectrum of memory timescales. This generalizes the traditional
    "long-term/short-term memory" into a continuum.

    The output is computed as (Eq. 30):
        y_t = FFN^(f_k)(FFN^(f_{k-1})(...FFN^(f_1)(x_t)))

    Each FFN^(f_l) is updated every C^(l) steps (Eq. 31):
        theta^(f_l)_{i+1} = theta^(f_l)_i - sum_{t} eta^(l)_t * grad
        if i mod C^(l) == 0, else unchanged

    Architecture (from paper Figure 3):
        - Highest frequency FFN: Updated frequently (e.g., every 16 tokens)
        - Lowest frequency FFN: Updated rarely (e.g., every 16M tokens)
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        chunk_sizes: Optional[List[int]] = None,
        expansion: int = 4,
        dropout: float = 0.0,
        learning_rates: Optional[List[float]] = None,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # Default chunk sizes: exponentially increasing
        if chunk_sizes is None:
            chunk_sizes = [16 * (16 ** i) for i in range(num_levels)]
        self.chunk_sizes = chunk_sizes

        # Default learning rates: decreasing for lower frequency
        if learning_rates is None:
            learning_rates = [0.01 / (2 ** i) for i in range(num_levels)]

        assert len(chunk_sizes) == num_levels
        assert len(learning_rates) == num_levels

        # Create FFN layers for each frequency level
        # Index 0 = highest frequency, index -1 = lowest frequency
        self.ffn_layers = nn.ModuleList([
            FrequencyFFN(
                dim=dim,
                expansion=expansion,
                chunk_size=chunk_sizes[i],
                dropout=dropout,
                learning_rate=learning_rates[i],
                momentum=momentum,
            )
            for i in range(num_levels)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)

        # Step counter for tracking updates
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        # Accumulated gradients for each level
        self._accumulated_grads: List[Dict[str, torch.Tensor]] = [
            {} for _ in range(num_levels)
        ]
        self._accumulated_counts: List[int] = [0] * num_levels

    def resetAccumulators(self):
        """Reset gradient accumulators."""
        for i in range(self.num_levels):
            self._accumulated_grads[i] = {}
            self._accumulated_counts[i] = 0

    def forward(
        self,
        x: torch.Tensor,
        enable_online_learning: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through CMS.

        Args:
            x: Input tensor (batch, seq_len, dim)
            enable_online_learning: If True, compute and accumulate local gradients

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        current = x

        if enable_online_learning:
            # Forward with gradient computation and accumulation
            for level_idx, ffn in enumerate(self.ffn_layers):
                current, grads = ffn.forwardWithLocalGrad(current)

                # Accumulate gradients
                for key, grad in grads.items():
                    if key not in self._accumulated_grads[level_idx]:
                        self._accumulated_grads[level_idx][key] = grad.clone()
                    else:
                        self._accumulated_grads[level_idx][key] += grad

                self._accumulated_counts[level_idx] += 1

                # Check if this level should be updated
                chunk_size = self.chunk_sizes[level_idx]
                if self._accumulated_counts[level_idx] >= chunk_size:
                    # Apply update
                    scale = 1.0 / chunk_size
                    ffn.applyOnlineUpdate(
                        self._accumulated_grads[level_idx],
                        scale=scale,
                    )
                    # Reset accumulator for this level
                    self._accumulated_grads[level_idx] = {}
                    self._accumulated_counts[level_idx] = 0
        else:
            # Standard forward pass
            for ffn in self.ffn_layers:
                current = ffn(current)

        output = self.final_norm(current)
        self.global_step += 1

        return output

    def forwardWithExplicitStep(
        self,
        x: torch.Tensor,
        step: int,
        accumulated_grads: Optional[List[Dict[str, torch.Tensor]]] = None,
        accumulated_counts: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], List[int]]:
        """
        Forward pass with explicit step tracking for external state management.

        Useful for distributed training or when state needs to be preserved
        across function calls.

        Args:
            x: Input tensor
            step: Current global step
            accumulated_grads: External gradient accumulators
            accumulated_counts: External accumulation counts

        Returns:
            output: Output tensor
            accumulated_grads: Updated gradient accumulators
            accumulated_counts: Updated counts
        """
        if accumulated_grads is None:
            accumulated_grads = [{} for _ in range(self.num_levels)]
        if accumulated_counts is None:
            accumulated_counts = [0] * self.num_levels

        current = x

        for level_idx, ffn in enumerate(self.ffn_layers):
            current, grads = ffn.forwardWithLocalGrad(current)

            # Accumulate
            for key, grad in grads.items():
                if key not in accumulated_grads[level_idx]:
                    accumulated_grads[level_idx][key] = grad.clone()
                else:
                    accumulated_grads[level_idx][key] += grad
            accumulated_counts[level_idx] += 1

            # Check for update
            chunk_size = self.chunk_sizes[level_idx]
            if accumulated_counts[level_idx] >= chunk_size:
                scale = 1.0 / chunk_size
                ffn.applyOnlineUpdate(accumulated_grads[level_idx], scale=scale)
                accumulated_grads[level_idx] = {}
                accumulated_counts[level_idx] = 0

        output = self.final_norm(current)

        return output, accumulated_grads, accumulated_counts

    def getLevelUpdateStatus(self, step: int) -> List[bool]:
        """
        Get which levels will be updated at a given step.

        Args:
            step: Global step number

        Returns:
            List of booleans indicating update status for each level
        """
        return [
            (step + 1) % chunk_size == 0
            for chunk_size in self.chunk_sizes
        ]

    def getUpdateSchedule(self, num_steps: int) -> Dict[int, List[int]]:
        """
        Get schedule of which levels are updated at each step.

        Args:
            num_steps: Number of steps to schedule

        Returns:
            Dict mapping step number to list of level indices to update
        """
        schedule = {}
        for step in range(num_steps):
            levels_to_update = []
            for level_idx, chunk_size in enumerate(self.chunk_sizes):
                if (step + 1) % chunk_size == 0:
                    levels_to_update.append(level_idx)
            if levels_to_update:
                schedule[step] = levels_to_update
        return schedule


class AdaptiveCMS(ContinuumMemorySystem):
    """
    Adaptive CMS with data-dependent frequency selection.

    Instead of fixed chunk sizes, learns to select which levels
    to update based on the input data characteristics (surprise).
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        chunk_sizes: Optional[List[int]] = None,
        expansion: int = 4,
        dropout: float = 0.0,
        learning_rates: Optional[List[float]] = None,
        surprise_threshold: float = 0.5,
    ):
        super().__init__(
            dim, num_levels, chunk_sizes, expansion, dropout, learning_rates
        )

        self.surprise_threshold = surprise_threshold

        # Surprise estimator: predicts whether to update each level
        self.surprise_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_levels),
            nn.Sigmoid(),
        )

        # Running statistics for surprise normalization
        self.register_buffer("surprise_mean", torch.zeros(num_levels))
        self.register_buffer("surprise_var", torch.ones(num_levels))
        self.register_buffer("surprise_count", torch.tensor(0.0))

    def computeSurprise(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute surprise signal based on prediction error.

        Args:
            x: Input tensor
            output: Output from FFN

        Returns:
            Surprise scores for each level (batch, num_levels)
        """
        # Use pooled representation
        x_pooled = x.mean(dim=1)  # (batch, dim)
        surprise_scores = self.surprise_estimator(x_pooled)
        return surprise_scores

    def forward(
        self,
        x: torch.Tensor,
        enable_online_learning: bool = False,
    ) -> torch.Tensor:
        """
        Forward with adaptive level selection based on surprise.
        """
        batch_size, seq_len, _ = x.shape
        current = x

        # Compute surprise scores
        surprise_scores = self.computeSurprise(x, x)  # Initial estimate

        for level_idx, ffn in enumerate(self.ffn_layers):
            if enable_online_learning:
                current, grads = ffn.forwardWithLocalGrad(current)

                # Adaptive update: only update if surprise exceeds threshold
                level_surprise = surprise_scores[:, level_idx].mean().item()

                # Update running statistics
                self.surprise_count += 1
                delta = level_surprise - self.surprise_mean[level_idx]
                self.surprise_mean[level_idx] += delta / self.surprise_count
                delta2 = level_surprise - self.surprise_mean[level_idx]
                self.surprise_var[level_idx] += delta * delta2

                # Normalize surprise
                std = (self.surprise_var[level_idx] / max(1, self.surprise_count - 1)).sqrt()
                normalized_surprise = (level_surprise - self.surprise_mean[level_idx]) / (std + 1e-6)

                # Accumulate gradients
                for key, grad in grads.items():
                    if key not in self._accumulated_grads[level_idx]:
                        self._accumulated_grads[level_idx][key] = grad.clone()
                    else:
                        self._accumulated_grads[level_idx][key] += grad
                self._accumulated_counts[level_idx] += 1

                # Update if surprise is high OR chunk is full
                should_update_surprise = normalized_surprise > self.surprise_threshold
                should_update_chunk = self._accumulated_counts[level_idx] >= self.chunk_sizes[level_idx]

                if should_update_surprise or should_update_chunk:
                    count = max(1, self._accumulated_counts[level_idx])
                    scale = 1.0 / count
                    ffn.applyOnlineUpdate(self._accumulated_grads[level_idx], scale=scale)
                    self._accumulated_grads[level_idx] = {}
                    self._accumulated_counts[level_idx] = 0
            else:
                current = ffn(current)

        output = self.final_norm(current)
        self.global_step += 1

        return output


class ParallelCMS(nn.Module):
    """
    Parallel Continuum Memory System.

    Instead of sequential processing through levels, processes
    all levels in parallel and combines outputs. This is more
    efficient but loses the hierarchical dependency.
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        chunk_sizes: Optional[List[int]] = None,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        if chunk_sizes is None:
            chunk_sizes = [16 * (16 ** i) for i in range(num_levels)]
        self.chunk_sizes = chunk_sizes

        # Parallel FFN layers
        self.ffn_layers = nn.ModuleList([
            FrequencyFFN(
                dim=dim,
                expansion=expansion,
                chunk_size=chunk_sizes[i],
                dropout=dropout,
            )
            for i in range(num_levels)
        ])

        # Learnable weights for combining level outputs
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

        # Combining projection
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with parallel processing and weighted combination."""
        # Process all levels in parallel
        level_outputs = [ffn(x) for ffn in self.ffn_layers]

        # Weighted combination
        weights = F.softmax(self.level_weights, dim=0)
        output = sum(w * out for w, out in zip(weights, level_outputs))

        output = self.norm(output)

        return output
