"""
Retention Gate modules for MIRAS framework.

Implements various retention mechanisms that control how memory retains
its past state during updates. These act as regularization on memory changes.

The general memory update with retention gate is:
    M_{t+1} = argmin_M [ L_attn(M; k, v) + lambda * R(M, M_t) ]

where R(M, M_t) is the retention penalty encouraging M to stay close to M_t.

Reference: MIRAS paper (2504.13173), Section 3.3 "Retention Gate"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import math


class RetentionGate(ABC, nn.Module):
    """
    Abstract base class for retention gates.

    Retention gates define how memory should retain its past state:
        R(M_{t+1}, M_t) = penalty for changing memory

    The gradient of this penalty is subtracted from the memory update.
    """

    def __init__(self, strength: float = 0.1, eps: float = 1e-6):
        """
        Args:
            strength: Retention strength (lambda in the formula)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.strength = strength
        self.eps = eps

    @abstractmethod
    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the retention penalty R(M_new, M_old).

        Args:
            memory_new: Proposed new memory state
            memory_old: Previous memory state

        Returns:
            Scalar or per-element penalty
        """
        pass

    @abstractmethod
    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of retention penalty w.r.t. memory_new.

        Args:
            memory_new: Proposed new memory state
            memory_old: Previous memory state

        Returns:
            Gradient dR/dM_new
        """
        pass

    def applyRetention(
        self,
        memory: torch.Tensor,
        update: torch.Tensor,
        learning_rate: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply retention-aware memory update.

        The update is modified to account for retention penalty:
            M_{t+1} = M_t - lr * (grad_attn + strength * grad_retention)

        Args:
            memory: Current memory state M_t
            update: Proposed update from attentional bias gradient
            learning_rate: Learning rate for the update

        Returns:
            Updated memory with retention applied
        """
        # Proposed new memory without retention
        memory_proposed = memory - learning_rate * update

        # Compute retention gradient at proposed state
        retention_grad = self.computeGradient(memory_proposed, memory)

        # Apply retention correction
        memory_new = memory_proposed - learning_rate * self.strength * retention_grad

        return memory_new


class L2RetentionGate(RetentionGate):
    """
    L2 retention gate (weight decay / Tikhonov regularization).

    Penalty: R(M_new, M_old) = 0.5 * ||M_new - M_old||_2^2

    This encourages small changes to memory, providing stability.
    Equivalent to exponential moving average decay.

    Gradient: M_new - M_old

    Reference: MIRAS Table 1, standard "L2" retention
    """

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """L2 penalty: 0.5 * ||M_new - M_old||_2^2"""
        diff = memory_new - memory_old
        return 0.5 * (diff ** 2).sum()

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient: M_new - M_old"""
        return memory_new - memory_old


class LqRetentionGate(RetentionGate):
    """
    Lq-norm retention gate for q in (1, 2).

    Penalty: R(M_new, M_old) = (1/q) * ||M_new - M_old||_q^q

    For q < 2, this is more permissive of large changes (compared to L2)
    while still penalizing many small changes. This provides:
    - Stability for gradual drift (many small changes penalized)
    - Flexibility for large updates when needed (sublinear penalty growth)

    Gradient: sign(M_new - M_old) * |M_new - M_old|^{q-1}

    Reference: MIRAS Table 1, "Lq stability" for Moneta model
    """

    def __init__(self, q: float = 1.5, strength: float = 0.1, eps: float = 1e-6):
        """
        Args:
            q: Norm power, must be in (1, 2]. q=2 reduces to L2, q->1 approaches L1.
            strength: Retention strength
            eps: Small constant for numerical stability
        """
        super().__init__(strength, eps)
        if not 1.0 < q <= 2.0:
            raise ValueError(f"q must be in (1, 2], got {q}")
        self.q = q

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Lq penalty: (1/q) * ||M_new - M_old||_q^q"""
        diff = memory_new - memory_old
        return (1.0 / self.q) * (torch.abs(diff) ** self.q).sum()

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient: sign(diff) * |diff|^{q-1}"""
        diff = memory_new - memory_old
        abs_diff = torch.abs(diff) + self.eps
        grad = torch.sign(diff) * (abs_diff ** (self.q - 1))
        return grad


class KLRetentionGate(RetentionGate):
    """
    KL-divergence retention gate.

    Penalty: R(M_new, M_old) = KL(M_new || M_old)
                              = sum(M_new * log(M_new / M_old))

    For positive memory values (e.g., attention weights), this penalizes
    changes that significantly alter the "distribution" of memory values.

    For general matrices, we apply softmax normalization or use a
    modified version based on relative entropy.

    Key property: Provides "soft thresholding" effect - small values in
    M_old become hard to increase (log penalty), while large values are
    easier to modify.

    Reference: MIRAS Table 1, "KL-divergence" for Memora model
    """

    def __init__(
        self,
        strength: float = 0.1,
        normalize: bool = True,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            strength: Retention strength
            normalize: Whether to apply softmax normalization
            temperature: Temperature for softmax normalization
            eps: Small constant for numerical stability
        """
        super().__init__(strength, eps)
        self.normalize = normalize
        self.temperature = temperature

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence penalty with optional normalization."""
        if self.normalize:
            # Flatten last dimensions and apply softmax
            shape = memory_new.shape
            m_new_flat = memory_new.view(*shape[:-2], -1) / self.temperature
            m_old_flat = memory_old.view(*shape[:-2], -1) / self.temperature

            p = F.softmax(m_new_flat, dim=-1)
            log_q = F.log_softmax(m_old_flat, dim=-1)

            kl = F.kl_div(log_q, p, reduction="sum")
        else:
            # Direct KL (requires positive values)
            m_new_pos = torch.abs(memory_new) + self.eps
            m_old_pos = torch.abs(memory_old) + self.eps

            # Normalize to sum to 1
            m_new_norm = m_new_pos / m_new_pos.sum()
            m_old_norm = m_old_pos / m_old_pos.sum()

            kl = (m_new_norm * (torch.log(m_new_norm) - torch.log(m_old_norm))).sum()

        return kl

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient of KL divergence.

        For normalized version: (softmax(M_new) - softmax(M_old)) / T
        """
        if self.normalize:
            shape = memory_new.shape
            m_new_flat = memory_new.view(*shape[:-2], -1) / self.temperature
            m_old_flat = memory_old.view(*shape[:-2], -1) / self.temperature

            p = F.softmax(m_new_flat, dim=-1)
            q = F.softmax(m_old_flat, dim=-1)

            # Gradient of KL(p||q) w.r.t. logits of p
            # = (1/T) * p * (1 + log(p) - log(q) - sum(p * (1 + log(p) - log(q))))
            # Simplified: (1/T) * (p - q) for small changes
            grad_flat = (p - q) / self.temperature
            grad = grad_flat.view(*shape)
        else:
            # Direct gradient
            m_new_pos = torch.abs(memory_new) + self.eps
            m_old_pos = torch.abs(memory_old) + self.eps

            total = m_new_pos.sum()
            m_new_norm = m_new_pos / total

            # d/dM_new of KL = (1/total) * (1 + log(M_new_norm) - log(M_old_norm))
            grad = (1.0 / total) * (1 + torch.log(m_new_norm) - torch.log(m_old_pos / m_old_pos.sum()))
            grad = grad * torch.sign(memory_new)  # Chain rule for abs

        return grad


class ElasticNetRetentionGate(RetentionGate):
    """
    Elastic Net retention gate.

    Penalty: R(M_new, M_old) = alpha * ||M_new||_1 + (1-alpha) * ||M_new||_2^2

    Combines L1 sparsity with L2 smoothness:
    - L1 component encourages sparse memory (many zero entries)
    - L2 component provides stability and smooth optimization

    Note: This regularizes the memory state itself, not just changes.
    For change-based regularization, see ElasticNetChangeRetention.

    Reference: MIRAS Table 1, "Elastic Net"
    """

    def __init__(
        self,
        alpha: float = 0.5,
        strength: float = 0.1,
        eps: float = 1e-6,
    ):
        """
        Args:
            alpha: L1 vs L2 balance (0 = pure L2, 1 = pure L1)
            strength: Overall retention strength
            eps: Small constant for numerical stability
        """
        super().__init__(strength, eps)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Elastic net: alpha * ||M||_1 + (1-alpha) * ||M||_2^2"""
        l1 = torch.abs(memory_new).sum()
        l2 = (memory_new ** 2).sum()
        return self.alpha * l1 + (1 - self.alpha) * l2

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient: alpha * sign(M) + 2*(1-alpha) * M"""
        l1_grad = torch.sign(memory_new)
        l2_grad = 2 * memory_new
        return self.alpha * l1_grad + (1 - self.alpha) * l2_grad


class BregmanRetentionGate(RetentionGate):
    """
    Bregman divergence retention gate.

    Penalty: D_psi(M_new, M_old) = psi(M_new) - psi(M_old) - <grad_psi(M_old), M_new - M_old>

    where psi is a strictly convex function. This generalizes many divergences:
    - psi(x) = ||x||_2^2 / 2 -> L2 distance
    - psi(x) = sum(x * log(x)) -> KL divergence (for positive x)
    - psi(x) = -sum(log(x)) -> Itakura-Saito divergence

    Reference: MIRAS Table 1, "Bregman divergence"
    """

    def __init__(
        self,
        psi_type: str = "l2",
        strength: float = 0.1,
        eps: float = 1e-6,
    ):
        """
        Args:
            psi_type: Type of generating function ('l2', 'entropy', 'log_barrier')
            strength: Retention strength
            eps: Small constant for numerical stability
        """
        super().__init__(strength, eps)
        self.psi_type = psi_type

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """Compute psi(x) based on selected type."""
        if self.psi_type == "l2":
            return 0.5 * (x ** 2).sum()
        elif self.psi_type == "entropy":
            # Negative entropy: sum(x * log(x)) for positive x
            x_pos = torch.abs(x) + self.eps
            return (x_pos * torch.log(x_pos)).sum()
        elif self.psi_type == "log_barrier":
            # Log barrier: -sum(log(x)) for positive x
            x_pos = torch.abs(x) + self.eps
            return -torch.log(x_pos).sum()
        else:
            raise ValueError(f"Unknown psi_type: {self.psi_type}")

    def gradPsi(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient of psi at x."""
        if self.psi_type == "l2":
            return x
        elif self.psi_type == "entropy":
            x_pos = torch.abs(x) + self.eps
            return (1 + torch.log(x_pos)) * torch.sign(x)
        elif self.psi_type == "log_barrier":
            x_pos = torch.abs(x) + self.eps
            return -torch.sign(x) / x_pos
        else:
            raise ValueError(f"Unknown psi_type: {self.psi_type}")

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Bregman divergence: psi(new) - psi(old) - <grad_psi(old), new - old>"""
        psi_new = self.psi(memory_new)
        psi_old = self.psi(memory_old)
        grad_old = self.gradPsi(memory_old)
        inner_prod = (grad_old * (memory_new - memory_old)).sum()
        return psi_new - psi_old - inner_prod

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient of Bregman divergence: grad_psi(new) - grad_psi(old)"""
        return self.gradPsi(memory_new) - self.gradPsi(memory_old)


class DeltaRuleRetentionGate(RetentionGate):
    """
    Delta rule style retention (forgetting mechanism from Nested Learning paper).

    Instead of a penalty-based retention, this implements the explicit
    forgetting term from Eq. 28-29:

        M_{t+1} = M_t - (M_t @ k) @ k^T - lr * gradient

    The first term (M_t @ k) @ k^T projects out the old association for key k,
    making room for the new association.

    This is not a penalty-based retention but rather an explicit forgetting
    mechanism integrated into the update rule.

    Reference: Nested Learning paper Eq. 28-29, Titans paper
    """

    def __init__(
        self,
        forget_strength: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        Args:
            forget_strength: Strength of forgetting (1.0 = full projection out)
            eps: Small constant for numerical stability
        """
        super().__init__(forget_strength, eps)
        self.forget_strength = forget_strength

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Not used for delta rule retention."""
        return torch.tensor(0.0, device=memory_new.device)

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Not used for delta rule retention."""
        return torch.zeros_like(memory_new)

    def computeForgetTerm(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the forgetting term (M @ k) @ k^T.

        This projects out the current association for key k from memory.

        Args:
            memory: Current memory (batch, dim_v, dim_k) or (batch, heads, dv, dk)
            key: Key to forget (batch, dim_k) or (batch, heads, dim_k)

        Returns:
            Forgetting term to subtract from memory
        """
        # Normalize key for stable updates
        key_norm = key / (key.norm(dim=-1, keepdim=True) + self.eps)

        if memory.dim() == 3:
            # (batch, dim_v, dim_k)
            predicted = torch.einsum("bvk,bk->bv", memory, key_norm)
            forget_term = torch.einsum("bv,bk->bvk", predicted, key_norm)
        else:
            # (batch, heads, dim_v, dim_k)
            predicted = torch.einsum("bhvk,bhk->bhv", memory, key_norm)
            forget_term = torch.einsum("bhv,bhk->bhvk", predicted, key_norm)

        return self.forget_strength * forget_term


class AdaptiveRetentionGate(RetentionGate):
    """
    Adaptive retention gate that adjusts strength based on context.

    Learns to modulate retention strength based on:
    - Surprise magnitude (high surprise = allow more change)
    - Memory utilization (full memory = allow more forgetting)

    This provides a learned balance between stability and plasticity.
    """

    def __init__(
        self,
        dim: int,
        base_strength: float = 0.1,
        eps: float = 1e-6,
    ):
        """
        Args:
            dim: Dimension for computing adaptive strength
            base_strength: Base retention strength
            eps: Small constant for numerical stability
        """
        super().__init__(base_strength, eps)
        self.dim = dim

        # Network to predict retention strength modifier
        self.strength_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        # Base retention uses L2
        self.base_retention = L2RetentionGate(base_strength, eps)

    def computeAdaptiveStrength(
        self,
        surprise: torch.Tensor,
        memory_utilization: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive retention strength.

        Args:
            surprise: Surprise signal (prediction error)
            memory_utilization: How full the memory is

        Returns:
            Modulated retention strength
        """
        # Concatenate inputs
        context = torch.cat([
            surprise.flatten(-2),
            memory_utilization.flatten(-2),
        ], dim=-1)

        # Predict modifier (0 to 1)
        modifier = self.strength_net(context)

        # Scale base strength: high surprise -> lower retention (more change allowed)
        # modifier close to 0 = low retention, modifier close to 1 = high retention
        return self.strength * (0.5 + 0.5 * modifier)

    def computePenalty(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2 penalty (adaptive strength applied in applyRetention)."""
        return self.base_retention.computePenalty(memory_new, memory_old)

    def computeGradient(
        self,
        memory_new: torch.Tensor,
        memory_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2 gradient (adaptive strength applied in applyRetention)."""
        return self.base_retention.computeGradient(memory_new, memory_old)


def createRetentionGate(
    gate_type: str,
    **kwargs,
) -> RetentionGate:
    """
    Factory function to create retention gate modules.

    Args:
        gate_type: One of 'l2', 'lq', 'kl', 'elastic_net', 'bregman', 'delta_rule'
        **kwargs: Additional arguments for the specific gate type

    Returns:
        RetentionGate instance
    """
    gate_map = {
        "l2": L2RetentionGate,
        "lq": LqRetentionGate,
        "kl": KLRetentionGate,
        "elastic_net": ElasticNetRetentionGate,
        "bregman": BregmanRetentionGate,
        "delta_rule": DeltaRuleRetentionGate,
        "adaptive": AdaptiveRetentionGate,
    }

    if gate_type not in gate_map:
        raise ValueError(f"Unknown gate type: {gate_type}. Available: {list(gate_map.keys())}")

    return gate_map[gate_type](**kwargs)
