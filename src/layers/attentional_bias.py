"""
Attentional Bias modules for MIRAS framework.

Implements various internal memory objectives (attentional bias) that determine
how the memory learns to associate keys with values.

Reference: MIRAS paper (2504.13173), Section 3.2 "Attentional Bias"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import math


class AttentionalBias(ABC, nn.Module):
    """
    Abstract base class for attentional bias objectives.

    Attentional bias defines the internal memory objective:
        L(M; k, v) = loss(M @ k, v) + regularization

    The gradient of this objective drives memory updates.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @abstractmethod
    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss between prediction and target.

        Args:
            predicted: M @ k (batch, dim_value) or (batch, heads, dim_value)
            target: v (batch, dim_value) or (batch, heads, dim_value)

        Returns:
            Scalar or per-sample loss
        """
        pass

    @abstractmethod
    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of loss w.r.t. predicted value.

        Args:
            predicted: M @ k
            target: v

        Returns:
            Gradient d_loss / d_predicted
        """
        pass

    def computeMemoryGradient(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient of loss w.r.t. memory matrix.

        For L(M; k, v), the gradient is:
            dL/dM = dL/d(Mk) @ k^T

        Args:
            memory: Memory matrix (batch, dim_value, dim_key) or (batch, heads, dv, dk)
            key: Key vector (batch, dim_key) or (batch, heads, dim_key)
            value: Value vector (batch, dim_value) or (batch, heads, dim_value)

        Returns:
            grad_memory: Gradient w.r.t. memory
            surprise: The prediction error (for monitoring)
        """
        # Handle different tensor shapes
        if memory.dim() == 3:
            # (batch, dim_value, dim_key)
            predicted = torch.einsum("bvk,bk->bv", memory, key)
            grad_pred = self.computeGradient(predicted, value)
            grad_memory = torch.einsum("bv,bk->bvk", grad_pred, key)
        else:
            # (batch, heads, dim_value, dim_key)
            predicted = torch.einsum("bhvk,bhk->bhv", memory, key)
            grad_pred = self.computeGradient(predicted, value)
            grad_memory = torch.einsum("bhv,bhk->bhvk", grad_pred, key)

        surprise = predicted - value
        return grad_memory, surprise


class L2AttentionalBias(AttentionalBias):
    """
    L2 (Mean Squared Error) attentional bias.

    Loss: L(M; k, v) = 0.5 * ||M @ k - v||_2^2

    This is the standard delta rule objective, equivalent to linear regression.
    Gradient is linear in the error, making updates proportional to surprise.

    Reference: MIRAS Table 1, "L2 regression"
    """

    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """L2 loss: 0.5 * ||predicted - target||_2^2"""
        diff = predicted - target
        return 0.5 * (diff ** 2).sum(dim=-1)

    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient: predicted - target"""
        return predicted - target


class LpAttentionalBias(AttentionalBias):
    """
    Lp-norm attentional bias for p in (1, 2).

    Loss: L(M; k, v) = (1/p) * ||M @ k - v||_p^p

    For p < 2, this is more robust to outliers in value space compared to L2.
    The gradient magnitude is bounded for large errors when p < 2.

    Gradient: sign(Mk - v) * |Mk - v|^{p-1}

    Reference: MIRAS Table 1, "Lp regression" for Moneta model
    """

    def __init__(self, p: float = 1.5, eps: float = 1e-6):
        """
        Args:
            p: Norm power, must be in (1, 2]. p=2 reduces to L2, p->1 approaches L1.
            eps: Small constant for numerical stability
        """
        super().__init__(eps)
        if not 1.0 < p <= 2.0:
            raise ValueError(f"p must be in (1, 2], got {p}")
        self.p = p

    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Lp loss: (1/p) * ||predicted - target||_p^p"""
        diff = predicted - target
        return (1.0 / self.p) * (torch.abs(diff) ** self.p).sum(dim=-1)

    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient: sign(diff) * |diff|^{p-1}

        For p < 2, gradient magnitude is sublinear in error magnitude,
        providing robustness to outliers.
        """
        diff = predicted - target
        abs_diff = torch.abs(diff) + self.eps
        # Gradient: sign(diff) * |diff|^{p-1}
        grad = torch.sign(diff) * (abs_diff ** (self.p - 1))
        return grad


class HuberAttentionalBias(AttentionalBias):
    """
    Huber loss attentional bias.

    Loss:
        L_delta(x) = 0.5 * x^2           if |x| <= delta
                   = delta * (|x| - 0.5*delta)  if |x| > delta

    Combines L2 for small errors (quadratic) with L1 for large errors (linear).
    This provides:
    - Smooth optimization near the optimum (quadratic behavior)
    - Robustness to outliers (bounded gradient for large errors)

    Reference: MIRAS Table 1, "Huber Loss" for Yaad model
    """

    def __init__(self, delta: float = 1.0, eps: float = 1e-6):
        """
        Args:
            delta: Threshold between quadratic and linear regions
            eps: Small constant for numerical stability
        """
        super().__init__(eps)
        self.delta = delta

    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss with threshold delta."""
        diff = predicted - target
        abs_diff = torch.abs(diff)

        # Quadratic region: 0.5 * x^2
        quadratic = 0.5 * (diff ** 2)

        # Linear region: delta * (|x| - 0.5 * delta)
        linear = self.delta * (abs_diff - 0.5 * self.delta)

        # Select based on threshold
        loss = torch.where(abs_diff <= self.delta, quadratic, linear)
        return loss.sum(dim=-1)

    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Huber gradient:
            x           if |x| <= delta
            delta * sign(x)  if |x| > delta

        Gradient is bounded by delta, providing robustness.
        """
        diff = predicted - target
        abs_diff = torch.abs(diff)

        # Quadratic gradient: x
        # Linear gradient: delta * sign(x)
        grad = torch.where(
            abs_diff <= self.delta,
            diff,
            self.delta * torch.sign(diff)
        )
        return grad


class KLAttentionalBias(AttentionalBias):
    """
    KL-divergence attentional bias.

    Loss: L(M; k, v) = KL(softmax(v) || softmax(M @ k))
                     = sum_i p_i * log(p_i / q_i)

    where p = softmax(v) and q = softmax(M @ k).

    Useful when values represent probability distributions or when
    we want relative rather than absolute error matching.

    Reference: MIRAS Table 1, "KL-divergence"
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-6):
        """
        Args:
            temperature: Softmax temperature (lower = sharper distributions)
            eps: Small constant for numerical stability
        """
        super().__init__(eps)
        self.temperature = temperature

    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """KL(softmax(target) || softmax(predicted))"""
        p = F.softmax(target / self.temperature, dim=-1)
        log_q = F.log_softmax(predicted / self.temperature, dim=-1)

        # KL = sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
        # We compute -sum(p * log(q)) as cross-entropy, then subtract entropy
        kl = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
        return kl

    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient of KL w.r.t. predicted logits.

        d KL / d predicted = (1/T) * (softmax(predicted/T) - softmax(target/T))
        """
        p = F.softmax(target / self.temperature, dim=-1)
        q = F.softmax(predicted / self.temperature, dim=-1)

        # Gradient: (q - p) / temperature
        grad = (q - p) / self.temperature
        return grad


class DotProductAttentionalBias(AttentionalBias):
    """
    Dot-product similarity attentional bias (Linear Attention).

    Loss: L(M; k, v) = -<M @ k, v> = -(M @ k)^T @ v

    This corresponds to maximizing similarity between prediction and target.
    Equivalent to the standard linear attention update rule.

    Reference: MIRAS Table 1, "Dot-product similarity"
    """

    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Negative dot product: -<predicted, target>"""
        return -(predicted * target).sum(dim=-1)

    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient: -target (direction to maximize similarity)"""
        return -target


class RobustAttentionalBias(AttentionalBias):
    """
    Robust attentional bias combining multiple loss functions.

    Uses a mixture of L2 and Huber losses with learned weighting,
    adapting to the error distribution during training.

    Loss: L = alpha * L_huber + (1-alpha) * L_l2

    where alpha is learned based on error statistics.
    """

    def __init__(
        self,
        delta: float = 1.0,
        initial_alpha: float = 0.5,
        adaptive: bool = True,
        eps: float = 1e-6,
    ):
        """
        Args:
            delta: Huber threshold
            initial_alpha: Initial mixture weight for Huber loss
            adaptive: Whether to adapt alpha based on error statistics
            eps: Small constant for numerical stability
        """
        super().__init__(eps)
        self.delta = delta
        self.adaptive = adaptive

        # Learnable mixture weight
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5

        if not adaptive:
            self.alpha_logit.requires_grad = False
            self.alpha_logit.fill_(math.log(initial_alpha / (1 - initial_alpha + eps)))

        self.l2_bias = L2AttentionalBias(eps)
        self.huber_bias = HuberAttentionalBias(delta, eps)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def computeLoss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture loss: alpha * Huber + (1-alpha) * L2"""
        l2_loss = self.l2_bias.computeLoss(predicted, target)
        huber_loss = self.huber_bias.computeLoss(predicted, target)
        return self.alpha * huber_loss + (1 - self.alpha) * l2_loss

    def computeGradient(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture gradient"""
        l2_grad = self.l2_bias.computeGradient(predicted, target)
        huber_grad = self.huber_bias.computeGradient(predicted, target)
        return self.alpha * huber_grad + (1 - self.alpha) * l2_grad


def createAttentionalBias(
    bias_type: str,
    **kwargs,
) -> AttentionalBias:
    """
    Factory function to create attentional bias modules.

    Args:
        bias_type: One of 'l2', 'lp', 'huber', 'kl', 'dot_product', 'robust'
        **kwargs: Additional arguments for the specific bias type

    Returns:
        AttentionalBias instance
    """
    bias_map = {
        "l2": L2AttentionalBias,
        "lp": LpAttentionalBias,
        "huber": HuberAttentionalBias,
        "kl": KLAttentionalBias,
        "dot_product": DotProductAttentionalBias,
        "robust": RobustAttentionalBias,
    }

    if bias_type not in bias_map:
        raise ValueError(f"Unknown bias type: {bias_type}. Available: {list(bias_map.keys())}")

    return bias_map[bias_type](**kwargs)
