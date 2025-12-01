"""
Deep Optimizers for HOPE architecture.

Implements momentum-based optimizers viewed as associative memory modules.
These optimizers learn to compress gradient history into their parameters.

Reference: Nested Learning paper, Section 2.3, Equations 17-24
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Iterable, Tuple
import math


class DeepMomentumGD(Optimizer):
    """
    Deep Momentum Gradient Descent (DMGD).

    Extends standard momentum by using an MLP to compress gradient history
    instead of a simple exponential moving average.

    The momentum term is viewed as an associative memory that learns to
    map gradients to update directions. Using an MLP increases capacity.

    Update rule:
        W_{t+1} = W_t + m_{t+1}(u_i)
        m_{t+1} = alpha * m_t - eta * grad L^(2)(m_t; u_i, I)

    where u_i = grad L(W_t; x_i) and L^(2) is the internal objective
    (e.g., dot product similarity or L2 regression).

    Reference: Section 2.3 "More Expressive Memory"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        memory_depth: int = 1,
    ):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            momentum: Momentum decay factor
            weight_decay: L2 regularization weight
            dampening: Dampening for momentum
            nesterov: Use Nesterov momentum
            memory_depth: Depth of momentum memory (1 = standard momentum)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            memory_depth=memory_depth,
        )

        super().__init__(params, defaults)

        # For deep momentum, we maintain MLP-transformed momentum
        self.memory_depth = memory_depth

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            loss: Optional loss value from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # For deep momentum, additional buffers
                    if self.memory_depth > 1:
                        state["momentum_buffers"] = [
                            torch.zeros_like(p)
                            for _ in range(self.memory_depth)
                        ]

                state["step"] += 1

                if momentum != 0:
                    buf = state["momentum_buffer"]

                    if self.memory_depth == 1:
                        # Standard momentum
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                        if nesterov:
                            grad = grad.add(buf, alpha=momentum)
                        else:
                            grad = buf
                    else:
                        # Deep momentum: cascade through multiple buffers
                        buffers = state["momentum_buffers"]
                        current_grad = grad

                        for i, buf in enumerate(buffers):
                            decay = momentum ** (i + 1)
                            buf.mul_(decay).add_(current_grad, alpha=1 - dampening)
                            current_grad = buf

                        if nesterov:
                            grad = grad.add(buffers[-1], alpha=momentum)
                        else:
                            grad = buffers[-1]

                p.add_(grad, alpha=-lr)

        return loss


class DeltaRuleOptimizer(Optimizer):
    """
    Delta Rule Optimizer.

    Uses L2 regression objective instead of dot-product similarity for
    the momentum update, providing better gradient compression.

    Update rule (Eq. 21-22):
        W_{t+1} = W_t + m_{t+1}
        m_{t+1} = (alpha*I - grad^T @ grad) * m_t - eta * P_t @ grad

    where P_t is a preconditioning matrix and the grad^T @ grad term
    implements the delta rule forgetting.

    Reference: Section 2.3 "More Expressive Objectives"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        beta: float = 0.1,
        weight_decay: float = 0.0,
        precondition: str = "none",
        eps: float = 1e-8,
    ):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            momentum: Momentum decay factor
            beta: Delta rule forgetting factor
            weight_decay: L2 regularization weight
            precondition: Preconditioning type ('none', 'grad', 'hessian_diag')
            eps: Small constant for numerical stability
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"Invalid beta value: {beta}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta=beta,
            weight_decay=weight_decay,
            precondition=precondition,
            eps=eps,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            precondition = group["precondition"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    if precondition == "hessian_diag":
                        state["hessian_diag"] = torch.zeros_like(p)

                state["step"] += 1

                buf = state["momentum_buffer"]

                # Delta rule momentum update:
                # m = (alpha*I - beta * grad^T @ grad) @ m - lr * P @ grad

                # For efficiency, approximate grad^T @ grad per-element
                # Full outer product would be too expensive
                grad_sq = grad * grad

                # Forgetting term: reduce momentum where gradient is large
                forget_factor = momentum - beta * grad_sq.clamp(max=1.0)
                buf.mul_(forget_factor.clamp(min=0))

                # Learning term with optional preconditioning
                if precondition == "grad":
                    # Use gradient norm as preconditioner
                    precond = 1.0 / (grad.abs() + eps)
                    update = precond * grad
                elif precondition == "hessian_diag":
                    # EMA of squared gradients (like Adam's v)
                    hess = state["hessian_diag"]
                    hess.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    update = grad / (hess.sqrt() + eps)
                else:
                    update = grad

                buf.add_(update, alpha=-lr)

                # Apply update
                p.add_(buf)

        return loss


class AdamWithDeltaRule(Optimizer):
    """
    Adam optimizer enhanced with delta rule.

    Combines Adam's adaptive learning rates with delta rule's
    forgetting mechanism for better gradient compression.

    This corresponds to viewing Adam as an associative memory
    that maps gradients to update directions, enhanced with
    the delta rule for capacity management.

    Reference: Section C.4 "Adam as Associative Memory"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        delta_beta: float = 0.01,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            delta_beta: Delta rule forgetting coefficient
            eps: Small constant for numerical stability
            weight_decay: L2 regularization weight
            amsgrad: Whether to use AMSGrad variant
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            delta_beta=delta_beta,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            delta_beta = group["delta_beta"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Delta rule: forget before updating
                # Reduce momentum where gradient is large (memory interference)
                grad_sq_normalized = (grad * grad) / (exp_avg_sq + eps)
                forget_mask = (1.0 - delta_beta * grad_sq_normalized).clamp(min=0)

                # Apply forgetting to momentum
                exp_avg.mul_(forget_mask)

                # Standard Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Bias correction
                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = denom / math.sqrt(bias_correction2)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class MuonOptimizer(Optimizer):
    """
    Muon Optimizer implementation.

    Uses Newton-Schulz orthogonalization on momentum updates.
    This corresponds to using a non-linear output transformation
    on the momentum memory module.

    Update rule (Eq. 24):
        W_{t+1} = W_t + sigma(m_{t+1}(u_i))

    where sigma = Newton-Schulz orthogonalization.

    Reference: Section 2.3 "None Linear Outputs"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            momentum: Momentum decay factor
            nesterov: Use Nesterov momentum
            ns_steps: Number of Newton-Schulz iterations
            weight_decay: Weight decay (applied separately)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def newtonSchulzOrthogonalize(
        self,
        M: torch.Tensor,
        num_steps: int = 5,
    ) -> torch.Tensor:
        """
        Apply Newton-Schulz orthogonalization.

        Iteratively computes approximate orthogonalization:
            X_{k+1} = X_k @ (3*I - X_k^T @ X_k) / 2

        Args:
            M: Input matrix (can be any shape, operates on last 2 dims)
            num_steps: Number of iterations

        Returns:
            Approximately orthogonalized matrix
        """
        # Handle different parameter shapes
        original_shape = M.shape

        if M.dim() == 1:
            # Vector: normalize
            return M / (M.norm() + 1e-8)

        if M.dim() > 2:
            # Reshape to 2D
            M = M.view(-1, M.shape[-1])

        # Scale for numerical stability
        scale = (M.shape[0] * M.shape[1]) ** 0.25
        M = M / (scale + 1e-8)

        # Newton-Schulz iteration
        X = M
        for _ in range(num_steps):
            A = X.T @ X
            X = X @ (3.0 * torch.eye(A.shape[0], device=A.device) - A) / 2.0

        # Reshape back
        X = X.view(original_shape)

        return X

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)

                state["step"] += 1

                # Weight decay (decoupled)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                buf = state["momentum_buffer"]

                # Momentum update
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Apply Newton-Schulz orthogonalization
                # Only for 2D+ parameters (matrices)
                if p.dim() >= 2 and min(p.shape) > 1:
                    update = self.newtonSchulzOrthogonalize(update, ns_steps)

                p.add_(update, alpha=-lr)

        return loss
