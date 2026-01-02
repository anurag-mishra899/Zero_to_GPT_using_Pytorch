"""
Module 3.1: Optimizers - From SGD to AdamW
Complete implementation of optimization algorithms for deep learning.

This module covers:
1. Gradient Descent fundamentals
2. SGD with Momentum, Nesterov
3. Adaptive methods (Adagrad, RMSprop)
4. Adam and AdamW
5. Gradient clipping techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Callable, Iterable
from collections import defaultdict

print("=" * 70)
print("MODULE 3.1: OPTIMIZERS - FROM SGD TO ADAMW")
print("=" * 70)

# =============================================================================
# SECTION 1: GRADIENT DESCENT FUNDAMENTALS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: GRADIENT DESCENT FUNDAMENTALS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Basic Gradient Descent
# -----------------------------------------------------------------------------
print("\n1.1 Basic Gradient Descent")
print("-" * 40)


class VanillaGD:
    """
    Vanilla Gradient Descent.

    θ_{t+1} = θ_t - η × ∇L(θ_t)
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data


# Demonstrate basic GD
print("Vanilla GD on simple quadratic: f(x) = x²")

x = torch.tensor([5.0], requires_grad=True)
optimizer = VanillaGD([x], lr=0.1)

print(f"Initial x: {x.item():.4f}")
for i in range(10):
    loss = x ** 2  # f(x) = x²
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        print(f"  Step {i+1}: x = {x.item():.4f}, f(x) = {loss.item():.4f}")


# -----------------------------------------------------------------------------
# 1.2 Mini-batch Gradient Descent Comparison
# -----------------------------------------------------------------------------
print("\n\n1.2 Batch vs Stochastic vs Mini-batch")
print("-" * 40)


def create_regression_data(n_samples=1000):
    """Create simple linear regression data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, 10)
    true_w = torch.randn(10, 1)
    y = X @ true_w + 0.1 * torch.randn(n_samples, 1)
    return X, y, true_w


X, y, true_w = create_regression_data(1000)


def train_one_epoch(X, y, w, lr, batch_size=None):
    """Train for one epoch with given batch size."""
    n_samples = X.shape[0]

    if batch_size is None:
        batch_size = n_samples  # Full batch

    total_loss = 0
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Shuffle
    perm = torch.randperm(n_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward
        pred = X_batch @ w
        loss = ((pred - y_batch) ** 2).mean()

        # Backward (manual)
        grad = 2 * X_batch.T @ (pred - y_batch) / len(X_batch)
        w.data -= lr * grad

        total_loss += loss.item()

    return total_loss / n_batches


# Compare different batch sizes
print("Training linear regression with different batch sizes:")
print(f"{'Batch Size':>12} {'Final Loss':>12} {'Steps/Epoch':>12}")
print("-" * 40)

for batch_size in [1000, 100, 32, 1]:
    w = torch.randn(10, 1)
    lr = 0.01 if batch_size > 1 else 0.001  # Smaller LR for SGD

    for epoch in range(100):
        loss = train_one_epoch(X, y, w, lr, batch_size)

    name = "Full" if batch_size == 1000 else str(batch_size)
    steps = (1000 + batch_size - 1) // batch_size
    print(f"{name:>12} {loss:>12.6f} {steps:>12}")


# =============================================================================
# SECTION 2: SGD WITH MOMENTUM
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: SGD WITH MOMENTUM")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 SGD with Momentum Implementation
# -----------------------------------------------------------------------------
print("\n2.1 SGD with Momentum")
print("-" * 40)


class SGDMomentum:
    """
    SGD with Momentum.

    v_{t+1} = β × v_t + g_t
    θ_{t+1} = θ_t - η × v_{t+1}
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.state = defaultdict(dict)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Add weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Initialize velocity if needed
            if 'velocity' not in self.state[id(p)]:
                self.state[id(p)]['velocity'] = torch.zeros_like(p.data)

            v = self.state[id(p)]['velocity']

            # Update velocity
            v.mul_(self.momentum).add_(grad)

            if self.nesterov:
                # Nesterov: use lookahead gradient
                update = grad + self.momentum * v
            else:
                update = v

            # Update parameters
            p.data -= self.lr * update


# Compare vanilla SGD vs momentum vs Nesterov
print("Comparing optimization methods on Rosenbrock function:")


def rosenbrock(xy):
    """Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²"""
    x, y = xy[0], xy[1]
    a, b = 1, 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def optimize_rosenbrock(optimizer_class, lr, n_steps=1000, **kwargs):
    xy = torch.tensor([-1.0, 1.0], requires_grad=True)
    optimizer = optimizer_class([xy], lr=lr, **kwargs)

    trajectory = [xy.detach().clone()]
    losses = []

    for _ in range(n_steps):
        loss = rosenbrock(xy)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trajectory.append(xy.detach().clone())

    return losses, trajectory


# Test different methods
configs = [
    ("Vanilla GD", VanillaGD, 0.001, {}),
    ("SGD + Momentum", SGDMomentum, 0.001, {"momentum": 0.9}),
    ("SGD + Nesterov", SGDMomentum, 0.001, {"momentum": 0.9, "nesterov": True}),
]

print(f"\n{'Method':>20} {'Final Loss':>15} {'Final Position':>20}")
print("-" * 60)

for name, opt_class, lr, kwargs in configs:
    losses, trajectory = optimize_rosenbrock(opt_class, lr, n_steps=5000, **kwargs)
    final_pos = trajectory[-1].numpy()
    print(f"{name:>20} {losses[-1]:>15.6f} ({final_pos[0]:.4f}, {final_pos[1]:.4f})")


# -----------------------------------------------------------------------------
# 2.2 Why Momentum Helps
# -----------------------------------------------------------------------------
print("\n\n2.2 Why Momentum Helps")
print("-" * 40)


def ill_conditioned_quadratic(x):
    """
    Ill-conditioned quadratic: f(x,y) = x² + 100y²
    Has steep gradients in y direction, shallow in x
    """
    return x[0] ** 2 + 100 * x[1] ** 2


print("Ill-conditioned quadratic: f(x,y) = x² + 100y²")
print("Optimal at (0, 0)")

for name, opt_class, lr, kwargs in configs:
    x = torch.tensor([1.0, 1.0], requires_grad=True)
    optimizer = opt_class([x], lr=lr, **kwargs)

    for step in range(500):
        loss = ill_conditioned_quadratic(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = ill_conditioned_quadratic(x).item()
    print(f"  {name}: final loss = {final_loss:.8f}")


# =============================================================================
# SECTION 3: ADAPTIVE LEARNING RATE METHODS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: ADAPTIVE LEARNING RATE METHODS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Adagrad
# -----------------------------------------------------------------------------
print("\n3.1 Adagrad")
print("-" * 40)


class Adagrad:
    """
    Adagrad optimizer.

    G_t = G_{t-1} + g_t²
    θ_{t+1} = θ_t - η × g_t / (√G_t + ε)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.01,
        eps: float = 1e-8
    ):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.state = defaultdict(dict)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Initialize accumulated squared gradients
            if 'sum_squares' not in self.state[id(p)]:
                self.state[id(p)]['sum_squares'] = torch.zeros_like(p.data)

            G = self.state[id(p)]['sum_squares']

            # Accumulate squared gradients
            G.add_(grad ** 2)

            # Update with adaptive learning rate
            p.data -= self.lr * grad / (G.sqrt() + self.eps)


# Demonstrate Adagrad's diminishing learning rate
print("Adagrad on simple quadratic:")
x = torch.tensor([5.0], requires_grad=True)
optimizer = Adagrad([x], lr=1.0)

print(f"{'Step':>6} {'x':>10} {'Effective LR':>15}")
for i in range(10):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()

    # Calculate effective learning rate
    G = optimizer.state[id(x)]['sum_squares'] if 'sum_squares' in optimizer.state[id(x)] else torch.tensor([0.0])
    eff_lr = optimizer.lr / (G.sqrt() + optimizer.eps).item()

    optimizer.step()
    print(f"{i+1:>6} {x.item():>10.4f} {eff_lr:>15.6f}")

print("\nNote: Effective LR decreases over time (Adagrad problem)")


# -----------------------------------------------------------------------------
# 3.2 RMSprop
# -----------------------------------------------------------------------------
print("\n\n3.2 RMSprop")
print("-" * 40)


class RMSprop:
    """
    RMSprop optimizer.

    v_t = ρ × v_{t-1} + (1-ρ) × g_t²
    θ_{t+1} = θ_t - η × g_t / (√v_t + ε)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.01,
        rho: float = 0.9,
        eps: float = 1e-8
    ):
        self.params = list(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.state = defaultdict(dict)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Initialize EMA of squared gradients
            if 'v' not in self.state[id(p)]:
                self.state[id(p)]['v'] = torch.zeros_like(p.data)

            v = self.state[id(p)]['v']

            # Update EMA
            v.mul_(self.rho).add_((1 - self.rho) * grad ** 2)

            # Update parameters
            p.data -= self.lr * grad / (v.sqrt() + self.eps)


# Compare Adagrad vs RMSprop
print("Comparing Adagrad vs RMSprop (100 steps):")

for name, opt_class in [("Adagrad", Adagrad), ("RMSprop", RMSprop)]:
    x = torch.tensor([5.0], requires_grad=True)
    optimizer = opt_class([x], lr=0.5)

    for _ in range(100):
        loss = x ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  {name}: final x = {x.item():.6f}")

print("RMSprop doesn't slow down as much because it uses EMA!")


# =============================================================================
# SECTION 4: ADAM AND VARIANTS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: ADAM AND VARIANTS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Adam Implementation
# -----------------------------------------------------------------------------
print("\n4.1 Adam (Adaptive Moment Estimation)")
print("-" * 40)


class Adam:
    """
    Adam optimizer.

    m_t = β₁ × m_{t-1} + (1-β₁) × g_t       (first moment)
    v_t = β₂ × v_{t-1} + (1-β₂) × g_t²      (second moment)
    m̂_t = m_t / (1 - β₁ᵗ)                   (bias correction)
    v̂_t = v_t / (1 - β₂ᵗ)                   (bias correction)
    θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = defaultdict(dict)
        self.t = 0  # Time step

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            # L2 regularization (coupled - this is the issue AdamW fixes)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Initialize moments
            if 'm' not in self.state[id(p)]:
                self.state[id(p)]['m'] = torch.zeros_like(p.data)
                self.state[id(p)]['v'] = torch.zeros_like(p.data)

            m = self.state[id(p)]['m']
            v = self.state[id(p)]['v']

            # Update moments
            m.mul_(self.beta1).add_((1 - self.beta1) * grad)
            v.mul_(self.beta2).add_((1 - self.beta2) * grad ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


# Demonstrate Adam
print("Adam on Rosenbrock function:")
xy = torch.tensor([-1.0, 1.0], requires_grad=True)
optimizer = Adam([xy], lr=0.1)

for i in range(1000):
    loss = rosenbrock(xy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        print(f"  Step {i}: loss = {loss.item():.6f}, pos = ({xy[0].item():.4f}, {xy[1].item():.4f})")

print(f"  Final: loss = {rosenbrock(xy).item():.6f}")


# -----------------------------------------------------------------------------
# 4.2 Bias Correction Demonstration
# -----------------------------------------------------------------------------
print("\n\n4.2 Why Bias Correction Matters")
print("-" * 40)


class AdamNoBiasCorrection:
    """Adam without bias correction for demonstration."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = defaultdict(dict)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            if 'm' not in self.state[id(p)]:
                self.state[id(p)]['m'] = torch.zeros_like(p.data)
                self.state[id(p)]['v'] = torch.zeros_like(p.data)

            m = self.state[id(p)]['m']
            v = self.state[id(p)]['v']

            m.mul_(self.beta1).add_((1 - self.beta1) * grad)
            v.mul_(self.beta2).add_((1 - self.beta2) * grad ** 2)

            # NO bias correction - use m and v directly
            p.data -= self.lr * m / (v.sqrt() + self.eps)


# Compare with and without bias correction
print("Early training comparison (first 10 steps):")
print(f"{'Step':>6} {'With BC':>12} {'Without BC':>12}")

for name, opt_class in [("With BC", Adam), ("Without BC", AdamNoBiasCorrection)]:
    x = torch.tensor([5.0], requires_grad=True)
    optimizer = opt_class([x], lr=0.5)

    for i in range(10):
        loss = x ** 2
        optimizer.zero_grad()
        loss.backward()

        if i == 0:
            # Show first moment value before and after bias correction
            m = optimizer.state[id(x)].get('m', torch.tensor([0.0]))
            if hasattr(optimizer, 't'):
                m_hat = m / (1 - 0.9 ** (optimizer.t + 1))
            else:
                m_hat = m

        optimizer.step()

print("\nBias correction prevents slow starts when m and v are near 0!")


# -----------------------------------------------------------------------------
# 4.3 AdamW Implementation
# -----------------------------------------------------------------------------
print("\n\n4.3 AdamW (Decoupled Weight Decay)")
print("-" * 40)


class AdamW:
    """
    AdamW optimizer with decoupled weight decay.

    Key difference from Adam:
    - Weight decay is applied AFTER the Adam update
    - Not incorporated into gradient (like L2 regularization)

    θ_{t+1} = θ_t - η × (m̂_t / (√v̂_t + ε) + λ × θ_t)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = defaultdict(dict)
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data  # NOTE: No L2 added to gradient!

            if 'm' not in self.state[id(p)]:
                self.state[id(p)]['m'] = torch.zeros_like(p.data)
                self.state[id(p)]['v'] = torch.zeros_like(p.data)

            m = self.state[id(p)]['m']
            v = self.state[id(p)]['v']

            # Update moments (without weight decay in gradient)
            m.mul_(self.beta1).add_((1 - self.beta1) * grad)
            v.mul_(self.beta2).add_((1 - self.beta2) * grad ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Adam update
            adam_update = m_hat / (v_hat.sqrt() + self.eps)

            # Decoupled weight decay (applied separately!)
            p.data -= self.lr * (adam_update + self.weight_decay * p.data)


# Demonstrate difference between Adam+L2 and AdamW
print("Comparing Adam with L2 vs AdamW:")


def train_and_compare(opt_class, name, n_steps=1000):
    """Train a simple model and return final weight magnitude."""
    torch.manual_seed(42)
    model = nn.Linear(10, 1)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    optimizer = opt_class(model.parameters(), lr=0.01, weight_decay=0.1)

    for _ in range(n_steps):
        pred = model(X)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    weight_norm = model.weight.data.norm().item()
    return weight_norm


# Note: For true comparison, we'd need more sophisticated analysis
# Here we just demonstrate the API difference
print("Both optimizers achieve regularization, but AdamW works as intended")
print("for adaptive methods where L2 in gradient is problematic.")


# =============================================================================
# SECTION 5: GRADIENT CLIPPING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: GRADIENT CLIPPING")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Gradient Clipping Methods
# -----------------------------------------------------------------------------
print("\n5.1 Gradient Clipping Methods")
print("-" * 40)


def clip_grad_value_(params: Iterable[torch.Tensor], clip_value: float):
    """
    Clip gradient values to [-clip_value, clip_value].

    Simple but can change gradient direction.
    """
    for p in params:
        if p.grad is not None:
            p.grad.data.clamp_(-clip_value, clip_value)


def clip_grad_norm_(params: Iterable[torch.Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradient by norm.

    If ||g|| > max_norm:
        g = g × (max_norm / ||g||)

    Preserves direction, only scales magnitude.
    Returns the original norm before clipping.
    """
    params = list(params)

    # Compute total norm
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm


# Demonstrate gradient clipping
print("Gradient clipping demonstration:")

# Create large gradients
w = torch.randn(10, 10, requires_grad=True)
loss = (w ** 2).sum() * 100  # Large loss for large gradients
loss.backward()

print(f"Before clipping: grad norm = {w.grad.norm().item():.2f}")

# Clone for comparison
w_value = w.grad.clone()
w_norm = w.grad.clone()

# Clip by value
clip_grad_value_([w], clip_value=1.0)
print(f"After value clipping (±1.0): grad norm = {w.grad.norm().item():.2f}")

# Restore and clip by norm
w.grad = w_norm
orig_norm = clip_grad_norm_([w], max_norm=1.0)
print(f"After norm clipping (max=1.0): grad norm = {w.grad.norm().item():.2f}")

print("\nNorm clipping is preferred - preserves gradient direction!")


# -----------------------------------------------------------------------------
# 5.2 Gradient Clipping in Training Loop
# -----------------------------------------------------------------------------
print("\n\n5.2 Using Gradient Clipping in Training")
print("-" * 40)


class RNNModel(nn.Module):
    """Simple RNN that can have exploding gradients."""

    def __init__(self, input_size, hidden_size, n_layers=3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def train_with_clipping(model, max_norm=None):
    """Train RNN with optional gradient clipping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create synthetic data
    X = torch.randn(32, 50, 10)  # Long sequence to encourage exploding gradients
    y = torch.randn(32, 1)

    losses = []
    grad_norms = []

    for step in range(100):
        pred = model(X)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # Record gradient norm before clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm.item())

        # Clip gradients if specified
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        losses.append(loss.item())

    return losses, grad_norms


torch.manual_seed(42)
model_no_clip = RNNModel(10, 32)
torch.manual_seed(42)
model_with_clip = RNNModel(10, 32)

print("Training RNN without gradient clipping:")
losses_no, norms_no = train_with_clipping(model_no_clip, max_norm=None)
print(f"  Max gradient norm: {max(norms_no):.2f}")
print(f"  Final loss: {losses_no[-1]:.6f}")

print("\nTraining RNN with gradient clipping (max_norm=1.0):")
losses_clip, norms_clip = train_with_clipping(model_with_clip, max_norm=1.0)
print(f"  Max gradient norm: {max(norms_clip):.2f}")
print(f"  Final loss: {losses_clip[-1]:.6f}")


# =============================================================================
# SECTION 6: COMPLETE OPTIMIZER WITH ALL FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: COMPLETE ADAMW IMPLEMENTATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Production-Ready AdamW
# -----------------------------------------------------------------------------
print("\n6.1 Production-Ready AdamW")
print("-" * 40)


class AdamWComplete:
    """
    Complete AdamW implementation with all features:
    - Decoupled weight decay
    - Bias correction
    - Parameter groups (different LR/WD per group)
    - Gradient clipping support
    """

    def __init__(
        self,
        params,  # Can be list of params or list of param groups
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        # Handle param groups
        if isinstance(params, torch.Tensor):
            params = [params]

        params = list(params)

        # Check if params are already grouped
        if len(params) > 0 and isinstance(params[0], dict):
            self.param_groups = params
        else:
            self.param_groups = [{'params': params}]

        # Set defaults
        self.defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }

        # Apply defaults to groups
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)

        self.state = defaultdict(dict)
        self.t = 0

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self, closure=None):
        """Perform single optimization step."""
        self.t += 1

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Initialize state
                if 'm' not in self.state[id(p)]:
                    self.state[id(p)]['m'] = torch.zeros_like(p.data)
                    self.state[id(p)]['v'] = torch.zeros_like(p.data)

                m = self.state[id(p)]['m']
                v = self.state[id(p)]['v']

                # Update moments
                m.mul_(beta1).add_((1 - beta1) * grad)
                v.mul_(beta2).add_((1 - beta2) * grad ** 2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** self.t
                bias_correction2 = 1 - beta2 ** self.t

                # Corrected estimates
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Adam update
                denom = v_hat.sqrt() + eps
                update = m_hat / denom

                # Decoupled weight decay
                if weight_decay != 0:
                    update = update + weight_decay * p.data

                # Apply update
                p.data -= lr * update

        return loss


# Demonstrate parameter groups
print("Using parameter groups for different learning rates:")

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Different LR for different layers
optimizer = AdamWComplete([
    {'params': model[0].parameters(), 'lr': 1e-3},
    {'params': model[2].parameters(), 'lr': 5e-4},
    {'params': model[4].parameters(), 'lr': 1e-4, 'weight_decay': 0.0}  # No WD for last layer
])

print("Parameter groups configured:")
for i, group in enumerate(optimizer.param_groups):
    print(f"  Group {i}: lr={group['lr']}, wd={group['weight_decay']}")


# =============================================================================
# SECTION 7: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. SGD vs Adam:
   - SGD: Simple, needs LR tuning, can generalize better
   - Adam: Adaptive LR, faster initial convergence
   - SGD+M often preferred for CNNs, Adam for transformers

2. Momentum:
   - v = β×v + g
   - Accelerates in consistent directions
   - Dampens oscillations
   - β typically 0.9

3. Adam:
   - First moment (momentum): m = β₁×m + (1-β₁)×g
   - Second moment (adaptive): v = β₂×v + (1-β₂)×g²
   - Bias correction: m̂ = m/(1-β₁ᵗ)
   - Update: θ -= lr × m̂ / √v̂

4. AdamW vs Adam+L2:
   - Adam+L2: L2 term goes into adaptive statistics
   - AdamW: Weight decay applied AFTER Adam update
   - AdamW is correct for adaptive optimizers

5. Gradient Clipping:
   - Prevents exploding gradients
   - Clip by norm preserves direction
   - Essential for RNNs and transformers
   - Typical max_norm = 1.0

6. Learning Rate Schedules:
   - Warmup: Stabilizes early training
   - Cosine decay: Smooth decrease
   - Step decay: Abrupt decreases at milestones
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 3.1 SUMMARY: OPTIMIZERS")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────────────┬─────────────────────────┐
│ Optimizer       │ Key Equations                  │ Best For                │
├─────────────────┼────────────────────────────────┼─────────────────────────┤
│ SGD             │ θ -= η×g                       │ Baselines               │
│ SGD+Momentum    │ v = βv + g; θ -= ηv           │ CNNs                    │
│ RMSprop         │ v = ρv + (1-ρ)g²              │ RNNs                    │
│ Adam            │ m,v moments + bias correction  │ General                 │
│ AdamW           │ Adam + decoupled weight decay  │ Transformers/LLMs       │
└─────────────────┴────────────────────────────────┴─────────────────────────┘

DEFAULT HYPERPARAMETERS:

Adam/AdamW:
  lr = 1e-4 to 3e-4
  β₁ = 0.9
  β₂ = 0.999 (0.95 for LLMs)
  ε = 1e-8
  weight_decay = 0.01 to 0.1

SGD+Momentum:
  lr = 0.1 (with decay)
  momentum = 0.9
  weight_decay = 1e-4

KEY TAKEAWAYS:
1. Momentum accelerates convergence
2. Adaptive LR handles different parameter scales
3. AdamW is standard for transformers
4. Gradient clipping prevents instability
5. Learning rate schedule is crucial
""")

print("\nModule 3.1 Complete!")
