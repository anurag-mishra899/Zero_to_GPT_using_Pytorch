"""
Module 2.2: Activation Functions - From ReLU to SwiGLU
Complete implementation of activation functions for deep learning and LLMs.

This module covers:
1. Classic activations (Sigmoid, Tanh)
2. Modern activations (ReLU, Leaky ReLU, ELU, GELU, SiLU)
3. Gated activations (GLU, SwiGLU, GeGLU) - Critical for LLMs
4. Numerical stability considerations
5. Comparison and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Callable

print("=" * 70)
print("MODULE 2.2: ACTIVATION FUNCTIONS - FROM ReLU TO SwiGLU")
print("=" * 70)

# =============================================================================
# SECTION 1: CLASSIC ACTIVATION FUNCTIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: CLASSIC ACTIVATION FUNCTIONS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Sigmoid
# -----------------------------------------------------------------------------
print("\n1.1 Sigmoid Function")
print("-" * 40)

class Sigmoid:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))

    Properties:
    - Output range: (0, 1)
    - Derivative: σ(x) * (1 - σ(x))
    - Max derivative: 0.25 at x=0

    Issues:
    - Saturates at extremes (vanishing gradients)
    - Not zero-centered
    - Computationally expensive (exp)
    """

    @staticmethod
    def forward_unstable(x: torch.Tensor) -> torch.Tensor:
        """Naive implementation - can overflow for large negative x."""
        return 1 / (1 + torch.exp(-x))

    @staticmethod
    def forward_stable(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable sigmoid implementation."""
        # For x >= 0: 1 / (1 + e^(-x))
        # For x < 0: e^x / (1 + e^x) - avoids overflow in exp(-x)
        return torch.where(
            x >= 0,
            1 / (1 + torch.exp(-x)),
            torch.exp(x) / (1 + torch.exp(x))
        )

    @staticmethod
    def backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Gradient of sigmoid: σ(x) * (1 - σ(x))"""
        sig = torch.sigmoid(x)
        return grad_output * sig * (1 - sig)


# Demonstrate sigmoid
x = torch.linspace(-10, 10, 21)
sigmoid_out = torch.sigmoid(x)
print(f"Input range: [{x.min():.1f}, {x.max():.1f}]")
print(f"Output range: [{sigmoid_out.min():.4f}, {sigmoid_out.max():.4f}]")

# Show saturation problem
x_extreme = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], requires_grad=True)
y = torch.sigmoid(x_extreme)
y.sum().backward()
print(f"\nSaturation demonstration:")
print(f"x values: {x_extreme.data.tolist()}")
print(f"σ(x) values: {[f'{v:.6f}' for v in y.data.tolist()]}")
print(f"Gradients: {[f'{v:.6f}' for v in x_extreme.grad.tolist()]}")
print("Note: Gradients nearly 0 at extremes - vanishing gradient problem!")


# -----------------------------------------------------------------------------
# 1.2 Tanh
# -----------------------------------------------------------------------------
print("\n\n1.2 Tanh Function")
print("-" * 40)

class Tanh:
    """
    Tanh activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                             = 2σ(2x) - 1

    Properties:
    - Output range: (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Derivative: 1 - tanh²(x)
    - Max derivative: 1 at x=0
    - Still saturates at extremes
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def forward_from_sigmoid(x: torch.Tensor) -> torch.Tensor:
        """tanh(x) = 2σ(2x) - 1"""
        return 2 * torch.sigmoid(2 * x) - 1

    @staticmethod
    def backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Gradient: 1 - tanh²(x)"""
        tanh_x = torch.tanh(x)
        return grad_output * (1 - tanh_x ** 2)


# Demonstrate tanh
x = torch.linspace(-5, 5, 11)
tanh_out = torch.tanh(x)
print(f"Input: {x.tolist()}")
print(f"tanh(x): {[f'{v:.4f}' for v in tanh_out.tolist()]}")

# Compare gradients with sigmoid
x_compare = torch.tensor([0.0], requires_grad=True)
sig_out = torch.sigmoid(x_compare)
sig_out.backward()
sig_grad = x_compare.grad.item()

x_compare2 = torch.tensor([0.0], requires_grad=True)
tanh_out = torch.tanh(x_compare2)
tanh_out.backward()
tanh_grad = x_compare2.grad.item()

print(f"\nGradient at x=0:")
print(f"  Sigmoid: {sig_grad:.4f}")
print(f"  Tanh: {tanh_grad:.4f}")
print("Tanh has 4x stronger gradient at origin!")


# =============================================================================
# SECTION 2: RELU FAMILY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: RELU FAMILY")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 ReLU
# -----------------------------------------------------------------------------
print("\n2.1 ReLU (Rectified Linear Unit)")
print("-" * 40)

class ReLU:
    """
    ReLU: f(x) = max(0, x)

    Properties:
    - Simple and fast
    - No saturation for positive values
    - Sparse activations
    - Derivative: 1 if x > 0, 0 otherwise

    Problem: Dying ReLU
    - If weights push all inputs negative, neuron dies
    - Gradient = 0, so weights never update
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))

    @staticmethod
    def forward_efficient(x: torch.Tensor) -> torch.Tensor:
        """More memory efficient using clamp."""
        return x.clamp(min=0)

    @staticmethod
    def backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Gradient: 1 if x > 0, 0 otherwise"""
        return grad_output * (x > 0).float()


# Demonstrate ReLU
x = torch.linspace(-5, 5, 11)
relu_out = F.relu(x)
print(f"Input: {x.tolist()}")
print(f"ReLU(x): {relu_out.tolist()}")

# Demonstrate dying ReLU problem
print("\nDying ReLU Demonstration:")
print("-" * 30)

class DyingReLUDemo(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with negative weights to simulate dying neuron
        self.linear = nn.Linear(10, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(-1.0)  # All negative weights

    def forward(self, x):
        return F.relu(self.linear(x))

demo = DyingReLUDemo()
x_positive = torch.ones(1, 10)  # Positive inputs
output = demo(x_positive)
output.backward()

print(f"Input (all positive): {x_positive[0, :3].tolist()}...")
print(f"After linear (weights=-1): {demo.linear(x_positive).item():.2f}")
print(f"After ReLU: {output.item():.2f}")
print(f"Gradient: {demo.linear.weight.grad}")
print("Neuron is DEAD - gradient is 0, weights will never update!")


# -----------------------------------------------------------------------------
# 2.2 Leaky ReLU
# -----------------------------------------------------------------------------
print("\n\n2.2 Leaky ReLU")
print("-" * 40)

class LeakyReLU:
    """
    Leaky ReLU: f(x) = x if x > 0, αx otherwise

    Properties:
    - Fixes dying ReLU (small gradient for negatives)
    - α is typically 0.01
    - Allows neurons to recover from "death"
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * x)

    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        grad = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, self.alpha))
        return grad_output * grad


# Demonstrate Leaky ReLU
x = torch.linspace(-5, 5, 11)
leaky = LeakyReLU(alpha=0.1)
leaky_out = leaky.forward(x)
print(f"Input: {x.tolist()}")
print(f"LeakyReLU(x, α=0.1): {[f'{v:.1f}' for v in leaky_out.tolist()]}")

# Compare with standard ReLU
x_neg = torch.tensor([-2.0], requires_grad=True)
relu_out = F.relu(x_neg)
relu_out.backward()
relu_grad = x_neg.grad.item()

x_neg2 = torch.tensor([-2.0], requires_grad=True)
leaky_out = F.leaky_relu(x_neg2, negative_slope=0.1)
leaky_out.backward()
leaky_grad = x_neg2.grad.item()

print(f"\nGradient at x=-2:")
print(f"  ReLU: {relu_grad}")
print(f"  Leaky ReLU (α=0.1): {leaky_grad}")


# -----------------------------------------------------------------------------
# 2.3 PReLU (Parametric ReLU)
# -----------------------------------------------------------------------------
print("\n\n2.3 PReLU (Parametric ReLU)")
print("-" * 40)

class PReLU(nn.Module):
    """
    PReLU: f(x) = x if x > 0, αx otherwise

    Unlike Leaky ReLU, α is LEARNED during training.
    Can have one α per channel or shared.
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.alpha)


# Demonstrate PReLU
prelu = PReLU(num_parameters=1, init=0.1)
print(f"Initial α: {prelu.alpha.item():.3f}")

# Simulate training
x = torch.randn(100, 1)
y_target = torch.where(x > 0, x, 0.3 * x)  # Target: LeakyReLU with α=0.3

optimizer = torch.optim.SGD(prelu.parameters(), lr=0.1)
for epoch in range(100):
    y_pred = prelu(x)
    loss = F.mse_loss(y_pred, y_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Learned α after training: {prelu.alpha.item():.3f}")
print("PReLU learned to approximate target α=0.3!")


# -----------------------------------------------------------------------------
# 2.4 ELU (Exponential Linear Unit)
# -----------------------------------------------------------------------------
print("\n\n2.4 ELU (Exponential Linear Unit)")
print("-" * 40)

class ELU:
    """
    ELU: f(x) = x if x > 0, α(e^x - 1) otherwise

    Properties:
    - Smooth everywhere (unlike ReLU)
    - Mean activations closer to zero
    - More expensive than ReLU (exponential)
    - α typically = 1.0
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        # For x > 0: derivative = 1
        # For x <= 0: derivative = α * e^x = ELU(x) + α
        grad = torch.where(x > 0, torch.ones_like(x), self.forward(x) + self.alpha)
        return grad_output * grad


# Demonstrate ELU
x = torch.linspace(-3, 3, 13)
elu = ELU(alpha=1.0)
elu_out = elu.forward(x)
print(f"Input: {[f'{v:.1f}' for v in x.tolist()]}")
print(f"ELU(x): {[f'{v:.2f}' for v in elu_out.tolist()]}")

# Compare smoothness
print("\nSmoothness comparison near x=0:")
x_near_zero = torch.tensor([-0.01, 0.0, 0.01])
print(f"  ReLU: {F.relu(x_near_zero).tolist()}")
print(f"  ELU:  {[f'{v:.4f}' for v in F.elu(x_near_zero).tolist()]}")


# =============================================================================
# SECTION 3: MODERN ACTIVATIONS (TRANSFORMERS)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODERN ACTIVATIONS (TRANSFORMERS)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 GELU (Gaussian Error Linear Unit)
# -----------------------------------------------------------------------------
print("\n3.1 GELU (Gaussian Error Linear Unit)")
print("-" * 40)

class GELU:
    """
    GELU: f(x) = x * Φ(x)

    Where Φ(x) is the CDF of standard normal distribution.

    Intuition: Weight input by "probability it's positive"

    Properties:
    - Smooth, differentiable everywhere
    - Non-monotonic (slight negative bump)
    - Used in BERT, GPT-2, GPT-3
    """

    @staticmethod
    def forward_exact(x: torch.Tensor) -> torch.Tensor:
        """Exact GELU using error function."""
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def forward_tanh_approx(x: torch.Tensor) -> torch.Tensor:
        """Tanh approximation - commonly used in libraries."""
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
        ))

    @staticmethod
    def forward_sigmoid_approx(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid approximation - fastest."""
        return x * torch.sigmoid(1.702 * x)

    @staticmethod
    def backward_exact(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Gradient of exact GELU."""
        cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        pdf = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        return grad_output * (cdf + x * pdf)


# Demonstrate GELU implementations
x = torch.linspace(-3, 3, 13)
gelu_exact = GELU.forward_exact(x)
gelu_tanh = GELU.forward_tanh_approx(x)
gelu_sigmoid = GELU.forward_sigmoid_approx(x)

print("Comparing GELU implementations:")
print(f"Input:          {[f'{v:.1f}' for v in x.tolist()]}")
print(f"Exact:          {[f'{v:.3f}' for v in gelu_exact.tolist()]}")
print(f"Tanh approx:    {[f'{v:.3f}' for v in gelu_tanh.tolist()]}")
print(f"Sigmoid approx: {[f'{v:.3f}' for v in gelu_sigmoid.tolist()]}")

# Max approximation error
max_error_tanh = (gelu_exact - gelu_tanh).abs().max().item()
max_error_sigmoid = (gelu_exact - gelu_sigmoid).abs().max().item()
print(f"\nMax approximation error:")
print(f"  Tanh approx: {max_error_tanh:.6f}")
print(f"  Sigmoid approx: {max_error_sigmoid:.6f}")


# -----------------------------------------------------------------------------
# 3.2 SiLU / Swish
# -----------------------------------------------------------------------------
print("\n\n3.2 SiLU / Swish")
print("-" * 40)

class SiLU:
    """
    SiLU (Sigmoid Linear Unit) / Swish: f(x) = x * σ(x)

    Properties:
    - Self-gated: input gates itself
    - Smooth, non-monotonic
    - Bounded below (~-0.28), unbounded above
    - Used in EfficientNet, many LLMs

    Note: Very similar to GELU in practice
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    @staticmethod
    def forward_with_beta(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Swish-β: f(x) = x * σ(βx)"""
        return x * torch.sigmoid(beta * x)

    @staticmethod
    def backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Derivative: σ(x) + x * σ(x) * (1 - σ(x))
                  = σ(x) * (1 + x * (1 - σ(x)))
        """
        sig = torch.sigmoid(x)
        return grad_output * (sig + x * sig * (1 - sig))


# Demonstrate SiLU
x = torch.linspace(-5, 5, 21)
silu_out = F.silu(x)
print(f"SiLU minimum: {silu_out.min().item():.4f}")
print(f"(occurs near x ≈ -1.28)")

# Compare GELU and SiLU
print("\nGELU vs SiLU comparison:")
x_compare = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
gelu_vals = F.gelu(x_compare)
silu_vals = F.silu(x_compare)
print(f"x:    {x_compare.tolist()}")
print(f"GELU: {[f'{v:.4f}' for v in gelu_vals.tolist()]}")
print(f"SiLU: {[f'{v:.4f}' for v in silu_vals.tolist()]}")
print("Very similar behavior!")


# =============================================================================
# SECTION 4: GATED ACTIVATIONS (LLM CRITICAL)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: GATED ACTIVATIONS (LLM CRITICAL)")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 GLU (Gated Linear Unit)
# -----------------------------------------------------------------------------
print("\n4.1 GLU (Gated Linear Unit)")
print("-" * 40)

class GLU(nn.Module):
    """
    GLU: f(x) = (x @ W1) * σ(x @ W2)

    Key insight: Split computation into "content" and "gate"
    - Content: What information to represent
    - Gate: What information to let through

    This is the foundation for SwiGLU, GeGLU, etc.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w_content = nn.Linear(d_model, d_ff, bias=bias)
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content = self.w_content(x)
        gate = torch.sigmoid(self.w_gate(x))
        return content * gate


# Demonstrate GLU
d_model, d_ff = 64, 256
glu = GLU(d_model, d_ff)
x = torch.randn(2, 10, d_model)  # (batch, seq, d_model)
output = glu(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")


# -----------------------------------------------------------------------------
# 4.2 SwiGLU (Used in LLaMA, PaLM)
# -----------------------------------------------------------------------------
print("\n\n4.2 SwiGLU (State-of-the-Art for LLMs)")
print("-" * 40)

class SwiGLU(nn.Module):
    """
    SwiGLU: f(x) = (x @ W_up) * SiLU(x @ W_gate) @ W_down

    Used in LLaMA, PaLM, and likely GPT-4.

    Architecture:
    - W_gate: Projects to intermediate dimension, applies SiLU
    - W_up: Projects to intermediate dimension (no activation)
    - Element-wise multiplication (gating)
    - W_down: Projects back to model dimension

    Why SwiGLU over GELU:
    - Better empirical performance at scale
    - Multiplicative interactions more expressive
    - Smoother optimization landscape
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        bias: bool = False,
        multiple_of: int = 256
    ):
        super().__init__()

        # Default FFN dimension for SwiGLU
        # Standard FFN uses 4 * d_model
        # SwiGLU uses (8/3) * d_model to match parameter count
        if d_ff is None:
            d_ff = int(8 * d_model / 3)

        # Round to multiple for efficiency
        d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path: apply SiLU activation
        gate = F.silu(self.w_gate(x))

        # Up path: linear projection (no activation)
        up = self.w_up(x)

        # Element-wise gating
        gated = gate * up

        # Down projection
        return self.w_down(gated)


# Demonstrate SwiGLU
d_model = 512
swiglu = SwiGLU(d_model)
x = torch.randn(2, 128, d_model)  # (batch, seq, d_model)
output = swiglu(x)

print(f"SwiGLU Configuration:")
print(f"  d_model: {d_model}")
print(f"  d_ff: {swiglu.d_ff}")
print(f"  Ratio: {swiglu.d_ff / d_model:.2f}x (≈8/3)")
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Parameter count comparison
class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

standard_ffn = StandardFFN(d_model)
standard_params = sum(p.numel() for p in standard_ffn.parameters())
swiglu_params = sum(p.numel() for p in swiglu.parameters())

print(f"\nParameter comparison:")
print(f"  Standard FFN (4x): {standard_params:,}")
print(f"  SwiGLU (8/3x): {swiglu_params:,}")
print(f"  Difference: {abs(standard_params - swiglu_params):,} ({100*abs(standard_params - swiglu_params)/standard_params:.1f}%)")


# -----------------------------------------------------------------------------
# 4.3 GeGLU
# -----------------------------------------------------------------------------
print("\n\n4.3 GeGLU")
print("-" * 40)

class GeGLU(nn.Module):
    """
    GeGLU: f(x) = (x @ W_up) * GELU(x @ W_gate) @ W_down

    Same as SwiGLU but uses GELU instead of SiLU.
    Used in some models like T5 1.1.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, bias: bool = False):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# Demonstrate GeGLU
geglu = GeGLU(d_model)
output = geglu(x)
print(f"GeGLU output shape: {output.shape}")


# -----------------------------------------------------------------------------
# 4.4 ReGLU
# -----------------------------------------------------------------------------
print("\n\n4.4 ReGLU")
print("-" * 40)

class ReGLU(nn.Module):
    """
    ReGLU: f(x) = (x @ W_up) * ReLU(x @ W_gate) @ W_down

    Simplest gated variant using ReLU.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, bias: bool = False):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# -----------------------------------------------------------------------------
# 4.5 Complete LLaMA-style FFN Block
# -----------------------------------------------------------------------------
print("\n\n4.5 Complete LLaMA-style FFN Block")
print("-" * 40)

class LLaMAFFN(nn.Module):
    """
    Feed-forward network as used in LLaMA.

    Architecture:
    - Pre-normalization (RMSNorm applied before FFN in the block)
    - SwiGLU activation
    - No bias terms

    In a transformer block:
    x = x + FFN(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 1.0
    ):
        super().__init__()

        # Compute FFN dimension
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = int(d_ff * ffn_dim_multiplier)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# Demonstrate LLaMA FFN
llama_ffn = LLaMAFFN(d_model=4096)
x = torch.randn(1, 1, 4096)
output = llama_ffn(x)
print(f"LLaMA FFN configuration:")
print(f"  d_model: 4096")
print(f"  d_ff: {llama_ffn.w_gate.out_features}")
print(f"  Parameters: {sum(p.numel() for p in llama_ffn.parameters()):,}")


# =============================================================================
# SECTION 5: NUMERICAL STABILITY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: NUMERICAL STABILITY")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Stable Softmax
# -----------------------------------------------------------------------------
print("\n5.1 Stable Softmax")
print("-" * 40)

def softmax_unstable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unstable softmax - can overflow."""
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable softmax - subtracts max before exp."""
    x_max = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# Demonstrate overflow problem
x_large = torch.tensor([1000.0, 1001.0, 1002.0])
print(f"Input with large values: {x_large.tolist()}")

try:
    unstable_result = softmax_unstable(x_large)
    print(f"Unstable softmax: {unstable_result.tolist()}")
except RuntimeError as e:
    print(f"Unstable softmax: OVERFLOW!")

stable_result = softmax_stable(x_large)
print(f"Stable softmax: {stable_result.tolist()}")
print(f"PyTorch softmax: {F.softmax(x_large, dim=0).tolist()}")


# -----------------------------------------------------------------------------
# 5.2 Log-Softmax for Numerical Stability
# -----------------------------------------------------------------------------
print("\n\n5.2 Log-Softmax")
print("-" * 40)

def log_softmax_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Naive: log(softmax(x)) - numerically unstable."""
    return torch.log(F.softmax(x, dim=dim))

def log_softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Stable log-softmax using log-sum-exp trick.
    log(softmax(x)) = x - log(sum(exp(x)))
                    = x - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = x.max(dim=dim, keepdim=True)[0]
    log_sum_exp = torch.log(torch.exp(x - x_max).sum(dim=dim, keepdim=True)) + x_max
    return x - log_sum_exp


# Demonstrate
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Input: {x.tolist()}")
print(f"Log-softmax (naive): {log_softmax_naive(x).tolist()}")
print(f"Log-softmax (stable): {[f'{v:.4f}' for v in log_softmax_stable(x).tolist()]}")
print(f"PyTorch F.log_softmax: {[f'{v:.4f}' for v in F.log_softmax(x, dim=0).tolist()]}")


# =============================================================================
# SECTION 6: ACTIVATION COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: ACTIVATION COMPARISON")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Gradient Flow Analysis
# -----------------------------------------------------------------------------
print("\n6.1 Gradient Flow Analysis")
print("-" * 40)

def analyze_gradient_flow(activation_fn: Callable, name: str, n_samples: int = 10000):
    """Analyze gradient statistics for an activation function."""
    x = torch.randn(n_samples, requires_grad=True)
    y = activation_fn(x)
    y.sum().backward()

    # Statistics
    grad_mean = x.grad.mean().item()
    grad_std = x.grad.std().item()
    grad_zero_frac = (x.grad == 0).float().mean().item()

    return {
        'mean': grad_mean,
        'std': grad_std,
        'zero_fraction': grad_zero_frac
    }


activations = {
    'ReLU': F.relu,
    'Leaky ReLU': lambda x: F.leaky_relu(x, 0.01),
    'GELU': F.gelu,
    'SiLU': F.silu,
    'Tanh': torch.tanh,
    'Sigmoid': torch.sigmoid
}

print(f"{'Activation':<15} {'Mean Grad':>12} {'Std Grad':>12} {'Zero Frac':>12}")
print("-" * 55)
for name, fn in activations.items():
    stats = analyze_gradient_flow(fn, name)
    print(f"{name:<15} {stats['mean']:>12.4f} {stats['std']:>12.4f} {stats['zero_fraction']:>12.2%}")


# -----------------------------------------------------------------------------
# 6.2 Output Statistics
# -----------------------------------------------------------------------------
print("\n\n6.2 Output Statistics (x ~ N(0,1))")
print("-" * 40)

def analyze_output_stats(activation_fn: Callable, name: str, n_samples: int = 10000):
    """Analyze output statistics for an activation function."""
    x = torch.randn(n_samples)
    y = activation_fn(x)

    return {
        'mean': y.mean().item(),
        'std': y.std().item(),
        'min': y.min().item(),
        'max': y.max().item()
    }


print(f"{'Activation':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 60)
for name, fn in activations.items():
    stats = analyze_output_stats(fn, name)
    print(f"{name:<15} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['min']:>10.4f} {stats['max']:>10.4f}")


# -----------------------------------------------------------------------------
# 6.3 FFN Performance Comparison
# -----------------------------------------------------------------------------
print("\n\n6.3 FFN Activation Performance Comparison")
print("-" * 40)

class FlexibleFFN(nn.Module):
    """FFN with configurable activation for comparison."""

    def __init__(self, d_model: int, activation: str = 'relu'):
        super().__init__()

        if activation in ['swiglu', 'geglu', 'reglu']:
            # Gated activations need 3 linear layers
            d_ff = int(8 * d_model / 3)
            self.is_gated = True
            self.w_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w_up = nn.Linear(d_model, d_ff, bias=False)
            self.w_down = nn.Linear(d_ff, d_model, bias=False)

            if activation == 'swiglu':
                self.gate_fn = F.silu
            elif activation == 'geglu':
                self.gate_fn = F.gelu
            else:  # reglu
                self.gate_fn = F.relu
        else:
            # Standard 2-layer FFN
            d_ff = 4 * d_model
            self.is_gated = False
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)

            if activation == 'relu':
                self.act_fn = F.relu
            elif activation == 'gelu':
                self.act_fn = F.gelu
            else:  # silu
                self.act_fn = F.silu

    def forward(self, x):
        if self.is_gated:
            return self.w_down(self.gate_fn(self.w_gate(x)) * self.w_up(x))
        else:
            return self.w2(self.act_fn(self.w1(x)))


# Compare forward pass times
import time

d_model = 1024
batch_size = 32
seq_len = 512
n_iterations = 100

print(f"Configuration: d_model={d_model}, batch={batch_size}, seq={seq_len}")
print(f"Running {n_iterations} iterations each\n")

ffn_configs = ['relu', 'gelu', 'silu', 'swiglu', 'geglu']

for config in ffn_configs:
    ffn = FlexibleFFN(d_model, activation=config)
    x = torch.randn(batch_size, seq_len, d_model)

    # Warmup
    for _ in range(10):
        _ = ffn(x)

    # Timed run
    start = time.time()
    for _ in range(n_iterations):
        _ = ffn(x)
    elapsed = time.time() - start

    params = sum(p.numel() for p in ffn.parameters())
    print(f"{config.upper():<10}: {elapsed*1000:.2f}ms total, {params/1e6:.2f}M params")


# =============================================================================
# SECTION 7: CUSTOM ACTIVATION FOR EXPERIMENTATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: CUSTOM ACTIVATION IMPLEMENTATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 7.1 Custom Activation with Autograd
# -----------------------------------------------------------------------------
print("\n7.1 Custom Activation with Autograd")
print("-" * 40)

class CustomSiLU(torch.autograd.Function):
    """
    Custom SiLU implementation using autograd.Function
    for complete control over forward and backward passes.
    """

    @staticmethod
    def forward(ctx, x):
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid_x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        x, sigmoid_x = ctx.saved_tensors
        # d/dx[x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
        grad_input = grad_output * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        return grad_input


# Verify custom implementation
x = torch.randn(100, requires_grad=True)
x_copy = x.detach().clone().requires_grad_(True)

# Custom
y_custom = CustomSiLU.apply(x)
y_custom.sum().backward()

# PyTorch
y_pytorch = F.silu(x_copy)
y_pytorch.sum().backward()

# Compare
forward_diff = (y_custom - y_pytorch).abs().max().item()
backward_diff = (x.grad - x_copy.grad).abs().max().item()

print(f"Forward difference: {forward_diff:.10f}")
print(f"Backward difference: {backward_diff:.10f}")
print("Custom implementation matches PyTorch!")


# -----------------------------------------------------------------------------
# 7.2 Learnable Activation Parameters
# -----------------------------------------------------------------------------
print("\n\n7.2 Learnable Activation (Swish-β)")
print("-" * 40)

class SwishBeta(nn.Module):
    """
    Swish with learnable β parameter: f(x) = x * σ(βx)

    When β=1: standard Swish/SiLU
    When β→∞: approaches ReLU
    When β=0: linear (f(x) = x/2)
    """

    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


# Demonstrate learning β
swish = SwishBeta(init_beta=1.0)
print(f"Initial β: {swish.beta.item():.3f}")

# Simple optimization to learn β
optimizer = torch.optim.Adam(swish.parameters(), lr=0.1)
x = torch.randn(1000)

# Target: ReLU-like behavior (high β)
target = F.relu(x)

for i in range(100):
    output = swish(x)
    loss = F.mse_loss(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 25 == 0:
        print(f"Step {i+1}: β = {swish.beta.item():.3f}, loss = {loss.item():.4f}")

print(f"\nFinal β: {swish.beta.item():.3f}")
print("β increased toward infinity, approximating ReLU!")


# =============================================================================
# SECTION 8: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. Why Non-linearity?
   - Without activation, deep network = single linear transform
   - Proof: f(x) = W2(W1x) = (W2W1)x = Wx

2. Dying ReLU Problem:
   - Negative inputs → zero gradient → weights stuck
   - Solutions: Leaky ReLU, PReLU, ELU, careful initialization

3. Why GELU in Transformers (BERT, GPT-2)?
   - Smooth gradient (no kink at 0)
   - Probabilistic interpretation
   - Works well with LayerNorm

4. Why SwiGLU in Modern LLMs (LLaMA, PaLM)?
   - Gating provides multiplicative interactions
   - Better empirical performance at scale
   - Uses 3 weight matrices (gate, up, down)
   - Dimension adjusted to (8/3) × d_model to match param count

5. GELU vs SiLU:
   - GELU: x × Φ(x) using error function
   - SiLU: x × σ(x) using sigmoid
   - Very similar in practice, SiLU slightly faster

6. Numerical Stability:
   - Softmax: subtract max before exp
   - Sigmoid: different formula for pos/neg
   - Log-softmax: use log-sum-exp trick
""")


# =============================================================================
# SECTION 9: COMPREHENSIVE ACTIVATION FUNCTION LIBRARY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: COMPREHENSIVE ACTIVATION LIBRARY")
print("=" * 70)

class ActivationLibrary:
    """
    A comprehensive collection of activation functions.
    Use this as reference for implementing activations in your projects.
    """

    # Classic
    @staticmethod
    def sigmoid(x): return torch.sigmoid(x)

    @staticmethod
    def tanh(x): return torch.tanh(x)

    # ReLU Family
    @staticmethod
    def relu(x): return F.relu(x)

    @staticmethod
    def leaky_relu(x, alpha=0.01): return F.leaky_relu(x, alpha)

    @staticmethod
    def elu(x, alpha=1.0): return F.elu(x, alpha)

    @staticmethod
    def selu(x): return F.selu(x)

    @staticmethod
    def relu6(x): return F.relu6(x)

    # Modern
    @staticmethod
    def gelu(x): return F.gelu(x)

    @staticmethod
    def gelu_tanh_approx(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def silu(x): return F.silu(x)

    @staticmethod
    def mish(x): return x * torch.tanh(F.softplus(x))

    # Softmax variants
    @staticmethod
    def softmax(x, dim=-1): return F.softmax(x, dim=dim)

    @staticmethod
    def log_softmax(x, dim=-1): return F.log_softmax(x, dim=dim)

    @staticmethod
    def softplus(x, beta=1.0): return F.softplus(x, beta=beta)

    # Hard variants (for efficiency)
    @staticmethod
    def hardtanh(x, min_val=-1, max_val=1):
        return F.hardtanh(x, min_val, max_val)

    @staticmethod
    def hardsigmoid(x): return F.hardsigmoid(x)

    @staticmethod
    def hardswish(x): return F.hardswish(x)


# Demonstrate library
print("\nActivation Library Examples:")
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x.tolist()}")
print(f"GELU: {[f'{v:.4f}' for v in ActivationLibrary.gelu(x).tolist()]}")
print(f"SiLU: {[f'{v:.4f}' for v in ActivationLibrary.silu(x).tolist()]}")
print(f"Mish: {[f'{v:.4f}' for v in ActivationLibrary.mish(x).tolist()]}")
print(f"HardSwish: {[f'{v:.4f}' for v in ActivationLibrary.hardswish(x).tolist()]}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 2.2 SUMMARY: ACTIVATION FUNCTIONS")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────┬─────────────────────────┐
│ Activation      │ Formula                │ Use Case                │
├─────────────────┼────────────────────────┼─────────────────────────┤
│ ReLU            │ max(0, x)              │ CNNs, general           │
│ Leaky ReLU      │ max(αx, x)             │ When dying ReLU occurs  │
│ GELU            │ x × Φ(x)               │ BERT, GPT-2, GPT-3      │
│ SiLU/Swish      │ x × σ(x)               │ EfficientNet, some LLMs │
│ SwiGLU          │ (xW₁) ⊗ SiLU(xW₂)     │ LLaMA, PaLM, GPT-4(?)   │
│ Softmax         │ exp(xᵢ)/Σexp(xⱼ)       │ Classification output   │
│ Sigmoid         │ 1/(1+e⁻ˣ)              │ Binary output, gates    │
└─────────────────┴────────────────────────┴─────────────────────────┘

KEY TAKEAWAYS:
1. Non-linearity is essential - without it, deep = shallow
2. ReLU revolutionized deep learning but has dying neuron problem
3. GELU/SiLU are smooth alternatives, better for transformers
4. SwiGLU dominates modern LLMs (LLaMA, PaLM)
5. Always consider numerical stability (softmax, sigmoid)
6. Gating provides multiplicative interactions - more expressive

MODERN LLM FFN ARCHITECTURE:
    Input → SwiGLU → Output
    where SwiGLU = W_down(SiLU(W_gate(x)) ⊗ W_up(x))
""")

print("\nModule 2.2 Complete!")
