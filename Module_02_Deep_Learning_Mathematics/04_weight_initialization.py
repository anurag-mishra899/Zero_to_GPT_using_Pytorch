"""
Module 2.4: Weight Initialization - From Xavier to Modern Practices
Complete implementation of initialization strategies for deep learning and LLMs.

This module covers:
1. Why initialization matters (variance analysis)
2. Xavier/Glorot initialization
3. He/Kaiming initialization
4. Modern transformer initialization
5. Practical debugging techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Callable

print("=" * 70)
print("MODULE 2.4: WEIGHT INITIALIZATION")
print("=" * 70)

# =============================================================================
# SECTION 1: WHY INITIALIZATION MATTERS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: WHY INITIALIZATION MATTERS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Demonstrating Bad Initialization
# -----------------------------------------------------------------------------
print("\n1.1 The Problem with Bad Initialization")
print("-" * 40)


class DeepNetwork(nn.Module):
    """Deep network for demonstrating initialization effects."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, init_std: float):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Custom initialization
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=init_std)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            activations.append(x)
        return x, activations


# Test different initializations
input_dim, hidden_dim, n_layers = 100, 100, 10
x = torch.randn(32, input_dim)

print(f"Network: {n_layers} layers, {hidden_dim} hidden units")
print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}")

for init_std, name in [(0.01, "Too Small"), (1.0, "Too Large"), (0.1, "Reasonable")]:
    net = DeepNetwork(input_dim, hidden_dim, n_layers, init_std)
    _, activations = net(x)

    final_std = activations[-1].std().item()
    final_mean = activations[-1].mean().item()

    print(f"\n{name} (std={init_std}):")
    print(f"  Layer-wise std: ", end="")
    for i in [0, 3, 6, 9]:
        print(f"L{i}:{activations[i+1].std():.4f} ", end="")
    print()
    print(f"  Final: mean={final_mean:.6f}, std={final_std:.6f}")


# -----------------------------------------------------------------------------
# 1.2 Gradient Analysis
# -----------------------------------------------------------------------------
print("\n\n1.2 Gradient Flow Analysis")
print("-" * 40)


def analyze_gradients(model: nn.Module, x: torch.Tensor) -> dict:
    """Analyze gradient magnitudes through the network."""
    model.zero_grad()

    # Forward
    output, _ = model(x)
    loss = output.sum()

    # Backward
    loss.backward()

    # Collect gradient stats
    grad_stats = {}
    for i, layer in enumerate(model.layers):
        grad = layer.weight.grad
        grad_stats[f"layer_{i}"] = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.abs().max().item()
        }

    return grad_stats


# Compare gradient flow
for init_std, name in [(0.01, "Too Small"), (1.0, "Too Large")]:
    net = DeepNetwork(input_dim, hidden_dim, n_layers, init_std)
    stats = analyze_gradients(net, x)

    print(f"\n{name} (std={init_std}) - Gradient std per layer:")
    for layer_name, s in stats.items():
        print(f"  {layer_name}: std={s['std']:.2e}")


# =============================================================================
# SECTION 2: VARIANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: VARIANCE ANALYSIS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 Forward Pass Variance
# -----------------------------------------------------------------------------
print("\n2.1 Forward Pass Variance Analysis")
print("-" * 40)


def forward_variance_analysis(n_in: int, n_out: int, std: float, activation: str = 'linear') -> dict:
    """
    Analyze variance propagation through a single layer.

    For y = Wx where w ~ N(0, std²) and x ~ N(0, 1):
    Var[y] = n_in × std² × Var[x] = n_in × std²
    """
    n_samples = 10000

    # Input
    x = torch.randn(n_samples, n_in)

    # Weight matrix
    W = torch.randn(n_out, n_in) * std

    # Linear transformation
    y = x @ W.T

    # Apply activation
    if activation == 'relu':
        y = F.relu(y)
    elif activation == 'tanh':
        y = torch.tanh(y)

    return {
        'input_var': x.var().item(),
        'output_var': y.var().item(),
        'theoretical_var': n_in * std ** 2,
        'ratio': y.var().item() / x.var().item()
    }


n_in, n_out = 512, 512

print(f"Layer: {n_in} → {n_out}")
print(f"\nLinear activation:")
for std in [0.01, 0.044, 0.1, 1.0]:  # 0.044 ≈ 1/√512
    stats = forward_variance_analysis(n_in, n_out, std, 'linear')
    print(f"  std={std:.3f}: Var[out]={stats['output_var']:.4f}, "
          f"ratio={stats['ratio']:.4f}, "
          f"theoretical={stats['theoretical_var']:.4f}")

print(f"\nWith ReLU:")
for std in [0.01, 0.063, 0.1, 1.0]:  # 0.063 ≈ √(2/512)
    stats = forward_variance_analysis(n_in, n_out, std, 'relu')
    print(f"  std={std:.3f}: Var[out]={stats['output_var']:.4f}, ratio={stats['ratio']:.4f}")


# -----------------------------------------------------------------------------
# 2.2 Deriving Correct Initialization
# -----------------------------------------------------------------------------
print("\n\n2.2 Deriving Correct Initialization")
print("-" * 40)

print("""
FORWARD PASS ANALYSIS:
----------------------
For a linear layer: y = Σᵢ wᵢxᵢ

Assumptions:
- w ~ N(0, σ²), x ~ N(0, 1)
- w and x are independent

Variance:
  Var[y] = Var[Σᵢ wᵢxᵢ]
         = Σᵢ E[wᵢ²]E[xᵢ²]     (independence, zero mean)
         = Σᵢ σ² × 1
         = n_in × σ²

To preserve variance (Var[y] = Var[x] = 1):
  n_in × σ² = 1
  σ = 1/√n_in

For RELU (halves variance):
  σ = √(2/n_in)  (He initialization)

For Tanh/Sigmoid (compromise forward/backward):
  σ = √(2/(n_in + n_out))  (Xavier initialization)
""")


# =============================================================================
# SECTION 3: CLASSIC INITIALIZATION METHODS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: CLASSIC INITIALIZATION METHODS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Xavier/Glorot Initialization
# -----------------------------------------------------------------------------
print("\n3.1 Xavier/Glorot Initialization")
print("-" * 40)


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Xavier normal initialization.

    W ~ N(0, √(2/(n_in + n_out)))
    """
    n_in, n_out = tensor.shape[-2], tensor.shape[-1]
    std = gain * math.sqrt(2.0 / (n_in + n_out))
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Xavier uniform initialization.

    W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
    """
    n_in, n_out = tensor.shape[-2], tensor.shape[-1]
    a = gain * math.sqrt(6.0 / (n_in + n_out))
    with torch.no_grad():
        tensor.uniform_(-a, a)
    return tensor


# Demonstrate Xavier
n_in, n_out = 512, 256
W = torch.empty(n_out, n_in)

xavier_normal_(W)
print(f"Xavier Normal (n_in={n_in}, n_out={n_out}):")
print(f"  Expected std: {math.sqrt(2/(n_in + n_out)):.6f}")
print(f"  Actual std:   {W.std():.6f}")

xavier_uniform_(W)
expected_std_uniform = math.sqrt(2/(n_in + n_out))  # For uniform, std = a/√3
actual_std = W.std()
print(f"\nXavier Uniform:")
print(f"  Actual std:   {actual_std:.6f}")
print(f"  Range: [{W.min():.4f}, {W.max():.4f}]")


# Verify variance preservation with tanh
print("\nVariance preservation test (Xavier + Tanh):")


class XavierTanhNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        for layer in self.layers:
            xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


dims = [256, 256, 256, 256, 256]
net = XavierTanhNet(dims)
x = torch.randn(1000, 256)

with torch.no_grad():
    acts = [x]
    for layer in net.layers:
        x = torch.tanh(layer(x))
        acts.append(x)

print("  Layer-wise variance:")
for i, act in enumerate(acts):
    print(f"    Layer {i}: var={act.var():.4f}")


# -----------------------------------------------------------------------------
# 3.2 He/Kaiming Initialization
# -----------------------------------------------------------------------------
print("\n\n3.2 He/Kaiming Initialization (for ReLU)")
print("-" * 40)


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,  # Negative slope for leaky ReLU
    mode: str = 'fan_in',
    nonlinearity: str = 'relu'
) -> torch.Tensor:
    """
    Kaiming (He) normal initialization.

    W ~ N(0, √(2/n_in)) for ReLU
    W ~ N(0, √(2/((1+α²)×n_in))) for Leaky ReLU
    """
    n_in, n_out = tensor.shape[-2], tensor.shape[-1]

    fan = n_in if mode == 'fan_in' else n_out

    # Calculate gain based on nonlinearity
    if nonlinearity == 'relu':
        gain = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = math.sqrt(2.0 / (1 + a ** 2))
    else:
        gain = 1.0

    std = gain / math.sqrt(fan)

    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'relu'
) -> torch.Tensor:
    """Kaiming uniform initialization."""
    n_in, n_out = tensor.shape[-2], tensor.shape[-1]
    fan = n_in if mode == 'fan_in' else n_out

    if nonlinearity == 'relu':
        gain = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = math.sqrt(2.0 / (1 + a ** 2))
    else:
        gain = 1.0

    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


# Demonstrate He initialization
W = torch.empty(n_out, n_in)
kaiming_normal_(W, mode='fan_in', nonlinearity='relu')

print(f"He Normal (n_in={n_in}, ReLU):")
print(f"  Expected std: {math.sqrt(2/n_in):.6f}")
print(f"  Actual std:   {W.std():.6f}")


# Verify variance preservation with ReLU
print("\nVariance preservation test (He + ReLU):")


class HeReLUNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        for layer in self.layers:
            kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


net = HeReLUNet(dims)
x = torch.randn(1000, 256)

with torch.no_grad():
    acts = [x]
    for layer in net.layers:
        x = F.relu(layer(x))
        acts.append(x)

print("  Layer-wise variance:")
for i, act in enumerate(acts):
    print(f"    Layer {i}: var={act.var():.4f}")


# Compare Xavier vs He for ReLU
print("\n\nComparison: Xavier vs He with ReLU (10 layers)")
print("-" * 40)


class ComparisonNet(nn.Module):
    def __init__(self, dims, init_method):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        for layer in self.layers:
            if init_method == 'xavier':
                xavier_normal_(layer.weight)
            else:
                kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


dims_deep = [256] * 11  # 10 layers
x = torch.randn(1000, 256)

for method in ['xavier', 'he']:
    net = ComparisonNet(dims_deep, method)
    with torch.no_grad():
        output = net(x)
    print(f"{method.capitalize():>8}: output var = {output.var():.6f}, mean = {output.mean():.6f}")

print("\nHe maintains variance much better with ReLU!")


# =============================================================================
# SECTION 4: MODERN TRANSFORMER INITIALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: MODERN TRANSFORMER INITIALIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 GPT-2 Style Initialization
# -----------------------------------------------------------------------------
print("\n4.1 GPT-2 Style Initialization")
print("-" * 40)


class GPT2Initialization:
    """
    GPT-2 initialization strategy.

    - All weights: N(0, 0.02)
    - Output projections: N(0, 0.02/√(2*n_layers))
    """

    def __init__(self, n_layers: int, std: float = 0.02):
        self.n_layers = n_layers
        self.std = std

    def init_weights(self, module: nn.Module, is_output_proj: bool = False):
        if isinstance(module, nn.Linear):
            if is_output_proj:
                # Scale by sqrt(2*n_layers) for output projections
                std = self.std / math.sqrt(2 * self.n_layers)
            else:
                std = self.std

            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.std)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


# Demonstrate GPT-2 initialization
print("GPT-2 initialization (12 layers):")
init = GPT2Initialization(n_layers=12, std=0.02)

linear = nn.Linear(768, 768)
init.init_weights(linear, is_output_proj=False)
print(f"  Regular linear std: {linear.weight.std():.4f}")

output_proj = nn.Linear(768, 768)
init.init_weights(output_proj, is_output_proj=True)
print(f"  Output projection std: {output_proj.weight.std():.4f}")
print(f"  Expected output std: {0.02 / math.sqrt(24):.4f}")


# -----------------------------------------------------------------------------
# 4.2 LLaMA Style Initialization
# -----------------------------------------------------------------------------
print("\n\n4.2 LLaMA Style Initialization")
print("-" * 40)


class LLaMAInitialization:
    """
    LLaMA-style initialization.

    - Embeddings: N(0, 1)
    - Linear layers: N(0, 1/√n_in)
    - Output projections: N(0, 1/(√n_in × √(2*n_layers)))
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers

    def init_weights(self, module: nn.Module, is_output_proj: bool = False):
        if isinstance(module, nn.Linear):
            n_in = module.weight.shape[1]
            std = 1.0 / math.sqrt(n_in)

            if is_output_proj:
                std = std / math.sqrt(2 * self.n_layers)

            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1.0)


# Demonstrate LLaMA initialization
print("LLaMA initialization (32 layers, d_model=4096):")
init = LLaMAInitialization(n_layers=32)

linear = nn.Linear(4096, 4096)
init.init_weights(linear, is_output_proj=False)
print(f"  Regular linear std: {linear.weight.std():.6f}")
print(f"  Expected: {1/math.sqrt(4096):.6f}")

output_proj = nn.Linear(4096, 4096)
init.init_weights(output_proj, is_output_proj=True)
print(f"  Output projection std: {output_proj.weight.std():.6f}")
print(f"  Expected: {1/math.sqrt(4096)/math.sqrt(64):.6f}")


# -----------------------------------------------------------------------------
# 4.3 Complete Transformer Initialization
# -----------------------------------------------------------------------------
print("\n\n4.3 Complete Transformer Block Initialization")
print("-" * 40)


class TransformerBlock(nn.Module):
    """Transformer block with proper initialization."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, layer_idx: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # Output projection

        # FFN (SwiGLU style)
        self.ln2 = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # Output projection

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following LLaMA/GPT conventions."""
        std = 1.0 / math.sqrt(self.d_model)
        output_std = std / math.sqrt(2 * self.n_layers)

        # Q, K, V projections: standard init
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=std)

        # Output projection: scaled init
        nn.init.normal_(self.o_proj.weight, std=output_std)

        # FFN: standard init for gate and up, scaled for down
        for layer in [self.w_gate, self.w_up]:
            nn.init.normal_(layer.weight, std=std)

        nn.init.normal_(self.w_down.weight, std=1.0 / math.sqrt(self.w_down.weight.shape[1]) / math.sqrt(2 * self.n_layers))

        # LayerNorm
        nn.init.ones_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)
        nn.init.ones_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)

    def forward(self, x):
        # Simplified forward (no actual attention)
        h = self.ln1(x)
        attn = self.o_proj(h)  # Simplified
        x = x + attn

        h = self.ln2(x)
        ffn = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        x = x + ffn

        return x


# Create and inspect initialization
block = TransformerBlock(d_model=512, n_heads=8, d_ff=1024, n_layers=12, layer_idx=0)

print("Transformer block weight statistics:")
for name, param in block.named_parameters():
    if 'weight' in name and param.dim() > 1:
        print(f"  {name}: std={param.std():.6f}")


# =============================================================================
# SECTION 5: SPECIAL INITIALIZATION CASES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: SPECIAL INITIALIZATION CASES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Embedding Initialization
# -----------------------------------------------------------------------------
print("\n5.1 Embedding Initialization")
print("-" * 40)


def init_embeddings(embedding: nn.Embedding, method: str = 'normal'):
    """Initialize embedding layer."""
    if method == 'normal':
        # Standard: N(0, 1)
        nn.init.normal_(embedding.weight, std=1.0)
    elif method == 'scaled_normal':
        # Scaled by embedding dim: N(0, 1/√d)
        nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(embedding.embedding_dim))
    elif method == 'uniform':
        # Uniform in small range
        nn.init.uniform_(embedding.weight, -0.1, 0.1)


vocab_size, embed_dim = 50000, 768

for method in ['normal', 'scaled_normal', 'uniform']:
    emb = nn.Embedding(vocab_size, embed_dim)
    init_embeddings(emb, method)
    print(f"  {method}: std={emb.weight.std():.4f}, "
          f"norm per token={emb.weight.norm(dim=1).mean():.4f}")


# -----------------------------------------------------------------------------
# 5.2 LSTM Initialization
# -----------------------------------------------------------------------------
print("\n\n5.2 LSTM Initialization (Forget Gate Bias)")
print("-" * 40)


def init_lstm(lstm: nn.LSTM, forget_bias: float = 1.0):
    """
    Initialize LSTM with forget gate bias.

    The forget gate bias is crucial for learning long-term dependencies.
    Setting it to 1.0 means the gate starts open (remembering).
    """
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            # Input-to-hidden weights
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-to-hidden weights (orthogonal helps)
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # Biases
            nn.init.zeros_(param)
            # Set forget gate bias
            # LSTM bias is ordered: [input, forget, cell, output]
            hidden_size = lstm.hidden_size
            param.data[hidden_size:2*hidden_size].fill_(forget_bias)


lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
init_lstm(lstm, forget_bias=1.0)

print("LSTM initialization:")
for name, param in lstm.named_parameters():
    if 'bias' in name:
        # Show forget gate bias (indices hidden_size to 2*hidden_size)
        h = lstm.hidden_size
        fg_bias = param.data[h:2*h].mean().item()
        print(f"  {name}: forget gate bias mean = {fg_bias:.2f}")


# -----------------------------------------------------------------------------
# 5.3 Residual Network Initialization
# -----------------------------------------------------------------------------
print("\n\n5.3 ResNet Initialization (Zero Init)")
print("-" * 40)


class ResidualBlock(nn.Module):
    """Residual block with zero initialization for the last layer."""

    def __init__(self, dim: int, use_zero_init: bool = True):
        super().__init__()
        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Linear(dim, dim)
        self.bn = nn.LayerNorm(dim)

        # Standard He initialization
        kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        if use_zero_init:
            # Zero initialize the last layer
            # This makes residual = 0 at init, so block is identity
            nn.init.zeros_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        residual = self.conv2(F.relu(self.conv1(x)))
        residual = self.bn(residual)
        return x + residual


# Compare with and without zero init
print("Testing residual block initialization:")
x = torch.randn(32, 256)

for use_zero_init in [False, True]:
    block = ResidualBlock(256, use_zero_init=use_zero_init)
    with torch.no_grad():
        out = block(x)

    # Check if output ≈ input (identity function)
    diff = (out - x).abs().mean().item()
    print(f"  Zero init = {use_zero_init}: |output - input| = {diff:.6f}")


# =============================================================================
# SECTION 6: DEBUGGING INITIALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: DEBUGGING INITIALIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Activation Statistics Monitor
# -----------------------------------------------------------------------------
print("\n6.1 Monitoring Activation Statistics")
print("-" * 40)


class ActivationMonitor:
    """Monitor activation statistics through a network."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.hooks = []

        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'dead_frac': (output == 0).float().mean().item()
            }
        return hook

    def analyze(self, x: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return self.activations

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


# Example usage
class SampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Test with bad vs good initialization
print("Bad initialization (std=0.01):")
net_bad = SampleNet()
for layer in [net_bad.fc1, net_bad.fc2, net_bad.fc3, net_bad.fc4]:
    nn.init.normal_(layer.weight, std=0.01)

monitor = ActivationMonitor(net_bad)
stats = monitor.analyze(torch.randn(100, 256))
for name, s in stats.items():
    print(f"  {name}: mean={s['mean']:.4f}, std={s['std']:.4f}, dead={s['dead_frac']:.2%}")
monitor.remove_hooks()

print("\nGood initialization (He):")
net_good = SampleNet()
for layer in [net_good.fc1, net_good.fc2, net_good.fc3, net_good.fc4]:
    kaiming_normal_(layer.weight, nonlinearity='relu')

monitor = ActivationMonitor(net_good)
stats = monitor.analyze(torch.randn(100, 256))
for name, s in stats.items():
    print(f"  {name}: mean={s['mean']:.4f}, std={s['std']:.4f}, dead={s['dead_frac']:.2%}")
monitor.remove_hooks()


# -----------------------------------------------------------------------------
# 6.2 Gradient Statistics Monitor
# -----------------------------------------------------------------------------
print("\n\n6.2 Monitoring Gradient Statistics")
print("-" * 40)


def check_gradient_flow(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Check gradient magnitudes through the network."""
    model.train()
    model.zero_grad()

    # Forward + backward
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()

    # Collect gradient stats
    print("Gradient flow analysis:")
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad = param.grad
            print(f"  {name}: "
                  f"grad_mean={grad.mean():.2e}, "
                  f"grad_std={grad.std():.2e}, "
                  f"grad_max={grad.abs().max():.2e}")


# Test gradient flow
net = SampleNet()
for layer in [net.fc1, net.fc2, net.fc3, net.fc4]:
    kaiming_normal_(layer.weight, nonlinearity='relu')

x = torch.randn(32, 256)
y = torch.randint(0, 10, (32,))
check_gradient_flow(net, x, y)


# =============================================================================
# SECTION 7: COMPLETE INITIALIZATION LIBRARY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: COMPLETE INITIALIZATION LIBRARY")
print("=" * 70)


class InitializationLibrary:
    """
    Comprehensive initialization library.
    """

    # Xavier/Glorot
    @staticmethod
    def xavier_uniform(tensor, gain=1.0):
        return nn.init.xavier_uniform_(tensor, gain=gain)

    @staticmethod
    def xavier_normal(tensor, gain=1.0):
        return nn.init.xavier_normal_(tensor, gain=gain)

    # Kaiming/He
    @staticmethod
    def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='relu'):
        return nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

    @staticmethod
    def kaiming_normal(tensor, a=0, mode='fan_in', nonlinearity='relu'):
        return nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)

    # Orthogonal (good for RNNs)
    @staticmethod
    def orthogonal(tensor, gain=1.0):
        return nn.init.orthogonal_(tensor, gain=gain)

    # Sparse
    @staticmethod
    def sparse(tensor, sparsity, std=0.01):
        return nn.init.sparse_(tensor, sparsity=sparsity, std=std)

    # Transformer-specific
    @staticmethod
    def transformer_linear(tensor, d_model, is_output_proj=False, n_layers=1):
        std = 1.0 / math.sqrt(d_model)
        if is_output_proj:
            std = std / math.sqrt(2 * n_layers)
        return nn.init.normal_(tensor, std=std)


# Demonstrate library
print("\nInitialization Library Examples:")
W = torch.empty(256, 512)

InitializationLibrary.xavier_normal(W)
print(f"Xavier Normal: std={W.std():.4f}")

InitializationLibrary.kaiming_normal(W)
print(f"Kaiming Normal: std={W.std():.4f}")

InitializationLibrary.orthogonal(W)
print(f"Orthogonal: std={W.std():.4f}, orthogonality check: {(W @ W.T - torch.eye(256)).abs().max():.6f}")

InitializationLibrary.transformer_linear(W, d_model=512, is_output_proj=False)
print(f"Transformer (d=512): std={W.std():.4f}")

InitializationLibrary.transformer_linear(W, d_model=512, is_output_proj=True, n_layers=12)
print(f"Transformer output (d=512, L=12): std={W.std():.4f}")


# =============================================================================
# SECTION 8: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. Why does initialization matter?
   - Bad init → vanishing/exploding gradients
   - Goal: preserve variance through layers
   - Symmetry breaking (not all zeros)

2. Xavier vs He:
   - Xavier: σ² = 2/(n_in + n_out) → tanh/sigmoid
   - He: σ² = 2/n_in → ReLU (compensates for halving)

3. Transformer initialization:
   - All weights: N(0, 0.02) or N(0, 1/√d)
   - Output projections: scaled by 1/√(2L)
   - Prevents variance explosion in residual stream

4. LSTM forget gate:
   - Initialize bias to 1.0
   - Keeps gate open initially (remembers)
   - Critical for long-term dependencies

5. Debugging:
   - Monitor activation statistics (mean, std)
   - Check for dead neurons (ReLU)
   - Verify gradient magnitudes
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 2.4 SUMMARY: WEIGHT INITIALIZATION")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────┬─────────────────────────┐
│ Method          │ Formula                │ Use Case                │
├─────────────────┼────────────────────────┼─────────────────────────┤
│ Xavier Normal   │ N(0, √(2/(n_in+n_out)))│ Tanh, Sigmoid           │
│ Xavier Uniform  │ U(-√6/(n+m), √6/(n+m)) │ Tanh, Sigmoid           │
│ He Normal       │ N(0, √(2/n_in))        │ ReLU                    │
│ He Uniform      │ U(-√6/n_in, √6/n_in)   │ ReLU                    │
│ Transformer     │ N(0, 0.02)             │ Transformers            │
│ Output Proj     │ N(0, σ/√(2L))          │ Residual outputs        │
│ Orthogonal      │ QR decomposition       │ RNNs                    │
└─────────────────┴────────────────────────┴─────────────────────────┘

KEY TAKEAWAYS:
1. Goal: Preserve variance through forward and backward passes
2. Xavier for tanh/sigmoid, He for ReLU
3. Transformers need scaled output projections
4. LSTM forget gate bias = 1.0 for long sequences
5. Always monitor activations during initial training

MODERN LLM INITIALIZATION:
  Embeddings → N(0, 1) or N(0, 1/√d)
  Linear → N(0, 1/√n_in)
  Output Projections → N(0, 1/(√n_in × √(2L)))
  LayerNorm → weight=1, bias=0
""")

print("\nModule 2.4 Complete!")
