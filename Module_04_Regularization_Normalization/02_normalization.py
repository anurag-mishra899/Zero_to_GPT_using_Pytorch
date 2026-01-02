"""
Module 4.2: Normalization Techniques - BatchNorm to RMSNorm
Complete implementation of normalization methods for deep learning.

This module covers:
1. Batch Normalization
2. Layer Normalization
3. RMS Normalization (used in LLaMA)
4. Group Normalization
5. Pre-Norm vs Post-Norm patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

print("=" * 70)
print("MODULE 4.2: NORMALIZATION TECHNIQUES")
print("=" * 70)

# =============================================================================
# SECTION 1: BATCH NORMALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: BATCH NORMALIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 BatchNorm Implementation
# -----------------------------------------------------------------------------
print("\n1.1 Batch Normalization")
print("-" * 40)


class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 1D inputs (N, C) or (N, C, L).

    Normalizes across batch dimension for each channel.

    Training: Use batch statistics, update running stats
    Eval: Use running statistics
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Learnable parameters
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))  # γ
            self.bias = nn.Parameter(torch.zeros(num_features))   # β
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Running statistics
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed: (N, C, L) -> use same logic
        if x.dim() == 2:
            # (N, C) -> normalize over N
            pass
        elif x.dim() == 3:
            # (N, C, L) -> normalize over N and L
            pass
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1

        if self.training:
            # Compute batch statistics
            if x.dim() == 2:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
            else:  # 3D
                mean = x.mean(dim=(0, 2))
                var = x.var(dim=(0, 2), unbiased=False)

            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        if x.dim() == 2:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:  # 3D
            x_norm = (x - mean.unsqueeze(0).unsqueeze(2)) / torch.sqrt(var.unsqueeze(0).unsqueeze(2) + self.eps)

        # Scale and shift
        if self.affine:
            if x.dim() == 2:
                x_norm = x_norm * self.weight + self.bias
            else:
                x_norm = x_norm * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)

        return x_norm


# Demonstrate BatchNorm
print("BatchNorm1d demonstration:")
bn = BatchNorm1d(num_features=4)
x = torch.randn(8, 4)  # (batch=8, features=4)

print(f"Input shape: {x.shape}")
print(f"Input stats per feature: mean={x.mean(dim=0).tolist()}")

bn.train()
out_train = bn(x)
print(f"Training output stats: mean={out_train.mean(dim=0).detach().tolist()}")

bn.eval()
out_eval = bn(x)
print(f"Eval output stats: mean={out_eval.mean(dim=0).detach().tolist()}")


# =============================================================================
# SECTION 2: LAYER NORMALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: LAYER NORMALIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 LayerNorm Implementation
# -----------------------------------------------------------------------------
print("\n2.1 Layer Normalization")
print("-" * 40)


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes across the last dimension(s) for each sample.

    y = γ × (x - μ) / √(σ² + ε) + β

    where μ and σ are computed across normalized_shape dimensions.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))  # γ
            self.bias = nn.Parameter(torch.zeros(normalized_shape))   # β
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine which dimensions to normalize over
        # For normalized_shape = (D,), normalize over last dim
        # For normalized_shape = (H, W), normalize over last 2 dims
        normalized_dims = tuple(range(-len(self.normalized_shape), 0))

        # Compute mean and variance
        mean = x.mean(dim=normalized_dims, keepdim=True)
        var = x.var(dim=normalized_dims, unbiased=False, keepdim=True)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


# Demonstrate LayerNorm
print("LayerNorm demonstration:")
ln = LayerNorm(normalized_shape=64)
x = torch.randn(4, 10, 64)  # (batch=4, seq=10, features=64)

print(f"Input shape: {x.shape}")
print(f"Input stats (first sample, first position):")
print(f"  mean={x[0, 0].mean().item():.4f}, std={x[0, 0].std().item():.4f}")

out = ln(x)
print(f"Output stats (first sample, first position):")
print(f"  mean={out[0, 0].mean().item():.4f}, std={out[0, 0].std().item():.4f}")

# Verify normalization
print("\nVerification: Each position should have mean≈0, std≈1")
sample_means = out[0].mean(dim=-1)
sample_stds = out[0].std(dim=-1)
print(f"  Position means: min={sample_means.min():.4f}, max={sample_means.max():.4f}")
print(f"  Position stds: min={sample_stds.min():.4f}, max={sample_stds.max():.4f}")


# -----------------------------------------------------------------------------
# 2.2 LayerNorm vs BatchNorm Comparison
# -----------------------------------------------------------------------------
print("\n\n2.2 LayerNorm vs BatchNorm Comparison")
print("-" * 40)


def compare_normalizations():
    """Compare BatchNorm and LayerNorm behavior."""
    batch, seq, features = 4, 10, 64
    x = torch.randn(batch, seq, features)

    # Reshape for BatchNorm: (batch, features, seq)
    x_bn = x.transpose(1, 2)

    bn = nn.BatchNorm1d(features)
    ln = nn.LayerNorm(features)

    # Training mode
    bn.train()
    out_bn = bn(x_bn).transpose(1, 2)
    out_ln = ln(x)

    print("Normalization comparison:")
    print(f"Input shape: {x.shape}")

    print(f"\nBatchNorm:")
    print(f"  Normalizes over: batch and sequence (for each feature)")
    print(f"  Running mean shape: {bn.running_mean.shape}")
    print(f"  Output var across batch: {out_bn[:, 0, 0].var().item():.4f}")

    print(f"\nLayerNorm:")
    print(f"  Normalizes over: features (for each position)")
    print(f"  Weight shape: {ln.weight.shape}")
    print(f"  Output var across features: {out_ln[0, 0].var().item():.4f}")


compare_normalizations()


# =============================================================================
# SECTION 3: RMS NORMALIZATION (LLM STANDARD)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: RMS NORMALIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 RMSNorm Implementation
# -----------------------------------------------------------------------------
print("\n3.1 RMS Normalization (used in LLaMA)")
print("-" * 40)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    y = x × γ / RMS(x)

    where RMS(x) = √(mean(x²))

    Key differences from LayerNorm:
    - No mean centering (no subtraction of mean)
    - No bias term (no β)
    - Faster computation
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # γ only, no β

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return x / rms * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Just the normalization without scaling."""
        return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)


# Alternative implementation (mathematically equivalent)
class RMSNormAlternative(nn.Module):
    """Alternative RMSNorm using rsqrt for efficiency."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using rsqrt (reciprocal square root) - often faster
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


# Demonstrate RMSNorm
print("RMSNorm demonstration:")
rmsnorm = RMSNorm(normalized_shape=64)
x = torch.randn(4, 10, 64)

print(f"Input shape: {x.shape}")
print(f"Input RMS (first position): {torch.sqrt((x[0, 0] ** 2).mean()).item():.4f}")

out = rmsnorm(x)
print(f"Output RMS (first position): {torch.sqrt((out[0, 0] ** 2).mean()).item():.4f}")

# Compare parameters
ln = nn.LayerNorm(64)
print(f"\nParameter comparison:")
print(f"  LayerNorm: {sum(p.numel() for p in ln.parameters())} params (γ and β)")
print(f"  RMSNorm: {sum(p.numel() for p in rmsnorm.parameters())} params (γ only)")


# -----------------------------------------------------------------------------
# 3.2 RMSNorm vs LayerNorm
# -----------------------------------------------------------------------------
print("\n\n3.2 RMSNorm vs LayerNorm Comparison")
print("-" * 40)


def compare_rmsnorm_layernorm():
    """Compare RMSNorm and LayerNorm outputs."""
    x = torch.randn(2, 8, 64)

    ln = nn.LayerNorm(64)
    rn = RMSNorm(64)

    # Make weights same for fair comparison
    with torch.no_grad():
        rn.weight.copy_(ln.weight)

    out_ln = ln(x)
    out_rn = rn(x)

    print("LayerNorm vs RMSNorm (same γ weights):")
    print(f"\nInput (first position):")
    print(f"  mean={x[0, 0].mean().item():.4f}, std={x[0, 0].std().item():.4f}")

    print(f"\nLayerNorm output:")
    print(f"  mean={out_ln[0, 0].mean().item():.4f}, std={out_ln[0, 0].std().item():.4f}")

    print(f"\nRMSNorm output:")
    print(f"  mean={out_rn[0, 0].mean().item():.4f}, std={out_rn[0, 0].std().item():.4f}")

    print(f"\nDifference: {(out_ln - out_rn).abs().mean().item():.6f}")


compare_rmsnorm_layernorm()


# -----------------------------------------------------------------------------
# 3.3 Performance Comparison
# -----------------------------------------------------------------------------
print("\n\n3.3 Performance Comparison")
print("-" * 40)

import time


def benchmark_norms(batch=32, seq=512, dim=1024, iterations=100):
    """Benchmark different normalization methods."""
    x = torch.randn(batch, seq, dim)

    ln = nn.LayerNorm(dim)
    rn = RMSNorm(dim)

    # Warmup
    for _ in range(10):
        _ = ln(x)
        _ = rn(x)

    # LayerNorm timing
    start = time.time()
    for _ in range(iterations):
        _ = ln(x)
    ln_time = time.time() - start

    # RMSNorm timing
    start = time.time()
    for _ in range(iterations):
        _ = rn(x)
    rn_time = time.time() - start

    print(f"Benchmark (batch={batch}, seq={seq}, dim={dim}, iters={iterations}):")
    print(f"  LayerNorm: {ln_time*1000:.2f}ms")
    print(f"  RMSNorm: {rn_time*1000:.2f}ms")
    print(f"  Speedup: {ln_time/rn_time:.2f}x")


benchmark_norms()


# =============================================================================
# SECTION 4: GROUP NORMALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: GROUP NORMALIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 GroupNorm Implementation
# -----------------------------------------------------------------------------
print("\n4.1 Group Normalization")
print("-" * 40)


class GroupNorm(nn.Module):
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group.
    Works well with small batch sizes (unlike BatchNorm).

    For input (N, C, H, W):
      Divides C channels into G groups
      Each group has C/G channels
      Normalizes each group separately
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) or (N, C)
        N, C = x.shape[:2]
        G = self.num_groups

        # Reshape to (N, G, C//G, *)
        x_grouped = x.view(N, G, C // G, -1)

        # Normalize each group
        mean = x_grouped.mean(dim=(2, 3), keepdim=True)
        var = x_grouped.var(dim=(2, 3), unbiased=False, keepdim=True)

        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)

        # Reshape back
        x_norm = x_norm.view_as(x)

        # Scale and shift
        if self.affine:
            if x.dim() == 2:
                x_norm = x_norm * self.weight + self.bias
            else:
                x_norm = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return x_norm


# Demonstrate GroupNorm
print("GroupNorm demonstration:")
gn = GroupNorm(num_groups=4, num_channels=16)
x = torch.randn(2, 16, 8, 8)  # (batch, channels, H, W)

print(f"Input shape: {x.shape}")
print(f"Groups: 4 (each with 4 channels)")

out = gn(x)
print(f"Output shape: {out.shape}")

# Check group statistics
for g in range(4):
    group_out = out[:, g*4:(g+1)*4]
    print(f"  Group {g} mean: {group_out.mean().item():.4f}, std: {group_out.std().item():.4f}")


# =============================================================================
# SECTION 5: PRE-NORM VS POST-NORM
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: PRE-NORM VS POST-NORM")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Pre-Norm Transformer Block
# -----------------------------------------------------------------------------
print("\n5.1 Pre-Norm vs Post-Norm Patterns")
print("-" * 40)


class PostNormBlock(nn.Module):
    """
    Post-Norm: Original Transformer style.

    x = LayerNorm(x + Sublayer(x))

    Used in original "Attention Is All You Need".
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.sublayer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize AFTER residual addition
        return self.norm(x + self.sublayer(x))


class PreNormBlock(nn.Module):
    """
    Pre-Norm: Modern Transformer style.

    x = x + Sublayer(LayerNorm(x))

    Used in GPT-2, LLaMA, and most modern LLMs.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize BEFORE sublayer, then residual
        return x + self.sublayer(self.norm(x))


class PreNormBlockRMS(nn.Module):
    """
    Pre-Norm with RMSNorm: LLaMA style.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.sublayer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # LLaMA uses SiLU
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sublayer(self.norm(x))


# Compare gradient flow
def analyze_gradient_flow():
    """Compare gradient magnitude through blocks."""
    d_model, d_ff = 256, 512
    n_layers = 12

    # Stack multiple blocks
    post_norm_layers = nn.ModuleList([PostNormBlock(d_model, d_ff) for _ in range(n_layers)])
    pre_norm_layers = nn.ModuleList([PreNormBlock(d_model, d_ff) for _ in range(n_layers)])

    x = torch.randn(4, 16, d_model, requires_grad=True)
    x_copy = x.detach().clone().requires_grad_(True)

    # Forward through post-norm
    h_post = x
    for layer in post_norm_layers:
        h_post = layer(h_post)
    loss_post = h_post.sum()
    loss_post.backward()

    # Forward through pre-norm
    h_pre = x_copy
    for layer in pre_norm_layers:
        h_pre = layer(h_pre)
    loss_pre = h_pre.sum()
    loss_pre.backward()

    print("Gradient analysis ({} layers):".format(n_layers))
    print(f"  Post-Norm input gradient norm: {x.grad.norm().item():.4f}")
    print(f"  Pre-Norm input gradient norm: {x_copy.grad.norm().item():.4f}")
    print("\nPre-Norm typically has better gradient flow!")


analyze_gradient_flow()


# =============================================================================
# SECTION 6: COMPLETE TRANSFORMER NORMALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: COMPLETE LLM NORMALIZATION PATTERN")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 LLaMA-style Transformer with RMSNorm
# -----------------------------------------------------------------------------
print("\n6.1 LLaMA-style Normalization Pattern")
print("-" * 40)


class LLaMABlock(nn.Module):
    """
    LLaMA-style transformer block with RMSNorm.

    Pre-norm architecture:
    - RMSNorm before attention
    - RMSNorm before FFN
    - No bias terms
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # Attention
        self.attn_norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN (SwiGLU)
        self.ffn_norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attention(self.attn_norm(x))
        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMAModel(nn.Module):
    """Minimal LLaMA-like model showing normalization pattern."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LLaMABlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)  # Final norm before output
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)  # Final normalization
        logits = self.output(h)

        return logits


# Demonstrate
print("LLaMA-style model normalization pattern:")
model = LLaMAModel(
    vocab_size=1000,
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=512
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nNormalization points:")
print("  - Before each attention layer (RMSNorm)")
print("  - Before each FFN layer (RMSNorm)")
print("  - After all layers, before output (RMSNorm)")


# =============================================================================
# SECTION 7: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. BatchNorm vs LayerNorm:
   - BatchNorm: Across batch (per channel)
   - LayerNorm: Across features (per sample)
   - BN for CNNs, LN for Transformers

2. RMSNorm:
   - y = γ × x / RMS(x)
   - No mean centering, no β
   - 10-20% faster than LayerNorm
   - Used in LLaMA, Gemma, Mistral

3. Pre-Norm vs Post-Norm:
   - Pre-Norm: x = x + Sublayer(Norm(x))
   - Post-Norm: x = Norm(x + Sublayer(x))
   - Pre-Norm has better gradient flow
   - All modern LLMs use Pre-Norm

4. Why LayerNorm for Transformers:
   - Batch-independent
   - Same train/eval behavior
   - Each position normalized independently

5. Implementation Details:
   - eps = 1e-5 or 1e-6
   - Parameters: γ (scale), β (shift for LN)
   - Compute mean/var, normalize, scale, shift
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 4.2 SUMMARY: NORMALIZATION")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬──────────────────────────┬─────────────────────────┐
│ Method          │ Normalize Over           │ Use Case                │
├─────────────────┼──────────────────────────┼─────────────────────────┤
│ BatchNorm       │ Batch, spatial dims      │ CNNs                    │
│ LayerNorm       │ Feature dimension        │ Transformers            │
│ RMSNorm         │ Feature (no mean)        │ LLMs (LLaMA)            │
│ GroupNorm       │ Channel groups           │ Small batch CNNs        │
└─────────────────┴──────────────────────────┴─────────────────────────┘

KEY EQUATIONS:

LayerNorm:
  μ = mean(x, dim=-1)
  σ² = var(x, dim=-1)
  y = γ × (x - μ) / √(σ² + ε) + β

RMSNorm:
  RMS = √(mean(x²))
  y = γ × x / RMS

LLM NORMALIZATION PATTERN:

  Pre-Norm with RMSNorm:
    for layer in layers:
      x = x + Attention(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    x = RMSNorm(x)  # Final norm
    logits = Linear(x)

KEY TAKEAWAYS:
1. LayerNorm for transformers (batch-independent)
2. RMSNorm for LLMs (faster, same quality)
3. Pre-Norm for stable training
4. Always final norm before output
5. No bias terms in modern LLMs
""")

print("\nModule 4.2 Complete!")
