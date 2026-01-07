"""
Module 7B: Positional Encodings Deep Dive
=========================================

This module provides hands-on implementations of:
1. Sinusoidal Positional Encoding (Original Transformer)
2. Learned Positional Embeddings (BERT, GPT-2)
3. Rotary Position Embeddings (RoPE) - LLaMA style
4. ALiBi (Attention with Linear Biases)
5. Relative Positional Encodings (T5 style)
6. Context Length Extension techniques

Run this file to see all positional encoding methods in action!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# 1. SINUSOIDAL POSITIONAL ENCODING
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - No learned parameters
    - Can extrapolate to any sequence length
    - Relative position via linear transformation
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            x + positional_encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for visualization."""
        return self.pe[:, :seq_len, :].squeeze(0)


# =============================================================================
# 2. LEARNED POSITIONAL EMBEDDINGS
# =============================================================================

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (BERT, GPT-2 style).

    Simple embedding table lookup for positions.

    Properties:
    - Learned parameters (max_len × d_model)
    - Cannot extrapolate beyond max_len
    - Task-specific patterns
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        # Register position indices as buffer
        self.register_buffer(
            'position_ids',
            torch.arange(max_len).expand((1, -1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            x + positional_embedding
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.embedding(position_ids)

        x = x + position_embeddings
        return self.dropout(x)


# =============================================================================
# 3. ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) as used in LLaMA.

    Key idea: Rotate Q and K vectors by position-dependent angles.
    The dot product then depends only on relative position!

    q_m · k_n = qᵀ R_{n-m} k

    Properties:
    - No additional parameters
    - Natural relative position encoding
    - Good extrapolation (with interpolation)
    - Applied to Q and K only (not V)
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for all positions
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len: int):
        """Precompute cos and sin values for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)

        # Duplicate for pairs: [θ₀, θ₀, θ₁, θ₁, ...]
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to Q and K.

        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)
            seq_len: Current sequence length

        Returns:
            Rotated q and k
        """
        # Get cached cos/sin for current sequence length
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation using the formula:
        x_rot = x * cos + rotate_half(x) * sin
        """
        # Rotate half: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1)
        x_rot = x_rot.reshape(x.shape)

        return x * cos + x_rot * sin


def apply_rope_simple(q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    """
    Simplified RoPE application for educational purposes.

    This shows the rotation more explicitly.
    """
    # q, k: (batch, heads, seq_len, head_dim)
    # freqs_cos, freqs_sin: (seq_len, head_dim/2)

    batch, heads, seq_len, head_dim = q.shape

    # Split into pairs: reshape to (..., head_dim/2, 2)
    q_pairs = q.reshape(batch, heads, seq_len, head_dim // 2, 2)
    k_pairs = k.reshape(batch, heads, seq_len, head_dim // 2, 2)

    # Expand freqs for broadcasting
    cos = freqs_cos.view(1, 1, seq_len, head_dim // 2, 1)
    sin = freqs_sin.view(1, 1, seq_len, head_dim // 2, 1)

    # Apply rotation to each pair
    # [x0, x1] @ [[cos, -sin], [sin, cos]] = [x0*cos - x1*sin, x0*sin + x1*cos]
    q_rotated = torch.stack([
        q_pairs[..., 0] * cos.squeeze(-1) - q_pairs[..., 1] * sin.squeeze(-1),
        q_pairs[..., 0] * sin.squeeze(-1) + q_pairs[..., 1] * cos.squeeze(-1)
    ], dim=-1)

    k_rotated = torch.stack([
        k_pairs[..., 0] * cos.squeeze(-1) - k_pairs[..., 1] * sin.squeeze(-1),
        k_pairs[..., 0] * sin.squeeze(-1) + k_pairs[..., 1] * cos.squeeze(-1)
    ], dim=-1)

    # Reshape back
    q_out = q_rotated.reshape(batch, heads, seq_len, head_dim)
    k_out = k_rotated.reshape(batch, heads, seq_len, head_dim)

    return q_out, k_out


# =============================================================================
# 4. ALiBi (ATTENTION WITH LINEAR BIASES)
# =============================================================================

class ALiBiPositionalEncoding(nn.Module):
    """
    ALiBi: Attention with Linear Biases.

    No position encoding! Instead, add a linear penalty to attention scores
    based on distance between tokens.

    attention = softmax(Q @ K^T / sqrt(d) + bias)
    bias[i,j] = -m × |i - j|

    Properties:
    - No additional parameters
    - Excellent extrapolation to longer sequences
    - Applied to attention scores (not embeddings)
    - Different slopes per head
    """

    def __init__(self, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads

        # Compute slopes for each head (geometric sequence)
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', torch.tensor(slopes).view(n_heads, 1, 1))

        # Precompute bias matrix
        self._precompute_bias(max_seq_len)

    def _get_slopes(self, n_heads: int):
        """
        Get slopes for each attention head.

        Geometric sequence from 2^(-8/n) to 2^(-8)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            # Handle non-power-of-2 number of heads
            closest_power = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power)
            extra_slopes = get_slopes_power_of_2(2 * closest_power)[0::2][:n_heads - closest_power]
            return slopes + extra_slopes

    def _precompute_bias(self, seq_len: int):
        """Precompute the ALiBi bias matrix."""
        # Distance matrix: |i - j|
        positions = torch.arange(seq_len)
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # Bias = -slope × distance
        # Shape: (n_heads, seq_len, seq_len)
        bias = -distance.float().unsqueeze(0) * self.slopes

        self.register_buffer('bias', bias, persistent=False)

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            attention_scores: (batch, n_heads, seq_len, seq_len)

        Returns:
            attention_scores + alibi_bias
        """
        seq_len = attention_scores.size(-1)

        # Get bias for current sequence length
        bias = self.bias[:, :seq_len, :seq_len]

        return attention_scores + bias

    def get_bias(self, seq_len: int) -> torch.Tensor:
        """Get ALiBi bias for visualization."""
        return self.bias[:, :seq_len, :seq_len]


# =============================================================================
# 5. RELATIVE POSITIONAL ENCODING (T5-style)
# =============================================================================

class T5RelativePositionalBias(nn.Module):
    """
    T5-style relative positional bias with bucketing.

    Learns a bias for each relative position, with logarithmic bucketing
    for distant positions.

    Properties:
    - Learned biases
    - Bucketing reduces parameters for long-range
    - Added to attention scores
    """

    def __init__(
        self,
        n_heads: int,
        n_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Learned relative position embeddings
        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Map relative positions to buckets.

        Nearby positions: fine-grained (one bucket each)
        Far positions: coarse-grained (logarithmic bucketing)
        """
        relative_buckets = 0

        if self.bidirectional:
            num_buckets = self.n_buckets // 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            num_buckets = self.n_buckets

        # Half buckets for exact positions (linear)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Other half for logarithmic bucketing
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position bias.

        Args:
            seq_len: Sequence length

        Returns:
            bias: (1, n_heads, seq_len, seq_len)
        """
        # Create position matrix
        positions = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Map to buckets
        relative_buckets = self._relative_position_bucket(relative_positions)

        # Look up biases
        values = self.relative_attention_bias(relative_buckets)  # (seq_len, seq_len, n_heads)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, seq_len, seq_len)

        return values


# =============================================================================
# 6. CONTEXT LENGTH EXTENSION
# =============================================================================

class RoPEWithPositionInterpolation(RotaryPositionalEmbedding):
    """
    RoPE with position interpolation for extending context length.

    If trained on max_len=2048, can extend to longer sequences by
    scaling positions: pos_new = pos × (train_len / target_len)
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000,
                 original_max_len: int = 2048):
        self.original_max_len = original_max_len
        super().__init__(dim, max_seq_len, base)

    def _precompute_freqs(self, seq_len: int):
        """Precompute with position interpolation."""
        # Scale positions if beyond original training length
        if seq_len > self.original_max_len:
            scale = self.original_max_len / seq_len
        else:
            scale = 1.0

        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t = t * scale  # Position interpolation!

        freqs = torch.outer(t, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)


class RoPEWithNTKScaling(RotaryPositionalEmbedding):
    """
    RoPE with NTK-aware scaling for better context extension.

    Instead of scaling positions, scale the frequency base.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000,
                 original_max_len: int = 2048):
        self.original_max_len = original_max_len
        super().__init__(dim, max_seq_len, base)

    def _precompute_freqs(self, seq_len: int):
        """Precompute with NTK-aware scaling."""
        # Adjust base if extending beyond original length
        if seq_len > self.original_max_len:
            scale = seq_len / self.original_max_len
            # NTK formula
            new_base = self.base * (scale ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            inv_freq = self.inv_freq

        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)


# =============================================================================
# 7. VISUALIZATION UTILITIES
# =============================================================================

def visualize_positional_encodings():
    """Visualize different positional encoding methods."""
    print("\n" + "=" * 70)
    print("VISUALIZING POSITIONAL ENCODINGS")
    print("=" * 70)

    d_model = 128
    seq_len = 64

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sinusoidal
    sin_pe = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
    encoding = sin_pe.get_encoding(seq_len).numpy()

    ax = axes[0, 0]
    im = ax.imshow(encoding.T, aspect='auto', cmap='RdBu')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Sinusoidal Positional Encoding')
    plt.colorbar(im, ax=ax)

    # 2. Learned (random initialization for visualization)
    learned_pe = LearnedPositionalEmbedding(d_model, max_len=seq_len)
    learned_encoding = learned_pe.embedding.weight.detach().numpy()

    ax = axes[0, 1]
    im = ax.imshow(learned_encoding.T, aspect='auto', cmap='RdBu')
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension')
    ax.set_title('Learned Positional Embedding (init)')
    plt.colorbar(im, ax=ax)

    # 3. RoPE frequencies
    rope = RotaryPositionalEmbedding(d_model, max_seq_len=seq_len)
    freqs = rope.cos_cached[:seq_len, :d_model//2].numpy()

    ax = axes[1, 0]
    im = ax.imshow(freqs.T, aspect='auto', cmap='RdBu')
    ax.set_xlabel('Position')
    ax.set_ylabel('Frequency dimension')
    ax.set_title('RoPE cos(mθ) values')
    plt.colorbar(im, ax=ax)

    # 4. ALiBi bias
    alibi = ALiBiPositionalEncoding(n_heads=8, max_seq_len=seq_len)
    bias = alibi.get_bias(seq_len)[0].numpy()  # First head

    ax = axes[1, 1]
    im = ax.imshow(bias, aspect='auto', cmap='viridis')
    ax.set_xlabel('Key position')
    ax.set_ylabel('Query position')
    ax.set_title('ALiBi Bias (head 0)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('positional_encodings_visualization.png', dpi=150)
    print("Saved visualization to 'positional_encodings_visualization.png'")
    plt.close()


def visualize_alibi_slopes():
    """Visualize ALiBi slopes across different heads."""
    print("\n" + "=" * 70)
    print("ALiBi SLOPES VISUALIZATION")
    print("=" * 70)

    n_heads = 8
    seq_len = 32

    alibi = ALiBiPositionalEncoding(n_heads=n_heads, max_seq_len=seq_len)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_heads:
            bias = alibi.get_bias(seq_len)[i].numpy()
            im = ax.imshow(bias, aspect='auto', cmap='viridis')
            ax.set_title(f'Head {i} (slope={alibi.slopes[i].item():.4f})')
            ax.set_xlabel('Key position')
            ax.set_ylabel('Query position')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('alibi_slopes_visualization.png', dpi=150)
    print("Saved to 'alibi_slopes_visualization.png'")
    plt.close()


# =============================================================================
# 8. DEMONSTRATIONS
# =============================================================================

def demo_sinusoidal():
    """Demonstrate sinusoidal positional encoding."""
    print("\n" + "=" * 70)
    print("1. SINUSOIDAL POSITIONAL ENCODING")
    print("=" * 70)

    d_model = 64
    seq_len = 16
    batch_size = 2

    pe = SinusoidalPositionalEncoding(d_model, max_len=1000)

    # Create random embeddings
    x = torch.randn(batch_size, seq_len, d_model)

    # Add positional encoding
    x_with_pos = pe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pos.shape}")

    # Show properties
    encoding = pe.get_encoding(seq_len)
    print(f"\nPositional encoding properties:")
    print(f"  Min value: {encoding.min().item():.4f}")
    print(f"  Max value: {encoding.max().item():.4f}")
    print(f"  Shape: {encoding.shape}")

    # Extrapolation test
    long_x = torch.randn(1, 2000, d_model)
    long_out = pe(long_x)
    print(f"\nExtrapolation test (2000 positions): {long_out.shape}")


def demo_learned():
    """Demonstrate learned positional embeddings."""
    print("\n" + "=" * 70)
    print("2. LEARNED POSITIONAL EMBEDDINGS")
    print("=" * 70)

    d_model = 64
    max_len = 512
    seq_len = 16
    batch_size = 2

    pe = LearnedPositionalEmbedding(d_model, max_len=max_len)

    # Create random embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pos = pe(x)

    print(f"Parameters: {sum(p.numel() for p in pe.parameters()):,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pos.shape}")

    # Show extrapolation limitation
    try:
        long_x = torch.randn(1, max_len + 1, d_model)
        pe(long_x)
    except ValueError as e:
        print(f"\nExtrapolation error (expected): {e}")


def demo_rope():
    """Demonstrate Rotary Position Embeddings."""
    print("\n" + "=" * 70)
    print("3. ROTARY POSITION EMBEDDINGS (RoPE)")
    print("=" * 70)

    batch_size = 2
    n_heads = 4
    seq_len = 16
    head_dim = 32

    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=1024)

    # Create random Q and K
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # Apply RoPE
    q_rot, k_rot = rope(q, k, seq_len)

    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    # Verify relative position property
    print("\nRelative position property:")
    # Q[pos=5] · K[pos=10] should depend only on relative distance (5)
    dot_original = (q[0, 0, 5] * k[0, 0, 10]).sum()
    dot_rotated = (q_rot[0, 0, 5] * k_rot[0, 0, 10]).sum()
    print(f"  Original Q·K at (5,10): {dot_original.item():.4f}")
    print(f"  Rotated Q·K at (5,10): {dot_rotated.item():.4f}")


def demo_alibi():
    """Demonstrate ALiBi."""
    print("\n" + "=" * 70)
    print("4. ALiBi (ATTENTION WITH LINEAR BIASES)")
    print("=" * 70)

    n_heads = 8
    seq_len = 16
    batch_size = 2

    alibi = ALiBiPositionalEncoding(n_heads=n_heads, max_seq_len=1024)

    # Create random attention scores
    attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)

    # Apply ALiBi
    attention_with_alibi = alibi(attention_scores)

    print(f"Attention scores shape: {attention_scores.shape}")
    print(f"With ALiBi shape: {attention_with_alibi.shape}")

    # Show bias effect
    bias = alibi.get_bias(seq_len)
    print(f"\nALiBi bias shape: {bias.shape}")
    print(f"Slopes (per head): {alibi.slopes.squeeze().tolist()}")

    # Show bias at different positions
    print(f"\nBias at distance 1: {bias[0, 0, 1].item():.4f}")
    print(f"Bias at distance 5: {bias[0, 4, 9].item():.4f}")
    print(f"Bias at distance 10: {bias[0, 5, 15].item():.4f}")


def demo_t5_relative():
    """Demonstrate T5-style relative position bias."""
    print("\n" + "=" * 70)
    print("5. T5-STYLE RELATIVE POSITION BIAS")
    print("=" * 70)

    n_heads = 8
    seq_len = 16

    t5_bias = T5RelativePositionalBias(n_heads=n_heads, n_buckets=32, max_distance=128)

    # Get bias
    bias = t5_bias(seq_len)

    print(f"Parameters: {sum(p.numel() for p in t5_bias.parameters()):,}")
    print(f"Bias shape: {bias.shape}")

    # Show bucketing
    print(f"\nBucketing examples:")
    for distance in [0, 1, 5, 10, 50, 100]:
        rel_pos = torch.tensor([distance])
        bucket = t5_bias._relative_position_bucket(rel_pos)
        print(f"  Distance {distance:3d} -> Bucket {bucket.item()}")


def demo_context_extension():
    """Demonstrate context length extension techniques."""
    print("\n" + "=" * 70)
    print("6. CONTEXT LENGTH EXTENSION")
    print("=" * 70)

    head_dim = 32
    original_len = 2048
    extended_len = 8192

    # Standard RoPE
    rope_standard = RotaryPositionalEmbedding(head_dim, max_seq_len=original_len)

    # RoPE with Position Interpolation
    rope_pi = RoPEWithPositionInterpolation(
        head_dim, max_seq_len=extended_len, original_max_len=original_len
    )

    # RoPE with NTK scaling
    rope_ntk = RoPEWithNTKScaling(
        head_dim, max_seq_len=extended_len, original_max_len=original_len
    )

    print(f"Original training length: {original_len}")
    print(f"Extended target length: {extended_len}")

    # Compare frequencies at different positions
    print("\nFrequency comparison at position 4096:")
    print(f"  Standard RoPE: Cannot handle (max={original_len})")
    print(f"  Position Interpolation: Position scaled to {4096 * original_len / extended_len:.0f}")
    print(f"  NTK-aware: Base scaled for smoother extrapolation")


def main():
    """Run all positional encoding demonstrations."""
    print("=" * 70)
    print("MODULE 7B: POSITIONAL ENCODINGS - DEMONSTRATIONS")
    print("=" * 70)

    demo_sinusoidal()
    demo_learned()
    demo_rope()
    demo_alibi()
    demo_t5_relative()
    demo_context_extension()

    # Visualizations (optional - requires matplotlib)
    try:
        visualize_positional_encodings()
        visualize_alibi_slopes()
    except Exception as e:
        print(f"\nVisualization skipped: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Takeaways:

1. SINUSOIDAL (Original Transformer):
   - No parameters, can extrapolate
   - Fixed pattern, relative position via linear transform

2. LEARNED (BERT, GPT-2):
   - Simple embedding lookup
   - Cannot extrapolate beyond max_len
   - Task-specific patterns

3. RoPE (LLaMA, Mistral):
   - Rotate Q, K by position
   - Natural relative position encoding
   - Good extrapolation with PI/NTK
   - Modern standard for LLMs

4. ALiBi (BLOOM, MPT):
   - No position encoding!
   - Add distance-based bias to attention
   - Excellent extrapolation
   - Different slopes per head

5. T5 Relative:
   - Learned relative position biases
   - Logarithmic bucketing for far positions
   - Good for encoder-decoder models

6. Context Extension:
   - Position Interpolation: Scale positions
   - NTK-aware: Scale frequency base
   - YaRN: Per-dimension adjustments
   - Fine-tuning: Most reliable
""")


if __name__ == "__main__":
    main()
