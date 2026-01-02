"""
Module 10.1: Modern LLM Architectures - LLaMA, RoPE, GQA
Complete implementation of modern LLM components

Covers:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation
- RMSNorm
- Complete LLaMA-style architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from dataclasses import dataclass
import math

print("=" * 70)
print("Module 10.1: Modern LLM Architectures - LLaMA, RoPE, GQA")
print("=" * 70)


# ===========================================================================
# Section 1: RMSNorm
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: RMSNorm")
print("=" * 70)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) * gamma
    RMS(x) = sqrt(mean(x^2))

    No mean centering, no beta parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Just the normalization, without scaling."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# Compare RMSNorm vs LayerNorm
print("\n--- Comparing RMSNorm vs LayerNorm ---")

dim = 512
batch_size, seq_len = 4, 128

rmsnorm = RMSNorm(dim)
layernorm = nn.LayerNorm(dim)

x = torch.randn(batch_size, seq_len, dim)

# Forward pass
out_rms = rmsnorm(x)
out_ln = layernorm(x)

print(f"Input shape: {x.shape}")
print(f"RMSNorm output shape: {out_rms.shape}")
print(f"LayerNorm output shape: {out_ln.shape}")

# Parameter count
params_rms = sum(p.numel() for p in rmsnorm.parameters())
params_ln = sum(p.numel() for p in layernorm.parameters())
print(f"\nRMSNorm parameters: {params_rms}")
print(f"LayerNorm parameters: {params_ln}")

# Timing comparison
import time

n_iters = 1000
x_test = torch.randn(8, 256, 512)

start = time.time()
for _ in range(n_iters):
    _ = rmsnorm(x_test)
rms_time = time.time() - start

start = time.time()
for _ in range(n_iters):
    _ = layernorm(x_test)
ln_time = time.time() - start

print(f"\nRMSNorm time: {rms_time:.3f}s")
print(f"LayerNorm time: {ln_time:.3f}s")
print(f"RMSNorm speedup: {ln_time/rms_time:.2f}x")


# ===========================================================================
# Section 2: Rotary Position Embeddings (RoPE)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: Rotary Position Embeddings (RoPE)")
print("=" * 70)


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE.

    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation

    Returns:
        freqs_cis: Complex exponentials (max_seq_len, dim/2)
    """
    # Compute frequencies: theta^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Compute position indices
    t = torch.arange(max_seq_len)

    # Outer product: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis for broadcasting with x."""
    ndim = x.ndim
    assert ndim >= 2
    shape = [1] * (ndim - 2) + [x.shape[-2], x.shape[-1]]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor (batch, seq_len, num_heads, head_dim)
        xk: Key tensor (batch, seq_len, num_heads, head_dim)
        freqs_cis: Precomputed frequencies (seq_len, head_dim/2)

    Returns:
        xq_out, xk_out: Rotated query and key tensors
    """
    # Reshape to complex numbers: (batch, seq, heads, dim) -> (batch, seq, heads, dim/2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex
    xq_c = torch.view_as_complex(xq_r)  # (batch, seq, heads, dim/2)
    xk_c = torch.view_as_complex(xk_r)

    # Reshape freqs for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_c)

    # Apply rotation (complex multiplication)
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings.

        Args:
            q: Query (batch, seq_len, num_heads, head_dim)
            k: Key (batch, seq_len, num_heads, head_dim)
            start_pos: Starting position (for KV-cache)
        """
        seq_len = q.shape[1]
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        return apply_rotary_emb(q, k, freqs_cis)


# Test RoPE
print("\n--- Testing Rotary Position Embeddings ---")

batch_size, seq_len, num_heads, head_dim = 2, 32, 8, 64

rope = RotaryEmbedding(head_dim, max_seq_len=512)

q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn(batch_size, seq_len, num_heads, head_dim)

q_rot, k_rot = rope(q, k)

print(f"Query shape: {q.shape}")
print(f"Key shape: {k.shape}")
print(f"Rotated Q shape: {q_rot.shape}")
print(f"Rotated K shape: {k_rot.shape}")

# Verify relative position property
print("\n--- Verifying Relative Position Property ---")

# Create test vectors at different positions
q_pos0 = torch.randn(1, 1, 1, head_dim)
k_pos0 = torch.randn(1, 1, 1, head_dim)

# Apply RoPE at different positions
freqs = precompute_freqs_cis(head_dim, 100)

q_at_5, k_at_5 = apply_rotary_emb(q_pos0, k_pos0, freqs[5:6])
q_at_10, k_at_15 = apply_rotary_emb(q_pos0, k_pos0, freqs[10:11])
_, k_at_20 = apply_rotary_emb(q_pos0, k_pos0, freqs[15:16])

# Dot products should depend on relative position
dot_5_5 = (q_at_5 * k_at_5).sum()  # Relative: 0
dot_10_15 = (q_at_10 * k_at_20).sum()  # Relative: 5

print(f"Dot product (pos 5, pos 5), relative=0: {dot_5_5:.4f}")
print(f"Dot product (pos 10, pos 15), relative=5: {dot_10_15:.4f}")


# Visualize RoPE frequencies
def visualize_rope():
    """Visualize RoPE frequency patterns."""
    dim = 64
    max_len = 100

    freqs_cis = precompute_freqs_cis(dim, max_len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Real part (cosine)
    real_part = freqs_cis.real.numpy()
    im1 = axes[0].imshow(real_part.T, aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    axes[0].set_title('RoPE Real Part (Cosine)')
    plt.colorbar(im1, ax=axes[0])

    # Imaginary part (sine)
    imag_part = freqs_cis.imag.numpy()
    im2 = axes[1].imshow(imag_part.T, aspect='auto', cmap='RdBu')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Dimension')
    axes[1].set_title('RoPE Imaginary Part (Sine)')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('/Users/anuragmishra/Documents/Zero_to_GPT/Module_10_Modern_LLM_Architectures/rope_visualization.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved RoPE visualization")

visualize_rope()


# ===========================================================================
# Section 3: Grouped Query Attention (GQA)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: Grouped Query Attention (GQA)")
print("=" * 70)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Multiple query heads share the same key and value heads.

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (num_heads must be divisible by this)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        # Projections
        self.wq = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            start_pos: Starting position for KV-cache
            kv_cache: Optional tuple of (cached_keys, cached_values)
            use_cache: Whether to return new KV-cache
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        xq = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        xq, xk = self.rope(xq, xk, start_pos)

        # Handle KV-cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            xk = torch.cat([cached_k, xk], dim=1)
            xv = torch.cat([cached_v, xv], dim=1)

        new_cache = (xk, xv) if use_cache else None

        # Repeat K, V to match number of Q heads
        # (batch, seq, n_kv_heads, head_dim) -> (batch, seq, n_heads, head_dim)
        xk = xk.repeat_interleave(self.num_groups, dim=2)
        xv = xv.repeat_interleave(self.num_groups, dim=2)

        # Transpose for attention: (batch, heads, seq, dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        kv_len = xk.size(2)
        mask = self.mask[:, :, start_pos:start_pos + seq_len, :kv_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, xv)

        # Reshape: (batch, heads, seq, dim) -> (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.wo(output)

        return output, new_cache


# Test GQA
print("\n--- Testing Grouped Query Attention ---")

d_model = 512
num_heads = 8
num_kv_heads = 2  # 4x reduction

gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

x = torch.randn(2, 32, d_model)
output, _ = gqa(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Num Q heads: {num_heads}")
print(f"Num KV heads: {num_kv_heads}")
print(f"KV-cache reduction: {num_heads // num_kv_heads}x")

# Parameter comparison
mha_params = 4 * d_model * d_model  # Q, K, V, O all same size
gqa_params = (d_model * num_heads * (d_model // num_heads) +  # Q
              2 * d_model * num_kv_heads * (d_model // num_heads) +  # K, V
              d_model * d_model)  # O

print(f"\nMHA parameters: {mha_params:,}")
print(f"GQA parameters: {sum(p.numel() for p in gqa.parameters()):,}")


# ===========================================================================
# Section 4: SwiGLU FFN
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: SwiGLU FFN")
print("=" * 70)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.

    SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up)
    """

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * x


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    FFN(x) = W_down(SwiGLU(W_up(x), W_gate(x)))

    Note: hidden_dim is typically 8/3 * d_model to match standard FFN param count
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()

        # Default hidden dim for SwiGLU (8/3 * d_model to match param count)
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
            # Round to multiple of 256 for efficiency
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up
        up = self.w_up(x)
        gate = self.w_gate(x)
        x = F.silu(gate) * up
        x = self.w_down(x)
        x = self.dropout(x)
        return x


# Compare standard FFN vs SwiGLU
print("\n--- Comparing Standard FFN vs SwiGLU ---")

d_model = 512
hidden_dim_std = 4 * d_model  # Standard: 4x expansion
hidden_dim_swiglu = int(8 * d_model / 3)  # SwiGLU: 8/3 x expansion

# Standard FFN
class StandardFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

ffn_std = StandardFFN(d_model, hidden_dim_std)
ffn_swiglu = SwiGLUFFN(d_model)

params_std = sum(p.numel() for p in ffn_std.parameters())
params_swiglu = sum(p.numel() for p in ffn_swiglu.parameters())

print(f"Standard FFN (4x): {params_std:,} parameters")
print(f"SwiGLU FFN (8/3x): {params_swiglu:,} parameters")
print(f"Ratio: {params_swiglu / params_std:.2f}")

# Test forward pass
x = torch.randn(2, 32, d_model)
out_std = ffn_std(x)
out_swiglu = ffn_swiglu(x)

print(f"\nInput: {x.shape}")
print(f"Standard FFN output: {out_std.shape}")
print(f"SwiGLU FFN output: {out_swiglu.shape}")


# ===========================================================================
# Section 5: Complete LLaMA Block
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Complete LLaMA Block")
print("=" * 70)


@dataclass
class LLaMAConfig:
    """Configuration for LLaMA model."""
    vocab_size: int = 32000
    d_model: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # None = same as num_heads (MHA)
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


# Predefined configs
LLAMA_CONFIGS = {
    'llama-7b': LLaMAConfig(
        d_model=4096, num_layers=32, num_heads=32, num_kv_heads=32
    ),
    'llama-13b': LLaMAConfig(
        d_model=5120, num_layers=40, num_heads=40, num_kv_heads=40
    ),
    'llama-70b': LLaMAConfig(
        d_model=8192, num_layers=80, num_heads=64, num_kv_heads=8  # GQA!
    ),
}


class LLaMABlock(nn.Module):
    """
    Single LLaMA transformer block.

    Pre-RMSNorm architecture with GQA and SwiGLU.
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__()

        num_kv_heads = config.num_kv_heads or config.num_heads

        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config.d_model, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Pre-RMSNorm.
        """
        # Attention with residual
        h, new_cache = self.attention(
            self.attention_norm(x),
            start_pos=start_pos,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        x = x + h

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_cache


# Test LLaMA block
print("\n--- Testing LLaMA Block ---")

config = LLaMAConfig(
    d_model=512,
    num_heads=8,
    num_kv_heads=2,  # GQA
    max_seq_len=256
)

block = LLaMABlock(config)

x = torch.randn(2, 32, config.d_model)
output, _ = block(x)

print(f"Config: d_model={config.d_model}, heads={config.num_heads}, kv_heads={config.num_kv_heads}")
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")


# ===========================================================================
# Section 6: Complete LLaMA Model
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: Complete LLaMA Model")
print("=" * 70)


class LLaMA(nn.Module):
    """
    Complete LLaMA model.
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            LLaMABlock(config) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.tok_embeddings.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            start_pos: Starting position for KV-cache
            kv_caches: List of KV-caches for each layer
            use_cache: Whether to return new KV-caches
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings (no position embedding - handled by RoPE)
        h = self.tok_embeddings(input_ids)

        # Process through layers
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, start_pos, kv_cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        # Final norm and output
        h = self.norm(h)
        logits = self.output(h)

        return logits, new_caches

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV-cache."""
        self.eval()

        # Initial forward pass (prefill)
        logits, kv_caches = self(input_ids, start_pos=0, use_cache=True)

        for _ in range(max_new_tokens):
            # Get last logits
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Forward with cache (only new token)
            start_pos = input_ids.size(1) - 1
            logits, kv_caches = self(next_token, start_pos=start_pos, kv_caches=kv_caches, use_cache=True)

        return input_ids


# Test LLaMA model
print("\n--- Testing LLaMA Model ---")

small_config = LLaMAConfig(
    vocab_size=1000,
    d_model=256,
    num_layers=4,
    num_heads=4,
    num_kv_heads=2,  # GQA
    max_seq_len=256
)

model = LLaMA(small_config)
print(f"Model parameters: {model.get_num_params():,}")

input_ids = torch.randint(0, small_config.vocab_size, (2, 32))
logits, _ = model(input_ids)

print(f"Input: {input_ids.shape}")
print(f"Logits: {logits.shape}")

# Test generation
print("\n--- Testing Generation with KV-Cache ---")
start_ids = torch.randint(0, small_config.vocab_size, (1, 5))
generated = model.generate(start_ids, max_new_tokens=20, temperature=0.8, top_k=50)
print(f"Start: {start_ids.shape}")
print(f"Generated: {generated.shape}")


# ===========================================================================
# Section 7: Parameter Count Estimation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 7: Parameter Count Estimation")
print("=" * 70)


def estimate_llama_params(config: LLaMAConfig) -> dict:
    """Estimate parameter counts for LLaMA."""
    d = config.d_model
    L = config.num_layers
    V = config.vocab_size
    n_heads = config.num_heads
    n_kv_heads = config.num_kv_heads or config.num_heads
    head_dim = d // n_heads

    # Hidden dim for SwiGLU (8/3 * d, rounded)
    hidden_dim = int(8 * d / 3)
    hidden_dim = ((hidden_dim + 255) // 256) * 256

    params = {
        'embedding': V * d,
        'attention_q': d * n_heads * head_dim * L,
        'attention_kv': 2 * d * n_kv_heads * head_dim * L,
        'attention_o': n_heads * head_dim * d * L,
        'ffn': 3 * d * hidden_dim * L,  # w_up, w_gate, w_down
        'norms': 2 * d * L + d,  # 2 per layer + final
    }

    params['total'] = sum(params.values())
    return params


print("\nLLaMA Model Parameter Estimates:")
print("-" * 60)

for name, config in LLAMA_CONFIGS.items():
    params = estimate_llama_params(config)
    print(f"\n{name}:")
    print(f"  Total: {params['total'] / 1e9:.2f}B")
    print(f"  Embedding: {params['embedding'] / 1e9:.2f}B")
    print(f"  Attention: {(params['attention_q'] + params['attention_kv'] + params['attention_o']) / 1e9:.2f}B")
    print(f"  FFN: {params['ffn'] / 1e9:.2f}B")


# ===========================================================================
# Section 8: KV-Cache Memory Analysis
# ===========================================================================
print("\n" + "=" * 70)
print("Section 8: KV-Cache Memory Analysis")
print("=" * 70)


def analyze_kv_cache_memory(
    d_model: int,
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2  # FP16
) -> dict:
    """Analyze KV-cache memory requirements."""

    head_dim = d_model // 32  # Assume 32 query heads

    # KV per layer per token: 2 (K, V) * n_kv_heads * head_dim * dtype_bytes
    kv_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes

    # Total per token (all layers)
    kv_per_token = kv_per_token_per_layer * num_layers

    # Total for sequence
    kv_per_seq = kv_per_token * seq_len

    # Total for batch
    total = kv_per_seq * batch_size

    return {
        'kv_per_token_bytes': kv_per_token,
        'kv_per_seq_mb': kv_per_seq / (1024 ** 2),
        'total_mb': total / (1024 ** 2),
        'total_gb': total / (1024 ** 3)
    }


print("\nKV-Cache Memory Analysis:")
print("-" * 70)

configs_analysis = [
    ('LLaMA-7B (MHA)', 4096, 32, 32, 4096),
    ('LLaMA-7B w/GQA-8', 4096, 32, 8, 4096),
    ('LLaMA-70B (GQA-8)', 8192, 80, 8, 4096),
    ('LLaMA-70B (MHA)', 8192, 80, 64, 4096),
]

print(f"{'Model':<25} {'KV Heads':<10} {'Seq Len':<10} {'KV Cache (GB)':<15}")
print("-" * 60)

for name, d_model, num_layers, num_kv_heads, seq_len in configs_analysis:
    mem = analyze_kv_cache_memory(d_model, num_layers, num_kv_heads, seq_len)
    print(f"{name:<25} {num_kv_heads:<10} {seq_len:<10} {mem['total_gb']:.2f}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 10.1 Summary: Modern LLM Architectures")
print("=" * 70)

print("""
Key Takeaways:
==============

1. RMSNorm:
   - Simpler than LayerNorm (no mean centering)
   - ~15% faster
   - Same quality in practice

2. RoPE (Rotary Position Embeddings):
   - Encodes position through rotation
   - Natural relative position encoding
   - Extrapolates to longer sequences
   - Applied to Q, K only

3. GQA (Grouped Query Attention):
   - Multiple Q heads share K, V heads
   - 4-8x reduction in KV-cache
   - Critical for long-context inference
   - Used in LLaMA-2-70B (8 KV heads for 64 Q heads)

4. SwiGLU:
   - Gated FFN: SiLU(gate) * up
   - More expressive than standard FFN
   - Hidden dim = 8d/3 to match param count

5. LLaMA Architecture:
   - RoPE for position
   - RMSNorm (pre-norm)
   - GQA (for 70B)
   - SwiGLU FFN
   - Weight tying

Files created:
- 01_modern_llm.md: Theory and concepts
- 01_modern_llm.py: This implementation file
- rope_visualization.png: RoPE frequency patterns
""")

print("\nModule 10.1 complete!")
