"""
Module 11.1: Efficient Attention - Flash Attention & KV-Cache
Implementation of efficient attention mechanisms

Covers:
- Flash Attention concepts (simplified)
- KV-Cache implementation
- Sliding Window Attention
- Paged Attention concepts
- Speculative Decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import math
import time


# Device detection for Apple Silicon (MPS) / CUDA / CPU
def get_device():
    """Get best available device: MPS > CUDA > CPU"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = get_device()
print(f"Using device: {DEVICE}")

print("=" * 70)
print("Module 11.1: Efficient Attention - Flash Attention & KV-Cache")
print("=" * 70)


# ===========================================================================
# Section 1: Standard vs Flash Attention Comparison
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: Standard vs Flash Attention Comparison")
print("=" * 70)


def standard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Standard attention implementation.

    Memory: O(n²) for attention matrix
    """
    d_k = query.size(-1)

    # Compute attention scores: O(n²) memory
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    # Softmax: still O(n²)
    attn_weights = F.softmax(scores, dim=-1)

    # Apply to values
    output = torch.matmul(attn_weights, value)

    return output


def flash_attention_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False
) -> torch.Tensor:
    """
    Use PyTorch's built-in Flash Attention (PyTorch 2.0+).

    Memory: O(n) - doesn't materialize full attention matrix
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal
    )


# Compare memory and speed
print("\n--- Comparing Standard vs Flash Attention ---")

def benchmark_attention(batch_size, num_heads, seq_len, head_dim, num_runs=10):
    """Benchmark attention implementations."""

    # Create inputs (use CPU for benchmarking to avoid device-specific issues)
    device = 'cpu'  # Benchmarking on CPU for consistency
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Clear cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    start = time.time()
    for _ in range(num_runs):
        _ = standard_attention(q, k, v)
    std_time = (time.time() - start) / num_runs

    # Flash attention (PyTorch built-in)
    start = time.time()
    for _ in range(num_runs):
        _ = flash_attention_pytorch(q, k, v, is_causal=True)
    flash_time = (time.time() - start) / num_runs

    return std_time, flash_time


# Test different sequence lengths
print(f"\n{'Seq Len':<10} {'Standard (ms)':<15} {'Flash (ms)':<15} {'Speedup':<10}")
print("-" * 50)

for seq_len in [256, 512, 1024, 2048]:
    std_time, flash_time = benchmark_attention(2, 8, seq_len, 64, num_runs=5)
    speedup = std_time / flash_time
    print(f"{seq_len:<10} {std_time*1000:<15.2f} {flash_time*1000:<15.2f} {speedup:<10.2f}x")


# ===========================================================================
# Section 2: Simplified Flash Attention (Educational)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: Simplified Flash Attention (Educational)")
print("=" * 70)


class SimplifiedFlashAttention:
    """
    Educational implementation of Flash Attention concepts.

    This is NOT the real Flash Attention (which requires CUDA kernels),
    but demonstrates the key ideas:
    - Block-wise processing
    - Online softmax computation
    - Avoiding materialization of full attention matrix
    """

    @staticmethod
    def online_softmax_update(
        m_prev: torch.Tensor,  # Previous max
        l_prev: torch.Tensor,  # Previous sum of exp
        o_prev: torch.Tensor,  # Previous output
        s_block: torch.Tensor,  # Current block scores
        v_block: torch.Tensor   # Current block values
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update softmax statistics incrementally (online algorithm).
        """
        # New maximum
        m_block = s_block.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_prev, m_block)

        # Rescale previous and compute new exponentials
        exp_prev = torch.exp(m_prev - m_new)
        exp_block = torch.exp(s_block - m_new)

        # Update sum
        l_new = l_prev * exp_prev + exp_block.sum(dim=-1, keepdim=True)

        # Update output with proper rescaling
        o_new = (o_prev * l_prev * exp_prev +
                 torch.matmul(exp_block, v_block)) / l_new

        return m_new, l_new, o_new

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_size: int = 64
    ) -> torch.Tensor:
        """
        Block-wise attention computation.

        Args:
            query: (batch, heads, seq_q, d)
            key: (batch, heads, seq_k, d)
            value: (batch, heads, seq_v, d)
            block_size: Size of each block
        """
        batch, heads, seq_q, d = query.shape
        seq_k = key.shape[2]
        scale = 1.0 / math.sqrt(d)

        # Initialize output and statistics
        output = torch.zeros_like(query)
        m = torch.full((batch, heads, seq_q, 1), float('-inf'), device=query.device)
        l = torch.zeros((batch, heads, seq_q, 1), device=query.device)

        # Process in blocks
        for i in range(0, seq_k, block_size):
            # Get current block
            k_block = key[:, :, i:i+block_size, :]
            v_block = value[:, :, i:i+block_size, :]

            # Compute block scores
            s_block = torch.matmul(query, k_block.transpose(-2, -1)) * scale

            # Update using online softmax
            m, l, output = SimplifiedFlashAttention.online_softmax_update(
                m, l, output, s_block, v_block
            )

        return output


# Test simplified flash attention
print("\n--- Testing Simplified Flash Attention ---")

batch, heads, seq_len, d = 2, 4, 256, 64

q = torch.randn(batch, heads, seq_len, d)
k = torch.randn(batch, heads, seq_len, d)
v = torch.randn(batch, heads, seq_len, d)

# Standard attention
out_standard = standard_attention(q, k, v)

# Simplified flash attention
out_flash = SimplifiedFlashAttention.forward(q, k, v, block_size=64)

# Compare outputs
diff = (out_standard - out_flash).abs().max()
print(f"Max difference between standard and block-wise: {diff:.6f}")
print(f"Outputs match: {diff < 1e-4}")


# ===========================================================================
# Section 3: KV-Cache Implementation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: KV-Cache Implementation")
print("=" * 70)


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        # Pre-allocate cache
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)
        self.v_cache = torch.zeros(cache_shape, device=self.device, dtype=self.dtype)

        # Track current position
        self.seq_len = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value and return full cached tensors.

        Args:
            key: New keys (batch, heads, new_seq, head_dim)
            value: New values (batch, heads, new_seq, head_dim)

        Returns:
            Tuple of (full_keys, full_values) including cache
        """
        new_seq_len = key.shape[2]

        # Store in cache
        self.k_cache[:, :, self.seq_len:self.seq_len + new_seq_len, :] = key
        self.v_cache[:, :, self.seq_len:self.seq_len + new_seq_len, :] = value

        # Update position
        self.seq_len += new_seq_len

        # Return cached values
        return (
            self.k_cache[:, :, :self.seq_len, :],
            self.v_cache[:, :, :self.seq_len, :]
        )

    def get_memory_usage(self) -> int:
        """Return memory usage in bytes."""
        return self.k_cache.numel() * 2 + self.v_cache.numel() * 2  # 2 bytes for FP16

    def clear(self):
        """Clear the cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


class CachedAttention(nn.Module):
    """
    Attention with KV-Cache support for efficient generation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward with optional KV-cache.
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Update cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Create new cache if requested
        new_cache = None
        if use_cache and kv_cache is None:
            new_cache = KVCache(batch_size, self.num_heads, self.max_seq_len, self.head_dim)
            k, v = new_cache.update(k, v)

        # Attention
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.wo(output)

        return output, new_cache if use_cache else kv_cache


# Test KV-Cache
print("\n--- Testing KV-Cache ---")

d_model, num_heads = 256, 4
attn = CachedAttention(d_model, num_heads)

# Simulate generation
prompt = torch.randn(1, 10, d_model)  # Initial prompt
print(f"Prompt shape: {prompt.shape}")

# Initial forward (create cache)
output, kv_cache = attn(prompt, use_cache=True)
print(f"After prompt - Cache seq_len: {kv_cache.seq_len}")

# Generate tokens one by one
for i in range(5):
    new_token = torch.randn(1, 1, d_model)
    output, kv_cache = attn(new_token, kv_cache=kv_cache)
    print(f"After token {i+1} - Cache seq_len: {kv_cache.seq_len}")

print(f"\nCache memory usage: {kv_cache.get_memory_usage() / 1024:.2f} KB")


# ===========================================================================
# Section 4: Sliding Window Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: Sliding Window Attention")
print("=" * 70)


class SlidingWindowAttention(nn.Module):
    """
    Attention with sliding window - each position only attends to
    the last W positions.

    Memory: O(n × W) instead of O(n²)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 4096
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sliding window attention mask.

        Position i can attend to positions [max(0, i-W+1), i]
        """
        # Create position indices
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # Causal: can't attend to future
        causal_mask = col_idx <= row_idx

        # Window: can't attend beyond window
        window_mask = row_idx - col_idx < self.window_size

        # Combine
        mask = causal_mask & window_mask

        # Convert to attention mask format
        mask = mask.float().masked_fill(~mask, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sliding window attention."""
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Create sliding window mask
        mask = self.create_sliding_window_mask(seq_len, x.device)

        # Attention with mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.wo(output)

        return output, attn_weights


# Test Sliding Window Attention
print("\n--- Testing Sliding Window Attention ---")

d_model, num_heads, window_size = 128, 4, 64
sliding_attn = SlidingWindowAttention(d_model, num_heads, window_size)

x = torch.randn(1, 256, d_model)
output, attn_weights = sliding_attn(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Window size: {window_size}")

# Visualize attention pattern
def visualize_sliding_window_attention(attn_weights, window_size, save_path):
    """Visualize sliding window attention pattern."""
    # Take first head
    attn = attn_weights[0, 0].detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full attention matrix
    axes[0].imshow(attn, cmap='Blues', aspect='auto')
    axes[0].set_title('Sliding Window Attention')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    # Zoom into a section
    start = 100
    end = min(150, attn.shape[0])
    axes[1].imshow(attn[start:end, max(0, start-window_size):end], cmap='Blues', aspect='auto')
    axes[1].set_title(f'Zoomed View (positions {start}-{end})')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sliding window visualization to {save_path}")

visualize_sliding_window_attention(
    attn_weights,
    window_size,
    '/Users/anuragmishra/Documents/Zero_to_GPT/Module_11_Efficient_Attention/sliding_window_attn.png'
)


# ===========================================================================
# Section 5: Rolling Buffer KV-Cache
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Rolling Buffer KV-Cache")
print("=" * 70)


class RollingBufferKVCache:
    """
    Rolling buffer KV-Cache for sliding window attention.

    Only stores the last W tokens, using circular buffer.
    Memory: O(W) instead of O(n)
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        window_size: int,
        head_dim: int,
        device: torch.device = None
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = head_dim
        self.device = device or torch.device('cpu')

        # Circular buffer
        cache_shape = (batch_size, num_heads, window_size, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=self.device)
        self.v_cache = torch.zeros(cache_shape, device=self.device)

        # Position tracking
        self.position = 0  # Next write position
        self.filled = 0    # How many positions are filled

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update rolling buffer with new key/value.

        Returns valid cached keys/values.
        """
        new_len = key.shape[2]

        for i in range(new_len):
            # Write to circular buffer position
            write_pos = (self.position + i) % self.window_size
            self.k_cache[:, :, write_pos, :] = key[:, :, i, :]
            self.v_cache[:, :, write_pos, :] = value[:, :, i, :]

        # Update position
        self.position = (self.position + new_len) % self.window_size
        self.filled = min(self.filled + new_len, self.window_size)

        # Return valid portion (need to handle wrap-around)
        if self.filled < self.window_size:
            return (
                self.k_cache[:, :, :self.filled, :],
                self.v_cache[:, :, :self.filled, :]
            )
        else:
            # Full buffer - return all (positions are valid)
            return self.k_cache, self.v_cache

    def get_memory_usage(self) -> int:
        """Memory usage in bytes (assuming FP32)."""
        return (self.k_cache.numel() + self.v_cache.numel()) * 4


# Test Rolling Buffer
print("\n--- Testing Rolling Buffer KV-Cache ---")

batch_size, num_heads, window_size, head_dim = 1, 4, 8, 32

rolling_cache = RollingBufferKVCache(batch_size, num_heads, window_size, head_dim)

print(f"Window size: {window_size}")

# Simulate adding tokens
for i in range(15):
    new_k = torch.randn(batch_size, num_heads, 1, head_dim)
    new_v = torch.randn(batch_size, num_heads, 1, head_dim)

    k_cached, v_cached = rolling_cache.update(new_k, new_v)
    print(f"Token {i+1}: cache size = {k_cached.shape[2]}, filled = {rolling_cache.filled}")

print(f"\nRolling buffer memory: {rolling_cache.get_memory_usage() / 1024:.2f} KB")
print(f"Standard cache would be: ~{15 * batch_size * num_heads * head_dim * 4 * 2 / 1024:.2f} KB")


# ===========================================================================
# Section 6: Speculative Decoding
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: Speculative Decoding")
print("=" * 70)


class SpeculativeDecoder:
    """
    Speculative decoding for faster generation.

    Uses a small draft model to generate candidates,
    then verifies with large model in parallel.
    """

    def __init__(
        self,
        large_model: nn.Module,
        draft_model: nn.Module,
        vocab_size: int,
        k: int = 4  # Number of draft tokens
    ):
        self.large_model = large_model
        self.draft_model = draft_model
        self.vocab_size = vocab_size
        self.k = k

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, int]:
        """
        One step of speculative decoding.

        Returns:
            new_tokens: Accepted tokens
            num_accepted: Number of draft tokens accepted
        """
        batch_size = input_ids.shape[0]

        # 1. Generate k draft tokens
        draft_tokens = []
        draft_context = input_ids.clone()

        for _ in range(self.k):
            draft_logits = self.draft_model(draft_context)[:, -1, :]
            draft_probs = F.softmax(draft_logits / temperature, dim=-1)
            next_token = torch.multinomial(draft_probs, 1)
            draft_tokens.append(next_token)
            draft_context = torch.cat([draft_context, next_token], dim=1)

        draft_tokens = torch.cat(draft_tokens, dim=1)  # (batch, k)

        # 2. Verify with large model (all at once!)
        verify_context = torch.cat([input_ids, draft_tokens], dim=1)
        large_logits = self.large_model(verify_context)

        # 3. Accept/reject tokens
        accepted_tokens = []
        num_accepted = 0

        for i in range(self.k):
            pos = input_ids.shape[1] + i - 1  # Position to verify
            if pos < 0:
                pos = 0

            large_probs = F.softmax(large_logits[:, pos, :] / temperature, dim=-1)
            draft_token = draft_tokens[:, i]

            # Simple acceptance: check if draft token has reasonable probability
            token_prob = large_probs.gather(1, draft_token.unsqueeze(1)).squeeze()

            # Simplified acceptance criterion
            if token_prob.item() > 0.1:  # Accept if prob > threshold
                accepted_tokens.append(draft_token)
                num_accepted += 1
            else:
                # Reject - sample from large model
                sampled = torch.multinomial(large_probs, 1).squeeze()
                accepted_tokens.append(sampled.unsqueeze(0))
                num_accepted += 1
                break

        return torch.cat(accepted_tokens, dim=0).unsqueeze(0), num_accepted


# Simulate speculative decoding (with dummy models)
print("\n--- Simulating Speculative Decoding ---")

class DummyModel(nn.Module):
    """Simple model for demonstration."""
    def __init__(self, vocab_size, d_model, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

vocab_size, d_model = 1000, 128

# Large model (slower)
large = DummyModel(vocab_size, d_model, 512)
# Draft model (faster)
draft = DummyModel(vocab_size, d_model, 64)

spec_decoder = SpeculativeDecoder(large, draft, vocab_size, k=4)

# Simulate generation
input_ids = torch.randint(0, vocab_size, (1, 10))
print(f"Initial context: {input_ids.shape}")

total_tokens = 0
total_accepted = 0

for step in range(5):
    new_tokens, num_accepted = spec_decoder.generate_step(input_ids)
    input_ids = torch.cat([input_ids, new_tokens], dim=1)
    total_tokens += new_tokens.shape[1]
    total_accepted += num_accepted
    print(f"Step {step+1}: Generated {new_tokens.shape[1]} tokens, accepted {num_accepted}/{spec_decoder.k}")

print(f"\nFinal sequence length: {input_ids.shape[1]}")
print(f"Acceptance rate: {total_accepted / (5 * spec_decoder.k):.2%}")


# ===========================================================================
# Section 7: Memory Analysis
# ===========================================================================
print("\n" + "=" * 70)
print("Section 7: Memory Analysis")
print("=" * 70)


def analyze_attention_memory(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """Analyze memory requirements for different attention methods."""

    # Standard attention: stores full n×n matrix per head
    standard = batch_size * num_heads * seq_len * seq_len * dtype_bytes

    # Flash attention: only block-sized matrices (approximate)
    block_size = 128
    flash = batch_size * num_heads * block_size * block_size * dtype_bytes * 2  # Q, K blocks

    # Linear attention: stores features
    linear = batch_size * num_heads * seq_len * head_dim * dtype_bytes * 2

    return {
        'standard_mb': standard / (1024**2),
        'flash_mb': flash / (1024**2),
        'linear_mb': linear / (1024**2)
    }


print("\nMemory Analysis for Different Attention Methods:")
print("-" * 70)

configs = [
    (2, 1024, 32, 128),
    (2, 4096, 32, 128),
    (2, 8192, 32, 128),
    (2, 32768, 32, 128),
]

print(f"{'Seq Len':<10} {'Standard (MB)':<15} {'Flash (MB)':<15} {'Linear (MB)':<15}")
print("-" * 55)

for batch, seq, heads, dim in configs:
    mem = analyze_attention_memory(batch, seq, heads, dim)
    print(f"{seq:<10} {mem['standard_mb']:<15.1f} {mem['flash_mb']:<15.1f} {mem['linear_mb']:<15.1f}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 11.1 Summary: Efficient Attention")
print("=" * 70)

print("""
Key Takeaways:
==============

1. Flash Attention:
   - Uses tiling + online softmax
   - Never materializes full n² attention matrix
   - O(n) memory instead of O(n²)
   - 2-4x speedup for long sequences
   - Built into PyTorch 2.0+

2. KV-Cache:
   - Store computed K, V during generation
   - Avoid recomputation of past tokens
   - Critical for efficient inference

3. KV-Cache Optimizations:
   - GQA/MQA: Reduce number of KV heads
   - Paged Attention: Virtual memory for cache
   - Quantization: INT8 KV-cache

4. Sliding Window:
   - Attend only to last W positions
   - O(nW) instead of O(n²)
   - Rolling buffer cache: O(W) memory
   - Effective context = n_layers × W

5. Speculative Decoding:
   - Draft model generates candidates
   - Large model verifies in parallel
   - 2-3x speedup typical

Files created:
- 01_efficient_attention.md: Theory and concepts
- 01_efficient_attention.py: This implementation file
- sliding_window_attn.png: Visualization
""")

print("\nModule 11.1 complete!")
