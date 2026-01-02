"""
Module 18: Interview Preparation - Code Reference

This file contains clean implementations of key concepts
for interview preparation and quick reference.

Contents:
1. Attention Mechanisms
2. Position Encodings
3. Transformer Components
4. Training Utilities
5. LoRA Implementation
6. Memory Estimation
7. Common Interview Coding Problems
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Section 1: Attention Mechanisms
# ============================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V

    Args:
        query: (batch, heads, seq_q, d_k)
        key: (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask: (batch, 1, seq_q, seq_k) or broadcastable

    Returns:
        output: (batch, heads, seq_q, d_v)
    """
    d_k = query.size(-1)

    # Attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax and apply to values
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Combined projection for efficiency
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V together
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Attention
        out = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Uses fewer KV heads than query heads.
    Each KV head is shared by multiple query heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_groups = num_heads // num_kv_heads

        # Separate projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # Project
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads to match Q heads
        K = K.repeat_interleave(self.kv_groups, dim=1)
        V = V.repeat_interleave(self.kv_groups, dim=1)

        # Attention
        out = scaled_dot_product_attention(Q, K, V, mask)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out)


# ============================================================================
# Section 2: Position Encodings
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Original sinusoidal position encoding from "Attention is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies rotation to query and key based on position.
    Enables relative position encoding and better length extrapolation.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key.

        Args:
            q, k: (batch, heads, seq, head_dim)
            positions: (seq,) position indices

        Returns:
            Rotated q, k
        """
        seq_len = q.size(2)

        if positions is None:
            positions = torch.arange(seq_len, device=q.device)

        # Compute angles
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq, dim)

        cos = emb.cos()[None, None, :, :]  # (1, 1, seq, dim)
        sin = emb.sin()[None, None, :, :]

        # Apply rotation
        q_rotated = self._rotate(q, cos, sin)
        k_rotated = self._rotate(k, cos, sin)

        return q_rotated, k_rotated

    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotation using complex multiplication."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        ], dim=-1).flatten(-2)
        return x_rotated


# ============================================================================
# Section 3: Transformer Components
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) * g
    where RMS(x) = sqrt(mean(x^2))

    Simpler and faster than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function for FFN.

    SwiGLU(x) = Swish(xW_1) ⊙ (xV)
    where Swish(x) = x × sigmoid(x)

    Used in LLaMA, PaLM, etc.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.silu(self.W(x)) * self.V(x))


class TransformerBlock(nn.Module):
    """
    Modern transformer block (LLaMA-style).

    Features:
    - Pre-normalization
    - RMSNorm
    - SwiGLU FFN
    - No bias
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# ============================================================================
# Section 4: Training Utilities
# ============================================================================

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0  # True where attention is allowed


def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute perplexity from logits and labels.

    PPL = exp(avg_cross_entropy)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )

    return math.exp(loss.item())


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.base_lr * (
                self.min_lr_ratio + (1 - self.min_lr_ratio) *
                0.5 * (1 + math.cos(math.pi * progress))
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# ============================================================================
# Section 5: LoRA Implementation
# ============================================================================

class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation.

    W' = W + (alpha/r) * B @ A

    W is frozen, only A and B are trained.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0
    ):
        super().__init__()

        # Frozen base weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + self.scale * lora

    def merge(self):
        """Merge LoRA into base weights for inference."""
        if not self.merged:
            self.weight.data += self.scale * (self.lora_B @ self.lora_A)
            self.merged = True


# ============================================================================
# Section 6: Memory Estimation
# ============================================================================

def estimate_model_memory(
    num_params: int,
    dtype: str = 'fp16'
) -> float:
    """Estimate model memory in GB."""
    bytes_per_param = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}
    return num_params * bytes_per_param[dtype] / 1e9


def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str = 'fp16'
) -> float:
    """Estimate KV-cache memory in GB."""
    bytes_per_element = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1}
    # K and V for each layer
    total_elements = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim
    return total_elements * bytes_per_element[dtype] / 1e9


def estimate_training_memory(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    optimizer: str = 'adam'
) -> dict:
    """
    Estimate training memory breakdown.

    Components:
    1. Model (FP32 master weights)
    2. Gradients (FP32)
    3. Optimizer states (Adam: 2x for m, v)
    4. Activations (rough estimate)
    """
    # Model and gradients
    model_mem = num_params * 4 / 1e9  # FP32
    grad_mem = num_params * 4 / 1e9   # FP32

    # Optimizer states
    if optimizer == 'adam':
        opt_mem = num_params * 8 / 1e9  # m and v
    else:
        opt_mem = 0

    # Activations (rough: 2 * batch * seq * hidden * layers)
    act_mem = 2 * batch_size * seq_len * hidden_size * num_layers * 2 / 1e9

    return {
        'model_gb': model_mem,
        'gradients_gb': grad_mem,
        'optimizer_gb': opt_mem,
        'activations_gb': act_mem,
        'total_gb': model_mem + grad_mem + opt_mem + act_mem
    }


# ============================================================================
# Section 7: Common Interview Coding Problems
# ============================================================================

def implement_attention_from_scratch(
    d_model: int = 512,
    num_heads: int = 8,
    seq_len: int = 100,
    batch_size: int = 2
) -> torch.Tensor:
    """
    Interview Problem: Implement multi-head attention from scratch.

    This is a common interview question. Show you understand:
    1. Q, K, V projections
    2. Scaled dot-product
    3. Softmax
    4. Multi-head splitting
    """
    head_dim = d_model // num_heads

    # Initialize projections
    W_q = torch.randn(d_model, d_model) * 0.02
    W_k = torch.randn(d_model, d_model) * 0.02
    W_v = torch.randn(d_model, d_model) * 0.02
    W_o = torch.randn(d_model, d_model) * 0.02

    # Input
    x = torch.randn(batch_size, seq_len, d_model)

    # Project Q, K, V
    Q = x @ W_q  # (B, N, D)
    K = x @ W_k
    V = x @ W_v

    # Reshape for multi-head: (B, N, D) -> (B, H, N, D/H)
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)

    # Reshape back: (B, H, N, D/H) -> (B, N, D)
    output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

    # Output projection
    output = output @ W_o

    return output


def implement_kv_cache_generation():
    """
    Interview Problem: Implement generation with KV-cache.

    Shows understanding of:
    1. Why KV-cache is needed
    2. How to update cache
    3. Memory vs compute trade-off
    """
    # Simplified model
    class SimpleLM(nn.Module):
        def __init__(self, vocab_size=100, d_model=64, num_heads=4):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.attn = MultiHeadAttention(d_model, num_heads)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, x, kv_cache=None):
            h = self.embed(x)

            # In real implementation, attn would use kv_cache
            h = self.attn(h)

            logits = self.lm_head(h)
            return logits, kv_cache

    model = SimpleLM()
    input_ids = torch.randint(0, 100, (1, 5))

    # Generation with simulated KV-cache
    generated = input_ids
    kv_cache = None

    for _ in range(10):
        # Only process last token if cache exists
        if kv_cache is not None:
            model_input = generated[:, -1:]
        else:
            model_input = generated

        logits, kv_cache = model(model_input, kv_cache)

        # Sample next token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def implement_dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Interview Problem: Implement DPO loss.

    L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    """
    chosen_log_ratio = policy_chosen_logprobs - ref_chosen_logprobs
    rejected_log_ratio = policy_rejected_logprobs - ref_rejected_logprobs

    loss = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio)).mean()
    return loss


# ============================================================================
# Demo and Testing
# ============================================================================

def run_all_demos():
    """Run all implementations to verify correctness."""
    print("=" * 60)
    print("Interview Code Reference - Verification")
    print("=" * 60)

    # Test attention
    print("\n1. Testing Multi-Head Attention...")
    attn = MultiHeadAttention(d_model=256, num_heads=8)
    x = torch.randn(2, 16, 256)
    out = attn(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape, "Attention shape mismatch!"
    print("   OK!")

    # Test GQA
    print("\n2. Testing Grouped Query Attention...")
    gqa = GroupedQueryAttention(d_model=256, num_heads=8, num_kv_heads=2)
    out = gqa(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   KV heads: 2 (4x reduction)")
    print("   OK!")

    # Test RoPE
    print("\n3. Testing RoPE...")
    rope = RotaryPositionalEmbedding(dim=64)
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    q_rot, k_rot = rope(q, k)
    print(f"   Applied rotation: {q.shape} -> {q_rot.shape}")
    print("   OK!")

    # Test RMSNorm
    print("\n4. Testing RMSNorm...")
    norm = RMSNorm(256)
    out = norm(x)
    print(f"   Normalized: {out.shape}")
    print("   OK!")

    # Test SwiGLU
    print("\n5. Testing SwiGLU FFN...")
    ffn = SwiGLU(256, 1024)
    out = ffn(x)
    print(f"   FFN: {x.shape} -> {out.shape}")
    print("   OK!")

    # Test Transformer Block
    print("\n6. Testing Transformer Block...")
    block = TransformerBlock(d_model=256, num_heads=8, d_ff=1024)
    mask = create_causal_mask(16, x.device)
    out = block(x, mask)
    print(f"   Block: {x.shape} -> {out.shape}")
    print("   OK!")

    # Test LoRA
    print("\n7. Testing LoRA...")
    lora = LoRALinear(256, 256, r=8, alpha=16)
    out = lora(x)
    print(f"   LoRA forward: {x.shape} -> {out.shape}")
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("   OK!")

    # Memory estimation
    print("\n8. Testing Memory Estimation...")
    mem = estimate_training_memory(
        num_params=7_000_000_000,
        batch_size=4,
        seq_len=2048,
        hidden_size=4096,
        num_layers=32
    )
    print(f"   LLaMA-7B training estimate:")
    for k, v in mem.items():
        print(f"   - {k}: {v:.1f} GB")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()
