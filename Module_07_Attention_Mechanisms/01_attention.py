"""
Module 7.1: Attention Mechanisms - From Seq2Seq to Self-Attention
Complete implementations from scratch

Covers:
- Bahdanau (additive) attention
- Luong (multiplicative) attention
- Scaled dot-product attention
- Multi-head attention
- Self-attention and cross-attention
- Causal masking for autoregressive models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math

print("=" * 70)
print("Module 7.1: Attention Mechanisms")
print("=" * 70)


# ===========================================================================
# Section 1: Bahdanau (Additive) Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: Bahdanau (Additive) Attention")
print("=" * 70)


class BahdanauAttention(nn.Module):
    """
    Additive attention mechanism (Bahdanau et al., 2014).

    score = v^T × tanh(W_q × query + W_k × key)

    Used in: Original seq2seq attention
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        """
        Args:
            query_dim: Dimension of query vectors (decoder hidden state)
            key_dim: Dimension of key vectors (encoder hidden states)
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()

        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Decoder state (batch_size, query_dim)
            keys: Encoder states (batch_size, seq_len, key_dim)
            values: Encoder states (batch_size, seq_len, value_dim)
            mask: Optional mask (batch_size, seq_len), True for valid positions

        Returns:
            context: Weighted sum of values (batch_size, value_dim)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = keys.size()

        # Project query and keys
        # query: (batch, query_dim) -> (batch, 1, hidden_dim)
        query_proj = self.W_query(query).unsqueeze(1)

        # keys: (batch, seq_len, key_dim) -> (batch, seq_len, hidden_dim)
        keys_proj = self.W_key(keys)

        # Compute scores: v^T × tanh(W_q × q + W_k × k)
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(torch.tanh(query_proj + keys_proj)).squeeze(-1)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute context as weighted sum
        # (batch, 1, seq_len) @ (batch, seq_len, value_dim) -> (batch, 1, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


# Test Bahdanau Attention
print("\n--- Testing Bahdanau Attention ---")
batch_size, seq_len = 4, 10
query_dim, key_dim, value_dim = 256, 512, 512
hidden_dim = 128

attention = BahdanauAttention(query_dim, key_dim, hidden_dim)

query = torch.randn(batch_size, query_dim)  # Decoder state
keys = torch.randn(batch_size, seq_len, key_dim)  # Encoder states
values = keys.clone()  # Usually same as keys

context, weights = attention(query, keys, values)

print(f"Query shape: {query.shape}")
print(f"Keys shape: {keys.shape}")
print(f"Context shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights sum: {weights.sum(dim=-1)}")  # Should be ~1


# ===========================================================================
# Section 2: Luong (Multiplicative) Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: Luong (Multiplicative) Attention")
print("=" * 70)


class LuongAttention(nn.Module):
    """
    Multiplicative attention mechanism (Luong et al., 2015).

    Three variants:
    - dot: score = query^T × key
    - general: score = query^T × W × key
    - concat: score = v^T × tanh(W × [query; key])
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        method: str = 'general'
    ):
        super().__init__()
        self.method = method

        if method == 'general':
            self.W = nn.Linear(key_dim, query_dim, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(query_dim + key_dim, query_dim, bias=False)
            self.v = nn.Linear(query_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, query_dim)
            keys: (batch_size, seq_len, key_dim)
            values: (batch_size, seq_len, value_dim)
            mask: Optional (batch_size, seq_len)

        Returns:
            context: (batch_size, value_dim)
            attention_weights: (batch_size, seq_len)
        """
        if self.method == 'dot':
            # score = query^T × key
            # (batch, 1, query_dim) @ (batch, key_dim, seq_len) -> (batch, 1, seq_len)
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)

        elif self.method == 'general':
            # score = query^T × W × key
            # First: W × keys -> (batch, seq_len, query_dim)
            keys_proj = self.W(keys)
            scores = torch.bmm(query.unsqueeze(1), keys_proj.transpose(1, 2)).squeeze(1)

        elif self.method == 'concat':
            # score = v^T × tanh(W × [query; key])
            batch_size, seq_len, _ = keys.size()
            query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
            concat = torch.cat([query_expanded, keys], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(-1)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Context
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


# Test Luong Attention variants
print("\n--- Testing Luong Attention Variants ---")

for method in ['dot', 'general', 'concat']:
    # For dot product, query_dim must equal key_dim
    q_dim = 256 if method == 'dot' else query_dim
    k_dim = 256 if method == 'dot' else key_dim

    attention = LuongAttention(q_dim, k_dim, method=method)

    query = torch.randn(batch_size, q_dim)
    keys = torch.randn(batch_size, seq_len, k_dim)
    values = keys.clone()

    context, weights = attention(query, keys, values)
    print(f"{method.capitalize()}: context={context.shape}, weights_sum={weights.sum(dim=-1).mean():.4f}")


# ===========================================================================
# Section 3: Scaled Dot-Product Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: Scaled Dot-Product Attention")
print("=" * 70)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: (batch, ..., seq_q, d_k)
        key: (batch, ..., seq_k, d_k)
        value: (batch, ..., seq_v, d_v) where seq_k == seq_v
        mask: Optional mask, -inf for positions to ignore
        dropout: Optional dropout layer

    Returns:
        output: (batch, ..., seq_q, d_v)
        attention_weights: (batch, ..., seq_q, seq_k)
    """
    d_k = query.size(-1)

    # Compute attention scores: QK^T / sqrt(d_k)
    # (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (before softmax)
    if mask is not None:
        scores = scores + mask  # mask should have -inf where we don't want attention

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout to attention weights
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Compute output
    # (..., seq_q, seq_k) @ (..., seq_v, d_v) -> (..., seq_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# Test scaled dot-product attention
print("\n--- Testing Scaled Dot-Product Attention ---")
batch_size, seq_len, d_k, d_v = 4, 10, 64, 64

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")


# Demonstrate scaling importance
print("\n--- Why Scaling by sqrt(d_k)? ---")

def show_scaling_effect(d_k_values=[16, 64, 256, 512]):
    """Show how variance grows without scaling."""
    print(f"{'d_k':<10} {'Unscaled Var':<15} {'Scaled Var':<15}")
    print("-" * 40)

    for d_k in d_k_values:
        q = torch.randn(1000, d_k)  # 1000 samples
        k = torch.randn(1000, d_k)

        # Unscaled dot product
        unscaled = (q * k).sum(dim=-1)
        unscaled_var = unscaled.var().item()

        # Scaled dot product
        scaled = unscaled / math.sqrt(d_k)
        scaled_var = scaled.var().item()

        print(f"{d_k:<10} {unscaled_var:<15.2f} {scaled_var:<15.2f}")

    print("\nObservation: Scaled variance stays ~1 regardless of d_k")

show_scaling_effect()


# ===========================================================================
# Section 4: Multi-Head Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: Multi-Head Attention")
print("=" * 70)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in projections
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # For storing attention weights (useful for visualization)
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_q, d_model)
            key: (batch_size, seq_k, d_model)
            value: (batch_size, seq_v, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            output: (batch_size, seq_q, d_model)
            attention_weights: Optional (batch_size, num_heads, seq_q, seq_k)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)    # (batch, seq_k, d_model)
        V = self.W_v(value)  # (batch, seq_v, d_model)

        # Reshape for multi-head: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Store for visualization
        self.attention_weights = attention_weights

        # Reshape back: (batch, num_heads, seq_q, d_k) -> (batch, seq_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(output)

        if return_attention:
            return output, attention_weights
        return output


# Test Multi-Head Attention
print("\n--- Testing Multi-Head Attention ---")
batch_size, seq_len, d_model, num_heads = 4, 20, 512, 8

mha = MultiHeadAttention(d_model, num_heads, dropout=0.1)

x = torch.randn(batch_size, seq_len, d_model)

# Self-attention (Q=K=V=x)
output = mha(x, x, x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"d_k per head: {mha.d_k}")

# With return attention
output, attn = mha(x, x, x, return_attention=True)
print(f"Attention weights shape: {attn.shape}")  # (batch, num_heads, seq, seq)


# ===========================================================================
# Section 5: Causal (Masked) Self-Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Causal (Masked) Self-Attention")
print("=" * 70)


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal attention mask.

    Position i can only attend to positions 0, 1, ..., i

    Returns:
        mask: (seq_len, seq_len) with 0 for valid, -inf for invalid
    """
    # Upper triangular matrix (excluding diagonal)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))

    if device is not None:
        mask = mask.to(device)

    return mask


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention for autoregressive models (GPT-style).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        # Pre-compute causal mask for efficiency
        # Register as buffer (not a parameter, but saved with model)
        causal_mask = create_causal_mask(max_seq_len)
        self.register_buffer('causal_mask', causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # Self-attention with causal mask
        return self.mha(x, x, x, mask=mask, return_attention=return_attention)


# Test Causal Self-Attention
print("\n--- Testing Causal Self-Attention ---")
batch_size, seq_len, d_model, num_heads = 2, 8, 256, 4

causal_attn = CausalSelfAttention(d_model, num_heads, max_seq_len=100)

x = torch.randn(batch_size, seq_len, d_model)
output, attn_weights = causal_attn(x, return_attention=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")

# Visualize causal mask
print("\nCausal mask (8x8):")
mask = create_causal_mask(8)
mask_display = mask.clone()
mask_display[mask == float('-inf')] = -1
mask_display[mask == 0] = 1
for i in range(8):
    row = mask_display[i].tolist()
    print("  ", ["▓" if v == 1 else "░" for v in row])
print("  ▓ = can attend, ░ = masked")

# Verify attention is causal
print("\nVerifying causal attention pattern:")
print("Position 0 attention (should only attend to 0):")
print(f"  Weights: {attn_weights[0, 0, 0, :].detach().numpy().round(3)}")
print("Position 4 attention (should attend to 0-4):")
print(f"  Weights: {attn_weights[0, 0, 4, :].detach().numpy().round(3)}")


# ===========================================================================
# Section 6: Cross-Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: Cross-Attention")
print("=" * 70)


class CrossAttention(nn.Module):
    """
    Cross-attention for encoder-decoder models.

    Query from decoder, Key/Value from encoder.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            decoder_states: (batch, decoder_seq_len, d_model) - queries
            encoder_states: (batch, encoder_seq_len, d_model) - keys & values
            encoder_mask: Optional mask for encoder padding

        Returns:
            output: (batch, decoder_seq_len, d_model)
        """
        # Query from decoder, Key/Value from encoder
        return self.mha(
            query=decoder_states,
            key=encoder_states,
            value=encoder_states,
            mask=encoder_mask,
            return_attention=return_attention
        )


# Test Cross-Attention
print("\n--- Testing Cross-Attention ---")
batch_size, enc_len, dec_len, d_model, num_heads = 2, 15, 10, 256, 4

cross_attn = CrossAttention(d_model, num_heads)

encoder_out = torch.randn(batch_size, enc_len, d_model)  # From encoder
decoder_states = torch.randn(batch_size, dec_len, d_model)  # Decoder hidden states

output, attn = cross_attn(decoder_states, encoder_out, return_attention=True)

print(f"Encoder states shape: {encoder_out.shape}")
print(f"Decoder states shape: {decoder_states.shape}")
print(f"Output shape: {output.shape}")
print(f"Cross-attention weights shape: {attn.shape}")
print("  (batch, num_heads, decoder_seq, encoder_seq)")


# ===========================================================================
# Section 7: Complete Transformer Attention Block
# ===========================================================================
print("\n" + "=" * 70)
print("Section 7: Complete Transformer Attention Block")
print("=" * 70)


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with:
    - Multi-head self-attention
    - Feedforward network
    - Layer normalization
    - Residual connections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pre-norm architecture:
        x = x + Attention(Norm(x))
        x = x + FF(Norm(x))
        """
        # Self-attention with residual
        x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))

        return x


class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block with:
    - Causal self-attention
    - Cross-attention to encoder
    - Feedforward network
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()

        # Causal self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Causal mask
        causal_mask = create_causal_mask(max_seq_len)
        self.register_buffer('causal_mask', causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, dec_seq_len, d_model)
            encoder_output: Encoder output (batch, enc_seq_len, d_model)
        """
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Causal self-attention
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, mask=causal_mask))

        # Cross-attention
        x_norm = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x_norm, encoder_output, encoder_mask))

        # Feedforward
        x = x + self.ff(self.norm3(x))

        return x


# Test Transformer Blocks
print("\n--- Testing Transformer Blocks ---")
batch_size, enc_len, dec_len = 2, 20, 15
d_model, num_heads, d_ff = 256, 4, 1024

encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
decoder_block = TransformerDecoderBlock(d_model, num_heads, d_ff)

# Encoder
enc_input = torch.randn(batch_size, enc_len, d_model)
enc_output = encoder_block(enc_input)
print(f"Encoder input: {enc_input.shape}")
print(f"Encoder output: {enc_output.shape}")

# Decoder
dec_input = torch.randn(batch_size, dec_len, d_model)
dec_output = decoder_block(dec_input, enc_output)
print(f"Decoder input: {dec_input.shape}")
print(f"Decoder output: {dec_output.shape}")


# ===========================================================================
# Section 8: Attention Visualization
# ===========================================================================
print("\n" + "=" * 70)
print("Section 8: Attention Visualization")
print("=" * 70)


def visualize_attention(
    attention_weights: torch.Tensor,
    query_tokens: list,
    key_tokens: list,
    title: str = "Attention Weights",
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: (query_len, key_len) tensor
        query_tokens: List of query token strings
        key_tokens: List of key token strings
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    weights = attention_weights.detach().numpy()

    im = ax.imshow(weights, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(key_tokens)))
    ax.set_yticks(range(len(query_tokens)))
    ax.set_xticklabels(key_tokens, rotation=45, ha='right')
    ax.set_yticklabels(query_tokens)

    ax.set_xlabel('Keys (Source)')
    ax.set_ylabel('Queries (Target)')
    ax.set_title(title)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    plt.close()


def visualize_multihead_attention(
    attention_weights: torch.Tensor,
    tokens: list,
    save_path: Optional[str] = None
):
    """
    Visualize attention from all heads.

    Args:
        attention_weights: (num_heads, seq_len, seq_len) tensor
    """
    num_heads = attention_weights.size(0)
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        weights = attention_weights[i].detach().numpy()

        im = ax.imshow(weights, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {i+1}')

        if len(tokens) <= 12:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)

    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Multi-Head Attention Patterns', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-head visualization to {save_path}")
    plt.close()


# Demo attention visualization
print("\n--- Creating Attention Visualizations ---")

# Create sample attention pattern
seq_len = 8
tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "dog"]

# Random attention weights (for demo)
d_model, num_heads = 256, 8
mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(1, seq_len, d_model)
_, attn_weights = mha(x, x, x, return_attention=True)

# Visualize single head
visualize_attention(
    attn_weights[0, 0],  # First batch, first head
    tokens, tokens,
    title="Self-Attention (Head 1)",
    save_path="/Users/anuragmishra/Documents/Zero_to_GPT/Module_07_Attention_Mechanisms/attention_single_head.png"
)

# Visualize all heads
visualize_multihead_attention(
    attn_weights[0],  # First batch, all heads
    tokens,
    save_path="/Users/anuragmishra/Documents/Zero_to_GPT/Module_07_Attention_Mechanisms/attention_all_heads.png"
)


# ===========================================================================
# Section 9: Attention Patterns Analysis
# ===========================================================================
print("\n" + "=" * 70)
print("Section 9: Attention Patterns Analysis")
print("=" * 70)


def analyze_attention_patterns(attn_weights: torch.Tensor) -> dict:
    """
    Analyze common attention patterns.

    Args:
        attn_weights: (num_heads, seq_len, seq_len)

    Returns:
        Dictionary of pattern metrics
    """
    num_heads, seq_len, _ = attn_weights.shape

    patterns = {}

    for h in range(num_heads):
        weights = attn_weights[h]

        # 1. Local attention (diagonal strength)
        diag_weight = torch.diag(weights).mean().item()

        # 2. Previous token attention
        if seq_len > 1:
            prev_weight = torch.diag(weights, diagonal=-1).mean().item()
        else:
            prev_weight = 0

        # 3. First token attention (CLS-like)
        first_token_weight = weights[:, 0].mean().item()

        # 4. Entropy (spread of attention)
        entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean().item()

        patterns[f'head_{h}'] = {
            'self_attention': diag_weight,
            'previous_token': prev_weight,
            'first_token': first_token_weight,
            'entropy': entropy
        }

    return patterns


# Analyze patterns
print("\nAttention Pattern Analysis:")
patterns = analyze_attention_patterns(attn_weights[0])

print(f"\n{'Head':<8} {'Self':<10} {'Prev':<10} {'First':<10} {'Entropy':<10}")
print("-" * 48)
for head, metrics in patterns.items():
    print(f"{head:<8} {metrics['self_attention']:<10.3f} "
          f"{metrics['previous_token']:<10.3f} "
          f"{metrics['first_token']:<10.3f} "
          f"{metrics['entropy']:<10.3f}")


# ===========================================================================
# Section 10: Efficient Attention Computation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 10: Efficient Attention Computation")
print("=" * 70)


class EfficientMultiHeadAttention(nn.Module):
    """
    Memory-efficient multi-head attention using fused operations.

    Key optimizations:
    1. Combined QKV projection
    2. Fused attention computation
    3. Memory-efficient reshaping
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Combined QKV projection (more efficient)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Efficient self-attention.

        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x)

        # Reshape: (batch, seq, 3 * d_model) -> (batch, seq, 3, num_heads, d_k)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)

        # Permute: (batch, seq, 3, heads, d_k) -> (3, batch, heads, seq, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into Q, K, V
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_v, d_v)
        output = torch.matmul(attn_weights, V)

        # Reshape back: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(output)

        return output


# Compare efficiency
print("\n--- Comparing Attention Implementations ---")

import time

batch_size, seq_len, d_model, num_heads = 8, 512, 512, 8

standard_mha = MultiHeadAttention(d_model, num_heads)
efficient_mha = EfficientMultiHeadAttention(d_model, num_heads)

x = torch.randn(batch_size, seq_len, d_model)

# Warmup
_ = standard_mha(x, x, x)
_ = efficient_mha(x)

# Timing
n_iters = 50

start = time.time()
for _ in range(n_iters):
    _ = standard_mha(x, x, x)
standard_time = (time.time() - start) / n_iters * 1000

start = time.time()
for _ in range(n_iters):
    _ = efficient_mha(x)
efficient_time = (time.time() - start) / n_iters * 1000

print(f"Standard MHA: {standard_time:.2f} ms")
print(f"Efficient MHA: {efficient_time:.2f} ms")
print(f"Speedup: {standard_time / efficient_time:.2f}x")


# ===========================================================================
# Section 11: Seq2Seq with Attention (Complete Example)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 11: Seq2Seq with Attention (Complete Example)")
print("=" * 70)


class Seq2SeqWithAttention(nn.Module):
    """
    Complete sequence-to-sequence model with attention.
    Uses Bahdanau-style attention between encoder and decoder.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Attention
        self.attention = BahdanauAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim * 2,  # Bidirectional encoder
            hidden_dim=attention_dim
        )

        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.decoder = nn.LSTM(
            embed_dim + hidden_dim * 2,  # Input + context
            hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, tgt_vocab_size)

        # Bridge: encoder final hidden -> decoder initial hidden
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode source sequence."""
        embedded = self.src_embedding(src)

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            encoder_outputs, (h_n, c_n) = self.encoder(packed)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_outputs, batch_first=True
            )
        else:
            encoder_outputs, (h_n, c_n) = self.encoder(embedded)

        # Reshape hidden states for decoder
        num_layers = self.encoder.num_layers
        batch_size = src.size(0)

        # h_n: (num_layers * 2, batch, hidden) -> (num_layers, batch, hidden * 2)
        h_n = h_n.view(num_layers, 2, batch_size, -1)
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        h_n = self.bridge_h(h_n)

        c_n = c_n.view(num_layers, 2, batch_size, -1)
        c_n = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        c_n = self.bridge_c(c_n)

        return encoder_outputs, (h_n, c_n)

    def decode_step(
        self,
        tgt_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Single decoding step with attention."""

        # Embed target token
        embedded = self.tgt_embedding(tgt_token)  # (batch, 1, embed_dim)

        # Get decoder hidden state for attention query
        h = hidden[0][-1]  # Last layer: (batch, hidden_dim)

        # Compute attention
        context, attn_weights = self.attention(
            query=h,
            keys=encoder_outputs,
            values=encoder_outputs,
            mask=src_mask
        )  # context: (batch, hidden_dim * 2)

        # Decoder input: [embedded; context]
        decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        # Decoder step
        output, hidden = self.decoder(decoder_input, hidden)

        # Project to vocabulary
        logits = self.output_proj(output)  # (batch, 1, vocab_size)

        return logits, hidden, attn_weights

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full forward pass with teacher forcing.

        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Encode
        encoder_outputs, hidden = self.encode(src, src_lengths)

        # Decode with teacher forcing
        batch_size, tgt_len = tgt.size()
        outputs = []

        for t in range(tgt_len):
            tgt_input = tgt[:, t:t+1]  # (batch, 1)
            logits, hidden, _ = self.decode_step(
                tgt_input, hidden, encoder_outputs
            )
            outputs.append(logits)

        # Stack: list of (batch, 1, vocab) -> (batch, tgt_len, vocab)
        outputs = torch.cat(outputs, dim=1)

        return outputs


# Test Seq2Seq with Attention
print("\n--- Testing Seq2Seq with Attention ---")
src_vocab, tgt_vocab = 5000, 8000
embed_dim, hidden_dim, attn_dim = 256, 512, 128

model = Seq2SeqWithAttention(
    src_vocab, tgt_vocab,
    embed_dim, hidden_dim, attn_dim
)

src = torch.randint(1, src_vocab, (4, 20))
tgt = torch.randint(1, tgt_vocab, (4, 15))

logits = model(src, tgt)
print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 7.1 Summary: Attention Mechanisms")
print("=" * 70)

print("""
Key Takeaways:
==============

1. Attention solves the bottleneck problem in seq2seq:
   - No need to compress entire sequence into single vector
   - Decoder can look at all encoder positions

2. Types of Attention:
   - Additive (Bahdanau): score = v^T tanh(W_q q + W_k k)
   - Multiplicative (Luong): score = q^T W k or q^T k
   - Scaled Dot-Product: score = (q^T k) / sqrt(d_k)

3. Scaled Dot-Product Attention:
   - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
   - Scaling prevents softmax saturation

4. Multi-Head Attention:
   - Multiple attention heads capture different patterns
   - MultiHead = Concat(head_i) W_O
   - Typical: d_k = d_model / num_heads

5. Self-Attention vs Cross-Attention:
   - Self: Q, K, V from same sequence
   - Cross: Q from decoder, K/V from encoder

6. Causal Masking:
   - For autoregressive models (GPT)
   - Position i can only attend to positions 0..i

7. Attention → Transformers:
   - Attention is the core mechanism
   - Combined with FFN, residuals, normalization
   - Foundation for all modern LLMs

Files created:
- 01_attention.md: Theory and concepts
- 01_attention.py: This implementation file
- attention_single_head.png: Single head visualization
- attention_all_heads.png: Multi-head visualization
""")

print("\nModule 7.1 complete!")
