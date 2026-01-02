"""
Module 8.1: The Transformer Architecture
Complete implementation from "Attention Is All You Need"

Covers:
- Token embeddings and positional encoding
- Multi-head attention
- Feed-forward networks
- Encoder and Decoder stacks
- Full encoder-decoder transformer
- Training utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math
import copy

print("=" * 70)
print("Module 8.1: The Transformer Architecture")
print("=" * 70)


# ===========================================================================
# Section 1: Positional Encoding
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: Positional Encoding")
print("=" * 70)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding (used in BERT, GPT-2).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


# Visualize positional encoding
def visualize_positional_encoding(d_model: int = 128, max_len: int = 100):
    """Visualize sinusoidal positional encoding patterns."""
    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0)
    x = torch.zeros(1, max_len, d_model)
    encoding = pe.pe[0, :max_len, :].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of encoding
    im = axes[0].imshow(encoding.T, aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    axes[0].set_title('Positional Encoding Heatmap')
    plt.colorbar(im, ax=axes[0])

    # Individual dimensions
    for dim in [0, 1, 2, 3, 10, 50]:
        if dim < d_model:
            axes[1].plot(encoding[:, dim], label=f'dim {dim}')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Positional Encoding by Dimension')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/anuragmishra/Documents/Zero_to_GPT/Module_08_Transformer_Architecture/positional_encoding.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved positional encoding visualization")

visualize_positional_encoding()


# Test positional encoding
print("\n--- Testing Positional Encoding ---")
d_model, seq_len = 512, 100

pe_sin = SinusoidalPositionalEncoding(d_model, max_len=1000)
pe_learned = LearnedPositionalEncoding(d_model, max_len=1000)

x = torch.randn(4, seq_len, d_model)

x_with_pe_sin = pe_sin(x)
x_with_pe_learned = pe_learned(x)

print(f"Input shape: {x.shape}")
print(f"With sinusoidal PE: {x_with_pe_sin.shape}")
print(f"With learned PE: {x_with_pe_learned.shape}")


# ===========================================================================
# Section 2: Multi-Head Attention
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: Multi-Head Attention")
print("=" * 70)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
    where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

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
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_v, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        # (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k) -> (batch, heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_v, d_k) -> (batch, heads, seq_q, d_k)
        context = torch.matmul(attn_weights, V)

        # Reshape: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final projection
        output = self.W_o(context)

        if return_attention:
            return output, attn_weights
        return output


# Test Multi-Head Attention
print("\n--- Testing Multi-Head Attention ---")
batch_size, seq_len, d_model, num_heads = 4, 20, 512, 8

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)

# Self-attention
output = mha(x, x, x)
print(f"Self-attention: input={x.shape}, output={output.shape}")


# ===========================================================================
# Section 3: Position-wise Feed-Forward Network
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: Position-wise Feed-Forward Network")
print("=" * 70)


class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2

    Applied independently to each position.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in LLaMA, PaLM).

    SwiGLU(x) = (x @ W_1 * SiLU(x @ W_gate)) @ W_2

    More expressive than standard FFN, used in modern LLMs.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.dropout(self.w2(F.silu(self.w_gate(x)) * self.w1(x)))


# Test FFN
print("\n--- Testing Feed-Forward Networks ---")
d_model, d_ff = 512, 2048

ffn_standard = PositionwiseFFN(d_model, d_ff)
ffn_swiglu = SwiGLUFFN(d_model, d_ff)

x = torch.randn(4, 20, d_model)

out_standard = ffn_standard(x)
out_swiglu = ffn_swiglu(x)

print(f"Input: {x.shape}")
print(f"Standard FFN output: {out_standard.shape}")
print(f"SwiGLU FFN output: {out_swiglu.shape}")

# Parameter count comparison
params_standard = sum(p.numel() for p in ffn_standard.parameters())
params_swiglu = sum(p.numel() for p in ffn_swiglu.parameters())
print(f"Standard FFN params: {params_standard:,}")
print(f"SwiGLU FFN params: {params_swiglu:,}")


# ===========================================================================
# Section 4: Transformer Encoder Layer
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: Transformer Encoder Layer")
print("=" * 70)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Consists of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    3. Residual connections
    4. Layer normalization (Pre-norm)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.pre_norm = pre_norm

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        if self.pre_norm:
            # Pre-norm: normalize before sublayer
            # Self-attention
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, x, x, mask)
            x = self.dropout1(x)
            x = residual + x

            # FFN
            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout2(x)
            x = residual + x
        else:
            # Post-norm: normalize after sublayer (original transformer)
            # Self-attention
            x = x + self.dropout1(self.self_attn(x, x, x, mask))
            x = self.norm1(x)

            # FFN
            x = x + self.dropout2(self.ffn(x))
            x = self.norm2(x)

        return x


# Test Encoder Layer
print("\n--- Testing Encoder Layer ---")
d_model, num_heads, d_ff = 512, 8, 2048

encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
x = torch.randn(4, 20, d_model)

output = encoder_layer(x)
print(f"Encoder layer: input={x.shape}, output={output.shape}")


# ===========================================================================
# Section 5: Transformer Decoder Layer
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Transformer Decoder Layer")
print("=" * 70)


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Consists of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention (to encoder)
    3. Position-wise feed-forward network
    4. Residual connections
    5. Layer normalization (Pre-norm)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.pre_norm = pre_norm

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Encoder output (batch, src_len, d_model)
            self_mask: Causal attention mask
            cross_mask: Cross-attention mask

        Returns:
            output: (batch, tgt_len, d_model)
        """
        if self.pre_norm:
            # Pre-norm
            # Masked self-attention
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, x, x, self_mask)
            x = self.dropout1(x)
            x = residual + x

            # Cross-attention
            residual = x
            x = self.norm2(x)
            x = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
            x = self.dropout2(x)
            x = residual + x

            # FFN
            residual = x
            x = self.norm3(x)
            x = self.ffn(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            # Post-norm
            x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, self_mask)))
            x = self.norm2(x + self.dropout2(self.cross_attn(x, encoder_output, encoder_output, cross_mask)))
            x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


# Test Decoder Layer
print("\n--- Testing Decoder Layer ---")
decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff)

tgt = torch.randn(4, 15, d_model)  # Target sequence
memory = torch.randn(4, 20, d_model)  # Encoder output

# Create causal mask
tgt_len = tgt.size(1)
causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))

output = decoder_layer(tgt, memory, self_mask=causal_mask)
print(f"Decoder layer: tgt={tgt.shape}, memory={memory.shape}, output={output.shape}")


# ===========================================================================
# Section 6: Full Transformer Encoder
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: Full Transformer Encoder")
print("=" * 70)


class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder with N layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation, pre_norm)
            for _ in range(num_layers)
        ])

        # Final layer norm (required for pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source token indices (batch, src_len)
            mask: Optional padding mask

        Returns:
            encoder_output: (batch, src_len, d_model)
        """
        # Embed and add positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.final_norm(x)

        return x


# Test Encoder
print("\n--- Testing Full Encoder ---")
vocab_size, d_model, num_heads, d_ff, num_layers = 10000, 512, 8, 2048, 6

encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)

src = torch.randint(0, vocab_size, (4, 30))
encoder_output = encoder(src)

print(f"Source tokens: {src.shape}")
print(f"Encoder output: {encoder_output.shape}")
print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")


# ===========================================================================
# Section 7: Full Transformer Decoder
# ===========================================================================
print("\n" + "=" * 70)
print("Section 7: Full Transformer Decoder")
print("=" * 70)


class TransformerDecoder(nn.Module):
    """
    Full Transformer Decoder with N layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, activation, pre_norm)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

        # Pre-compute causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(max_len))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal (look-ahead) mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target token indices (batch, tgt_len)
            encoder_output: Encoder output (batch, src_len, d_model)
            cross_mask: Optional cross-attention mask

        Returns:
            decoder_output: (batch, tgt_len, d_model)
        """
        tgt_len = tgt.size(1)

        # Embed and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Get causal mask for this sequence length
        causal_mask = self.causal_mask[:tgt_len, :tgt_len]

        # Pass through layers
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, cross_mask)

        # Final normalization
        x = self.final_norm(x)

        return x


# Test Decoder
print("\n--- Testing Full Decoder ---")
decoder = TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers)

tgt = torch.randint(0, vocab_size, (4, 25))
decoder_output = decoder(tgt, encoder_output)

print(f"Target tokens: {tgt.shape}")
print(f"Encoder output: {encoder_output.shape}")
print(f"Decoder output: {decoder_output.shape}")


# ===========================================================================
# Section 8: Complete Transformer (Encoder-Decoder)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 8: Complete Transformer (Encoder-Decoder)")
print("=" * 70)


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.

    Follows the architecture from "Attention Is All You Need".
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True,
        tie_weights: bool = False
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff,
            num_encoder_layers, max_len, dropout, activation, pre_norm
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_decoder_layers, max_len, dropout, activation, pre_norm
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Optional: tie embedding and output weights
        if tie_weights and src_vocab_size == tgt_vocab_size:
            self.output_proj.weight = self.decoder.embedding.weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            src_mask: Source padding mask
            cross_mask: Cross-attention mask

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # Encode source
        encoder_output = self.encoder(src, src_mask)

        # Decode target
        decoder_output = self.decoder(tgt, encoder_output, cross_mask)

        # Project to vocabulary
        logits = self.output_proj(decoder_output)

        return logits

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence only."""
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode given encoder output."""
        decoder_output = self.decoder(tgt, encoder_output, cross_mask)
        logits = self.output_proj(decoder_output)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        start_token: int = 1,
        end_token: int = 2,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive generation using greedy decoding.

        Args:
            src: Source sequence (batch, src_len)
            max_len: Maximum generation length
            start_token: Start of sequence token ID
            end_token: End of sequence token ID
            temperature: Sampling temperature

        Returns:
            generated: Generated sequences (batch, gen_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        encoder_output = self.encode(src)

        # Start with BOS token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Generate tokens one by one
        for _ in range(max_len - 1):
            # Decode
            logits = self.decode(generated, encoder_output)

            # Get next token logits
            next_logits = logits[:, -1, :] / temperature

            # Greedy: take argmax
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have ended
            if (next_token == end_token).all():
                break

        return generated


# Test Complete Transformer
print("\n--- Testing Complete Transformer ---")
src_vocab, tgt_vocab = 10000, 8000
d_model, num_heads, d_ff = 512, 8, 2048
num_layers = 6

model = Transformer(
    src_vocab_size=src_vocab,
    tgt_vocab_size=tgt_vocab,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers
)

src = torch.randint(0, src_vocab, (4, 30))
tgt = torch.randint(0, tgt_vocab, (4, 25))

logits = model(src, tgt)

print(f"Source: {src.shape}")
print(f"Target: {tgt.shape}")
print(f"Logits: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================================
# Section 9: Decoder-Only Transformer (GPT-style)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 9: Decoder-Only Transformer (GPT-style)")
print("=" * 70)


class GPTBlock(nn.Module):
    """Single GPT block: causal self-attention + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Used in: GPT-2, GPT-3, LLaMA, etc.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        num_layers: int = 12,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (weight tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(max_len))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        # Causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # Pass through blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        if return_hidden:
            return logits, x
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self(input_ids)[:, -1, :]

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Test GPT
print("\n--- Testing GPT Model ---")
vocab_size, d_model, num_heads, d_ff, num_layers = 50257, 768, 12, 3072, 12

gpt = GPT(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=1024)

input_ids = torch.randint(0, vocab_size, (2, 50))
logits = gpt(input_ids)

print(f"Input: {input_ids.shape}")
print(f"Logits: {logits.shape}")
print(f"GPT parameters: {sum(p.numel() for p in gpt.parameters()):,}")

# Generate
generated = gpt.generate(input_ids[:, :10], max_new_tokens=20, temperature=0.8, top_k=50)
print(f"Generated: {generated.shape}")


# ===========================================================================
# Section 10: Training Utilities
# ===========================================================================
print("\n" + "=" * 70)
print("Section 10: Training Utilities")
print("=" * 70)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.

    Instead of hard targets [0, 0, 1, 0], use soft targets [ε/V, ε/V, 1-ε, ε/V]
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.padding_idx = padding_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            targets: (batch * seq_len,)

        Returns:
            loss: Scalar loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.vocab_size - 2))  # Exclude padding and true label
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth_targets[:, self.padding_idx] = 0

            # Mask padding positions
            mask = (targets != self.padding_idx).float().unsqueeze(1)
            smooth_targets = smooth_targets * mask

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = loss.mean()

        return loss


class TransformerLRScheduler:
    """
    Learning rate scheduler from "Attention Is All You Need".

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """Update learning rate."""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )


# Visualize LR schedule
def visualize_lr_schedule():
    """Visualize transformer learning rate schedule."""
    d_model = 512
    warmup_steps = 4000
    total_steps = 100000

    steps = range(1, total_steps + 1)
    lrs = []

    for step in steps:
        lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        lrs.append(lr)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Transformer Learning Rate Schedule')
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label=f'Warmup end ({warmup_steps} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/anuragmishra/Documents/Zero_to_GPT/Module_08_Transformer_Architecture/lr_schedule.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved LR schedule visualization")

visualize_lr_schedule()


# Test training utilities
print("\n--- Testing Training Utilities ---")

# Label smoothing loss
criterion = LabelSmoothingLoss(vocab_size=8000, smoothing=0.1)
logits = torch.randn(32 * 25, 8000)  # Flattened batch
targets = torch.randint(0, 8000, (32 * 25,))
loss = criterion(logits, targets)
print(f"Label smoothing loss: {loss.item():.4f}")

# Compare with standard cross-entropy
ce_loss = F.cross_entropy(logits, targets)
print(f"Standard CE loss: {ce_loss.item():.4f}")


# ===========================================================================
# Section 11: Model Configurations
# ===========================================================================
print("\n" + "=" * 70)
print("Section 11: Model Configurations")
print("=" * 70)


def get_model_config(model_name: str) -> dict:
    """Get configuration for various transformer models."""
    configs = {
        'transformer-base': {
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'num_layers': 6,
            'dropout': 0.1
        },
        'transformer-big': {
            'd_model': 1024,
            'num_heads': 16,
            'd_ff': 4096,
            'num_layers': 6,
            'dropout': 0.3
        },
        'gpt2-small': {
            'd_model': 768,
            'num_heads': 12,
            'd_ff': 3072,
            'num_layers': 12,
            'dropout': 0.1
        },
        'gpt2-medium': {
            'd_model': 1024,
            'num_heads': 16,
            'd_ff': 4096,
            'num_layers': 24,
            'dropout': 0.1
        },
        'gpt2-large': {
            'd_model': 1280,
            'num_heads': 20,
            'd_ff': 5120,
            'num_layers': 36,
            'dropout': 0.1
        },
        'gpt2-xl': {
            'd_model': 1600,
            'num_heads': 25,
            'd_ff': 6400,
            'num_layers': 48,
            'dropout': 0.1
        },
        'bert-base': {
            'd_model': 768,
            'num_heads': 12,
            'd_ff': 3072,
            'num_layers': 12,
            'dropout': 0.1
        },
        'bert-large': {
            'd_model': 1024,
            'num_heads': 16,
            'd_ff': 4096,
            'num_layers': 24,
            'dropout': 0.1
        },
    }
    return configs.get(model_name, configs['transformer-base'])


# Print configurations
print("\n--- Model Configurations ---")
print(f"{'Model':<20} {'d_model':<10} {'heads':<8} {'d_ff':<8} {'layers':<8}")
print("-" * 60)
for model_name in ['transformer-base', 'gpt2-small', 'gpt2-medium', 'bert-base']:
    config = get_model_config(model_name)
    print(f"{model_name:<20} {config['d_model']:<10} {config['num_heads']:<8} "
          f"{config['d_ff']:<8} {config['num_layers']:<8}")


# Parameter count estimation
def estimate_params(config: dict, vocab_size: int) -> int:
    """Estimate parameter count for GPT-style model."""
    d = config['d_model']
    h = config['num_heads']
    ff = config['d_ff']
    L = config['num_layers']
    V = vocab_size

    # Embeddings
    embedding_params = V * d + 1024 * d  # Token + position

    # Per-layer params
    # Attention: 4 * d * d (Q, K, V, O)
    # FFN: d * ff + ff * d
    layer_params = 4 * d * d + 2 * d * ff

    # Total (excluding biases for simplicity)
    total = embedding_params + L * layer_params

    return total


print("\n--- Estimated Parameter Counts ---")
vocab_size = 50257  # GPT-2 vocab size
for model_name in ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    config = get_model_config(model_name)
    params = estimate_params(config, vocab_size)
    print(f"{model_name}: ~{params / 1e6:.0f}M parameters")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 8.1 Summary: Transformer Architecture")
print("=" * 70)

print("""
Key Takeaways:
==============

1. Transformer Architecture:
   - Encoder: Self-attention + FFN (bidirectional)
   - Decoder: Causal self-attention + Cross-attention + FFN
   - Residual connections + Layer normalization

2. Key Components:
   - Multi-head attention: Diverse attention patterns
   - Positional encoding: Sinusoidal or learned
   - Position-wise FFN: Non-linear transformation

3. Attention:
   - Scaled dot-product: softmax(QK^T / sqrt(d_k)) V
   - Causal masking for autoregressive models
   - Cross-attention connects encoder and decoder

4. Modern Improvements:
   - Pre-norm (more stable than post-norm)
   - GELU activation (smoother than ReLU)
   - SwiGLU FFN (used in LLaMA)
   - Weight tying (embedding = output projection)

5. Variants:
   - Encoder-only (BERT): Bidirectional, for understanding
   - Decoder-only (GPT): Causal, for generation
   - Encoder-decoder (T5): For seq2seq tasks

6. Training:
   - Label smoothing: Soft targets
   - Warmup + decay: LR schedule
   - Gradient clipping: Stability

Files created:
- 01_transformer.md: Theory and concepts
- 01_transformer.py: This implementation file
- positional_encoding.png: PE visualization
- lr_schedule.png: Learning rate schedule
""")

print("\nModule 8.1 complete!")
