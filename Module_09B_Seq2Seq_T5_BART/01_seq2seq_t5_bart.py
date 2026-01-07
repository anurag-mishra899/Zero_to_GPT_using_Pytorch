"""
Seq2Seq Models: T5 & BART - Hands-on Implementation
====================================================

This module covers:
1. Encoder-Decoder architecture from scratch
2. T5-style span corruption pre-training
3. BART-style denoising pre-training
4. Cross-attention mechanisms
5. Training and fine-tuning seq2seq models
6. Generation with beam search

Author: Zero to GPT Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import math
from dataclasses import dataclass
import random
import numpy as np


# =============================================================================
# Section 1: Configuration
# =============================================================================

@dataclass
class Seq2SeqConfig:
    """Configuration for Encoder-Decoder model."""
    vocab_size: int = 32000
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_relative_positions: bool = True  # T5-style


# =============================================================================
# Section 2: Position Encodings
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (for BART-style)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class T5RelativePositionBias(nn.Module):
    """
    T5-style relative position bias.

    Instead of absolute positions, adds learned bias based on
    relative distance between tokens.
    """

    def __init__(
        self,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Map relative positions to bucket indices.

        Uses logarithmic bucketing for positions beyond half the buckets.
        """
        ret = 0
        n = -relative_position

        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Logarithmic bucketing for larger distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) /
            math.log(self.max_distance / max_exact) *
            (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias.

        Returns: (1, n_heads, query_length, key_length)
        """
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]

        relative_position = memory_position - context_position
        bucket = self._relative_position_bucket(relative_position)

        values = self.relative_attention_bias(bucket)  # (q_len, k_len, n_heads)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, q_len, k_len)

        return values


# =============================================================================
# Section 3: Attention Mechanisms
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative position bias."""

    def __init__(self, config: Seq2SeqConfig, is_cross_attention: bool = False):
        super().__init__()

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, q_len, d_model)
            key: (batch, k_len, d_model)
            value: (batch, v_len, d_model)
            attention_mask: (batch, 1, q_len, k_len) or broadcastable
            position_bias: (1, n_heads, q_len, k_len) for relative positions

        Returns:
            output: (batch, q_len, d_model)
        """
        B, q_len, _ = query.shape
        k_len = key.shape[1]

        # Project
        Q = self.q_proj(query).view(B, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add relative position bias if provided
        if position_bias is not None:
            scores = scores + position_bias

        # Apply mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, q_len, self.d_model)

        return self.out_proj(out)


# =============================================================================
# Section 4: Transformer Blocks
# =============================================================================

class EncoderBlock(nn.Module):
    """Single encoder transformer block."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()

        self.self_attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.ln1(x)
        x = self.self_attention(x, x, x, attention_mask, position_bias)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class DecoderBlock(nn.Module):
    """Single decoder transformer block with cross-attention."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()

        # Self-attention (causal)
        self.self_attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Cross-attention to encoder
        self.cross_attention = MultiHeadAttention(config, is_cross_attention=True)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # FFN
        self.ln3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        self_position_bias: Optional[torch.Tensor] = None,
        cross_position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention (causal)
        residual = x
        x = self.ln1(x)
        x = self.self_attention(x, x, x, self_attention_mask, self_position_bias)
        x = residual + x

        # Cross-attention to encoder
        residual = x
        x = self.ln2(x)
        x = self.cross_attention(x, encoder_output, encoder_output,
                                 cross_attention_mask, cross_position_bias)
        x = residual + x

        # FFN
        residual = x
        x = self.ln3(x)
        x = self.ffn(x)
        x = residual + x

        return x


# =============================================================================
# Section 5: Complete Encoder-Decoder Model
# =============================================================================

class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        if config.use_relative_positions:
            self.position_bias = T5RelativePositionBias(
                config.n_heads, bidirectional=True
            )
            self.position_embedding = None
        else:
            self.position_embedding = SinusoidalPositionalEncoding(
                config.d_model, config.max_seq_len
            )
            self.position_bias = None

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.n_encoder_layers)
        ])

        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            encoder_output: (batch, seq_len, d_model)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embed_tokens(input_ids)

        # Position encoding
        if self.position_embedding is not None:
            x = self.position_embedding(x)

        x = self.dropout(x)

        # Convert attention mask to attention bias
        if attention_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -1e9
        else:
            extended_mask = None

        # Get relative position bias
        if self.position_bias is not None:
            position_bias = self.position_bias(L, L, device)
        else:
            position_bias = None

        # Encoder layers
        for layer in self.layers:
            x = layer(x, extended_mask, position_bias)

        return self.final_ln(x)


class Decoder(nn.Module):
    """Transformer decoder stack with cross-attention."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        if config.use_relative_positions:
            self.position_bias = T5RelativePositionBias(
                config.n_heads, bidirectional=False  # Causal
            )
            self.position_embedding = None
        else:
            self.position_embedding = SinusoidalPositionalEncoding(
                config.d_model, config.max_seq_len
            )
            self.position_bias = None

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_decoder_layers)
        ])

        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, dec_len) decoder input ids
            encoder_output: (batch, enc_len, d_model)
            encoder_attention_mask: (batch, enc_len)
            decoder_attention_mask: (batch, dec_len)

        Returns:
            decoder_output: (batch, dec_len, d_model)
        """
        B, dec_len = input_ids.shape
        enc_len = encoder_output.shape[1]
        device = input_ids.device

        # Token embeddings
        x = self.embed_tokens(input_ids)

        # Position encoding
        if self.position_embedding is not None:
            x = self.position_embedding(x)

        x = self.dropout(x)

        # Causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones(dec_len, dec_len, device=device), diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) * -1e9

        # Combine with padding mask if provided
        if decoder_attention_mask is not None:
            dec_pad_mask = (1.0 - decoder_attention_mask[:, None, None, :]) * -1e9
            self_attention_mask = causal_mask + dec_pad_mask
        else:
            self_attention_mask = causal_mask

        # Cross-attention mask
        if encoder_attention_mask is not None:
            cross_attention_mask = (1.0 - encoder_attention_mask[:, None, None, :]) * -1e9
        else:
            cross_attention_mask = None

        # Position biases
        if self.position_bias is not None:
            self_position_bias = self.position_bias(dec_len, dec_len, device)
        else:
            self_position_bias = None

        # Decoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_output,
                self_attention_mask, cross_attention_mask,
                self_position_bias, None  # No position bias for cross-attention
            )

        return self.final_ln(x)


class Seq2SeqModel(nn.Module):
    """
    Complete Encoder-Decoder model.

    Can be configured for T5-style or BART-style.
    """

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Shared embeddings between encoder and decoder (T5-style)
        self.decoder.embed_tokens.weight = self.encoder.embed_tokens.weight

        # Output projection (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
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
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, src_len) source tokens
            decoder_input_ids: (batch, tgt_len) decoder input (shifted labels)
            attention_mask: (batch, src_len) source mask
            decoder_attention_mask: (batch, tgt_len) target mask
            labels: (batch, tgt_len) targets for loss computation

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Encode
        encoder_output = self.encoder(input_ids, attention_mask)

        # Decode
        decoder_output = self.decoder(
            decoder_input_ids, encoder_output,
            attention_mask, decoder_attention_mask
        )

        # Project to vocabulary
        logits = self.lm_head(decoder_output)

        output = {'logits': logits, 'encoder_output': encoder_output}

        # Compute loss
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            output['loss'] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        """
        Generate output sequence.

        Args:
            input_ids: (batch, src_len) source tokens
            attention_mask: (batch, src_len)
            max_length: Maximum generation length
            num_beams: Beam size (1 = greedy)
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy/beam search

        Returns:
            generated_ids: (batch, gen_len)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Encode once
        encoder_output = self.encoder(input_ids, attention_mask)

        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device
        )

        if num_beams > 1:
            return self._beam_search(
                encoder_output, attention_mask, decoder_input_ids,
                max_length, num_beams
            )

        # Greedy or sampling decoding
        for _ in range(max_length - 1):
            decoder_output = self.decoder(
                decoder_input_ids, encoder_output, attention_mask
            )
            logits = self.lm_head(decoder_output[:, -1, :])

            if do_sample:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            if (next_token == self.config.eos_token_id).all():
                break

        return decoder_input_ids

    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        initial_input: torch.Tensor,
        max_length: int,
        num_beams: int
    ) -> torch.Tensor:
        """Simple beam search implementation."""
        device = encoder_output.device
        batch_size = encoder_output.shape[0]

        # Expand for beams
        encoder_output = encoder_output.repeat_interleave(num_beams, dim=0)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.repeat_interleave(num_beams, dim=0)

        # Initialize beams: (batch * num_beams, 1)
        beam_input = initial_input.repeat_interleave(num_beams, dim=0)
        beam_scores = torch.zeros(batch_size * num_beams, device=device)
        beam_scores[1::num_beams] = -1e9  # Only first beam active initially

        for step in range(max_length - 1):
            decoder_output = self.decoder(beam_input, encoder_output, encoder_mask)
            logits = self.lm_head(decoder_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            vocab_size = log_probs.shape[-1]

            # Add current beam scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs

            # Reshape for beam selection
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Select top beams
            top_scores, top_indices = torch.topk(next_scores, num_beams, dim=-1)

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beam scores and sequences
            beam_scores = top_scores.view(-1)

            # Gather previous sequences
            beam_indices_expanded = beam_indices.view(-1)
            batch_beam_offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            beam_indices_global = (batch_beam_offset + beam_indices).view(-1)

            beam_input = beam_input[beam_indices_global]
            beam_input = torch.cat([beam_input, token_indices.view(-1, 1)], dim=1)

            # Check for EOS
            if (token_indices == self.config.eos_token_id).all():
                break

        # Return best beam for each batch
        best_beam_indices = torch.arange(0, batch_size * num_beams, num_beams, device=device)
        return beam_input[best_beam_indices]


# =============================================================================
# Section 6: Pre-training Data Creation
# =============================================================================

def create_t5_span_corruption_data(
    tokens: List[int],
    noise_density: float = 0.15,
    mean_span_length: int = 3,
    sentinel_start_id: int = 32000
) -> Tuple[List[int], List[int]]:
    """
    Create T5-style span corruption training data.

    Args:
        tokens: Original token ids
        noise_density: Fraction of tokens to corrupt
        mean_span_length: Average span length
        sentinel_start_id: Starting ID for sentinel tokens

    Returns:
        (corrupted_tokens, target_tokens)
    """
    n = len(tokens)
    num_noise_tokens = int(n * noise_density)
    num_spans = max(1, num_noise_tokens // mean_span_length)

    # Sample span start positions
    possible_starts = list(range(n))
    random.shuffle(possible_starts)
    span_starts = sorted(possible_starts[:num_spans])

    # Generate spans
    spans = []
    last_end = 0
    for start in span_starts:
        if start < last_end:
            continue

        # Sample span length
        length = max(1, int(np.random.exponential(mean_span_length)))
        end = min(start + length, n)

        spans.append((start, end))
        last_end = end

    # Create corrupted input and target
    corrupted = []
    target = []

    sentinel_id = sentinel_start_id
    last_end = 0

    for start, end in spans:
        # Add tokens before span
        corrupted.extend(tokens[last_end:start])

        # Add sentinel to both
        corrupted.append(sentinel_id)
        target.append(sentinel_id)

        # Add span tokens to target
        target.extend(tokens[start:end])

        sentinel_id += 1
        last_end = end

    # Add remaining tokens
    corrupted.extend(tokens[last_end:])

    return corrupted, target


def create_bart_denoising_data(
    tokens: List[int],
    mask_token_id: int,
    mask_ratio: float = 0.3,
    poisson_lambda: float = 3.0
) -> Tuple[List[int], List[int]]:
    """
    Create BART-style text infilling data.

    Randomly masks spans of tokens with a single mask token.

    Args:
        tokens: Original token ids
        mask_token_id: ID of mask token
        mask_ratio: Fraction of tokens to mask
        poisson_lambda: Mean span length

    Returns:
        (corrupted_tokens, original_tokens)
    """
    n = len(tokens)
    num_to_mask = int(n * mask_ratio)

    # Sample span lengths from Poisson
    span_lengths = []
    total_masked = 0
    while total_masked < num_to_mask:
        length = max(1, np.random.poisson(poisson_lambda))
        span_lengths.append(length)
        total_masked += length

    # Sample span positions
    num_spans = len(span_lengths)
    max_start = n - sum(span_lengths)

    if max_start <= 0:
        # Fall back to single span
        span_starts = [0]
        span_lengths = [num_to_mask]
    else:
        span_starts = sorted(random.sample(range(max_start), min(num_spans, max_start)))

    # Create corrupted sequence
    corrupted = []
    i = 0
    span_idx = 0

    while i < n:
        if span_idx < len(span_starts) and i == span_starts[span_idx]:
            # Insert mask and skip span
            corrupted.append(mask_token_id)
            i += span_lengths[span_idx]
            span_idx += 1

            # Adjust remaining span starts
            for j in range(span_idx, len(span_starts)):
                span_starts[j] -= span_lengths[span_idx - 1] - 1
        else:
            corrupted.append(tokens[i])
            i += 1

    return corrupted, tokens


# =============================================================================
# Section 7: Training Utilities
# =============================================================================

def shift_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int
) -> torch.Tensor:
    """
    Shift input ids one position to the right for decoder input.

    Args:
        input_ids: (batch, seq_len) target tokens
        pad_token_id: Padding token ID
        decoder_start_token_id: Start token ID for decoder

    Returns:
        shifted: (batch, seq_len) shifted tokens
    """
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id

    # Replace -100 (ignore index) with pad
    shifted = shifted.masked_fill(shifted == -100, pad_token_id)

    return shifted


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for seq2seq training."""

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed distribution
        smooth = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
        smooth.scatter_(-1, targets.unsqueeze(-1).clamp(0), self.confidence)

        # Mask padding
        mask = (targets != self.ignore_index).float().unsqueeze(-1)

        loss = -(smooth * log_probs * mask).sum() / mask.sum()
        return loss


class Seq2SeqTrainer:
    """Training loop for seq2seq models."""

    def __init__(
        self,
        model: Seq2SeqModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_label_smoothing: bool = True,
        gradient_clip: float = 1.0,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.device = device

        if use_label_smoothing:
            self.criterion = LabelSmoothingLoss(
                model.config.vocab_size,
                smoothing=0.1,
                ignore_index=model.config.pad_token_id
            )
        else:
            self.criterion = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        labels = batch['labels'].to(self.device)

        # Create decoder input (shift right)
        decoder_input_ids = shift_right(
            labels,
            self.model.config.pad_token_id,
            self.model.config.bos_token_id
        )

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        if self.criterion is not None:
            loss = self.criterion(outputs['logits'], labels)
        else:
            loss = outputs['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)

                decoder_input_ids = shift_right(
                    labels,
                    self.model.config.pad_token_id,
                    self.model.config.bos_token_id
                )

                outputs = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Count non-padding tokens
                mask = (labels != self.model.config.pad_token_id)
                num_tokens = mask.sum().item()

                total_loss += outputs['loss'].item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {'loss': avg_loss, 'perplexity': perplexity}


# =============================================================================
# Section 8: Simple Dataset
# =============================================================================

class SimpleSeq2SeqDataset(Dataset):
    """Simple dataset for demonstration."""

    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        max_source_len: int = 128,
        max_target_len: int = 64
    ):
        self.examples = []

        for src, tgt in zip(source_texts, target_texts):
            # Simple character-level tokenization
            src_tokens = [ord(c) % 1000 for c in src[:max_source_len]]
            tgt_tokens = [ord(c) % 1000 for c in tgt[:max_target_len]]

            # Pad
            src_pad = [0] * (max_source_len - len(src_tokens))
            tgt_pad = [0] * (max_target_len - len(tgt_tokens))

            self.examples.append({
                'input_ids': torch.tensor(src_tokens + src_pad),
                'attention_mask': torch.tensor([1] * len(src_tokens) + [0] * len(src_pad)),
                'labels': torch.tensor(tgt_tokens + tgt_pad)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Section 9: Demonstrations
# =============================================================================

def demo_span_corruption():
    """Demonstrate T5-style span corruption."""
    print("=" * 60)
    print("Demo: T5 Span Corruption")
    print("=" * 60)

    # Simulate tokens
    tokens = list(range(100, 120))  # 20 tokens
    print(f"\nOriginal tokens: {tokens}")

    corrupted, target = create_t5_span_corruption_data(
        tokens,
        noise_density=0.15,
        mean_span_length=3,
        sentinel_start_id=32000
    )

    print(f"Corrupted (input): {corrupted}")
    print(f"Target: {target}")


def demo_bart_denoising():
    """Demonstrate BART-style denoising."""
    print("\n" + "=" * 60)
    print("Demo: BART Text Infilling")
    print("=" * 60)

    tokens = list(range(100, 120))
    print(f"\nOriginal tokens: {tokens}")

    corrupted, original = create_bart_denoising_data(
        tokens,
        mask_token_id=3,  # [MASK]
        mask_ratio=0.3,
        poisson_lambda=3.0
    )

    print(f"Corrupted (input): {corrupted}")
    print(f"Target (reconstruct): {original}")


def demo_model_forward():
    """Demonstrate model forward pass."""
    print("\n" + "=" * 60)
    print("Demo: Seq2Seq Model Forward Pass")
    print("=" * 60)

    config = Seq2SeqConfig(
        vocab_size=1000,
        d_model=128,
        d_ff=256,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        max_seq_len=64
    )

    model = Seq2SeqModel(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    batch_size = 2
    src_len = 20
    tgt_len = 15

    input_ids = torch.randint(4, 1000, (batch_size, src_len))
    attention_mask = torch.ones(batch_size, src_len)
    labels = torch.randint(4, 1000, (batch_size, tgt_len))

    # Create decoder input
    decoder_input_ids = shift_right(labels, config.pad_token_id, config.bos_token_id)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")


def demo_generation():
    """Demonstrate text generation."""
    print("\n" + "=" * 60)
    print("Demo: Text Generation")
    print("=" * 60)

    config = Seq2SeqConfig(
        vocab_size=1000,
        d_model=128,
        d_ff=256,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        max_seq_len=64
    )

    model = Seq2SeqModel(config)

    # Source input
    input_ids = torch.randint(4, 1000, (1, 20))
    attention_mask = torch.ones(1, 20)

    print(f"\nSource tokens: {input_ids[0].tolist()}")

    # Greedy generation
    generated = model.generate(
        input_ids,
        attention_mask,
        max_length=30,
        num_beams=1
    )
    print(f"Greedy output: {generated[0].tolist()}")

    # Beam search
    generated_beam = model.generate(
        input_ids,
        attention_mask,
        max_length=30,
        num_beams=3
    )
    print(f"Beam search output: {generated_beam[0].tolist()}")


def demo_training():
    """Demonstrate training loop."""
    print("\n" + "=" * 60)
    print("Demo: Training Loop")
    print("=" * 60)

    config = Seq2SeqConfig(
        vocab_size=1000,
        d_model=128,
        d_ff=256,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        max_seq_len=64
    )

    model = Seq2SeqModel(config)

    # Create dummy dataset
    src_texts = [f"source text number {i}" for i in range(50)]
    tgt_texts = [f"target {i}" for i in range(50)]

    dataset = SimpleSeq2SeqDataset(src_texts, tgt_texts, max_source_len=32, max_target_len=16)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Setup trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Seq2SeqTrainer(
        model, optimizer,
        use_label_smoothing=True,
        gradient_clip=1.0
    )

    # Train for a few steps
    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        epoch_loss = 0
        for batch in dataloader:
            loss = trainer.train_step(batch)
            epoch_loss += loss

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    # Evaluate
    eval_results = trainer.evaluate(dataloader)
    print(f"\nEvaluation: Loss = {eval_results['loss']:.4f}, PPL = {eval_results['perplexity']:.2f}")


if __name__ == "__main__":
    print("Seq2Seq Models: T5 & BART - Hands-on Implementation")
    print("=" * 60)

    demo_span_corruption()
    demo_bart_denoising()
    demo_model_forward()
    demo_generation()
    demo_training()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
