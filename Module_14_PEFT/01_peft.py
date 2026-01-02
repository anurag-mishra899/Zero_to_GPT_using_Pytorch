"""
Module 14: Parameter-Efficient Fine-Tuning (PEFT) - Implementation

This module covers:
1. LoRA (Low-Rank Adaptation)
2. QLoRA concepts
3. Adapters
4. Prefix Tuning
5. Complete training examples
"""

import math
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Section 1: LoRA Implementation
# ============================================================================

class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation.

    Instead of fine-tuning full weight matrix W, we learn:
    W' = W + (alpha/r) * B @ A

    Where:
    - W: Original frozen weights (d_out x d_in)
    - A: Down projection (r x d_in)
    - B: Up projection (d_out x r)
    - r: Rank (much smaller than d_in, d_out)
    - alpha: Scaling factor

    The original forward is: y = Wx
    With LoRA: y = Wx + (alpha/r) * BAx
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merge_weights = merge_weights

        # Original frozen weights
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # LoRA matrices
        # A: (r, in_features) - projects to low rank
        # B: (out_features, r) - projects back to full dimension
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Flag for merged state
        self.merged = False

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize LoRA parameters.

        Critical: B is initialized to zero, so LoRA contribution is zero at start.
        This means fine-tuning starts exactly from pretrained weights.
        """
        # Initialize A with scaled normal distribution
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Initialize B to zero - crucial for starting from pretrained
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge LoRA weights into base weights for inference."""
        if not self.merged:
            # W_merged = W + (alpha/r) * B @ A
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights for continued training."""
        if self.merged:
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        If merged: Just use merged weights
        If not merged: y = Wx + (alpha/r) * B(A(x))
        """
        if self.merged:
            # Use merged weights (no LoRA computation)
            return F.linear(x, self.weight, self.bias)

        # Original path (frozen)
        original_output = F.linear(x, self.weight, self.bias)

        # LoRA path
        # x: (..., in_features)
        # A: (r, in_features) -> A @ x.T or x @ A.T
        # After A: (..., r)
        # B: (out_features, r) -> B @ ... or ... @ B.T
        # After B: (..., out_features)

        x_dropped = self.dropout(x)
        lora_output = F.linear(F.linear(x_dropped, self.lora_A), self.lora_B)

        return original_output + self.scaling * lora_output

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count trainable and total parameters in a model with LoRA.

    Returns:
        (trainable_params, total_params, percentage)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    return trainable, total, percentage


# ============================================================================
# Section 2: LoRA-Enhanced Transformer
# ============================================================================

class LoRAMultiHeadAttention(nn.Module):
    """
    Multi-head attention with LoRA on Q and V projections.

    This is the typical configuration - LoRA on query and value.
    Key projection usually doesn't benefit as much from LoRA.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        apply_lora_to: List[str] = ["q", "v"]
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.apply_lora_to = apply_lora_to

        # Projections - using LoRA for specified ones
        if "q" in apply_lora_to:
            self.q_proj = LoRALinear(d_model, d_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.q_proj = nn.Linear(d_model, d_model)

        if "k" in apply_lora_to:
            self.k_proj = LoRALinear(d_model, d_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.k_proj = nn.Linear(d_model, d_model)

        if "v" in apply_lora_to:
            self.v_proj = LoRALinear(d_model, d_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.v_proj = nn.Linear(d_model, d_model)

        if "o" in apply_lora_to:
            self.o_proj = LoRALinear(d_model, d_model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.o_proj(attn_output)

    def freeze_base_weights(self):
        """Freeze non-LoRA parameters for fine-tuning."""
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False


class LoRATransformerBlock(nn.Module):
    """Transformer block with LoRA-enhanced attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        lora_r: int = 8,
        lora_alpha: float = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        # LoRA attention
        self.attention = LoRAMultiHeadAttention(
            d_model, num_heads,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            apply_lora_to=["q", "v"]
        )

        # Standard FFN (can also add LoRA here)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.dropout(self.attention(self.norm1(x), mask))

        # FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# ============================================================================
# Section 3: Adapters
# ============================================================================

class Adapter(nn.Module):
    """
    Adapter module for PEFT.

    Architecture:
    Input -> Down project -> Nonlinearity -> Up project -> Add residual

    This is inserted after attention and FFN in transformer blocks.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 1e-3
    ):
        super().__init__()

        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize up projection near zero for residual stability
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        # Down project
        hidden = self.down_proj(x)

        # Nonlinearity
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        # Up project
        output = self.up_proj(hidden)

        # Residual
        return x + output


class AdapterTransformerBlock(nn.Module):
    """Transformer block with adapters."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        adapter_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # Standard attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Standard FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # Adapters - one after attention, one after FFN
        self.adapter_attn = Adapter(d_model, adapter_dim)
        self.adapter_ffn = Adapter(d_model, adapter_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Adapter after attention
        x = self.adapter_attn(x)

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        # Adapter after FFN
        x = self.adapter_ffn(x)

        return x

    def freeze_non_adapter(self):
        """Freeze everything except adapters."""
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False


# ============================================================================
# Section 4: Prefix Tuning
# ============================================================================

class PrefixTuning(nn.Module):
    """
    Prefix Tuning for PEFT.

    Adds learnable prefix tokens to keys and values in attention.
    These prefixes are task-specific while the model remains frozen.

    K = [K_prefix; K_input]
    V = [V_prefix; V_input]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        prefix_len: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()

        self.prefix_len = prefix_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # Learnable prefix for each layer
        # Shape: (num_layers, 2, prefix_len, num_heads, head_dim)
        # 2 is for key and value
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_layers, 2, prefix_len, num_heads, self.head_dim) * 0.02
        )

        # Optional MLP for reparameterization (helps training)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def get_prefix(self, layer_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prefix keys and values for a specific layer.

        Returns:
            prefix_k: (batch_size, num_heads, prefix_len, head_dim)
            prefix_v: (batch_size, num_heads, prefix_len, head_dim)
        """
        # Get prefix for this layer
        prefix = self.prefix_embeddings[layer_idx]  # (2, prefix_len, num_heads, head_dim)

        prefix_k = prefix[0]  # (prefix_len, num_heads, head_dim)
        prefix_v = prefix[1]

        # Expand for batch
        prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, prefix_len, heads, head_dim)
        prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Transpose to (batch, heads, prefix_len, head_dim)
        prefix_k = prefix_k.transpose(1, 2)
        prefix_v = prefix_v.transpose(1, 2)

        return self.dropout(prefix_k), self.dropout(prefix_v)


class PrefixAttention(nn.Module):
    """Multi-head attention with prefix tuning."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        prefix_tuning: PrefixTuning,
        layer_idx: int
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx
        self.prefix_tuning = prefix_tuning

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Standard Q, K, V projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Get prefix keys and values
        prefix_k, prefix_v = self.prefix_tuning.get_prefix(self.layer_idx, batch_size)

        # Concatenate prefix with input K, V
        # K: (batch, heads, prefix_len + seq_len, head_dim)
        K = torch.cat([prefix_k, K], dim=2)
        V = torch.cat([prefix_v, V], dim=2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Extend mask for prefix if provided
        if mask is not None:
            prefix_len = prefix_k.shape[2]
            prefix_mask = torch.ones(batch_size, 1, seq_len, prefix_len, device=x.device)
            mask = torch.cat([prefix_mask, mask], dim=-1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)


# ============================================================================
# Section 5: Prompt Tuning
# ============================================================================

class PromptTuning(nn.Module):
    """
    Prompt Tuning for PEFT.

    Learns soft prompt embeddings that are prepended to the input.
    Only the prompt embeddings are trained; everything else is frozen.

    Input: [P1, P2, ..., Pn, x1, x2, ..., xm]
    Where P1..Pn are learnable soft prompts
    """

    def __init__(
        self,
        num_prompt_tokens: int,
        d_model: int,
        init_from_vocab: bool = False,
        vocab_size: Optional[int] = None,
        embedding_layer: Optional[nn.Embedding] = None
    ):
        super().__init__()

        self.num_prompt_tokens = num_prompt_tokens
        self.d_model = d_model

        if init_from_vocab and embedding_layer is not None:
            # Initialize from random vocab tokens
            random_indices = torch.randint(0, vocab_size, (num_prompt_tokens,))
            self.prompt_embeddings = nn.Parameter(
                embedding_layer.weight[random_indices].clone()
            )
        else:
            # Random initialization
            self.prompt_embeddings = nn.Parameter(
                torch.randn(num_prompt_tokens, d_model) * 0.02
            )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompts to input embeddings.

        Args:
            input_embeddings: (batch, seq_len, d_model)

        Returns:
            (batch, num_prompts + seq_len, d_model)
        """
        batch_size = input_embeddings.shape[0]

        # Expand prompts for batch
        prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prompts with input
        return torch.cat([prompts, input_embeddings], dim=1)


# ============================================================================
# Section 6: QLoRA Components (Simulation)
# ============================================================================

class QuantizedLinear(nn.Module):
    """
    Simulated 4-bit quantized linear layer for QLoRA.

    In real QLoRA:
    - Weights stored in 4-bit NF4 format
    - Dequantized to FP16 during forward
    - Only LoRA matrices receive gradients

    This is a simulation that demonstrates the concept.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 64
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size

        # Simulated quantized weights (stored as int8 for demo)
        # Real implementation would use custom CUDA kernels
        num_blocks = (in_features * out_features + block_size - 1) // block_size
        self.quantized_weight = nn.Parameter(
            torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )

        # Scales for each block
        self.scales = nn.Parameter(
            torch.ones(num_blocks),
            requires_grad=False
        )

        # Zero points for asymmetric quantization
        self.zero_points = nn.Parameter(
            torch.zeros(num_blocks),
            requires_grad=False
        )

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def dequantize(self) -> torch.Tensor:
        """Dequantize weights to FP16."""
        # Simple linear dequantization for demo
        # Real NF4 uses special quantization levels
        weight_fp = self.quantized_weight.float()

        # Apply scale (simplified - real version is block-wise)
        scale = self.scales.mean()
        weight_fp = weight_fp * scale / 8.0  # Normalize to reasonable range

        return weight_fp.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dequantized weights."""
        # Dequantize on-the-fly
        weight = self.dequantize()

        # Convert input to match weight dtype
        x = x.to(weight.dtype)

        return F.linear(x, weight, self.bias.to(weight.dtype))

    def memory_footprint(self) -> Dict[str, float]:
        """Calculate memory footprint."""
        # 4-bit = 0.5 bytes per weight
        weight_memory = self.in_features * self.out_features * 0.5 / (1024 * 1024)

        # Scales and zero points
        num_blocks = len(self.scales)
        scale_memory = num_blocks * 4 / (1024 * 1024)  # FP32

        return {
            'weight_mb': weight_memory,
            'scale_mb': scale_memory,
            'total_mb': weight_memory + scale_memory
        }


class QLoRALinear(nn.Module):
    """
    Complete QLoRA layer: Quantized base + FP16 LoRA.

    The base weights are quantized to 4-bit and frozen.
    LoRA matrices remain in FP16 and are trainable.
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

        # Quantized base layer (frozen)
        self.base = QuantizedLinear(in_features, out_features)

        # FP16 LoRA (trainable)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: dequantized base + LoRA."""
        # Base path (4-bit dequantized)
        base_output = self.base(x)

        # LoRA path (FP16)
        x_fp16 = x.half()
        lora_output = F.linear(F.linear(self.dropout(x_fp16), self.lora_A.half()), self.lora_B.half())

        return base_output + self.scaling * lora_output.to(base_output.dtype)


# ============================================================================
# Section 7: Complete PEFT Model Example
# ============================================================================

class PEFTLanguageModel(nn.Module):
    """
    Complete language model with PEFT support.

    Supports:
    - LoRA
    - Adapters
    - Prefix Tuning
    - Prompt Tuning

    Can be configured to use any combination.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        # PEFT options
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: float = 16,
        use_adapter: bool = False,
        adapter_dim: int = 64,
        use_prefix: bool = False,
        prefix_len: int = 10,
        use_prompt: bool = False,
        num_prompt_tokens: int = 10
    ):
        super().__init__()

        self.d_model = d_model
        self.use_prefix = use_prefix
        self.use_prompt = use_prompt

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Prompt tuning
        if use_prompt:
            self.prompt_tuning = PromptTuning(num_prompt_tokens, d_model)
        else:
            self.prompt_tuning = None

        # Prefix tuning
        if use_prefix:
            self.prefix_tuning = PrefixTuning(d_model, num_heads, num_layers, prefix_len)
        else:
            self.prefix_tuning = None

        # Transformer blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_lora:
                layer = LoRATransformerBlock(
                    d_model, num_heads, d_ff,
                    lora_r=lora_r, lora_alpha=lora_alpha
                )
            elif use_adapter:
                layer = AdapterTransformerBlock(
                    d_model, num_heads, d_ff,
                    adapter_dim=adapter_dim
                )
            else:
                # Standard transformer block
                layer = nn.TransformerEncoderLayer(
                    d_model, num_heads, d_ff,
                    dropout=0.1, batch_first=True
                )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Apply prompt tuning (prepend soft prompts)
        if self.prompt_tuning is not None:
            x = self.prompt_tuning(x)
            seq_len = x.shape[1]  # Updated seq_len

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = ~mask  # Invert for attention

        # Forward through layers
        for layer in self.layers:
            if isinstance(layer, (LoRATransformerBlock, AdapterTransformerBlock)):
                x = layer(x, mask)
            else:
                x = layer(x, src_mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        output = {'logits': logits}

        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss

        return output

    def freeze_base_model(self):
        """Freeze all non-PEFT parameters."""
        for name, param in self.named_parameters():
            is_peft = any(peft_name in name for peft_name in [
                'lora_', 'adapter', 'prefix', 'prompt'
            ])
            if not is_peft:
                param.requires_grad = False

    def get_trainable_params_info(self) -> str:
        """Get info about trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"


# ============================================================================
# Section 8: Training Utilities
# ============================================================================

def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base weights for inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge LoRA weights for continued training."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA weights."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state[name] = param.data.clone()
    torch.save(lora_state, path)
    print(f"Saved LoRA weights to {path}")


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA weights into model."""
    lora_state = torch.load(path)
    model_state = model.state_dict()

    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)

    print(f"Loaded LoRA weights from {path}")


# ============================================================================
# Section 9: Demo Functions
# ============================================================================

def demo_lora():
    """Demonstrate LoRA basics."""
    print("=" * 60)
    print("LoRA Demonstration")
    print("=" * 60)

    # Create LoRA linear layer
    in_features, out_features = 512, 512
    rank = 8

    lora_layer = LoRALinear(in_features, out_features, r=rank, alpha=16)

    # Initialize base weights (simulating pretrained)
    nn.init.normal_(lora_layer.weight, std=0.02)

    print(f"\nLayer dimensions: {in_features} -> {out_features}")
    print(f"LoRA rank: {rank}")

    # Parameter count
    full_params = in_features * out_features + out_features
    lora_params = rank * in_features + out_features * rank
    print(f"\nFull parameters: {full_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Parameter reduction: {full_params/lora_params:.1f}x")

    # Test forward pass
    x = torch.randn(2, 10, in_features)  # batch=2, seq=10
    y = lora_layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test merging
    print("\nTesting weight merging...")
    y_before_merge = lora_layer(x).clone()
    lora_layer.merge()
    y_after_merge = lora_layer(x)
    print(f"Output same after merge: {torch.allclose(y_before_merge, y_after_merge, atol=1e-5)}")


def demo_complete_model():
    """Demonstrate complete PEFT model."""
    print("\n" + "=" * 60)
    print("Complete PEFT Model Demonstration")
    print("=" * 60)

    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 2

    # Create model with LoRA
    model = PEFTLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        use_lora=True,
        lora_r=8,
        use_adapter=False,
        use_prefix=False,
        use_prompt=False
    )

    # Freeze base model
    model.freeze_base_model()

    print("\nModel with LoRA:")
    print(model.get_trainable_params_info())

    # Create model with adapters
    model_adapter = PEFTLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        use_lora=False,
        use_adapter=True,
        adapter_dim=32
    )
    model_adapter.freeze_base_model()
    print("\nModel with Adapters:")
    print(model_adapter.get_trainable_params_info())

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (2, 16))
    labels = torch.randint(0, vocab_size, (2, 16))

    output = model(input_ids, labels)
    print(f"\nForward pass:")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")


def demo_qlora_memory():
    """Demonstrate QLoRA memory savings."""
    print("\n" + "=" * 60)
    print("QLoRA Memory Comparison")
    print("=" * 60)

    in_features = 4096
    out_features = 4096

    # Standard linear
    standard = nn.Linear(in_features, out_features)
    standard_memory = sum(p.numel() * p.element_size() for p in standard.parameters())

    # Quantized linear
    quantized = QuantizedLinear(in_features, out_features)
    quant_memory = quantized.memory_footprint()

    # QLoRA
    qlora = QLoRALinear(in_features, out_features, r=8)
    lora_memory = sum(
        p.numel() * p.element_size()
        for p in qlora.parameters()
        if 'lora_' in str(type(p)) or p.requires_grad
    )

    print(f"\nLayer: {in_features} -> {out_features}")
    print(f"\nFP32 Linear: {standard_memory / 1024 / 1024:.2f} MB")
    print(f"4-bit Quantized: {quant_memory['total_mb']:.2f} MB")
    print(f"Memory reduction: {standard_memory / 1024 / 1024 / quant_memory['total_mb']:.1f}x")


def demo_peft_comparison():
    """Compare different PEFT methods."""
    print("\n" + "=" * 60)
    print("PEFT Methods Comparison")
    print("=" * 60)

    vocab_size = 32000
    d_model = 512
    num_heads = 8
    num_layers = 6

    configs = [
        ("Full Fine-tuning", dict(use_lora=False, use_adapter=False, use_prefix=False, use_prompt=False)),
        ("LoRA (r=8)", dict(use_lora=True, lora_r=8, use_adapter=False, use_prefix=False, use_prompt=False)),
        ("LoRA (r=64)", dict(use_lora=True, lora_r=64, use_adapter=False, use_prefix=False, use_prompt=False)),
        ("Adapter (d=64)", dict(use_lora=False, use_adapter=True, adapter_dim=64, use_prefix=False, use_prompt=False)),
        ("Prompt (n=20)", dict(use_lora=False, use_adapter=False, use_prefix=False, use_prompt=True, num_prompt_tokens=20)),
    ]

    print(f"\nBase model: {num_layers} layers, d_model={d_model}, {vocab_size} vocab")
    print("-" * 60)
    print(f"{'Method':<25} {'Trainable':>15} {'Total':>15} {'Ratio':>10}")
    print("-" * 60)

    for name, config in configs:
        model = PEFTLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            **config
        )

        if name != "Full Fine-tuning":
            model.freeze_base_model()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = trainable / total * 100

        print(f"{name:<25} {trainable:>15,} {total:>15,} {ratio:>9.2f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("Module 14: Parameter-Efficient Fine-Tuning (PEFT)")
    print("=" * 60)

    demo_lora()
    demo_complete_model()
    demo_qlora_memory()
    demo_peft_comparison()

    print("\n" + "=" * 60)
    print("Module 14 Complete!")
    print("=" * 60)
    print("\nKey concepts covered:")
    print("1. LoRA - Low-Rank Adaptation with trainable A, B matrices")
    print("2. QLoRA - 4-bit quantized base + FP16 LoRA")
    print("3. Adapters - Bottleneck layers inserted in transformer")
    print("4. Prefix Tuning - Learnable prefix for K, V in attention")
    print("5. Prompt Tuning - Soft prompt tokens prepended to input")
    print("6. Memory and parameter comparisons")


if __name__ == "__main__":
    main()
