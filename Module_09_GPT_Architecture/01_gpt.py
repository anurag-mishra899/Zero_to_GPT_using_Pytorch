"""
Module 9.1: GPT Architecture - Decoder-Only Transformers
Complete GPT-2 style implementation with generation

Covers:
- GPT architecture (decoder-only transformer)
- Causal attention
- Training with next-token prediction
- Text generation strategies
- KV-cache for efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import math
from dataclasses import dataclass

print("=" * 70)
print("Module 9.1: GPT Architecture - Decoder-Only Transformers")
print("=" * 70)


# ===========================================================================
# Section 1: GPT Configuration
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: GPT Configuration")
print("=" * 70)


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 50257  # GPT-2 vocab size
    max_position_embeddings: int = 1024
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_weights: bool = True


# Predefined configurations
GPT2_CONFIGS = {
    'gpt2-small': GPTConfig(
        d_model=768, num_heads=12, num_layers=12, d_ff=3072
    ),
    'gpt2-medium': GPTConfig(
        d_model=1024, num_heads=16, num_layers=24, d_ff=4096
    ),
    'gpt2-large': GPTConfig(
        d_model=1280, num_heads=20, num_layers=36, d_ff=5120
    ),
    'gpt2-xl': GPTConfig(
        d_model=1600, num_heads=25, num_layers=48, d_ff=6400
    ),
}

# Print configurations
print("\nGPT-2 Model Configurations:")
print(f"{'Model':<15} {'d_model':<10} {'heads':<8} {'layers':<8} {'d_ff':<8}")
print("-" * 55)
for name, config in GPT2_CONFIGS.items():
    print(f"{name:<15} {config.d_model:<10} {config.num_heads:<8} "
          f"{config.num_layers:<8} {config.d_ff:<8}")


# ===========================================================================
# Section 2: Core Components
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: Core Components")
print("=" * 70)


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for GPT.

    Each position can only attend to previous positions.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        self.scale = math.sqrt(self.d_k)

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.max_position_embeddings,
                                  config.max_position_embeddings)).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            layer_past: Optional tuple of (past_key, past_value) for KV-cache
            use_cache: Whether to return new KV-cache

        Returns:
            output: (batch, seq_len, d_model)
            present: Optional new KV-cache
        """
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Handle KV-cache
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        present = (k, v) if use_cache else None

        # Compute attention scores
        # (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k) -> (batch, heads, seq_q, seq_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal mask
        key_len = k.size(-2)
        query_len = q.size(-2)
        causal_mask = self.bias[:, :, key_len - query_len:key_len, :key_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply attention to values
        output = torch.matmul(attn_probs, v)

        # Reshape: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.c_proj(output)
        output = self.resid_dropout(output)

        return output, present


class MLP(nn.Module):
    """
    GPT-2 style MLP (Feed-Forward Network).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    """
    Single GPT transformer block.

    Pre-LayerNorm architecture:
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Self-attention with residual
        attn_out, present = self.attn(self.ln_1(x), layer_past, use_cache)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.ln_2(x))

        return x, present


# Test components
print("\n--- Testing Core Components ---")
config = GPTConfig(d_model=256, num_heads=4, num_layers=4, d_ff=1024,
                   max_position_embeddings=512, vocab_size=1000)

attn = CausalSelfAttention(config)
mlp = MLP(config)
block = GPTBlock(config)

x = torch.randn(2, 20, config.d_model)
attn_out, _ = attn(x)
mlp_out = mlp(x)
block_out, _ = block(x)

print(f"Input: {x.shape}")
print(f"Attention output: {attn_out.shape}")
print(f"MLP output: {mlp_out.shape}")
print(f"Block output: {block_out.shape}")


# ===========================================================================
# Section 3: Complete GPT Model
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: Complete GPT Model")
print("=" * 70)


class GPT(nn.Module):
    """
    Complete GPT-2 style language model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids: (batch, seq_len)
            past_key_values: Optional list of KV-cache for each layer
            use_cache: Whether to return new KV-cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            present_key_values: Optional new KV-cache
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Get past length for position encoding
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
        else:
            past_length = 0

        # Position indices
        positions = torch.arange(past_length, past_length + seq_len, device=device)

        # Embeddings
        token_emb = self.wte(input_ids)
        pos_emb = self.wpe(positions)
        x = self.drop(token_emb + pos_emb)

        # Transformer blocks
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, layer_past, use_cache)
            if use_cache:
                present_key_values.append(present)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)

        return logits, present_key_values

    def get_num_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


# Test GPT model
print("\n--- Testing GPT Model ---")
config = GPTConfig(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    max_position_embeddings=1024
)

model = GPT(config)
print(f"GPT-2 Small architecture")
print(f"Total parameters: {model.get_num_params():,}")

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 64))
logits, _ = model(input_ids)
print(f"Input: {input_ids.shape}")
print(f"Logits: {logits.shape}")


# ===========================================================================
# Section 4: Training
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: Training")
print("=" * 70)


class GPTTrainer:
    """Training utilities for GPT."""

    def __init__(
        self,
        model: GPT,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 1000,
        max_steps: int = 100000
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # Separate weight decay for different parameter types
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'ln' in name or 'bias' in name or 'wpe' in name or 'wte' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=betas)

        self.step = 0

    def get_lr(self) -> float:
        """Cosine decay with warmup."""
        if self.step < self.warmup_steps:
            return self.optimizer.defaults['lr'] * self.step / self.warmup_steps
        else:
            progress = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.optimizer.defaults['lr'] * 0.5 * (1 + math.cos(math.pi * progress))

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Update learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        logits, _ = self.model(input_ids)

        # Compute loss (shift logits and labels)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update
        self.optimizer.step()
        self.step += 1

        return loss.item()


# Test training
print("\n--- Testing Training ---")
small_config = GPTConfig(
    vocab_size=1000,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_position_embeddings=256
)

small_model = GPT(small_config)
trainer = GPTTrainer(small_model, learning_rate=1e-3, warmup_steps=10, max_steps=100)

# Simulate training
for step in range(5):
    batch = torch.randint(0, small_config.vocab_size, (4, 32))
    loss = trainer.train_step(batch, batch)  # Labels = input (causal LM)
    print(f"Step {step}: loss = {loss:.4f}, lr = {trainer.get_lr():.6f}")


# ===========================================================================
# Section 5: Text Generation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Text Generation")
print("=" * 70)


class GPTGenerator:
    """Text generation utilities for GPT."""

    def __init__(self, model: GPT):
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            repetition_penalty: Penalty for repeated tokens
            use_cache: Use KV-cache for efficiency

        Returns:
            generated: Complete sequence (batch, seq_len + max_new_tokens)
        """
        self.model.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Track generated tokens for repetition penalty
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Get input for this step
            if use_cache and past_key_values is not None:
                # Only need the last token if using cache
                current_input = generated[:, -1:]
            else:
                current_input = generated

            # Forward pass
            logits, past_key_values = self.model(
                current_input,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache
            )

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_logits[i, token_id] /= repetition_penalty

            # Temperature scaling
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def generate_beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        length_penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Beam search generation.
        """
        self.model.eval()
        batch_size = input_ids.size(0)
        assert batch_size == 1, "Beam search only supports batch_size=1"

        device = input_ids.device
        vocab_size = self.model.config.vocab_size

        # Initialize beams
        # beam_scores: (num_beams,)
        # beam_tokens: (num_beams, seq_len)
        beam_scores = torch.zeros(num_beams, device=device)
        beam_tokens = input_ids.expand(num_beams, -1).clone()

        for step in range(max_new_tokens):
            # Forward pass for all beams
            logits, _ = self.model(beam_tokens)
            next_logits = logits[:, -1, :]  # (num_beams, vocab_size)

            # Compute scores for all possible next tokens
            next_scores = F.log_softmax(next_logits, dim=-1)  # (num_beams, vocab_size)

            # Add beam scores
            next_scores = next_scores + beam_scores.unsqueeze(-1)

            # Flatten and get top-k
            next_scores = next_scores.view(-1)  # (num_beams * vocab_size)
            top_scores, top_indices = torch.topk(next_scores, num_beams)

            # Convert to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beams
            beam_tokens = torch.cat([
                beam_tokens[beam_indices],
                token_indices.unsqueeze(-1)
            ], dim=-1)
            beam_scores = top_scores

        # Apply length penalty
        final_scores = beam_scores / (beam_tokens.size(1) ** length_penalty)

        # Return best beam
        best_beam = final_scores.argmax()
        return beam_tokens[best_beam].unsqueeze(0)


# Test generation
print("\n--- Testing Generation ---")
generator = GPTGenerator(small_model)

input_ids = torch.randint(0, small_config.vocab_size, (1, 10))
print(f"Input: {input_ids.shape}")

# Different sampling strategies
generated_greedy = generator.generate(input_ids, max_new_tokens=20, temperature=0.0001)
generated_temp = generator.generate(input_ids, max_new_tokens=20, temperature=0.8)
generated_topk = generator.generate(input_ids, max_new_tokens=20, temperature=0.8, top_k=50)
generated_topp = generator.generate(input_ids, max_new_tokens=20, temperature=0.8, top_p=0.9)

print(f"Greedy output: {generated_greedy.shape}")
print(f"Temperature sampling: {generated_temp.shape}")
print(f"Top-k sampling: {generated_topk.shape}")
print(f"Top-p sampling: {generated_topp.shape}")


# ===========================================================================
# Section 6: KV-Cache Analysis
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: KV-Cache Analysis")
print("=" * 70)


def analyze_kv_cache():
    """Analyze KV-cache memory and speedup."""

    print("\nKV-Cache Memory Analysis:")
    print("-" * 50)

    configs = [
        ('GPT-2 Small', 768, 12, 1024),
        ('GPT-2 Medium', 1024, 24, 1024),
        ('GPT-2 Large', 1280, 36, 1024),
        ('GPT-2 XL', 1600, 48, 1024),
        ('GPT-3 175B', 12288, 96, 2048),
    ]

    print(f"{'Model':<15} {'KV per layer':<15} {'Total KV (GB)':<15}")
    print("-" * 45)

    for name, d_model, n_layers, max_len in configs:
        # Per layer: 2 (K, V) * seq_len * d_model * 2 bytes (fp16)
        kv_per_layer = 2 * max_len * d_model * 2  # bytes
        total_kv = kv_per_layer * n_layers / (1024 ** 3)  # GB

        print(f"{name:<15} {kv_per_layer/1024:.1f} KB{'':<5} {total_kv:.3f}")

    # Speed comparison
    print("\n\nGeneration Speed Comparison:")
    print("-" * 50)

    import time

    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_position_embeddings=512
    )
    model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 10))
    generator = GPTGenerator(model)

    # Without cache
    start = time.time()
    with torch.no_grad():
        _ = generator.generate(input_ids, max_new_tokens=50, use_cache=False)
    no_cache_time = time.time() - start

    # With cache
    start = time.time()
    with torch.no_grad():
        _ = generator.generate(input_ids, max_new_tokens=50, use_cache=True)
    cache_time = time.time() - start

    print(f"Without KV-cache: {no_cache_time*1000:.1f} ms")
    print(f"With KV-cache: {cache_time*1000:.1f} ms")
    print(f"Speedup: {no_cache_time/cache_time:.2f}x")

analyze_kv_cache()


# ===========================================================================
# Section 7: Perplexity Calculation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 7: Perplexity Calculation")
print("=" * 70)


def calculate_perplexity(
    model: GPT,
    input_ids: torch.Tensor,
    stride: int = 512
) -> float:
    """
    Calculate perplexity on a sequence.

    Args:
        model: GPT model
        input_ids: Token IDs (1, seq_len)
        stride: Stride for sliding window

    Returns:
        perplexity: exp(average_loss)
    """
    model.eval()
    seq_len = input_ids.size(1)
    max_len = model.config.max_position_embeddings

    nlls = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_len, seq_len)
        target_len = end_loc - begin_loc
        input_chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            logits, _ = model(input_chunk)

            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_chunk[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )

            nlls.append(loss)

        if end_loc == seq_len:
            break

    # Compute perplexity
    all_nlls = torch.cat(nlls)
    perplexity = torch.exp(all_nlls.mean()).item()

    return perplexity


# Test perplexity calculation
print("\n--- Testing Perplexity Calculation ---")
test_sequence = torch.randint(0, small_config.vocab_size, (1, 200))
ppl = calculate_perplexity(small_model, test_sequence, stride=100)
print(f"Test perplexity: {ppl:.2f}")
print(f"(Random model should have PPL close to vocab_size = {small_config.vocab_size})")


# ===========================================================================
# Section 8: Attention Visualization
# ===========================================================================
print("\n" + "=" * 70)
print("Section 8: Attention Visualization")
print("=" * 70)


def visualize_attention_patterns(
    model: GPT,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    save_path: Optional[str] = None
):
    """Visualize attention patterns for a given layer."""
    model.eval()

    # Hook to capture attention weights
    attention_weights = []

    def hook_fn(module, input, output):
        # Get attention from the output
        # Need to access internal attention computation
        pass

    # For visualization, we'll compute attention manually
    with torch.no_grad():
        x = model.wte(input_ids) + model.wpe(torch.arange(input_ids.size(1)))
        x = model.drop(x)

        for i, block in enumerate(model.blocks):
            if i == layer_idx:
                # Get Q, K for attention visualization
                ln_out = block.ln_1(x)
                qkv = block.attn.c_attn(ln_out)
                q, k, v = qkv.split(model.config.d_model, dim=-1)

                batch_size, seq_len = input_ids.shape
                num_heads = model.config.num_heads
                d_k = model.config.d_model // num_heads

                q = q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
                k = k.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

                # Apply causal mask
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))

                attn_weights = F.softmax(scores, dim=-1)
                break

            x, _ = block(x)

    # Visualize
    attn_weights = attn_weights[0].detach().numpy()  # First batch item
    num_heads = attn_weights.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(min(4, num_heads)):
        ax = axes[i]
        im = ax.imshow(attn_weights[i], cmap='Blues', aspect='auto')
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Attention Patterns (Layer {layer_idx})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    plt.close()


# Visualize attention
input_ids = torch.randint(0, small_config.vocab_size, (1, 20))
visualize_attention_patterns(
    small_model, input_ids, layer_idx=0,
    save_path='/Users/anuragmishra/Documents/Zero_to_GPT/Module_09_GPT_Architecture/attention_patterns.png'
)


# ===========================================================================
# Section 9: Sampling Strategy Comparison
# ===========================================================================
print("\n" + "=" * 70)
print("Section 9: Sampling Strategy Comparison")
print("=" * 70)


def compare_sampling_strategies():
    """Compare different sampling strategies."""

    # Create simple logits distribution
    vocab_size = 100
    logits = torch.randn(vocab_size)

    # Make some tokens more likely
    logits[0:10] += 3  # Top tokens

    print("\nSampling Strategy Comparison:")
    print("-" * 60)

    strategies = [
        ('Greedy', lambda l: F.softmax(l / 0.001, dim=-1)),
        ('T=0.5', lambda l: F.softmax(l / 0.5, dim=-1)),
        ('T=1.0', lambda l: F.softmax(l / 1.0, dim=-1)),
        ('T=1.5', lambda l: F.softmax(l / 1.5, dim=-1)),
    ]

    probs_original = F.softmax(logits, dim=-1)

    print(f"{'Strategy':<15} {'Top-1 Prob':<12} {'Top-10 Prob':<12} {'Entropy':<12}")
    print("-" * 55)

    for name, transform in strategies:
        probs = transform(logits)
        top1 = probs.max().item()
        top10 = probs[:10].sum().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        print(f"{name:<15} {top1:<12.4f} {top10:<12.4f} {entropy:<12.4f}")

    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, transform) in enumerate(strategies):
        ax = axes[idx // 2, idx % 2]
        probs = transform(logits).numpy()
        ax.bar(range(vocab_size), probs, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Token ID')
        ax.set_ylabel('Probability')
        ax.set_xlim(-1, 30)  # Show first 30 tokens

    plt.tight_layout()
    plt.savefig('/Users/anuragmishra/Documents/Zero_to_GPT/Module_09_GPT_Architecture/sampling_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved sampling comparison visualization")

compare_sampling_strategies()


# ===========================================================================
# Section 10: Model Size Estimation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 10: Model Size Estimation")
print("=" * 70)


def estimate_model_size(config: GPTConfig) -> Dict[str, int]:
    """Estimate parameter counts for each component."""
    V = config.vocab_size
    D = config.d_model
    H = config.num_heads
    L = config.num_layers
    F = config.d_ff

    params = {
        'token_embedding': V * D,
        'position_embedding': config.max_position_embeddings * D,
        'attention_per_layer': 4 * D * D,  # Q, K, V, O projections
        'mlp_per_layer': D * F + F * D,  # Two linear layers
        'ln_per_layer': 4 * D,  # 2 layer norms with weight + bias
        'final_ln': 2 * D,
    }

    params['total_per_layer'] = params['attention_per_layer'] + params['mlp_per_layer'] + params['ln_per_layer']
    params['all_layers'] = params['total_per_layer'] * L
    params['lm_head'] = 0 if config.tie_weights else V * D

    params['total'] = (
        params['token_embedding'] +
        params['position_embedding'] +
        params['all_layers'] +
        params['final_ln'] +
        params['lm_head']
    )

    return params


# Print model size breakdown
print("\nModel Size Breakdown (GPT-2 Small):")
print("-" * 50)

config = GPT2_CONFIGS['gpt2-small']
params = estimate_model_size(config)

print(f"Token embedding: {params['token_embedding']:,}")
print(f"Position embedding: {params['position_embedding']:,}")
print(f"Attention per layer: {params['attention_per_layer']:,}")
print(f"MLP per layer: {params['mlp_per_layer']:,}")
print(f"Total per layer: {params['total_per_layer']:,}")
print(f"All {config.num_layers} layers: {params['all_layers']:,}")
print(f"Total parameters: {params['total']:,}")

# Compare all GPT-2 sizes
print("\n\nGPT-2 Model Sizes:")
print("-" * 40)
print(f"{'Model':<15} {'Parameters':<15}")
print("-" * 30)

for name, config in GPT2_CONFIGS.items():
    params = estimate_model_size(config)
    print(f"{name:<15} {params['total']:,}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 9.1 Summary: GPT Architecture")
print("=" * 70)

print("""
Key Takeaways:
==============

1. GPT Architecture:
   - Decoder-only transformer
   - Causal (masked) self-attention
   - Pre-LayerNorm (normalize before sublayer)
   - GELU activation
   - Learned position embeddings

2. Training:
   - Next token prediction (causal LM)
   - Cross-entropy loss
   - AdamW optimizer with weight decay
   - Cosine LR schedule with warmup

3. Generation:
   - Autoregressive (one token at a time)
   - Sampling strategies: temperature, top-k, top-p
   - Beam search for deterministic output
   - Repetition penalty for diversity

4. KV-Cache:
   - Store past key/value vectors
   - Avoid recomputation during generation
   - Significant speedup (O(n²) → O(n) per step)
   - Memory trade-off

5. Scaling:
   - GPT-2: 117M → 1.5B parameters
   - GPT-3: 125M → 175B parameters
   - Chinchilla: tokens ≈ 20 × parameters

6. In-Context Learning:
   - Few-shot learning via prompts
   - No gradient updates needed
   - Emergent ability at scale

Files created:
- 01_gpt.md: Theory and concepts
- 01_gpt.py: This implementation file
- attention_patterns.png: Attention visualization
- sampling_comparison.png: Sampling strategies
""")

print("\nModule 9.1 complete!")
