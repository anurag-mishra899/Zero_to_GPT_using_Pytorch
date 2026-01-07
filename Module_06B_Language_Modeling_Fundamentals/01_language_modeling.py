"""
Language Modeling Fundamentals - Hands-on Implementation
========================================================

This module covers:
1. Causal Language Modeling (GPT-style)
2. Masked Language Modeling (BERT-style)
3. Perplexity calculation
4. Teacher forcing vs free running
5. Training dynamics (LR schedules, gradient clipping)
6. Practical training loop

Author: Zero to GPT Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt


# =============================================================================
# Section 1: Core Language Model Components
# =============================================================================

@dataclass
class LMConfig:
    """Configuration for Language Model"""
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 3


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive language modeling.
    Key: Each position can only attend to previous positions.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # Q, K, V projections
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        # Register as buffer so it moves with the model
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape

        # Compute Q, K, V
        q = self.query(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, n_heads, L, head_dim)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: (B, n_heads, L, L)

        # Apply causal mask (prevent attending to future)
        causal_mask = self.causal_mask[:L, :L]
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply padding mask if provided
        if attention_mask is not None:
            # Expand mask: (B, 1, 1, L)
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


class BidirectionalSelfAttention(nn.Module):
    """
    Bidirectional self-attention for masked language modeling (BERT-style).
    Each position can attend to all positions.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape

        q = self.query(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Only apply padding mask (no causal mask)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm (modern convention)"""

    def __init__(self, config: LMConfig, causal: bool = True):
        super().__init__()

        self.attention = (
            CausalSelfAttention(config) if causal
            else BidirectionalSelfAttention(config)
        )

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm residual connection
        x = x + self.attention(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# Section 2: Causal Language Model (GPT-style)
# =============================================================================

class CausalLanguageModel(nn.Module):
    """
    Autoregressive language model (GPT-style).
    Predicts next token given previous tokens.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, causal=True)
            for _ in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(config.d_model)

        # Output head (tied weights with token embedding is common)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) - same as input_ids for CLM

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Create position ids
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        output = {'logits': logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            output['loss'] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: (batch, seq_len) - prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens
            top_p: Nucleus sampling threshold

        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate if exceeds max length
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else \
                       input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self(idx_cond)
            logits = outputs['logits'][:, -1, :]  # Last position

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS
            if (next_token == self.config.eos_token_id).all():
                break

        return input_ids


# =============================================================================
# Section 3: Masked Language Model (BERT-style)
# =============================================================================

class MaskedLanguageModel(nn.Module):
    """
    Masked language model (BERT-style).
    Predicts masked tokens using bidirectional context.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks (bidirectional)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, causal=False)
            for _ in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(config.d_model)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) - includes [MASK] tokens
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) - original tokens, -100 for non-masked

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, L = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_final(x)
        logits = self.mlm_head(x)

        output = {'logits': logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100  # Only compute loss on masked tokens
            )
            output['loss'] = loss

        return output


def create_mlm_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    special_token_ids: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM training inputs by masking random tokens.

    BERT strategy:
    - 15% of tokens selected for prediction
    - Of selected: 80% [MASK], 10% random, 10% unchanged

    Args:
        input_ids: (batch, seq_len) original tokens
        mask_token_id: ID of [MASK] token
        vocab_size: Size of vocabulary
        mask_prob: Probability of selecting token for prediction
        special_token_ids: List of special tokens to never mask

    Returns:
        masked_input_ids: Input with masks applied
        labels: Original tokens at masked positions, -100 elsewhere
    """
    if special_token_ids is None:
        special_token_ids = []

    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Create mask for positions we could potentially mask
    probability_matrix = torch.full(input_ids.shape, mask_prob)

    # Don't mask special tokens
    for special_id in special_token_ids:
        probability_matrix.masked_fill_(input_ids == special_id, value=0.0)

    # Randomly select positions to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels: -100 for non-masked positions (ignored in loss)
    labels[~masked_indices] = -100

    # For selected positions, apply BERT masking strategy
    indices_replaced = torch.bernoulli(
        torch.full(input_ids.shape, 0.8)
    ).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_token_id

    # 10% random replacement
    indices_random = torch.bernoulli(
        torch.full(input_ids.shape, 0.5)  # 0.5 of remaining 20% = 10%
    ).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
    masked_input_ids[indices_random] = random_words[indices_random]

    # 10% unchanged (remaining positions)

    return masked_input_ids, labels


# =============================================================================
# Section 4: Perplexity Calculation
# =============================================================================

def compute_perplexity(
    model: CausalLanguageModel,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Compute perplexity for a causal language model.

    Perplexity = exp(average cross-entropy loss per token)

    Lower is better:
    - PPL = 1: Perfect prediction
    - PPL = vocab_size: Random guessing
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask, labels=input_ids)

            # Get per-token loss
            shift_logits = outputs['logits'][:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:]
            else:
                shift_mask = torch.ones_like(shift_labels)

            loss = F.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                reduction='none',
                ignore_index=model.config.pad_token_id
            )

            loss = loss.view(shift_labels.shape)
            valid_mask = (shift_labels != model.config.pad_token_id).float()

            total_loss += (loss * valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def compute_perplexity_streaming(
    model: CausalLanguageModel,
    text_tokens: torch.Tensor,
    stride: int = 512,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Compute perplexity with sliding window for long texts.

    This handles texts longer than model's context length.

    Args:
        model: Language model
        text_tokens: (1, total_length) - full text tokenized
        stride: How much to slide window each step
        device: Compute device
    """
    model.eval()
    max_len = model.config.max_seq_len
    seq_len = text_tokens.size(1)

    nlls = []
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_len, seq_len)
            trg_len = end_loc - prev_end_loc  # Target length for this window

            input_ids = text_tokens[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()

            # Only compute loss on new tokens (not overlap from previous window)
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs['loss'].item() * trg_len)

            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

    ppl = math.exp(sum(nlls) / prev_end_loc)
    return ppl


# =============================================================================
# Section 5: Teacher Forcing vs Free Running
# =============================================================================

class TeacherForcingTrainer:
    """
    Demonstrates teacher forcing during training.
    Standard approach - always use ground truth as input.
    """

    def __init__(self, model: CausalLanguageModel, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> float:
        """
        Standard teacher forcing: ground truth at every position.
        """
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(input_ids, attention_mask, labels=input_ids)
        loss = outputs['loss']

        loss.backward()
        self.optimizer.step()

        return loss.item()


class ScheduledSamplingTrainer:
    """
    Scheduled sampling: gradually replace ground truth with model predictions.
    Helps bridge train-test gap (exposure bias).
    """

    def __init__(
        self,
        model: CausalLanguageModel,
        optimizer,
        sampling_schedule: str = 'linear'
    ):
        self.model = model
        self.optimizer = optimizer
        self.sampling_schedule = sampling_schedule
        self.current_epoch = 0
        self.total_epochs = 1

    def get_sampling_probability(self) -> float:
        """
        Probability of using model's own prediction instead of ground truth.
        Starts at 0 (full teacher forcing) and increases.
        """
        if self.sampling_schedule == 'linear':
            return self.current_epoch / self.total_epochs
        elif self.sampling_schedule == 'inverse_sigmoid':
            k = 5
            return 1 - k / (k + math.exp(self.current_epoch / self.total_epochs * k))
        else:
            return 0.0

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> float:
        """
        Scheduled sampling training step.
        With some probability, use model's prediction as next input.
        """
        self.model.train()
        self.optimizer.zero_grad()

        B, L = input_ids.shape
        device = input_ids.device
        p_sample = self.get_sampling_probability()

        # Start with BOS token
        current_input = input_ids[:, :1]
        total_loss = 0.0

        for t in range(1, L):
            # Forward pass for current position
            outputs = self.model(current_input)
            logits = outputs['logits'][:, -1, :]

            # Get ground truth and predicted next tokens
            gt_next = input_ids[:, t:t+1]
            pred_next = logits.argmax(dim=-1, keepdim=True)

            # Scheduled sampling: choose ground truth or prediction
            use_prediction = torch.rand(B, 1, device=device) < p_sample
            next_token = torch.where(use_prediction, pred_next, gt_next)

            # Compute loss against ground truth
            loss_t = F.cross_entropy(logits, input_ids[:, t])
            total_loss += loss_t

            # Append chosen token for next step
            current_input = torch.cat([current_input, next_token], dim=1)

        loss = total_loss / (L - 1)
        loss.backward()
        self.optimizer.step()

        return loss.item()


# =============================================================================
# Section 6: Learning Rate Schedules
# =============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    Standard for transformer training.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress))

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


class WarmupLinearScheduler:
    """Linear warmup followed by linear decay to 0."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            return self.max_lr * (1 - progress)

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


def visualize_lr_schedules():
    """Visualize different learning rate schedules."""
    total_steps = 10000
    warmup_steps = 1000
    max_lr = 1e-3

    # Dummy optimizer
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([dummy_param], lr=max_lr)

    schedules = {
        'Warmup + Cosine': WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, max_lr, min_lr=max_lr * 0.1
        ),
        'Warmup + Linear': WarmupLinearScheduler(
            optimizer, warmup_steps, total_steps, max_lr
        )
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, scheduler in schedules.items():
        lrs = []
        scheduler.current_step = 0
        for _ in range(total_steps):
            lrs.append(scheduler.get_lr())
            scheduler.current_step += 1
        ax.plot(lrs, label=name)

    ax.axvline(x=warmup_steps, color='gray', linestyle='--', label='End of warmup')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lr_schedules.png', dpi=150)
    plt.close()
    print("Saved lr_schedules.png")


# =============================================================================
# Section 7: Label Smoothing
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing regularization.

    Instead of one-hot targets, use soft targets:
    - Correct class: 1 - smoothing
    - Other classes: smoothing / (vocab_size - 1)

    Benefits:
    - Prevents overconfident predictions
    - Better generalization
    - Helps with calibration
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size)
            targets: (batch * seq_len,)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smooth target distribution
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), self.confidence)

        # Mask out ignored indices
        mask = (targets != self.ignore_index).float().unsqueeze(-1)

        # Compute loss
        loss = -(smooth_targets * log_probs * mask).sum(dim=-1)

        # Average over non-ignored positions
        return loss.sum() / mask.sum()


# =============================================================================
# Section 8: Complete Training Loop
# =============================================================================

class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.examples = []

        for text in texts:
            # Simple character-level tokenization for demo
            tokens = [ord(c) % 1000 for c in text]  # Map to 0-999
            tokens = [1] + tokens + [2]  # Add BOS, EOS

            if len(tokens) <= max_length:
                # Pad
                padding = [0] * (max_length - len(tokens))
                attention_mask = [1] * len(tokens) + [0] * len(padding)
                tokens = tokens + padding
            else:
                # Truncate
                tokens = tokens[:max_length]
                attention_mask = [1] * max_length

            self.examples.append({
                'input_ids': torch.tensor(tokens),
                'attention_mask': torch.tensor(attention_mask)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_language_model(
    model: CausalLanguageModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int = 3,
    max_lr: float = 1e-4,
    warmup_ratio: float = 0.1,
    gradient_clip: float = 1.0,
    device: torch.device = torch.device('cpu'),
    use_label_smoothing: bool = False
):
    """
    Complete training loop with all best practices.
    """
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps, max_lr, min_lr=max_lr * 0.1
    )

    # Optional label smoothing
    if use_label_smoothing:
        criterion = LabelSmoothingLoss(
            model.config.vocab_size,
            smoothing=0.1,
            ignore_index=model.config.pad_token_id
        )
    else:
        criterion = None

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': [], 'lr': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, labels=input_ids)

            if criterion is not None:
                # Use label smoothing loss
                shift_logits = outputs['logits'][:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, model.config.vocab_size),
                    shift_labels.view(-1)
                )
            else:
                loss = outputs['loss']

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            lr = scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(lr)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask, labels=input_ids)
                val_loss += outputs['loss'].item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_ppl = math.exp(avg_val_loss)

        history['val_loss'].append(avg_val_loss)
        history['val_ppl'].append(val_ppl)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        print(f"  LR: {lr:.2e}")

    return history


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Perplexity
    axes[0, 1].plot(history['val_ppl'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Validation Perplexity')
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    axes[1, 1].axis('off')
    summary = f"""
    Training Summary
    ----------------
    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Val Loss: {history['val_loss'][-1]:.4f}
    Final Val PPL: {history['val_ppl'][-1]:.2f}
    Best Val PPL: {min(history['val_ppl']):.2f}
    """
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("Saved training_history.png")


# =============================================================================
# Section 9: Demonstrations
# =============================================================================

def demo_causal_vs_masked():
    """Demonstrate difference between causal and masked attention."""
    print("=" * 60)
    print("Demo: Causal vs Masked Attention")
    print("=" * 60)

    config = LMConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)

    # Create sample input
    input_ids = torch.randint(4, 100, (2, 10))  # Batch of 2, length 10

    # Causal LM
    clm = CausalLanguageModel(config)
    clm_output = clm(input_ids, labels=input_ids)
    print(f"\nCausal LM output shape: {clm_output['logits'].shape}")
    print(f"Causal LM loss: {clm_output['loss'].item():.4f}")

    # Masked LM
    mlm = MaskedLanguageModel(config)
    masked_input, labels = create_mlm_inputs(
        input_ids,
        mask_token_id=config.mask_token_id,
        vocab_size=config.vocab_size,
        special_token_ids=[0, 1, 2, 3]
    )
    mlm_output = mlm(masked_input, labels=labels)
    print(f"\nMasked LM output shape: {mlm_output['logits'].shape}")
    print(f"Masked LM loss: {mlm_output['loss'].item():.4f}")

    # Show masking
    print(f"\nOriginal tokens: {input_ids[0].tolist()}")
    print(f"Masked tokens:   {masked_input[0].tolist()}")
    print(f"Labels (masked): {labels[0].tolist()}")


def demo_perplexity():
    """Demonstrate perplexity calculation."""
    print("\n" + "=" * 60)
    print("Demo: Perplexity Calculation")
    print("=" * 60)

    config = LMConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model = CausalLanguageModel(config)

    # Create dummy data
    texts = ["hello world " * 10 for _ in range(32)]
    dataset = SimpleTextDataset(texts, None, max_length=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Compute perplexity
    ppl = compute_perplexity(model, dataloader, torch.device('cpu'))
    print(f"\nPerplexity (untrained model): {ppl:.2f}")
    print(f"For reference, random guessing would give PPL ≈ {config.vocab_size}")


def demo_generation():
    """Demonstrate text generation with different sampling strategies."""
    print("\n" + "=" * 60)
    print("Demo: Text Generation")
    print("=" * 60)

    config = LMConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2)
    model = CausalLanguageModel(config)

    # Prompt (using ASCII values as tokens)
    prompt_text = "Hello"
    prompt_tokens = torch.tensor([[config.bos_token_id] + [ord(c) for c in prompt_text]])

    print(f"\nPrompt: '{prompt_text}'")

    # Different sampling strategies
    strategies = [
        {'temperature': 1.0, 'top_k': None, 'top_p': None, 'name': 'Standard (T=1.0)'},
        {'temperature': 0.5, 'top_k': None, 'top_p': None, 'name': 'Low temp (T=0.5)'},
        {'temperature': 1.0, 'top_k': 10, 'top_p': None, 'name': 'Top-k (k=10)'},
        {'temperature': 1.0, 'top_k': None, 'top_p': 0.9, 'name': 'Nucleus (p=0.9)'},
    ]

    for strategy in strategies:
        generated = model.generate(
            prompt_tokens.clone(),
            max_new_tokens=20,
            temperature=strategy['temperature'],
            top_k=strategy['top_k'],
            top_p=strategy['top_p']
        )

        # Convert back to text (valid ASCII only)
        gen_tokens = generated[0].tolist()
        gen_text = ''.join([chr(t) if 32 <= t < 127 else '?' for t in gen_tokens])
        print(f"\n{strategy['name']}: {gen_text[:50]}...")


def demo_teacher_forcing():
    """Demonstrate teacher forcing vs scheduled sampling."""
    print("\n" + "=" * 60)
    print("Demo: Teacher Forcing vs Scheduled Sampling")
    print("=" * 60)

    config = LMConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model = CausalLanguageModel(config)

    # Show scheduled sampling probability over epochs
    print("\nScheduled Sampling Probability by Epoch:")
    trainer = ScheduledSamplingTrainer(model, None, sampling_schedule='linear')
    trainer.total_epochs = 10

    for epoch in range(11):
        trainer.current_epoch = epoch
        p = trainer.get_sampling_probability()
        bar = '█' * int(p * 20) + '░' * (20 - int(p * 20))
        print(f"  Epoch {epoch:2d}: [{bar}] {p:.2f}")


def demo_full_training():
    """Full training demonstration (small scale)."""
    print("\n" + "=" * 60)
    print("Demo: Full Training Loop")
    print("=" * 60)

    # Small config for demo
    config = LMConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.1
    )

    model = CausalLanguageModel(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy data
    train_texts = [f"this is training sentence number {i} " * 3 for i in range(100)]
    val_texts = [f"this is validation sentence {i} " * 3 for i in range(20)]

    train_dataset = SimpleTextDataset(train_texts, None, max_length=64)
    val_dataset = SimpleTextDataset(val_texts, None, max_length=64)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Train
    history = train_language_model(
        model,
        train_loader,
        val_loader,
        num_epochs=3,
        max_lr=1e-3,
        warmup_ratio=0.1,
        gradient_clip=1.0,
        device=torch.device('cpu'),
        use_label_smoothing=True
    )

    # Plot
    plot_training_history(history)


if __name__ == "__main__":
    print("Language Modeling Fundamentals - Hands-on Implementation")
    print("=" * 60)

    # Run demos
    demo_causal_vs_masked()
    demo_perplexity()
    demo_generation()
    demo_teacher_forcing()

    # Uncomment for full training demo (takes a few minutes)
    # demo_full_training()

    # Visualize learning rate schedules
    visualize_lr_schedules()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
