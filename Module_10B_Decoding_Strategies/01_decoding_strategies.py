"""
Decoding Strategies for Text Generation - Hands-on Implementation
==================================================================

This module covers:
1. Greedy decoding
2. Beam search with variants
3. Temperature, top-k, top-p sampling
4. Repetition penalties
5. Speculative decoding
6. Constrained/guided generation

Author: Zero to GPT Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Callable, Set
from dataclasses import dataclass
from collections import Counter
import math
import heapq


# =============================================================================
# Section 1: Basic Decoding Functions
# =============================================================================

def greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Greedy decoding - always pick highest probability token.

    Args:
        model: Language model
        input_ids: (batch, seq_len) prompt tokens
        max_new_tokens: Maximum tokens to generate
        eos_token_id: Stop when this token is generated
        pad_token_id: Padding token ID

    Returns:
        generated_ids: (batch, seq_len + new_tokens)
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Track which sequences are done
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Get next token logits
            next_logits = logits[:, -1, :]

            # Greedy selection
            next_tokens = next_logits.argmax(dim=-1)

            # Mask done sequences with pad
            next_tokens = torch.where(done, pad_token_id, next_tokens)

            # Append
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Update done status
            if eos_token_id is not None:
                done = done | (next_tokens == eos_token_id)

            if done.all():
                break

    return input_ids


# =============================================================================
# Section 2: Temperature Sampling
# =============================================================================

def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from logits with temperature scaling.

    Temperature effects:
    - T < 1.0: Sharper distribution (more confident)
    - T = 1.0: Original distribution
    - T > 1.0: Flatter distribution (more random)
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def temperature_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None
) -> torch.Tensor:
    """Generate text using temperature sampling."""
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            next_logits = logits[:, -1, :]

            next_tokens = sample_with_temperature(next_logits, temperature)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if eos_token_id and (next_tokens == eos_token_id).all():
                break

    return input_ids


# =============================================================================
# Section 3: Top-k Sampling
# =============================================================================

def top_k_filtering(
    logits: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Filter logits to keep only top-k tokens.

    Args:
        logits: (batch, vocab_size)
        k: Number of top tokens to keep

    Returns:
        filtered_logits: Same shape, non-top-k set to -inf
    """
    if k <= 0:
        return logits

    # Get top k values
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)

    # Mask tokens below threshold
    filtered = torch.where(
        logits < min_top_k,
        torch.full_like(logits, float('-inf')),
        logits
    )
    return filtered


def top_k_sample(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0
) -> torch.Tensor:
    """Sample from top-k tokens."""
    filtered = top_k_filtering(logits, k)
    return sample_with_temperature(filtered, temperature)


# =============================================================================
# Section 4: Top-p (Nucleus) Sampling
# =============================================================================

def top_p_filtering(
    logits: torch.Tensor,
    p: float,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps smallest set of tokens whose cumulative probability >= p.

    Args:
        logits: (batch, vocab_size)
        p: Cumulative probability threshold
        min_tokens_to_keep: Always keep at least this many tokens

    Returns:
        filtered_logits: Same shape, filtered tokens set to -inf
    """
    if p >= 1.0:
        return logits

    # Sort in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff
    sorted_indices_to_remove = cumulative_probs > p

    # Shift right to keep first token above threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Keep minimum tokens
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[:, :min_tokens_to_keep] = False

    # Set filtered tokens to -inf
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Unsort
    original_indices = sorted_indices.argsort(dim=-1)
    filtered_logits = sorted_logits.gather(-1, original_indices)

    return filtered_logits


def top_p_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """Sample from nucleus of tokens."""
    filtered = top_p_filtering(logits, p)
    return sample_with_temperature(filtered, temperature)


# =============================================================================
# Section 5: Min-p Sampling
# =============================================================================

def min_p_filtering(
    logits: torch.Tensor,
    min_p: float = 0.1
) -> torch.Tensor:
    """
    Filter tokens below min_p * max_probability.

    Adaptive threshold based on model confidence.
    """
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = max_prob * min_p

    # Mask tokens below threshold
    filtered = logits.clone()
    filtered[probs < threshold] = float('-inf')

    return filtered


def min_p_sample(
    logits: torch.Tensor,
    min_p: float = 0.1,
    temperature: float = 1.0
) -> torch.Tensor:
    """Sample using min-p filtering."""
    filtered = min_p_filtering(logits, min_p)
    return sample_with_temperature(filtered, temperature)


# =============================================================================
# Section 6: Repetition Penalties
# =============================================================================

def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeating tokens.

    Divides logits of previously generated tokens by penalty if > 1,
    multiplies if < 1.

    Args:
        logits: (batch, vocab_size) current step logits
        generated_tokens: (batch, seq_len) previously generated tokens
        penalty: > 1 discourages repetition, < 1 encourages it
    """
    if penalty == 1.0:
        return logits

    batch_size = logits.shape[0]

    for batch_idx in range(batch_size):
        unique_tokens = generated_tokens[batch_idx].unique()

        for token in unique_tokens:
            if token >= 0:  # Skip padding
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] /= penalty
                else:
                    logits[batch_idx, token] *= penalty

    return logits


def apply_presence_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 0.5
) -> torch.Tensor:
    """
    Presence penalty - subtract penalty for each unique token present.

    Unlike repetition penalty, only penalizes presence, not frequency.
    """
    if penalty == 0.0:
        return logits

    batch_size = logits.shape[0]

    for batch_idx in range(batch_size):
        unique_tokens = generated_tokens[batch_idx].unique()
        for token in unique_tokens:
            if token >= 0:
                logits[batch_idx, token] -= penalty

    return logits


def apply_frequency_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float = 0.5
) -> torch.Tensor:
    """
    Frequency penalty - penalty scales with token frequency.

    penalty * count for each token.
    """
    if penalty == 0.0:
        return logits

    batch_size = logits.shape[0]

    for batch_idx in range(batch_size):
        token_counts = Counter(generated_tokens[batch_idx].tolist())
        for token, count in token_counts.items():
            if token >= 0:
                logits[batch_idx, token] -= penalty * count

    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_tokens: List[int],
    n: int = 3
) -> torch.Tensor:
    """
    Prevent repeating n-grams by setting their probability to -inf.
    """
    if len(generated_tokens) < n - 1:
        return logits

    # Get recent context (n-1 tokens)
    recent = tuple(generated_tokens[-(n-1):])

    # Find all n-grams in history
    banned_tokens = set()
    for i in range(len(generated_tokens) - n + 1):
        if tuple(generated_tokens[i:i+n-1]) == recent:
            banned_tokens.add(generated_tokens[i + n - 1])

    # Ban these tokens
    for token in banned_tokens:
        logits[:, token] = float('-inf')

    return logits


# =============================================================================
# Section 7: Beam Search
# =============================================================================

@dataclass
class BeamHypothesis:
    """Single beam hypothesis."""
    tokens: List[int]
    score: float
    is_done: bool = False


class BeamSearchDecoder:
    """
    Beam search decoder with length normalization and n-gram blocking.
    """

    def __init__(
        self,
        num_beams: int = 4,
        max_length: int = 50,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0
    ):
        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def _length_normalized_score(self, score: float, length: int) -> float:
        """Apply length normalization."""
        return score / (length ** self.length_penalty)

    def _get_banned_tokens(self, tokens: List[int]) -> Set[int]:
        """Get tokens that would create repeated n-grams."""
        if self.no_repeat_ngram_size <= 0 or len(tokens) < self.no_repeat_ngram_size - 1:
            return set()

        banned = set()
        n = self.no_repeat_ngram_size
        recent = tuple(tokens[-(n-1):])

        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i+n-1]) == recent:
                banned.add(tokens[i + n - 1])

        return banned

    def decode(
        self,
        model: nn.Module,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Run beam search decoding.

        Args:
            model: Language model
            input_ids: (1, seq_len) prompt tokens (batch size 1)

        Returns:
            best_sequence: (1, seq_len + generated)
        """
        model.eval()
        device = input_ids.device
        prompt_len = input_ids.shape[1]

        # Initialize beams
        beams = [BeamHypothesis(
            tokens=input_ids[0].tolist(),
            score=0.0
        )]

        with torch.no_grad():
            for step in range(self.max_length):
                all_candidates = []

                for beam in beams:
                    if beam.is_done:
                        all_candidates.append(beam)
                        continue

                    # Forward pass
                    beam_input = torch.tensor([beam.tokens], device=device)
                    outputs = model(beam_input)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

                    # Get banned tokens (n-gram blocking)
                    banned = self._get_banned_tokens(beam.tokens)
                    for token in banned:
                        log_probs[token] = float('-inf')

                    # Get top tokens
                    top_log_probs, top_tokens = torch.topk(log_probs, self.num_beams * 2)

                    for log_prob, token in zip(top_log_probs, top_tokens):
                        token = token.item()
                        new_tokens = beam.tokens + [token]
                        new_score = beam.score + log_prob.item()

                        is_done = (self.eos_token_id is not None and
                                  token == self.eos_token_id)

                        all_candidates.append(BeamHypothesis(
                            tokens=new_tokens,
                            score=new_score,
                            is_done=is_done
                        ))

                # Select top beams by normalized score
                all_candidates.sort(
                    key=lambda h: self._length_normalized_score(h.score, len(h.tokens) - prompt_len),
                    reverse=True
                )
                beams = all_candidates[:self.num_beams]

                # Early stopping
                if self.early_stopping and all(b.is_done for b in beams):
                    break

        # Return best beam
        best = max(beams, key=lambda h: self._length_normalized_score(h.score, len(h.tokens) - prompt_len))
        return torch.tensor([best.tokens], device=device)


# =============================================================================
# Section 8: Diverse Beam Search
# =============================================================================

class DiverseBeamSearchDecoder:
    """
    Diverse beam search - maintains diversity between beam groups.
    """

    def __init__(
        self,
        num_beams: int = 6,
        num_beam_groups: int = 3,
        diversity_penalty: float = 0.5,
        max_length: int = 50,
        eos_token_id: Optional[int] = None
    ):
        assert num_beams % num_beam_groups == 0
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.beams_per_group = num_beams // num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def decode(
        self,
        model: nn.Module,
        input_ids: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Returns list of diverse sequences (one per beam group).
        """
        model.eval()
        device = input_ids.device

        # Initialize groups
        groups = []
        for g in range(self.num_beam_groups):
            groups.append([BeamHypothesis(
                tokens=input_ids[0].tolist(),
                score=0.0
            )])

        with torch.no_grad():
            for step in range(self.max_length):
                # Track tokens selected by previous groups (for diversity)
                previous_group_tokens = []

                for group_idx, group in enumerate(groups):
                    all_candidates = []

                    for beam in group:
                        if beam.is_done:
                            all_candidates.append(beam)
                            continue

                        beam_input = torch.tensor([beam.tokens], device=device)
                        outputs = model(beam_input)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

                        # Apply diversity penalty
                        for prev_tokens in previous_group_tokens:
                            for token in prev_tokens:
                                log_probs[token] -= self.diversity_penalty

                        top_log_probs, top_tokens = torch.topk(log_probs, self.beams_per_group * 2)

                        for log_prob, token in zip(top_log_probs, top_tokens):
                            token = token.item()
                            new_tokens = beam.tokens + [token]
                            new_score = beam.score + log_prob.item()
                            is_done = token == self.eos_token_id if self.eos_token_id else False

                            all_candidates.append(BeamHypothesis(
                                tokens=new_tokens,
                                score=new_score,
                                is_done=is_done
                            ))

                    # Keep top beams for this group
                    all_candidates.sort(key=lambda h: h.score, reverse=True)
                    groups[group_idx] = all_candidates[:self.beams_per_group]

                    # Track tokens for diversity
                    group_tokens = set()
                    for beam in groups[group_idx]:
                        if len(beam.tokens) > len(input_ids[0]):
                            group_tokens.add(beam.tokens[-1])
                    previous_group_tokens.append(group_tokens)

        # Return best from each group
        results = []
        for group in groups:
            best = max(group, key=lambda h: h.score)
            results.append(torch.tensor([best.tokens], device=device))

        return results


# =============================================================================
# Section 9: Speculative Decoding
# =============================================================================

class SpeculativeDecoder:
    """
    Speculative decoding for faster inference.

    Uses a small draft model to propose tokens, verified by target model.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        k: int = 4,
        eos_token_id: Optional[int] = None
    ):
        """
        Args:
            target_model: Large target model
            draft_model: Small draft model
            k: Number of tokens to speculate
            eos_token_id: End of sequence token
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.k = k
        self.eos_token_id = eos_token_id

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50
    ) -> torch.Tensor:
        """
        Generate using speculative decoding.

        Returns generated sequence.
        """
        self.target_model.eval()
        self.draft_model.eval()
        device = input_ids.device

        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # Draft k tokens
            draft_tokens = []
            draft_probs = []
            current = generated.clone()

            for _ in range(self.k):
                draft_out = self.draft_model(current)
                draft_logits = draft_out.logits if hasattr(draft_out, 'logits') else draft_out
                draft_prob = F.softmax(draft_logits[:, -1, :], dim=-1)

                # Sample from draft
                next_token = torch.multinomial(draft_prob, 1)
                draft_tokens.append(next_token.item())
                draft_probs.append(draft_prob[0, next_token.item()].item())

                current = torch.cat([current, next_token], dim=-1)

            # Verify with target model (single forward pass for all positions)
            target_out = self.target_model(current)
            target_logits = target_out.logits if hasattr(target_out, 'logits') else target_out

            # Verify each draft token
            accepted = []
            for i, (token, draft_p) in enumerate(zip(draft_tokens, draft_probs)):
                # Position in target output
                pos = generated.shape[1] + i - 1
                target_prob = F.softmax(target_logits[:, pos, :], dim=-1)
                target_p = target_prob[0, token].item()

                # Acceptance probability
                accept_prob = min(1.0, target_p / (draft_p + 1e-10))

                if torch.rand(1).item() < accept_prob:
                    accepted.append(token)
                else:
                    # Sample from residual distribution
                    residual = torch.clamp(target_prob - F.softmax(
                        self.draft_model(generated[:, :pos+1]).logits[:, -1, :], dim=-1
                    ), min=0)
                    if residual.sum() > 0:
                        residual = residual / residual.sum()
                        new_token = torch.multinomial(residual, 1)
                    else:
                        new_token = torch.multinomial(target_prob, 1)
                    accepted.append(new_token.item())
                    break

            # Add accepted tokens
            accepted_tensor = torch.tensor([accepted], device=device)
            generated = torch.cat([generated, accepted_tensor], dim=-1)
            tokens_generated += len(accepted)

            # Check for EOS
            if self.eos_token_id is not None and accepted[-1] == self.eos_token_id:
                break

        return generated


# =============================================================================
# Section 10: Constrained Decoding
# =============================================================================

class ConstrainedDecoder:
    """
    Decoder with token-level constraints.

    Supports:
    - Required tokens (must appear)
    - Banned tokens (must not appear)
    - Valid token sets (at specific positions)
    """

    def __init__(
        self,
        required_tokens: Optional[Set[int]] = None,
        banned_tokens: Optional[Set[int]] = None,
        valid_token_fn: Optional[Callable[[List[int]], Set[int]]] = None
    ):
        self.required_tokens = required_tokens or set()
        self.banned_tokens = banned_tokens or set()
        self.valid_token_fn = valid_token_fn
        self.remaining_required = set(self.required_tokens)

    def apply_constraints(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        remaining_steps: int
    ) -> torch.Tensor:
        """Apply constraints to logits."""
        # Ban tokens
        for token in self.banned_tokens:
            logits[:, token] = float('-inf')

        # Valid token function (e.g., grammar constraints)
        if self.valid_token_fn is not None:
            valid = self.valid_token_fn(generated_tokens)
            for token in range(logits.shape[-1]):
                if token not in valid:
                    logits[:, token] = float('-inf')

        # Force required tokens near end
        if self.remaining_required and remaining_steps <= len(self.remaining_required):
            # Must include a required token
            forced_token = next(iter(self.remaining_required))
            logits[:, :] = float('-inf')
            logits[:, forced_token] = 0

        return logits

    def update_required(self, token: int):
        """Update remaining required tokens after generation."""
        self.remaining_required.discard(token)


# =============================================================================
# Section 11: Comprehensive Generator
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    min_new_tokens: int = 0

    # Sampling
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    min_p: float = 0.0

    # Repetition
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram_size: int = 0

    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False

    # Tokens
    eos_token_id: Optional[int] = None
    pad_token_id: int = 0


class TextGenerator:
    """
    Comprehensive text generator combining all strategies.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Generate text with specified configuration.

        Automatically selects beam search vs sampling based on config.
        """
        if config.num_beams > 1:
            return self._beam_search_generate(input_ids, config)
        else:
            return self._sample_generate(input_ids, config)

    def _sample_generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """Generate using sampling strategies."""
        self.model.eval()
        device = input_ids.device
        generated = input_ids.clone()

        with torch.no_grad():
            for step in range(config.max_new_tokens):
                # Forward pass
                outputs = self.model(generated)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                next_logits = logits[:, -1, :].clone()

                # Apply repetition penalties
                if config.repetition_penalty != 1.0:
                    next_logits = apply_repetition_penalty(
                        next_logits, generated, config.repetition_penalty
                    )

                if config.presence_penalty != 0.0:
                    next_logits = apply_presence_penalty(
                        next_logits, generated, config.presence_penalty
                    )

                if config.frequency_penalty != 0.0:
                    next_logits = apply_frequency_penalty(
                        next_logits, generated, config.frequency_penalty
                    )

                if config.no_repeat_ngram_size > 0:
                    next_logits = apply_no_repeat_ngram(
                        next_logits, generated[0].tolist(), config.no_repeat_ngram_size
                    )

                # Apply filtering
                if config.top_k > 0:
                    next_logits = top_k_filtering(next_logits, config.top_k)

                if config.top_p < 1.0:
                    next_logits = top_p_filtering(next_logits, config.top_p)

                if config.min_p > 0.0:
                    next_logits = min_p_filtering(next_logits, config.min_p)

                # Sample
                if config.do_sample:
                    next_token = sample_with_temperature(next_logits, config.temperature)
                else:
                    next_token = next_logits.argmax(dim=-1)

                # Append
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

                # Check stopping conditions
                if step >= config.min_new_tokens:
                    if config.eos_token_id is not None:
                        if (next_token == config.eos_token_id).all():
                            break

        return generated

    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """Generate using beam search."""
        decoder = BeamSearchDecoder(
            num_beams=config.num_beams,
            max_length=config.max_new_tokens,
            length_penalty=config.length_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            early_stopping=config.early_stopping,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id
        )
        return decoder.decode(self.model, input_ids)


# =============================================================================
# Section 12: Demonstrations
# =============================================================================

class SimpleLanguageModel(nn.Module):
    """Simple LM for demonstrations."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 4, batch_first=True),
            num_layers=2
        )
        self.output = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        x = self.transformer(x, mask=mask.to(x.device))
        logits = self.output(x)
        return type('Output', (), {'logits': logits})()


def demo_sampling_strategies():
    """Demonstrate different sampling strategies."""
    print("=" * 60)
    print("Demo: Sampling Strategies")
    print("=" * 60)

    # Create dummy logits (simulating model output)
    vocab_size = 100
    logits = torch.randn(1, vocab_size)

    # Make a few tokens much more likely
    logits[0, 5] = 10.0  # Very likely
    logits[0, 10] = 8.0
    logits[0, 15] = 6.0
    logits[0, 20] = 4.0

    print("\nOriginal top-5 tokens:", torch.topk(logits, 5).indices[0].tolist())

    # Greedy
    greedy_token = logits.argmax(dim=-1).item()
    print(f"\nGreedy: {greedy_token}")

    # Temperature
    print("\nTemperature sampling (10 samples each):")
    for temp in [0.5, 1.0, 2.0]:
        samples = [sample_with_temperature(logits.clone(), temp).item() for _ in range(10)]
        print(f"  T={temp}: {samples}")

    # Top-k
    print("\nTop-k sampling (k=3, 10 samples):")
    samples = [top_k_sample(logits.clone(), k=3).item() for _ in range(10)]
    print(f"  {samples}")

    # Top-p
    print("\nTop-p sampling (p=0.9, 10 samples):")
    samples = [top_p_sample(logits.clone(), p=0.9).item() for _ in range(10)]
    print(f"  {samples}")


def demo_repetition_penalties():
    """Demonstrate repetition penalties."""
    print("\n" + "=" * 60)
    print("Demo: Repetition Penalties")
    print("=" * 60)

    vocab_size = 100
    logits = torch.randn(1, vocab_size)
    logits[0, 5] = 5.0  # Token 5 is most likely

    # Simulate having generated token 5 multiple times
    generated = torch.tensor([[5, 5, 5, 10, 5]])

    print(f"\nOriginal top token: {logits.argmax(dim=-1).item()}")
    print(f"Generated tokens: {generated[0].tolist()}")

    # Repetition penalty
    penalized = apply_repetition_penalty(logits.clone(), generated, penalty=1.5)
    print(f"\nWith repetition penalty (1.5): top token = {penalized.argmax(dim=-1).item()}")

    # Frequency penalty
    freq_penalized = apply_frequency_penalty(logits.clone(), generated, penalty=1.0)
    print(f"With frequency penalty (1.0): top token = {freq_penalized.argmax(dim=-1).item()}")


def demo_beam_search():
    """Demonstrate beam search."""
    print("\n" + "=" * 60)
    print("Demo: Beam Search")
    print("=" * 60)

    model = SimpleLanguageModel(vocab_size=100)

    prompt = torch.randint(0, 100, (1, 5))
    print(f"\nPrompt: {prompt[0].tolist()}")

    # Regular beam search
    decoder = BeamSearchDecoder(
        num_beams=4,
        max_length=10,
        length_penalty=1.0,
        eos_token_id=99
    )

    result = decoder.decode(model, prompt)
    print(f"Beam search result: {result[0].tolist()}")

    # Diverse beam search
    diverse_decoder = DiverseBeamSearchDecoder(
        num_beams=6,
        num_beam_groups=3,
        diversity_penalty=0.5,
        max_length=10,
        eos_token_id=99
    )

    diverse_results = diverse_decoder.decode(model, prompt)
    print(f"\nDiverse beam search results:")
    for i, result in enumerate(diverse_results):
        print(f"  Group {i}: {result[0].tolist()}")


def demo_generator():
    """Demonstrate comprehensive generator."""
    print("\n" + "=" * 60)
    print("Demo: Comprehensive Generator")
    print("=" * 60)

    model = SimpleLanguageModel(vocab_size=100)
    generator = TextGenerator(model)

    prompt = torch.randint(0, 100, (1, 5))
    print(f"\nPrompt: {prompt[0].tolist()}")

    # Different configurations
    configs = [
        ("Greedy", GenerationConfig(do_sample=False, max_new_tokens=10)),
        ("Sampling T=0.7", GenerationConfig(do_sample=True, temperature=0.7, max_new_tokens=10)),
        ("Top-p 0.9", GenerationConfig(do_sample=True, top_p=0.9, max_new_tokens=10)),
        ("Top-k 10", GenerationConfig(do_sample=True, top_k=10, max_new_tokens=10)),
        ("Beam search", GenerationConfig(num_beams=4, max_new_tokens=10)),
    ]

    for name, config in configs:
        result = generator.generate(prompt.clone(), config)
        print(f"\n{name}: {result[0].tolist()}")


if __name__ == "__main__":
    print("Decoding Strategies - Hands-on Implementation")
    print("=" * 60)

    demo_sampling_strategies()
    demo_repetition_penalties()
    demo_beam_search()
    demo_generator()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
