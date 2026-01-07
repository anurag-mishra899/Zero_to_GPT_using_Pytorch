"""
Module 8B: BERT & Encoder Models
================================

This module provides hands-on implementations of:
1. BERT architecture from scratch
2. Masked Language Modeling (MLM)
3. Next Sentence Prediction (NSP)
4. Fine-tuning for classification
5. Fine-tuning for token classification (NER)
6. Fine-tuning for question answering
7. Using pre-trained BERT models

Run this file to see BERT concepts in action!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# =============================================================================
# 1. BERT CONFIGURATION
# =============================================================================

@dataclass
class BERTConfig:
    """Configuration for BERT model."""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # FFN hidden size (4 * hidden)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2  # Segment types (sentence A, B)
    layer_norm_eps: float = 1e-12

    # Special token IDs
    pad_token_id: int = 0
    cls_token_id: int = 101
    sep_token_id: int = 102
    mask_token_id: int = 103


# =============================================================================
# 2. BERT EMBEDDINGS
# =============================================================================

class BERTEmbeddings(nn.Module):
    """
    BERT embedding layer.

    Combines three embeddings:
    1. Token embeddings (from vocabulary)
    2. Position embeddings (learned, not sinusoidal)
    3. Segment embeddings (sentence A vs B)

    Input: [CLS] token1 token2 [SEP] token3 token4 [SEP]
           |      |     |      |     |      |      |
    Token: E_CLS  E_t1  E_t2  E_SEP  E_t3   E_t4  E_SEP
           +      +     +      +     +      +      +
    Pos:   E_0    E_1   E_2    E_3   E_4    E_5    E_6
           +      +     +      +     +      +      +
    Seg:   E_A    E_A   E_A    E_A   E_B    E_B    E_B
           =      =     =      =     =      =      =
    Output: Final embedding vectors
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids buffer (not a parameter)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            token_type_ids: Segment IDs (batch_size, seq_len), 0 for sent A, 1 for sent B
            position_ids: Position IDs (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
        """
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Get embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Sum all embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds

        # LayerNorm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# =============================================================================
# 3. BERT SELF-ATTENTION
# =============================================================================

class BERTSelfAttention(nn.Module):
    """
    BERT multi-head self-attention.

    Key difference from GPT: NO CAUSAL MASK!
    Each token can attend to all other tokens (bidirectional).
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape: (batch, seq, all_heads) -> (batch, heads, seq, head_size)"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, 1, seq_len) with 0 for real, -inf for padding

        Returns:
            context: (batch_size, seq_len, hidden_size)
            attention_probs: (batch_size, num_heads, seq_len, seq_len)
        """
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask (for padding)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back: (batch, heads, seq, head_size) -> (batch, seq, hidden)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))

        return context_layer, attention_probs


class BERTAttention(nn.Module):
    """BERT attention with output projection and residual."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.self = BERTSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention
        self_output, attention_probs = self.self(hidden_states, attention_mask)

        # Output projection
        attention_output = self.output(self_output)
        attention_output = self.dropout(attention_output)

        # Add & Norm (post-norm like original BERT)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        return attention_output, attention_probs


# =============================================================================
# 4. BERT FEED-FORWARD
# =============================================================================

class BERTIntermediate(nn.Module):
    """BERT intermediate (first half of FFN)."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """BERT output (second half of FFN with residual)."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # Residual + LayerNorm
        return hidden_states


# =============================================================================
# 5. BERT LAYER
# =============================================================================

class BERTLayer(nn.Module):
    """Single BERT encoder layer."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)

        # FFN
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attention_probs


# =============================================================================
# 6. BERT ENCODER
# =============================================================================

class BERTEncoder(nn.Module):
    """Stack of BERT layers."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.layers = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(attention_probs)

        return hidden_states, all_attentions


# =============================================================================
# 7. BERT MODEL
# =============================================================================

class BERTModel(nn.Module):
    """
    BERT base model (embeddings + encoder).

    Outputs contextual representations for all tokens.
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask to extended format for attention computation.

        Input mask: 1 for real tokens, 0 for padding
        Output mask: 0 for real tokens, -inf for padding
        """
        # Expand to (batch, 1, 1, seq_len) for broadcasting
        extended_mask = attention_mask[:, None, None, :]

        # Convert: 1 -> 0, 0 -> -inf
        extended_mask = (1.0 - extended_mask) * torch.finfo(torch.float32).min

        return extended_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len), 1 for real, 0 for padding
            token_type_ids: (batch_size, seq_len), 0 for sent A, 1 for sent B

        Returns:
            Dict with:
                last_hidden_state: (batch_size, seq_len, hidden_size)
                pooler_output: (batch_size, hidden_size) - [CLS] representation
                attentions: Optional list of attention weights
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Get extended attention mask
        extended_mask = self.get_extended_attention_mask(attention_mask)

        # Embeddings
        embeddings = self.embeddings(input_ids, token_type_ids)

        # Encoder
        encoder_output, attentions = self.encoder(embeddings, extended_mask, output_attentions)

        # Pooler output ([CLS] token)
        pooler_output = encoder_output[:, 0]  # First token is [CLS]

        return {
            'last_hidden_state': encoder_output,
            'pooler_output': pooler_output,
            'attentions': attentions,
        }


# =============================================================================
# 8. BERT FOR MASKED LANGUAGE MODELING
# =============================================================================

class BERTForMaskedLM(nn.Module):
    """
    BERT for Masked Language Model pre-training.

    Predicts original tokens for [MASK] positions.
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.bert = BERTModel(config)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) - with some tokens replaced by [MASK]
            labels: (batch, seq_len) - original tokens, -100 for non-masked

        Returns:
            Dict with:
                loss: MLM loss (if labels provided)
                logits: (batch, seq_len, vocab_size)
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_states = outputs['last_hidden_state']

        # Predict tokens
        prediction_scores = self.mlm_head(hidden_states)

        result = {'logits': prediction_scores}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1, self.bert.config.vocab_size), labels.view(-1))
            result['loss'] = loss

        return result


# =============================================================================
# 9. BERT FOR SEQUENCE CLASSIFICATION
# =============================================================================

class BERTForSequenceClassification(nn.Module):
    """
    BERT for sequence classification (sentiment, topic, etc.).

    Uses [CLS] token representation for classification.
    """

    def __init__(self, config: BERTConfig, num_labels: int):
        super().__init__()
        self.bert = BERTModel(config)
        self.num_labels = num_labels

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            labels: (batch,) - class labels

        Returns:
            Dict with loss and logits
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs['pooler_output']

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {'logits': logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss

        return result


# =============================================================================
# 10. BERT FOR TOKEN CLASSIFICATION (NER)
# =============================================================================

class BERTForTokenClassification(nn.Module):
    """
    BERT for token classification (NER, POS tagging).

    Classifies each token independently.
    """

    def __init__(self, config: BERTConfig, num_labels: int):
        super().__init__()
        self.bert = BERTModel(config)
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) - label for each token

        Returns:
            Dict with loss and logits
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs['last_hidden_state']

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        result = {'logits': logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            result['loss'] = loss

        return result


# =============================================================================
# 11. BERT FOR QUESTION ANSWERING
# =============================================================================

class BERTForQuestionAnswering(nn.Module):
    """
    BERT for extractive question answering.

    Predicts start and end positions of answer in context.
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.bert = BERTModel(config)

        # Output: 2 scores per token (start, end probability)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) - [CLS] question [SEP] context [SEP]
            start_positions: (batch,) - start index of answer
            end_positions: (batch,) - end index of answer

        Returns:
            Dict with loss, start_logits, end_logits
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs['last_hidden_state']

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        result = {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            result['loss'] = (start_loss + end_loss) / 2

        return result


# =============================================================================
# 12. MLM DATA CREATION
# =============================================================================

def create_mlm_data(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: List[int],
    mlm_probability: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masked language model training data.

    15% of tokens are selected for prediction:
    - 80% are replaced with [MASK]
    - 10% are replaced with random token
    - 10% are unchanged

    Args:
        input_ids: (batch, seq_len) - original token IDs
        vocab_size: Size of vocabulary
        mask_token_id: ID of [MASK] token
        special_token_ids: List of special token IDs to not mask
        mlm_probability: Probability of selecting token for prediction

    Returns:
        masked_input_ids: (batch, seq_len) - input with masking
        labels: (batch, seq_len) - original IDs for masked, -100 for others
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Create probability matrix
    probability_matrix = torch.full(input_ids.shape, mlm_probability)

    # Don't mask special tokens
    for special_id in special_token_ids:
        probability_matrix.masked_fill_(input_ids == special_id, 0.0)

    # Sample tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_token_id

    # 10% -> random token
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
    masked_input_ids[indices_random] = random_words[indices_random]

    # 10% -> unchanged (already done)

    return masked_input_ids, labels


# =============================================================================
# 13. DEMOS
# =============================================================================

def demo_bert_architecture():
    """Demonstrate BERT architecture."""
    print("=" * 70)
    print("BERT ARCHITECTURE DEMO")
    print("=" * 70)

    # Create small config for demo
    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
    )

    model = BERTModel(config)
    print(f"\nBERT Config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  FFN size: {config.intermediate_size}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len).long()

    # Mark second half as sentence B
    token_type_ids[:, seq_len // 2:] = 1

    outputs = model(input_ids, attention_mask, token_type_ids, output_attentions=True)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {outputs['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs['pooler_output'].shape}")
    print(f"Number of attention matrices: {len(outputs['attentions'])}")
    print(f"Attention shape: {outputs['attentions'][0].shape}")


def demo_mlm():
    """Demonstrate Masked Language Modeling."""
    print("\n" + "=" * 70)
    print("MASKED LANGUAGE MODELING DEMO")
    print("=" * 70)

    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )

    model = BERTForMaskedLM(config)

    # Create sample data
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(100, config.vocab_size, (batch_size, seq_len))

    # Add special tokens
    input_ids[:, 0] = config.cls_token_id
    input_ids[:, -1] = config.sep_token_id

    print(f"\nOriginal input: {input_ids[0].tolist()}")

    # Create MLM data
    masked_input, labels = create_mlm_data(
        input_ids,
        vocab_size=config.vocab_size,
        mask_token_id=config.mask_token_id,
        special_token_ids=[config.cls_token_id, config.sep_token_id, config.pad_token_id],
        mlm_probability=0.15,
    )

    print(f"Masked input:   {masked_input[0].tolist()}")
    print(f"Labels:         {labels[0].tolist()}")

    # Forward pass
    outputs = model(masked_input, labels=labels)

    print(f"\nMLM Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")


def demo_classification():
    """Demonstrate sequence classification."""
    print("\n" + "=" * 70)
    print("SEQUENCE CLASSIFICATION DEMO")
    print("=" * 70)

    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )

    num_labels = 3  # e.g., positive, negative, neutral
    model = BERTForSequenceClassification(config, num_labels)

    # Create sample data
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_labels, (batch_size,))

    outputs = model(input_ids, labels=labels)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Labels: {labels.tolist()}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Get predictions
    predictions = outputs['logits'].argmax(dim=-1)
    print(f"Predictions: {predictions.tolist()}")


def demo_token_classification():
    """Demonstrate token classification (NER)."""
    print("\n" + "=" * 70)
    print("TOKEN CLASSIFICATION (NER) DEMO")
    print("=" * 70)

    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )

    # NER labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    num_labels = 7
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

    model = BERTForTokenClassification(config, num_labels)

    # Create sample data
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_labels, (batch_size, seq_len))

    outputs = model(input_ids, labels=labels)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Get predictions
    predictions = outputs['logits'].argmax(dim=-1)
    print(f"\nSample predictions (first sequence):")
    print(f"  Predicted: {[label_names[p] for p in predictions[0].tolist()]}")


def demo_question_answering():
    """Demonstrate extractive question answering."""
    print("\n" + "=" * 70)
    print("QUESTION ANSWERING DEMO")
    print("=" * 70)

    config = BERTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )

    model = BERTForQuestionAnswering(config)

    # Create sample data
    # Input format: [CLS] question [SEP] context [SEP]
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Simulate answer positions in context
    start_positions = torch.tensor([20, 25])  # Answer starts at these positions
    end_positions = torch.tensor([25, 30])    # Answer ends at these positions

    outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Start logits shape: {outputs['start_logits'].shape}")
    print(f"End logits shape: {outputs['end_logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Get predictions
    start_pred = outputs['start_logits'].argmax(dim=-1)
    end_pred = outputs['end_logits'].argmax(dim=-1)
    print(f"\nTrue start positions: {start_positions.tolist()}")
    print(f"Predicted start: {start_pred.tolist()}")
    print(f"True end positions: {end_positions.tolist()}")
    print(f"Predicted end: {end_pred.tolist()}")


def demo_huggingface_bert():
    """Demonstrate using Hugging Face pre-trained BERT."""
    print("\n" + "=" * 70)
    print("HUGGING FACE BERT DEMO")
    print("=" * 70)

    try:
        from transformers import BertTokenizer, BertModel, BertForSequenceClassification
        from transformers import pipeline
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
        return

    # 1. Basic BERT embeddings
    print("\n1. Getting BERT Embeddings:")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    text = "Hello, how are you doing today?"
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"   Input: {text}")
    print(f"   Tokens: {tokenizer.tokenize(text)}")
    print(f"   Hidden state shape: {outputs.last_hidden_state.shape}")
    print(f"   [CLS] embedding shape: {outputs.pooler_output.shape}")

    # 2. Sentiment analysis pipeline
    print("\n2. Sentiment Analysis Pipeline:")
    sentiment_pipeline = pipeline("sentiment-analysis")

    texts = [
        "I love this product! It's amazing!",
        "This is terrible. Worst experience ever.",
        "It's okay, nothing special."
    ]

    for text in texts:
        result = sentiment_pipeline(text)[0]
        print(f"   '{text[:40]}...' -> {result['label']} ({result['score']:.3f})")

    # 3. Fill mask (MLM)
    print("\n3. Fill Mask (MLM):")
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    masked_text = "The capital of France is [MASK]."
    predictions = fill_mask(masked_text)

    print(f"   Input: '{masked_text}'")
    print(f"   Top predictions:")
    for pred in predictions[:3]:
        print(f"      {pred['token_str']}: {pred['score']:.3f}")


def main():
    """Run all BERT demonstrations."""
    print("=" * 70)
    print("MODULE 8B: BERT & ENCODER MODELS - DEMONSTRATIONS")
    print("=" * 70)

    demo_bert_architecture()
    demo_mlm()
    demo_classification()
    demo_token_classification()
    demo_question_answering()
    demo_huggingface_bert()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Takeaways:

1. BERT ARCHITECTURE:
   - Encoder-only transformer (no decoder)
   - Bidirectional attention (each token sees all others)
   - [CLS] token for classification, [SEP] for segments

2. PRE-TRAINING:
   - MLM: Predict 15% masked tokens (80% MASK, 10% random, 10% same)
   - NSP: Predict if sentence B follows A (often removed in variants)

3. FINE-TUNING:
   - Classification: Use [CLS] hidden state
   - Token classification (NER): Use all token hidden states
   - Question Answering: Predict start/end positions

4. BERT VARIANTS:
   - RoBERTa: Better pre-training (no NSP, more data)
   - ALBERT: Parameter efficient (factorized embeddings)
   - DistilBERT: Smaller, faster (knowledge distillation)

5. BERT vs GPT:
   - BERT: Bidirectional, good for understanding
   - GPT: Causal, good for generation
   - Use BERT for classification, NER, QA
   - Use GPT for text generation, chat
""")


if __name__ == "__main__":
    main()
