# Seq2Seq Models: T5 & BART

## Overview

Encoder-decoder (seq2seq) models like T5 and BART combine the best of both worlds - bidirectional encoding and autoregressive generation. This module covers their architectures, pre-training objectives, and how to train them effectively.

## Table of Contents
1. [Encoder-Decoder Architecture](#encoder-decoder-architecture)
2. [T5: Text-to-Text Transfer Transformer](#t5-text-to-text-transfer-transformer)
3. [BART: Denoising Autoencoder](#bart-denoising-autoencoder)
4. [Pre-training Objectives](#pre-training-objectives)
5. [Training Seq2Seq Models](#training-seq2seq-models)
6. [Fine-tuning Strategies](#fine-tuning-strategies)
7. [Comparison & When to Use What](#comparison--when-to-use-what)
8. [Interview Questions](#interview-questions)

---

## Encoder-Decoder Architecture

### The Core Idea

```
┌─────────────────────────────────────────────────────────────┐
│                    Encoder-Decoder Model                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "Translate English to French: Hello world"          │
│         ↓                                                   │
│  ┌─────────────────────┐                                    │
│  │      ENCODER        │  ← Bidirectional attention         │
│  │  (sees full input)  │  ← Builds rich representations     │
│  └──────────┬──────────┘                                    │
│             │ encoder hidden states                         │
│             ↓                                               │
│  ┌─────────────────────┐                                    │
│  │      DECODER        │  ← Causal attention (left-to-right)│
│  │  (generates output) │  ← Cross-attention to encoder      │
│  └──────────┬──────────┘                                    │
│             ↓                                               │
│  Output: "Bonjour le monde"                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Three Types of Attention

1. **Encoder Self-Attention**: Bidirectional, each token attends to all tokens
2. **Decoder Self-Attention**: Causal, each token only attends to previous tokens
3. **Cross-Attention**: Decoder attends to encoder hidden states

### Why Encoder-Decoder?

| Architecture | Strengths | Best For |
|--------------|-----------|----------|
| Encoder-only (BERT) | Deep understanding | Classification, NER |
| Decoder-only (GPT) | Generation | Text generation, chat |
| Encoder-Decoder | Both | Translation, summarization |

**Key Advantage**: The encoder processes the entire input bidirectionally, then the decoder generates output while attending to the full input context.

---

## T5: Text-to-Text Transfer Transformer

### The T5 Philosophy

**Everything is text-to-text**: Every NLP task is framed as generating text from text.

```
Classification:
  Input:  "mnli premise: A man is sleeping. hypothesis: A man is awake."
  Output: "contradiction"

Translation:
  Input:  "translate English to German: Hello, how are you?"
  Output: "Hallo, wie geht es Ihnen?"

Summarization:
  Input:  "summarize: The quick brown fox jumped over the lazy dog..."
  Output: "A fox jumped over a dog."

Question Answering:
  Input:  "question: What is the capital? context: France's capital is Paris."
  Output: "Paris"
```

### T5 Architecture Details

```python
T5Config:
    vocab_size: 32128           # SentencePiece vocabulary
    d_model: 768                # Hidden dimension (T5-base)
    d_ff: 2048                  # FFN intermediate dimension
    num_heads: 12               # Attention heads
    num_encoder_layers: 12
    num_decoder_layers: 12
    relative_attention: True    # T5-style relative position bias
    layer_norm_epsilon: 1e-6
```

### Key Innovations

**1. Relative Position Embeddings**
Unlike absolute positions, T5 uses learned biases based on relative distance:
```
Attention(Q, K, V) = softmax((QK^T + B) / √d_k) V
```
Where B is a learned bias matrix based on position difference.

**2. Simplified Layer Normalization**
- Pre-norm (before attention and FFN)
- No bias terms in layer norm
- RMSNorm variant in later versions

**3. Shared Embeddings**
Input embeddings shared with output projection layer.

### T5 Model Sizes

| Model | Parameters | Layers | d_model | d_ff |
|-------|------------|--------|---------|------|
| T5-small | 60M | 6 | 512 | 2048 |
| T5-base | 220M | 12 | 768 | 3072 |
| T5-large | 770M | 24 | 1024 | 4096 |
| T5-3B | 3B | 24 | 1024 | 16384 |
| T5-11B | 11B | 24 | 1024 | 65536 |

### Pre-training: Span Corruption

T5's pre-training task masks contiguous spans of tokens:

```
Original: "Thank you for inviting me to your party last week"
Masked:   "Thank you <X> me to your party <Y> week"
Target:   "<X> for inviting <Y> last"
```

**Span Selection**:
- 15% of tokens corrupted
- Mean span length: 3 tokens
- Spans replaced with sentinel tokens (<X>, <Y>, etc.)

**Why Spans?**
- More efficient than single-token masking
- Forces model to learn multi-token relationships
- Reduces computational cost

---

## BART: Denoising Autoencoder

### BART Philosophy

BART = **B**idirectional **A**uto-**R**egressive **T**ransformer

Combines BERT's bidirectional encoder with GPT's autoregressive decoder through denoising pre-training.

### Architecture

BART follows the standard transformer encoder-decoder:

```python
BARTConfig:
    vocab_size: 50265           # BPE vocabulary (like GPT-2)
    d_model: 1024               # Hidden dimension (BART-large)
    encoder_layers: 12
    decoder_layers: 12
    encoder_attention_heads: 16
    decoder_attention_heads: 16
    encoder_ffn_dim: 4096
    decoder_ffn_dim: 4096
    activation_function: "gelu"
    max_position_embeddings: 1024
```

### Pre-training: Denoising

BART uses multiple noise functions during pre-training:

**1. Token Masking** (like BERT)
```
Original: "A B C D E"
Corrupted: "A [MASK] C D E"
```

**2. Token Deletion**
```
Original: "A B C D E"
Corrupted: "A C D E"
(Model must detect and recover missing tokens)
```

**3. Text Infilling**
```
Original: "A B C D E"
Corrupted: "A [MASK] D E"
(Single mask replaces variable-length span)
```

**4. Sentence Permutation**
```
Original: "Sent1. Sent2. Sent3."
Corrupted: "Sent3. Sent1. Sent2."
```

**5. Document Rotation**
```
Original: "A B C D E"
Corrupted: "C D E A B"
(Rotate to random start position)
```

### BART's Best Configuration

After ablation studies, BART authors found:
- **Text Infilling** (spans with Poisson λ=3)
- **Sentence Shuffling**

This combination works best for downstream tasks.

### BART vs T5

| Aspect | T5 | BART |
|--------|-----|------|
| Input Format | Text-to-text prefixes | Raw text |
| Pre-training | Span corruption | Multiple noising |
| Position Encoding | Relative | Absolute |
| Tokenizer | SentencePiece | BPE (GPT-2) |
| Best Tasks | General NLP | Summarization, generation |

---

## Pre-training Objectives

### Span Corruption (T5)

```python
def create_t5_pretraining_data(text, tokenizer, noise_density=0.15, mean_span_length=3):
    """Create T5-style span corruption training data."""
    tokens = tokenizer.encode(text)

    # Calculate number of spans to mask
    num_noise_tokens = int(len(tokens) * noise_density)
    num_spans = max(1, int(num_noise_tokens / mean_span_length))

    # Sample span start positions
    span_starts = sorted(random.sample(range(len(tokens) - mean_span_length), num_spans))

    # Create masked input and target
    masked_tokens = []
    target_tokens = []
    sentinel_id = 32099  # Start of sentinel tokens

    last_end = 0
    for i, start in enumerate(span_starts):
        # Add tokens before span
        masked_tokens.extend(tokens[last_end:start])

        # Add sentinel
        masked_tokens.append(sentinel_id - i)
        target_tokens.append(sentinel_id - i)

        # Determine span end
        end = min(start + random.randint(1, mean_span_length * 2), len(tokens))

        # Add span to target
        target_tokens.extend(tokens[start:end])

        last_end = end

    # Add remaining tokens
    masked_tokens.extend(tokens[last_end:])

    return masked_tokens, target_tokens
```

### Denoising (BART)

```python
def create_bart_pretraining_data(text, tokenizer, mask_ratio=0.3, poisson_lambda=3):
    """Create BART-style denoising training data."""
    tokens = tokenizer.encode(text)

    # Text infilling with Poisson span lengths
    num_to_mask = int(len(tokens) * mask_ratio)
    masked = 0
    corrupted_tokens = []
    i = 0

    while i < len(tokens) and masked < num_to_mask:
        if random.random() < mask_ratio:
            # Sample span length from Poisson
            span_length = np.random.poisson(poisson_lambda)
            span_length = max(1, min(span_length, len(tokens) - i))

            # Replace span with single mask token
            corrupted_tokens.append(tokenizer.mask_token_id)
            masked += span_length
            i += span_length
        else:
            corrupted_tokens.append(tokens[i])
            i += 1

    # Add any remaining tokens
    corrupted_tokens.extend(tokens[i:])

    return corrupted_tokens, tokens  # Target is original
```

### Comparison of Objectives

| Objective | Pro | Con |
|-----------|-----|-----|
| Span Corruption | Efficient, multi-token context | Fixed span distribution |
| Token Masking | Simple, well-understood | Single token at a time |
| Text Infilling | Variable length recovery | More complex |
| Sentence Shuffle | Document understanding | Less word-level signal |

---

## Training Seq2Seq Models

### The Training Loop

```python
def train_seq2seq_step(model, batch, optimizer, scheduler):
    """Single training step for encoder-decoder model."""
    model.train()

    # Unpack batch
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    # Create decoder input (shift right)
    decoder_input_ids = shift_right(labels, pad_token_id=0)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )

    loss = outputs.loss

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss.item()
```

### Shift Right for Decoder

The decoder input is the target shifted right by one position:

```
Target:     [BOS] "The" "cat" "sat" [EOS]
Decoder In: [PAD] [BOS] "The" "cat" "sat"
```

```python
def shift_right(input_ids, pad_token_id, decoder_start_token_id=None):
    """Shift input ids one position to the right."""
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id if decoder_start_token_id else pad_token_id
    return shifted
```

### Cross-Attention Implementation

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        """
        decoder_hidden: (batch, dec_len, d_model) - queries
        encoder_output: (batch, enc_len, d_model) - keys/values
        """
        B, dec_len, _ = decoder_hidden.shape
        _, enc_len, _ = encoder_output.shape

        # Project Q from decoder, K/V from encoder
        Q = self.query(decoder_hidden)
        K = self.key(encoder_output)
        V = self.value(encoder_output)

        # Reshape for multi-head attention
        Q = Q.view(B, dec_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, enc_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, enc_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, heads, dec_len, enc_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply encoder padding mask
        if encoder_mask is not None:
            scores = scores.masked_fill(
                encoder_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, dec_len, -1)
        return self.out(out)
```

### Label Smoothing

Essential for seq2seq training:

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (batch * seq_len, vocab_size)
        targets: (batch * seq_len,)
        """
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed distribution
        smoothed = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))
        smoothed.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        # Mask padding
        mask = (targets != self.ignore_index).float()

        loss = -(smoothed * log_probs).sum(dim=-1)
        return (loss * mask).sum() / mask.sum()
```

---

## Fine-tuning Strategies

### Task-Specific Fine-tuning

**Translation**:
```python
# Prefix format (T5-style)
input_text = "translate English to German: Hello, how are you?"
target_text = "Hallo, wie geht es Ihnen?"
```

**Summarization**:
```python
# BART-style (no prefix needed, trained on summarization)
input_text = "Article text here..."
target_text = "Summary of the article."

# T5-style
input_text = "summarize: Article text here..."
target_text = "Summary of the article."
```

**Question Answering** (Generative):
```python
input_text = "question: What is the capital of France? context: Paris is the capital of France."
target_text = "Paris"
```

### Hyperparameters for Fine-tuning

```python
fine_tuning_config = {
    # Smaller learning rate than pre-training
    "learning_rate": 3e-5,  # or 1e-4 for larger datasets

    # Warmup
    "warmup_ratio": 0.1,

    # Batch size (with gradient accumulation)
    "per_device_batch_size": 8,
    "gradient_accumulation_steps": 4,

    # Regularization
    "weight_decay": 0.01,
    "label_smoothing": 0.1,

    # Early stopping
    "patience": 3,
    "metric_for_best_model": "eval_loss",  # or BLEU/ROUGE

    # Length settings
    "max_source_length": 512,
    "max_target_length": 128,
}
```

### Data Augmentation for Seq2Seq

**1. Back-Translation**:
```
Original: EN → DE: "Hello" → "Hallo"
Synthetic: DE → EN: "Hallo" → "Hi there"
Use "Hi there" → "Hallo" as additional training
```

**2. Paraphrase Augmentation**:
```
Original: "summarize: The cat sat on the mat."
Paraphrase: "summarize: A cat was sitting on a mat."
Same target for both.
```

**3. Noise Injection**:
```
# Add typos/noise to source (model learns robustness)
Original: "Translate: Hello world"
Noisy: "Trnslate: Helo wrold"
```

### Efficient Fine-tuning: LoRA for Seq2Seq

```python
# Apply LoRA to attention layers
lora_config = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj",  # Attention
        "encoder_attn.q_proj",  # Cross-attention
        "encoder_attn.k_proj",
        "encoder_attn.v_proj",
    ],
    "lora_dropout": 0.1,
}
```

---

## Comparison & When to Use What

### T5 vs BART vs GPT

| Aspect | T5 | BART | GPT-3/4 |
|--------|-----|------|---------|
| Architecture | Enc-Dec | Enc-Dec | Decoder-only |
| Pre-training | Span corruption | Denoising | CLM |
| Input format | Text-to-text | Natural | Prompt |
| Summarization | Good | Excellent | Good |
| Translation | Excellent | Good | Good |
| Classification | Good | Good | Good (few-shot) |
| Generation quality | Good | Excellent | Excellent |
| Efficiency | Medium | Medium | High (no encoder) |

### Decision Guide

**Use T5 when**:
- Multiple diverse tasks
- Need consistent text-to-text interface
- Translation is important
- Want to leverage massive pre-training

**Use BART when**:
- Summarization is primary task
- Denoising robustness needed
- Want GPT-2 style tokenizer
- Generation quality is priority

**Use Decoder-only when**:
- Pure generation tasks
- Instruction following / chat
- Want to leverage latest LLMs
- Computational efficiency matters

### Task-Model Recommendations

| Task | Best Model | Why |
|------|------------|-----|
| Translation | T5, mT5 | Multi-task pre-training |
| Summarization | BART, PEGASUS | Denoising pre-training |
| QA (Generative) | T5 | Text-to-text format |
| Data-to-Text | T5, BART | Structured input handling |
| Dialogue | GPT, T5 | Natural generation |

---

## Interview Questions

### Conceptual

**Q1: Explain the three types of attention in an encoder-decoder model.**

1. **Encoder Self-Attention**: Bidirectional - each encoder token attends to all encoder tokens. Builds rich contextual representations of the input.

2. **Decoder Self-Attention**: Causal - each decoder token only attends to previous decoder tokens. Maintains autoregressive property for generation.

3. **Cross-Attention**: Decoder tokens attend to all encoder tokens. Allows decoder to "look at" the input while generating.

**Q2: Why does T5 use span corruption instead of single-token masking?**

Span corruption:
1. More efficient (fewer sentinel tokens = shorter sequences)
2. Forces learning multi-token dependencies
3. Better for generation (must produce coherent spans)
4. Reduces training cost while maintaining quality

Single-token gives dense signal but less "generation-like" task.

**Q3: How does BART differ from BERT + GPT?**

BART jointly trains the encoder-decoder through denoising:
- Encoder and decoder are trained together from scratch
- Cross-attention is learned end-to-end
- Can use multiple noise types simultaneously
- Optimized for seq2seq, not just classification + generation separately

Simply concatenating BERT encoder + GPT decoder wouldn't have aligned representations.

**Q4: What is "shift right" and why is it needed?**

Shift right prepends a start token to decoder inputs during training:
```
Target: "The cat sat" [EOS]
Decoder input: [BOS] "The cat sat"
```

This allows teacher forcing: at position i, the decoder predicts token i while seeing tokens 0 to i-1 as input. Without shift, position 0 would see its own target.

### Coding

**Q5: Implement cross-attention masking for variable-length batches.**

```python
def create_cross_attention_mask(encoder_mask, decoder_len):
    """
    Create mask for cross-attention.

    Args:
        encoder_mask: (batch, encoder_len) - 1 for valid, 0 for pad
        decoder_len: int - decoder sequence length

    Returns:
        mask: (batch, decoder_len, encoder_len)
    """
    batch_size, encoder_len = encoder_mask.shape

    # Expand encoder mask to cover all decoder positions
    # Each decoder position can attend to same encoder positions
    mask = encoder_mask.unsqueeze(1).expand(-1, decoder_len, -1)

    return mask
```

**Q6: Implement the T5 relative position bias.**

```python
class T5RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, num_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        # Bucketize relative positions
        num_buckets = self.num_buckets // 2
        ret = (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Logarithmic bucketing for larger distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_length, key_length, device):
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position

        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, heads, q_len, k_len)

        return values
```

### System Design

**Q7: Design a training pipeline for multilingual T5.**

```
Architecture:
┌─────────────────────────────────────────────────────┐
│               Data Pipeline                         │
├─────────────────────────────────────────────────────┤
│ 1. Language-balanced sampling                       │
│    - Upsample low-resource languages               │
│    - Temperature sampling: P(L) ∝ |D_L|^α          │
│                                                     │
│ 2. Multi-task data mixing                          │
│    - Translation: XX → EN, EN → XX, XX → YY        │
│    - Span corruption (all languages)               │
│    - Task prefixes: "translate X to Y:", etc.      │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Model Architecture                    │
├─────────────────────────────────────────────────────┤
│ - Shared encoder-decoder                            │
│ - Shared SentencePiece tokenizer (100K vocab)      │
│ - Language embeddings (optional)                    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Distributed Training                  │
├─────────────────────────────────────────────────────┤
│ - Data parallelism across nodes                     │
│ - Model parallelism for large variants             │
│ - Gradient checkpointing for memory               │
│ - Mixed precision (BF16)                            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Evaluation                            │
├─────────────────────────────────────────────────────┤
│ - Per-language BLEU for translation                │
│ - Cross-lingual transfer (train EN, test XX)       │
│ - Zero-shot translation (unseen pairs)             │
└─────────────────────────────────────────────────────┘
```

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| Encoder-Decoder | Bidirectional understanding + autoregressive generation |
| T5 | Text-to-text, span corruption, relative positions |
| BART | Denoising autoencoder, text infilling |
| Cross-Attention | Decoder attends to encoder for context |
| Shift Right | Teacher forcing with start token |
| Label Smoothing | Essential regularization for seq2seq |
| Fine-tuning | Lower LR, task prefixes, careful hyperparameters |

---

## References

1. [Exploring Transfer Learning with T5](https://arxiv.org/abs/1910.10683) - Raffel et al., 2019
2. [BART: Denoising Seq2Seq Pre-training](https://arxiv.org/abs/1910.13461) - Lewis et al., 2019
3. [mT5: Multilingual T5](https://arxiv.org/abs/2010.11934) - Xue et al., 2020
4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
5. [PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) - Zhang et al., 2020
