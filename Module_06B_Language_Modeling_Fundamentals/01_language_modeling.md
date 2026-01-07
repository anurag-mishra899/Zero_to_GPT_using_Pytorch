# Language Modeling Fundamentals

## Overview

Language modeling is the foundational task in modern NLP - the art of predicting text. This module covers the mathematical foundations, different paradigms, training techniques, and evaluation metrics that power everything from GPT to BERT.

## Table of Contents
1. [What is Language Modeling?](#what-is-language-modeling)
2. [Types of Language Models](#types-of-language-models)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Training Objectives](#training-objectives)
5. [Perplexity - The Core Metric](#perplexity---the-core-metric)
6. [Teacher Forcing](#teacher-forcing)
7. [Training Dynamics](#training-dynamics)
8. [Practical Considerations](#practical-considerations)
9. [Interview Questions](#interview-questions)

---

## What is Language Modeling?

**Language modeling** is the task of predicting the probability distribution over sequences of words/tokens. At its core, a language model answers: "Given some context, what comes next?"

### The Fundamental Question

For a sequence of tokens $w_1, w_2, ..., w_n$, a language model estimates:

$$P(w_1, w_2, ..., w_n)$$

Using the chain rule of probability:

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, ..., w_{i-1})$$

### Real-World Analogy

Think of autocomplete on your phone:
- Context: "I'm running late, be there in"
- Model predicts: "5" (high probability), "minutes" (high), "elephant" (very low)

This simple capability underpins:
- Text generation (GPT)
- Machine translation
- Speech recognition
- Code completion (Copilot)

---

## Types of Language Models

### 1. Causal/Autoregressive Language Models (CLM)

**Definition**: Predicts the next token based only on previous tokens (left-to-right).

```
Input:  "The cat sat on the"
Output: P(next_token | "The cat sat on the")
```

**Characteristics**:
- Unidirectional attention (can only look left)
- Natural for generation tasks
- Used in: GPT, GPT-2, GPT-3, LLaMA, Claude

**Training Objective**:
$$\mathcal{L}_{CLM} = -\sum_{i=1}^{n} \log P(w_i | w_1, ..., w_{i-1})$$

**Attention Mask** (causal):
```
     w1  w2  w3  w4
w1 [  1   0   0   0 ]
w2 [  1   1   0   0 ]
w3 [  1   1   1   0 ]
w4 [  1   1   1   1 ]
```

### 2. Masked Language Models (MLM)

**Definition**: Predicts randomly masked tokens using bidirectional context.

```
Input:  "The [MASK] sat on the [MASK]"
Output: P([MASK]_1 = "cat"), P([MASK]_2 = "mat")
```

**Characteristics**:
- Bidirectional attention (sees full context)
- Excellent for understanding tasks
- Used in: BERT, RoBERTa, ALBERT

**Training Objective**:
$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(w_i | w_{\backslash \mathcal{M}})$$

Where $\mathcal{M}$ is the set of masked positions.

**Typical Masking Strategy** (BERT):
- 15% of tokens selected for prediction
- Of those: 80% → [MASK], 10% → random token, 10% → unchanged

### 3. Prefix Language Models

**Definition**: Bidirectional attention on a prefix, causal attention for generation.

```
Prefix (bidirectional): "Translate English to French:"
Generation (causal):    "Le chat..."
```

**Characteristics**:
- Hybrid approach
- Good for conditional generation
- Used in: T5 (partially), UniLM

### 4. Permutation Language Models

**Definition**: Predicts tokens in random order, enabling bidirectional context without [MASK].

**Used in**: XLNet

**Key Insight**: By training on all possible permutations, the model learns bidirectional dependencies while remaining autoregressive.

### Comparison Table

| Aspect | CLM (GPT) | MLM (BERT) | Prefix LM |
|--------|-----------|------------|-----------|
| Attention | Unidirectional | Bidirectional | Hybrid |
| Best for | Generation | Understanding | Conditional Gen |
| Can generate? | Yes (natural) | No (needs tricks) | Yes |
| Training signal | Every token | ~15% tokens | Varies |
| Context usage | Left only | Full | Prefix: full |

---

## Mathematical Foundations

### Cross-Entropy Loss

The workhorse of language model training:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{V} y_i \log(\hat{y}_i)$$

Where:
- $V$ = vocabulary size
- $y_i$ = one-hot ground truth (1 for correct token, 0 otherwise)
- $\hat{y}_i$ = model's predicted probability for token $i$

**Simplified** (since $y$ is one-hot):
$$\mathcal{L}_{CE} = -\log(\hat{y}_{correct})$$

### Why Cross-Entropy?

1. **Maximum Likelihood Estimation**: Minimizing CE = maximizing likelihood of data
2. **KL Divergence**: CE = Entropy(true) + KL(true || predicted)
3. **Gradient Properties**: Strong gradients when confident and wrong

### Softmax Temperature

Controls the "sharpness" of the probability distribution:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T = 1.0$: Standard softmax
- $T < 1.0$: Sharper distribution (more confident)
- $T > 1.0$: Flatter distribution (more random)

**Training**: Always $T = 1.0$
**Inference**: Adjust for desired creativity level

---

## Training Objectives

### Standard Causal LM Loss

```python
def causal_lm_loss(logits, labels, ignore_index=-100):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len) - shifted by 1
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss
```

### Key Insight: The Shift

In causal LM, we predict position $i+1$ from position $i$:

```
Input tokens:   [BOS]  The   cat   sat
Labels:          The   cat   sat   [EOS]
Position:         0     1     2     3
```

The model at position 0 predicts "The", at position 1 predicts "cat", etc.

### Masked LM Loss

```python
def mlm_loss(logits, labels, mask_positions):
    """
    Only compute loss on masked positions
    """
    masked_logits = logits[mask_positions]
    masked_labels = labels[mask_positions]

    loss = F.cross_entropy(masked_logits, masked_labels)
    return loss
```

### Next Sentence Prediction (NSP)

Binary classification: Are two sentences consecutive?

```python
def nsp_loss(pooled_output, nsp_labels):
    """
    pooled_output: [CLS] representation
    nsp_labels: 0 (consecutive) or 1 (random)
    """
    nsp_logits = nsp_classifier(pooled_output)
    loss = F.cross_entropy(nsp_logits, nsp_labels)
    return loss
```

**Note**: RoBERTa showed NSP might not be necessary and could even hurt performance.

---

## Perplexity - The Core Metric

### Definition

Perplexity measures how "surprised" the model is by the test data:

$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | context)\right)$$

Or equivalently:
$$PPL = \exp(Cross Entropy Loss)$$

### Intuition

- **PPL = 1**: Perfect prediction (probability 1 for every token)
- **PPL = V**: Random guessing among vocabulary of size V
- **PPL = 10**: On average, model is as uncertain as choosing among 10 equally likely tokens

### Real-World Benchmarks

| Model | WikiText-103 PPL | Parameters |
|-------|------------------|------------|
| GPT-2 Small | ~37 | 117M |
| GPT-2 Medium | ~26 | 345M |
| GPT-2 Large | ~22 | 774M |
| GPT-3 | ~20 | 175B |
| LLaMA-7B | ~5.5 | 7B |

### Computing Perplexity

```python
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            # Shift for causal LM
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            # Compute loss per token
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            )

            # Only count non-padded tokens
            loss = loss.reshape(shift_labels.shape)
            total_loss += (loss * shift_mask).sum().item()
            total_tokens += shift_mask.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
```

### Perplexity Gotchas

1. **Vocabulary Size Matters**: Can't compare PPL across different tokenizers
2. **Context Length**: Longer context usually = lower PPL
3. **Domain Shift**: Train on news, test on code = high PPL
4. **Subword vs Word**: Subword PPL ≠ word-level PPL

### Bits Per Character (BPC)

Alternative metric for character-level models:

$$BPC = \frac{\log_2(PPL)}{\text{avg chars per token}}$$

---

## Teacher Forcing

### What is Teacher Forcing?

During training, we feed the **ground truth** previous tokens instead of the model's own predictions.

```
Training with Teacher Forcing:
Step 1: Input=[BOS]     → Predict "The"   (use ground truth "The" for step 2)
Step 2: Input="The"     → Predict "cat"   (use ground truth "cat" for step 3)
Step 3: Input="cat"     → Predict "sat"   (use ground truth "sat" for step 4)

Inference (no teacher forcing):
Step 1: Input=[BOS]     → Predict "The"   (use prediction "The" for step 2)
Step 2: Input="The"     → Predict "dog"   (ERROR COMPOUNDS!)
Step 3: Input="dog"     → Predict "barked"
```

### Why Use Teacher Forcing?

1. **Parallelization**: Can compute all positions simultaneously
2. **Stable Training**: Prevents error accumulation during training
3. **Faster Convergence**: Direct signal at every position

### The Exposure Bias Problem

**Issue**: Model never sees its own mistakes during training, but must handle them during inference.

**Symptoms**:
- Degenerative repetition: "I think I think I think..."
- Incoherent long generations
- Sensitivity to prompt format

### Mitigation Strategies

#### 1. Scheduled Sampling
Gradually replace ground truth with model predictions:

```python
def scheduled_sampling(epoch, total_epochs):
    """Linear schedule: start with teacher forcing, end with free running"""
    teacher_forcing_ratio = 1.0 - (epoch / total_epochs)
    return teacher_forcing_ratio
```

#### 2. Curriculum Learning
Start with easy (short) sequences, progress to hard (long):

```python
def curriculum_length(epoch, max_length=512, start_length=64):
    return min(start_length + epoch * 32, max_length)
```

#### 3. Label Smoothing
Softens the one-hot targets:

```python
def label_smoothed_loss(logits, labels, smoothing=0.1):
    vocab_size = logits.size(-1)
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (vocab_size - 1)

    # Create smoothed target distribution
    true_dist = torch.full_like(logits, smooth_value)
    true_dist.scatter_(-1, labels.unsqueeze(-1), confidence)

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(true_dist * log_probs).sum(dim=-1).mean()
    return loss
```

---

## Training Dynamics

### Learning Rate Schedules

#### Warmup + Cosine Decay (Most Common)

```python
def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=0):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**Why Warmup?**
- Large gradients early can destabilize Adam's moment estimates
- Allows model to "find its footing" before large updates

#### Typical Values
- Warmup: 1-5% of total steps
- Peak LR: 1e-4 to 6e-4 (scales with batch size)
- Final LR: 10% of peak or 0

### Batch Size Considerations

**Critical Insight**: Learning rate and batch size are coupled!

Linear scaling rule (approximate):
$$lr_{new} = lr_{base} \times \frac{batch_{new}}{batch_{base}}$$

**Modern LLM Training**:
- Batch sizes: 1M-4M tokens per step
- Achieved via gradient accumulation
- Effective batch size = micro_batch × accumulation_steps × num_gpus

### Gradient Clipping

Prevents exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**When to use**: Always for transformers! Attention can produce large gradients.

### Loss Spikes

**Common Causes**:
1. Learning rate too high
2. Bad data batch (very long sequence, weird tokens)
3. Numerical instability (need gradient clipping or mixed precision fixes)

**Solutions**:
- Lower learning rate
- Skip batches with anomalous loss
- Use gradient clipping
- Check for inf/nan in gradients

---

## Practical Considerations

### Sequence Packing

Don't waste compute on padding! Pack multiple sequences:

```
Before (wasteful):
Batch 1: [The cat sat] [PAD] [PAD] [PAD] [PAD]
Batch 2: [A] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

After (efficient):
Batch 1: [The cat sat] [SEP] [A] [SEP] [Hello world]
```

**Implementation**: Use attention mask to prevent cross-sequence attention.

### Context Length During Training

**Strategy**: Train on shorter sequences first, then longer.

```python
# Chinchilla/LLaMA approach
context_lengths = [512, 1024, 2048, 4096]
for length in context_lengths:
    train_with_context_length(model, length, steps=10000)
```

### Data Quality > Data Quantity

**Key Findings from LLaMA/Chinchilla**:
- Deduplicated data trains better
- Higher quality = faster convergence
- Mix diverse domains for generalization

### Numerical Precision

| Precision | Memory | Speed | Quality |
|-----------|--------|-------|---------|
| FP32 | 4 bytes | 1x | Best |
| FP16 | 2 bytes | 2x | Good (with loss scaling) |
| BF16 | 2 bytes | 2x | Great (recommended) |
| FP8 | 1 byte | 4x | Emerging |

**Recommendation**: Use BF16 with FP32 master weights for stability.

---

## Interview Questions

### Conceptual Questions

**Q1: Why can't BERT generate text naturally like GPT?**

BERT uses bidirectional attention - it sees future tokens during training. For generation, it would need to:
1. Generate all [MASK] tokens simultaneously (parallel decoding)
2. Or use iterative refinement (slow, suboptimal)

GPT's causal attention naturally supports autoregressive generation.

**Q2: What's the relationship between cross-entropy loss and perplexity?**

$$PPL = e^{CE}$$

If CE loss = 2.0, then PPL = e² ≈ 7.4. The model is as uncertain as choosing among ~7 equally likely tokens.

**Q3: Why do we use warmup in learning rate schedules?**

Adam maintains running averages of gradients (momentum). Early in training:
- Few samples seen → noisy gradient estimates
- Large LR + noisy gradients → unstable training

Warmup allows Adam to build reliable moment estimates before taking large steps.

**Q4: Explain the exposure bias problem and one solution.**

During training, the model always sees ground truth context (teacher forcing). During inference, it sees its own predictions, which may contain errors that compound.

Solution: Scheduled sampling - gradually replace ground truth with model predictions during training to expose the model to its own mistakes.

### Coding Questions

**Q5: Implement perplexity calculation for a batch.**

```python
def batch_perplexity(logits, labels, pad_token_id):
    """
    logits: (B, L, V)
    labels: (B, L)
    """
    # Shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten
    B, L, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction='none',
        ignore_index=pad_token_id
    )

    # Reshape and compute per-sequence
    loss = loss.view(B, L)
    mask = (shift_labels != pad_token_id).float()
    seq_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)

    # Perplexity
    perplexity = torch.exp(seq_loss)
    return perplexity.mean().item()
```

**Q6: Implement causal attention mask.**

```python
def create_causal_mask(seq_len, device):
    """Creates lower triangular mask for causal attention"""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    return mask  # True = masked positions

# Usage in attention
# attn_weights.masked_fill_(causal_mask, float('-inf'))
```

### System Design Questions

**Q7: How would you scale language model training to 1000 GPUs?**

1. **Data Parallelism**: Distribute batches across GPUs, sync gradients
2. **Tensor Parallelism**: Split attention heads / FFN across GPUs (within node)
3. **Pipeline Parallelism**: Split layers across GPUs (across nodes)
4. **ZeRO Optimization**: Shard optimizer states, gradients, and parameters
5. **Gradient Checkpointing**: Trade compute for memory
6. **Mixed Precision**: BF16 forward/backward, FP32 optimizer

**Q8: Your model's perplexity is 50 on training data but 500 on test data. Diagnose.**

Severe overfitting! Possible causes:
1. Training data too small
2. Model too large
3. Trained too long
4. Train/test domain mismatch

Solutions:
1. More training data (or data augmentation)
2. Regularization (dropout, weight decay)
3. Early stopping
4. Check data pipeline for leakage

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| CLM | Left-to-right, good for generation (GPT) |
| MLM | Bidirectional, good for understanding (BERT) |
| Perplexity | exp(CE loss), lower is better |
| Teacher Forcing | Use ground truth during training, fast but causes exposure bias |
| Warmup | Stabilizes Adam by building reliable moment estimates |
| Cross-Entropy | Standard LM loss, equivalent to maximum likelihood |
| Sequence Packing | Avoid padding waste, pack multiple sequences |
| Gradient Clipping | Essential for transformer stability |

---

## Next Steps

After mastering language modeling fundamentals:
1. **Module 07**: Transformer Architecture (attention in detail)
2. **Module 08**: GPT Architecture (decoder-only specifics)
3. **Module 17**: Decoding Strategies (generation techniques)
4. **Module 19**: LLM Finetuning (adapting pre-trained models)

---

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al., 2019
3. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 Paper
4. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
5. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla Paper
