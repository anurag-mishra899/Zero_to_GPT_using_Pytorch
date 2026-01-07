# Decoding Strategies for Text Generation

## Overview

How do you turn a language model into a text generator? Decoding strategies bridge the gap between probability distributions and actual text. This module covers everything from basic greedy decoding to advanced techniques like speculative decoding used in production LLMs.

## Table of Contents
1. [The Decoding Problem](#the-decoding-problem)
2. [Deterministic Strategies](#deterministic-strategies)
3. [Stochastic Strategies](#stochastic-strategies)
4. [Advanced Techniques](#advanced-techniques)
5. [Speculative Decoding](#speculative-decoding)
6. [Guided Generation](#guided-generation)
7. [Practical Considerations](#practical-considerations)
8. [Interview Questions](#interview-questions)

---

## The Decoding Problem

### From Probabilities to Text

At each step, the model outputs logits $z \in \mathbb{R}^{V}$ over the vocabulary. Decoding converts these to actual tokens.

```
Model Output:           Decoding Strategy:        Generated Token:
┌────────────────┐     ┌─────────────────────┐   ┌──────────┐
│ logits: [2.1,  │     │ - Greedy            │   │  "the"   │
│  3.5, 1.2,...] │ ──▶ │ - Beam Search       │ ──▶│  (id=42) │
│                │     │ - Sampling          │   │          │
└────────────────┘     └─────────────────────┘   └──────────┘
```

### Why Decoding Matters

Different strategies produce vastly different outputs:

```
Prompt: "The meaning of life is"

Greedy:     "The meaning of life is to be happy and healthy."
Beam(k=5):  "The meaning of life is to find purpose and meaning."
Sampling:   "The meaning of life is dancing in the rain at midnight."
Top-p(0.9): "The meaning of life is a question philosophers have pondered."
```

### The Tradeoff Triangle

```
                    Quality
                      /\
                     /  \
                    /    \
                   /      \
                  /        \
                 /          \
           Diversity ───────── Speed
```

- **Quality**: Coherent, factual, grammatical
- **Diversity**: Creative, varied, surprising
- **Speed**: Low latency, high throughput

---

## Deterministic Strategies

### Greedy Decoding

**Algorithm**: Always pick the highest probability token.

```python
def greedy_decode(model, prompt, max_length):
    tokens = prompt
    for _ in range(max_length):
        logits = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
    return tokens
```

**Pros**:
- Fastest
- Deterministic
- Simple to implement

**Cons**:
- Often repetitive
- Misses better sequences (greedy != optimal)
- No diversity

**When to Use**: Factual Q&A, structured outputs, when speed is critical.

### Beam Search

**Algorithm**: Keep top-k partial sequences at each step.

```
Step 1:    "The" → [("The cat", -1.2), ("The dog", -1.5), ("The man", -1.8)]
Step 2:    Each expanded → keep top 3 overall
           [("The cat sat", -2.1), ("The cat was", -2.3), ("The dog ran", -2.4)]
...
Final:     Return highest scoring complete sequence
```

**Key Formula**:
$$\text{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t | y_{<t})$$

Where $\alpha$ is length normalization (typically 0.6-1.0).

**Pros**:
- Better than greedy
- Finds high-probability sequences
- Good for translation/summarization

**Cons**:
- Computationally expensive (k× model calls)
- Still prone to repetition
- Favors generic outputs

**Hyperparameters**:
- `num_beams`: Beam width (4-10 typical)
- `length_penalty`: α for length normalization
- `no_repeat_ngram_size`: Block repeated n-grams
- `early_stopping`: Stop when all beams hit EOS

### Beam Search Variants

**1. Diverse Beam Search**
Forces diversity between beam groups:
```python
penalty = diversity_strength * rank_in_group
adjusted_score = score - penalty
```

**2. Constrained Beam Search**
Requires certain tokens/phrases in output:
```python
# Must include "Python" somewhere
constraints = [PhrasalConstraint(["Python"])]
```

**3. Contrastive Search**
Balances probability and diversity:
$$\text{score} = (1-\alpha) \cdot \log P(x) - \alpha \cdot \max_{v \in V_{prev}} \text{sim}(h_x, h_v)$$

---

## Stochastic Strategies

### Temperature Sampling

**Core Idea**: Adjust logit distribution sharpness before sampling.

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T < 1.0 | Sharper (more confident) | Factual tasks |
| T = 1.0 | Standard | Balanced |
| T > 1.0 | Flatter (more random) | Creative tasks |

```python
def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Top-k Sampling

**Algorithm**: Only sample from top k most likely tokens.

```python
def top_k_sampling(logits, k=50):
    # Get top k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k)

    # Zero out everything else
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, top_k_indices, top_k_values)

    # Sample
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Problem**: Fixed k doesn't adapt to distribution shape.
- Sharp distribution (confident): k=50 includes garbage
- Flat distribution (uncertain): k=50 might exclude good options

### Top-p (Nucleus) Sampling

**Algorithm**: Sample from smallest set of tokens whose cumulative probability ≥ p.

```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff
    cutoff_mask = cumulative_probs > p
    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
    cutoff_mask[..., 0] = False

    sorted_logits[cutoff_mask] = float('-inf')

    # Unsort and sample
    logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Advantages over Top-k**:
- Adapts to distribution shape
- Sharp dist → fewer tokens, flat dist → more tokens
- Generally produces more coherent text

### Typical Sampling

**Insight**: Humans produce text with "typical" probability, not highest.

$$\text{typicality}(x) = -\log P(x) - H$$

Where H is the entropy of the distribution. Select tokens close to expected surprisal.

### Min-p Sampling

**Algorithm**: Keep tokens with probability ≥ p × max_probability.

```python
def min_p_sampling(logits, min_p=0.1):
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max()
    threshold = max_prob * min_p
    probs[probs < threshold] = 0
    probs = probs / probs.sum()  # Renormalize
    return torch.multinomial(probs, num_samples=1)
```

**Advantage**: Scales naturally with model confidence.

---

## Advanced Techniques

### Repetition Penalties

**Problem**: LMs often repeat themselves.

**Solution 1**: Presence Penalty
```python
def apply_presence_penalty(logits, generated_tokens, penalty=1.0):
    for token in set(generated_tokens):
        logits[:, token] -= penalty
    return logits
```

**Solution 2**: Frequency Penalty
```python
def apply_frequency_penalty(logits, generated_tokens, penalty=1.0):
    token_counts = Counter(generated_tokens)
    for token, count in token_counts.items():
        logits[:, token] -= penalty * count
    return logits
```

**Solution 3**: No Repeat N-gram
```python
def block_ngram_repeats(logits, generated_tokens, n=3):
    if len(generated_tokens) >= n - 1:
        recent = tuple(generated_tokens[-(n-1):])
        # Find all tokens that would create repeated n-gram
        for i in range(len(generated_tokens) - n + 1):
            if tuple(generated_tokens[i:i+n-1]) == recent:
                next_token = generated_tokens[i+n-1]
                logits[:, next_token] = float('-inf')
    return logits
```

### Entropy-Based Sampling

Adjust sampling based on model uncertainty:

```python
def entropy_sampling(logits, low_entropy_temp=0.5, high_entropy_temp=1.5, threshold=2.0):
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum()

    if entropy < threshold:
        temp = low_entropy_temp   # Confident → be conservative
    else:
        temp = high_entropy_temp  # Uncertain → explore more

    return sample_with_temperature(logits, temp)
```

### Classifier-Free Guidance

Used in models trained with conditional dropout:

$$\text{logits}_{guided} = \text{logits}_{uncond} + \gamma \cdot (\text{logits}_{cond} - \text{logits}_{uncond})$$

Where $\gamma > 1$ strengthens the conditioning signal.

---

## Speculative Decoding

### The Latency Problem

LLM inference is memory-bound, not compute-bound. Most time is spent loading model weights, not computing.

**Key Insight**: We can "speculate" multiple tokens in parallel, verify them efficiently.

### How It Works

```
1. Draft Model (small, fast) generates k tokens: [t1, t2, t3, t4]

2. Target Model (large, slow) verifies in ONE forward pass:
   - P(t1|context) > threshold? ✓ Accept
   - P(t2|context, t1) > threshold? ✓ Accept
   - P(t3|context, t1, t2) > threshold? ✗ Reject
   - Stop, sample correct token from target

3. Result: Generated 2 tokens with ~1 target model call
```

### Algorithm

```python
def speculative_decode(target_model, draft_model, prompt, k=4):
    while not done:
        # Draft k tokens
        draft_tokens = []
        for _ in range(k):
            draft_logits = draft_model(prompt + draft_tokens)
            draft_token = draft_logits.argmax(-1)
            draft_tokens.append(draft_token)

        # Verify with target (single forward pass!)
        target_logits = target_model(prompt + draft_tokens)

        # Accept/reject each
        accepted = []
        for i, token in enumerate(draft_tokens):
            p_target = softmax(target_logits[i])[token]
            p_draft = softmax(draft_logits[i])[token]

            # Acceptance criterion
            if random.random() < min(1, p_target / p_draft):
                accepted.append(token)
            else:
                # Sample from residual distribution
                residual = max(0, p_target - p_draft)
                new_token = sample(residual)
                accepted.append(new_token)
                break

        prompt = prompt + accepted
```

### Speedup Analysis

- **Acceptance rate** α: Fraction of draft tokens accepted
- **Speedup**: $\frac{k \cdot \alpha + 1}{1 + \epsilon}$ where ε is draft model overhead

Typical speedups: 2-3× with well-matched draft models.

### Draft Model Strategies

1. **Smaller version of target**: GPT-4 + GPT-3.5
2. **Distilled model**: Trained to match target
3. **N-gram model**: Simple, fast, works for common patterns
4. **Same model, early exit**: Use earlier layers as draft

---

## Guided Generation

### Structured Output

Force model to produce valid JSON, code, etc.

**Grammar-Constrained Decoding**:
```python
def constrained_decode(model, prompt, grammar):
    tokens = prompt
    for _ in range(max_length):
        logits = model(tokens)

        # Mask invalid tokens based on grammar state
        valid_tokens = grammar.get_valid_tokens(tokens)
        logits[:, ~valid_tokens] = float('-inf')

        next_token = sample(logits)
        tokens = torch.cat([tokens, next_token])

        if grammar.is_complete(tokens):
            break
    return tokens
```

### Outlines Library Example
```python
from outlines import models, generate

model = models.transformers("gpt2")

# JSON schema constraint
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

generator = generate.json(model, schema)
result = generator("Generate a person:")
# Always valid JSON matching schema
```

### Logit Bias

Manually adjust token probabilities:

```python
def apply_logit_bias(logits, bias_dict):
    """
    bias_dict: {token_id: bias_value}
    Positive = more likely, negative = less likely
    """
    for token_id, bias in bias_dict.items():
        logits[:, token_id] += bias
    return logits

# Example: Encourage formal language
formal_tokens = tokenizer.encode(" please")
logit_bias = {t: 2.0 for t in formal_tokens}
```

---

## Practical Considerations

### Choosing a Strategy

| Task | Recommended Strategy |
|------|---------------------|
| Factual Q&A | Greedy or low-temp sampling |
| Creative Writing | Top-p (0.9) + temp (0.7-1.0) |
| Code Generation | Low temp (0.2) + top-p (0.95) |
| Translation | Beam search (4-5 beams) |
| Summarization | Beam + length penalty |
| Chat | Top-p (0.9) + frequency penalty |

### Common Hyperparameter Ranges

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| temperature | 0.0-2.0 | 0.7-1.0 most common |
| top_p | 0.8-0.99 | 0.9-0.95 typical |
| top_k | 10-100 | 40-50 typical |
| frequency_penalty | 0.0-2.0 | 0.5-1.0 typical |
| presence_penalty | 0.0-2.0 | 0.5-1.0 typical |
| repetition_penalty | 1.0-2.0 | 1.1-1.3 typical |

### Stopping Criteria

```python
class StoppingCriteria:
    def __init__(self, stop_tokens, max_length, stop_strings=None):
        self.stop_tokens = stop_tokens
        self.max_length = max_length
        self.stop_strings = stop_strings or []

    def should_stop(self, generated_tokens, generated_text):
        # Max length
        if len(generated_tokens) >= self.max_length:
            return True

        # EOS token
        if generated_tokens[-1] in self.stop_tokens:
            return True

        # Stop strings
        for stop_str in self.stop_strings:
            if stop_str in generated_text:
                return True

        return False
```

### Batched Generation

Process multiple sequences in parallel:

```python
def batched_generate(model, prompts, **kwargs):
    # Pad to same length
    padded = pad_sequences(prompts)

    # Generate
    outputs = model.generate(padded, **kwargs)

    # Unpad
    return [unpad(o) for o in outputs]
```

---

## Interview Questions

### Conceptual

**Q1: Why does greedy decoding often produce repetitive text?**

Greedy always picks the highest probability token. If "the" is high-probability after many contexts, it gets picked repeatedly. The model has no mechanism to explore alternatives or maintain variety - it just maximizes local probability at each step.

**Q2: Explain the difference between top-k and top-p sampling.**

Top-k: Fixed number of tokens (always sample from top 50).
Top-p: Dynamic number based on cumulative probability (sample from tokens covering 90% probability mass).

Top-p adapts to distribution shape - uses fewer tokens when confident, more when uncertain. Top-k can include garbage tokens (if k too large) or miss good tokens (if k too small).

**Q3: How does speculative decoding achieve speedup?**

LLM inference is memory-bound - most time is loading weights, not computing. Speculative decoding:
1. Uses a small draft model to propose multiple tokens
2. Verifies all proposals in ONE target model forward pass
3. Accepts verified tokens, resamples rejected ones

The key is parallel verification - instead of k sequential target calls, we do one batched call covering k positions.

**Q4: Why use length normalization in beam search?**

Without it, beam search prefers shorter sequences (fewer probability multiplications = higher total probability). Length normalization divides score by length^α, making fair comparisons across different lengths.

### Coding

**Q5: Implement top-p sampling from scratch.**

```python
def top_p_sample(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    # Sort probabilities descending
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probability
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff (first position where cum_prob > p)
    # Keep one more token to ensure we have probability mass
    cutoff = (cum_probs > p).long().argmax(dim=-1, keepdim=True)

    # Create mask
    indices = torch.arange(probs.size(-1), device=probs.device)
    mask = indices.unsqueeze(0) > cutoff

    # Zero out tokens beyond cutoff
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from filtered distribution
    sampled_index = torch.multinomial(sorted_probs, 1)

    # Map back to original indices
    return sorted_indices.gather(-1, sampled_index)
```

**Q6: Implement beam search with length penalty.**

```python
def beam_search(model, prompt, num_beams=4, max_length=50, length_penalty=0.6):
    # Initialize beams: (sequence, score)
    beams = [(prompt, 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in beams:
            if seq[-1] == EOS_TOKEN:
                all_candidates.append((seq, score))
                continue

            logits = model(seq)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            # Get top k tokens
            top_log_probs, top_indices = torch.topk(log_probs, num_beams)

            for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
                new_score = score + log_prob.item()

                # Apply length normalization
                normalized_score = new_score / (len(new_seq) ** length_penalty)
                all_candidates.append((new_seq, normalized_score))

        # Keep top beams
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:num_beams]

        # Early stop if all beams ended
        if all(seq[0, -1] == EOS_TOKEN for seq, _ in beams):
            break

    return beams[0][0]  # Return best sequence
```

---

## Summary

| Strategy | Quality | Diversity | Speed | Best For |
|----------|---------|-----------|-------|----------|
| Greedy | Medium | None | Fastest | Factual tasks |
| Beam Search | High | Low | Slow | Translation |
| Temperature | Varies | High (T>1) | Fast | Creative |
| Top-k | Good | Medium | Fast | General |
| Top-p | Good | Medium-High | Fast | Chat, creative |
| Speculative | High | Depends | Fast | Production |

### Key Takeaways

1. **No universal best strategy** - depends on task
2. **Temperature controls randomness**, top-p/k control vocabulary
3. **Repetition penalties are essential** for long generation
4. **Speculative decoding** is key for production speed
5. **Combine strategies** (e.g., top-p + temperature + frequency penalty)

---

## References

1. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Holtzman et al., 2019 (Top-p)
2. [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) - Top-k origins
3. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - Leviathan et al., 2022
4. [Contrastive Search](https://arxiv.org/abs/2210.14140) - Su et al., 2022
5. [Typical Decoding](https://arxiv.org/abs/2202.00666) - Meister et al., 2022
