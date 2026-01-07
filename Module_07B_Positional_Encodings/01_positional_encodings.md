# Module 7B: Positional Encodings Deep Dive

## Table of Contents
1. [Why Position Matters](#1-why-position-matters)
2. [Sinusoidal Positional Encoding](#2-sinusoidal-positional-encoding)
3. [Learned Positional Embeddings](#3-learned-positional-embeddings)
4. [Rotary Position Embeddings (RoPE)](#4-rotary-position-embeddings-rope)
5. [ALiBi (Attention with Linear Biases)](#5-alibi-attention-with-linear-biases)
6. [Relative Positional Encodings](#6-relative-positional-encodings)
7. [Context Length Extension](#7-context-length-extension)
8. [Comparison and Selection](#8-comparison-and-selection)
9. [Interview Questions](#9-interview-questions)
10. [Summary](#10-summary)

---

## 1. Why Position Matters

### 1.1 The Permutation Invariance Problem

Self-attention is inherently **position-agnostic**:

```python
# Attention only depends on content, not position
attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

# These produce identical outputs!
attention(["dog", "bites", "man"])
attention(["man", "bites", "dog"])
```

**Problem**: "Dog bites man" and "Man bites dog" have very different meanings!

### 1.2 What Position Information Provides

| Information | Example |
|-------------|---------|
| **Word order** | Subject before verb in English |
| **Sentence structure** | Beginning/middle/end patterns |
| **Local context** | Adjacent words matter more |
| **Long-range dependencies** | Distance between related words |

### 1.3 Position Encoding Goals

1. **Unique representation** for each position
2. **Bounded values** (not growing with position)
3. **Distance awareness** (nearby positions are similar)
4. **Extrapolation** (handle longer sequences than training)
5. **Efficiency** (not too expensive to compute)

### 1.4 Evolution of Position Encodings

```
2017: Sinusoidal (Original Transformer)
      ↓
2018: Learned Absolute (BERT, GPT-2)
      ↓
2020: Relative Position (T5, Transformer-XL)
      ↓
2021: RoPE (RoFormer, LLaMA)
      ↓
2022: ALiBi (BLOOM, MPT)
```

---

## 2. Sinusoidal Positional Encoding

### 2.1 The Original Formula

From "Attention Is All You Need" (Vaswani et al., 2017):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
  pos = position in sequence (0, 1, 2, ...)
  i = dimension index (0, 1, 2, ..., d_model/2 - 1)
  d_model = embedding dimension
```

### 2.2 Intuition Behind the Formula

**Each dimension has a different frequency**:
```
Dimension 0-1: Very high frequency (changes every position)
Dimension 2-3: Lower frequency
...
Dimension d-2, d-1: Very low frequency (long wavelength)

Like a binary counter, but continuous!
Position:  0     1     2     3     4     ...
Dim 0-1:  [~]   [~]   [~]   [~]   [~]    (fast oscillation)
Dim 2-3:  [~~]  [~~]  [~~]  [~~]  [~~]   (medium)
...
Dim d:    [~~~~~~~~~~~~~~~~~~~~~~~~]     (slow)
```

### 2.3 Why Sin and Cos?

**Key property**: Position (pos + k) can be expressed as a linear transformation of position (pos):

```
PE(pos + k) = Linear_k(PE(pos))

Using trig identities:
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)

This allows the model to learn relative positions!
```

### 2.4 Properties

| Property | Sinusoidal |
|----------|------------|
| **Parameters** | None (computed) |
| **Extrapolation** | Yes (formula works for any pos) |
| **Relative position** | Yes (via linear transformation) |
| **Bounded** | Yes (sin/cos in [-1, 1]) |
| **Used by** | Original Transformer, some T5 variants |

### 2.5 Limitations

- Not learned - might not be optimal for specific tasks
- Absolute positions - doesn't directly encode relative distances
- Fixed frequency pattern - may not match task needs

---

## 3. Learned Positional Embeddings

### 3.1 Concept

Replace fixed sinusoidal with **learned embedding table**:

```python
# Learned position embeddings
position_embeddings = nn.Embedding(max_position, d_model)

# Usage
positions = torch.arange(seq_len)
pos_embeds = position_embeddings(positions)
output = token_embeds + pos_embeds
```

### 3.2 BERT and GPT-2 Style

**BERT**:
```
Input: [CLS] token1 token2 [SEP]
Token:  E_CLS  E_t1   E_t2  E_SEP
  +       +      +      +     +
Pos:    E_0    E_1    E_2   E_3
  =       =      =      =     =
Output: [...]  [...]  [...] [...]
```

**GPT-2**:
```python
# GPT-2 uses additive position embeddings
wpe = nn.Embedding(1024, 768)  # Max 1024 positions
wte = nn.Embedding(vocab_size, 768)

h = wte(input_ids) + wpe(position_ids)
```

### 3.3 Properties

| Property | Learned |
|----------|---------|
| **Parameters** | max_position × d_model |
| **Extrapolation** | No (only positions seen in training) |
| **Flexibility** | High (learned for task) |
| **Used by** | BERT, GPT-2, RoBERTa |

### 3.4 Advantages and Limitations

**Advantages**:
- Can learn task-specific patterns
- Simple implementation
- Often works as well as sinusoidal

**Limitations**:
- Cannot extrapolate beyond max_position
- Requires more parameters
- No built-in relative position awareness

---

## 4. Rotary Position Embeddings (RoPE)

### 4.1 Overview

RoPE (Su et al., 2021) encodes position by **rotating** query and key vectors:

```
Instead of: x + PE
Do: rotate(x, θ × position)

Key insight: The dot product of rotated vectors depends on
their RELATIVE position, not absolute!
```

### 4.2 The Rotation Matrix

For a 2D subspace (pair of dimensions):

```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]

For position m:
R(mθ) = [cos(mθ)  -sin(mθ)]
        [sin(mθ)   cos(mθ)]
```

Applied to query/key pairs:
```
q_m = R(mθ) × q    (rotate query by position m)
k_n = R(nθ) × k    (rotate key by position n)
```

### 4.3 Why Rotation Encodes Relative Position

```
Dot product of rotated vectors:
q_m · k_n = (R_m × q)ᵀ × (R_n × k)
          = qᵀ × R_mᵀ × R_n × k
          = qᵀ × R_{n-m} × k

The rotation only depends on (n - m) = relative position!
```

### 4.4 Full RoPE Formula

For d-dimensional vectors, apply rotation to pairs of dimensions:

```
RoPE(x, m) = [x_0 cos(mθ_0) - x_1 sin(mθ_0)]
             [x_0 sin(mθ_0) + x_1 cos(mθ_0)]
             [x_2 cos(mθ_1) - x_3 sin(mθ_1)]
             [x_2 sin(mθ_1) + x_3 cos(mθ_1)]
             ...

where θ_i = base^(-2i/d), base = 10000 typically
```

### 4.5 Implementation

```python
def apply_rope(q, k, freqs_cos, freqs_sin):
    # q, k: (batch, heads, seq_len, head_dim)
    # freqs_cos, freqs_sin: (seq_len, head_dim/2)

    # Reshape to pairs
    q_r = q.reshape(*q.shape[:-1], -1, 2)  # (..., head_dim/2, 2)
    k_r = k.reshape(*k.shape[:-1], -1, 2)

    # Apply rotation
    q_out = torch.stack([
        q_r[..., 0] * freqs_cos - q_r[..., 1] * freqs_sin,
        q_r[..., 0] * freqs_sin + q_r[..., 1] * freqs_cos
    ], dim=-1).flatten(-2)

    k_out = torch.stack([
        k_r[..., 0] * freqs_cos - k_r[..., 1] * freqs_sin,
        k_r[..., 0] * freqs_sin + k_r[..., 1] * freqs_cos
    ], dim=-1).flatten(-2)

    return q_out, k_out
```

### 4.6 Properties

| Property | RoPE |
|----------|------|
| **Parameters** | None (computed) |
| **Extrapolation** | Good (with position interpolation) |
| **Relative position** | Yes (built-in) |
| **Applied to** | Q and K only (not V) |
| **Used by** | LLaMA, Mistral, GPT-NeoX, PaLM |

### 4.7 Why RoPE is Popular

1. **Relative position naturally**: No need to learn it
2. **Efficient**: No additional parameters
3. **Extrapolation**: Works beyond training length (with tricks)
4. **Modern standard**: Used by most open LLMs

---

## 5. ALiBi (Attention with Linear Biases)

### 5.1 Overview

ALiBi (Press et al., 2022) takes a completely different approach:
**No position encoding at all!** Instead, add a bias to attention scores based on distance.

```
Standard attention:
  attention = softmax(Q @ K.T / sqrt(d))

ALiBi attention:
  attention = softmax(Q @ K.T / sqrt(d) + bias)

where bias[i,j] = -m × |i - j|
```

### 5.2 The Bias Matrix

```
For a sequence of length 4:
         j=0   j=1   j=2   j=3
i=0  [   0,   -m,  -2m,  -3m ]
i=1  [  -m,    0,   -m,  -2m ]
i=2  [ -2m,   -m,    0,   -m ]
i=3  [ -3m,  -2m,   -m,    0 ]

m = slope (different per head, typically 2^(-8/n_heads))
```

### 5.3 Multiple Heads with Different Slopes

```python
# Different slopes for different heads
# Creates geometric sequence: 2^(-8/n), 2^(-7/n), ..., 2^(-1/n)
def get_alibi_slopes(n_heads):
    ratio = 2 ** (-8 / n_heads)
    return [ratio ** i for i in range(1, n_heads + 1)]

# Example for 8 heads:
# [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
```

### 5.4 Why ALiBi Works

**Key insight**: The linear penalty naturally discounts distant tokens.

```
Position 100 vs 101: bias difference = -m
Position 100 vs 200: bias difference = -100m

Distant tokens get much lower attention (after softmax)
Local context naturally emphasized
```

### 5.5 Properties

| Property | ALiBi |
|----------|-------|
| **Parameters** | None (fixed biases) |
| **Extrapolation** | Excellent (linear bias extends naturally) |
| **Added to** | Attention scores (not embeddings) |
| **Relative position** | Yes (distance-based bias) |
| **Used by** | BLOOM, MPT, Falcon |

### 5.6 Advantages

1. **Zero additional parameters**
2. **Excellent extrapolation**: Works well beyond training length
3. **Simple implementation**: Just add a bias matrix
4. **No position embeddings needed**: Simpler input

---

## 6. Relative Positional Encodings

### 6.1 T5-Style Relative Position

T5 uses learned **relative position biases** added to attention:

```
attention[i,j] = softmax(Q_i @ K_j + bias[i-j])

bias is a learned parameter indexed by relative position
```

### 6.2 Bucketed Relative Positions

T5 uses **logarithmic bucketing** for relative positions:

```
Relative positions: -128, ..., -1, 0, 1, ..., 127, ...
Bucketed to: 32 buckets (16 for each direction)

Nearby positions: fine-grained (separate bucket each)
Far positions: coarse-grained (grouped into same bucket)
```

### 6.3 Transformer-XL Style

Transformer-XL decomposes attention into content and position components:

```
A_{i,j} = E_i W_q (E_j W_k)^T           # Content-to-content
        + E_i W_q (R_{i-j} W_{k,R})^T   # Content-to-position
        + u (E_j W_k)^T                  # Global content bias
        + v (R_{i-j} W_{k,R})^T          # Global position bias

where R_{i-j} is sinusoidal encoding of relative position
```

### 6.4 Properties

| Property | T5 Relative | Transformer-XL |
|----------|-------------|----------------|
| **Parameters** | Small (buckets × heads) | Medium |
| **Extrapolation** | Limited (bucket OOV) | Good |
| **Complexity** | Low | High |
| **Used by** | T5, FLAN-T5 | Transformer-XL |

---

## 7. Context Length Extension

### 7.1 The Problem

Models are trained with fixed context length (e.g., 2048 tokens).
How do we extend to longer sequences?

```
Trained on: 2048 tokens
Want to use: 8192 tokens

Position embeddings for positions 2049-8192 are:
- Not learned (for learned embeddings)
- Have different characteristics (for computed methods)
```

### 7.2 Position Interpolation (PI)

**For RoPE**: Scale positions to fit training range:

```
Original: position m uses angle θ × m
Extended: position m uses angle θ × (m × L_train / L_new)

Example: Position 5000 with 2048 training, 8192 target
  Original: θ × 5000
  Interpolated: θ × (5000 × 2048 / 8192) = θ × 1250
```

### 7.3 NTK-Aware Scaling

Adjust the frequency base instead of positions:

```
Original: θ_i = base^(-2i/d), base = 10000
NTK-aware: θ_i = (base × scale)^(-2i/d)

where scale = (L_new / L_train)^(d/(d-2))
```

### 7.4 YaRN (Yet Another RoPE Extension)

Combines multiple techniques:
1. NTK-aware interpolation
2. Attention scaling
3. Per-dimension adjustments

```python
# YaRN modifies frequencies differently by dimension
# Low-frequency dimensions: interpolate
# High-frequency dimensions: extrapolate
```

### 7.5 Dynamic NTK (Code LLaMA approach)

Dynamically adjust scaling based on sequence length:

```python
if seq_len > max_trained_len:
    scale = seq_len / max_trained_len
    base_new = base * scale ** (dim / (dim - 2))
```

### 7.6 Extension Comparison

| Method | Quality | Speed | Complexity |
|--------|---------|-------|------------|
| Position Interpolation | Good | Fast | Low |
| NTK-Aware | Better | Fast | Low |
| YaRN | Best | Medium | Medium |
| Fine-tuning | Excellent | Slow | High |

---

## 8. Comparison and Selection

### 8.1 Method Comparison

| Method | Params | Extrapolation | Relative | Used By |
|--------|--------|---------------|----------|---------|
| Sinusoidal | 0 | Good | Indirect | Original Transformer |
| Learned | max_pos × d | Poor | No | BERT, GPT-2 |
| RoPE | 0 | Good | Yes | LLaMA, Mistral |
| ALiBi | 0 | Excellent | Yes | BLOOM, MPT |
| T5 Relative | Small | Limited | Yes | T5, FLAN |

### 8.2 When to Use What

**Sinusoidal**:
- Simple baseline
- When you need extrapolation without learning
- Limited modern use

**Learned Absolute**:
- Short, fixed-length sequences
- When position patterns are task-specific
- Simpler implementation

**RoPE** (Recommended for most cases):
- Modern LLMs
- When you need extrapolation
- When relative position matters
- Standard choice for open LLMs

**ALiBi**:
- When extrapolation is critical
- Simpler than RoPE
- Good for very long sequences

**T5-Style Relative**:
- Encoder-decoder models
- When you want learned relative biases

### 8.3 Implementation Complexity

```
Easiest to implement:
1. Learned absolute (just nn.Embedding)
2. Sinusoidal (compute from formula)
3. ALiBi (add bias matrix to attention)
4. RoPE (rotation in Q, K)
5. T5 relative (bucketed bias lookup)
```

---

## 9. Interview Questions

### Q1: Explain why position encoding is necessary in transformers.

**Answer**:

Self-attention is **permutation-invariant**:
```
attention("dog bites man") = attention("man bites dog")
```

This is because attention only computes:
```
softmax(Q @ K.T / sqrt(d)) @ V
```

There's nothing in this formula that knows position!

**Consequences without position**:
- Word order is lost
- "I love you" = "You love I"
- Grammar and syntax can't be learned

**Position encoding adds**:
- Unique representation per position
- Allows model to learn position-dependent patterns
- Enables understanding of sequence structure

### Q2: Compare sinusoidal vs learned positional embeddings.

**Answer**:

**Sinusoidal (Original Transformer)**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Pros:
- No parameters to learn
- Can extrapolate to longer sequences
- Encodes relative position via linear transformation
- Bounded values

Cons:
- Fixed pattern, not task-specific
- May not be optimal for all tasks

**Learned (BERT, GPT-2)**:
```python
pos_emb = nn.Embedding(max_len, d_model)
```

Pros:
- Can learn task-specific patterns
- Often performs as well or better
- Simple implementation

Cons:
- Cannot extrapolate beyond max_len
- Additional parameters
- Position max_len+1 is undefined

### Q3: Explain Rotary Position Embeddings (RoPE).

**Answer**:

**Core idea**: Encode position by rotating query/key vectors.

**Formula** (for 2D):
```
q_m = R(mθ) @ q = [cos(mθ), -sin(mθ)] @ [q_0]
                   [sin(mθ),  cos(mθ)]   [q_1]
```

**Why it works**:
```
q_m · k_n = q^T @ R_m^T @ R_n @ k
          = q^T @ R_{n-m} @ k

The dot product only depends on (n-m), the RELATIVE position!
```

**Key properties**:
- Applied to Q and K only (not V)
- Different frequencies for different dimension pairs
- No additional parameters
- Natural relative position encoding
- Good extrapolation with position interpolation

**Used by**: LLaMA, Mistral, GPT-NeoX, PaLM

### Q4: What is ALiBi and why is it good for long sequences?

**Answer**:

**ALiBi (Attention with Linear Biases)**:
- No position encoding in input!
- Add linear penalty to attention scores based on distance

```
attention = softmax(Q @ K.T / sqrt(d) + bias)
bias[i,j] = -m × |i - j|
```

**Why good for extrapolation**:
1. Linear bias extends naturally to any length
2. No position embeddings to go out-of-distribution
3. Tested: trained on 1024, works at 2048+

**Different slopes per head**:
```python
# Geometric sequence of slopes
slopes = [2^(-8/n), 2^(-7/n), ..., 2^(-1/n)]
```
- Some heads focus locally (steep slope)
- Some heads look far (gentle slope)

### Q5: How do you extend a model to longer context lengths?

**Answer**:

**For RoPE-based models** (most modern LLMs):

1. **Position Interpolation (PI)**:
```python
# Scale position to fit training range
effective_pos = pos × (train_len / target_len)
```
Simple but may lose fine-grained position info.

2. **NTK-Aware Scaling**:
```python
# Adjust frequency base
new_base = base × scale^(d/(d-2))
```
Better preserves position resolution.

3. **YaRN**:
- Different treatment for different frequency dimensions
- Best quality, more complex

4. **Fine-tuning**:
- Train briefly on longer sequences
- Most reliable but requires compute

**For ALiBi models**:
- Often works out-of-the-box
- Linear bias naturally extends

**For learned embeddings**:
- Most difficult to extend
- Options: interpolation, extrapolation tricks, fine-tuning

---

## 10. Summary

### Method Quick Reference

| Method | Formula | Key Insight |
|--------|---------|-------------|
| **Sinusoidal** | sin/cos with varying frequencies | Linear transform = relative position |
| **Learned** | nn.Embedding(max_pos, d) | Task-specific patterns |
| **RoPE** | Rotate Q, K by position | Rotation preserves relative position in dot product |
| **ALiBi** | Linear penalty on attention scores | Distance penalty, no embeddings |

### Key Equations

**Sinusoidal**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**RoPE**:
```
q_m · k_n = qᵀ R_{n-m} k
```

**ALiBi**:
```
attention = softmax(QKᵀ/√d - m×|i-j|)
```

### Recommendations

| Scenario | Recommendation |
|----------|----------------|
| Modern LLM | RoPE (standard) |
| Need extrapolation | ALiBi or RoPE + PI |
| Simple/short sequences | Learned absolute |
| Encoder-decoder | T5-style relative |
| Just starting out | Sinusoidal (simple baseline) |

### Key Takeaways

1. **Position encoding is essential**: Transformers are permutation-invariant
2. **RoPE is the modern standard**: Used by LLaMA, Mistral, most open LLMs
3. **ALiBi excels at extrapolation**: No position embeddings = natural extension
4. **Learned embeddings are simple but limited**: Can't extrapolate
5. **Extension techniques exist**: PI, NTK, YaRN for longer contexts
6. **Choice matters for long-context**: RoPE or ALiBi if you need >2K tokens
