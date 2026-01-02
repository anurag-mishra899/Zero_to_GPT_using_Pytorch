# Module 4.2: Normalization Techniques - BatchNorm to RMSNorm

## Table of Contents
1. [Why Normalization?](#1-why-normalization)
2. [Batch Normalization](#2-batch-normalization)
3. [Layer Normalization](#3-layer-normalization)
4. [RMS Normalization](#4-rms-normalization)
5. [Other Normalization Methods](#5-other-normalization-methods)
6. [Pre-Norm vs Post-Norm](#6-pre-norm-vs-post-norm)
7. [Interview Questions](#7-interview-questions)
8. [Summary](#8-summary)

---

## 1. Why Normalization?

### 1.1 The Internal Covariate Shift Problem

As training progresses, the distribution of inputs to each layer changes:
- Layer 2's input = Layer 1's output
- As Layer 1's weights update, Layer 2's input distribution shifts
- Each layer constantly adjusts to new input statistics

This slows training and requires careful initialization.

### 1.2 Benefits of Normalization

| Benefit | Explanation |
|---------|-------------|
| **Faster training** | Allows higher learning rates |
| **Stable gradients** | Prevents vanishing/exploding |
| **Less sensitive to init** | Robust to initialization choices |
| **Regularization** | Slight regularization effect |
| **Smoother loss landscape** | Easier optimization |

### 1.3 General Normalization Formula

```
y = γ × (x - μ) / √(σ² + ε) + β

where:
  μ = mean of x (computed over some dimensions)
  σ² = variance of x (computed over same dimensions)
  γ = learnable scale (initialized to 1)
  β = learnable shift (initialized to 0)
  ε = small constant for numerical stability (1e-5)
```

Different normalizations differ in **which dimensions** compute μ and σ².

### 1.4 Comparison Overview

| Method | Normalize Over | Best For |
|--------|---------------|----------|
| BatchNorm | Batch, H, W | CNNs |
| LayerNorm | Features | Transformers |
| RMSNorm | Features (no mean) | LLMs |
| InstanceNorm | H, W | Style transfer |
| GroupNorm | Groups of channels | Small batch |

---

## 2. Batch Normalization

### 2.1 BatchNorm Formula

For input x of shape (N, C, H, W):

```
μ_c = (1 / N×H×W) × Σₙ Σₕ Σ_w x_{n,c,h,w}   (mean per channel)
σ²_c = (1 / N×H×W) × Σₙ Σₕ Σ_w (x_{n,c,h,w} - μ_c)²

y_{n,c,h,w} = γ_c × (x_{n,c,h,w} - μ_c) / √(σ²_c + ε) + β_c
```

Key: Statistics computed **across batch and spatial dimensions, per channel**.

### 2.2 Training vs Inference

**Training**:
- Use batch statistics (μ_batch, σ²_batch)
- Update running estimates using exponential moving average:
```
μ_running = momentum × μ_running + (1-momentum) × μ_batch
σ²_running = momentum × σ²_running + (1-momentum) × σ²_batch
```

**Inference**:
- Use running statistics (fixed)
- Deterministic output (same input → same output)

### 2.3 BatchNorm Properties

**Pros**:
- Very effective for CNNs
- Allows much higher learning rates
- Regularization effect (noise from batch statistics)

**Cons**:
- Requires sufficiently large batch sizes
- Different behavior train vs eval
- Problematic for RNNs/transformers (sequence length varies)
- Can't use with batch size 1

### 2.4 When NOT to Use BatchNorm

1. **Small batch sizes**: Statistics unreliable
2. **Sequence models**: Length varies, batch mixing problematic
3. **Online learning**: No batches
4. **Transformers**: LayerNorm preferred

---

## 3. Layer Normalization

### 3.1 LayerNorm Formula

For input x of shape (N, S, D) or (N, D):

```
# For each sample, normalize over features
μ_n = (1/D) × Σ_d x_{n,d}
σ²_n = (1/D) × Σ_d (x_{n,d} - μ_n)²

y_{n,d} = γ_d × (x_{n,d} - μ_n) / √(σ²_n + ε) + β_d
```

Key: Statistics computed **per sample, across features**.

### 3.2 LayerNorm vs BatchNorm

```
Input shape: (Batch, Sequence, Features) or (Batch, Features)

BatchNorm: Normalize across Batch (and spatial dims)
  Statistics: One per feature
  Parameters: 2 × Features (γ, β per feature)

LayerNorm: Normalize across Features
  Statistics: One per sample
  Parameters: 2 × Features (γ, β per feature)
```

### 3.3 LayerNorm Properties

**Pros**:
- Independent of batch size (works with batch=1)
- Same behavior train and eval
- Works well for sequences (each position independent)
- Standard for transformers

**Cons**:
- Doesn't have BatchNorm's regularization effect
- May not work as well for CNNs

### 3.4 LayerNorm in Transformers

**Standard usage**:
```python
# Pre-LN (modern):
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-LN (original):
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Why Pre-LN is preferred**:
- More stable training
- Easier gradient flow
- Can often train without warmup

---

## 4. RMS Normalization

### 4.1 RMSNorm Formula

Simplified LayerNorm without mean centering:

```
RMS(x) = √((1/D) × Σ_d x_d²)

y = γ × x / RMS(x)
```

No subtraction of mean, no β (bias) parameter.

### 4.2 Why RMSNorm?

**Computational savings**:
- No mean computation
- No mean subtraction
- Fewer parameters (no β)
- ~10-20% faster than LayerNorm

**Empirical finding**: Mean centering not necessary for good performance.

### 4.3 RMSNorm vs LayerNorm

```
LayerNorm:
  y = γ × (x - μ) / √(σ² + ε) + β
  Parameters: 2D (γ and β)
  Operations: mean, variance, subtract, divide, scale, shift

RMSNorm:
  y = γ × x / RMS(x)
  Parameters: D (only γ)
  Operations: square, mean, sqrt, divide, scale
```

### 4.4 RMSNorm in LLMs

Used in:
- LLaMA (all versions)
- Gemma
- Mistral
- Many other modern LLMs

**Why LLMs prefer RMSNorm**:
- Faster (matters at scale)
- Empirically works just as well
- Simpler implementation

---

## 5. Other Normalization Methods

### 5.1 Instance Normalization

Normalize each sample, each channel independently:

```
For input (N, C, H, W):
μ_{n,c} = mean over H, W
σ²_{n,c} = var over H, W
```

**Use case**: Style transfer (removes style information)

### 5.2 Group Normalization

Divide channels into groups, normalize within each:

```
For input (N, C, H, W) with G groups:
Each group has C/G channels
Normalize each group independently (like LayerNorm per group)
```

**Use case**: Small batch sizes (better than BatchNorm when batch < 32)

### 5.3 Weight Normalization

Normalize weights instead of activations:

```
w = g × v / ||v||

g = learnable scalar
v = weight vector
```

**Use case**: Sometimes used in sequence models

---

## 6. Pre-Norm vs Post-Norm

### 6.1 Post-Norm (Original Transformer)

```python
# Normalize AFTER residual addition
x = LayerNorm(x + Sublayer(x))
```

**Properties**:
- Original "Attention Is All You Need"
- Harder to train deep models
- Requires warmup
- Output distribution changes after each layer

### 6.2 Pre-Norm (Modern)

```python
# Normalize BEFORE sublayer, then residual
x = x + Sublayer(LayerNorm(x))
```

**Properties**:
- More stable training
- Better gradient flow
- Often no warmup needed
- Residual stream stays in original distribution

### 6.3 Why Pre-Norm is Better

**Gradient flow analysis**:
```
Post-Norm: Gradients must flow through normalization
  - Each norm can disrupt gradient flow
  - Deeper = more disruption

Pre-Norm: Residual provides direct gradient path
  - Skip connection bypasses normalization
  - Gradients flow directly back
```

**Stability**:
- Post-Norm: Output scale can grow with depth
- Pre-Norm: Residual stream maintains scale

### 6.4 Final Layer Norm

Both Pre-Norm and Post-Norm need final normalization:

```python
# After all transformer layers
x = LayerNorm(x)  # Final LN before output
logits = Linear(x)
```

This is sometimes called "RMSNorm" or "final_norm" in LLM code.

---

## 7. Interview Questions

### Q1: Explain the difference between BatchNorm and LayerNorm.

**Answer**:

**BatchNorm** (for CNNs):
```
Input: (Batch, Channels, Height, Width)
Normalize: Across Batch, H, W for each Channel
Statistics: C means, C variances (one per channel)
```
- Requires large batch sizes
- Different train/eval behavior
- Uses running statistics at inference

**LayerNorm** (for Transformers):
```
Input: (Batch, Sequence, Features)
Normalize: Across Features for each sample
Statistics: Computed per sample at runtime
```
- Works with any batch size
- Same train/eval behavior
- No running statistics

**When to use**:
- CNNs → BatchNorm
- Transformers/RNNs → LayerNorm

### Q2: What is RMSNorm and why do LLMs use it?

**Answer**:

**RMSNorm formula**:
```
RMS(x) = √(mean(x²))
y = γ × x / RMS(x)
```

**Differences from LayerNorm**:
1. No mean centering (no subtraction)
2. No bias term (no β)
3. Simpler computation

**Why LLMs use it**:
1. **Speed**: 10-20% faster (no mean computation)
2. **Fewer parameters**: Half the parameters (no β)
3. **Works just as well**: Empirically equivalent quality
4. **Scale matters**: At LLM scale, small speedups add up

**Used in**: LLaMA, Gemma, Mistral, etc.

### Q3: Explain Pre-Norm vs Post-Norm in transformers.

**Answer**:

**Post-Norm** (original):
```python
x = LayerNorm(x + Attention(x))
```
- Normalize after residual addition
- Gradients flow through normalization
- Harder to train deep models
- Requires warmup

**Pre-Norm** (modern):
```python
x = x + Attention(LayerNorm(x))
```
- Normalize before sublayer
- Residual provides clean gradient path
- More stable, easier to train
- Often no warmup needed

**Why Pre-Norm is preferred**:
- Better gradient flow (residual bypasses norm)
- Maintains scale of residual stream
- Standard in all modern LLMs

### Q4: Why can't we use BatchNorm for transformers?

**Answer**:

**Issues with BatchNorm for transformers**:

1. **Variable sequence lengths**:
   - Different sequences in batch have different lengths
   - Statistics would mix information across positions

2. **Batch dependency**:
   - Each sample depends on other samples in batch
   - Problematic for autoregressive generation

3. **Train/eval mismatch**:
   - Running statistics computed on training batches
   - Inference with batch=1 uses different statistics

4. **Sequence independence**:
   - Each position should be processed independently
   - BatchNorm mixes information across positions

**LayerNorm solution**:
- Normalizes each position independently
- No batch dependency
- Same behavior train/eval

### Q5: How do you implement LayerNorm efficiently?

**Answer**:

**Naive implementation**:
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
y = (x - mean) / sqrt(var + eps)
y = y * gamma + beta
```

**Fused implementation** (faster):
- Single kernel computes mean, var, normalize, scale, shift
- Avoids multiple memory reads/writes
- Uses Welford's algorithm for numerical stability

**Key optimizations**:
1. Fused CUDA kernel
2. Welford's algorithm for variance
3. Mixed precision (compute stats in FP32)

---

## 8. Summary

### Quick Reference

| Method | Normalize Over | Parameters | Use Case |
|--------|---------------|------------|----------|
| BatchNorm | Batch, H, W | 2C | CNNs |
| LayerNorm | Features | 2D | Transformers |
| RMSNorm | Features | D | LLMs |
| GroupNorm | Channel groups | 2C | Small batch |
| InstanceNorm | H, W | 2C | Style transfer |

### Key Equations

**LayerNorm**:
```
μ = mean(x, dim=-1)
σ² = var(x, dim=-1)
y = γ × (x - μ) / √(σ² + ε) + β
```

**RMSNorm**:
```
RMS = √(mean(x²))
y = γ × x / RMS
```

### Modern LLM Configuration

```
Normalization: RMSNorm
Position: Pre-norm
  x = x + Attention(RMSNorm(x))
  x = x + FFN(RMSNorm(x))

Final layer: RMSNorm before output projection

Parameters:
  γ only (no β)
  eps = 1e-5 or 1e-6
```

### Key Takeaways

1. **LayerNorm for transformers**: Independent of batch, same train/eval
2. **RMSNorm for LLMs**: Faster, works just as well
3. **Pre-Norm preferred**: Better gradient flow, more stable
4. **BatchNorm for CNNs**: Still best for computer vision
5. **Always use final norm**: Before output projection
