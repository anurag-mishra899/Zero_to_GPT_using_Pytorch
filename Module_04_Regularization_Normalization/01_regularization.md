# Module 4.1: Regularization Techniques

## Table of Contents
1. [Why Regularization?](#1-why-regularization)
2. [L1 and L2 Regularization](#2-l1-and-l2-regularization)
3. [Dropout](#3-dropout)
4. [Other Regularization Techniques](#4-other-regularization-techniques)
5. [Regularization in LLMs](#5-regularization-in-llms)
6. [Interview Questions](#6-interview-questions)
7. [Summary](#7-summary)

---

## 1. Why Regularization?

### 1.1 The Overfitting Problem

**Overfitting**: Model learns training data too well, fails to generalize.

```
Training loss: Very low
Validation loss: High and increasing
Gap between train/val: Large
```

**Signs of overfitting**:
- Training accuracy >> Validation accuracy
- Validation loss increases while training loss decreases
- Model memorizes training examples

### 1.2 Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

Bias: Error from overly simple models (underfitting)
Variance: Error from sensitivity to training data (overfitting)
```

**High bias**: Model too simple, misses patterns
**High variance**: Model too complex, memorizes noise

**Regularization reduces variance** (complexity) at cost of slightly increased bias.

### 1.3 Types of Regularization

| Category | Techniques |
|----------|------------|
| **Explicit** | L1, L2 weight decay |
| **Implicit** | Dropout, data augmentation |
| **Architectural** | Smaller models, early stopping |
| **Data-based** | More data, augmentation |

---

## 2. L1 and L2 Regularization

### 2.1 L2 Regularization (Ridge / Weight Decay)

Add squared magnitude of weights to loss:

```
L_total = L_original + λ × Σᵢ wᵢ²

where λ is regularization strength
```

**Gradient**:
```
∂L_total/∂w = ∂L_original/∂w + 2λw

Weight update:
w_new = w - η(∇L + 2λw)
      = w(1 - 2ηλ) - η∇L
      ↑ Weights "decay" toward 0
```

**Properties**:
- Pushes weights toward 0 (but rarely exactly 0)
- Smooth, differentiable
- Equivalent to Gaussian prior on weights
- Prevents any single weight from being too large

### 2.2 L1 Regularization (Lasso)

Add absolute magnitude of weights to loss:

```
L_total = L_original + λ × Σᵢ |wᵢ|
```

**Gradient**:
```
∂L_total/∂w = ∂L_original/∂w + λ × sign(w)
```

**Properties**:
- Drives weights exactly to 0 (sparsity)
- Feature selection (eliminates unimportant features)
- Non-differentiable at 0 (use subgradient)
- Equivalent to Laplacian prior on weights

### 2.3 L1 vs L2 Comparison

| Property | L1 | L2 |
|----------|----|----|
| Penalty | Σ\|w\| | Σw² |
| Sparsity | Yes (exact zeros) | No |
| Solution | Diamond corners | Circular |
| Robustness to outliers | Better | Worse |
| Feature selection | Yes | No |

**Why L2 is preferred in deep learning**:
- Smooth gradients (no kink at 0)
- Distributed small weights often better than sparse
- Works well with adaptive optimizers

### 2.4 Weight Decay vs L2 Regularization

**Important distinction for Adam**:

**L2 Regularization** (add to loss):
```
loss = original_loss + λ × ||w||²
gradient = ∇L + 2λw
```

**Weight Decay** (multiply weights):
```
w_new = w × (1 - λ) - η × ∇L
```

**For SGD**: These are equivalent (λ_decay = 2ηλ_L2)

**For Adam**: They are NOT equivalent!
- L2: Regularization term goes into Adam's statistics
- Weight decay: Applied after Adam update
- **Use AdamW** (decoupled weight decay) for transformers

### 2.5 Choosing λ

```
λ too small: No regularization effect
λ too large: Underfitting (weights too small)
λ just right: Balance between fit and generalization
```

**Typical values**:
- CNNs: 1e-4 to 5e-4
- Transformers: 0.01 to 0.1
- Fine-tuning: 0.01

---

## 3. Dropout

### 3.1 What is Dropout?

Randomly set neuron outputs to 0 during training:

```
During training:
  y = x × mask    where mask ~ Bernoulli(p)

During inference:
  y = x × p       (scale by keep probability)

Or (inverted dropout, more common):
  During training: y = (x × mask) / p
  During inference: y = x
```

### 3.2 Why Dropout Works

**Multiple explanations**:

1. **Ensemble effect**:
   - Each forward pass uses different subset of neurons
   - Like training 2^n different networks
   - Final model = average of all sub-networks

2. **Prevents co-adaptation**:
   - Neurons can't rely on specific other neurons
   - Forces redundant representations
   - More robust features

3. **Implicit regularization**:
   - Similar effect to L2 regularization
   - But adaptive (stronger where needed)

### 3.3 Dropout Rates

| Layer Type | Typical Rate |
|------------|--------------|
| Input layer | 0.2 or no dropout |
| Hidden layers | 0.5 |
| Before softmax | 0.5 |
| Attention (transformers) | 0.1 |
| Residual connections | 0.1 |

### 3.4 Dropout Variants

**Standard Dropout**:
```
mask = Bernoulli(p) for each element
y = x × mask / p
```

**Spatial Dropout** (for CNNs):
```
mask = Bernoulli(p) for each channel
y[:, c, :, :] = x[:, c, :, :] × mask[c] / p
```

**DropConnect**:
```
mask = Bernoulli(p) for each weight
y = (W × mask) × x / p
```

**DropPath / Stochastic Depth** (for ResNets):
```
# Randomly drop entire residual block
if training and random() < drop_prob:
    output = input  # Skip the block
else:
    output = input + block(input) / (1 - drop_prob)
```

### 3.5 Dropout in Transformers

**Where dropout is applied**:
1. After attention weights (attention_dropout)
2. After attention output projection
3. After FFN layers
4. In residual connections

**Important**: Always apply dropout BEFORE residual addition!

```python
# Correct
x = x + dropout(attention(x))

# Wrong (drops residual too)
x = dropout(x + attention(x))
```

---

## 4. Other Regularization Techniques

### 4.1 Early Stopping

Stop training when validation loss stops improving:

```
Best approach:
1. Monitor validation loss
2. Save model at each improvement
3. Stop after N epochs without improvement
4. Restore best model
```

**Patience**: Number of epochs to wait before stopping (typically 5-10)

### 4.2 Data Augmentation

Create synthetic training data:

**For images**:
- Random crops, flips, rotations
- Color jittering
- CutOut, MixUp, CutMix

**For text**:
- Back-translation
- Synonym replacement
- Random word dropout

**For LLMs**:
- Different tokenizations
- Document shuffling
- Span corruption (T5)

### 4.3 Label Smoothing

Soft targets instead of hard one-hot:

```
Hard: [0, 0, 1, 0]
Soft (ε=0.1): [0.025, 0.025, 0.925, 0.025]

Formula: y_smooth = (1-ε)y + ε/K
```

**Benefits**:
- Prevents overconfidence
- Better calibration
- Implicit regularization

**Common in transformers**: ε = 0.1

### 4.4 Gradient Clipping

Limit gradient magnitude:

```
if ||∇L|| > max_norm:
    ∇L = ∇L × (max_norm / ||∇L||)
```

Not traditional regularization, but prevents:
- Exploding gradients
- Large parameter updates
- Training instability

### 4.5 Mixup

Combine pairs of training examples:

```
x_mix = λ × x_a + (1-λ) × x_b
y_mix = λ × y_a + (1-λ) × y_b

where λ ~ Beta(α, α)
```

**Benefits**:
- Smoother decision boundaries
- Better generalization
- Works for many tasks

---

## 5. Regularization in LLMs

### 5.1 Dropout Usage

**Pre-training**:
- Light dropout (0.0-0.1)
- Or no dropout at all (GPT-3)
- More data = less need for dropout

**Fine-tuning**:
- Moderate dropout (0.1-0.3)
- Higher for smaller datasets

**Where to apply**:
```python
# Attention
attn_output = dropout(softmax(QK^T/√d) @ V)
output = dropout(attn_output @ W_o)

# FFN
ffn_output = dropout(act(x @ W1) @ W2)

# Residual (apply dropout to branch, not sum)
x = x + dropout(sublayer(x))
```

### 5.2 Weight Decay

**Standard LLM configuration**:
```
weight_decay = 0.1
Apply to: All weights EXCEPT
  - Biases
  - LayerNorm parameters
  - Embeddings (sometimes)
```

**Why exclude some parameters**:
- Biases: 1D, no overfitting risk
- LayerNorm: Scale/shift, not weights
- Embeddings: Some argue they should be regularized

### 5.3 Effective Regularization Through Scale

Large models regularize differently:
- More parameters = more capacity
- But also more implicit regularization
- Training on massive data = regularization through diversity

**Observation**: Large LLMs often train well without explicit regularization.

---

## 6. Interview Questions

### Q1: Explain the difference between L1 and L2 regularization.

**Answer**:

**L2 (Ridge)**:
```
Penalty: λ × Σw²
Gradient: 2λw
```
- Pushes weights toward 0 but rarely exactly 0
- Smooth, differentiable everywhere
- Penalizes large weights more (quadratic)
- Results in small, distributed weights

**L1 (Lasso)**:
```
Penalty: λ × Σ|w|
Gradient: λ × sign(w)
```
- Drives weights exactly to 0 (sparsity)
- Performs feature selection
- Not differentiable at 0
- Linear penalty regardless of magnitude

**When to use**:
- L2: Default for deep learning (smooth gradients)
- L1: When sparsity/feature selection is desired
- Elastic Net: Combination of both

### Q2: Why use AdamW instead of Adam with L2 regularization?

**Answer**:

In standard Adam with L2 regularization:
```
g_regularized = g + λw  # L2 added to gradient
# This goes into Adam's moment estimates m and v
```

Problem: The regularization term affects the adaptive learning rate computation, which is not intended.

In AdamW (decoupled):
```
# Adam update computed on original gradient
adam_update = m / (√v + ε)
# Weight decay applied separately
w = w - lr × (adam_update + λ × w)
```

**Benefits of AdamW**:
- Weight decay works as intended
- Better generalization
- Standard for transformers

### Q3: Explain dropout and why it works.

**Answer**:

**What**: Randomly zero out neurons with probability p during training.

**How**:
```python
# Training
mask = Bernoulli(keep_prob)
output = input * mask / keep_prob  # Inverted dropout

# Inference
output = input  # No changes
```

**Why it works**:

1. **Ensemble**: Each forward pass uses different sub-network; final model averages all 2^n possible networks.

2. **Co-adaptation**: Neurons can't rely on specific others, forcing robust representations.

3. **Regularization**: Similar to L2 but adaptive; stronger regularization where needed.

**Key implementation detail**: Scale by 1/keep_prob during training (inverted dropout) so inference needs no changes.

### Q4: What regularization techniques are used in modern LLMs?

**Answer**:

1. **Weight Decay** (always):
   - λ = 0.1 typically
   - Exclude biases, LayerNorm

2. **Dropout** (sometimes):
   - Pre-training: 0.0-0.1 or none
   - Fine-tuning: 0.1-0.3
   - Applied in attention and FFN

3. **Label Smoothing**:
   - ε = 0.1 common
   - Prevents overconfidence

4. **Gradient Clipping**:
   - max_norm = 1.0
   - Stability, not traditional regularization

5. **Data Regularization**:
   - Massive diverse training data
   - Document shuffling
   - Different tokenizations

**Note**: Very large models often need less explicit regularization due to implicit regularization from scale.

---

## 7. Summary

### Quick Reference

| Technique | Effect | When to Use |
|-----------|--------|-------------|
| L2 | Small weights | Always (as weight decay) |
| L1 | Sparse weights | Feature selection |
| Dropout | Random zeros | Medium datasets |
| Early stopping | Stop at best val | Always monitor |
| Label smoothing | Soft targets | Classification |
| Data augmentation | More diversity | Limited data |

### Key Equations

**L2 Regularization**:
```
L = L_original + λ × ||w||²
∂L/∂w = ∂L_original/∂w + 2λw
```

**Dropout (Inverted)**:
```
Training: y = x × mask / p, mask ~ Bernoulli(p)
Inference: y = x
```

**Label Smoothing**:
```
y_smooth = (1-ε)y + ε/K
```

### LLM Regularization Config

```
Weight Decay:
  value = 0.1
  exclude = ['bias', 'LayerNorm']

Dropout:
  attention = 0.0-0.1
  ffn = 0.0-0.1
  residual = 0.0-0.1

Label Smoothing:
  epsilon = 0.1

Gradient Clipping:
  max_norm = 1.0
```

### Key Takeaways

1. **L2/Weight decay** is standard for all deep learning
2. **Use AdamW**, not Adam + L2
3. **Dropout**: Light for pre-training, more for fine-tuning
4. **Label smoothing**: Standard for transformers (ε=0.1)
5. **More data** = less need for explicit regularization
