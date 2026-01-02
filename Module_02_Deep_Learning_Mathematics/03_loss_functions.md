# Module 2.3: Loss Functions for Deep Learning and LLMs

## Table of Contents
1. [Loss Function Fundamentals](#1-loss-function-fundamentals)
2. [Regression Losses](#2-regression-losses)
3. [Classification Losses](#3-classification-losses)
4. [Language Model Losses](#4-language-model-losses)
5. [Contrastive and Embedding Losses](#5-contrastive-and-embedding-losses)
6. [Advanced LLM Losses](#6-advanced-llm-losses)
7. [Numerical Stability](#7-numerical-stability)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Loss Function Fundamentals

### 1.1 What is a Loss Function?

A loss function (also called cost function or objective function) measures how well a model's predictions match the target values.

```
Loss = f(predictions, targets)

Goal: Minimize loss through gradient descent
```

### 1.2 Properties of Good Loss Functions

| Property | Why It Matters |
|----------|----------------|
| **Differentiable** | Required for gradient-based optimization |
| **Bounded below** | Ensures optimization has a target (usually ≥ 0) |
| **Convex** | Guarantees global minimum (nice to have, not required) |
| **Meaningful** | Correlates with actual task performance |
| **Numerically stable** | Works with float32/16 without overflow |

### 1.3 Loss vs Metric

```
Loss Function: Used during training (must be differentiable)
    - Cross-entropy, MSE, etc.

Evaluation Metric: Used to measure real-world performance
    - Accuracy, F1, BLEU, perplexity
    - May not be differentiable (e.g., accuracy)
```

**Important distinction**:
- We minimize **loss** during training
- We report **metrics** for model comparison
- Sometimes they're the same (e.g., MSE), sometimes not (accuracy)

---

## 2. Regression Losses

### 2.1 Mean Squared Error (MSE) / L2 Loss

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²

Gradient: ∂MSE/∂ŷᵢ = (2/n)(ŷᵢ - yᵢ)
```

**Properties**:
- Penalizes large errors heavily (squared)
- Sensitive to outliers
- Assumes Gaussian noise in data
- Always positive, minimum at 0

**Use cases**:
- Regression tasks
- Reconstruction loss (autoencoders)
- Not for classification (use cross-entropy)

### 2.2 Mean Absolute Error (MAE) / L1 Loss

```
MAE = (1/n) Σ |yᵢ - ŷᵢ|

Gradient: ∂MAE/∂ŷᵢ = (1/n) × sign(ŷᵢ - yᵢ)
```

**Properties**:
- Linear penalty (not squared)
- Robust to outliers
- Non-differentiable at 0 (use subgradient)
- Assumes Laplacian noise

**When to use MAE over MSE**:
- Data has outliers
- Equal penalty for all error sizes desired
- Median prediction wanted (vs mean for MSE)

### 2.3 Huber Loss (Smooth L1)

```
Huber(y, ŷ) =
    0.5(y - ŷ)²           if |y - ŷ| ≤ δ
    δ|y - ŷ| - 0.5δ²      otherwise

δ is the threshold (typically 1.0)
```

**Properties**:
- Combines MSE (small errors) and MAE (large errors)
- Quadratic for small errors → smooth gradients
- Linear for large errors → robust to outliers
- Differentiable everywhere

**Use cases**:
- Regression with potential outliers
- Object detection (bounding box regression)
- Reinforcement learning (TD error)

### 2.4 Comparison Table

| Loss | Sensitivity to Outliers | Gradient at 0 | Statistical Assumption |
|------|------------------------|---------------|----------------------|
| MSE | High | Smooth (0) | Gaussian noise |
| MAE | Low | Undefined | Laplacian noise |
| Huber | Medium | Smooth (0) | Hybrid |

---

## 3. Classification Losses

### 3.1 Binary Cross-Entropy (BCE)

For binary classification where ŷ ∈ (0, 1) is probability of positive class:

```
BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]

where:
  y ∈ {0, 1} is the true label
  ŷ ∈ (0, 1) is predicted probability

Gradient: ∂BCE/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
                  = (ŷ - y) / (ŷ(1-ŷ))
```

**Derivation from maximum likelihood**:
```
P(y|x) = ŷʸ × (1-ŷ)^(1-y)    (Bernoulli distribution)

Log-likelihood: log P = y log(ŷ) + (1-y) log(1-ŷ)

Negative log-likelihood = BCE
```

**Properties**:
- Measures divergence between predicted and true distributions
- Penalizes confident wrong predictions heavily
- Works with sigmoid output

### 3.2 BCE with Logits (More Stable)

When input is logits z (before sigmoid):

```
BCE_logits(y, z) = max(z, 0) - z×y + log(1 + e^(-|z|))

This is equivalent to: -[y log(σ(z)) + (1-y) log(1-σ(z))]
but numerically stable
```

**Why more stable**:
- Avoids computing σ(z) explicitly (overflow/underflow)
- Single formula handles both positive and negative z
- Used in practice: `F.binary_cross_entropy_with_logits`

### 3.3 Categorical Cross-Entropy

For multi-class classification with C classes:

```
CE = -Σᵢ yᵢ log(ŷᵢ)

where:
  y is one-hot encoded [0,0,1,0,...] or class index
  ŷ is softmax output (probabilities summing to 1)

For class index c:
  CE = -log(ŷ_c)
```

**Combined with Softmax**:
```
ŷᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)    (softmax)

CE = -log(exp(z_c) / Σⱼ exp(zⱼ))
   = -z_c + log(Σⱼ exp(zⱼ))
   = -z_c + LogSumExp(z)
```

**Gradient (combined CE + Softmax)**:
```
∂CE/∂zᵢ = ŷᵢ - yᵢ = softmax(z)ᵢ - yᵢ

This is remarkably simple!
For the correct class c: gradient = ŷ_c - 1
For other classes i: gradient = ŷᵢ
```

### 3.4 Cross-Entropy with Label Smoothing

Prevents overconfidence by softening hard labels:

```
y_smooth = (1 - ε)y + ε/K

where:
  ε = smoothing factor (e.g., 0.1)
  K = number of classes
  y = original one-hot label

CE_smooth = -Σᵢ y_smooth,ᵢ × log(ŷᵢ)
```

**Why label smoothing**:
- Prevents model from being too confident
- Acts as regularization
- Improves calibration
- Used in most modern transformers (ε = 0.1 common)

**Example**:
```
Original label: [0, 0, 1, 0] (class 2)
With ε=0.1, K=4:
  Smoothed: [0.025, 0.025, 0.925, 0.025]
```

### 3.5 Focal Loss (For Class Imbalance)

Addresses class imbalance by down-weighting easy examples:

```
FL(p) = -αₜ(1 - pₜ)^γ log(pₜ)

where:
  pₜ = probability of correct class
  α = class weight
  γ = focusing parameter (typically 2)
```

**How it works**:
- Easy examples (high pₜ): (1-pₜ)^γ is small → low loss
- Hard examples (low pₜ): (1-pₜ)^γ is large → high loss
- Focuses training on hard, misclassified examples

**Use cases**:
- Object detection (many background, few objects)
- Imbalanced classification

---

## 4. Language Model Losses

### 4.1 Causal Language Modeling Loss (GPT-style)

The fundamental LLM training objective:

```
L = -Σₜ log P(xₜ | x₁, x₂, ..., xₜ₋₁)

For a sequence x = [x₁, x₂, ..., xₙ]:
  L = -(1/n) Σₜ log softmax(logits[t])_{xₜ}
```

**Implementation**:
```python
# Logits: [batch, seq_len, vocab_size]
# Labels: [batch, seq_len] - token IDs

# Shift for next-token prediction
shift_logits = logits[..., :-1, :]  # [batch, seq-1, vocab]
shift_labels = labels[..., 1:]       # [batch, seq-1]

# Cross-entropy (flattened)
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1)
)
```

**Key aspects**:
- **Shifting**: Predict next token from current position
- **Autoregressive**: Each position only sees past tokens
- **Per-token loss**: Average over all positions
- **Ignore padding**: Use `ignore_index=-100`

### 4.2 Masked Language Modeling Loss (BERT-style)

Predict randomly masked tokens:

```
L = -(1/|M|) Σᵢ∈M log P(xᵢ | x_masked)

where M is the set of masked positions (typically 15% of tokens)
```

**Masking strategy**:
```
For each selected position (15% of tokens):
  80%: Replace with [MASK]
  10%: Replace with random token
  20%: Keep original token
```

**Why this strategy**:
- [MASK] doesn't appear during inference → 10% random prevents overfitting to [MASK]
- Keep 10% original → model doesn't assume all masked positions need changing

### 4.3 Perplexity

The standard metric for language models:

```
Perplexity = exp(Cross-Entropy Loss)
           = exp(-(1/n) Σₜ log P(xₜ | x<t))
```

**Interpretation**:
- Average "branching factor" at each position
- How many choices the model is uncertain between
- **Lower is better**

**Example perplexities**:
```
Random model (vocab 50k): ~50,000
GPT-2: ~20-40 (depending on dataset)
GPT-3: ~15-25
State-of-art: < 10 on some benchmarks
```

**Comparing models**:
- Only compare on same test set!
- Different tokenizers = different perplexity (can't compare!)
- Bits-per-character normalizes across tokenizers

### 4.4 Sequence-to-Sequence Loss

For encoder-decoder models (T5, BART):

```
L = -Σₜ log P(yₜ | y<t, encoder(x))

where:
  x = input sequence (encoded once)
  y = output sequence (generated autoregressively)
```

---

## 5. Contrastive and Embedding Losses

### 5.1 Contrastive Loss

Learn embeddings where similar items are close:

```
L = (1-y) × D² + y × max(0, margin - D)²

where:
  D = ||f(x₁) - f(x₂)||  (distance between embeddings)
  y = 0 if similar, 1 if dissimilar
  margin = minimum distance for dissimilar pairs
```

**Effect**:
- Similar pairs (y=0): Minimize distance D
- Dissimilar pairs (y=1): Push apart until distance > margin

### 5.2 Triplet Loss

More effective than contrastive loss:

```
L = max(0, D(a, p) - D(a, n) + margin)

where:
  a = anchor sample
  p = positive (same class as anchor)
  n = negative (different class from anchor)
  D = distance function
```

**Goal**: D(anchor, positive) < D(anchor, negative) - margin

**Hard negative mining**: Select negatives that are close to anchor for faster learning.

### 5.3 InfoNCE / Contrastive Loss (SimCLR, CLIP)

The modern contrastive learning objective:

```
L = -log(exp(sim(z, z⁺)/τ) / Σⱼ exp(sim(z, zⱼ)/τ))

where:
  z = anchor embedding
  z⁺ = positive embedding (augmented version or paired item)
  zⱼ = all embeddings in batch (including negatives)
  τ = temperature (typically 0.07-0.5)
  sim = cosine similarity
```

**Used in**:
- CLIP (image-text matching)
- SimCLR (self-supervised vision)
- Sentence transformers

**Temperature effect**:
- Low τ → sharper distribution → hard negatives matter more
- High τ → softer distribution → all negatives contribute equally

### 5.4 Cosine Embedding Loss

Direct optimization of cosine similarity:

```
L = 1 - cos(x₁, x₂)     if y = 1 (similar)
    max(0, cos(x₁, x₂) - margin)   if y = -1 (dissimilar)
```

---

## 6. Advanced LLM Losses

### 6.1 Knowledge Distillation Loss

Transfer knowledge from large (teacher) to small (student) model:

```
L = α × L_CE(student_logits, hard_labels)
  + (1-α) × T² × KL(softmax(student_logits/T) || softmax(teacher_logits/T))

where:
  T = temperature (typically 2-20)
  α = weighting factor
```

**Why temperature**:
- Softens probability distribution
- Reveals "dark knowledge" in teacher's non-top predictions
- Higher T → softer distributions → more knowledge transfer

**Components**:
1. **Hard loss**: Standard CE with true labels
2. **Soft loss**: KL divergence with teacher's soft predictions

### 6.2 DPO Loss (Direct Preference Optimization)

Modern alternative to RLHF for alignment:

```
L_DPO = -log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

where:
  y_w = preferred (winning) response
  y_l = dispreferred (losing) response
  π_θ = policy (model being trained)
  π_ref = reference policy (original model)
  β = strength parameter
```

**Simplified interpretation**:
- Increase probability of preferred responses relative to reference
- Decrease probability of dispreferred responses relative to reference
- No need for reward model or RL!

### 6.3 RLHF Loss Components

Reinforcement Learning from Human Feedback:

**Reward Model Training**:
```
L_reward = -log σ(r_θ(x, y_w) - r_θ(x, y_l))

Train to assign higher reward to preferred response
```

**PPO Policy Loss**:
```
L_PPO = -min(r_t(θ) × Â_t, clip(r_t(θ), 1-ε, 1+ε) × Â_t)

where:
  r_t(θ) = π_θ(a|s) / π_θ_old(a|s)  (probability ratio)
  Â_t = advantage estimate
  ε = clipping parameter (e.g., 0.2)
```

**KL Penalty**:
```
L = L_policy - β × KL(π_θ || π_ref)

Prevents drifting too far from original model
```

### 6.4 SFT (Supervised Fine-Tuning) Loss

Standard fine-tuning on instruction data:

```
L_SFT = -Σₜ∈response log P(xₜ | x<t, instruction)

Only compute loss on response tokens, not instruction
```

**Implementation**:
```python
# Labels: instruction tokens = -100 (ignored), response tokens = actual IDs
labels = torch.where(is_response_token, input_ids, -100)
loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

---

## 7. Numerical Stability

### 7.1 Log-Sum-Exp Trick

Prevent overflow in softmax/cross-entropy:

```
log(Σᵢ exp(xᵢ)) = max(x) + log(Σᵢ exp(xᵢ - max(x)))

This is numerically stable because:
  exp(xᵢ - max(x)) ≤ 1 for all i
```

**Implementation**:
```python
def log_sum_exp(x, dim=-1):
    max_x = x.max(dim=dim, keepdim=True)[0]
    return max_x + torch.log(torch.exp(x - max_x).sum(dim=dim, keepdim=True))
```

### 7.2 Cross-Entropy with Logits

Always use the combined version:

```python
# WRONG (unstable)
probs = F.softmax(logits, dim=-1)
loss = -torch.log(probs[target])

# RIGHT (stable)
loss = F.cross_entropy(logits, target)
```

### 7.3 Clipping Probabilities

When computing log(p), clip to avoid log(0):

```python
# Clip probabilities to avoid log(0)
eps = 1e-7
log_probs = torch.log(probs.clamp(min=eps))

# Or use log_softmax directly
log_probs = F.log_softmax(logits, dim=-1)
```

### 7.4 Mixed Precision Considerations

With FP16/BF16 training:

```python
# Compute loss in FP32 for stability
with torch.cuda.amp.autocast(enabled=False):
    loss = F.cross_entropy(
        logits.float(),  # Cast to FP32
        labels
    )
```

---

## 8. Interview Questions

### Q1: Explain why cross-entropy is used for classification instead of MSE.

**Answer**:

1. **Gradient magnitude**:
   - MSE gradient: 2(ŷ - y) × ŷ(1-ŷ) [due to sigmoid derivative]
   - CE gradient: ŷ - y (much cleaner, no sigmoid derivative)
   - CE has stronger gradients for wrong predictions

2. **Probabilistic interpretation**:
   - CE = negative log-likelihood
   - Minimizing CE = maximizing likelihood
   - MSE doesn't have clean probabilistic meaning for classification

3. **Optimization landscape**:
   - CE is convex w.r.t. logits (with softmax)
   - MSE has problematic plateaus due to sigmoid saturation

4. **Matching output distribution**:
   - Cross-entropy naturally pairs with softmax
   - Measures divergence between probability distributions

### Q2: What is label smoothing and why use it?

**Answer**:

**What**: Replace hard labels [0,0,1,0] with soft labels [0.025, 0.025, 0.925, 0.025]

```
y_smooth = (1-ε)y + ε/K
```

**Benefits**:
1. **Prevents overconfidence**: Model doesn't push logits to infinity
2. **Better calibration**: Predicted probabilities more reliable
3. **Regularization**: Acts as entropy regularizer
4. **Generalization**: Often improves test performance

**Common value**: ε = 0.1 in transformers

### Q3: Explain perplexity and how it relates to cross-entropy.

**Answer**:

```
Perplexity = exp(Cross-Entropy)
           = exp(-(1/n) Σ log P(xₜ|x<t))
```

**Interpretation**:
- Average number of equally likely choices at each position
- If perplexity = 10, model is as uncertain as choosing from 10 options
- **Lower is better**

**Key points**:
- Perplexity 1 = perfect prediction
- Perplexity = vocab_size = random guessing
- Can only compare models with same tokenizer!

### Q4: How does the loss differ between GPT (causal LM) and BERT (masked LM)?

**Answer**:

**GPT (Causal LM)**:
```
L = -Σₜ log P(xₜ | x₁, ..., xₜ₋₁)
```
- Predict next token from all previous tokens
- Unidirectional (left-to-right)
- Loss on all tokens

**BERT (Masked LM)**:
```
L = -Σᵢ∈M log P(xᵢ | x_context)
```
- Predict masked tokens from surrounding context
- Bidirectional
- Loss only on masked positions (15%)

**Trade-offs**:
- GPT: Natural for generation
- BERT: Better for understanding tasks
- Modern preference: GPT-style for scaling

### Q5: What is DPO and how does it differ from RLHF?

**Answer**:

**RLHF**:
1. Train reward model from preferences
2. Use RL (PPO) to maximize reward
3. Complex, unstable, requires reward model

**DPO**:
1. Direct optimization from preferences
2. No separate reward model needed
3. Simpler, more stable

**DPO Loss**:
```
L = -log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
```

**Key insight**: DPO derives the optimal policy directly from preferences without RL.

---

## 9. Summary

### Quick Reference

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | Σ(y-ŷ)² | Regression |
| MAE | Σ\|y-ŷ\| | Robust regression |
| BCE | -[y log(ŷ) + (1-y)log(1-ŷ)] | Binary classification |
| CE | -Σᵢ yᵢ log(ŷᵢ) | Multi-class classification |
| CE (LM) | -Σₜ log P(xₜ\|x<t) | Language modeling |
| InfoNCE | -log(exp(sim⁺/τ)/Σexp(simⱼ/τ)) | Contrastive learning |
| KD | α×CE + (1-α)×T²×KL | Distillation |
| DPO | -log σ(β × Δ log-probs) | Preference alignment |

### Key Takeaways

1. **Cross-entropy for classification**: Natural pairing with softmax, clean gradients
2. **Label smoothing**: Standard in transformers (ε=0.1), prevents overconfidence
3. **LM loss is cross-entropy**: Over vocabulary at each position
4. **Perplexity = exp(CE)**: Standard LM metric, lower is better
5. **Numerical stability matters**: Use `*_with_logits` functions
6. **DPO simplifies alignment**: No RL needed, direct optimization from preferences

### Modern LLM Training Stack

```
Pre-training:   Causal LM loss (next token prediction)
                ↓
SFT:           CE loss on response tokens only
                ↓
Alignment:     DPO/RLHF (preference optimization)
```
