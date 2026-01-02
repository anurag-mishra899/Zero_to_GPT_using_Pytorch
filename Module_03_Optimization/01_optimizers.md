# Module 3.1: Optimizers - From SGD to AdamW

## Table of Contents
1. [Gradient Descent Fundamentals](#1-gradient-descent-fundamentals)
2. [SGD and Momentum](#2-sgd-and-momentum)
3. [Adaptive Learning Rate Methods](#3-adaptive-learning-rate-methods)
4. [Adam and Variants](#4-adam-and-variants)
5. [Optimizer Selection Guide](#5-optimizer-selection-guide)
6. [Implementation Details](#6-implementation-details)
7. [Interview Questions](#7-interview-questions)
8. [Summary](#8-summary)

---

## 1. Gradient Descent Fundamentals

### 1.1 The Optimization Problem

Neural network training is an optimization problem:

```
θ* = argmin_θ L(θ)

where:
  θ = model parameters (weights)
  L = loss function
  θ* = optimal parameters
```

### 1.2 Gradient Descent Update Rule

Basic update:
```
θ_{t+1} = θ_t - η × ∇L(θ_t)

where:
  η = learning rate
  ∇L = gradient of loss w.r.t. parameters
```

### 1.3 Types of Gradient Descent

**Batch Gradient Descent**:
```
∇L = (1/N) Σᵢ ∇L(xᵢ, yᵢ; θ)

- Uses ALL training examples
- Slow per update
- Smooth convergence
- Memory intensive
```

**Stochastic Gradient Descent (SGD)**:
```
∇L ≈ ∇L(xᵢ, yᵢ; θ)  (single random example)

- Fast per update
- Noisy gradients
- Can escape local minima
- High variance
```

**Mini-batch Gradient Descent** (Most Common):
```
∇L ≈ (1/B) Σᵢ∈batch ∇L(xᵢ, yᵢ; θ)

- Balance of speed and stability
- Batch size B typically 32-512
- Enables GPU parallelism
- Standard in practice
```

### 1.4 Learning Rate Importance

```
η too small:
  - Very slow convergence
  - May get stuck in local minima

η too large:
  - Overshoots minima
  - Diverges (loss → ∞)
  - Oscillates without converging

η just right:
  - Steady progress toward minimum
  - Good balance of speed and stability
```

---

## 2. SGD and Momentum

### 2.1 Vanilla SGD

```
Update rule:
  θ_{t+1} = θ_t - η × g_t

where g_t = ∇L(θ_t)
```

**Problems**:
1. Slow convergence in ravines (one direction much steeper)
2. Oscillates across steep dimensions
3. Gets stuck in saddle points

### 2.2 SGD with Momentum

Key insight: Accumulate gradients to build "velocity"

```
v_{t+1} = β × v_t + g_t
θ_{t+1} = θ_t - η × v_{t+1}

where:
  v = velocity (accumulated gradients)
  β = momentum coefficient (typically 0.9)
```

**Equivalent formulation** (more common):
```
v_{t+1} = β × v_t + η × g_t
θ_{t+1} = θ_t - v_{t+1}
```

**Why momentum helps**:
- Accelerates in consistent gradient directions
- Dampens oscillations in inconsistent directions
- Effectively increases learning rate in low-curvature directions
- Helps escape shallow local minima

**Physical analogy**:
- Ball rolling down hill
- Builds up speed in consistent direction
- Momentum carries it through small bumps

### 2.3 Nesterov Accelerated Gradient (NAG)

Key insight: Look ahead before computing gradient

```
# Standard momentum: compute gradient at current position
v_{t+1} = β × v_t + η × ∇L(θ_t)

# Nesterov: compute gradient at "lookahead" position
v_{t+1} = β × v_t + η × ∇L(θ_t - β × v_t)
θ_{t+1} = θ_t - v_{t+1}
```

**Why Nesterov is better**:
- Corrects overshooting
- Gradient at lookahead position is more accurate
- Faster convergence in practice

**PyTorch implementation** (equivalent form):
```
v_{t+1} = β × v_t + g_t
θ_{t+1} = θ_t - η × (g_t + β × v_{t+1})
```

### 2.4 Momentum Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| β = 0 | No momentum | Vanilla SGD |
| β = 0.9 | Standard | Good for most cases |
| β = 0.99 | High momentum | Very smooth, slower to adapt |

---

## 3. Adaptive Learning Rate Methods

### 3.1 The Problem with Fixed Learning Rate

Different parameters need different learning rates:
- Frequent features → smaller updates (already well-trained)
- Rare features → larger updates (need more training)

### 3.2 Adagrad (2011)

Adapt learning rate based on historical gradient magnitudes:

```
# Accumulate squared gradients
G_t = G_{t-1} + g_t²

# Update with adaptive learning rate
θ_{t+1} = θ_t - η × g_t / (√G_t + ε)

where ε ≈ 1e-8 (numerical stability)
```

**Effect**:
- Parameters with large historical gradients get smaller updates
- Parameters with small historical gradients get larger updates

**Problem**:
- G_t monotonically increases
- Learning rate → 0 over time
- Training stops prematurely

### 3.3 RMSprop (2012)

Fix Adagrad's diminishing learning rate with exponential moving average:

```
# Exponential moving average of squared gradients
v_t = ρ × v_{t-1} + (1-ρ) × g_t²

# Update
θ_{t+1} = θ_t - η × g_t / (√v_t + ε)

where ρ ≈ 0.9 (decay rate)
```

**Improvements**:
- Forgets old gradients (exponential decay)
- Learning rate doesn't vanish
- Adapts to recent gradient magnitudes

### 3.4 Adadelta (2012)

RMSprop without learning rate hyperparameter:

```
# Accumulated gradient
v_t = ρ × v_{t-1} + (1-ρ) × g_t²

# Accumulated updates
u_t = ρ × u_{t-1} + (1-ρ) × Δθ_t²

# Update (no learning rate!)
Δθ_t = -√(u_{t-1} + ε) / √(v_t + ε) × g_t
θ_{t+1} = θ_t + Δθ_t
```

Less commonly used today; Adam is preferred.

---

## 4. Adam and Variants

### 4.1 Adam (Adaptive Moment Estimation, 2014)

Combines momentum AND adaptive learning rates:

```
# First moment (mean of gradients, like momentum)
m_t = β₁ × m_{t-1} + (1-β₁) × g_t

# Second moment (mean of squared gradients, like RMSprop)
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²

# Bias correction (crucial for early training)
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

# Update
θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)
```

**Default hyperparameters**:
```
β₁ = 0.9    (momentum)
β₂ = 0.999  (RMSprop-like)
ε = 1e-8    (numerical stability)
η = 0.001   (learning rate)
```

### 4.2 Why Bias Correction?

At initialization (t=0):
```
m_0 = 0, v_0 = 0

m_1 = (1-β₁) × g_1  (biased toward 0!)
v_1 = (1-β₂) × g_1²  (biased toward 0!)
```

The correction compensates:
```
m̂_1 = m_1 / (1-β₁) = g_1  (unbiased!)

As t → ∞: (1-β^t) → 1, so m̂ → m
```

### 4.3 AdamW (Adam with Decoupled Weight Decay, 2017)

**The Problem with L2 Regularization in Adam**:

L2 regularization adds λθ to gradient:
```
g̃ = g + λθ  (regularized gradient)
```

In Adam, this becomes part of the adaptive terms, which is wrong!
The weight decay should be applied AFTER adaptation.

**AdamW Solution**:
```
# Standard Adam update (without L2 in gradient)
m_t = β₁ × m_{t-1} + (1-β₁) × g_t
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²

# Bias correction
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

# Update with DECOUPLED weight decay
θ_{t+1} = θ_t - η × (m̂_t / (√v̂_t + ε) + λ × θ_t)
                    ↑ Adam update            ↑ Weight decay
```

**Why AdamW is better**:
- Weight decay works as intended (actual regularization)
- Better generalization
- Standard for transformers and LLMs

### 4.4 Adam Variants Comparison

| Optimizer | Momentum | Adaptive LR | Weight Decay | Use Case |
|-----------|----------|-------------|--------------|----------|
| SGD+M | Yes | No | L2 (coupled) | CNNs |
| Adam | Yes | Yes | L2 (coupled) | General |
| AdamW | Yes | Yes | Decoupled | Transformers |
| LAMB | Yes | Yes | Decoupled + Layer-wise | Large batch |

### 4.5 Other Adam Variants

**RAdam (Rectified Adam)**:
- Addresses variance issue in early training
- Automatically warms up learning rate

**AdaFactor**:
- Memory-efficient (factorizes second moment)
- Used in T5, large models

**8-bit Adam**:
- Quantized optimizer states
- Much lower memory usage
- Used in bitsandbytes library

---

## 5. Optimizer Selection Guide

### 5.1 Quick Decision Tree

```
Task: Computer Vision (CNN)?
  → SGD + Momentum (0.9) + Weight Decay
  → Learning rate: 0.1, decay 10x at epochs [30, 60, 90]

Task: NLP / Transformers?
  → AdamW
  → β₁=0.9, β₂=0.999, ε=1e-8
  → Learning rate: 1e-4 to 5e-4
  → Weight decay: 0.01 to 0.1

Task: LLM Pre-training?
  → AdamW
  → β₁=0.9, β₂=0.95 (lower for stability)
  → Learning rate: peak 3e-4, with warmup + cosine decay
  → Weight decay: 0.1

Task: Fine-tuning?
  → AdamW with lower learning rate
  → Learning rate: 1e-5 to 5e-5
  → Maybe freeze some layers
```

### 5.2 Hyperparameter Guidelines

**Learning Rate**:
```
SGD: Start with 0.1, decay 10x every ~30 epochs
Adam/AdamW: Start with 1e-3 to 3e-4

Rule of thumb:
- Too high: Loss oscillates or diverges
- Too low: Very slow progress
- Just right: Steady decrease in loss
```

**Weight Decay**:
```
CNNs: 1e-4 to 5e-4
Transformers: 0.01 to 0.1
Fine-tuning: 0.01

Note: Weight decay helps prevent overfitting
Higher values = stronger regularization
```

**Batch Size**:
```
Larger batch → can use larger learning rate
Linear scaling rule: lr_new = lr_base × (batch_new / batch_base)

But: larger batch may hurt generalization
Sweet spot usually 32-512
```

### 5.3 Common Mistakes

1. **Not using learning rate schedule**:
   - Always decay learning rate during training
   - Warmup helps with Adam

2. **Wrong weight decay**:
   - Use AdamW, not Adam with L2
   - Don't apply weight decay to biases/LayerNorm

3. **Same LR for all parameters**:
   - Embeddings may need smaller LR
   - Pre-trained layers need smaller LR than new layers

4. **Ignoring gradient clipping**:
   - Essential for RNNs and transformers
   - Typically clip to 1.0

---

## 6. Implementation Details

### 6.1 Gradient Clipping

**Clip by value**:
```
g = clip(g, -max_val, max_val)

Simple but can change gradient direction
```

**Clip by norm** (preferred):
```
norm = ||g||₂
if norm > max_norm:
    g = g × (max_norm / norm)

Preserves gradient direction, just scales magnitude
```

**Clip by global norm** (most common for LLMs):
```
global_norm = √(Σᵢ ||gᵢ||²)
if global_norm > max_norm:
    scale = max_norm / global_norm
    gᵢ = gᵢ × scale  for all parameters

All gradients scaled by same factor
```

### 6.2 Learning Rate Schedules

**Step Decay**:
```
lr = lr_0 × γ^(epoch // step_size)

Example: lr_0=0.1, γ=0.1, step_size=30
  Epoch 0-29: lr=0.1
  Epoch 30-59: lr=0.01
  Epoch 60-89: lr=0.001
```

**Cosine Annealing**:
```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))

Smooth decay from lr_max to lr_min over T steps
Standard for LLM training
```

**Warmup + Cosine** (LLM standard):
```
if t < warmup_steps:
    lr = lr_max × t / warmup_steps  (linear warmup)
else:
    lr = cosine_decay(t - warmup_steps)  (cosine decay)
```

**OneCycle**:
```
lr increases to max, then decreases
Also cycles momentum (opposite direction)
Can train faster with larger max_lr
```

### 6.3 Excluding Parameters from Weight Decay

Some parameters shouldn't have weight decay:
- Biases (1D tensors)
- LayerNorm parameters
- Embeddings (sometimes)

```python
# Create parameter groups
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if 'bias' in name or 'LayerNorm' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=3e-4)
```

---

## 7. Interview Questions

### Q1: Explain the difference between SGD with momentum and Adam.

**Answer**:

**SGD with Momentum**:
```
v_t = β × v_{t-1} + g_t
θ_t = θ_{t-1} - η × v_t
```
- Uses momentum to accelerate in consistent gradient directions
- Same learning rate for all parameters
- Needs careful LR tuning per problem

**Adam**:
```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t    (momentum)
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²   (adaptive LR)
θ_t = θ_{t-1} - η × m̂_t / √v̂_t
```
- Has momentum (first moment)
- Has adaptive per-parameter learning rate (second moment)
- More hyperparameters but more robust
- Faster initial convergence

**When to use which**:
- SGD+M: CNNs, when you can tune carefully
- Adam: Default, especially NLP/transformers

### Q2: Why use AdamW instead of Adam with L2 regularization?

**Answer**:

**Adam with L2**:
```
g̃ = g + λθ  (add L2 to gradient)
Then apply Adam to g̃
```

Problem: The weight decay term goes into the adaptive statistics:
- Parameters with larger L2 contribution get smaller updates
- Weight decay doesn't work as intended

**AdamW**:
```
Apply Adam to original g
Then: θ = θ - η × λ × θ  (separate weight decay)
```

Benefits:
- Weight decay truly decouples from optimization
- Acts as proper regularization
- Better generalization
- Standard for transformers

### Q3: What is the bias correction in Adam and why is it needed?

**Answer**:

Adam initializes m₀ = 0 and v₀ = 0.

After first step:
```
m_1 = 0.1 × g_1    (if β₁ = 0.9)
v_1 = 0.001 × g_1² (if β₂ = 0.999)
```

These are biased toward 0!

**Bias correction**:
```
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
```

At t=1:
```
m̂_1 = m_1 / (1-0.9) = m_1 / 0.1 = g_1  (correct!)
```

As t → ∞: correction factor → 1, becomes unnecessary.

### Q4: Explain gradient clipping and when to use it.

**Answer**:

**What**: Limit gradient magnitude to prevent exploding gradients.

**Types**:
1. Clip by value: `g = clip(g, -max, max)`
2. Clip by norm: `g = g × min(1, max_norm / ||g||)`
3. Clip by global norm: Scale all gradients by same factor

**When to use**:
- RNNs/LSTMs (long sequences cause exploding gradients)
- Transformers (standard practice, clip to 1.0)
- When seeing NaN losses

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Q5: Describe the warmup + cosine decay schedule used in LLM training.

**Answer**:

```
Warmup phase (t < warmup_steps):
  lr = lr_max × (t / warmup_steps)

Decay phase (t >= warmup_steps):
  progress = (t - warmup_steps) / (total_steps - warmup_steps)
  lr = lr_final + 0.5 × (lr_max - lr_final) × (1 + cos(π × progress))
```

**Why warmup**:
- Adam's adaptive statistics need time to stabilize
- High LR initially causes instability
- Typically 1-5% of total steps

**Why cosine decay**:
- Smooth decrease, no sudden drops
- Allows fine-grained convergence at end
- Better than step decay for transformers

---

## 8. Summary

### Quick Reference

| Optimizer | Update Rule | Best For |
|-----------|-------------|----------|
| SGD | θ = θ - η×g | Simple baselines |
| SGD+M | v = βv + g; θ = θ - η×v | CNNs |
| Adam | m/v moments + adaptive | General |
| AdamW | Adam + decoupled decay | Transformers |

### Key Equations

```
SGD:        θ = θ - η × g

Momentum:   v = β×v + g
            θ = θ - η × v

Adam:       m = β₁×m + (1-β₁)×g
            v = β₂×v + (1-β₂)×g²
            θ = θ - η × m̂ / √v̂

AdamW:      θ = θ - η × (m̂/√v̂ + λ×θ)
```

### Default Hyperparameters

**Adam/AdamW**:
```
β₁ = 0.9
β₂ = 0.999 (0.95 for LLMs)
ε = 1e-8
lr = 1e-4 to 3e-4
weight_decay = 0.01 to 0.1
```

**SGD+Momentum**:
```
momentum = 0.9
lr = 0.1 (decay 10x every ~30 epochs)
weight_decay = 1e-4
```

### LLM Training Recipe

```
Optimizer: AdamW
  β₁ = 0.9
  β₂ = 0.95
  ε = 1e-8
  weight_decay = 0.1

Learning Rate:
  peak = 3e-4
  warmup = 2000 steps
  decay = cosine to 1e-5

Gradient Clipping:
  max_norm = 1.0
```
