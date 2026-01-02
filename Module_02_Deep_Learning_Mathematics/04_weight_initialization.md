# Module 2.4: Weight Initialization - From Xavier to Modern Practices

## Table of Contents
1. [Why Initialization Matters](#1-why-initialization-matters)
2. [The Variance Problem](#2-the-variance-problem)
3. [Classic Initialization Methods](#3-classic-initialization-methods)
4. [Modern Initialization](#4-modern-initialization)
5. [Initialization for Specific Architectures](#5-initialization-for-specific-architectures)
6. [Practical Guidelines](#6-practical-guidelines)
7. [Interview Questions](#7-interview-questions)
8. [Summary](#8-summary)

---

## 1. Why Initialization Matters

### 1.1 The Problem

Neural networks are optimized using gradient descent. Bad initialization can cause:

**Vanishing Gradients**:
```
Layer 100 gradient = Layer 1 gradient × (W₁ × W₂ × ... × W₉₉)

If |Wᵢ| < 1 for all i:
  Gradient → 0 as depth increases
  Early layers don't learn
```

**Exploding Gradients**:
```
If |Wᵢ| > 1 for all i:
  Gradient → ∞ as depth increases
  NaN losses, training fails
```

**Saturated Activations**:
```
If initial values too large:
  sigmoid(large_value) ≈ 1 or 0
  Gradient ≈ 0 (saturation)
```

### 1.2 The Goal

Initialize weights so that:
1. **Variance is preserved** through forward pass
2. **Gradient variance is preserved** through backward pass
3. **Activations don't saturate** (stay in active region)
4. **Symmetry is broken** (neurons learn different features)

### 1.3 What NOT to Do

**Zero Initialization**:
```python
W = torch.zeros(out_features, in_features)
# BAD: All neurons identical, no symmetry breaking
# All gradients identical → all weights update identically
# Network can only learn one feature
```

**Constant Initialization**:
```python
W = torch.full((out_features, in_features), 0.5)
# BAD: Same problem as zeros
```

**Too Large Values**:
```python
W = torch.randn(out_features, in_features) * 10
# BAD: Activations explode, saturate
```

**Too Small Values**:
```python
W = torch.randn(out_features, in_features) * 0.0001
# BAD: Gradients vanish, very slow learning
```

---

## 2. The Variance Problem

### 2.1 Forward Pass Analysis

Consider a single neuron:
```
y = Σᵢ wᵢxᵢ    (sum over n_in inputs)

Assumptions:
- w and x are independent
- E[w] = E[x] = 0
- Var[w] = σ²_w, Var[x] = σ²_x
```

**Variance of output**:
```
Var[y] = Var[Σᵢ wᵢxᵢ]
       = Σᵢ Var[wᵢxᵢ]           (independence)
       = Σᵢ E[wᵢ²]E[xᵢ²]        (since E[w]=E[x]=0)
       = Σᵢ Var[w]·Var[x]
       = n_in × σ²_w × σ²_x
```

**To preserve variance** (Var[y] = Var[x]):
```
n_in × σ²_w × σ²_x = σ²_x
σ²_w = 1/n_in
```

### 2.2 Backward Pass Analysis

For gradients:
```
∂L/∂x = W^T × ∂L/∂y

Var[∂L/∂x] = n_out × σ²_w × Var[∂L/∂y]
```

**To preserve gradient variance**:
```
σ²_w = 1/n_out
```

### 2.3 The Conflict

Forward pass wants: σ²_w = 1/n_in
Backward pass wants: σ²_w = 1/n_out

**Solutions**:
- Xavier: Use average → σ²_w = 2/(n_in + n_out)
- He: Account for ReLU which zeros half → σ²_w = 2/n_in

---

## 3. Classic Initialization Methods

### 3.1 Xavier/Glorot Initialization (2010)

For **tanh/sigmoid** activations:

**Uniform**:
```
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Var[W] = 2/(n_in + n_out)
```

**Normal**:
```
W ~ N(0, √(2/(n_in + n_out)))

Var[W] = 2/(n_in + n_out)
```

**Derivation**:
```
Compromise between forward (1/n_in) and backward (1/n_out):
σ² = 2/(n_in + n_out)

For uniform U(-a, a): Var = a²/3
So: a²/3 = 2/(n_in + n_out)
    a = √(6/(n_in + n_out))
```

**When to use**:
- Tanh activations
- Sigmoid activations
- Linear layers without ReLU

### 3.2 He/Kaiming Initialization (2015)

For **ReLU** activations:

**Key Insight**: ReLU zeros out negative values, effectively halving the variance.

**Uniform**:
```
W ~ U(-√(6/n_in), √(6/n_in))

Var[W] = 2/n_in
```

**Normal**:
```
W ~ N(0, √(2/n_in))

Var[W] = 2/n_in
```

**Derivation**:
```
ReLU: y = max(0, x)
For x ~ N(0, σ²): E[ReLU(x)²] = σ²/2

So after ReLU, variance halves.
To compensate: σ²_w = 2/n_in (factor of 2)
```

**For Leaky ReLU (slope α for negatives)**:
```
σ²_w = 2 / ((1 + α²) × n_in)
```

**When to use**:
- ReLU activations
- Leaky ReLU, PReLU, ELU
- Most modern CNNs

### 3.3 Comparison

| Method | Variance | Best For |
|--------|----------|----------|
| Xavier Uniform | 2/(n_in + n_out) | Tanh, Sigmoid |
| Xavier Normal | 2/(n_in + n_out) | Tanh, Sigmoid |
| He Uniform | 2/n_in | ReLU |
| He Normal | 2/n_in | ReLU |

---

## 4. Modern Initialization

### 4.1 Fixup Initialization

For **very deep residual networks** (100+ layers) without normalization:

```
For residual branch:
- Scale last layer of each block by 1/√L (L = number of blocks)
- Zero-initialize the last layer of each block

For non-residual parts:
- Standard He initialization
```

**Why it works**:
- Prevents gradient explosion in deep resnets
- Allows training without BatchNorm

### 4.2 Transformer Initialization

**Standard transformer (GPT-2 style)**:

```python
# Embeddings
embedding.weight ~ N(0, 0.02)

# Linear layers
linear.weight ~ N(0, 0.02)

# Output projection (before residual)
output_proj.weight ~ N(0, 0.02/√(2*n_layers))

# Why /√(2*n_layers)?
# Each layer has 2 residual connections (attention + FFN)
# This prevents variance from exploding
```

**Modern LLM (LLaMA style)**:

```python
# Use smaller initial scale
embedding.weight ~ N(0, 1/√d_model)

# Linear layers
linear.weight ~ N(0, 1/√n_in)

# Output projection scaled
output_proj.weight ~ N(0, 1/(√n_in × √(2*n_layers)))
```

### 4.3 Initialization with Pre-Layer Norm

When using Pre-LN transformers:

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

- LayerNorm normalizes before each sub-layer
- Makes initialization less critical
- Standard N(0, 0.02) usually works

### 4.4 μP (Maximal Update Parameterization)

For **hyperparameter transfer** across model sizes:

```python
# Scale initialization and learning rate by width
# This allows hyperparameters to transfer from small to large models

# Embedding: N(0, 1)
# Hidden: N(0, 1/√width)
# Output: N(0, 1/width)

# Learning rate scaled by:
# Embedding: constant
# Hidden: 1/width
# Output: 1/width
```

**Why μP matters**:
- Tune hyperparameters on small model
- Transfer directly to large model
- Saves significant compute

---

## 5. Initialization for Specific Architectures

### 5.1 Transformers

**Attention weights**:
```python
# Q, K, V projections
W_q, W_k, W_v ~ N(0, 1/√d_model)

# Output projection (scaled for residual)
W_o ~ N(0, 1/√d_model / √(2*n_layers))
```

**FFN weights**:
```python
# SwiGLU
W_gate ~ N(0, 1/√d_model)
W_up ~ N(0, 1/√d_model)
W_down ~ N(0, 1/√d_ff / √(2*n_layers))
```

**Why scale output projections?**:
```
After n_layers, each with 2 residual connections:
  x_final = x_0 + Σ(residual_i)

Variance: Var[x_final] = Var[x_0] + 2*n_layers*Var[residual]

To keep variance constant:
  Var[residual] should be ~ 1/(2*n_layers)
  So std[weight] ~ 1/√(2*n_layers)
```

### 5.2 LSTMs/RNNs

**Standard LSTM**:
```python
# Input-to-hidden
W_ih ~ U(-√(1/hidden_size), √(1/hidden_size))

# Hidden-to-hidden
W_hh ~ U(-√(1/hidden_size), √(1/hidden_size))

# Forget gate bias
b_f = 1.0  # Important! Helps remember at start of training
```

**Why forget gate bias = 1?**:
- Sigmoid(1) ≈ 0.73 → mostly passes through
- Prevents LSTM from forgetting everything initially
- Critical for long sequences

### 5.3 CNNs

**Standard ConvNet**:
```python
# Conv layers with ReLU: He initialization
conv.weight ~ N(0, √(2/(k*k*c_in)))

where k = kernel size, c_in = input channels
```

**ResNet specifics**:
```python
# Zero-initialize residual branch's last BN
# This makes residual blocks identity at start
bn.weight = 0  # Gamma
bn.bias = 0    # Beta
```

### 5.4 Embedding Layers

```python
# Word embeddings
embedding.weight ~ N(0, 1)
# or
embedding.weight ~ N(0, 1/√d_model)

# Positional embeddings
# Usually same as word embeddings
# Or learned from data (sinusoidal doesn't need init)
```

---

## 6. Practical Guidelines

### 6.1 Quick Reference

| Architecture | Activation | Initialization |
|--------------|------------|----------------|
| MLP | ReLU | He Normal |
| MLP | Tanh/Sigmoid | Xavier Normal |
| CNN | ReLU | He Normal |
| Transformer | GELU/SiLU | N(0, 0.02) or N(0, 1/√n) |
| LSTM | Tanh/Sigmoid | Xavier + bias_f=1 |
| Embedding | - | N(0, 1) or N(0, 1/√d) |

### 6.2 PyTorch Defaults

```python
nn.Linear:     U(-√(1/n_in), √(1/n_in))  # Kaiming uniform
nn.Conv2d:     U(-√(1/(k*k*c)), √(1/(k*k*c)))
nn.Embedding:  N(0, 1)
nn.LSTM:       U(-√(1/h), √(1/h))
```

### 6.3 When to Override Defaults

1. **Deep networks without normalization**: Use Fixup or careful scaling
2. **Pre-LN transformers**: N(0, 0.02) is common
3. **Very deep models**: Scale output projections by 1/√depth
4. **Transfer learning**: Often keep pretrained, reinit head
5. **μP for scaling**: Follow μP rules exactly

### 6.4 Debugging Initialization

```python
def check_init(model, sample_input):
    """Check if initialization is reasonable."""
    model.eval()

    # Forward pass
    activations = {}
    def hook(name):
        def fn(m, i, o):
            activations[name] = o.detach()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_hook(hook(name))

    _ = model(sample_input)

    # Check statistics
    print("Activation statistics:")
    for name, act in activations.items():
        print(f"{name}: mean={act.mean():.4f}, std={act.std():.4f}")

    # Want: mean ≈ 0, std ≈ 1 (or at least not exploding/vanishing)
```

### 6.5 Common Mistakes

1. **Forgetting to init after model creation**:
```python
# Wrong: init then create modules
self.apply(init_weights)  # Runs before submodules exist!

# Right: create then init
self._init_weights()  # Called after __init__ completes
```

2. **Not handling different layer types**:
```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    # Don't forget LayerNorm, etc.
```

3. **Reinitializing pretrained weights**:
```python
# Wrong: init entire model
model = load_pretrained()
model.apply(init_weights)  # Destroys pretrained weights!

# Right: only init new layers
model.classifier.apply(init_weights)
```

---

## 7. Interview Questions

### Q1: Explain Xavier initialization and when to use it.

**Answer**:

Xavier initialization sets weight variance to preserve signal through forward and backward passes:

```
W ~ N(0, √(2/(n_in + n_out)))
```

**Derivation**:
- Forward: Want Var[output] = Var[input] → need σ² = 1/n_in
- Backward: Want Var[grad] = Var[grad_next] → need σ² = 1/n_out
- Compromise: σ² = 2/(n_in + n_out)

**When to use**:
- Tanh or sigmoid activations
- Linear layers
- NOT for ReLU (use He instead)

### Q2: Why does ReLU need different initialization than tanh?

**Answer**:

ReLU zeros out negative values, effectively halving the variance:

```
For x ~ N(0, σ²):
  E[ReLU(x)²] = σ²/2
  (Half the values are 0, half pass through)
```

To compensate, He initialization doubles the variance:

```
Xavier: σ² = 2/(n_in + n_out)
He:     σ² = 2/n_in
```

The factor of 2 compensates for ReLU's variance-halving effect.

### Q3: How do transformers handle initialization?

**Answer**:

**Key insight**: Residual connections accumulate variance.

```
After L layers with 2 residuals each:
  Var[output] ≈ 1 + 2L × Var[residual]
```

**Solutions**:

1. **Scale output projections**:
```python
W_o ~ N(0, std/√(2L))  # L = number of layers
```

2. **Pre-LayerNorm**:
```
Normalize before each sub-layer
Makes initialization less critical
```

3. **Standard practice (GPT-2)**:
```python
All weights ~ N(0, 0.02)
Output proj ~ N(0, 0.02/√(2L))
```

### Q4: What is the forget gate bias trick in LSTMs?

**Answer**:

Initialize forget gate bias to 1 (or larger):

```python
lstm.bias_hh[hidden_size:2*hidden_size] = 1.0
```

**Why**:
- Forget gate controls how much old state to keep
- sigmoid(1) ≈ 0.73 → mostly keeps old state
- Without this, LSTM forgets everything at initialization
- Crucial for learning long-term dependencies

### Q5: Explain μP (Maximal Update Parameterization).

**Answer**:

μP is an initialization + learning rate scheme that allows hyperparameter transfer across model sizes:

**Key idea**: Scale initialization and LR by width so updates have same effect regardless of size.

```
Standard parameterization:
  - Init: constant std
  - LR: constant
  - Updates don't transfer across widths

μP:
  - Init: std ~ 1/√width (for hidden layers)
  - LR: ~ 1/width (for hidden layers)
  - Updates transfer across widths!
```

**Benefit**: Tune hyperparameters on small model (cheap), transfer to large model (expensive) without re-tuning.

---

## 8. Summary

### Quick Reference

| Initialization | Formula | Use Case |
|----------------|---------|----------|
| Xavier Normal | N(0, √(2/(n_in+n_out))) | Tanh, Sigmoid |
| Xavier Uniform | U(-√(6/(n_in+n_out)), ...) | Tanh, Sigmoid |
| He Normal | N(0, √(2/n_in)) | ReLU |
| He Uniform | U(-√(6/n_in), ...) | ReLU |
| Transformer | N(0, 0.02) | Transformers |
| Scaled | N(0, σ/√(2L)) | Output projections |

### Key Takeaways

1. **Variance preservation**: Goal is to keep signal stable through layers
2. **Xavier for tanh/sigmoid**: Balanced for symmetric activations
3. **He for ReLU**: Extra factor of 2 for half-zeroed values
4. **Transformers need scaling**: Output projections scaled by 1/√(depth)
5. **LSTM forget bias**: Initialize to 1 for long-term memory
6. **Pre-LN helps**: Normalization before layers reduces init sensitivity

### Modern LLM Initialization

```python
def init_llm_weights(model, n_layers):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            nn.init.normal_(param, std=1/math.sqrt(d_model))
        elif 'output_proj' in name or 'w_down' in name:
            nn.init.normal_(param, std=1/math.sqrt(n_in)/math.sqrt(2*n_layers))
        elif param.dim() > 1:
            nn.init.normal_(param, std=1/math.sqrt(n_in))
        else:
            nn.init.zeros_(param)
```
