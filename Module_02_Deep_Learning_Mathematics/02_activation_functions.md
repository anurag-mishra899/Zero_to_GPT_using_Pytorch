# Module 2.2: Activation Functions - From ReLU to SwiGLU

## Table of Contents
1. [Why Activation Functions?](#1-why-activation-functions)
2. [Classic Activations](#2-classic-activations)
3. [Modern Activations](#3-modern-activations)
4. [Gated Activations (LLM Critical)](#4-gated-activations-llm-critical)
5. [Activation in Different Architectures](#5-activation-in-different-architectures)
6. [Mathematical Properties](#6-mathematical-properties)
7. [Implementation Details](#7-implementation-details)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Why Activation Functions?

### 1.1 The Non-Linearity Problem

Without activation functions, neural networks are just linear transformations:

```
Layer 1: h₁ = W₁x + b₁
Layer 2: h₂ = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

This collapses to a single linear transformation: **y = Wx + b**

No matter how many layers, without non-linearity, the network can only learn linear functions!

### 1.2 What Makes a Good Activation?

| Property | Why It Matters |
|----------|---------------|
| **Non-linear** | Enables learning complex functions |
| **Differentiable** | Required for gradient-based optimization |
| **Non-saturating** | Prevents vanishing gradients |
| **Zero-centered** | Helps optimization dynamics |
| **Computationally efficient** | Training speed |
| **Smooth** | Better optimization landscape |

### 1.3 Evolution of Activations

```
1990s: Sigmoid, Tanh (saturate, vanishing gradients)
   ↓
2010: ReLU (simple, fast, but dying ReLU problem)
   ↓
2015: Leaky ReLU, ELU, PReLU (address dying ReLU)
   ↓
2017: GELU (smooth, used in BERT/GPT)
   ↓
2020: SwiGLU, GeGLU (gated, state-of-the-art for LLMs)
```

---

## 2. Classic Activations

### 2.1 Sigmoid

```
σ(x) = 1 / (1 + e^(-x))

Range: (0, 1)
Derivative: σ(x)(1 - σ(x))
Max derivative: 0.25 (at x=0)
```

**Properties**:
- Smooth, differentiable everywhere
- Output bounded (0, 1) - good for probabilities
- **Problem**: Saturates at extremes → vanishing gradients
- **Problem**: Not zero-centered → slower convergence

**Use cases**:
- Output layer for binary classification
- Gates in LSTM/GRU

### 2.2 Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        = 2σ(2x) - 1

Range: (-1, 1)
Derivative: 1 - tanh²(x)
Max derivative: 1 (at x=0)
```

**Properties**:
- Zero-centered (better than sigmoid)
- Still saturates at extremes
- Stronger gradients than sigmoid (max = 1 vs 0.25)

**Use cases**:
- Hidden layers in RNNs (historically)
- Attention scores normalization (sometimes)

### 2.3 Sigmoid vs Tanh Comparison

```
           Sigmoid              Tanh
Output:    (0, 1)              (-1, 1)
Centered:  No                  Yes
Max grad:  0.25                1.0
Vanishing: Severe              Moderate
```

---

## 3. Modern Activations

### 3.1 ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)

Derivative:
  1 if x > 0
  0 if x < 0
  undefined at x = 0 (typically use 0 or 1)
```

**Properties**:
- Simple, fast computation
- No saturation for positive values
- Sparse activation (neurons can be off)
- **Problem**: "Dying ReLU" - neurons stuck at 0

**Why dying ReLU happens**:
```
If weights push all inputs negative:
  output = 0 for all inputs
  gradient = 0 for all inputs
  weights never update
  neuron is "dead"
```

### 3.2 Leaky ReLU

```
LeakyReLU(x) = x if x > 0
               αx if x ≤ 0

where α is small (e.g., 0.01)

Derivative:
  1 if x > 0
  α if x ≤ 0
```

**Properties**:
- Fixes dying ReLU (gradient always flows)
- Small negative slope allows recovery
- α is hyperparameter (or learned in PReLU)

### 3.3 ELU (Exponential Linear Unit)

```
ELU(x) = x if x > 0
         α(e^x - 1) if x ≤ 0

Derivative:
  1 if x > 0
  α·e^x = ELU(x) + α if x ≤ 0
```

**Properties**:
- Smooth everywhere (unlike ReLU)
- Negative values push mean toward zero
- More expensive than ReLU (exponential)

### 3.4 GELU (Gaussian Error Linear Unit)

```
GELU(x) = x · Φ(x)

where Φ(x) is the CDF of standard normal distribution

Approximation:
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        ≈ x · σ(1.702x)  (faster approximation)
```

**Properties**:
- Smooth, non-monotonic
- Combines properties of dropout and ReLU
- Used in BERT, GPT-2, GPT-3
- Small negative values get small negative outputs

**Why GELU for transformers**:
- Smoother gradient landscape
- Implicit regularization effect
- Works well with layer normalization

### 3.5 SiLU / Swish

```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))

Derivative:
SiLU'(x) = σ(x) + x · σ(x) · (1 - σ(x))
         = σ(x)(1 + x(1 - σ(x)))
         = SiLU(x) + σ(x)(1 - SiLU(x))
```

**Properties**:
- Smooth, non-monotonic
- Self-gated (input gates itself)
- Bounded below (~-0.28), unbounded above
- Used in EfficientNet, some LLMs

---

## 4. Gated Activations (LLM Critical)

### 4.1 Gated Linear Units (GLU)

The key insight: **split input and use half to gate the other half**

```
GLU(x) = (xW₁ + b₁) ⊗ σ(xW₂ + b₂)

where:
  First half: linear transformation (the "content")
  Second half: sigmoid gate (0 to 1)
  ⊗: element-wise multiplication
```

**Why gating helps**:
- Provides multiplicative interactions
- Allows network to learn what to pass through
- Smoother optimization landscape

### 4.2 SwiGLU (Used in LLaMA, PaLM)

```
SwiGLU(x) = (xW₁) ⊗ SiLU(xW₂)
          = (xW₁) ⊗ (xW₂ · σ(xW₂))

where SiLU(x) = x · σ(x)
```

**In transformer FFN**:
```python
# Standard FFN:
def ffn(x):
    return W2 @ relu(W1 @ x)  # hidden_dim = 4 * d_model

# SwiGLU FFN (LLaMA style):
def swiglu_ffn(x):
    gate = silu(W_gate @ x)
    up = W_up @ x
    return W_down @ (gate * up)  # hidden_dim = 8/3 * d_model
```

**Why SwiGLU**:
- Better performance than ReLU/GELU in LLMs
- Gating provides smoother gradients
- Used in LLaMA, PaLM, GPT-4 (speculated)

### 4.3 GeGLU

```
GeGLU(x) = (xW₁) ⊗ GELU(xW₂)
```

Similar to SwiGLU but uses GELU instead of SiLU.

### 4.4 FFN Dimension with Gated Activations

Standard FFN has inner dimension 4×d_model.

With gated activations (2 projections instead of 1), to match parameter count:
```
SwiGLU/GeGLU inner dimension = (8/3) × d_model ≈ 2.67 × d_model
```

This keeps total parameters similar while adding gating.

---

## 5. Activation in Different Architectures

### 5.1 CNNs

```
Typically: ReLU or variants
Why: Simple, fast, works well for image features
```

### 5.2 RNNs/LSTMs

```
Gates: Sigmoid (0-1 range needed)
State: Tanh (zero-centered)
```

### 5.3 Transformers (BERT, GPT-2)

```
FFN activation: GELU
Why: Smoother, better optimization
```

### 5.4 Modern LLMs (LLaMA, PaLM)

```
FFN activation: SwiGLU or GeGLU
Why: Better performance at scale
```

### 5.5 Summary Table

| Architecture | Activation | Reason |
|--------------|------------|--------|
| CNN | ReLU | Fast, sparse |
| LSTM gates | Sigmoid | Need [0,1] |
| LSTM state | Tanh | Zero-centered |
| BERT/GPT-2 | GELU | Smooth gradients |
| LLaMA/PaLM | SwiGLU | Best at scale |
| Output (classification) | Softmax | Probabilities |
| Output (regression) | None/Linear | Unbounded |

---

## 6. Mathematical Properties

### 6.1 Gradient Flow Comparison

At initialization (x ~ N(0,1)), expected gradient magnitude:

| Activation | E[|f'(x)|] | Gradient Flow |
|------------|------------|---------------|
| Sigmoid | ~0.2 | Poor (vanishing) |
| Tanh | ~0.4 | Moderate |
| ReLU | 0.5 | Good (but sparse) |
| GELU | ~0.6 | Good |
| SiLU | ~0.6 | Good |

### 6.2 Mean Activation Shift

For x ~ N(0,1):

| Activation | E[f(x)] | Zero-centered? |
|------------|---------|----------------|
| Sigmoid | 0.5 | No |
| Tanh | 0 | Yes |
| ReLU | ~0.4 | No |
| GELU | ~0.17 | Slightly |
| SiLU | ~0.08 | Nearly |

### 6.3 Lipschitz Constants

Lipschitz constant bounds how fast output changes with input:

| Activation | Lipschitz | Implication |
|------------|-----------|-------------|
| ReLU | 1 | Stable |
| Sigmoid | 0.25 | Very stable |
| Tanh | 1 | Stable |
| GELU | ~1.13 | Nearly stable |
| SiLU | ~1.1 | Nearly stable |

---

## 7. Implementation Details

### 7.1 Numerical Stability

**Sigmoid**:
```python
# Unstable for large negative x
def sigmoid_unstable(x):
    return 1 / (1 + np.exp(-x))  # exp(-x) overflows for large x

# Stable
def sigmoid_stable(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
```

**Softmax**:
```python
# Unstable
def softmax_unstable(x):
    return np.exp(x) / np.exp(x).sum()  # exp overflow

# Stable (subtract max)
def softmax_stable(x):
    x_max = x.max()
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum()
```

### 7.2 Approximate GELU

```python
# Exact GELU (slow)
def gelu_exact(x):
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

# Tanh approximation (fast, used in many libraries)
def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

# Sigmoid approximation (fastest)
def gelu_sigmoid(x):
    return x * torch.sigmoid(1.702 * x)
```

### 7.3 SwiGLU Implementation

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU activation
        up = self.w_up(x)              # Linear (no activation)
        return self.w_down(gate * up)  # Gated output
```

---

## 8. Interview Questions

### Q1: Why did transformers switch from ReLU to GELU?

**Answer**:
1. **Smoothness**: GELU is smooth everywhere; ReLU has a kink at 0
2. **Probabilistic interpretation**: GELU weights inputs by "how likely they are to be positive"
3. **Better gradients**: No dead neurons, smoother gradient flow
4. **Empirical performance**: GELU consistently outperforms ReLU in transformers
5. **Compatibility with LayerNorm**: Smooth activations work better with normalization

### Q2: Explain the dying ReLU problem and solutions.

**Answer**:
**Problem**: If a ReLU neuron's input is always negative:
- Output is always 0
- Gradient is always 0
- Weights never update
- Neuron is permanently "dead"

**Solutions**:
1. **Leaky ReLU**: Small negative slope (α=0.01) keeps gradient flowing
2. **PReLU**: Learn the negative slope
3. **ELU**: Smooth exponential for negatives
4. **Careful initialization**: He initialization keeps activations positive
5. **Lower learning rate**: Prevents weights from going too negative

### Q3: What is SwiGLU and why is it used in modern LLMs?

**Answer**:
```
SwiGLU(x) = (xW₁) ⊗ SiLU(xW₂)
```

- **Gated architecture**: Part of input controls what passes through
- **SiLU activation**: Smooth, self-gated non-linearity
- **Benefits**:
  - Multiplicative interactions (more expressive)
  - Smoother optimization landscape
  - Better performance at scale (empirically proven)
- **Used in**: LLaMA, PaLM, likely GPT-4

### Q4: Compare GELU and SiLU mathematically.

**Answer**:
```
GELU(x) = x · Φ(x)     where Φ is normal CDF
SiLU(x) = x · σ(x)     where σ is sigmoid

At x=0: Both = 0
For large x: Both ≈ x
For large negative x: Both ≈ 0

Key difference:
- GELU uses error function (normal CDF)
- SiLU uses sigmoid

Practically:
- Very similar behavior
- SiLU slightly faster (no erf)
- Both smooth, self-gated
```

### Q5: Why is gating beneficial in neural networks?

**Answer**:
1. **Selective information flow**: Network learns what to pass through
2. **Multiplicative interactions**: More expressive than additive
3. **Gradient flow**: Gating provides alternative gradient paths
4. **Implicit attention**: Similar to attention mechanism in spirit
5. **Optimization**: Creates smoother loss landscape

Examples:
- LSTM: forget/input/output gates
- Attention: softmax as gating
- SwiGLU: learned gating in FFN

---

## 9. Summary

### Quick Reference

| Activation | Formula | Use Case |
|------------|---------|----------|
| ReLU | max(0, x) | CNNs, general |
| GELU | x·Φ(x) | Transformers (BERT/GPT-2) |
| SiLU | x·σ(x) | EfficientNet, some LLMs |
| SwiGLU | (xW₁)⊗SiLU(xW₂) | LLaMA, PaLM |
| Softmax | exp(xᵢ)/Σexp(xⱼ) | Classification output |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output, gates |

### Key Takeaways

1. **Activation = Non-linearity**: Without it, deep networks collapse to linear
2. **ReLU revolutionized deep learning**: Simple, fast, no vanishing gradient (for positives)
3. **GELU/SiLU are smoother alternatives**: Better optimization for transformers
4. **Gated activations (SwiGLU) dominate LLMs**: Best performance at scale
5. **Match activation to architecture**: Different architectures have different optimal choices

### Modern LLM Stack

```
Embedding → [N × TransformerBlock] → LayerNorm → Linear → Softmax

TransformerBlock:
  LayerNorm → Attention → Residual
  LayerNorm → SwiGLU FFN → Residual
```
