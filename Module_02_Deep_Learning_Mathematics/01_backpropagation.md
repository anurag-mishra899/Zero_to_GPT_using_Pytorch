# Module 2.1: Backpropagation - The Heart of Deep Learning

## Table of Contents
1. [The Learning Problem](#1-the-learning-problem)
2. [Forward Pass Mathematics](#2-forward-pass-mathematics)
3. [The Chain Rule](#3-the-chain-rule)
4. [Backpropagation Derivation](#4-backpropagation-derivation)
5. [Gradient Flow Through Layers](#5-gradient-flow-through-layers)
6. [Computational Graph Perspective](#6-computational-graph-perspective)
7. [Numerical Gradient Checking](#7-numerical-gradient-checking)
8. [Common Issues](#8-common-issues)
9. [Interview Deep Dives](#9-interview-deep-dives)
10. [Summary](#10-summary)

---

## 1. The Learning Problem

### 1.1 What Are We Optimizing?

Neural network training is an optimization problem:

```
minimize L(θ) = (1/N) Σ loss(f(x_i; θ), y_i)

where:
  θ = all learnable parameters (weights, biases)
  f(x; θ) = neural network function
  L = average loss over dataset
```

### 1.2 Gradient Descent

To minimize loss, we update parameters in the direction of steepest descent:

```
θ_new = θ_old - η * ∂L/∂θ

where:
  η = learning rate
  ∂L/∂θ = gradient of loss w.r.t. parameters
```

**The Problem**: How do we compute ∂L/∂θ for millions of parameters efficiently?

**The Solution**: Backpropagation - systematic application of the chain rule.

---

## 2. Forward Pass Mathematics

### 2.1 Single Neuron

```
Input: x ∈ R^n
Weights: w ∈ R^n, bias: b ∈ R

Linear: z = w·x + b = Σ(w_i * x_i) + b
Activation: a = σ(z)

Output: a (scalar)
```

### 2.2 Fully Connected Layer

```
Input: x ∈ R^n (or batch: X ∈ R^(batch × n))
Weights: W ∈ R^(m × n), bias: b ∈ R^m

Linear: Z = XW^T + b    (batch × m)
Activation: A = σ(Z)    (batch × m)

Parameter count: m*n + m
```

### 2.3 Multi-Layer Network

```
Layer 1: Z₁ = XW₁^T + b₁,  A₁ = σ(Z₁)
Layer 2: Z₂ = A₁W₂^T + b₂, A₂ = σ(Z₂)
...
Layer L: Z_L = A_{L-1}W_L^T + b_L, Ŷ = softmax(Z_L)

Loss: L = CrossEntropy(Ŷ, Y)
```

---

## 3. The Chain Rule

### 3.1 Univariate Chain Rule

For f(g(x)):
```
df/dx = df/dg * dg/dx
```

### 3.2 Multivariate Chain Rule

For f(g₁(x), g₂(x), ..., gₙ(x)):
```
∂f/∂x = Σᵢ (∂f/∂gᵢ * ∂gᵢ/∂x)
```

### 3.3 Vector Chain Rule (Jacobians)

For f: R^n → R^m and g: R^m → R^k:
```
∂(g∘f)/∂x = ∂g/∂f * ∂f/∂x

where ∂g/∂f is the k×m Jacobian of g
      ∂f/∂x is the m×n Jacobian of f
```

### 3.4 Practical Form (Vector-Jacobian Product)

We don't compute full Jacobians. Instead, we compute VJPs:

```
For loss L (scalar) and intermediate z:
∂L/∂x = (∂L/∂z)^T * (∂z/∂x)
      = upstream_gradient^T * local_jacobian
```

This is computed efficiently without forming the full Jacobian.

---

## 4. Backpropagation Derivation

### 4.1 Simple Network Example

Consider: x → [Linear] → z → [ReLU] → a → [Linear] → o → [Softmax+CE] → L

```
Forward:
  z = W₁x + b₁
  a = ReLU(z)
  o = W₂a + b₂
  L = CrossEntropyLoss(softmax(o), y)

Backward (apply chain rule from output to input):
  ∂L/∂o = softmax(o) - y_onehot     (derivative of softmax+CE)
  ∂L/∂W₂ = (∂L/∂o)^T · a            (outer product)
  ∂L/∂b₂ = ∂L/∂o
  ∂L/∂a = W₂^T · (∂L/∂o)            (propagate to previous layer)
  ∂L/∂z = ∂L/∂a ⊙ ReLU'(z)          (element-wise, ReLU'(z) = 1 if z>0 else 0)
  ∂L/∂W₁ = (∂L/∂z)^T · x
  ∂L/∂b₁ = ∂L/∂z
  ∂L/∂x = W₁^T · (∂L/∂z)            (if needed)
```

### 4.2 General Pattern

For each layer with input a_in, output a_out, and parameters θ:

```
Forward: a_out = f(a_in; θ)

Backward (given ∂L/∂a_out):
  ∂L/∂θ = ∂L/∂a_out * ∂a_out/∂θ     (parameter gradient)
  ∂L/∂a_in = ∂L/∂a_out * ∂a_out/∂a_in (gradient to pass backward)
```

### 4.3 Matrix Calculus Notation

For Y = XW + b with loss L:

```
Given: dL/dY (upstream gradient, same shape as Y)

dL/dW = X^T @ (dL/dY)    shape: (in, out)
dL/db = sum(dL/dY, dim=0) shape: (out,)
dL/dX = (dL/dY) @ W^T    shape: (batch, in)
```

---

## 5. Gradient Flow Through Layers

### 5.1 Linear Layer

```python
# Forward
Y = X @ W + b  # (batch, out) = (batch, in) @ (in, out) + (out,)

# Backward (given dY = dL/dY)
dW = X.T @ dY      # (in, out)
db = dY.sum(dim=0) # (out,)
dX = dY @ W.T      # (batch, in)
```

### 5.2 ReLU

```python
# Forward
Y = max(0, X)

# Backward
dX = dY * (X > 0)  # Gradient is 1 where X > 0, else 0
```

### 5.3 Sigmoid

```python
# Forward
Y = 1 / (1 + exp(-X))  # = σ(X)

# Backward
dX = dY * Y * (1 - Y)  # σ'(x) = σ(x)(1 - σ(x))
```

### 5.4 Tanh

```python
# Forward
Y = tanh(X)

# Backward
dX = dY * (1 - Y**2)  # tanh'(x) = 1 - tanh²(x)
```

### 5.5 Softmax (with Cross-Entropy)

```python
# Forward
P = softmax(X)  # P_i = exp(X_i) / Σ exp(X_j)
L = -Σ Y_true * log(P)  # Cross-entropy

# Backward (combined, much simpler!)
dX = P - Y_true  # Elegant result!
```

### 5.6 Layer Normalization

```python
# Forward
mean = X.mean(dim=-1, keepdim=True)
var = X.var(dim=-1, keepdim=True)
X_norm = (X - mean) / sqrt(var + eps)
Y = gamma * X_norm + beta

# Backward (simplified)
# Complex due to mean/var dependencies
# dX involves terms for: direct path, mean path, var path
```

### 5.7 Batch Normalization

Similar to LayerNorm but normalizes across batch dimension:
- Different gradient flow
- Running statistics complicate backward pass

---

## 6. Computational Graph Perspective

### 6.1 Nodes and Edges

```
Computational Graph for y = (a + b) * c:

    [a]     [b]
      \     /
       [+]      [c]
         \     /
          [*]
           |
          [y]

Each node stores:
- Output value (forward pass)
- Gradient w.r.t. output (backward pass)
- How to compute local gradients
```

### 6.2 Forward Pass

Traverse graph from inputs to outputs, computing values:

```python
# Topological order: a, b, c, (a+b), y
a.value = 2
b.value = 3
c.value = 4
add_node.value = a.value + b.value  # 5
y.value = add_node.value * c.value  # 20
```

### 6.3 Backward Pass

Traverse graph from outputs to inputs, computing gradients:

```python
# Start with dy/dy = 1
y.grad = 1

# Propagate through multiplication
add_node.grad = y.grad * c.value  # 1 * 4 = 4
c.grad = y.grad * add_node.value  # 1 * 5 = 5

# Propagate through addition
a.grad = add_node.grad * 1  # 4
b.grad = add_node.grad * 1  # 4
```

### 6.4 Multiple Paths (Fan-out)

When a node is used multiple times, gradients sum:

```
y = a * a  (a used twice)

Forward: y = a²
Backward: da/dy = 2a  (sum of two paths, each contributing a)
```

---

## 7. Numerical Gradient Checking

### 7.1 Finite Differences

Verify analytical gradients with numerical approximation:

```python
def numerical_gradient(f, x, eps=1e-5):
    """
    Compute gradient numerically using central difference.
    More accurate than forward difference.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad
```

### 7.2 Gradient Check Implementation

```python
def gradient_check(model, x, y, eps=1e-5, threshold=1e-7):
    """
    Compare analytical and numerical gradients.
    """
    # Compute analytical gradient
    loss = model.forward(x, y)
    model.backward()

    for name, param in model.named_parameters():
        # Numerical gradient
        param_flat = param.data.view(-1)
        num_grad = torch.zeros_like(param_flat)

        for i in range(len(param_flat)):
            # f(θ + ε)
            param_flat[i] += eps
            loss_plus = model.forward(x, y)

            # f(θ - ε)
            param_flat[i] -= 2 * eps
            loss_minus = model.forward(x, y)

            # Restore
            param_flat[i] += eps

            num_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # Compare
        anal_grad = param.grad.view(-1)
        diff = torch.abs(anal_grad - num_grad)
        relative_error = diff / (torch.abs(anal_grad) + torch.abs(num_grad) + 1e-8)

        if relative_error.max() > threshold:
            print(f"Gradient check FAILED for {name}")
            print(f"Max relative error: {relative_error.max()}")
        else:
            print(f"Gradient check PASSED for {name}")
```

### 7.3 Common Gradient Check Mistakes

1. **Not using double precision**: Use float64 for gradient checking
2. **eps too large or small**: 1e-5 to 1e-7 typically works
3. **Forgetting to zero gradients**: Call `zero_grad()` before backward
4. **Stochastic operations**: Disable dropout, use same random state

---

## 8. Common Issues

### 8.1 Vanishing Gradients

**Problem**: Gradients become tiny as they propagate backward.

**Causes**:
- Sigmoid/tanh saturation (gradients → 0 at extremes)
- Many layers multiplying small gradients
- Poor initialization

**Solutions**:
- Use ReLU/GELU activations
- Skip connections (ResNet)
- Proper initialization (He, Xavier)
- Batch/Layer normalization

### 8.2 Exploding Gradients

**Problem**: Gradients become huge, causing NaN/Inf.

**Causes**:
- Large weights multiplying gradients
- Unstable recurrent connections
- Poor initialization

**Solutions**:
- Gradient clipping
- Proper initialization
- Learning rate warmup
- Gradient normalization

### 8.3 Dead ReLU

**Problem**: Neurons output 0 for all inputs, gradient is always 0.

**Causes**:
- Large negative bias
- Large learning rate pushing weights negative
- Poor initialization

**Solutions**:
- Leaky ReLU, ELU, GELU
- Careful initialization
- Lower learning rate

### 8.4 Gradient Checking Fails

**Debug steps**:
1. Check for bugs in forward pass
2. Verify backward pass logic
3. Use double precision
4. Check for non-differentiable operations
5. Disable stochastic operations (dropout)

---

## 9. Interview Deep Dives

### Q1: Derive backprop for a single linear layer

**Answer**:
```
Forward: Y = XW + b
  X: (batch, in)
  W: (in, out)
  b: (out,)
  Y: (batch, out)

Backward (given dL/dY):
  dL/dW = X^T @ dL/dY
        = (in, batch) @ (batch, out) = (in, out) ✓

  dL/db = sum(dL/dY, dim=0)
        = (out,) ✓

  dL/dX = dL/dY @ W^T
        = (batch, out) @ (out, in) = (batch, in) ✓
```

### Q2: Why is softmax + cross-entropy gradient so simple?

**Answer**:
The combined gradient `P - Y_true` comes from:

```
L = -Σ y_i * log(p_i)  where p_i = exp(z_i) / Σexp(z_j)

∂L/∂z_i = Σ_j (∂L/∂p_j * ∂p_j/∂z_i)

After math (using quotient rule for softmax):
∂L/∂z_i = p_i - y_i

This elegant form is why we use this combination in practice.
```

### Q3: Explain vanishing gradients mathematically

**Answer**:
For sigmoid σ(x), the derivative is σ(x)(1-σ(x)), maximum 0.25 at x=0.

In an L-layer network:
```
∂L/∂W₁ = ∂L/∂a_L * ∂a_L/∂a_{L-1} * ... * ∂a_2/∂a_1 * ∂a_1/∂W₁
```

Each ∂aᵢ/∂aᵢ₋₁ involves sigmoid derivative (≤0.25).
After L layers: gradient ≤ 0.25^L

For L=10: 0.25^10 ≈ 10^-6 (vanished!)

### Q4: How do skip connections help?

**Answer**:
In ResNet: y = F(x) + x

Gradient: ∂L/∂x = ∂L/∂y * ∂y/∂x = ∂L/∂y * (∂F/∂x + 1)

The "+1" term provides a direct gradient path that doesn't diminish.
Even if ∂F/∂x → 0, gradient still flows through identity connection.

### Q5: Implement backward pass for batch norm

**Answer** (simplified, ignoring running stats):
```python
def batchnorm_backward(dout, cache):
    x, x_norm, mean, var, gamma, eps = cache
    N, D = x.shape

    # Gradient w.r.t. gamma and beta
    dgamma = (dout * x_norm).sum(dim=0)
    dbeta = dout.sum(dim=0)

    # Gradient w.r.t. x_norm
    dx_norm = dout * gamma

    # Gradient w.r.t. variance
    dvar = (dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5)).sum(dim=0)

    # Gradient w.r.t. mean
    dmean = (dx_norm * -1 / sqrt(var + eps)).sum(dim=0)
    dmean += dvar * (-2 * (x - mean)).mean(dim=0)

    # Gradient w.r.t. x
    dx = dx_norm / sqrt(var + eps)
    dx += dvar * 2 * (x - mean) / N
    dx += dmean / N

    return dx, dgamma, dbeta
```

---

## 10. Summary

### Key Concepts

1. **Backpropagation** = Chain rule applied systematically through computational graph
2. **Forward pass**: Compute outputs and cache intermediates
3. **Backward pass**: Compute gradients from output to input using cached values
4. **VJP**: Vector-Jacobian product - efficient gradient computation without full Jacobians

### Gradient Formulas for Common Layers

| Layer | Forward | Backward (dL/dX given dL/dY) |
|-------|---------|------------------------------|
| Linear: Y=XW+b | Y=XW+b | dX=dY@W^T, dW=X^T@dY, db=sum(dY) |
| ReLU | Y=max(0,X) | dX=dY*(X>0) |
| Sigmoid | Y=σ(X) | dX=dY*Y*(1-Y) |
| Softmax+CE | L=-Σy*log(p) | dX=P-Y |
| LayerNorm | Y=γ*norm(X)+β | Complex (mean/var paths) |

### Best Practices

1. **Gradient checking**: Always verify new backward implementations
2. **Use double precision** for gradient checking
3. **Watch for vanishing/exploding**: Monitor gradient norms during training
4. **Use appropriate activations**: ReLU/GELU for deep networks
5. **Skip connections**: Essential for very deep networks
