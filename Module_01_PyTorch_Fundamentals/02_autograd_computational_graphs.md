# Module 1.2: Autograd and Computational Graphs

## Table of Contents
1. [What is Automatic Differentiation?](#1-what-is-automatic-differentiation)
2. [Computational Graphs](#2-computational-graphs)
3. [PyTorch Autograd Basics](#3-pytorch-autograd-basics)
4. [Gradient Computation In-Depth](#4-gradient-computation-in-depth)
5. [Controlling Gradient Flow](#5-controlling-gradient-flow)
6. [Common Pitfalls and Solutions](#6-common-pitfalls-and-solutions)
7. [Advanced Autograd Features](#7-advanced-autograd-features)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. What is Automatic Differentiation?

### 1.1 The Problem

Training neural networks requires computing gradients of a loss function with respect to millions (or billions) of parameters. Doing this manually is:
- Error-prone
- Time-consuming
- Impractical for complex architectures

### 1.2 Three Approaches to Differentiation

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Symbolic** | Derive formulas analytically | Exact | Expression explosion |
| **Numerical** | Finite differences: (f(x+h) - f(x))/h | Simple | Slow, numerical errors |
| **Automatic** | Chain rule on computational graph | Fast, exact | Memory overhead |

### 1.3 Automatic Differentiation (AD)

AD decomposes complex functions into elementary operations, then applies the chain rule systematically.

Two modes:
- **Forward mode**: Compute derivatives alongside values (good for few inputs, many outputs)
- **Reverse mode (Backprop)**: Compute derivatives backward from output (good for many inputs, few outputs)

Neural networks have millions of inputs (parameters) and one output (loss), so **reverse mode** is used.

### 1.4 Why This Matters for Interviews

Understanding autograd is crucial because:
1. You need to debug gradient issues (vanishing, exploding, NaN)
2. Custom layers require understanding gradient flow
3. Advanced techniques (gradient clipping, accumulation) need this knowledge
4. Memory optimization depends on understanding the computational graph

---

## 2. Computational Graphs

### 2.1 What is a Computational Graph?

A computational graph represents a mathematical expression as a **directed acyclic graph (DAG)**:
- **Nodes**: Operations (add, multiply, relu, etc.) or inputs
- **Edges**: Data flow (tensors)

### 2.2 Example: Simple Expression

For `y = (a + b) * c`:

```
    [a]     [b]
      \     /
       [add]    [c]
         \      /
          [mul]
            |
           [y]
```

### 2.3 Forward vs Backward Pass

```
FORWARD PASS (compute output):
───────────────────────────────
a=2, b=3, c=4

     [2]     [3]
       \     /
        [5]      [4]    (2+3=5)
         \      /
          [20]          (5*4=20)
            |
           [y=20]

BACKWARD PASS (compute gradients):
───────────────────────────────
dy/dy = 1 (start with 1)

dy/d(a+b) = dy/dy * d(mul)/d(a+b) = 1 * c = 4
dy/dc = dy/dy * d(mul)/dc = 1 * (a+b) = 5

dy/da = dy/d(a+b) * d(add)/da = 4 * 1 = 4
dy/db = dy/d(a+b) * d(add)/db = 4 * 1 = 4

Result: da=4, db=4, dc=5
```

### 2.4 PyTorch's Dynamic Computational Graph

PyTorch uses **dynamic** (define-by-run) graphs:
- Graph is built on-the-fly during forward pass
- New graph each iteration
- Enables dynamic control flow (if/else, loops based on data)

Compare to TensorFlow 1.x (static graphs):
- Graph defined once, then executed
- Faster for fixed architectures
- No dynamic control flow

```python
# PyTorch: Dynamic graph
for i in range(n):
    if x[i] > 0:
        y = model_a(x[i])
    else:
        y = model_b(x[i])
    # Works! Graph built based on actual runtime condition
```

### 2.5 Graph Nodes in PyTorch

Each tensor that requires gradients has a `grad_fn` attribute pointing to the function that created it:

```python
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

c = a + b  # c.grad_fn = <AddBackward0>
d = c * 2  # d.grad_fn = <MulBackward0>

# Trace back the graph
print(d.grad_fn)                    # <MulBackward0>
print(d.grad_fn.next_functions)     # ((<AddBackward0>, 0),)
print(d.grad_fn.next_functions[0][0].next_functions)  # AccumulateGrad nodes
```

---

## 3. PyTorch Autograd Basics

### 3.1 Enabling Gradient Tracking

```python
# Method 1: At creation
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Method 2: After creation
x = torch.tensor([1.0, 2.0, 3.0])
x.requires_grad_(True)  # In-place modification

# Method 3: Using requires_grad in operations
x = torch.tensor([1.0, 2.0, 3.0])
x = x.requires_grad_()
```

### 3.2 The .backward() Method

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# Compute gradients
y.backward()  # dy/dx = 2x + 3 = 2*2 + 3 = 7

print(x.grad)  # tensor([7.])
```

### 3.3 Gradient Accumulation

**Critical**: Gradients are **accumulated** by default!

```python
x = torch.tensor([2.0], requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(x.grad)  # 4.0

# Second backward - gradients ACCUMULATE!
y2 = x ** 3
y2.backward()
print(x.grad)  # 4.0 + 12.0 = 16.0

# Must zero gradients between batches!
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(x.grad)  # 4.0 (fresh gradient)
```

### 3.4 Non-Scalar Outputs

`.backward()` expects a scalar. For non-scalar outputs, you must provide a `gradient` argument:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # [1, 4, 9] - not scalar!

# This fails:
# y.backward()  # RuntimeError: grad can be implicitly created only for scalar outputs

# Solution 1: Pass gradient vector (Jacobian-vector product)
y.backward(torch.ones_like(y))  # Sum of all gradients
print(x.grad)  # [2, 4, 6]

# Solution 2: Sum to scalar
x.grad.zero_()
y = x ** 2
y.sum().backward()
print(x.grad)  # [2, 4, 6]
```

### 3.5 Leaf Tensors and .grad

Only **leaf tensors** have `.grad` populated:

```python
a = torch.tensor([2.0], requires_grad=True)  # Leaf
b = a * 2  # Not a leaf (has grad_fn)

c = b.sum()
c.backward()

print(a.grad)   # tensor([2.])
print(b.grad)   # None! (not a leaf)
print(a.is_leaf)  # True
print(b.is_leaf)  # False
```

To retain gradients for non-leaf tensors:
```python
b.retain_grad()
c.backward()
print(b.grad)  # Now available
```

---

## 4. Gradient Computation In-Depth

### 4.1 The Chain Rule

For composite function f(g(x)):
```
df/dx = df/dg * dg/dx
```

Neural network with layers:
```
Loss = L(f3(f2(f1(x))))

dL/dx = dL/df3 * df3/df2 * df2/df1 * df1/dx
```

### 4.2 Vector-Jacobian Products (VJP)

For function f: R^n → R^m, the Jacobian is:
```
J[i,j] = df_i / dx_j   (m × n matrix)
```

Backpropagation computes **vector-Jacobian products** (VJPs):
```
v^T @ J   where v is the upstream gradient
```

This is efficient because:
- We never explicitly form the full Jacobian
- VJP can be computed in O(n) instead of O(n²)

### 4.3 Example: Linear Layer Gradients

```python
# y = Wx + b
# W: (out_features, in_features)
# x: (batch, in_features)
# b: (out_features,)

# Forward: y = x @ W.T + b
# Shape: (batch, in_features) @ (in_features, out_features) + (out_features,)
#      = (batch, out_features)

# Backward (given upstream gradient dy):
# dL/dW = dy.T @ x
# dL/dx = dy @ W
# dL/db = dy.sum(dim=0)
```

### 4.4 Gradient Flow Through Common Operations

| Operation | Forward | Backward (dy is upstream grad) |
|-----------|---------|-------------------------------|
| y = x + c | y = x + c | dx = dy |
| y = x * c | y = x * c | dx = dy * c |
| y = x @ W | y = x @ W | dx = dy @ W.T, dW = x.T @ dy |
| y = relu(x) | max(0, x) | dx = dy * (x > 0) |
| y = sigmoid(x) | σ(x) | dx = dy * σ(x) * (1 - σ(x)) |
| y = softmax(x) | exp(x)/sum(exp(x)) | Complex (see below) |
| y = x.sum() | sum | dx = ones_like(x) * dy |
| y = x.mean() | mean | dx = ones_like(x) * dy / n |

### 4.5 Softmax + Cross-Entropy Gradient

This is asked in interviews frequently!

```python
# Softmax: p_i = exp(x_i) / sum(exp(x_j))
# Cross-entropy: L = -sum(y_true * log(p))

# For one-hot y_true with correct class k:
# L = -log(p_k)

# Combined gradient (elegant result!):
# dL/dx_i = p_i - y_true_i

# For one-hot:
# dL/dx_i = p_i (for i ≠ k)
# dL/dx_k = p_k - 1
```

This is why PyTorch has `CrossEntropyLoss` that takes logits directly - it's numerically stable and efficient.

---

## 5. Controlling Gradient Flow

### 5.1 torch.no_grad()

Disable gradient computation for efficiency:

```python
# During inference
with torch.no_grad():
    predictions = model(x)

# Faster and uses less memory
# No computational graph is built
```

### 5.2 torch.inference_mode()

Even more efficient than no_grad (PyTorch 1.9+):

```python
with torch.inference_mode():
    predictions = model(x)

# More aggressive optimizations than no_grad
# Cannot modify tensors afterward
```

### 5.3 .detach()

Create a tensor that shares data but doesn't track gradients:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

y_detached = y.detach()  # Shares memory, no grad tracking
y_detached.requires_grad  # False

# Useful for:
# 1. Stopping gradient flow to part of network
# 2. Using tensor for logging without affecting gradients
# 3. Creating target values for losses
```

### 5.4 Freezing Parameters

```python
# Method 1: requires_grad = False
for param in model.encoder.parameters():
    param.requires_grad = False

# Method 2: torch.no_grad() context
with torch.no_grad():
    frozen_output = frozen_model(x)

# Method 3: detach intermediates
features = encoder(x).detach()  # Stop gradients here
output = decoder(features)
```

### 5.5 Gradient Clipping

Prevent exploding gradients:

```python
# Method 1: Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Method 2: Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# Typical training loop:
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # After backward!
optimizer.step()
```

### 5.6 Gradient Accumulation

For effective larger batch sizes with limited memory:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()  # Gradients accumulate

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. Common Pitfalls and Solutions

### 6.1 In-Place Operations

In-place operations can break autograd:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

# This breaks the graph!
y.add_(1)  # In-place operation
# RuntimeError: one of the variables needed for gradient computation
# has been modified by an inplace operation

# Solution: Use out-of-place operations
y = y + 1  # Creates new tensor
```

### 6.2 Forgetting to Zero Gradients

```python
# WRONG - gradients accumulate!
for batch in dataloader:
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
    # x.grad keeps growing!

# CORRECT
for batch in dataloader:
    optimizer.zero_grad()  # Clear gradients first!
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
```

### 6.3 Using Tensor for Scalar Operations

```python
# WRONG - creates unnecessary graph
total_loss = 0
for batch in dataloader:
    loss = criterion(model(batch), targets)
    total_loss += loss  # Accumulates entire graph!

# CORRECT - use .item() for scalar extraction
total_loss = 0
for batch in dataloader:
    loss = criterion(model(batch), targets)
    total_loss += loss.item()  # Just the number, no graph
```

### 6.4 Leaf Variable Modification

```python
x = torch.tensor([1.0], requires_grad=True)

# This fails:
x = x * 2  # x is no longer a leaf!
x.backward()  # Error: x has no grad

# Solution: Use separate variable
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
y.backward()
print(x.grad)  # Works!
```

### 6.5 Double Backward Issues

By default, intermediate results are freed after backward:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
# y.backward()  # Error! Graph already freed

# Solution: retain_graph=True
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
y.backward()  # Works now (but be careful of grad accumulation!)
```

---

## 7. Advanced Autograd Features

### 7.1 Custom Autograd Functions

Define custom forward and backward passes:

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
x = torch.randn(3, requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(x.grad)
```

### 7.2 Gradient Hooks

Inspect or modify gradients during backward:

```python
def print_grad(grad):
    print(f"Gradient: {grad}")
    return grad  # Return modified gradient (or None for no change)

x = torch.tensor([1.0, 2.0], requires_grad=True)
x.register_hook(print_grad)  # Called during backward

y = x.sum() * 2
y.backward()  # Prints: Gradient: tensor([2., 2.])
```

### 7.3 Higher-Order Gradients

Compute gradients of gradients:

```python
x = torch.tensor([3.0], requires_grad=True)
y = x ** 3  # y = x³

# First derivative: dy/dx = 3x²
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {grad1}")  # 27.0 = 3 * 9

# Second derivative: d²y/dx² = 6x
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d²y/dx² = {grad2}")  # 18.0 = 6 * 3
```

### 7.4 torch.autograd.grad()

More control than .backward():

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Get gradients without populating .grad
grads = torch.autograd.grad(
    outputs=y.sum(),
    inputs=x,
    create_graph=False,  # Set True for higher-order gradients
    retain_graph=False,
)
print(grads[0])  # tensor([2., 4., 6.])
```

### 7.5 Gradient Checkpointing

Trade compute for memory - recompute forward pass during backward:

```python
from torch.utils.checkpoint import checkpoint

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 1000)

    def forward(self, x):
        # Checkpoint layer2 - don't save activations
        x = self.layer1(x)
        x = checkpoint(self.layer2, x)  # Recomputed during backward
        x = self.layer3(x)
        return x
```

---

## 8. Interview Questions

### Q1: Explain the difference between `.backward()` and `torch.autograd.grad()`

**Answer**:
- `.backward()`: Populates `.grad` attribute of leaf tensors, modifies state
- `torch.autograd.grad()`: Returns gradients as tensors, no state modification

Use `grad()` when you need gradients as values (e.g., for gradient penalty in GANs) or for higher-order derivatives.

### Q2: Why does PyTorch accumulate gradients by default?

**Answer**: This design supports:
1. **Gradient accumulation**: Simulate larger batch sizes by accumulating over multiple forward passes
2. **Multiple losses**: Compute gradients from multiple loss terms without additional computation
3. **RNN training**: Accumulate gradients through time steps

The user must explicitly call `optimizer.zero_grad()` or `tensor.grad.zero_()`.

### Q3: What is the difference between `torch.no_grad()` and `.detach()`?

**Answer**:
- `torch.no_grad()`: Context manager, disables gradient computation for all operations within
- `.detach()`: Creates a tensor view that doesn't require gradients, but the original still might

```python
with torch.no_grad():
    y = x * 2  # No graph built at all

y = x.detach() * 2  # x still tracks grads, but y doesn't connect to x's graph
```

### Q4: How does gradient clipping work? When is it needed?

**Answer**: Gradient clipping limits gradient magnitudes to prevent exploding gradients.

```python
# Clip by norm: scales all gradients if total norm exceeds threshold
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

# Clip by value: clips each gradient element to [-clip, clip]
torch.nn.utils.clip_grad_value_(parameters, clip_value=1.0)
```

Needed for:
- RNNs (especially vanilla RNNs)
- Very deep networks
- Transformers (often use clip_norm=1.0)
- When training is unstable (loss spikes)

### Q5: Implement gradient penalty for WGAN-GP

**Answer**:
```python
def gradient_penalty(discriminator, real, fake):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,  # Need this for second-order gradient
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty
```

### Q6: Why might you see NaN gradients? How do you debug?

**Answer**: Common causes:
1. **Division by zero**: `1/x` where x=0
2. **Log of zero/negative**: `log(x)` where x≤0
3. **Exploding gradients**: Values overflow to inf
4. **sqrt of negative**: Due to numerical errors

Debugging:
```python
# Check for NaN/Inf in gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")

# Use anomaly detection
torch.autograd.set_detect_anomaly(True)  # Slow but helpful
```

---

## 9. Summary

### Key Concepts

1. **Computational Graph**: DAG of operations, built dynamically in PyTorch
2. **Reverse-mode AD**: Efficient for many inputs, few outputs (like neural networks)
3. **Gradient Accumulation**: Gradients add up by default - zero them between batches!
4. **Leaf Tensors**: Only leaves have `.grad` populated
5. **Control Flow**: Use `no_grad()`, `detach()`, `requires_grad` to control gradient computation

### Training Loop Pattern

```python
model.train()
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # 3. Backward pass
        loss.backward()

        # 4. (Optional) Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Update weights
        optimizer.step()
```

### Quick Reference

| Goal | Method |
|------|--------|
| Enable gradients | `requires_grad=True` or `.requires_grad_()` |
| Compute gradients | `.backward()` or `torch.autograd.grad()` |
| Zero gradients | `optimizer.zero_grad()` or `.grad.zero_()` |
| Disable gradients | `torch.no_grad()` or `torch.inference_mode()` |
| Stop gradient flow | `.detach()` |
| Clip gradients | `clip_grad_norm_()` or `clip_grad_value_()` |
| Debug NaN gradients | `torch.autograd.set_detect_anomaly(True)` |
| Higher-order gradients | `create_graph=True` in backward/grad |
