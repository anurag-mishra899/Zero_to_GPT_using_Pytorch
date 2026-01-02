# Module 1.1: Tensors and Tensor Operations in PyTorch

## Table of Contents
1. [What is a Tensor?](#1-what-is-a-tensor)
2. [Tensor Creation Methods](#2-tensor-creation-methods)
3. [Data Types (dtypes)](#3-data-types-dtypes)
4. [Memory Layout: Strides and Contiguity](#4-memory-layout-strides-and-contiguity)
5. [Tensor Operations](#5-tensor-operations)
6. [Broadcasting](#6-broadcasting)
7. [Views vs Copies](#7-views-vs-copies)
8. [GPU Operations](#8-gpu-operations)
9. [Interview Questions](#9-interview-questions)
10. [Summary](#10-summary)

---

## 1. What is a Tensor?

### 1.1 The Core Abstraction

A **tensor** is the fundamental data structure in PyTorch and modern deep learning. It is a generalization of scalars, vectors, and matrices to arbitrary dimensions:

```
Scalar (0D tensor):     42                          Shape: ()
Vector (1D tensor):     [1, 2, 3]                   Shape: (3,)
Matrix (2D tensor):     [[1, 2], [3, 4]]            Shape: (2, 2)
3D tensor:              [[[1,2],[3,4]],[[5,6],[7,8]]] Shape: (2, 2, 2)
```

### 1.2 Why Tensors Instead of NumPy Arrays?

PyTorch tensors provide three critical capabilities beyond NumPy:

| Feature | NumPy | PyTorch Tensor |
|---------|-------|----------------|
| GPU Acceleration | No | Yes (CUDA/ROCm) |
| Automatic Differentiation | No | Yes (autograd) |
| Optimized for DL | Partial | Yes |
| Distributed Computing | No | Yes |

### 1.3 Tensor Properties

Every tensor has these fundamental properties:

```python
t = torch.randn(2, 3, 4)

t.shape      # torch.Size([2, 3, 4]) - dimensions
t.dtype      # torch.float32 - data type
t.device     # cpu or cuda:0 - where tensor lives
t.stride()   # (12, 4, 1) - memory layout
t.is_contiguous()  # True/False - memory arrangement
t.requires_grad    # True/False - track gradients?
```

### 1.4 Tensor Terminology in Deep Learning

In neural networks, tensor dimensions often have semantic meaning:

```
Image batch:  (batch_size, channels, height, width)     - NCHW format
             (batch_size, height, width, channels)     - NHWC format

Sequence:     (batch_size, sequence_length, features)   - NLP common
             (sequence_length, batch_size, features)   - PyTorch RNN default

Attention:    (batch, heads, seq_len, d_k)             - Multi-head attention
```

---

## 2. Tensor Creation Methods

### 2.1 From Python Data

```python
# From list - dtype inferred
torch.tensor([1, 2, 3])           # int64
torch.tensor([1.0, 2.0, 3.0])     # float32
torch.tensor([[1, 2], [3, 4]])    # 2D tensor

# With explicit dtype
torch.tensor([1, 2, 3], dtype=torch.float32)
```

**Important**: `torch.tensor()` always copies data. For zero-copy from NumPy, use `torch.from_numpy()`.

### 2.2 Factory Functions

```python
# Zeros and Ones
torch.zeros(3, 4)           # 3x4 matrix of zeros
torch.ones(2, 3, 4)         # 3D tensor of ones
torch.zeros_like(x)         # Same shape/dtype/device as x
torch.ones_like(x)

# Identity and Diagonal
torch.eye(4)                # 4x4 identity matrix
torch.diag(torch.tensor([1,2,3]))  # Diagonal matrix

# Ranges
torch.arange(0, 10, 2)      # [0, 2, 4, 6, 8] - like Python range
torch.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1] - evenly spaced

# Empty (uninitialized - faster but contains garbage)
torch.empty(3, 4)           # Allocates memory, no initialization
```

### 2.3 Random Tensors (Critical for Initialization)

```python
torch.manual_seed(42)  # For reproducibility

# Uniform distribution [0, 1)
torch.rand(3, 3)

# Standard normal (mean=0, std=1)
torch.randn(3, 3)

# Normal with custom mean/std
torch.normal(mean=0, std=0.02, size=(3, 3))

# Random integers
torch.randint(low=0, high=10, size=(3, 3))

# Random permutation (useful for shuffling)
torch.randperm(10)  # Random permutation of 0-9
```

### 2.4 From NumPy (Zero-Copy)

```python
import numpy as np

np_array = np.array([1, 2, 3])

# Shares memory! Changes to one affect the other
t = torch.from_numpy(np_array)

# Back to NumPy (also shares memory if on CPU)
np_back = t.numpy()

# For GPU tensors, must move to CPU first
t_gpu = t.cuda()
np_from_gpu = t_gpu.cpu().numpy()
```

---

## 3. Data Types (dtypes)

### 3.1 Common dtypes in Deep Learning

| dtype | Bits | Range | Use Case |
|-------|------|-------|----------|
| `torch.float32` | 32 | ±3.4e38 | Default for training |
| `torch.float16` | 16 | ±65504 | Mixed precision training |
| `torch.bfloat16` | 16 | ±3.4e38 | Better range than fp16 |
| `torch.float64` | 64 | ±1.8e308 | High precision (rare in DL) |
| `torch.int64` | 64 | ±9.2e18 | Indices, labels |
| `torch.int32` | 32 | ±2.1e9 | Smaller indices |
| `torch.bool` | 8 | True/False | Masks |

### 3.2 bfloat16 vs float16

This is an **important interview topic**:

```
float16 (IEEE 754):
  Sign: 1 bit | Exponent: 5 bits | Mantissa: 10 bits
  - High precision, limited range (max ~65504)
  - Can overflow during training

bfloat16 (Brain Float):
  Sign: 1 bit | Exponent: 8 bits | Mantissa: 7 bits
  - Same range as float32 (max ~3.4e38)
  - Lower precision but no overflow issues
  - Preferred for training on modern hardware (TPUs, A100+)
```

### 3.3 Type Conversion

```python
t = torch.tensor([1, 2, 3])

# Convert dtype
t.float()     # to float32
t.double()    # to float64
t.half()      # to float16
t.int()       # to int32
t.long()      # to int64
t.bool()      # to bool

# Generic conversion
t.to(torch.float16)
t.type(torch.FloatTensor)
```

---

## 4. Memory Layout: Strides and Contiguity

### 4.1 How Tensors are Stored in Memory

Tensors are stored as **1D arrays in memory**. The `stride` tells us how to navigate this 1D array to access multi-dimensional elements.

```
Matrix (2x3):
[[1, 2, 3],
 [4, 5, 6]]

Memory (row-major): [1, 2, 3, 4, 5, 6]

Stride: (3, 1)
- To move to next row: skip 3 elements
- To move to next column: skip 1 element

Accessing element [i, j]:
memory_offset = i * stride[0] + j * stride[1]
             = i * 3 + j * 1
```

### 4.2 Stride Calculation Formula

For a contiguous (row-major) tensor with shape `(d0, d1, d2, ..., dn)`:

```
stride[i] = d[i+1] * d[i+2] * ... * d[n]
stride[n] = 1  (last dimension)

Example: shape (2, 3, 4)
stride[2] = 1
stride[1] = 4 * 1 = 4
stride[0] = 3 * 4 = 12
Result: stride = (12, 4, 1)
```

### 4.3 Transpose and Strides

**Key insight**: Transpose doesn't move data, it just swaps strides!

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # Shape (2, 3), Stride (3, 1)

t_T = t.T  # Shape (3, 2), Stride (1, 3)

# Same memory! Just different interpretation
t.storage().data_ptr() == t_T.storage().data_ptr()  # True
```

This is why transpose is O(1) - no data copying!

### 4.4 Contiguity

A tensor is **contiguous** if its elements are stored in memory in the order you'd expect from iterating through dimensions left-to-right (row-major order).

```python
t = torch.randn(2, 3)
t.is_contiguous()  # True

t_T = t.T
t_T.is_contiguous()  # False! Strides are "backwards"

# Make contiguous (creates a copy)
t_T_contig = t_T.contiguous()
t_T_contig.is_contiguous()  # True
```

### 4.5 When Contiguity Matters

1. **`.view()` requires contiguous tensors**
2. **Some CUDA kernels are optimized for contiguous memory**
3. **Non-contiguous tensors have cache-unfriendly access patterns**

```python
t = torch.randn(2, 3)
t_T = t.T  # Non-contiguous

# This fails:
try:
    t_T.view(6)
except RuntimeError:
    print("view() needs contiguous tensor!")

# Solutions:
t_T.contiguous().view(6)  # Make contiguous first
t_T.reshape(6)            # reshape handles non-contiguous
```

---

## 5. Tensor Operations

### 5.1 Element-wise Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Arithmetic
a + b   # torch.add(a, b)
a - b   # torch.sub(a, b)
a * b   # torch.mul(a, b) - ELEMENT-WISE, not matmul!
a / b   # torch.div(a, b)
a ** 2  # torch.pow(a, 2)

# Math functions
torch.sqrt(a)
torch.exp(a)
torch.log(a)
torch.sin(a)
torch.abs(a)
torch.clamp(a, min=0, max=2)  # Clip values
```

### 5.2 Matrix Multiplication (Critical!)

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# Matrix multiplication - 4 equivalent ways:
C = A @ B              # Python 3.5+ operator (recommended)
C = torch.mm(A, B)     # Only 2D tensors
C = torch.matmul(A, B) # Supports broadcasting & batched
C = A.mm(B)            # Method form

# Result shape: (2, 3) @ (3, 4) = (2, 4)
```

### 5.3 Batched Matrix Multiplication

This is **extremely common** in transformers:

```python
# Attention scores: Q @ K^T for each batch and head
batch, heads, seq_len, d_k = 4, 8, 512, 64

Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)

# Batched matmul across batch and head dimensions
scores = torch.matmul(Q, K.transpose(-2, -1))
# Shape: (4, 8, 512, 512)

# Alternative for 3D tensors only:
torch.bmm(Q.view(-1, seq_len, d_k),
          K.view(-1, seq_len, d_k).transpose(-2, -1))
```

### 5.4 Reduction Operations

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Full tensor reductions
t.sum()       # 21.0 (scalar)
t.mean()      # 3.5
t.std()       # Standard deviation
t.var()       # Variance
t.min()       # 1.0
t.max()       # 6.0
t.prod()      # Product of all elements

# Along specific dimension
t.sum(dim=0)   # [5, 7, 9] - sum columns (along rows)
t.sum(dim=1)   # [6, 15] - sum rows (along columns)
t.mean(dim=0)  # [2.5, 3.5, 4.5]

# Keep dimensions for broadcasting
t.sum(dim=1, keepdim=True)  # Shape (2, 1) instead of (2,)

# Max/Min with indices (for argmax)
values, indices = t.max(dim=1)  # values=[3, 6], indices=[2, 2]
```

### 5.5 Comparison Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])

a > b   # [False, False, True]
a >= b  # [False, True, True]
a == b  # [False, True, False]
a != b  # [True, False, True]

# Element-wise max/min
torch.maximum(a, b)  # [2, 2, 3]
torch.minimum(a, b)  # [1, 2, 2]

# Check conditions
torch.all(a > 0)     # True if all elements satisfy
torch.any(a > 2)     # True if any element satisfies
```

### 5.6 Shape Manipulation

```python
t = torch.randn(2, 3, 4)

# Reshape (may copy if non-contiguous)
t.reshape(6, 4)
t.reshape(-1, 4)     # -1 infers dimension

# View (never copies, requires contiguous)
t.view(6, 4)
t.view(-1)           # Flatten

# Transpose
t.T                  # Only for 2D
t.transpose(0, 1)    # Swap dimensions 0 and 1
t.permute(2, 0, 1)   # Arbitrary dimension order

# Add/remove dimensions
t.unsqueeze(0)       # Add dim at position 0: (1, 2, 3, 4)
t.unsqueeze(-1)      # Add at end: (2, 3, 4, 1)
t.squeeze()          # Remove all size-1 dimensions
t.squeeze(0)         # Remove size-1 dim at position 0

# Expand (broadcast without copying)
t = torch.randn(1, 3)
t.expand(4, 3)       # Shape (4, 3), shares memory!

# Repeat (copies data)
t.repeat(4, 2)       # Shape (4, 6), new memory
```

### 5.7 Indexing and Slicing

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Basic indexing (same as NumPy)
t[0]          # [1, 2, 3] - first row
t[0, 1]       # 2 - element at (0, 1)
t[:, 1]       # [2, 5, 8] - second column
t[1:, :2]     # [[4, 5], [7, 8]] - slicing

# Advanced indexing
indices = torch.tensor([0, 2])
t[indices]    # [[1, 2, 3], [7, 8, 9]] - select rows 0 and 2

# Boolean indexing
mask = t > 5
t[mask]       # [6, 7, 8, 9] - 1D tensor of matching elements

# Gather (useful in RL, NLP)
# Select elements along a dimension using indices
src = torch.tensor([[1, 2], [3, 4]])
idx = torch.tensor([[0, 0], [1, 0]])
torch.gather(src, dim=1, index=idx)  # [[1, 1], [4, 3]]
```

---

## 6. Broadcasting

### 6.1 What is Broadcasting?

Broadcasting allows operations between tensors of different shapes by **virtually expanding** dimensions to make them compatible.

### 6.2 Broadcasting Rules

```
1. Align shapes from the RIGHT
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1
3. Missing dimensions are treated as 1

Example:
A:     (4, 3)
B:        (3,)  →  treated as (1, 3)
─────────────
Result: (4, 3)  →  B broadcasts to (4, 3)
```

### 6.3 Broadcasting Examples

```python
# Scalar broadcast
t = torch.tensor([1, 2, 3])
t + 10  # [11, 12, 13] - 10 broadcasts to [10, 10, 10]

# Vector to matrix
matrix = torch.randn(4, 3)  # (4, 3)
vector = torch.randn(3)     # (3,) → (1, 3) → (4, 3)
matrix + vector  # Works!

# Column vector to matrix
col = torch.randn(4, 1)     # (4, 1) → (4, 3)
matrix + col  # Works!

# Both broadcast
A = torch.randn(4, 1)  # (4, 1) → (4, 3)
B = torch.randn(1, 3)  # (1, 3) → (4, 3)
A + B  # Shape (4, 3)
```

### 6.4 Common Broadcasting Patterns

#### Pattern 1: Batch operations
```python
# Apply same transformation to all samples
batch = torch.randn(32, 10)   # 32 samples, 10 features
weights = torch.randn(10)     # Per-feature weights
batch * weights  # (32, 10) * (10,) → (32, 10)
```

#### Pattern 2: Outer product
```python
a = torch.tensor([1, 2, 3])     # (3,)
b = torch.tensor([4, 5])        # (2,)

# Reshape for broadcasting
outer = a[:, None] * b[None, :]  # (3, 1) * (1, 2) → (3, 2)
# [[4, 5], [8, 10], [12, 15]]
```

#### Pattern 3: Normalization
```python
# Normalize each feature (column) to zero mean
data = torch.randn(100, 5)  # 100 samples, 5 features
mean = data.mean(dim=0, keepdim=True)  # (1, 5)
centered = data - mean  # (100, 5) - (1, 5) → (100, 5)
```

### 6.5 Broadcasting Failures

```python
A = torch.randn(4, 3)
B = torch.randn(4,)  # (4,) → (1, 4) - doesn't match!

# This fails:
# A + B  # RuntimeError: size mismatch

# Fix by reshaping B:
A + B[:, None]  # (4, 3) + (4, 1) → works!
```

---

## 7. Views vs Copies

### 7.1 The Distinction

| Operation | Memory | Gradient Flow | When |
|-----------|--------|---------------|------|
| View | Shared | Yes | Fast, alias |
| Copy | New | Yes | Independent data |
| Detach | Shared | **No** | Stop gradients |

### 7.2 View Operations (Share Memory)

```python
t = torch.tensor([[1, 2, 3], [4, 5, 6]])

# These create VIEWS (share memory):
v1 = t.view(6)          # Reshape
v2 = t[0]               # Indexing (basic)
v3 = t.T                # Transpose
v4 = t.transpose(0, 1)  # Explicit transpose
v5 = t.unsqueeze(0)     # Add dimension
v6 = t.squeeze()        # Remove dimension
v7 = t.expand(2, 2, 3)  # Broadcast expansion

# Modifying view affects original!
v1[0] = 100
print(t[0, 0])  # 100
```

### 7.3 Copy Operations (New Memory)

```python
t = torch.randn(2, 3)

# These create COPIES (new memory):
c1 = t.clone()                    # Explicit copy
c2 = t.contiguous()               # Copy if non-contiguous
c3 = t.reshape(6)                 # Copy if non-contiguous
c4 = t[torch.tensor([0, 1])]      # Advanced indexing
c5 = t + 0                        # Any computation

# Modifying copy doesn't affect original
c1[0, 0] = 100
print(t[0, 0])  # Unchanged
```

### 7.4 Detach (Stop Gradients)

```python
# During training
x = torch.randn(3, requires_grad=True)
y = x * 2

# Detach: share memory, but no gradient flow
y_detach = y.detach()  # View, no grad
y_clone = y.clone()    # Copy, grad flows

# Common pattern: detach AND copy
y_safe = y.detach().clone()  # Independent, no grad
```

### 7.5 How to Check

```python
t = torch.randn(2, 3)
v = t.view(6)

# Check if same storage
same_memory = t.storage().data_ptr() == v.storage().data_ptr()

# Check memory address
print(t.data_ptr())
print(v.data_ptr())  # Same if view
```

---

## 8. GPU Operations

### 8.1 Device Management

```python
# Check CUDA availability
torch.cuda.is_available()       # True/False
torch.cuda.device_count()       # Number of GPUs
torch.cuda.current_device()     # Current GPU index
torch.cuda.get_device_name(0)   # GPU name

# Define device (best practice)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')  # Specific GPU
device = torch.device('mps')     # Apple Silicon
```

### 8.2 Moving Tensors

```python
t = torch.randn(3, 3)

# To GPU
t_gpu = t.to(device)
t_gpu = t.cuda()          # Shorthand for CUDA
t_gpu = t.cuda(0)         # Specific GPU

# To CPU
t_cpu = t_gpu.cpu()
t_cpu = t_gpu.to('cpu')

# Create directly on device
t_direct = torch.randn(3, 3, device=device)
t_like = torch.zeros_like(t_gpu)  # Same device as t_gpu
```

### 8.3 Important Rules

```python
# 1. All tensors in an operation must be on same device
t_cpu = torch.randn(3)
t_gpu = torch.randn(3, device='cuda')
# t_cpu + t_gpu  # RuntimeError!

# 2. Models must be moved too
model = MyModel()
model.to(device)

# 3. Non-blocking transfer (overlap with computation)
t = t.to(device, non_blocking=True)

# 4. Pin memory for faster CPU→GPU transfer
data_loader = DataLoader(dataset, pin_memory=True)
```

### 8.4 Memory Management

```python
# Check memory usage
torch.cuda.memory_allocated()     # Currently allocated
torch.cuda.memory_reserved()      # Reserved by caching allocator
torch.cuda.max_memory_allocated() # Peak usage

# Free cache
torch.cuda.empty_cache()

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```

---

## 9. Interview Questions

### Q1: What are strides and why do they matter?

**Answer**: Strides define how many memory positions to skip when moving along each tensor dimension. They enable efficient operations like transpose (just swap strides, O(1)) without copying data. Understanding strides is crucial for:
- Knowing when `.contiguous()` is needed
- Optimizing custom CUDA kernels
- Debugging performance issues

### Q2: Difference between `.view()` and `.reshape()`?

**Answer**:
- `.view()`: Requires contiguous tensor, always returns a view
- `.reshape()`: Works on any tensor, returns view if possible, copy if not

Use `.view()` when you know the tensor is contiguous and want to guarantee no copy. Use `.reshape()` when you don't care or aren't sure about contiguity.

### Q3: When would you use bfloat16 vs float16?

**Answer**:
- **float16**: Higher precision (10 mantissa bits), but limited range (max ~65504). Can overflow during training.
- **bfloat16**: Lower precision (7 mantissa bits), but same range as float32. Preferred for training because it doesn't overflow.

Modern GPUs (A100+) and TPUs are optimized for bfloat16 training.

### Q4: Explain broadcasting with an example.

**Answer**: Broadcasting allows operations between tensors of different shapes by virtually expanding dimensions. Rules:
1. Align shapes from right
2. Dimensions match if equal or one is 1

Example: Centering data
```python
data = torch.randn(100, 5)  # (100, 5)
mean = data.mean(dim=0)     # (5,) → (1, 5)
centered = data - mean      # (100, 5) - (1, 5) → (100, 5)
```

### Q5: How do you implement softmax from scratch?

**Answer**:
```python
def softmax(x, dim=-1):
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

The subtraction of max prevents overflow when computing `exp()`.

### Q6: What's the difference between `.clone()` and `.detach()`?

**Answer**:
- `.clone()`: Creates copy with new memory, gradients still flow
- `.detach()`: Returns view (shared memory), but detached from computation graph (no gradients)
- `.detach().clone()`: Copy that is also detached (common pattern for logging/visualization)

---

## 10. Summary

### Key Concepts

1. **Tensors** are n-dimensional arrays with GPU support and autograd
2. **Strides** determine memory layout; transpose swaps strides without copying
3. **Contiguity** matters for `.view()` and performance
4. **Broadcasting** enables operations between different shapes
5. **Views share memory**, copies don't
6. **Always use `device`** consistently in your code

### Quick Reference

| Operation | Creates View? | Requires Contiguous? |
|-----------|--------------|---------------------|
| `view()` | Yes | Yes |
| `reshape()` | Maybe | No |
| `transpose()` | Yes | No |
| `clone()` | No (copy) | No |
| `contiguous()` | No (copy) | No |
| `squeeze/unsqueeze` | Yes | No |

### Best Practices

1. Define `device` once at the start
2. Use `torch.zeros_like()` / `torch.ones_like()` to preserve device
3. Use `.reshape()` unless you specifically need `.view()`
4. Use `bfloat16` for training if hardware supports it
5. Always check shapes when debugging - most errors are shape mismatches
