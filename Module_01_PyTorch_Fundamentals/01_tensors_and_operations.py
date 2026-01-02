"""
Module 1.1: Tensors and Tensor Operations in PyTorch
=====================================================

This module covers:
1. Tensor creation methods
2. Data types (dtypes)
3. Memory layout (strides, contiguity)
4. Tensor operations
5. Broadcasting
6. Views vs Copies
7. GPU operations

Run this file to see all examples in action.
"""

import torch
import numpy as np

print("=" * 70)
print("MODULE 1.1: TENSORS AND TENSOR OPERATIONS")
print("=" * 70)

# =============================================================================
# SECTION 1: TENSOR CREATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: TENSOR CREATION")
print("=" * 70)

# 1.1 From Python data structures
print("\n--- 1.1 From Python Data ---")
t1 = torch.tensor([1, 2, 3])  # int64 by default
t2 = torch.tensor([1.0, 2.0, 3.0])  # float32 by default
t3 = torch.tensor([[1, 2], [3, 4]])  # 2D tensor

print(f"From int list: {t1}, dtype: {t1.dtype}")
print(f"From float list: {t2}, dtype: {t2.dtype}")
print(f"2D tensor shape: {t3.shape}")

# 1.2 Specifying dtype explicitly
print("\n--- 1.2 Explicit dtype ---")
t_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
t_float16 = torch.tensor([1, 2, 3], dtype=torch.float16)
t_bfloat16 = torch.tensor([1, 2, 3], dtype=torch.bfloat16)

print(f"float32: {t_float32.dtype}, bytes/element: {t_float32.element_size()}")
print(f"float16: {t_float16.dtype}, bytes/element: {t_float16.element_size()}")
print(f"bfloat16: {t_bfloat16.dtype}, bytes/element: {t_bfloat16.element_size()}")

# 1.3 Factory functions
print("\n--- 1.3 Factory Functions ---")
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3, 4)
eye = torch.eye(4)
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)

print(f"zeros shape: {zeros.shape}")
print(f"ones shape: {ones.shape}")
print(f"eye (identity):\n{eye}")
print(f"arange: {arange}")
print(f"linspace: {linspace}")

# 1.4 Random tensors
print("\n--- 1.4 Random Tensors ---")
torch.manual_seed(42)

uniform = torch.rand(3, 3)  # Uniform [0, 1)
normal = torch.randn(3, 3)  # Standard normal
custom_normal = torch.normal(mean=0, std=0.02, size=(3, 3))
randint = torch.randint(low=0, high=10, size=(3, 3))

print(f"Uniform [0,1):\n{uniform}")
print(f"\nStandard Normal:\n{normal}")
print(f"\nCustom Normal (std=0.02):\n{custom_normal}")
print(f"\nRandom integers [0,10):\n{randint}")

# 1.5 Creating tensors like another
print("\n--- 1.5 Tensors Like Another ---")
x = torch.randn(2, 3, device='cpu', dtype=torch.float32)
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x)

print(f"Original: dtype={x.dtype}, device={x.device}, shape={x.shape}")
print(f"zeros_like: dtype={zeros_like.dtype}, device={zeros_like.device}")
print("This preserves dtype, device, and shape!")

# 1.6 From NumPy (zero-copy)
print("\n--- 1.6 From NumPy (Zero-Copy) ---")
np_array = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(np_array)

print(f"NumPy array: {np_array}")
print(f"Torch tensor: {t_from_np}")

# Modify numpy, torch changes too!
np_array[0] = 100
print(f"After modifying NumPy: tensor = {t_from_np}")
print("WARNING: from_numpy shares memory!")


# =============================================================================
# SECTION 2: MEMORY LAYOUT (STRIDES AND CONTIGUITY)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: MEMORY LAYOUT (STRIDES AND CONTIGUITY)")
print("=" * 70)

# 2.1 Understanding strides
print("\n--- 2.1 Understanding Strides ---")
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(f"Tensor:\n{t}")
print(f"Shape: {t.shape}")
print(f"Stride: {t.stride()}")
print("\nStride (3, 1) means:")
print("  - Skip 3 elements to move to next row")
print("  - Skip 1 element to move to next column")
print("\nMemory layout: [1, 2, 3, 4, 5, 6]")

# 2.2 Transpose and strides
print("\n--- 2.2 Transpose Changes Strides, Not Data ---")
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
t_T = t.T

print(f"Original: shape={t.shape}, stride={t.stride()}")
print(f"Transposed: shape={t_T.shape}, stride={t_T.stride()}")
print(f"Same storage? {t.storage().data_ptr() == t_T.storage().data_ptr()}")
print("\nKey insight: Transpose just swaps strides!")

# 2.3 Contiguity
print("\n--- 2.3 Contiguous vs Non-Contiguous ---")
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
t_T = t.T

print(f"Original is contiguous: {t.is_contiguous()}")
print(f"Transposed is contiguous: {t_T.is_contiguous()}")

# Make contiguous
t_contig = t_T.contiguous()
print(f"After .contiguous(): {t_contig.is_contiguous()}")
print(f"Original stride: {t_T.stride()} -> Contiguous stride: {t_contig.stride()}")

# 2.4 When contiguity matters
print("\n--- 2.4 When Contiguity Matters ---")
t = torch.randn(2, 3)
t_T = t.T

# view() requires contiguous
try:
    t_T.view(-1)
except RuntimeError as e:
    print(f"view() error: {str(e)[:60]}...")

# Solutions
print("\nSolutions:")
print(f"1. t_T.contiguous().view(-1): shape = {t_T.contiguous().view(-1).shape}")
print(f"2. t_T.reshape(-1): shape = {t_T.reshape(-1).shape}")

# 2.5 Stride calculation demonstration
print("\n--- 2.5 Stride Calculation ---")
t = torch.randn(2, 3, 4)
print(f"Shape: {t.shape}")
print(f"Stride: {t.stride()}")
print("\nFor contiguous tensor (2, 3, 4):")
print("  stride[2] = 1")
print("  stride[1] = 4 * 1 = 4")
print("  stride[0] = 3 * 4 = 12")
print("  Result: (12, 4, 1)")


# =============================================================================
# SECTION 3: TENSOR OPERATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: TENSOR OPERATIONS")
print("=" * 70)

# 3.1 Element-wise operations
print("\n--- 3.1 Element-wise Operations ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}  (element-wise, NOT matrix multiplication!)")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")
print(f"sqrt(a) = {torch.sqrt(a)}")
print(f"exp(a) = {torch.exp(a)}")
print(f"log(a) = {torch.log(a)}")

# 3.2 Matrix multiplication
print("\n--- 3.2 Matrix Multiplication ---")
A = torch.randn(2, 3)
B = torch.randn(3, 4)

C1 = A @ B  # Python operator (recommended)
C2 = torch.mm(A, B)  # Only 2D
C3 = torch.matmul(A, B)  # General (supports batched)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"A @ B shape: {C1.shape}")
print(f"All methods equal: {torch.allclose(C1, C2) and torch.allclose(C2, C3)}")

# 3.3 Batched matrix multiplication
print("\n--- 3.3 Batched Matrix Multiplication ---")
batch_size, heads, seq_len, d_k = 4, 8, 10, 64

Q = torch.randn(batch_size, heads, seq_len, d_k)
K = torch.randn(batch_size, heads, seq_len, d_k)

# Batched: Q @ K^T for each (batch, head)
attention_scores = torch.matmul(Q, K.transpose(-2, -1))

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Attention scores shape: {attention_scores.shape}")
print("This is exactly what happens in transformer attention!")

# 3.4 Reduction operations
print("\n--- 3.4 Reduction Operations ---")
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(f"Tensor:\n{t}\n")
print(f"Sum (all): {t.sum()}")
print(f"Sum (dim=0, along rows): {t.sum(dim=0)}")
print(f"Sum (dim=1, along cols): {t.sum(dim=1)}")
print(f"\nMean (all): {t.mean()}")
print(f"Mean (dim=0): {t.mean(dim=0)}")

# Max with indices
values, indices = t.max(dim=1)
print(f"\nMax per row: values={values}, indices={indices}")

# keepdim
print(f"\nSum dim=1, keepdim=False: shape = {t.sum(dim=1).shape}")
print(f"Sum dim=1, keepdim=True: shape = {t.sum(dim=1, keepdim=True).shape}")

# 3.5 Shape manipulation
print("\n--- 3.5 Shape Manipulation ---")
t = torch.arange(12).reshape(3, 4)
print(f"Original:\n{t}")
print(f"Shape: {t.shape}")

print(f"\nt.view(4, 3):\n{t.view(4, 3)}")
print(f"t.view(-1): {t.view(-1)}")  # Flatten
print(f"t.T:\n{t.T}")
print(f"t.unsqueeze(0) shape: {t.unsqueeze(0).shape}")
print(f"t.unsqueeze(-1) shape: {t.unsqueeze(-1).shape}")


# =============================================================================
# SECTION 4: BROADCASTING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: BROADCASTING")
print("=" * 70)

# 4.1 Basic broadcasting
print("\n--- 4.1 Basic Broadcasting ---")
t = torch.tensor([1.0, 2.0, 3.0])
result = t + 10
print(f"[1, 2, 3] + 10 = {result}")
print("10 broadcasts to [10, 10, 10]")

# 4.2 Vector to matrix
print("\n--- 4.2 Vector to Matrix Broadcasting ---")
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
vector = torch.tensor([10.0, 20.0, 30.0])

print(f"Matrix shape: {matrix.shape}")
print(f"Vector shape: {vector.shape}")
print(f"Matrix + Vector:\n{matrix + vector}")
print("\nVector (3,) → (1, 3) → broadcasts to (2, 3)")

# 4.3 Broadcasting rules demonstration
print("\n--- 4.3 Broadcasting Rules ---")
print("""
Rules:
1. Align shapes from RIGHT
2. Dimensions compatible if: equal OR one is 1
3. Missing dimensions treated as 1

Example:
A:     (4, 3)
B:        (3,)  →  (1, 3)  →  (4, 3)
Result: (4, 3)
""")

A = torch.ones(4, 3)
B = torch.tensor([1.0, 2.0, 3.0])
print(f"A (ones) shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"A + B:\n{A + B}")

# 4.4 Outer product via broadcasting
print("\n--- 4.4 Outer Product via Broadcasting ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0])

# Reshape for broadcasting
outer = a.unsqueeze(1) * b.unsqueeze(0)
print(f"a shape: {a.shape} → unsqueeze(1) → {a.unsqueeze(1).shape}")
print(f"b shape: {b.shape} → unsqueeze(0) → {b.unsqueeze(0).shape}")
print(f"Outer product (3, 1) * (1, 2) → (3, 2):\n{outer}")

# 4.5 Normalization pattern
print("\n--- 4.5 Normalization Pattern ---")
data = torch.randn(4, 3)
print(f"Data:\n{data}\n")

mean = data.mean(dim=0, keepdim=True)
std = data.std(dim=0, keepdim=True)
normalized = (data - mean) / (std + 1e-5)

print(f"Mean shape: {mean.shape}")
print(f"Broadcasting: (4, 3) - (1, 3) → (4, 3)")
print(f"\nNormalized:\n{normalized}")
print(f"\nVerify - new mean: {normalized.mean(dim=0)}")
print(f"Verify - new std: {normalized.std(dim=0)}")


# =============================================================================
# SECTION 5: VIEWS VS COPIES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: VIEWS VS COPIES")
print("=" * 70)

# 5.1 View operations
print("\n--- 5.1 View Operations (Share Memory) ---")
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
v = t.view(6)

print(f"Original:\n{t}")
print(f"View: {v}")
print(f"Same storage: {t.storage().data_ptr() == v.storage().data_ptr()}")

v[0] = 100
print(f"\nAfter v[0] = 100:")
print(f"View: {v}")
print(f"Original:\n{t}")
print("Both changed because they share memory!")

# 5.2 Copy operations
print("\n--- 5.2 Copy Operations (New Memory) ---")
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
c = t.clone()

c[0, 0] = 100
print(f"Original:\n{t}")
print(f"Clone:\n{c}")
print(f"Same storage: {t.storage().data_ptr() == c.storage().data_ptr()}")
print("Clone has new memory - original unchanged!")

# 5.3 Summary table
print("\n--- 5.3 View vs Copy Summary ---")
print("""
Operation         | Memory    | Gradient | Use Case
------------------|-----------|----------|----------
view()            | Shared    | Yes      | Reshape contiguous
reshape()         | Maybe     | Yes      | General reshape
clone()           | New       | Yes      | Independent copy
detach()          | Shared    | NO       | Stop gradients
detach().clone()  | New       | NO       | Safe copy for logging
transpose()       | Shared    | Yes      | Change dimension order
contiguous()      | New*      | Yes      | Fix memory layout
                  | (*if non-contiguous)
""")


# =============================================================================
# SECTION 6: GPU OPERATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: GPU/ACCELERATOR OPERATIONS")
print("=" * 70)

# 6.1 Device management
print("\n--- 6.1 Device Management ---")

# Check MPS (Apple Silicon)
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f"MPS (Apple Silicon) available: {mps_available}")

# Check CUDA (NVIDIA)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Best practice - device selection priority: MPS > CUDA > CPU
def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = get_device()
print(f"\nUsing device: {device}")

# 6.2 Moving tensors
print("\n--- 6.2 Moving Tensors ---")
t_cpu = torch.randn(3, 3)
print(f"Original device: {t_cpu.device}")

t_device = t_cpu.to(device)
print(f"After .to(device): {t_device.device}")

t_direct = torch.randn(3, 3, device=device)
print(f"Created directly on device: {t_direct.device}")

# 6.3 Common patterns
print("\n--- 6.3 Common GPU Patterns ---")
print("""
# In training loop:
for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    output = model(batch_x)
    loss = criterion(output, batch_y)
    ...

# Model must also be on device:
model = MyModel().to(device)

# Non-blocking transfer (overlaps with computation):
batch_x = batch_x.to(device, non_blocking=True)

# Pin memory for faster CPU→GPU:
DataLoader(dataset, pin_memory=True)
""")


# =============================================================================
# SECTION 7: INTERVIEW QUESTIONS - IMPLEMENTATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: INTERVIEW IMPLEMENTATIONS")
print("=" * 70)

# Q1: Implement softmax from scratch
print("\n--- Q1: Softmax Implementation ---")

def softmax(x, dim=-1):
    """
    Numerically stable softmax implementation.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    For numerical stability, we subtract max before exp:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

logits = torch.tensor([[1.0, 2.0, 3.0],
                       [1.0, 1.0, 1.0]])
probs = softmax(logits, dim=-1)

print(f"Logits:\n{logits}")
print(f"\nSoftmax:\n{probs}")
print(f"Sum per row (should be 1): {probs.sum(dim=-1)}")
print(f"Matches torch.softmax: {torch.allclose(probs, torch.softmax(logits, dim=-1))}")


# Q2: Implement layer normalization
print("\n--- Q2: Layer Normalization Implementation ---")

def layer_norm(x, eps=1e-5):
    """
    Layer normalization - normalizes across features (last dimension).

    LayerNorm(x) = (x - mean) / sqrt(var + eps)

    Unlike BatchNorm, this normalizes within each sample.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

x = torch.randn(2, 4)  # 2 samples, 4 features
x_norm = layer_norm(x)

print(f"Input:\n{x}")
print(f"\nLayer Normalized:\n{x_norm}")
print(f"Mean per sample (should be ~0): {x_norm.mean(dim=-1)}")
print(f"Std per sample (should be ~1): {x_norm.std(dim=-1)}")


# Q3: Implement batch matrix multiply using einsum
print("\n--- Q3: Einsum for Batched Operations ---")

batch, n, m, p = 4, 3, 4, 5
A = torch.randn(batch, n, m)
B = torch.randn(batch, m, p)

# Standard way
C1 = torch.bmm(A, B)

# Einsum way (more flexible)
C2 = torch.einsum('bnm,bmp->bnp', A, B)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"Result shape: {C1.shape}")
print(f"bmm == einsum: {torch.allclose(C1, C2)}")

# Einsum attention scores
print("\nEinsum for attention (Q @ K^T):")
Q = torch.randn(2, 8, 10, 64)  # batch, heads, seq, d_k
K = torch.randn(2, 8, 10, 64)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Scores shape: {scores.shape}")


# Q4: Implement attention scores
print("\n--- Q4: Scaled Dot-Product Attention ---")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Queries (batch, heads, seq_len, d_k)
        K: Keys (batch, heads, seq_len, d_k)
        V: Values (batch, heads, seq_len, d_v)
        mask: Optional mask (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

    Returns:
        output: (batch, heads, seq_len, d_v)
        attention_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# Example
batch, heads, seq_len, d_k = 2, 4, 8, 32
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Q, K, V shape: ({batch}, {heads}, {seq_len}, {d_k})")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Weights sum to 1: {weights.sum(dim=-1)[0, 0, 0]:.4f}")


# =============================================================================
# SECTION 8: PRACTICE EXERCISES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: PRACTICE EXERCISES")
print("=" * 70)

print("""
EXERCISE 1: Stride Calculation
------------------------------
Given a tensor with shape (2, 3, 4, 5), calculate the strides.

Answer:
stride[3] = 1
stride[2] = 5 * 1 = 5
stride[1] = 4 * 5 = 20
stride[0] = 3 * 20 = 60
Result: (60, 20, 5, 1)
""")

t = torch.randn(2, 3, 4, 5)
print(f"Verification - actual stride: {t.stride()}")

print("""
EXERCISE 2: Broadcasting
------------------------
What is the result shape of:
A = torch.randn(3, 1, 4)
B = torch.randn(2, 4)
C = A + B

Answer:
A: (3, 1, 4)
B:    (2, 4) → (1, 2, 4)
-----------------------
Result: (3, 2, 4)
""")

A = torch.randn(3, 1, 4)
B = torch.randn(2, 4)
C = A + B
print(f"Verification - actual shape: {C.shape}")

print("""
EXERCISE 3: View Failure
------------------------
Why does this fail?
    t = torch.randn(4, 3)
    t_permuted = t.permute(1, 0)
    t_permuted.view(12)

Answer:
permute() changes strides but not memory layout, making the tensor
non-contiguous. view() requires contiguous tensors.

Fix: t_permuted.contiguous().view(12) or t_permuted.reshape(12)
""")

print("\n" + "=" * 70)
print("END OF MODULE 1.1")
print("=" * 70)
