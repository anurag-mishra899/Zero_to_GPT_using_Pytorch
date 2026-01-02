# Module 14: Parameter-Efficient Fine-Tuning (PEFT)

## Table of Contents
1. [Why PEFT?](#1-why-peft)
2. [LoRA (Low-Rank Adaptation)](#2-lora-low-rank-adaptation)
3. [QLoRA](#3-qlora)
4. [Other PEFT Methods](#4-other-peft-methods)
5. [Practical Considerations](#5-practical-considerations)
6. [Interview Questions](#6-interview-questions)
7. [Summary](#7-summary)

---

## 1. Why PEFT?

### 1.1 The Full Fine-tuning Problem

Full fine-tuning is expensive:
```
LLaMA-70B:
  Parameters: 70B × 4 bytes = 280 GB (FP32)
  Gradients: 280 GB
  Optimizer states: 560 GB (Adam)
  Total: ~1.1 TB per training run

Problems:
  - Requires massive GPU memory
  - Expensive to store one copy per task
  - Risk of catastrophic forgetting
  - Slow to train
```

### 1.2 PEFT Solution

Train only a small subset of parameters:
```
Full fine-tuning: Train all 70B parameters
PEFT: Train only 0.1-1% of parameters

Benefits:
  - Much less memory (only train small adapter)
  - Store tiny adapters per task (<100 MB vs 280 GB)
  - Faster training
  - Less overfitting
  - Can share base model across tasks
```

### 1.3 PEFT Taxonomy

```
1. Additive Methods:
   - Adapters: Add small layers
   - Prefix Tuning: Add learnable prefixes
   - Prompt Tuning: Learn soft prompts

2. Selective Methods:
   - BitFit: Train only biases
   - Sparse fine-tuning: Select subset of parameters

3. Reparameterization:
   - LoRA: Low-rank weight updates
   - DoRA: Decomposed LoRA
```

---

## 2. LoRA (Low-Rank Adaptation)

### 2.1 Key Insight

Weight updates during fine-tuning are low-rank:
```
Full fine-tuning:
  W_new = W_old + ΔW

Observation:
  ΔW often has low intrinsic rank
  i.e., can be approximated by low-rank matrices

LoRA insight:
  Instead of learning full ΔW (d × d)
  Learn ΔW = BA where:
    B: d × r (down projection)
    A: r × d (up projection)
    r << d (typical: r = 8 to 64)
```

### 2.2 Mathematical Formulation

```
Original linear layer:
  h = Wx

With LoRA:
  h = Wx + (BA)x
  h = Wx + B(Ax)

Forward pass:
  1. Original path: Wx (frozen)
  2. LoRA path: B(Ax) (trainable)
  3. Sum both paths
```

### 2.3 Parameter Savings

```
Original weight: W ∈ ℝ^(d × k)
  Parameters: d × k

LoRA matrices: A ∈ ℝ^(r × k), B ∈ ℝ^(d × r)
  Parameters: r × k + d × r = r(d + k)

Example (d = 4096, k = 4096, r = 8):
  Full: 4096 × 4096 = 16.7M
  LoRA: 8 × (4096 + 4096) = 65.5K
  Reduction: 255×
```

### 2.4 Initialization

Critical for training stability:
```python
# A: Random Gaussian initialization
A = torch.randn(r, k) / math.sqrt(r)

# B: Zero initialization
B = torch.zeros(d, r)

# Result: BA = 0 at start
# Fine-tuning starts from pretrained weights
```

### 2.5 Scaling Factor

LoRA uses scaling factor α:
```
ΔW = (α/r) × BA

Common: α = r (so α/r = 1)
Or: α = 2r (for 2× contribution)

Why scale?
  - Prevents magnitude change when adjusting r
  - Keeps learning rate stable across rank choices
```

### 2.6 Which Layers to Apply LoRA?

```
Transformer options:
  - Query (Wq): Common choice
  - Key (Wk): Less common
  - Value (Wv): Common choice
  - Output projection (Wo): Effective
  - MLP layers: Can help

Typical configuration:
  Apply to: Wq, Wv
  Rank: r = 8 to 64
  Alpha: α = 16 to 32
```

### 2.7 Merging LoRA

After training, merge into base weights:
```python
# Merge LoRA into original weights
W_merged = W + (alpha / r) * B @ A

# Now: h = W_merged @ x
# No inference overhead!
```

---

## 3. QLoRA

### 3.1 Motivation

LoRA still requires loading full model:
```
LLaMA-70B:
  Base model: 140 GB (FP16)
  Still too large for consumer GPUs!

QLoRA solution:
  Quantize base model to 4-bit
  Keep LoRA adapters in FP16/BF16
```

### 3.2 4-bit NormalFloat (NF4)

Quantization-aware data type:
```
Normal Float 4-bit:
  - Optimized for normally distributed weights
  - Quantization levels match Gaussian distribution
  - Better than uniform 4-bit

Quantile-based quantization:
  q_i = Φ^(-1)((i + 0.5) / 16)  for i = 0..15
  where Φ is standard normal CDF
```

### 3.3 Double Quantization

Quantize the quantization constants:
```
Block-wise quantization:
  W_quant = round(W / scale) × scale

Problem: Need to store scale per block (FP32)
  64 elements/block × 4 bytes = significant memory

Double quantization:
  Quantize scales to FP8
  Memory reduction: ~0.5 bit per parameter
```

### 3.4 Paged Optimizers

Handle memory spikes during training:
```
Problem: Gradient checkpointing causes memory spikes
  Spike can exceed GPU memory → OOM

Paged optimizer:
  Automatically offload optimizer states to CPU
  Page back when needed
  Handles spikes gracefully
```

### 3.5 QLoRA Training Flow

```
Forward:
  1. Dequantize block of weights (4-bit → FP16)
  2. Compute: h = W_dequant × x + B(Ax)
  3. Keep LoRA in FP16 for gradients

Backward:
  1. Gradients only flow through LoRA
  2. Base model frozen (no gradients needed)
  3. Update LoRA parameters in FP16

Memory:
  Base model: 4-bit (~0.5 bytes/param)
  LoRA: 16-bit (2 bytes/param for small r)
  Optimizer: Only for LoRA params
```

### 3.6 Memory Comparison

```
Fine-tuning LLaMA-65B:

Full FP16:
  Model: 130 GB
  Gradients: 130 GB
  Optimizer: 260 GB
  Total: 520 GB (8× A100-80GB)

LoRA FP16:
  Model: 130 GB
  LoRA params: ~50 MB
  Gradients: ~50 MB
  Optimizer: ~100 MB
  Total: ~130 GB (2× A100-80GB)

QLoRA:
  Model (4-bit): 32 GB
  LoRA params: ~50 MB
  Gradients: ~50 MB
  Optimizer: ~100 MB
  Total: ~33 GB (1× A100-40GB!)
```

---

## 4. Other PEFT Methods

### 4.1 Adapters

Add small bottleneck layers:
```
Original: x → LayerNorm → Attention → Add → LayerNorm → FFN → Add

With Adapter:
  x → LayerNorm → Attention → Add → Adapter → LayerNorm → FFN → Add → Adapter

Adapter architecture:
  Down project: d → r
  Nonlinearity: GELU/ReLU
  Up project: r → d
  Residual: Add input
```

```python
class Adapter(nn.Module):
    def __init__(self, d_model, r=64):
        super().__init__()
        self.down = nn.Linear(d_model, r)
        self.up = nn.Linear(r, d_model)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))
```

### 4.2 Prefix Tuning

Add learnable prefix to keys and values:
```
Original attention:
  Q, K, V from input x

With prefix:
  K = [K_prefix; K_input]
  V = [V_prefix; V_input]

K_prefix, V_prefix are learnable (per layer)
Input tokens attend to prefix tokens
```

```python
class PrefixTuning(nn.Module):
    def __init__(self, num_layers, d_model, prefix_len=10):
        super().__init__()
        # Learnable prefix for each layer
        self.prefix_k = nn.Parameter(torch.randn(num_layers, prefix_len, d_model))
        self.prefix_v = nn.Parameter(torch.randn(num_layers, prefix_len, d_model))
```

### 4.3 Prompt Tuning

Learn soft tokens prepended to input:
```
Hard prompts: "Translate to French: {input}"
Soft prompts: [P1][P2]...[Pn] {input}

P1...Pn are learnable embeddings
Not tied to vocabulary
Optimized for the task
```

### 4.4 IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Scale activations instead of adding:
```
LoRA: h = Wx + BAx (additive)
IA3: h = (l_v ⊙ V)x for values, similar for keys, FFN

l_v: Learnable vector (element-wise scaling)
⊙: Element-wise multiplication

Fewer parameters than LoRA
Fast inference (fuse into weights)
```

### 4.5 DoRA (Weight-Decomposed Low-Rank Adaptation)

Decompose into magnitude and direction:
```
Weight decomposition:
  W = m × (W / ||W||)
    = magnitude × direction

DoRA:
  Adapt magnitude: m (learnable vector)
  Adapt direction: V + BA (LoRA on direction)

Better than LoRA at same parameter count
More stable training
```

### 4.6 Comparison

| Method | Parameters | Inference Cost | Quality |
|--------|------------|----------------|---------|
| Full FT | 100% | Baseline | Best |
| LoRA | 0.1-1% | No overhead* | Very good |
| QLoRA | 0.1-1% | Quantization | Good |
| Adapters | 1-5% | +Forward pass | Good |
| Prefix | <0.1% | +Attention | Moderate |
| Prompt | <0.01% | +Embedding | Task-specific |

*After merging

---

## 5. Practical Considerations

### 5.1 Rank Selection

```
Lower rank (r = 4-8):
  - Fewer parameters
  - Faster training
  - May underfit complex tasks

Higher rank (r = 64-256):
  - More capacity
  - Better for complex tasks
  - More memory/compute

Guidelines:
  - Start with r = 8
  - Increase if underfitting
  - r = 64 usually sufficient for most tasks
```

### 5.2 Learning Rate

```
LoRA typically needs higher LR than full FT:
  Full FT: 1e-5 to 5e-5
  LoRA: 1e-4 to 3e-4

Why higher?
  - Fewer parameters, can afford larger steps
  - Scale factor α/r affects effective LR
```

### 5.3 Which Layers?

```
Attention layers (recommended):
  - Wq, Wv: Standard choice
  - Wk: Sometimes helps
  - Wo: Effective for some tasks

MLP layers:
  - Gate/Up projection: Can help
  - Down projection: Usually less important

Start with attention, add MLP if needed
```

### 5.4 Multiple LoRA Adapters

```
Task switching:
  Load different LoRA for different tasks
  Base model stays the same

Combining LoRAs:
  W = W_base + LoRA_task1 + LoRA_task2
  Can blend adapters

LoRA arithmetic:
  Interpolate: 0.7 × LoRA_A + 0.3 × LoRA_B
  Negate: W - LoRA (remove capability)
```

### 5.5 Common Issues

```
1. LoRA not training:
   - LR too low (try 1e-4)
   - Alpha/rank ratio wrong
   - Target modules not set correctly

2. Overfitting:
   - Rank too high
   - Add dropout (0.05-0.1)
   - Reduce epochs

3. Poor performance:
   - Apply to more layers
   - Increase rank
   - Check data quality
```

---

## 6. Interview Questions

### Q1: Explain LoRA and why it works.

**Answer**:

**LoRA** (Low-Rank Adaptation) freezes pretrained weights and injects trainable low-rank matrices.

**Why it works**:
```
1. Weight updates are low-rank:
   Research shows ΔW during fine-tuning has low intrinsic dimensionality
   Even though ΔW is d×d, it lies in low-rank subspace

2. Formulation:
   Original: h = Wx
   LoRA: h = Wx + BAx
   Where B ∈ ℝ^(d×r), A ∈ ℝ^(r×d), r << d

3. Parameter efficiency:
   Full: d × d parameters
   LoRA: r(d + d) = 2rd parameters
   With r=8, d=4096: 255× reduction
```

**Implementation details**:
```python
# Initialization (critical!)
A = randn / sqrt(r)  # Normal init
B = zeros            # Zero init

# Scaling factor
scale = alpha / r

# Forward
output = x @ W + scale * (x @ A.T @ B.T)
```

**Benefits**:
- No inference latency after merging
- Can store multiple adapters per base model
- Less memory during training

### Q2: What is QLoRA and how does it enable training on consumer GPUs?

**Answer**:

**QLoRA** = Quantized LoRA

**Key innovations**:

1. **4-bit NormalFloat (NF4)**:
```
Quantize base model to 4-bit
NF4 optimized for neural net weight distribution
0.5 bytes/param instead of 2 bytes (FP16)
```

2. **Double Quantization**:
```
Block-wise quantization needs scales
Quantize the scales too (FP32 → FP8)
Saves ~0.5 bits per parameter
```

3. **Paged Optimizers**:
```
Offload optimizer states to CPU
Page back during optimizer step
Handles memory spikes gracefully
```

**Memory comparison for 65B model**:
```
Full FP16 training: ~520 GB
LoRA FP16: ~130 GB
QLoRA: ~33 GB (fits on 1× A100-40GB!)
```

**Trade-offs**:
- Slight quality loss from quantization
- Training slower (dequantization overhead)
- But enables training models otherwise impossible

### Q3: Compare LoRA, Adapters, and Prefix Tuning.

**Answer**:

| Aspect | LoRA | Adapters | Prefix Tuning |
|--------|------|----------|---------------|
| **Location** | Parallel to weights | Serial (new layers) | Attention prefix |
| **Parameters** | 0.1-1% | 1-5% | <0.1% |
| **Inference** | None (after merge) | +Forward pass | +Attention |
| **Architecture** | Low-rank matrices | Bottleneck MLP | Learnable K, V |

**LoRA**:
```
W' = W + BA (low-rank)
Pros: No inference cost, flexible
Cons: Need to choose rank, target layers
```

**Adapters**:
```
h = x + Up(GELU(Down(x)))
Pros: More expressive, residual connection
Cons: Inference overhead, more parameters
```

**Prefix Tuning**:
```
K = [K_prefix; K_input], V = [V_prefix; V_input]
Pros: Very few parameters
Cons: Limited capacity, longer context
```

**When to use what**:
- LoRA: Default choice, good balance
- Adapters: Need more capacity, OK with overhead
- Prefix: Very few parameters, specific tasks

### Q4: How do you choose LoRA hyperparameters?

**Answer**:

**Rank (r)**:
```
r = 8: Good starting point
r = 16-64: Complex tasks
r = 128-256: Near full fine-tuning

Rule: Start small, increase if underfitting
```

**Alpha (α)**:
```
Common: α = 2r or α = r
Effect: Scales learning rate effectively
Higher α = stronger adaptation
```

**Target modules**:
```
Standard: q_proj, v_proj (attention)
Extended: + k_proj, o_proj
Maximum: + mlp layers

More modules = more capacity = more memory
```

**Learning rate**:
```
Higher than full FT: 1e-4 to 3e-4
Scale with rank (lower rank, higher LR)
```

**Dropout**:
```
lora_dropout = 0.05-0.1
Helps prevent overfitting
Especially with high rank
```

### Q5: How do you serve multiple LoRA adapters efficiently?

**Answer**:

**Single adapter serving**:
```
1. Merge into base weights:
   W_merged = W_base + (α/r) × BA
   No runtime overhead

2. Or: Keep separate, add at runtime
   Slight overhead but switchable
```

**Multiple adapter serving**:

```
Approach 1: Batched LoRA
  Group requests by adapter
  Process each group with respective LoRA
  Efficient for high-throughput

Approach 2: S-LoRA (Scalable serving)
  Store all LoRAs in unified memory
  Custom CUDA kernels for efficient multi-adapter
  Supports 1000s of adapters concurrently
```

**S-LoRA technique**:
```python
# Unified paging for adapters
# Single base model forward
base_output = base_model(x)

# Batched LoRA application
for adapter_id, batch_indices in grouped_requests:
    lora_output[batch_indices] = lora_forward(
        x[batch_indices],
        adapters[adapter_id]
    )

output = base_output + lora_output
```

**Key insight**: Base model computation is shared, only LoRA differs.

---

## 7. Summary

### Key Techniques

| Method | Memory | Training Speed | Inference | Best For |
|--------|--------|----------------|-----------|----------|
| Full FT | 100% | 1× | Baseline | Best quality |
| LoRA | 10-20% | 1.5-2× | Same | General use |
| QLoRA | 5-10% | 0.5-1× | Same | Consumer GPUs |
| Adapters | 15-25% | 1-1.5× | Slower | Multi-task |
| Prefix | 5-10% | 2-3× | Slower | Few-shot |

### LoRA Configuration Template

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Alpha (often 2×r)
    target_modules=[         # Which layers
        "q_proj", "v_proj",
        "k_proj", "o_proj",
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",             # Don't train biases
    task_type="CAUSAL_LM"    # Task type
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

### Key Takeaways

1. **LoRA is the default choice**: Best balance of efficiency and quality
2. **QLoRA enables consumer GPU training**: 4-bit base + FP16 LoRA
3. **Rank 8-64 covers most cases**: Start small, increase if needed
4. **Apply to attention layers first**: q, v projections minimum
5. **Merge for inference**: Zero runtime overhead after merging
6. **Multiple adapters are cheap**: Store many task-specific adapters
