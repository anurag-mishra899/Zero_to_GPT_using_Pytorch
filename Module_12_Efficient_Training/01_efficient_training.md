# Module 12.1: Efficient Training Techniques

## Table of Contents
1. [Mixed Precision Training](#1-mixed-precision-training)
2. [Gradient Checkpointing](#2-gradient-checkpointing)
3. [Gradient Accumulation](#3-gradient-accumulation)
4. [Memory Optimization](#4-memory-optimization)
5. [Data Loading Optimization](#5-data-loading-optimization)
6. [Training Stability](#6-training-stability)
7. [Interview Questions](#7-interview-questions)
8. [Summary](#8-summary)

---

## 1. Mixed Precision Training

### 1.1 Why Mixed Precision?

**FP32** (32-bit floating point):
```
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits
- Range: ±3.4e38
- Memory: 4 bytes
```

**FP16** (16-bit):
```
- 1 sign bit
- 5 exponent bits
- 10 mantissa bits
- Range: ±65504
- Memory: 2 bytes
```

**BF16** (Brain Float 16):
```
- 1 sign bit
- 8 exponent bits (same as FP32!)
- 7 mantissa bits
- Range: ±3.4e38 (same as FP32!)
- Memory: 2 bytes
```

### 1.2 Benefits of Mixed Precision

| Benefit | Impact |
|---------|--------|
| Memory | 2x reduction (model + activations) |
| Speed | 2-4x speedup (tensor cores) |
| Batch size | Can fit larger batches |
| Training | Often equivalent accuracy |

### 1.3 The Mixed Precision Strategy

**Keep some things in FP32**:
```
FP32 (master weights):
- Weight parameters (for updates)
- Gradient accumulation
- Loss scaling

FP16/BF16 (computation):
- Forward pass
- Backward pass
- Gradient computation
```

### 1.4 Loss Scaling

**Problem**: FP16 has limited range, small gradients underflow to 0.

**Solution**: Scale loss before backward, unscale gradients before update.

```python
# Scale loss
scaled_loss = loss * loss_scale

# Backward
scaled_loss.backward()

# Unscale gradients
for param in model.parameters():
    param.grad /= loss_scale

# Update
optimizer.step()
```

### 1.5 Dynamic Loss Scaling

Automatically adjust loss scale:
```
1. Start with large scale (e.g., 2^16)
2. If NaN/Inf gradients:
   - Skip update
   - Reduce scale by factor (e.g., 0.5)
3. If N consecutive good steps:
   - Increase scale (e.g., 2x)
```

### 1.6 PyTorch AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)

    # Backward with scaling
    scaler.scale(loss).backward()

    # Unscale and update
    scaler.step(optimizer)
    scaler.update()
```

### 1.7 BF16 vs FP16

| Aspect | FP16 | BF16 |
|--------|------|------|
| Range | ±65504 | ±3.4e38 |
| Precision | Higher | Lower |
| Loss scaling | Required | Usually not needed |
| Support | Older GPUs | A100+, TPUs |
| Stability | May need tuning | More stable |

**Recommendation**: Use BF16 if available (simpler, more stable).

---

## 2. Gradient Checkpointing

### 2.1 The Memory Problem

During forward pass, save activations for backward:
```
Layer 1 → save a1 → Layer 2 → save a2 → ... → Layer N → save aN

Memory: O(N × batch_size × hidden_dim)

For GPT-3 (96 layers):
  96 × 2048 × 12288 × 2 bytes ≈ 4.8 GB per sequence!
```

### 2.2 Gradient Checkpointing Idea

**Trade compute for memory**:
```
Standard:
  Forward: Save all activations
  Backward: Use saved activations

Checkpointing:
  Forward: Save only checkpoint activations
  Backward: Recompute activations between checkpoints
```

### 2.3 Memory vs Compute Trade-off

```
No checkpointing:
  Memory: O(N)
  Compute: 1x forward + 1x backward

Full checkpointing (every layer):
  Memory: O(1)
  Compute: 1x forward + 2x forward + 1x backward ≈ 2x overhead

Checkpoint every √N layers:
  Memory: O(√N)
  Compute: ~30% overhead
```

### 2.4 Implementation

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        # Use checkpointing
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        # Actual computation
        x = self.attention(x)
        x = self.ffn(x)
        return x
```

### 2.5 Segment-wise Checkpointing

Checkpoint groups of layers (segments):
```
Layers 0-7:   Checkpoint at input
Layers 8-15:  Checkpoint at input
Layers 16-23: Checkpoint at input
Layers 24-31: No checkpoint (outputs needed anyway)
```

**Memory**: O(N/k) where k = checkpoint frequency

---

## 3. Gradient Accumulation

### 3.1 The Problem

**Large effective batch sizes** improve training but require:
```
Memory ∝ batch_size × seq_len × hidden_dim

For batch_size=32, seq_len=2048, hidden_dim=4096:
  Memory ≈ 32 × 2048 × 4096 × 2 bytes × activations ≈ massive
```

### 3.2 Gradient Accumulation Solution

Simulate large batch by accumulating gradients:
```
Effective batch = micro_batch × accumulation_steps

Instead of:
  batch_size = 64

Do:
  micro_batch = 8
  accumulation_steps = 8
  → Same effective batch, 8x less memory
```

### 3.3 Implementation

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward + backward
    loss = model(batch)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()

    # Update only every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3.4 Considerations

```
1. Learning rate: May need adjustment
   - Some scale LR with effective batch size
   - Linear scaling rule: LR ∝ batch_size

2. BatchNorm: Statistics computed per micro-batch
   - May need SyncBatchNorm or GroupNorm

3. Gradient sync: In distributed, sync only on update step
```

---

## 4. Memory Optimization

### 4.1 Memory Breakdown

For training a transformer:
```
1. Model parameters: 4 bytes × num_params (FP32 master)
2. Gradients: 4 bytes × num_params
3. Optimizer states: 8-12 bytes × num_params (Adam)
4. Activations: O(batch × seq × hidden × layers)
5. Temporary buffers: Varies
```

### 4.2 Optimizer Memory

**Adam optimizer states**:
```
Per parameter:
  - First moment (m): 4 bytes
  - Second moment (v): 4 bytes

Total: 8 bytes per parameter

For 7B model:
  7B × 8 bytes = 56 GB just for optimizer!
```

### 4.3 Memory-Efficient Optimizers

**8-bit Adam (bitsandbytes)**:
```
Store m, v in 8-bit
~4x memory reduction for optimizer states

For 7B model:
  56 GB → 14 GB
```

**Adafactor**:
```
Factor optimizer states across dimensions
Much less memory than Adam
Used in T5, PaLM
```

### 4.4 Activation Memory Strategies

```
1. Gradient Checkpointing: Recompute activations
2. Activation Compression: Quantize stored activations
3. Offloading: Move activations to CPU
4. Selective Saving: Only save necessary activations
```

### 4.5 CPU Offloading

Move some data to CPU memory:
```python
# DeepSpeed ZeRO-Offload
{
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
    },
    "offload_param": {
        "device": "cpu"
    }
}
```

**Trade-off**: Slower (CPU-GPU transfer) but more memory.

---

## 5. Data Loading Optimization

### 5.1 DataLoader Best Practices

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,        # Multiple workers
    pin_memory=True,      # Faster GPU transfer
    prefetch_factor=2,    # Prefetch batches
    persistent_workers=True,  # Keep workers alive
)
```

### 5.2 Efficient Data Formats

**Memory-mapped files**:
```python
# NumPy memory map
data = np.memmap('data.bin', dtype='float32', mode='r', shape=(N, D))

# Access without loading entire file
batch = data[indices]
```

**Arrow/Parquet** for large datasets:
```python
from datasets import load_dataset
dataset = load_dataset('path', streaming=True)  # Streaming
```

### 5.3 Tokenization Optimization

**Pre-tokenize** dataset:
```
1. Tokenize entire dataset once
2. Save tokenized version
3. Load pre-tokenized during training

Saves tokenization time every epoch
```

**On-the-fly tokenization** for very large datasets:
```python
def collate_fn(examples):
    # Tokenize in collate function
    return tokenizer(examples, padding=True, truncation=True)
```

### 5.4 Packing Sequences

Combine short sequences to fill context:
```
Instead of:
  [seq1, PAD, PAD, PAD]  # 75% padding
  [seq2, PAD, PAD]
  [seq3]

Do:
  [seq1, seq2, PAD]      # Better utilization
  [seq3, PAD, PAD]
```

---

## 6. Training Stability

### 6.1 Gradient Clipping

Prevent exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Types**:
- Clip by norm: Scale if total norm > threshold
- Clip by value: Clamp each element to range

### 6.2 Learning Rate Warmup

Start with small LR, gradually increase:
```python
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return max_lr  # Then decay
```

**Why**:
- Early gradients are noisy (random weights)
- Large steps can diverge
- Warmup stabilizes early training

### 6.3 Initialization

**GPT-style initialization**:
```python
# Scale residual layers by 1/√N
for layer in residual_layers:
    layer.weight.data /= math.sqrt(num_layers)

# This prevents residual stream from growing
```

### 6.4 Loss Spikes

Common causes and fixes:
```
1. Learning rate too high → Reduce LR
2. Bad batch → Gradient clipping
3. NaN in loss → Check for log(0), division by 0
4. FP16 overflow → Loss scaling
5. Data issue → Validate data pipeline
```

### 6.5 Monitoring

Key metrics to track:
```
- Loss (train/val)
- Gradient norm (before/after clipping)
- Learning rate
- GPU memory
- Throughput (tokens/sec)
- Loss scale (if using AMP)
```

---

## 7. Interview Questions

### Q1: Explain mixed precision training.

**Answer**:

**Mixed precision** uses lower precision (FP16/BF16) for computation while maintaining FP32 master weights.

**Strategy**:
```
FP32: Master weights, gradient accumulation
FP16/BF16: Forward pass, backward pass, gradients
```

**Benefits**:
- 2x memory reduction
- 2-4x speedup (tensor cores)
- Larger batch sizes

**Loss scaling** (for FP16):
```python
# Scale loss to prevent gradient underflow
scaled_loss = loss * scale
scaled_loss.backward()
gradients /= scale
```

**BF16 vs FP16**:
- BF16: Same range as FP32, no loss scaling needed
- FP16: Better precision, requires loss scaling

### Q2: What is gradient checkpointing?

**Answer**:

**Problem**: Forward pass saves all activations for backward → O(N) memory.

**Solution**: Save only checkpoints, recompute activations during backward.

```
Standard: Save a1, a2, ..., aN (O(N) memory)

Checkpointing: Save a1, a_k, a_2k, ... (O(N/k) memory)
  During backward, recompute a2 to a_k from a1, etc.
```

**Trade-off**:
- Memory: O(√N) with optimal checkpointing
- Compute: ~30% overhead (recomputation)

**Implementation**:
```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(layer, input, use_reentrant=False)
```

### Q3: How does gradient accumulation work?

**Answer**:

**Purpose**: Simulate large batch sizes with limited memory.

```python
accumulation_steps = 4
effective_batch = micro_batch * accumulation_steps

for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps  # Normalize
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Key points**:
- Divide loss by accumulation steps (gradient normalization)
- Only call optimizer.step() every N steps
- Equivalent to larger batch (same gradients)
- Memory proportional to micro_batch, not effective_batch

### Q4: What are the main components of GPU memory during training?

**Answer**:

```
1. Model Parameters
   - FP32 master: 4 bytes × params
   - FP16 copy: 2 bytes × params (if mixed precision)

2. Gradients
   - Same size as parameters (4 bytes typically)

3. Optimizer States (Adam)
   - First moment (m): 4 bytes × params
   - Second moment (v): 4 bytes × params
   - Total: 8 bytes × params

4. Activations
   - O(batch × seq × hidden × layers)
   - Often largest component

5. Temporary Buffers
   - MatMul intermediates
   - Gradient computation buffers
```

**Example for 7B model**:
```
Parameters: 7B × 4 = 28 GB
Gradients: 7B × 4 = 28 GB
Optimizer: 7B × 8 = 56 GB
Activations: Variable (10-100+ GB)
Total: 112 GB + activations
```

### Q5: Explain learning rate warmup.

**Answer**:

**Purpose**: Stabilize early training.

**Problem without warmup**:
- Initial weights are random
- Gradients are noisy/unreliable
- Large LR → divergence

**Warmup schedule**:
```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
else:
    # Decay schedule (cosine, linear, etc.)
```

**Typical warmup**:
- LLMs: 2000-4000 steps
- Fine-tuning: 100-500 steps
- BERT: 10% of total steps

**Why it works**:
- Gradients become reliable as weights improve
- Gradually increase step size
- Prevents early divergence

---

## 8. Summary

### Key Techniques

| Technique | Memory | Compute | Complexity |
|-----------|--------|---------|------------|
| Mixed Precision | 2x ↓ | 2-4x ↑ | Low |
| Gradient Checkpointing | √N ↓ | 30% ↓ | Low |
| Gradient Accumulation | k× ↓ | Same | Low |
| 8-bit Optimizer | 4x ↓ (optim) | Same | Medium |
| CPU Offloading | Variable | 2-10x ↓ | Medium |

### Training Configuration Template

```python
# Mixed precision
scaler = GradScaler()
autocast_dtype = torch.bfloat16  # or torch.float16

# Gradient accumulation
accumulation_steps = 8

# Gradient checkpointing
model.gradient_checkpointing_enable()

# Gradient clipping
max_grad_norm = 1.0

# Warmup
warmup_steps = 2000
```

### Key Takeaways

1. **Mixed precision is essential**: Always use BF16/FP16
2. **Gradient checkpointing for large models**: Trade compute for memory
3. **Gradient accumulation for large batches**: Simulate any batch size
4. **Monitor everything**: Loss, gradients, memory, throughput
5. **Warmup stabilizes training**: Especially for large LR
