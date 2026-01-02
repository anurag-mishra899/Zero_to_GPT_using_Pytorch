# Module 13: Distributed Training

## Table of Contents
1. [Why Distributed Training?](#1-why-distributed-training)
2. [Communication Primitives](#2-communication-primitives)
3. [Data Parallelism](#3-data-parallelism)
4. [Distributed Data Parallel (DDP)](#4-distributed-data-parallel-ddp)
5. [Model Parallelism](#5-model-parallelism)
6. [Fully Sharded Data Parallel (FSDP)](#6-fully-sharded-data-parallel-fsdp)
7. [ZeRO Optimization](#7-zero-optimization)
8. [Practical Considerations](#8-practical-considerations)
9. [Interview Questions](#9-interview-questions)
10. [Summary](#10-summary)

---

## 1. Why Distributed Training?

### 1.1 The Scale Problem

Modern LLMs require massive compute:
```
GPT-3 (175B):
  - Training: ~3.14e23 FLOPs
  - Single A100 (312 TFLOPS): ~11,600 days
  - 1000 A100s: ~12 days

LLaMA-70B:
  - Memory for weights: 70B × 4 bytes = 280 GB
  - Single A100: 80 GB HBM
  - Need at least 4 GPUs just for weights!
```

### 1.2 Scaling Strategies

```
1. Data Parallelism:
   - Same model on each GPU
   - Different data batches
   - Average gradients

2. Model Parallelism:
   - Split model across GPUs
   - Tensor parallelism: split layers
   - Pipeline parallelism: split stages

3. Hybrid:
   - Combine data + model parallelism
   - Modern approach for large models
```

### 1.3 Memory Breakdown (Revisited)

For training a model with P parameters:
```
1. Model parameters: P × 4 bytes (FP32)
2. Gradients: P × 4 bytes
3. Optimizer states (Adam):
   - First moment (m): P × 4 bytes
   - Second moment (v): P × 4 bytes
4. Activations: O(batch × seq × hidden × layers)

Total per GPU (data parallel):
  P × 16 bytes + activations

For 7B model:
  7B × 16 = 112 GB (without activations!)
```

---

## 2. Communication Primitives

### 2.1 Collective Operations

**All-Reduce**: Reduce + broadcast result to all
```
GPU 0: [1, 2]    →    GPU 0: [10, 14]
GPU 1: [3, 4]    →    GPU 1: [10, 14]
GPU 2: [6, 8]    →    GPU 2: [10, 14]

Sum: [1+3+6, 2+4+8] = [10, 14] on ALL GPUs
```

**All-Gather**: Gather tensors from all to all
```
GPU 0: [A]       →    GPU 0: [A, B, C]
GPU 1: [B]       →    GPU 1: [A, B, C]
GPU 2: [C]       →    GPU 2: [A, B, C]
```

**Reduce-Scatter**: Reduce + scatter results
```
GPU 0: [1,2,3]   →    GPU 0: [6]     (sum of position 0: 1+2+3)
GPU 1: [2,3,4]   →    GPU 1: [9]     (sum of position 1: 2+3+4)
GPU 2: [3,4,5]   →    GPU 2: [12]    (sum of position 2: 3+4+5)
```

**Broadcast**: Send from one to all
```
GPU 0: [A]       →    GPU 0: [A]
GPU 1: []        →    GPU 1: [A]
GPU 2: []        →    GPU 2: [A]
```

### 2.2 Communication Costs

```
Ring All-Reduce:
  - Data volume: 2(N-1)/N × data_size ≈ 2 × data_size
  - Steps: 2(N-1) for N GPUs
  - Bandwidth optimal

Tree All-Reduce:
  - Latency: O(log N)
  - Better for small messages

For gradient sync (P parameters):
  - All-reduce: 2P × bytes_per_param
  - Time: 2P × bytes / bandwidth
```

### 2.3 NCCL

NVIDIA Collective Communications Library:
```python
import torch.distributed as dist

# Initialize
dist.init_process_group(backend='nccl')

# All-reduce
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-gather
dist.all_gather(tensor_list, tensor)

# Reduce-scatter
dist.reduce_scatter(output, input_list)
```

---

## 3. Data Parallelism

### 3.1 Basic Concept

```
Each GPU:
  1. Has full copy of model
  2. Processes different data batch
  3. Computes gradients independently
  4. Synchronizes gradients (all-reduce)
  5. Updates model identically
```

### 3.2 Naive Data Parallel

```python
# PyTorch DataParallel (simple but inefficient)
model = nn.DataParallel(model)

# Limitations:
# - Single process, GIL bottleneck
# - Gradient all-reduce on GPU 0
# - Unbalanced memory
```

### 3.3 Gradient Synchronization

```
Forward pass: Independent on each GPU
Backward pass: Independent computation
After backward: All-reduce gradients

All-reduce averages gradients:
  g_synced = (g_0 + g_1 + ... + g_N) / N

Equivalent to single GPU with N× batch size
```

### 3.4 Scaling Efficiency

```
Ideal: N GPUs → N× throughput
Reality: Communication overhead

Efficiency = N × single_GPU_throughput / actual_throughput

Factors:
- Gradient size (P parameters)
- Network bandwidth
- Computation/communication overlap
```

---

## 4. Distributed Data Parallel (DDP)

### 4.1 Architecture

```
DDP improvements over DataParallel:
1. Multi-process: One process per GPU
2. No GIL bottleneck
3. Overlapped communication
4. Gradient bucketing
```

### 4.2 Gradient Bucketing

Group gradients for efficient all-reduce:
```
Instead of:
  all_reduce(grad_1)  # Small, high latency overhead
  all_reduce(grad_2)
  ...

Do:
  bucket = [grad_1, grad_2, ..., grad_k]  # ~25MB default
  all_reduce(bucket)  # Single operation, amortized latency
```

### 4.3 Overlap Communication with Computation

```
Timeline without overlap:
  [Forward][Backward][All-Reduce][Update]

Timeline with overlap:
  [Forward][Backward layer N | All-Reduce bucket 1]
           [Backward layer N-1 | All-Reduce bucket 2]
           ...
  [Update]

Key insight: Backward computes gradients in reverse order
             Start all-reduce for early gradients immediately
```

### 4.4 DDP Implementation

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=world_size,
    rank=rank
)

# Create model and move to GPU
model = Model().to(rank)
model = DDP(model, device_ids=[rank])

# Training loop (same as single GPU!)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # Gradients auto-synced
    optimizer.step()
```

### 4.5 Launch Script

```bash
# torchrun (recommended)
torchrun --nproc_per_node=4 train.py

# Environment variables set automatically:
# - LOCAL_RANK: GPU index on this node
# - RANK: Global rank
# - WORLD_SIZE: Total processes
```

### 4.6 DistributedSampler

Ensure each GPU sees different data:
```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

dataloader = DataLoader(dataset, sampler=sampler)

# Important: Set epoch for proper shuffling
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        ...
```

---

## 5. Model Parallelism

### 5.1 When Data Parallelism Isn't Enough

```
Problem: Model too large for single GPU
  - 70B model: 280 GB in FP32
  - Single A100: 80 GB

Solution: Split model across GPUs
```

### 5.2 Tensor Parallelism

Split individual layers across GPUs:
```
Linear layer: Y = XW where W is (d_in × d_out)

Column parallel:
  W = [W1 | W2]  (split columns)
  GPU 0: Y1 = X @ W1
  GPU 1: Y2 = X @ W2
  Y = [Y1 | Y2]

Row parallel:
  W = [W1]      (split rows)
      [W2]
  X = [X1 | X2] (split input)
  GPU 0: Y1 = X1 @ W1
  GPU 1: Y2 = X2 @ W2
  Y = Y1 + Y2  (all-reduce)
```

### 5.3 Tensor Parallelism for Transformers

```
MLP: H → 4H → H

Column parallel for first linear:
  [X] → GPU 0: X @ W1[:, :2H] = H1
        GPU 1: X @ W1[:, 2H:] = H2

Row parallel for second linear:
  GPU 0: H1 @ W2[:2H, :] = O1
  GPU 1: H2 @ W2[2H:, :] = O2
  All-reduce: O = O1 + O2

Communication: 1 all-reduce per MLP
```

### 5.4 Attention Parallelism

```
Split heads across GPUs:
  32 heads, 4 GPUs → 8 heads per GPU

Each GPU:
  - Computes attention for its heads
  - Output: (batch, seq, 8, head_dim)

After attention:
  - All-gather or concatenate
  - Apply output projection
```

### 5.5 Pipeline Parallelism

Split model into sequential stages:
```
Stage 0 (GPU 0): Embedding + Layers 0-7
Stage 1 (GPU 1): Layers 8-15
Stage 2 (GPU 2): Layers 16-23
Stage 3 (GPU 3): Layers 24-31 + LM Head

Forward:
  GPU 0 → send activations → GPU 1 → send → GPU 2 → send → GPU 3
```

### 5.6 Pipeline Bubble

```
Naive pipeline (4 stages, 4 micro-batches):
  GPU 0: [F1][F2][F3][F4][  ][  ][  ][B1][B2][B3][B4]
  GPU 1: [  ][F1][F2][F3][F4][  ][B1][B2][B3][B4][  ]
  GPU 2: [  ][  ][F1][F2][F3][F4][B1][B2][B3][B4][  ]
  GPU 3: [  ][  ][  ][F1][F2][F3][B4][B3][B2][B1][  ]

Bubble = idle time
Bubble fraction = (p-1) / (m + p - 1)
  p = pipeline stages
  m = micro-batches

For p=4, m=4: bubble = 3/7 ≈ 43% waste!
For p=4, m=32: bubble = 3/35 ≈ 9%
```

### 5.7 1F1B Schedule

Interleave forward and backward:
```
1F1B (1 Forward, 1 Backward):
  GPU 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7][B8]
  GPU 1: [  ][F1][F2][F3][B1][F4][B2][F5][B3][F6][B4][B5][B6][B7][B8]
  ...

Benefits:
- Same bubble fraction
- But: Lower memory (don't store all activations)
- Memory: O(p) instead of O(m)
```

---

## 6. Fully Sharded Data Parallel (FSDP)

### 6.1 The Insight

```
Problem with DDP:
  - Each GPU stores: full params + full grads + full optimizer states
  - Redundant storage across GPUs

FSDP insight:
  - Shard everything across GPUs
  - Gather when needed, discard after
```

### 6.2 FSDP Operations

```
Each GPU stores: 1/N of params, grads, optimizer states

Forward pass:
  1. All-gather parameters (reconstruct full layer)
  2. Compute forward
  3. Discard gathered parameters

Backward pass:
  1. All-gather parameters (need for gradient computation)
  2. Compute gradients
  3. Reduce-scatter gradients (each GPU gets 1/N)
  4. Discard gathered parameters

Update:
  - Each GPU updates its 1/N of parameters
  - Using its 1/N of gradients and optimizer states
```

### 6.3 Memory Savings

```
DDP (per GPU):
  Parameters: P × 4 bytes
  Gradients: P × 4 bytes
  Optimizer: P × 8 bytes (Adam)
  Total: P × 16 bytes

FSDP (per GPU):
  Parameters: P/N × 4 bytes
  Gradients: P/N × 4 bytes
  Optimizer: P/N × 8 bytes
  Total: P/N × 16 bytes

N GPUs → N× memory reduction!
```

### 6.4 Communication Cost

```
DDP:
  - All-reduce gradients: 2P (reduce + broadcast)

FSDP:
  - Forward: All-gather params per layer: P
  - Backward: All-gather params + Reduce-scatter grads: 2P
  - Total: 3P (1.5× more than DDP)

Trade-off: More communication for less memory
```

### 6.5 FSDP in PyTorch

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

# Sharding strategies
# FULL_SHARD: Shard params, grads, optimizer (most memory efficient)
# SHARD_GRAD_OP: Shard grads + optimizer only (less communication)
# NO_SHARD: DDP mode

# Mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

# Wrap model
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
)
```

### 6.6 FSDP Wrapping Strategies

```python
# Auto wrap based on size
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=1e6,  # Wrap layers > 1M params
)

# Or wrap specific modules
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

---

## 7. ZeRO Optimization

### 7.1 ZeRO Stages

DeepSpeed ZeRO (Zero Redundancy Optimizer):

```
ZeRO-1: Shard optimizer states
  Memory: P × 4 (params) + P × 4 (grads) + P/N × 8 (optimizer)
  Reduction: ~4× for Adam

ZeRO-2: Shard optimizer + gradients
  Memory: P × 4 (params) + P/N × 4 (grads) + P/N × 8 (optimizer)
  Reduction: ~8× for Adam

ZeRO-3: Shard everything (params + grads + optimizer)
  Memory: P/N × 4 + P/N × 4 + P/N × 8 = P/N × 16
  Reduction: ~N×

ZeRO-3 ≈ FSDP
```

### 7.2 ZeRO-Offload

Move data to CPU when not needed:
```
ZeRO-Offload:
  - Optimizer states on CPU
  - Gradients on CPU
  - Parameters on GPU

ZeRO-Infinity:
  - Offload to NVMe SSD
  - Enables training models larger than GPU+CPU memory
```

### 7.3 DeepSpeed Configuration

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5
    }
}
```

### 7.4 ZeRO vs FSDP Comparison

| Feature | ZeRO-3 | FSDP |
|---------|--------|------|
| Framework | DeepSpeed | PyTorch native |
| Sharding | All | All |
| CPU offload | Yes | Yes (PyTorch 2.1+) |
| NVMe offload | Yes | No |
| Activation checkpointing | Built-in | Separate |
| Ease of use | Config file | Python API |

---

## 8. Practical Considerations

### 8.1 Choosing a Strategy

```
Model fits on 1 GPU with room:
  → Single GPU training

Model fits on 1 GPU (barely):
  → Gradient checkpointing + Mixed precision

Model needs multiple GPUs (memory):
  → FSDP or ZeRO-3

Model needs multiple GPUs (speed):
  → DDP

Very large model (>100B):
  → Tensor + Pipeline + Data parallelism (3D)
```

### 8.2 Multi-Node Training

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py
```

### 8.3 Debugging Distributed Training

```python
# Check ranks
print(f"Global rank: {dist.get_rank()}")
print(f"Local rank: {local_rank}")
print(f"World size: {dist.get_world_size()}")

# Synchronize for debugging
dist.barrier()
if dist.get_rank() == 0:
    print("Only master prints this")

# Save checkpoints only on rank 0
if dist.get_rank() == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')
dist.barrier()  # Wait for save to complete
```

### 8.4 Common Issues

```
1. Hanging:
   - Barrier mismatch (not all ranks reach barrier)
   - Different control flow based on data
   - Fix: Ensure identical control flow

2. NCCL timeout:
   - Slow GPU or network issue
   - Fix: Set NCCL_DEBUG=INFO for diagnosis
   - Increase timeout: dist.init_process_group(..., timeout=timedelta(minutes=30))

3. OOM on some GPUs:
   - Uneven data distribution
   - Fix: Use DistributedSampler properly

4. Gradient explosion with scaling:
   - Learning rate too high for effective batch
   - Fix: Scale LR with sqrt(batch_size) or linear with warmup
```

### 8.5 Checkpointing with FSDP

```python
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

# Save full checkpoint (gather on rank 0)
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, 'checkpoint.pt')

# Or save sharded (each rank saves its shard)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
    torch.save(state_dict, f'checkpoint_rank{rank}.pt')
```

---

## 9. Interview Questions

### Q1: Explain DDP and how it differs from DataParallel.

**Answer**:

**DataParallel (DP)**:
```
- Single process, multiple threads
- Model replicated to all GPUs
- GIL limits parallelism
- All gradient reduction on GPU 0
- Memory imbalanced (GPU 0 does extra work)
```

**Distributed Data Parallel (DDP)**:
```
- Multi-process (one per GPU)
- No GIL bottleneck
- Ring all-reduce for gradients
- Gradient bucketing for efficiency
- Overlaps backward computation with communication
```

**Key DDP optimizations**:
1. **Bucketing**: Group small gradients for single all-reduce
2. **Overlap**: Start all-reduce while still computing gradients
3. **No redundant broadcast**: Each process has identical model

**Code difference**:
```python
# DP (not recommended)
model = nn.DataParallel(model)

# DDP (recommended)
dist.init_process_group('nccl')
model = DDP(model, device_ids=[local_rank])
```

### Q2: What is FSDP and when would you use it?

**Answer**:

**FSDP** = Fully Sharded Data Parallel

**Concept**: Shard model parameters, gradients, and optimizer states across GPUs.

**Operations**:
```
Forward:
  1. All-gather parameters for current layer
  2. Compute forward
  3. Free gathered parameters

Backward:
  1. All-gather parameters
  2. Compute gradients
  3. Reduce-scatter gradients
  4. Free gathered parameters
```

**Memory comparison** (N GPUs, P parameters):
```
DDP:  P × 16 bytes per GPU
FSDP: P/N × 16 bytes per GPU
```

**When to use**:
- Model doesn't fit on single GPU (with DDP)
- Need to train larger batch sizes
- Memory is the bottleneck, not communication

**Trade-off**: 1.5× more communication than DDP.

### Q3: Explain tensor parallelism vs pipeline parallelism.

**Answer**:

**Tensor Parallelism**:
```
Split individual layers across GPUs

Example (Linear: Y = XW):
  Column split: W = [W1 | W2]
    GPU 0: Y1 = X @ W1
    GPU 1: Y2 = X @ W2

Communication: All-reduce after each split layer
Best for: Large layers, high-bandwidth interconnect (NVLink)
```

**Pipeline Parallelism**:
```
Split model sequentially into stages

Stage 0: Layers 0-7
Stage 1: Layers 8-15
...

Communication: Point-to-point activation transfer
Problem: Pipeline bubble (idle time)
Solution: Micro-batching, 1F1B schedule
```

**Comparison**:
| Aspect | Tensor | Pipeline |
|--------|--------|----------|
| Communication | All-reduce per layer | Point-to-point |
| Memory | Activations split | Full activations per stage |
| Bubble | None | Yes (mitigated by micro-batching) |
| Granularity | Within layer | Between layers |

### Q4: What are ZeRO stages and how do they differ?

**Answer**:

**ZeRO** = Zero Redundancy Optimizer (DeepSpeed)

**ZeRO-1**: Shard optimizer states only
```
Per GPU: Full params + Full grads + 1/N optimizer
Memory reduction: ~4× (for Adam's m, v)
Communication: Same as DDP
```

**ZeRO-2**: Shard optimizer + gradients
```
Per GPU: Full params + 1/N grads + 1/N optimizer
Memory reduction: ~8×
Communication: Reduce-scatter for grads
```

**ZeRO-3**: Shard everything
```
Per GPU: 1/N params + 1/N grads + 1/N optimizer
Memory reduction: N×
Communication: All-gather params in forward/backward
```

**When to use each**:
- ZeRO-1: Easy win, no code changes
- ZeRO-2: Need more memory, tolerate some overhead
- ZeRO-3: Model doesn't fit otherwise (≈ FSDP)

### Q5: How would you debug a hanging distributed training job?

**Answer**:

**Common causes and solutions**:

1. **Barrier mismatch**:
```python
# BAD: Different control flow
if rank == 0:
    dist.barrier()  # Only rank 0 reaches this

# GOOD: All ranks execute same code
dist.barrier()
if rank == 0:
    print("After barrier")
```

2. **Data-dependent control flow**:
```python
# BAD: Batch size might differ
if len(batch) > 0:
    loss.backward()  # Some ranks skip

# GOOD: Ensure identical batches
sampler = DistributedSampler(dataset, drop_last=True)
```

3. **Debugging tools**:
```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Increase timeout
dist.init_process_group(..., timeout=timedelta(minutes=60))

# Check which operation hangs
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

4. **Systematic approach**:
- Add barriers with prints to find where it hangs
- Check all ranks have same tensor shapes
- Verify network connectivity between nodes
- Check for GPU errors: `nvidia-smi -q`

---

## 10. Summary

### Strategy Selection

| Scenario | Strategy |
|----------|----------|
| Model fits on 1 GPU | Single GPU + AMP |
| Need faster training | DDP |
| Model barely fits (memory) | DDP + Gradient Checkpointing |
| Model doesn't fit | FSDP or ZeRO-3 |
| Very large model (>100B) | Tensor + Pipeline + Data (3D) |

### Communication Costs

| Method | Communication Volume |
|--------|---------------------|
| DDP | 2P (all-reduce) |
| FSDP | 3P (all-gather + reduce-scatter) |
| Tensor Parallel | O(batch × seq × hidden) per layer |
| Pipeline Parallel | O(batch × seq × hidden) per stage |

### Memory per GPU

| Method | Params | Grads | Optimizer |
|--------|--------|-------|-----------|
| DDP | P | P | P × 2 (Adam) |
| ZeRO-1 | P | P | P/N × 2 |
| ZeRO-2 | P | P/N | P/N × 2 |
| ZeRO-3/FSDP | P/N | P/N | P/N × 2 |

### Key Takeaways

1. **DDP is the starting point**: Multi-process, efficient, easy to use
2. **FSDP/ZeRO-3 for memory**: When model doesn't fit with DDP
3. **Tensor parallelism for speed**: When you have fast interconnect
4. **Pipeline parallelism for very large models**: Handle bubble with micro-batching
5. **Always profile**: Communication vs computation balance varies
