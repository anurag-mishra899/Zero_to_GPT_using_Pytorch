# Module 11.1: Efficient Attention - Flash Attention & KV-Cache

## Table of Contents
1. [The Attention Bottleneck](#1-the-attention-bottleneck)
2. [Flash Attention](#2-flash-attention)
3. [KV-Cache Optimizations](#3-kv-cache-optimizations)
4. [Sliding Window Attention](#4-sliding-window-attention)
5. [Other Efficient Attention Methods](#5-other-efficient-attention-methods)
6. [Speculative Decoding](#6-speculative-decoding)
7. [Interview Questions](#7-interview-questions)
8. [Summary](#8-summary)

---

## 1. The Attention Bottleneck

### 1.1 Memory Complexity

Standard attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Memory for attention matrix: O(n²)
  - Q × K^T produces (seq_len × seq_len) matrix
  - For seq_len = 4096, d = 128: 4096² × 4 bytes = 64 MB per head!
  - With 32 heads: 2 GB just for attention matrices
```

### 1.2 Compute Complexity

```
Compute: O(n² × d)
  - Matrix multiply: Q(n×d) × K^T(d×n) = n² × d ops
  - Softmax: n² ops
  - Value multiply: n² × d ops

Scaling:
  - seq_len = 1024 → 1M ops per head
  - seq_len = 4096 → 16M ops per head
  - seq_len = 32768 → 1B ops per head
```

### 1.3 The Memory Bandwidth Problem

**Modern GPUs are memory-bound, not compute-bound**:
```
A100 GPU:
  - Compute: 312 TFLOPS (BF16)
  - HBM bandwidth: 2 TB/s
  - HBM size: 80 GB

For attention:
  - Read Q, K: 2 × n × d × 2 bytes
  - Write attention matrix: n² × 2 bytes
  - Read attention matrix + V: n² + n×d × 2 bytes
  - Write output: n × d × 2 bytes

Multiple reads/writes to HBM (slow!) for intermediate results
```

### 1.4 IO Complexity

```
Standard attention IO: O(n² + nd)
  - Write/read attention matrix (n²) dominates

Flash Attention IO: O(nd)
  - Never materializes full attention matrix
  - Huge speedup for long sequences
```

---

## 2. Flash Attention

### 2.1 Key Insight

**Don't materialize the full attention matrix**:
```
Standard:
1. Compute S = QK^T          (n² memory)
2. Compute P = softmax(S)    (n² memory)
3. Compute O = PV            (read n² from memory)

Flash Attention:
1. Process in blocks
2. Compute attention incrementally
3. Never store full n² matrix
```

### 2.2 The Algorithm

**Tiling + Online Softmax**:
```
For each block of Q (size B_q):
  For each block of K, V (size B_kv):
    1. Load Q_block, K_block, V_block from HBM to SRAM
    2. Compute S_block = Q_block × K_block^T (in SRAM)
    3. Update running softmax statistics (online algorithm)
    4. Update output O_block incrementally
  Store O_block to HBM
```

### 2.3 Online Softmax

**Problem**: Softmax requires knowing all values first:
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

**Online solution** (process incrementally):
```
For new block b with values x_b:
  m_new = max(m_old, max(x_b))           # Update max
  l_new = l_old × exp(m_old - m_new) +   # Rescale old sum
          sum(exp(x_b - m_new))           # Add new
  O_new = O_old × (l_old/l_new) × exp(m_old - m_new) +
          softmax_local(x_b) × V_b × ...  # Rescale output
```

### 2.4 Memory Savings

```
Standard attention:
  Memory: O(n²) for attention matrix

Flash Attention:
  Memory: O(n) - only block-sized matrices in SRAM
  SRAM usage: O(B_q × B_kv) where B << n

Example (seq_len = 8192, head_dim = 128):
  Standard: 8192² × 4 = 256 MB per head
  Flash: ~100 KB per head (block size dependent)
```

### 2.5 Flash Attention v2 Improvements

```
v1 → v2:
  1. Better parallelism (parallelize over seq_len, not just batch/heads)
  2. Reduced register spilling
  3. Better work partitioning between warps
  4. ~2x speedup over v1
```

### 2.6 Usage in PyTorch

```python
import torch.nn.functional as F

# PyTorch 2.0+ has built-in Flash Attention
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,  # For causal masking
    scale=None
)

# Automatically uses Flash Attention when available
# Falls back to standard attention otherwise
```

---

## 3. KV-Cache Optimizations

### 3.1 Basic KV-Cache

During autoregressive generation:
```
Without cache:
  Step 1: Process [t1]
  Step 2: Process [t1, t2]
  Step 3: Process [t1, t2, t3]
  ...
  Redundant computation!

With cache:
  Step 1: Process [t1], cache K1, V1
  Step 2: Process [t2], use cached K1,V1, cache K2,V2
  Step 3: Process [t3], use cached K1,V1,K2,V2, cache K3,V3
```

### 3.2 KV-Cache Memory Problem

```
Per token, per layer:
  K: n_kv_heads × head_dim × 2 bytes (FP16)
  V: n_kv_heads × head_dim × 2 bytes

For LLaMA-70B (8 KV heads, head_dim=128, 80 layers):
  Per token: 8 × 128 × 2 × 2 × 80 = 327 KB

For 4096 context:
  4096 × 327 KB = 1.3 GB per sequence

For batch of 16:
  16 × 1.3 GB = 21 GB just for KV-cache!
```

### 3.3 Paged Attention (vLLM)

**Problem**: KV-cache wastes memory with fragmentation.

**Solution**: Manage KV-cache like virtual memory:
```
Instead of:
  Allocate contiguous [max_seq_len × kv_size] per sequence

Do:
  Allocate fixed-size blocks (e.g., 16 tokens)
  Use page table to map logical → physical blocks
  Allocate blocks on demand
```

**Benefits**:
- No memory waste from max_len allocation
- Efficient memory sharing (copy-on-write for beam search)
- Better GPU memory utilization

### 3.4 Continuous Batching

```
Standard batching:
  Wait for all sequences in batch to complete
  New requests queue up

Continuous batching:
  Process multiple requests simultaneously
  When one finishes, immediately add new request
  Much better GPU utilization
```

### 3.5 Multi-Query Attention (MQA) & GQA

Reduce KV-cache by sharing:
```
MHA: 32 Q heads, 32 K heads, 32 V heads → full cache
GQA: 32 Q heads, 8 K heads, 8 V heads → 4x reduction
MQA: 32 Q heads, 1 K head, 1 V head → 32x reduction
```

---

## 4. Sliding Window Attention

### 4.1 Concept

**Limit attention to local window**:
```
Standard: Position i attends to all positions [0, i]
Sliding Window: Position i attends to [max(0, i-W), i]
  where W = window size
```

### 4.2 Memory Benefits

```
Standard: O(n²) attention matrix
Sliding Window: O(n × W) where W << n

Example: n=32768, W=4096
  Standard: 32768² = 1B elements
  Sliding: 32768 × 4096 = 134M elements (8x reduction)
```

### 4.3 Rolling Buffer Cache

```
Instead of caching all past KV:
  Cache only last W tokens
  Cache[(i mod W)] = KV_i

Memory: O(W) instead of O(n)
```

### 4.4 Effective Context

**How sliding window still captures long context**:
```
Layer 1: Token i sees [i-W, i]
Layer 2: Token i effectively sees [i-2W, i]
...
Layer L: Token i effectively sees [i-L×W, i]

For L=32, W=4096: Effective context = 131K tokens!
```

### 4.5 Used In

```
Mistral-7B: W=4096, 32 layers → 128K effective context
Mistral uses both sliding window + full attention layers
```

---

## 5. Other Efficient Attention Methods

### 5.1 Linear Attention

Replace softmax with kernel approximation:
```
Standard: Attention = softmax(QK^T) × V

Linear: Attention = φ(Q) × (φ(K)^T × V)
  where φ is feature map

Complexity: O(n × d²) instead of O(n² × d)
```

**Examples**: Linear Transformer, Performer, RWKV

### 5.2 Sparse Attention

Only compute attention for subset of positions:
```
Patterns:
- Local: Attend to nearby positions
- Strided: Attend to every k-th position
- Random: Random sparse pattern
- Learned: Learn which positions to attend

Examples: Longformer, BigBird
```

### 5.3 ALiBi (Attention with Linear Biases)

**No position embeddings, bias attention scores**:
```
Standard: Attention(Q, K, V) with position embedding added to input
ALiBi: Attention(Q, K, V) with position bias added to scores

bias[i, j] = -m × |i - j|
  where m is head-specific slope

Simpler, better length extrapolation
```

**Used in**: BLOOM, MPT

### 5.4 Comparison

| Method | Complexity | Memory | Quality | Training |
|--------|------------|--------|---------|----------|
| Standard | O(n²d) | O(n²) | Best | Easy |
| Flash Attention | O(n²d) | O(n) | Same | Easy |
| Sliding Window | O(nWd) | O(nW) | Good | Easy |
| Linear | O(nd²) | O(nd) | Lower | Harder |
| Sparse | O(nkd) | O(nk) | Good | Complex |

---

## 6. Speculative Decoding

### 6.1 The Problem

Autoregressive generation is slow:
```
Each token requires:
  1. Full forward pass through large model
  2. Memory bandwidth bound
  3. Can't parallelize (depends on previous tokens)
```

### 6.2 Speculative Decoding Idea

Use small "draft" model to generate candidates:
```
1. Draft model generates k tokens quickly
2. Large model verifies all k tokens in parallel
3. Accept prefix that matches, reject rest
4. Repeat from last accepted position
```

### 6.3 Algorithm

```python
while not done:
    # Draft phase: small model generates k tokens
    draft_tokens = draft_model.generate(context, k)

    # Verify phase: large model scores all k tokens in parallel
    logits = large_model(context + draft_tokens)

    # Accept/reject
    for i in range(k):
        if accept(draft_tokens[i], logits[i]):
            context.append(draft_tokens[i])
        else:
            # Sample from large model for this position
            context.append(sample(logits[i]))
            break
```

### 6.4 Speedup

```
If draft model acceptance rate = p:
  Expected accepted tokens per step = 1/(1-p) - 1

Example: p = 0.8
  Expected tokens = 4 per large model call
  Speedup ≈ 2-3x (draft model overhead + verification)
```

### 6.5 Requirements

- Draft model must be much faster
- Should have similar distribution (high acceptance)
- Common: Use smaller version of same architecture

---

## 7. Interview Questions

### Q1: Explain Flash Attention and why it's faster.

**Answer**:

**Problem**: Standard attention is memory-bound, not compute-bound.
```
Standard attention:
1. Compute QK^T (n² elements) → Write to HBM
2. Compute softmax → Read/write n² to HBM
3. Multiply by V → Read n² from HBM

Multiple HBM transfers for O(n²) intermediate data
```

**Flash Attention Solution**:
```
1. Process attention in blocks (tiles)
2. Never materialize full n² attention matrix
3. Use online softmax to compute incrementally
4. Keep all intermediate results in fast SRAM
```

**Key techniques**:
- **Tiling**: Process Q, K, V in blocks that fit in SRAM
- **Online softmax**: Update softmax statistics incrementally
- **Recomputation**: In backward pass, recompute attention (cheaper than loading from HBM)

**Speedup**: 2-4x for long sequences due to reduced memory IO.

### Q2: What is KV-cache and how do you optimize it?

**Answer**:

**KV-cache**: Store computed key and value vectors during generation.
```
Without cache: Recompute K, V for all past tokens each step
With cache: Only compute K, V for new token, reuse cached values
```

**Optimizations**:

1. **GQA/MQA**: Reduce KV heads
   ```
   MHA: 32 KV heads → Full cache
   GQA: 8 KV heads → 4x reduction
   ```

2. **Paged Attention**: Virtual memory for KV-cache
   - Allocate blocks on demand
   - No fragmentation
   - Memory sharing for beam search

3. **Sliding Window**: Only cache last W tokens
   ```
   Memory: O(W) instead of O(n)
   ```

4. **Quantization**: Store KV in INT8
   ```
   FP16 → INT8 = 2x memory reduction
   ```

### Q3: Explain sliding window attention.

**Answer**:

**Concept**: Each position only attends to W previous positions (not all).
```
Position i attends to: [max(0, i-W), i]
Instead of: [0, i]
```

**Benefits**:
- Memory: O(n×W) instead of O(n²)
- Compute: O(n×W×d) instead of O(n²×d)
- Can use rolling buffer cache: O(W) memory

**Effective context**:
```
Layer 1: See [i-W, i]
Layer 2: See [i-2W, i] (through layer 1's output)
Layer L: See [i-LW, i]

With L=32, W=4096: Effective context = 128K tokens
```

**Used in**: Mistral-7B (W=4096)

### Q4: What is speculative decoding?

**Answer**:

**Problem**: Autoregressive generation is slow (one token at a time).

**Solution**: Use fast draft model to generate candidates, verify in parallel.

```python
# 1. Draft model generates k tokens quickly
draft = draft_model.generate(k_tokens)

# 2. Large model verifies all k tokens (parallel!)
probs = large_model.forward(all_k_tokens)

# 3. Accept matching prefix
for i in range(k):
    if draft[i] matches large_model:
        accept(draft[i])
    else:
        sample_from_large_model()
        break
```

**Key insight**: Verification is parallel, so we process k tokens in time of 1.

**Speedup**: 2-3x typical (depends on acceptance rate).

### Q5: Compare different efficient attention methods.

**Answer**:

| Method | How it works | Complexity | Trade-off |
|--------|-------------|------------|-----------|
| **Flash Attention** | Tiling + online softmax | O(n²) compute, O(n) memory | No quality loss, engineering complexity |
| **Sliding Window** | Local attention only | O(nW) | Limited direct context, uses stacked layers |
| **GQA/MQA** | Share K,V heads | O(n²) with reduced KV | Slight quality loss for MQA |
| **Linear Attention** | Kernel approximation | O(nd²) | Quality loss, training challenges |
| **Sparse Attention** | Subset of positions | O(nk) | Pattern design complexity |
| **ALiBi** | Bias instead of PE | O(n²) | Better extrapolation, simpler |

**When to use what**:
- Flash Attention: Always (no downside)
- GQA: Large models needing KV-cache reduction
- Sliding Window: Very long sequences
- Linear: Extreme sequence lengths (100K+)

---

## 8. Summary

### Key Optimizations

| Optimization | Target | Benefit |
|--------------|--------|---------|
| Flash Attention | Memory IO | 2-4x speedup, O(n) memory |
| KV-Cache | Generation | Avoid recomputation |
| GQA/MQA | KV-Cache size | 4-32x reduction |
| Paged Attention | Memory efficiency | No fragmentation |
| Sliding Window | Very long seq | O(nW) complexity |
| Speculative Decoding | Generation speed | 2-3x speedup |

### Key Equations

**Flash Attention Memory**:
```
Standard: O(n² + nd) HBM access
Flash: O(nd) HBM access
```

**KV-Cache Size**:
```
Per token = 2 × n_kv_heads × head_dim × n_layers × dtype_bytes
```

**Sliding Window Effective Context**:
```
Effective = n_layers × window_size
```

### Key Takeaways

1. **Flash Attention is standard**: Always use it (built into PyTorch 2.0+)
2. **KV-cache is critical**: Use GQA + paged attention for production
3. **Sliding window for long context**: Mistral uses W=4096
4. **Speculative decoding**: 2-3x generation speedup
5. **Memory bandwidth is the bottleneck**: Optimize for memory access, not FLOPs
