# Module 10.1: Modern LLM Architectures - LLaMA, RoPE, GQA

## Table of Contents
1. [Evolution of LLM Architectures](#1-evolution-of-llm-architectures)
2. [Rotary Position Embeddings (RoPE)](#2-rotary-position-embeddings-rope)
3. [Grouped Query Attention (GQA)](#3-grouped-query-attention-gqa)
4. [SwiGLU Activation](#4-swiglu-activation)
5. [RMSNorm](#5-rmsnorm)
6. [LLaMA Architecture](#6-llama-architecture)
7. [Other Modern Architectures](#7-other-modern-architectures)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Evolution of LLM Architectures

### 1.1 From GPT to Modern LLMs

```
GPT-2 (2019)         → LLaMA (2023)
------------------------------------------
Post-LayerNorm       → Pre-LayerNorm
LayerNorm            → RMSNorm
Learned PE           → RoPE
Multi-Head Attention → Grouped Query Attention
GELU activation      → SwiGLU activation
Standard FFN         → SwiGLU FFN
```

### 1.2 Key Innovations

| Innovation | Problem Solved | Used In |
|------------|---------------|---------|
| RoPE | Better length generalization | LLaMA, Mistral |
| GQA/MQA | Reduce KV-cache memory | LLaMA-2, Gemma |
| SwiGLU | More expressive FFN | LLaMA, PaLM |
| RMSNorm | Faster normalization | LLaMA, Gemma |
| Pre-Norm | Stable training | All modern LLMs |

### 1.3 Why These Changes Matter

**Memory efficiency**:
- GQA reduces KV-cache by 4-8x
- Critical for long-context inference

**Training stability**:
- RMSNorm + Pre-Norm = more stable
- Can train deeper models without warmup

**Length generalization**:
- RoPE extrapolates to longer sequences
- No retraining needed for longer context

---

## 2. Rotary Position Embeddings (RoPE)

### 2.1 The Problem with Learned PE

Learned position embeddings:
```
PE = nn.Embedding(max_len, d_model)
x = token_emb + PE[positions]
```

**Limitations**:
- Can't generalize beyond trained max_len
- Absolute positions, not relative
- No inherent structure

### 2.2 RoPE Key Idea

**Encode position through rotation**:
```
Instead of: x + PE
Do: rotate(x, θ × position)

Key insight:
  dot(rotate(q, θ×m), rotate(k, θ×n)) depends on (m - n)
  → Captures RELATIVE position naturally!
```

### 2.3 RoPE Formula

For 2D subspace (pair of dimensions):
```
[cos(mθ)  -sin(mθ)] [q_1]
[sin(mθ)   cos(mθ)] [q_2]

Where:
  m = position in sequence
  θ = frequency (different for each dimension pair)
```

**Full formula**:
```
RoPE(x_m, m) = R_Θ,m × x_m

where R_Θ,m is rotation matrix with angles θ_i × m

θ_i = base^(-2i/d)  (base = 10000 typically)
```

### 2.4 Why Rotation Encodes Relative Position

```
q_m · k_n = (R_m q) · (R_n k)
         = q^T R_m^T R_n k
         = q^T R_{n-m} k

The dot product only depends on (n - m) = relative position!
```

### 2.5 RoPE Benefits

| Benefit | Explanation |
|---------|-------------|
| Relative positions | Attention depends on (m-n), not absolute m, n |
| Length extrapolation | Can extend to longer sequences |
| No learned params | Just rotation based on position |
| Efficient | Applied only to Q, K (not V) |

### 2.6 Extending RoPE (Position Interpolation)

To use longer contexts than trained:
```
Original: θ × m
Extended: θ × (m × old_len / new_len)

Example: Trained on 2048, want 8192
  m = 5000 → m' = 5000 × (2048 / 8192) = 1250
```

This is called **Position Interpolation (PI)** or **NTK-aware scaling**.

---

## 3. Grouped Query Attention (GQA)

### 3.1 The KV-Cache Problem

Standard multi-head attention (MHA):
```
Heads: 32
d_model: 4096
d_k = d_v = 128

KV-cache per token:
  32 heads × 128 dims × 2 (K, V) = 8192 values
  × 4 bytes = 32 KB per token per layer

For 32 layers, 4096 context:
  32 KB × 32 × 4096 = 4 GB just for KV-cache!
```

### 3.2 Multi-Query Attention (MQA)

**Single shared K, V for all heads**:
```
Standard MHA:      32 Q heads, 32 K heads, 32 V heads
MQA:               32 Q heads, 1 K head, 1 V head

KV-cache reduction: 32x
```

**Trade-off**: Some quality loss due to less expressive K, V.

### 3.3 Grouped Query Attention (GQA)

**Compromise between MHA and MQA**:
```
Group queries into G groups
Each group shares K, V

Example: 32 Q heads, 8 K/V heads (4 groups)
  Q heads 0-3 share K_0, V_0
  Q heads 4-7 share K_1, V_1
  ...
```

### 3.4 GQA Configurations

```
              Q heads    K/V heads    KV ratio
MHA:            32          32          1x
GQA-8:          32           8          4x
GQA-4:          32           4          8x
MQA:            32           1         32x
```

**LLaMA-2 uses GQA-8** (8 K/V heads for 32 Q heads)

### 3.5 GQA Implementation

```python
# Q: (batch, n_heads, seq, d_k)
# K, V: (batch, n_kv_heads, seq, d_k)

# Repeat K, V to match Q heads
n_rep = n_heads // n_kv_heads
K = K.repeat_interleave(n_rep, dim=1)
V = V.repeat_interleave(n_rep, dim=1)

# Standard attention
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
output = softmax(scores) @ V
```

### 3.6 GQA Benefits

| Aspect | Benefit |
|--------|---------|
| Memory | 4-8x reduction in KV-cache |
| Speed | Faster inference |
| Quality | Minimal degradation vs MHA |
| Batch size | Can fit larger batches |

---

## 4. SwiGLU Activation

### 4.1 Standard FFN

```
FFN(x) = W_2 × GELU(W_1 × x)

Input → Linear → GELU → Linear → Output
  d        d×4f            d
```

### 4.2 GLU (Gated Linear Unit)

**Idea**: Gate the activation:
```
GLU(x) = σ(W_1 × x) ⊙ (W_2 × x)

One path is sigmoid gate, other is linear
Output is element-wise product
```

### 4.3 SwiGLU Formula

**SwiGLU** = Swish + GLU:
```
SwiGLU(x) = Swish(W_gate × x) ⊙ (W_1 × x)
          = (W_gate × x × σ(W_gate × x)) ⊙ (W_1 × x)

Full FFN with SwiGLU:
  FFN(x) = W_2 × SwiGLU(x)
         = W_2 × (Swish(W_gate × x) ⊙ (W_1 × x))
```

### 4.4 Parameter Count

```
Standard FFN:
  W_1: d × 4d
  W_2: 4d × d
  Total: 8d²

SwiGLU FFN:
  W_1: d × (8d/3)
  W_gate: d × (8d/3)
  W_2: (8d/3) × d
  Total: 8d² (same, but different distribution)

Note: Hidden dim is 8d/3 (not 4d) to keep param count similar
```

### 4.5 Why SwiGLU Works Better

**Hypothesis**:
- Gating provides multiplicative interactions
- More expressive than simple activation
- Swish (SiLU) is smooth unlike ReLU

**Empirical**: PaLM showed 5-10% improvement on benchmarks.

---

## 5. RMSNorm

### 5.1 LayerNorm Review

```
LayerNorm(x) = γ × (x - μ) / σ + β

where:
  μ = mean(x)
  σ = sqrt(var(x) + ε)
  γ, β = learned parameters
```

### 5.2 RMSNorm Formula

**Remove mean centering**:
```
RMSNorm(x) = γ × x / RMS(x)

where:
  RMS(x) = sqrt(mean(x²) + ε)
```

**No subtraction of mean, no β parameter.**

### 5.3 Why RMSNorm?

**Empirical finding**: Mean centering not necessary for good performance.

**Benefits**:
- Faster (no mean computation)
- Fewer parameters (no β)
- ~10-20% speedup vs LayerNorm
- Same quality in practice

### 5.4 Comparison

| Aspect | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Mean subtraction | Yes | No |
| Parameters | 2d (γ, β) | d (γ only) |
| Operations | Mean + Var + Norm | Square + Mean + Norm |
| Speed | Slower | ~15% faster |
| Usage | BERT, GPT-2 | LLaMA, Gemma |

---

## 6. LLaMA Architecture

### 6.1 LLaMA Configurations

| Model | Layers | d_model | Heads | KV Heads | d_ff |
|-------|--------|---------|-------|----------|------|
| LLaMA-7B | 32 | 4096 | 32 | 32 | 11008 |
| LLaMA-13B | 40 | 5120 | 40 | 40 | 13824 |
| LLaMA-33B | 60 | 6656 | 52 | 52 | 17920 |
| LLaMA-65B | 80 | 8192 | 64 | 64 | 22016 |

**LLaMA-2** (with GQA):
| Model | Layers | d_model | Q Heads | KV Heads |
|-------|--------|---------|---------|----------|
| LLaMA-2-7B | 32 | 4096 | 32 | 32 |
| LLaMA-2-13B | 40 | 5120 | 40 | 40 |
| LLaMA-2-70B | 80 | 8192 | 64 | 8 |

### 6.2 LLaMA Block

```python
def llama_block(x, freqs_cis):
    # Pre-RMSNorm attention
    h = x + attention(rmsnorm(x), freqs_cis)

    # Pre-RMSNorm FFN
    out = h + swiglu_ffn(rmsnorm(h))

    return out
```

### 6.3 Full Architecture

```
Input tokens
    ↓
Token Embedding (no position embedding added here!)
    ↓
┌──────────────────────────────────────┐
│  RMSNorm                             │
│       ↓                              │
│  RoPE Self-Attention (GQA)           │  × N layers
│       ↓                              │
│  + Residual                          │
│       ↓                              │
│  RMSNorm                             │
│       ↓                              │
│  SwiGLU FFN                          │
│       ↓                              │
│  + Residual                          │
└──────────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
LM Head (Linear)
    ↓
Logits
```

### 6.4 Key Differences from GPT-2

| Component | GPT-2 | LLaMA |
|-----------|-------|-------|
| Position | Learned absolute | RoPE |
| Normalization | LayerNorm | RMSNorm |
| Norm position | Pre-norm | Pre-norm |
| Activation | GELU | SiLU/Swish |
| FFN | Standard | SwiGLU |
| Attention | MHA | GQA (LLaMA-2-70B) |
| Context length | 1024 | 2048/4096 |

### 6.5 LLaMA Training Details

```
Training tokens: 1-1.4T tokens (LLaMA-1)
                 2T tokens (LLaMA-2)

Data: CommonCrawl, Wikipedia, Books, Code, etc.

Optimizer: AdamW
  β1 = 0.9
  β2 = 0.95
  weight_decay = 0.1

LR schedule:
  Warmup: 2000 steps
  Cosine decay to 10% of peak

Batch size: 4M tokens
Context: 2048 tokens (LLaMA-1), 4096 (LLaMA-2)
```

---

## 7. Other Modern Architectures

### 7.1 Mistral

**Key innovations**:
- Sliding Window Attention
- Rolling Buffer KV-Cache

```
Sliding Window (size W):
  Position i attends to [max(0, i-W), i]

Rolling Buffer:
  Cache[i % W] = KV_i
  Memory: O(W) instead of O(seq_len)
```

### 7.2 Mixtral (Mixture of Experts)

**Sparse MoE**:
```
FFN(x) = Σ_i gate_i(x) × Expert_i(x)

Only top-k experts are activated per token
E.g., 8 experts, top-2 routing → 2/8 = 25% active
```

**Benefits**:
- More parameters with same compute
- Specialization of experts

### 7.3 Gemma

**Google's open LLM**:
- Based on Gemini architecture
- Uses GQA, RMSNorm, GeGLU
- Multi-Query Attention for 2B model

### 7.4 Architecture Comparison

| Model | RoPE | GQA | FFN | Norm | Context |
|-------|------|-----|-----|------|---------|
| LLaMA-2 | Yes | Yes (70B) | SwiGLU | RMSNorm | 4096 |
| Mistral | Yes | Yes | SwiGLU | RMSNorm | 8192 (sliding) |
| Mixtral | Yes | Yes | MoE | RMSNorm | 32768 |
| Gemma | Yes | Yes | GeGLU | RMSNorm | 8192 |

---

## 8. Interview Questions

### Q1: Explain Rotary Position Embeddings (RoPE).

**Answer**:

**Problem**: Learned position embeddings:
- Can't generalize beyond max trained length
- Don't inherently encode relative positions

**RoPE Solution**: Encode position through rotation:
```
RoPE(q_m) = R_m × q_m

where R_m is rotation matrix with angles θ × m
```

**Key Property**:
```
q_m · k_n = (R_m × q)^T × (R_n × k)
         = q^T × R_{m-n} × k

Dot product depends only on relative position (m-n)!
```

**Benefits**:
- Natural relative position encoding
- Extrapolates to longer sequences
- No learned parameters
- Applied to Q, K only (not V)

### Q2: What is Grouped Query Attention (GQA)?

**Answer**:

**Problem**: KV-cache memory grows with:
- Sequence length
- Number of layers
- Number of heads

For long sequences, this becomes prohibitive.

**MHA**: Each query head has its own K, V
```
Q: 32 heads, K: 32 heads, V: 32 heads
```

**GQA**: Multiple query heads share K, V
```
Q: 32 heads, K: 8 heads, V: 8 heads
Queries 0-3 share K_0, V_0
Queries 4-7 share K_1, V_1
...
```

**Benefits**:
- 4x reduction in KV-cache (for 32→8)
- Minimal quality loss
- Faster inference

**Used in**: LLaMA-2-70B, Mistral, Gemma

### Q3: Explain SwiGLU activation.

**Answer**:

**Standard FFN**:
```
FFN(x) = W_2 × GELU(W_1 × x)
```

**SwiGLU FFN**:
```
SwiGLU(x) = Swish(W_gate × x) ⊙ (W_1 × x)
FFN(x) = W_2 × SwiGLU(x)
```

**Components**:
- **Swish**: x × sigmoid(x) - smooth activation
- **GLU**: Gating mechanism (element-wise product)
- **SwiGLU**: Swish-gated linear unit

**Why better**:
- Multiplicative interactions through gating
- More expressive than simple activation
- ~5-10% improvement on benchmarks

**Parameter adjustment**:
- Hidden dim = 8d/3 (not 4d) to match param count
- Three weight matrices instead of two

### Q4: Compare LayerNorm and RMSNorm.

**Answer**:

**LayerNorm**:
```
y = γ × (x - mean(x)) / std(x) + β
```
- Subtracts mean (centering)
- Has bias parameter β
- 2d parameters (γ and β)

**RMSNorm**:
```
y = γ × x / RMS(x)
RMS(x) = sqrt(mean(x²))
```
- No mean subtraction
- No bias parameter
- d parameters (γ only)

**Why RMSNorm**:
- ~15% faster (fewer operations)
- Fewer parameters
- Empirically equivalent quality
- Used in all modern LLMs (LLaMA, Mistral, Gemma)

### Q5: What are the key components of the LLaMA architecture?

**Answer**:

**LLaMA architecture**:
```
1. Token Embedding (no position added)
2. N × Transformer Blocks:
   - Pre-RMSNorm
   - RoPE Self-Attention (GQA in 70B)
   - Residual connection
   - Pre-RMSNorm
   - SwiGLU FFN
   - Residual connection
3. Final RMSNorm
4. LM Head
```

**Key innovations**:
- **RoPE**: Rotary position embeddings for relative positions
- **RMSNorm**: Faster normalization
- **SwiGLU**: Gated FFN for expressiveness
- **GQA**: Reduced KV-cache (LLaMA-2-70B)
- **Pre-norm**: Stable training

**Differences from GPT-2**:
| GPT-2 | LLaMA |
|-------|-------|
| Learned PE | RoPE |
| LayerNorm | RMSNorm |
| GELU | SiLU |
| Standard FFN | SwiGLU |
| MHA | GQA (70B) |

---

## 9. Summary

### Key Innovations

| Innovation | Benefit |
|------------|---------|
| RoPE | Relative positions, length extrapolation |
| GQA | 4-8x KV-cache reduction |
| SwiGLU | More expressive FFN |
| RMSNorm | ~15% faster normalization |
| Pre-norm | Stable training |

### LLaMA Recipe

```
Position encoding: RoPE
Normalization: RMSNorm (pre-norm)
Attention: Multi-head or GQA
FFN: SwiGLU (hidden = 8d/3)
Activation: SiLU (Swish)
Context: 2048-4096 tokens
```

### Key Equations

**RoPE**:
```
RoPE(x, m) = R_Θ,m × x
where R applies rotation by θ_i × m for each dimension pair
```

**SwiGLU**:
```
SwiGLU(x) = SiLU(x × W_gate) ⊙ (x × W_1)
FFN(x) = W_2 × SwiGLU(x)
```

**RMSNorm**:
```
RMSNorm(x) = x / RMS(x) × γ
RMS(x) = sqrt(mean(x²))
```

### Key Takeaways

1. **RoPE is standard**: All modern LLMs use rotary embeddings
2. **GQA for efficiency**: Critical for long-context inference
3. **SwiGLU for quality**: Gating improves FFN expressiveness
4. **RMSNorm for speed**: Simpler normalization works just as well
5. **Pre-norm always**: More stable than post-norm
