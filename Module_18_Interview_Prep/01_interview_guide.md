# Module 18: LLM Interview Preparation Guide

## Table of Contents
1. [Interview Structure](#1-interview-structure)
2. [Core Concepts Checklist](#2-core-concepts-checklist)
3. [Common Interview Questions](#3-common-interview-questions)
4. [System Design Questions](#4-system-design-questions)
5. [Coding Questions](#5-coding-questions)
6. [Behavioral Preparation](#6-behavioral-preparation)
7. [Quick Reference Sheets](#7-quick-reference-sheets)
8. [Final Preparation Tips](#8-final-preparation-tips)

---

## 1. Interview Structure

### 1.1 Typical LLM Interview Rounds

```
Round 1: Technical Screen (45-60 min)
  - ML fundamentals
  - LLM architecture basics
  - Simple coding

Round 2: Deep Dive (60-90 min)
  - Transformer details
  - Training techniques
  - Implementation questions

Round 3: System Design (60 min)
  - Design LLM serving system
  - Scale considerations
  - Trade-offs discussion

Round 4: Coding (45-60 min)
  - Implement attention
  - Debug training code
  - Optimize inference

Round 5: Behavioral/Culture (45 min)
  - Past projects
  - Collaboration style
  - Research direction
```

### 1.2 What Interviewers Look For

```
Technical Depth:
  - Can explain concepts clearly
  - Understands trade-offs
  - Knows implementation details

Problem Solving:
  - Structured approach
  - Asks clarifying questions
  - Considers edge cases

Communication:
  - Clear explanations
  - Can adjust to audience level
  - Admits when unsure

Research Awareness:
  - Knows recent papers
  - Understands trends
  - Has opinions on approaches
```

---

## 2. Core Concepts Checklist

### 2.1 Transformer Architecture (Essential)

```
□ Self-attention mechanism
  - Q, K, V projections
  - Scaled dot-product
  - Why sqrt(d_k) scaling
  - Softmax for weights

□ Multi-head attention
  - Why multiple heads
  - Concatenation + projection
  - Parameter count

□ Position encodings
  - Sinusoidal (original)
  - Learned embeddings
  - RoPE (modern)
  - ALiBi

□ Feed-forward network
  - Two linear layers
  - Activation functions
  - SwiGLU (modern)

□ Layer normalization
  - Pre-norm vs post-norm
  - RMSNorm

□ Residual connections
  - Gradient flow
  - Identity mapping
```

### 2.2 Training (Essential)

```
□ Pretraining
  - Next token prediction
  - Cross-entropy loss
  - Large-scale data

□ Supervised Fine-tuning
  - Instruction following
  - Demonstration data

□ Alignment (RLHF/DPO)
  - Reward modeling
  - PPO basics
  - DPO loss

□ Optimization
  - AdamW
  - Learning rate schedules
  - Gradient clipping
  - Warmup

□ Efficient training
  - Mixed precision (AMP)
  - Gradient checkpointing
  - Gradient accumulation
```

### 2.3 Scaling & Distributed (Important)

```
□ Data parallelism (DDP)
  - All-reduce gradients
  - Synchronized training

□ Model parallelism
  - Tensor parallelism
  - Pipeline parallelism

□ FSDP / ZeRO
  - Memory sharding
  - Communication patterns

□ Memory breakdown
  - Parameters
  - Gradients
  - Optimizer states
  - Activations
```

### 2.4 Inference Optimization (Important)

```
□ KV-cache
  - Purpose
  - Memory cost
  - GQA/MQA reduction

□ Batching strategies
  - Static vs continuous
  - Why continuous is better

□ PagedAttention
  - Block allocation
  - Memory efficiency

□ Speculative decoding
  - Draft + verify
  - Expected speedup

□ Quantization
  - INT8, INT4
  - GPTQ, AWQ
  - Quality vs speed
```

### 2.5 Modern Architectures (Good to Know)

```
□ Architectural choices
  - Pre-norm (modern)
  - Parallel attention + FFN
  - No bias terms

□ Position encodings
  - RoPE implementation
  - How it enables extrapolation

□ Attention variants
  - GQA (grouped query)
  - Sliding window
  - Flash Attention

□ Activation functions
  - SwiGLU
  - Why gating helps
```

---

## 3. Common Interview Questions

### 3.1 Fundamentals

**Q: Explain the transformer attention mechanism.**
```
Answer structure:
1. Q, K, V projections from input
2. Attention = softmax(QK^T / sqrt(d_k)) × V
3. sqrt(d_k) scaling prevents softmax saturation
4. Multi-head: Split into h heads, concat, project
5. O(n²) complexity for sequence length n
```

**Q: Why do transformers use layer normalization instead of batch norm?**
```
Answer:
1. Sequence lengths vary (padding issues)
2. LayerNorm normalizes across features, not batch
3. Works with single examples (inference)
4. RMSNorm: Simpler, just scale (no centering)
```

**Q: Explain the KV-cache and its memory implications.**
```
Answer:
1. During generation, cache K, V from past tokens
2. Avoid recomputing for each new token
3. Memory: layers × heads × head_dim × seq_len × 2 × bytes
4. For LLaMA-70B at 4096 tokens: ~1.3GB per sequence
5. Mitigation: GQA (fewer KV heads), quantization, paging
```

### 3.2 Training

**Q: Explain mixed precision training.**
```
Answer:
1. FP16/BF16 for forward/backward (2x faster, less memory)
2. FP32 master weights for updates (precision)
3. Loss scaling for FP16 (prevent underflow)
4. BF16 preferred (same range as FP32, no scaling needed)
5. Benefits: 2x memory, 2-4x speed, same accuracy
```

**Q: What is gradient checkpointing and when would you use it?**
```
Answer:
1. Trade compute for memory
2. Don't save all activations in forward
3. Recompute during backward
4. ~30% compute overhead for sqrt(N) memory
5. Use when: Model doesn't fit, need larger batch
```

**Q: How does DPO differ from RLHF?**
```
Answer:
RLHF:
  - Train reward model
  - Use PPO to optimize
  - Complex, multiple models

DPO:
  - Direct from preferences
  - No reward model needed
  - L = -log σ(β × (log_ratio_chosen - log_ratio_rejected))
  - Simpler, similar results
```

### 3.3 Scaling

**Q: Explain the difference between DDP and FSDP.**
```
Answer:
DDP:
  - Each GPU has full model
  - All-reduce gradients
  - Memory: full model per GPU

FSDP:
  - Shard params, grads, optimizer
  - All-gather params when needed
  - Reduce-scatter gradients
  - Memory: 1/N per GPU
  - 1.5x more communication
```

**Q: What is tensor parallelism?**
```
Answer:
1. Split layers across GPUs
2. Column parallel: Split weight columns
3. Row parallel: Split weight rows
4. For MLP: Column then Row (1 all-reduce)
5. For attention: Split heads across GPUs
6. Best with fast interconnect (NVLink)
```

### 3.4 Inference

**Q: What is PagedAttention?**
```
Answer:
1. KV-cache fragmentation problem
2. Treat cache like virtual memory
3. Fixed-size blocks (e.g., 16 tokens)
4. Page table maps logical → physical
5. Benefits: No waste, sharing, 2-4x more sequences
```

**Q: Explain speculative decoding.**
```
Answer:
1. Draft model generates k tokens quickly
2. Target model verifies all k in parallel
3. Accept matching prefix, sample from target at first mismatch
4. Speedup ≈ 1/(1-acceptance_rate)
5. Typical: 2-3x speedup with good draft model
```

---

## 4. System Design Questions

### 4.1 Design an LLM Serving System

```
Requirements:
- Handle 1000 requests/minute
- P99 latency < 2 seconds
- Support multiple model sizes
- Cost-effective

Components:
1. Load Balancer
   - Route by model/priority
   - Health checks

2. Request Queue
   - Priority queues
   - Rate limiting
   - Timeout handling

3. Inference Servers
   - vLLM for serving
   - Continuous batching
   - PagedAttention

4. Model Storage
   - Pre-quantized models
   - Hot/cold storage

Scaling:
- Horizontal: Add more GPUs
- Vertical: Larger GPUs
- Autoscaling based on queue depth

Trade-offs:
- Latency vs throughput (batch size)
- Cost vs quality (quantization)
- Single vs multi-node (complexity)
```

### 4.2 Design a Training Pipeline for LLM

```
Requirements:
- Train 70B model
- Multi-node (16 nodes × 8 GPUs)
- Checkpoint frequently
- Resume from failures

Components:
1. Data Pipeline
   - Tokenized datasets
   - Distributed sampler
   - Streaming for scale

2. Training Framework
   - FSDP or DeepSpeed ZeRO-3
   - Mixed precision (BF16)
   - Gradient checkpointing

3. Checkpointing
   - Sharded checkpoints
   - Async saving
   - Cloud storage

4. Monitoring
   - Loss curves
   - Gradient norms
   - GPU utilization

Failure Handling:
- Regular checkpoints
- Elastic training (handle node failures)
- Automatic restart

Optimizations:
- Overlap communication/compute
- Prefetch data
- Compile with torch.compile
```

### 4.3 Design a RAG System

```
Requirements:
- Query over 1M documents
- Low latency (< 500ms)
- High relevance

Components:
1. Indexing Pipeline
   - Chunk documents
   - Generate embeddings
   - Store in vector DB

2. Retrieval
   - Embed query
   - ANN search (HNSW/IVF)
   - Rerank with cross-encoder

3. Generation
   - Prompt with context
   - LLM generates answer
   - Stream response

4. Evaluation
   - Relevance metrics
   - Faithfulness (hallucination)
   - User feedback

Optimizations:
- Hybrid search (semantic + keyword)
- Query expansion
- Document caching
- Pre-compute common queries
```

---

## 5. Coding Questions

### 5.1 Implement Multi-Head Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, N, _ = x.shape

        # Project and reshape
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out)
```

### 5.2 Implement RoPE

```python
def rotary_embedding(x, seq_len, dim):
    """Apply rotary position embedding."""
    # Create position indices
    position = torch.arange(seq_len, device=x.device).unsqueeze(1)

    # Create dimension indices
    dim_idx = torch.arange(0, dim, 2, device=x.device).float()
    freq = 1.0 / (10000 ** (dim_idx / dim))

    # Compute angles
    angles = position * freq  # (seq_len, dim/2)

    # Split x into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]

    # Apply rotation
    cos_angles = angles.cos()
    sin_angles = angles.sin()

    x_rotated = torch.stack([
        x1 * cos_angles - x2 * sin_angles,
        x1 * sin_angles + x2 * cos_angles
    ], dim=-1).flatten(-2)

    return x_rotated
```

### 5.3 Implement LoRA Layer

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(r, in_features) / r)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        # Frozen base + trainable LoRA
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)

    def merge(self):
        # Merge for inference
        self.linear.weight.data += self.scale * self.lora_B @ self.lora_A
```

### 5.4 Implement KV-Cache Generation

```python
def generate_with_kv_cache(model, input_ids, max_length):
    """Generate with KV-cache."""
    kv_cache = None
    generated = input_ids

    for _ in range(max_length):
        # Only process new token if cache exists
        if kv_cache is not None:
            input_for_model = generated[:, -1:]
        else:
            input_for_model = generated

        # Forward with cache
        logits, kv_cache = model(input_for_model, kv_cache=kv_cache)

        # Sample next token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # Check for EOS
        if (next_token == EOS_TOKEN).all():
            break

    return generated
```

---

## 6. Behavioral Preparation

### 6.1 Project Discussion Template

```
STAR Format:
- Situation: Context and challenge
- Task: Your specific responsibility
- Action: What you did (technical details)
- Result: Outcome and learnings

Example Questions:
1. "Tell me about a challenging ML project."
2. "Describe a time you had to debug a training issue."
3. "How did you scale a system?"
4. "Tell me about a failure and what you learned."
```

### 6.2 Key Projects to Prepare

```
For each project, prepare:
1. Problem statement (1-2 sentences)
2. Technical approach (key decisions)
3. Challenges faced (and solutions)
4. Results (quantitative if possible)
5. Learnings (what would you do differently)
```

### 6.3 Research Awareness

```
Be prepared to discuss:
1. Recent papers you've read
   - What was interesting
   - Limitations you noticed

2. Current LLM landscape
   - Major players and their approaches
   - Open vs closed source debate

3. Your opinions on
   - Scaling laws
   - Emergent abilities
   - Safety considerations

4. Future directions
   - What excites you
   - What problems remain
```

---

## 7. Quick Reference Sheets

### 7.1 Memory Formulas

```
Model Memory:
  FP32: params × 4 bytes
  FP16: params × 2 bytes

Training Memory per GPU (DDP):
  Params + Grads + Optimizer = params × 16 bytes (Adam FP32)

FSDP Memory per GPU:
  (Params + Grads + Optimizer) / N GPUs

KV-Cache per Sequence:
  2 × layers × kv_heads × head_dim × seq_len × 2 bytes (FP16)

Example (LLaMA-70B):
  Model: 70B × 2 = 140 GB (FP16)
  KV per seq (4096 tokens): 1.3 GB
```

### 7.2 Complexity Formulas

```
Attention:
  Time: O(n² × d)
  Memory: O(n²) standard, O(n) Flash Attention

Transformer Block:
  Attention: 4 × d × d (Q, K, V, O projections)
  FFN: 2 × d × 4d (assuming 4x hidden)
  Total: 12 × d² per block

Full Transformer:
  Parameters: 12 × L × d² + vocab × d
  FLOPs per token: 2 × params (forward)
```

### 7.3 Key Numbers

```
Models:
  GPT-3: 175B params
  LLaMA-2 70B: 70B params
  GPT-4: ~1.8T params (rumored MoE)

Training:
  LLaMA-2 70B: ~1.7M GPU hours
  Chinchilla optimal: 20 tokens per parameter

Hardware:
  A100-80GB: 80GB HBM, 2TB/s bandwidth, 312 TFLOPS (BF16)
  H100-80GB: 80GB HBM, 3.35TB/s bandwidth, 989 TFLOPS (BF16)
```

### 7.4 Key Hyperparameters

```
Training:
  Learning rate: 1e-4 to 3e-4 (pretraining)
  Batch size: 2M-4M tokens
  Warmup: 2000-4000 steps
  Weight decay: 0.1

Fine-tuning:
  Learning rate: 1e-5 to 5e-5
  LoRA r: 8-64
  LoRA alpha: 16-32

Inference:
  Temperature: 0.7-1.0
  Top-p: 0.9-0.95
  Top-k: 50
```

---

## 8. Final Preparation Tips

### 8.1 Week Before Interview

```
Technical:
□ Review all module summaries
□ Practice whiteboard coding
□ Review your past projects
□ Read 2-3 recent papers

Practical:
□ Test video/audio setup
□ Prepare quiet space
□ Have water ready
□ Get good sleep
```

### 8.2 During Interview

```
DO:
✓ Ask clarifying questions
✓ Think out loud
✓ Draw diagrams
✓ Mention trade-offs
✓ Admit when you don't know

DON'T:
✗ Rush to answer
✗ Pretend to know everything
✗ Get stuck silently
✗ Argue with interviewer
✗ Speak negatively about past work
```

### 8.3 Question Framework

```
When asked a technical question:
1. Clarify: "Do you mean X or Y?"
2. Structure: "There are three aspects to this..."
3. Explain: Walk through each point
4. Trade-offs: "The trade-off is..."
5. Check: "Does that answer your question?"
```

### 8.4 What Sets Candidates Apart

```
Good candidate:
- Knows concepts
- Can explain clearly
- Implements correctly

Great candidate:
- Understands trade-offs
- Has opinions backed by experience
- Asks insightful questions
- Thinks about edge cases
- Connects to real-world applications
```

---

## Summary

### Essential Topics (Must Know)

1. **Transformer attention** - Can implement from scratch
2. **Position encodings** - Sinusoidal, RoPE, ALiBi
3. **Training** - Mixed precision, gradient checkpointing
4. **RLHF/DPO** - High-level understanding
5. **KV-cache** - Why needed, memory cost
6. **Distributed training** - DDP vs FSDP

### Important Topics (Should Know)

1. **Modern architectures** - LLaMA innovations
2. **Flash Attention** - Why faster
3. **LoRA/QLoRA** - When and why
4. **Quantization** - INT8 vs INT4
5. **Serving** - Continuous batching, PagedAttention

### Good to Know

1. **Speculative decoding**
2. **Tensor/Pipeline parallelism details**
3. **Recent research directions**
4. **Constitutional AI**
5. **Multi-modal extensions**

### Final Advice

```
1. Understand, don't memorize
   - Be able to derive, not recite
   - Know WHY, not just WHAT

2. Practice implementation
   - Code attention from scratch
   - Debug training issues

3. Stay current
   - Follow key researchers
   - Read paper abstracts at least

4. Be honest
   - "I don't know" is better than BS
   - "I would approach it by..." shows thinking

5. Show enthusiasm
   - Ask about their work
   - Discuss what excites you
```

Good luck with your interviews!
