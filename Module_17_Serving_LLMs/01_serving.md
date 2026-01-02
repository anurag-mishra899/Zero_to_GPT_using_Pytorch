# Module 17: Serving LLMs in Production

## Table of Contents
1. [LLM Inference Challenges](#1-llm-inference-challenges)
2. [Batching Strategies](#2-batching-strategies)
3. [KV-Cache Management](#3-kv-cache-management)
4. [vLLM and PagedAttention](#4-vllm-and-pagedattention)
5. [Serving Frameworks](#5-serving-frameworks)
6. [Optimization Techniques](#6-optimization-techniques)
7. [Production Deployment](#7-production-deployment)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. LLM Inference Challenges

### 1.1 Autoregressive Bottleneck

LLM generation is sequential:
```
Generate token 1 → Generate token 2 → ... → Generate token N

Each step requires:
  - Full forward pass through model
  - Can't parallelize across tokens
  - Memory bound (low compute utilization)

Result:
  - High latency per token
  - Poor GPU utilization
  - Expensive to scale
```

### 1.2 Memory Challenges

```
Model memory:
  LLaMA-70B: 140 GB (FP16)

KV-Cache per sequence:
  Per layer: 2 × n_heads × head_dim × seq_len × 2 bytes
  All layers: 80 × 2 × 8 × 128 × 4096 × 2 = 1.3 GB per sequence!

For batch of 32:
  KV-Cache: 32 × 1.3 GB = 42 GB

Total: 140 + 42 = 182 GB (exceeds A100 80GB!)
```

### 1.3 Throughput vs Latency Trade-offs

```
Single request (low latency):
  - Small batch → Low throughput
  - GPU underutilized

Batched requests (high throughput):
  - Large batch → Higher latency
  - Better GPU utilization

Challenge: Balance both for production
```

### 1.4 Key Metrics

```
Latency metrics:
  - Time to First Token (TTFT): Time until first token generated
  - Inter-Token Latency (ITL): Time between tokens
  - End-to-End Latency: Total request time

Throughput metrics:
  - Requests per second (RPS)
  - Tokens per second (TPS)
  - GPU utilization (%)

Cost metrics:
  - Cost per 1M tokens
  - GPU hours per request
```

---

## 2. Batching Strategies

### 2.1 Static Batching

Traditional approach:
```
Wait for batch → Process all → Return results

Problems:
  - Must wait for batch to fill
  - Short sequences wait for long ones
  - Poor GPU utilization
  - High latency
```

### 2.2 Dynamic Batching

Process requests as they arrive:
```
Request queue → Batch when ready → Process

Better than static:
  - Don't wait for full batch
  - Timeout-based batching

Still has problem:
  - Sequences of different lengths
  - Padding waste
```

### 2.3 Continuous Batching (Iteration-Level Batching)

vLLM's approach:
```
At each decode step:
  1. Process all sequences in batch
  2. When sequence finishes, remove it
  3. Add new sequences immediately
  4. No waiting!

Benefits:
  - No padding
  - Maximum GPU utilization
  - Low latency for short sequences
  - High throughput
```

### 2.4 Continuous Batching Visualization

```
Time →  t1    t2    t3    t4    t5    t6

Static:
  Seq A: [===][===][===][===]
  Seq B: [===][===][===][===][===][===]
  Seq C: [   ][   ][   ][   ][   ][   ]  ← Waits!

Continuous:
  Seq A: [===][===][===][===]
  Seq B: [===][===][===][===][===][===]
  Seq C:                [===][===][===]  ← Starts when A finishes
```

### 2.5 Chunked Prefill

Optimize prefill for long prompts:
```
Problem:
  Long prompt prefill blocks decode steps
  Other sequences wait

Solution:
  Chunk prefill into smaller pieces
  Interleave with decode steps
  Better latency distribution
```

---

## 3. KV-Cache Management

### 3.1 KV-Cache Basics

```
During generation:
  Store K, V for all past tokens
  Don't recompute on each step

Memory per token (per layer):
  K: n_kv_heads × head_dim × 2 bytes
  V: n_kv_heads × head_dim × 2 bytes

Total per sequence (LLaMA-70B, 4096 tokens):
  80 layers × 8 heads × 128 dim × 4096 tokens × 2 × 2 = 1.3 GB
```

### 3.2 KV-Cache Problems

```
1. Memory fragmentation:
   - Different sequences have different lengths
   - Allocate max_seq_len per sequence → Waste
   - Dynamic allocation → Fragmentation

2. Memory waste:
   - Pre-allocate for max_seq_len
   - Most sequences shorter
   - Up to 80% memory wasted!

3. Scaling issues:
   - Can't fit many concurrent sequences
   - Limits throughput
```

### 3.3 PagedAttention Solution

Treat KV-cache like virtual memory:
```
Traditional:
  Allocate contiguous [max_seq_len × kv_size] per sequence
  Waste memory for short sequences

Paged:
  Allocate fixed-size blocks (e.g., 16 tokens)
  Sequences use blocks on demand
  Page table maps logical → physical blocks
```

### 3.4 PagedAttention Benefits

```
1. No fragmentation:
   - All blocks same size
   - Easy to allocate/deallocate

2. No waste:
   - Allocate only what's needed
   - Return blocks when done

3. Memory sharing:
   - Beam search: Share common prefix
   - Parallel sampling: Copy-on-write

4. More sequences:
   - Better memory utilization
   - 2-4x more concurrent requests
```

### 3.5 KV-Cache Compression

Further optimizations:
```
1. KV-Cache quantization:
   - FP16 → INT8: 2x reduction
   - FP16 → INT4: 4x reduction
   - Small quality impact

2. Attention sinks:
   - Keep only recent + initial tokens
   - Streaming for very long contexts

3. Grouped Query Attention (GQA):
   - Fewer KV heads
   - 4-8x reduction vs MHA
```

---

## 4. vLLM and PagedAttention

### 4.1 vLLM Architecture

```
Components:
1. Scheduler: Manages request queue, batching
2. Block Manager: Allocates KV-cache blocks
3. Worker: Runs model inference
4. Tokenizer: Handles tokenization

Flow:
  Request → Scheduler → Block allocation → Worker → Stream tokens
```

### 4.2 vLLM Scheduling

```
Priority-based scheduling:
  - FCFS (First Come First Served) default
  - Priority queues for different request types

Preemption:
  - If memory full, preempt long sequences
  - Resume later with cached state
  - Maintains fairness
```

### 4.3 vLLM Usage

```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # For multi-GPU
    gpu_memory_utilization=0.9,
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

# Generate
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 4.4 vLLM Server Mode

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2

# Query (OpenAI-compatible API)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "Hello, world!",
        "max_tokens": 100
    }'
```

### 4.5 vLLM Performance

```
Compared to HuggingFace:
  - 10-24x higher throughput
  - Similar or better latency

Key optimizations:
  - PagedAttention (memory efficiency)
  - Continuous batching (GPU utilization)
  - CUDA graphs (kernel launch overhead)
  - Optimized kernels
```

---

## 5. Serving Frameworks

### 5.1 Framework Comparison

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| vLLM | Throughput, memory | Production serving |
| TGI (HF) | Ease of use | Quick deployment |
| TensorRT-LLM | Max performance | NVIDIA optimization |
| llama.cpp | CPU/local | Edge/local |
| Ollama | Simplicity | Local development |

### 5.2 Text Generation Inference (TGI)

HuggingFace's serving solution:
```bash
# Docker deployment
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf

# Features:
# - Continuous batching
# - Flash attention
# - Quantization support
# - Token streaming
```

### 5.3 TensorRT-LLM

NVIDIA's optimized inference:
```python
# Build optimized engine
trtllm-build --model_dir ./model \
    --dtype float16 \
    --output_dir ./engine \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512

# Features:
# - Maximum GPU utilization
# - INT8/FP8 quantization
# - Multi-GPU support
# - In-flight batching
```

### 5.4 llama.cpp

CPU-friendly inference:
```bash
# Run locally
./main -m llama-2-7b.gguf \
    -p "Hello, world!" \
    -n 100

# Features:
# - GGUF quantization formats
# - CPU optimized (AVX, ARM NEON)
# - Metal (Apple Silicon)
# - Low memory footprint
```

### 5.5 Choosing a Framework

```
High throughput, production:
  → vLLM or TensorRT-LLM

Quick deployment, HuggingFace models:
  → TGI

Maximum performance, NVIDIA GPUs:
  → TensorRT-LLM

Local/CPU deployment:
  → llama.cpp or Ollama

Research/development:
  → HuggingFace Transformers
```

---

## 6. Optimization Techniques

### 6.1 Speculative Decoding

Use small model to draft, large model to verify:
```
1. Draft model generates k tokens quickly
2. Large model verifies all k in parallel
3. Accept matching prefix
4. Repeat

Speedup: 2-3x for well-matched draft/target models
```

### 6.2 CUDA Graphs

Capture and replay GPU operations:
```python
# Without CUDA graphs
for step in range(steps):
    launch_kernel_1()  # CPU overhead
    launch_kernel_2()  # CPU overhead
    launch_kernel_3()  # CPU overhead

# With CUDA graphs
graph = capture_graph(kernels)
for step in range(steps):
    replay_graph(graph)  # Single launch!

Benefit: Eliminates kernel launch overhead
Typically 10-20% speedup
```

### 6.3 Kernel Fusion

Combine multiple operations:
```
Unfused:
  LayerNorm → Attention → Dropout → Add
  (4 kernel launches, 4 memory roundtrips)

Fused:
  FusedAttentionBlock
  (1 kernel launch, 1 memory roundtrip)

Flash Attention is a form of kernel fusion
```

### 6.4 Tensor Parallelism for Serving

Split model across GPUs:
```
Single GPU:
  Request → GPU 0 → Response

Tensor Parallel (2 GPUs):
  Request → [GPU 0, GPU 1] → Response
  Each GPU has half the model
  Lower latency, same throughput

Pipeline Parallel:
  Not recommended for serving
  (pipeline bubble hurts latency)
```

### 6.5 Prefix Caching

Cache common prompt prefixes:
```
Many requests share prefix:
  "You are a helpful assistant. User: {query}"

Cache the KV for prefix:
  First request: Compute full KV
  Subsequent: Reuse prefix KV, compute only query

Benefit: Faster TTFT for repeated prefixes
```

---

## 7. Production Deployment

### 7.1 Capacity Planning

```
Estimate required GPUs:
  Target throughput: 100 requests/min
  Avg tokens per request: 500
  Tokens per second per GPU: 1000 (example)

  Required TPS: 100 × 500 / 60 = 833 TPS
  Required GPUs: 833 / 1000 = 1 GPU (with margin: 2 GPUs)
```

### 7.2 Autoscaling

```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-serving
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: gpu-utilization
      target:
        type: Utilization
        averageUtilization: 70
```

### 7.3 Load Balancing

```
Strategies:
1. Round-robin: Simple, may cause imbalance
2. Least-connections: Better for variable request length
3. GPU-aware: Route based on memory availability

Considerations:
  - Sticky sessions for stateful (chat)
  - Health checks for GPU OOM
  - Graceful degradation
```

### 7.4 Monitoring

Key metrics to track:
```
Latency:
  - P50, P95, P99 latency
  - TTFT distribution
  - Queue wait time

Throughput:
  - Requests per second
  - Tokens per second
  - Batch size distribution

Resources:
  - GPU utilization
  - GPU memory usage
  - KV-cache utilization

Errors:
  - OOM rate
  - Timeout rate
  - Error rate by type
```

### 7.5 Cost Optimization

```
Strategies:
1. Right-size GPUs:
   - A100 80GB vs 40GB based on model
   - Consider memory vs compute needs

2. Spot/preemptible instances:
   - 60-70% cost savings
   - Good for batch workloads

3. Quantization:
   - INT8: Same GPU, 2x batch
   - INT4: Smaller GPU possible

4. Caching:
   - Cache common responses
   - Prefix caching for repeated prompts

5. Request routing:
   - Simple queries → smaller model
   - Complex → larger model
```

---

## 8. Interview Questions

### Q1: Explain continuous batching and why it's better than static batching.

**Answer**:

**Static batching**:
```
Collect N requests → Process together → Return all

Problems:
- Wait for batch to fill (latency)
- All sequences padded to max length (waste)
- Short sequences wait for long ones
- GPU idle during collection
```

**Continuous batching**:
```
Process at iteration level:
- Each decode step can add/remove sequences
- No waiting for batch formation
- No padding (sequences different lengths)
- New requests start immediately

Benefits:
- Higher GPU utilization (no idle time)
- Lower latency (no waiting)
- Better throughput (no padding)
- Fair scheduling (short requests finish first)
```

**Implementation**:
```python
while requests:
    # Remove finished sequences
    batch = [s for s in batch if not s.done]

    # Add new sequences (if memory available)
    while new_requests and memory_available():
        batch.append(new_requests.pop())

    # Single decode step for entire batch
    outputs = model.decode(batch)

    # Stream finished tokens
    for seq, output in zip(batch, outputs):
        if output.is_eos:
            seq.done = True
            yield seq.response
```

### Q2: What is PagedAttention and how does it improve LLM serving?

**Answer**:

**Problem with traditional KV-cache**:
```
Allocate contiguous memory per sequence
Size: max_seq_len × kv_size

Issues:
1. Memory waste: Most sequences < max_seq_len
2. Fragmentation: Can't reuse freed memory efficiently
3. Low utilization: Up to 80% memory wasted
```

**PagedAttention solution**:
```
Treat KV-cache like virtual memory:
1. Divide memory into fixed-size blocks (e.g., 16 tokens)
2. Page table maps logical position → physical block
3. Allocate blocks on demand
4. Return blocks when sequence finishes
```

**Benefits**:
```
1. No waste: Allocate only needed blocks
2. No fragmentation: All blocks same size
3. Memory sharing:
   - Beam search: Share common prefix
   - Copy-on-write for parallel sampling
4. More sequences: 2-4x improvement
```

**Memory comparison** (LLaMA-7B, 2048 max):
```
Traditional: 2048 × 32 heads × 128 dim × 32 layers × 2 × 2 = 512 MB/seq
If avg seq = 512: 75% wasted

Paged (block=16): Only allocate for actual length
512 actual → 512/16 = 32 blocks = 128 MB/seq
```

### Q3: How would you optimize LLM serving for low latency vs high throughput?

**Answer**:

**For low latency (real-time chat)**:
```
1. Smaller batch size
   - Less time waiting for batch
   - Faster per-request processing

2. Speculative decoding
   - Draft model generates k tokens
   - Verify in parallel
   - 2-3x speedup

3. Tensor parallelism
   - Split model across GPUs
   - Lower latency per token

4. Optimized prefill
   - Chunked prefill
   - Don't block decode with long prompts

5. Prefix caching
   - Cache common prompt prefixes
   - Faster TTFT
```

**For high throughput (batch processing)**:
```
1. Larger batch size
   - Better GPU utilization
   - Higher tokens/second

2. Continuous batching
   - No idle time
   - Maximum utilization

3. Aggressive quantization
   - INT4 weights
   - More sequences in memory

4. Data parallelism
   - Multiple model replicas
   - Scale horizontally

5. Request batching
   - Group similar-length requests
   - Efficient memory usage
```

**Trade-offs**:
| Optimization | Latency | Throughput |
|--------------|---------|------------|
| Larger batch | + | ++ |
| Smaller batch | -- | - |
| Tensor parallel | -- | = |
| Speculative | -- | = |
| Quantization | + | ++ |

### Q4: Explain the memory hierarchy and why LLM inference is memory-bound.

**Answer**:

**GPU memory hierarchy**:
```
Registers: ~1 TB/s, KB size (fastest)
Shared Memory (SRAM): ~10 TB/s, MB size
L2 Cache: ~3 TB/s, MB size
HBM (Global Memory): ~2 TB/s, 80 GB (A100)
```

**Why LLM inference is memory-bound**:
```
Compute: A100 has 312 TFLOPS (BF16)
Bandwidth: A100 has 2 TB/s

For matrix multiply Y = XW:
  Compute: 2 × M × N × K FLOPs
  Memory: (M×K + K×N + M×N) × 2 bytes

Arithmetic intensity = FLOPs / Bytes

For small batch (decode):
  X: (1, d), W: (d, d)
  Compute: 2d² FLOPs
  Memory: 2d² bytes (dominated by loading W)
  Intensity: ~1 FLOP/byte

Required bandwidth for full compute:
  312 TFLOPS / 1 FLOP/byte = 312 TB/s
  Available: 2 TB/s
  → Memory bound by 150x!
```

**Solutions**:
```
1. Batching: Increase arithmetic intensity
2. Quantization: Reduce memory movement
3. Kernel fusion: Fewer memory roundtrips
4. Speculative decoding: More compute per memory access
```

### Q5: How do you handle GPU OOM errors in production LLM serving?

**Answer**:

**Prevention**:
```
1. Memory estimation:
   Model + KV-cache + activations + overhead
   Leave 10-20% margin

2. Request limits:
   - Max sequence length
   - Max concurrent requests
   - Max batch tokens

3. Admission control:
   - Reject if would exceed memory
   - Queue with timeout
```

**Handling OOM**:
```
1. Preemption:
   - Evict low-priority sequences
   - Resume later from checkpoint
   - vLLM does this automatically

2. Graceful degradation:
   - Reduce batch size
   - Quantize on-the-fly (emergency)
   - Route to backup service

3. Recovery:
   - Restart worker
   - Clear KV-cache
   - Re-queue affected requests
```

**Monitoring**:
```python
# Check memory before new request
def can_accept_request(request, current_memory):
    estimated_kv = estimate_kv_cache(request.max_tokens)
    return current_memory + estimated_kv < GPU_MEMORY * 0.9

# Alert on high memory
if gpu_memory_used / gpu_memory_total > 0.85:
    alert("High GPU memory usage")
    consider_preemption()
```

---

## 9. Summary

### Key Concepts

| Concept | Purpose | Impact |
|---------|---------|--------|
| Continuous batching | No idle time | 10-20x throughput |
| PagedAttention | Memory efficiency | 2-4x more requests |
| Speculative decoding | Faster decode | 2-3x latency |
| Quantization | Memory reduction | 2-4x more requests |
| Tensor parallelism | Lower latency | Near-linear speedup |

### Framework Selection

```
High throughput production:
  → vLLM

Maximum NVIDIA performance:
  → TensorRT-LLM

Quick deployment:
  → TGI (HuggingFace)

Local/CPU:
  → llama.cpp / Ollama
```

### Key Metrics

```
Latency:
  - TTFT: Time to first token
  - ITL: Inter-token latency
  - E2E: End-to-end latency

Throughput:
  - TPS: Tokens per second
  - RPS: Requests per second

Efficiency:
  - GPU utilization
  - Memory utilization
  - Cost per 1M tokens
```

### Best Practices

1. **Use continuous batching** (vLLM, TGI)
2. **Enable PagedAttention** for memory efficiency
3. **Quantize if possible** (INT8 minimal loss)
4. **Monitor GPU memory** and preempt if needed
5. **Cache common prefixes** for repeated prompts
6. **Set appropriate limits** (max tokens, max batch)
7. **Use tensor parallelism** for latency-sensitive apps

### Key Takeaways

1. **LLM inference is memory-bound** - optimize memory access
2. **Continuous batching is essential** for production
3. **PagedAttention solves KV-cache fragmentation**
4. **vLLM is the current standard** for high-throughput serving
5. **Speculative decoding** helps latency-sensitive applications
6. **Monitor and set limits** to prevent OOM
