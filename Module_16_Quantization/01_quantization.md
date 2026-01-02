# Module 16: LLM Quantization

## Table of Contents
1. [Why Quantization?](#1-why-quantization)
2. [Quantization Fundamentals](#2-quantization-fundamentals)
3. [Post-Training Quantization (PTQ)](#3-post-training-quantization-ptq)
4. [Weight-Only Quantization](#4-weight-only-quantization)
5. [Advanced Methods (GPTQ, AWQ)](#5-advanced-methods-gptq-awq)
6. [Quantization-Aware Training (QAT)](#6-quantization-aware-training-qat)
7. [Practical Deployment](#7-practical-deployment)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Why Quantization?

### 1.1 The Memory Problem

LLMs require massive memory:
```
LLaMA-70B in FP16:
  Parameters: 70B × 2 bytes = 140 GB
  KV-Cache: Additional 10-50 GB
  Total: 150-200 GB

Available hardware:
  A100: 80 GB
  Consumer GPU: 8-24 GB

Quantization enables:
  INT8: 70 GB (fits on 1 A100)
  INT4: 35 GB (fits on 2× RTX 4090)
```

### 1.2 Benefits of Quantization

```
1. Memory reduction:
   FP16 → INT8: 2× reduction
   FP16 → INT4: 4× reduction

2. Speed improvement:
   - Less memory bandwidth needed
   - Specialized INT8/INT4 kernels
   - 1.5-3× speedup typical

3. Cost reduction:
   - Smaller GPUs
   - More concurrent users
   - Lower cloud costs
```

### 1.3 Quantization Types

```
1. Weight-only quantization:
   - Quantize weights, compute in FP16
   - Best for memory-bound workloads
   - Minimal accuracy loss

2. Weight + Activation quantization:
   - Quantize both weights and activations
   - Greater speedup potential
   - More accuracy challenges

3. KV-Cache quantization:
   - Quantize cached keys/values
   - Reduces memory during generation
   - Important for long contexts
```

---

## 2. Quantization Fundamentals

### 2.1 Linear Quantization

Map continuous values to discrete integers:
```
Quantize:
  x_q = round(x / scale) + zero_point

Dequantize:
  x = (x_q - zero_point) × scale

Where:
  scale = (max - min) / (2^bits - 1)
  zero_point = round(-min / scale)
```

### 2.2 Symmetric vs Asymmetric

**Symmetric** (zero_point = 0):
```
Range: [-max_abs, +max_abs]
scale = max_abs / (2^(bits-1) - 1)

For INT8: [-127, 127] maps to [-max, +max]

Pros: Simpler, no zero_point storage
Cons: Wastes range if data is not centered
```

**Asymmetric**:
```
Range: [min, max]
scale = (max - min) / (2^bits - 1)
zero_point = round(-min / scale)

For INT8: [0, 255] maps to [min, max]

Pros: Full range utilization
Cons: Need to store/compute zero_point
```

### 2.3 Per-Tensor vs Per-Channel

**Per-tensor**:
```
One scale/zero_point for entire tensor
Simple but less accurate
All values share same range
```

**Per-channel** (recommended for weights):
```
Different scale/zero_point per output channel
Much better accuracy
Standard for weight quantization

For Linear(in, out):
  scales: (out,)
  one scale per output feature
```

### 2.4 Block-wise Quantization

Modern approach for LLMs:
```
Divide tensor into blocks (e.g., 128 elements)
Quantize each block independently

Benefits:
- Fine-grained range adaptation
- Handles outliers better
- Standard for 4-bit quantization

Memory overhead:
  Scales per block: 1 FP16 per 128 elements
  = 2 bytes / 128 = 0.015625 bytes per element
  For INT4: 0.5 + 0.015625 ≈ 0.52 bytes per element
```

### 2.5 Quantization Precision Formats

| Format | Bits | Range | Use Case |
|--------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | Training |
| FP16 | 16 | ±65504 | Training/Inference |
| BF16 | 16 | ±3.4e38 | Training |
| INT8 | 8 | -128 to 127 | Inference |
| INT4 | 4 | -8 to 7 | Inference |
| NF4 | 4 | (special) | QLoRA |

---

## 3. Post-Training Quantization (PTQ)

### 3.1 Overview

Quantize after training without retraining:
```
Process:
1. Take pretrained FP16 model
2. Collect calibration data (optional)
3. Compute quantization parameters
4. Convert weights to lower precision

No training required!
```

### 3.2 Dynamic Quantization

Quantize weights statically, activations at runtime:
```python
# PyTorch dynamic quantization
model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

Pros:
- No calibration data needed
- Works out of the box

Cons:
- Activation quantization overhead at runtime
- Less optimal than static
```

### 3.3 Static Quantization

Pre-compute activation ranges using calibration:
```python
# 1. Prepare model
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 2. Calibrate with representative data
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

# 3. Convert
torch.quantization.convert(model, inplace=True)
```

### 3.4 Calibration Methods

**MinMax**:
```
Track min/max values during calibration
scale = (max - min) / 255
Simple but sensitive to outliers
```

**Histogram/Percentile**:
```
Build histogram of values
Use percentiles (e.g., 99.9%) as range
More robust to outliers
```

**MSE Optimization**:
```
Find scale that minimizes:
  L = ||x - dequant(quant(x))||²
Optimal but slower
```

### 3.5 Handling Outliers

LLMs have outlier activations that break quantization:
```
Problem:
  Few very large values dominate range
  Most values get few quantization levels
  Accuracy degrades significantly

Solutions:
1. Percentile clipping (ignore outliers)
2. Mixed precision (keep outliers in FP16)
3. Per-channel scaling
4. Outlier-aware methods (LLM.int8())
```

---

## 4. Weight-Only Quantization

### 4.1 Why Weight-Only?

LLM inference is memory-bound:
```
Matrix multiply: Y = XW
  X: (batch, seq, hidden)   - Activations
  W: (hidden, hidden)       - Weights

For small batch (inference):
  Time dominated by loading W from memory
  Compute is fast once loaded

Weight quantization:
  Reduces memory bandwidth
  Dequantize on-the-fly
  Compute in FP16
```

### 4.2 INT8 Weight Quantization

```
Per-channel symmetric INT8:
  W_q = round(W / scale)
  scale = max(|W|, dim=input_dim) / 127

Storage: 1 byte per weight
Accuracy: Very close to FP16
```

### 4.3 INT4 Weight Quantization

```
Block-wise INT4:
  Block size: 32-128 elements
  Per-block scale and zero_point

Storage: 0.5 bytes + scale overhead
  ≈ 0.5625 bytes per weight with 128 block size

Accuracy: Slight degradation
  - Need calibration or advanced methods
  - Fine for most tasks
```

### 4.4 Group Quantization

```
Groups: Similar to blocks
  Divide hidden dimension into groups
  Each group has own scale

Example (hidden=4096, group=128):
  32 groups per row
  32 scales per row
  Fine-grained adaptation
```

---

## 5. Advanced Methods (GPTQ, AWQ)

### 5.1 GPTQ (Post-Training Quantization for GPT)

Optimal quantization by minimizing reconstruction error:
```
Problem:
  Find W_q that minimizes ||WX - W_qX||²
  Where X is calibration data

GPTQ approach:
  1. Quantize weights column by column
  2. Use Hessian (X^T X) to guide quantization order
  3. Adjust remaining weights to compensate error

Key insight:
  Quantization error propagates
  Compensate in remaining weights
```

### 5.2 GPTQ Algorithm

```
For each column i:
  1. Quantize: w_q[i] = quant(w[i])
  2. Error: δ = w[i] - w_q[i]
  3. Compensate: w[j>i] -= δ × H_inv[i,j] / H_inv[i,i]

Where H = X^T X (Hessian approximation)
Process columns in order of H diagonal (sensitivity)
```

### 5.3 GPTQ Properties

```
Pros:
- Very accurate INT4 quantization
- One-shot (no iterative training)
- Fast quantization (~hours for 70B)

Cons:
- Needs calibration data
- Memory intensive during quantization
- Order-dependent (can affect reproducibility)
```

### 5.4 AWQ (Activation-Aware Weight Quantization)

Protect important weights:
```
Observation:
  Not all weights equally important
  Some channels have large activations (salient)
  Quantization error on these hurts more

AWQ approach:
  1. Find salient channels (high activation magnitude)
  2. Scale up salient weights before quantization
  3. Scale down activations correspondingly
  4. Salient channels get more precision
```

### 5.5 AWQ Algorithm

```
1. Profile activations to find salience:
   s_i = mean(|X[:, i]|)

2. For salient channels (top k%):
   W'[:, i] = W[:, i] × scale_factor

3. Quantize W' (salient channels better preserved)

4. At runtime:
   Output = (W'_q × X) / scale_factor
   Or equivalently: X' = X / scale_factor
```

### 5.6 AWQ Properties

```
Pros:
- Better accuracy than basic INT4
- Simpler than GPTQ
- No iterative optimization

Cons:
- Still needs calibration data
- Slightly slower inference (scaling)
- Less optimal than GPTQ for some models
```

### 5.7 GPTQ vs AWQ Comparison

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Method | Error compensation | Salience scaling |
| Accuracy | Slightly better | Very good |
| Speed | Slower quantization | Faster |
| Inference | Standard | Needs scaling |
| Complexity | Higher | Lower |

---

## 6. Quantization-Aware Training (QAT)

### 6.1 Overview

Train with quantization simulation:
```
Forward pass:
  W_q = fake_quant(W)  # Quantize then dequantize
  Y = X @ W_q          # Use "quantized" weights

Backward pass:
  Straight-through estimator
  Gradient flows through fake_quant

Model learns to be robust to quantization
```

### 6.2 Straight-Through Estimator (STE)

Gradient for non-differentiable quantization:
```
Forward: y = round(x)
Backward: dy/dx = 1 (pretend identity)

Intuition:
  round() has zero gradient almost everywhere
  STE allows gradient to flow
  Model learns to avoid quantization errors
```

### 6.3 QAT Process

```
1. Insert fake quantization layers:
   - After weights
   - After activations (optional)

2. Train with quantization simulation:
   - Normal training loop
   - fake_quant in forward pass
   - STE in backward pass

3. Convert to actual quantized model:
   - Replace fake_quant with real quant
   - Deploy
```

### 6.4 When to Use QAT

```
QAT makes sense when:
- PTQ accuracy is unacceptable
- Have compute for retraining
- Need maximum accuracy at low bits

For LLMs:
- Usually PTQ is sufficient
- QAT expensive at scale
- GPTQ/AWQ often good enough
```

---

## 7. Practical Deployment

### 7.1 Framework Choices

```
1. llama.cpp / GGML:
   - CPU-focused, works on consumer hardware
   - Various quantization formats (Q4_0, Q4_K_M, etc.)
   - Good for local deployment

2. vLLM:
   - GPU server deployment
   - AWQ, GPTQ support
   - High throughput

3. TensorRT-LLM:
   - NVIDIA's optimized inference
   - INT8, INT4, FP8
   - Maximum GPU performance

4. bitsandbytes:
   - Easy integration with HuggingFace
   - INT8, NF4 (QLoRA)
   - Good for training and inference
```

### 7.2 Quantization Formats in GGML

```
Q4_0: Basic 4-bit (block_size=32)
  - 4.5 bits effective
  - Fastest, lowest quality

Q4_K_M: K-quant medium
  - Variable block sizes
  - Better quality

Q5_K_M: K-quant medium 5-bit
  - Good quality/size balance

Q8_0: 8-bit
  - Nearly lossless
  - 2x model size reduction
```

### 7.3 Memory Estimation

```python
def estimate_memory(num_params, quant_type):
    """Estimate model memory in GB."""
    bits_per_param = {
        'fp32': 32,
        'fp16': 16,
        'int8': 8,
        'int4': 4.5,  # With overhead
        'nf4': 4.5,
    }

    bits = bits_per_param[quant_type]
    bytes_needed = num_params * bits / 8
    gb = bytes_needed / 1e9

    return gb

# Example: LLaMA-70B
print(f"FP16: {estimate_memory(70e9, 'fp16'):.1f} GB")  # 140 GB
print(f"INT8: {estimate_memory(70e9, 'int8'):.1f} GB")  # 70 GB
print(f"INT4: {estimate_memory(70e9, 'int4'):.1f} GB")  # 39 GB
```

### 7.4 Accuracy vs Compression

| Method | Bits | Accuracy | Memory | Speed |
|--------|------|----------|--------|-------|
| FP16 | 16 | Baseline | 100% | 1.0x |
| INT8 | 8 | ~99.5% | 50% | 1.5x |
| GPTQ-4 | 4 | ~98% | 28% | 2.0x |
| AWQ-4 | 4 | ~97% | 28% | 1.8x |
| Basic Q4 | 4 | ~95% | 28% | 2.0x |

### 7.5 Best Practices

```
1. Start with weight-only quantization
   - INT8 for minimal degradation
   - INT4 for aggressive compression

2. Use calibration data
   - 128-512 samples usually sufficient
   - Representative of deployment distribution

3. Evaluate on target tasks
   - Perplexity not enough
   - Test downstream performance

4. Consider mixed precision
   - Keep sensitive layers in higher precision
   - Usually attention projections, embeddings
```

---

## 8. Interview Questions

### Q1: Explain symmetric vs asymmetric quantization.

**Answer**:

**Symmetric quantization**:
```
Maps [-α, α] to [-127, 127] (INT8)
Zero maps to zero
scale = α / 127
x_q = round(x / scale)

Pros:
- Simpler computation
- No zero_point storage
- Efficient for signed data

Cons:
- Wastes range if data is biased
- Less accurate for non-symmetric distributions
```

**Asymmetric quantization**:
```
Maps [min, max] to [0, 255] (INT8)
scale = (max - min) / 255
zero_point = round(-min / scale)
x_q = round(x / scale) + zero_point

Pros:
- Full range utilization
- Better for biased distributions (ReLU outputs)

Cons:
- Need to store zero_point
- Extra computation in dequantization
```

**When to use**:
- Weights: Usually symmetric (centered around 0)
- Activations: Asymmetric if after ReLU (non-negative)

### Q2: What is GPTQ and how does it work?

**Answer**:

**GPTQ** minimizes quantization error using Hessian information.

**Key insight**: Quantization errors compound across columns. Compensate errors in remaining weights.

**Algorithm**:
```
Input: Weight matrix W, calibration data X
Output: Quantized weights W_q

1. Compute Hessian: H = X^T @ X
2. Compute inverse Hessian: H_inv

For column i in order of H diagonal:
  3. Quantize: w_q[i] = quant(w[i])
  4. Compute error: δ = w[i] - dequant(w_q[i])
  5. Update remaining: w[j>i] -= δ × H_inv[i,j] / H_inv[i,i]
```

**Why Hessian?**
- H_ii indicates sensitivity of column i
- Quantize less sensitive columns first
- More room to compensate errors

**Results**: Near-lossless INT4 for many models.

### Q3: Explain the difference between PTQ and QAT.

**Answer**:

**Post-Training Quantization (PTQ)**:
```
Process:
1. Take trained FP16 model
2. Determine quantization parameters (scale, zero_point)
3. Convert weights/activations to lower precision

Pros:
- Fast (no training)
- Works with any pretrained model
- Sufficient for 8-bit, often 4-bit

Cons:
- May lose accuracy at very low bits
- Limited ability to recover from quantization error
```

**Quantization-Aware Training (QAT)**:
```
Process:
1. Insert fake quantization operators
2. Train with simulated quantization
3. Model learns to be robust to quantization
4. Convert to actual quantized model

Pros:
- Better accuracy at low bits
- Model adapts to quantization
- Optimal for extreme quantization (2-3 bit)

Cons:
- Requires training compute
- Need training data
- Slower iteration
```

**For LLMs**: PTQ (GPTQ, AWQ) usually sufficient; QAT rarely needed.

### Q4: How do you handle activation outliers in LLM quantization?

**Answer**:

**The problem**:
```
LLMs have sparse outliers in activations
Few channels have values 10-100x larger
Standard quantization:
  scale = max(|x|) / 127
  Most values get few quantization levels
  Accuracy degrades
```

**Solutions**:

1. **Percentile clipping**:
```
Use 99.9th percentile instead of max
Clip outliers to this value
Trade outlier accuracy for majority accuracy
```

2. **Mixed-precision decomposition (LLM.int8())**:
```
Identify outlier features (>6 std)
Keep outlier dimensions in FP16
Quantize rest to INT8
Merge results
```

3. **Per-channel quantization**:
```
Separate scale per channel
Outlier channels get appropriate scale
Other channels unaffected
```

4. **Activation-aware scaling (AWQ)**:
```
Scale up important weights before quantization
Important = channels with large activations
Preserves accuracy on salient features
```

### Q5: What factors determine if you should use INT8 vs INT4?

**Answer**:

**Use INT8 when**:
```
1. Need near-lossless quality
   - <0.5% accuracy drop typical
   - Safe default choice

2. Memory not critical constraint
   - Have enough GPU memory
   - 2x compression sufficient

3. Speed matters
   - INT8 kernels very mature
   - Consistent speedup

4. Simple deployment
   - Wide framework support
   - Less complexity
```

**Use INT4 when**:
```
1. Memory is primary constraint
   - Need to fit on smaller GPU
   - Running large models locally

2. Can tolerate ~2-5% accuracy drop
   - Chat/assistant tasks often OK
   - Some reasoning tasks sensitive

3. Using advanced methods
   - GPTQ, AWQ minimize degradation
   - Calibration data available

4. Trading quality for throughput
   - More concurrent requests
   - Cost optimization
```

**Decision framework**:
```
Start with INT8 → Test accuracy
If memory constrained → Try GPTQ-4/AWQ-4
Evaluate on YOUR tasks → Pick acceptable quality
```

---

## 9. Summary

### Quantization Basics

```
Quantize: x_q = round(x / scale) + zero_point
Dequantize: x = (x_q - zero_point) × scale

Symmetric: zero_point = 0, for centered data
Asymmetric: full range utilization
Per-channel: different scale per output feature
Block-wise: different scale per block (32-128 elements)
```

### Method Comparison

| Method | Type | Bits | Accuracy | Complexity |
|--------|------|------|----------|------------|
| Dynamic PTQ | PTQ | 8 | Good | Low |
| Static PTQ | PTQ | 8 | Better | Medium |
| GPTQ | PTQ | 4 | Very good | High |
| AWQ | PTQ | 4 | Very good | Medium |
| QAT | Training | Any | Best | High |

### Memory Formula

```
Memory (GB) = Parameters × Bits / 8 / 1e9

7B model:
  FP16: 14 GB
  INT8: 7 GB
  INT4: ~4 GB (with overhead)

70B model:
  FP16: 140 GB
  INT8: 70 GB
  INT4: ~40 GB
```

### Best Practices

1. **Start with INT8** weight-only quantization
2. **Use calibration data** (128-512 samples)
3. **GPTQ/AWQ for INT4** (much better than naive)
4. **Evaluate downstream** (not just perplexity)
5. **Consider mixed precision** for sensitive layers

### Key Takeaways

1. **Quantization is essential** for LLM deployment
2. **Weight-only quantization** is usually sufficient
3. **GPTQ/AWQ** make INT4 practical with minimal loss
4. **Calibration data** significantly improves quality
5. **Trade-offs exist** - choose based on your constraints
