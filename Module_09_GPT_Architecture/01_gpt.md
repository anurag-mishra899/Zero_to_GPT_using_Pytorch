# Module 9.1: GPT Architecture - Decoder-Only Transformers

## Table of Contents
1. [GPT Overview](#1-gpt-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Training Objective](#3-training-objective)
4. [GPT-2 and GPT-3](#4-gpt-2-and-gpt-3)
5. [Text Generation](#5-text-generation)
6. [Scaling Laws](#6-scaling-laws)
7. [Prompt Engineering](#7-prompt-engineering)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. GPT Overview

### 1.1 What is GPT?

**GPT** = Generative Pre-trained Transformer

Key characteristics:
- **Decoder-only**: No encoder, no cross-attention
- **Autoregressive**: Predicts next token given previous tokens
- **Causal masking**: Each position only sees previous positions
- **Pre-trained**: Learned from massive text corpora
- **Generative**: Can generate coherent text

### 1.2 Evolution

```
GPT-1 (2018):   117M params, 12 layers
GPT-2 (2019):   1.5B params, 48 layers
GPT-3 (2020):   175B params, 96 layers
GPT-4 (2023):   ~1.8T params (rumored), mixture of experts
```

### 1.3 GPT vs BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (left-to-right) | Bidirectional |
| Training | Next token prediction | Masked LM + NSP |
| Primary use | Generation | Understanding |
| Pre-training | Autoregressive LM | Masked tokens |

### 1.4 Why Decoder-Only?

**For generation**:
- Natural fit for autoregressive generation
- Each token predicted from left context
- No need for encoder (input = context)

**Simplicity**:
- Single stack of blocks
- No cross-attention complexity
- Easier to scale

---

## 2. Architecture Deep Dive

### 2.1 Overall Structure

```
Input tokens: [t_1, t_2, ..., t_n]
       ↓
Token Embedding (vocab_size × d_model)
       ↓
+ Positional Embedding (max_len × d_model)
       ↓
Dropout
       ↓
[  Transformer Block  ] × N layers
       ↓
Layer Norm (final)
       ↓
LM Head (Linear: d_model → vocab_size)
       ↓
Output logits: [l_1, l_2, ..., l_n]
```

### 2.2 Transformer Block

Each block contains:
```python
def transformer_block(x, mask):
    # Pre-norm attention
    residual = x
    x = layer_norm(x)
    x = causal_self_attention(x, mask)
    x = dropout(x)
    x = residual + x

    # Pre-norm FFN
    residual = x
    x = layer_norm(x)
    x = ffn(x)
    x = dropout(x)
    x = residual + x

    return x
```

### 2.3 Causal Self-Attention

```
Query, Key, Value from same input:
  Q = X @ W_Q
  K = X @ W_K
  V = X @ W_V

Attention scores:
  scores = Q @ K^T / sqrt(d_k)

Causal mask (prevent looking at future):
  scores[i, j] = -inf  if j > i

Output:
  output = softmax(scores) @ V
```

### 2.4 Position Embeddings

**GPT-2 uses learned position embeddings**:
```python
# Position embedding table
pos_embedding = nn.Embedding(max_position, d_model)

# During forward pass
positions = torch.arange(seq_len)
x = token_emb(tokens) + pos_embedding(positions)
```

**Key property**: Each position has its own learned vector (not sinusoidal).

### 2.5 Weight Tying

**Input embedding = Output projection**:
```python
# Share weights
self.lm_head.weight = self.token_embedding.weight
```

**Benefits**:
- Fewer parameters
- Better generalization
- Consistent token representations

---

## 3. Training Objective

### 3.1 Language Modeling Objective

**Next token prediction**:
```
Given: "The cat sat on the"
Predict: "mat"

Loss = -log P(mat | The cat sat on the)
```

**For a sequence**:
```
L = -Σ_t log P(x_t | x_1, x_2, ..., x_{t-1})

Averaged over all positions and all sequences
```

### 3.2 Training with Teacher Forcing

**Teacher forcing**: Use ground truth tokens as input (not model predictions).

```
Input:  [BOS, The, cat, sat, on,  the]
Target: [The, cat, sat, on,  the, mat]

At position 3:
  Input to model: [BOS, The, cat]
  Predict: "sat"
  Loss: -log P(sat | BOS, The, cat)
```

**Parallel computation**: All positions computed simultaneously.

### 3.3 Cross-Entropy Loss

```python
def compute_loss(logits, targets):
    # logits: (batch, seq_len, vocab_size)
    # targets: (batch, seq_len)

    # Shift: predict next token
    shift_logits = logits[:, :-1, :]  # All but last
    shift_targets = targets[:, 1:]     # All but first

    # Flatten
    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_targets.reshape(-1)
    )
    return loss
```

### 3.4 Perplexity

**Perplexity** = exp(average_loss)

```
PPL = exp(-1/N × Σ log P(x_t | x_{<t}))
```

**Interpretation**:
- PPL = 100 means model is "as confused as" choosing from 100 options
- Lower is better
- GPT-3: ~20 perplexity on common benchmarks

---

## 4. GPT-2 and GPT-3

### 4.1 GPT-2 Configurations

| Model | Layers | d_model | Heads | Params |
|-------|--------|---------|-------|--------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 762M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |

### 4.2 GPT-2 Specifics

**Differences from original Transformer**:
```
1. Decoder-only (no encoder)
2. Pre-LayerNorm (not Post-LayerNorm)
3. Learned position embeddings (not sinusoidal)
4. GELU activation (not ReLU)
5. Weight tying (embedding = output)
6. Modified initialization
```

**Initialization**:
```python
# Scale residual layer weights by 1/sqrt(N) where N = num_layers
# Prevents residual stream from growing with depth
scale = 1.0 / math.sqrt(num_layers)
```

### 4.3 GPT-3 Configurations

| Model | Layers | d_model | Heads | d_ff | Params |
|-------|--------|---------|-------|------|--------|
| Small | 12 | 768 | 12 | 3072 | 125M |
| Medium | 24 | 1024 | 16 | 4096 | 350M |
| Large | 24 | 1536 | 16 | 6144 | 760M |
| XL | 24 | 2048 | 24 | 8192 | 1.3B |
| 2.7B | 32 | 2560 | 32 | 10240 | 2.7B |
| 6.7B | 32 | 4096 | 32 | 16384 | 6.7B |
| 13B | 40 | 5140 | 40 | 20480 | 13B |
| 175B | 96 | 12288 | 96 | 49152 | 175B |

### 4.4 GPT-3 Specifics

**Alternating dense and sparse attention** (for 175B):
- Every other layer uses sparse attention patterns
- Reduces computation for long sequences

**Context length**: 2048 tokens (vs 1024 for GPT-2)

### 4.5 In-Context Learning

**GPT-3's key discovery**: Few-shot learning via prompts.

```
Prompt: "Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>"

Model output: "fromage"
```

**No fine-tuning needed** - model learns from examples in the prompt.

---

## 5. Text Generation

### 5.1 Autoregressive Generation

```
Input: "The cat"
Step 1: P(sat|The cat) → sample "sat"
Step 2: P(on|The cat sat) → sample "on"
Step 3: P(the|The cat sat on) → sample "the"
...
Continue until EOS or max_length
```

### 5.2 Sampling Strategies

**Greedy Decoding**:
```python
next_token = logits.argmax(dim=-1)
```
- Always pick highest probability
- Deterministic
- Can be repetitive/boring

**Temperature Sampling**:
```python
logits = logits / temperature
probs = softmax(logits)
next_token = multinomial(probs)
```
- temperature < 1: More focused/conservative
- temperature > 1: More random/creative
- temperature = 1: Original distribution

**Top-k Sampling**:
```python
# Keep only top k tokens
top_k_logits, top_k_indices = topk(logits, k)
probs = softmax(top_k_logits)
next_token = top_k_indices[multinomial(probs)]
```
- Restricts sampling to k most likely tokens
- Typical k: 40-100

**Top-p (Nucleus) Sampling**:
```python
# Keep tokens until cumulative probability >= p
sorted_probs = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
cutoff = find_first(cumsum >= p)
# Sample from tokens before cutoff
```
- Dynamic vocabulary size
- Typical p: 0.9-0.95
- Often combined with temperature

### 5.3 Repetition Penalty

Penalize tokens that have appeared before:
```python
for token in generated_tokens:
    logits[token] /= repetition_penalty
```
- penalty > 1: Reduce probability of repeated tokens
- Typical: 1.1-1.3

### 5.4 Beam Search

Keep top-k hypotheses at each step:
```
Step 1: Top-3 words after "The" → ["cat", "dog", "man"]
Step 2: For each, get top-3 next words
        "The cat" → ["sat", "ran", "ate"]
        "The dog" → ["barked", "ran", "slept"]
        ...
        Keep top-3 overall sequences
Step 3: Continue...
```
- Better for translation/summarization
- Less good for open-ended generation (deterministic)

### 5.5 KV-Cache for Efficient Generation

**Problem**: At step t, we recompute attention for all previous tokens.

**Solution**: Cache key and value vectors:
```python
# At step t:
# Only compute Q, K, V for new token
new_k, new_v = compute_kv(new_token)

# Append to cache
k_cache = concat(k_cache, new_k)
v_cache = concat(v_cache, new_v)

# Attention with full K, V
output = attention(new_q, k_cache, v_cache)
```

**Speedup**: O(n²) → O(n) per step

---

## 6. Scaling Laws

### 6.1 Kaplan Scaling Laws (2020)

**Key findings**:
```
Performance ∝ N^α × D^β × C^γ

where:
  N = model parameters
  D = dataset size
  C = compute budget
  α ≈ 0.076, β ≈ 0.095, γ ≈ 0.050
```

**Implications**:
- Larger models are more sample-efficient
- Given fixed compute, prefer larger models with less data
- Performance improves predictably with scale

### 6.2 Chinchilla Scaling Laws (2022)

**Updated findings**:
```
Optimal: N ∝ D (parameters should scale with data)

For compute-optimal training:
  Tokens ≈ 20 × Parameters
```

**Example**:
- 70B parameter model → train on ~1.4T tokens
- Previous models were often undertrained

### 6.3 Emergent Abilities

Capabilities that appear suddenly at certain scales:
```
Scale        Emergent Ability
~10B         Basic arithmetic
~100B        Chain-of-thought reasoning
~175B        Complex multi-step reasoning
```

**Not predictable** from smaller model performance.

---

## 7. Prompt Engineering

### 7.1 Zero-Shot

No examples, just instruction:
```
Classify the sentiment of this review as positive or negative:
"This movie was absolutely fantastic!"
Sentiment:
```

### 7.2 Few-Shot

Include examples in prompt:
```
Classify sentiment:
"Great product!" → positive
"Terrible service." → negative
"Loved the food!" → positive
"Worst experience ever." →
```

### 7.3 Chain-of-Thought (CoT)

Encourage step-by-step reasoning:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each.
   How many tennis balls does he have now?

A: Roger starts with 5 balls.
   2 cans × 3 balls = 6 balls.
   5 + 6 = 11 balls.
   Answer: 11
```

### 7.4 Best Practices

```
1. Be specific and clear
2. Provide context/examples
3. Specify output format
4. Use step-by-step for complex tasks
5. Experiment with temperature
```

---

## 8. Interview Questions

### Q1: Explain how GPT generates text autoregressively.

**Answer**:

**Autoregressive generation**:
```
1. Start with prompt tokens: [t_1, ..., t_n]
2. Feed through model to get logits for all positions
3. Take logits at position n: P(next | t_1, ..., t_n)
4. Sample next token t_{n+1} from distribution
5. Append to sequence: [t_1, ..., t_n, t_{n+1}]
6. Repeat from step 2 until EOS or max_length
```

**Key points**:
- Each token conditioned on all previous tokens
- Causal masking ensures no future information leaks during training
- Same process for training (with teacher forcing) and inference

**Efficiency trick**: KV-cache stores previous key/value vectors:
- Don't recompute K, V for previous tokens
- Only compute for new token
- O(n²) → O(n) per step

### Q2: What is the difference between Pre-LayerNorm and Post-LayerNorm?

**Answer**:

**Post-LayerNorm (Original Transformer)**:
```python
x = LayerNorm(x + Sublayer(x))
```
- Normalize after residual addition
- Gradients must flow through LayerNorm
- Harder to train deep models
- Requires warmup

**Pre-LayerNorm (GPT-2, modern LLMs)**:
```python
x = x + Sublayer(LayerNorm(x))
```
- Normalize before sublayer
- Residual provides clean gradient path
- More stable training
- Can train deeper models

**Why Pre-Norm is preferred**:
- Skip connection bypasses normalization
- Direct gradient flow through residual stream
- GPT-2, GPT-3, LLaMA all use Pre-Norm

### Q3: Explain temperature, top-k, and top-p sampling.

**Answer**:

**Temperature**:
```python
logits = logits / temperature
probs = softmax(logits)
```
- Controls distribution sharpness
- T < 1: More confident (peaked distribution)
- T > 1: More random (flat distribution)
- T = 0: Equivalent to greedy (argmax)

**Top-k Sampling**:
```python
# Only consider top k tokens
top_k_probs = probs.topk(k)
```
- Limits vocabulary to k most likely tokens
- Prevents very unlikely tokens
- Fixed cutoff regardless of probability mass

**Top-p (Nucleus) Sampling**:
```python
# Keep smallest set with cumulative prob >= p
sorted_probs = sort(probs, descending=True)
cumsum = cumsum(sorted_probs)
nucleus = probs[cumsum < p]
```
- Dynamic vocabulary size
- Adapts to confidence level
- Often combined with temperature

**Typical values**:
- Temperature: 0.7-1.0
- Top-k: 40-100
- Top-p: 0.9-0.95

### Q4: What are scaling laws and why do they matter?

**Answer**:

**Scaling laws** describe how model performance improves with:
- Model size (parameters N)
- Dataset size (tokens D)
- Compute budget (FLOPs C)

**Key findings**:

**Kaplan (2020)**:
```
Loss ∝ N^(-0.076) × D^(-0.095)
```
- Larger models more sample-efficient
- Prioritize model size over data size

**Chinchilla (2022)**:
```
Optimal: N ∝ D (balanced scaling)
Rule: Tokens ≈ 20 × Parameters
```
- Previous models undertrained
- 70B model → 1.4T tokens optimal

**Why it matters**:
1. Predictable performance gains
2. Optimal resource allocation
3. Training budget planning
4. Architecture decisions

### Q5: What is in-context learning?

**Answer**:

**In-context learning**: Model learns from examples in the prompt without gradient updates.

**Example**:
```
Q: What is the capital of France?
A: Paris

Q: What is the capital of Japan?
A: Tokyo

Q: What is the capital of Brazil?
A:
```
Model predicts "Brasilia" without fine-tuning.

**Types**:
- **Zero-shot**: Just instruction, no examples
- **One-shot**: Single example
- **Few-shot**: Multiple examples

**Why it works**:
- GPT-3 trained on diverse data with natural patterns
- Model learns to recognize and continue patterns
- Larger models better at in-context learning

**Limitations**:
- Context length limits number of examples
- Not as good as fine-tuning for specific tasks
- Sensitive to prompt formatting

### Q6: Explain KV-cache and why it's important.

**Answer**:

**Problem**: During generation, at step t:
```
Input: [t_1, t_2, ..., t_t]
Need to compute attention over all previous tokens
Naive: O(t²) computation per step
Total for n tokens: O(n³)
```

**KV-Cache Solution**:
```python
# Cache key and value vectors from previous steps
k_cache = []  # List of key vectors
v_cache = []  # List of value vectors

for new_token in generation:
    # Compute K, V for new token only
    new_k, new_v = compute_kv(new_token)

    # Append to cache
    k_cache.append(new_k)
    v_cache.append(new_v)

    # Compute query for new token
    new_q = compute_q(new_token)

    # Attention: new_q against all cached K, V
    output = attention(new_q, k_cache, v_cache)
```

**Complexity**:
- Per step: O(t) instead of O(t²)
- Total: O(n²) instead of O(n³)

**Memory trade-off**:
- Cache size: O(n × d_model × layers)
- For 175B model with 2048 context: ~3GB per sequence
- Batch size limited by cache memory

---

## 9. Summary

### Key Architecture Points

| Component | GPT Specifics |
|-----------|---------------|
| Architecture | Decoder-only |
| Attention | Causal (masked) |
| Position | Learned embeddings |
| Normalization | Pre-LayerNorm |
| Activation | GELU |
| Output | Weight tied with embedding |

### Key Equations

**Language Model Loss**:
```
L = -Σ_t log P(x_t | x_{<t})
```

**Perplexity**:
```
PPL = exp(L)
```

**Temperature Sampling**:
```
P(x) = softmax(logits / T)
```

### Key Takeaways

1. **GPT is decoder-only**: Causal attention, autoregressive generation

2. **Pre-LayerNorm is standard**: Better gradient flow, more stable

3. **Scaling matters**: Performance predictably improves with scale

4. **In-context learning**: Few-shot learning without fine-tuning

5. **Efficient generation**: KV-cache essential for fast inference

6. **Sampling matters**: Temperature, top-k, top-p for quality generation

7. **Prompt engineering**: Critical for getting good results
