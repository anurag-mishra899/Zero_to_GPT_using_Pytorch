# Module 7.1: Attention Mechanisms - From Seq2Seq to Self-Attention

## Table of Contents
1. [The Attention Problem](#1-the-attention-problem)
2. [Seq2Seq Attention (Bahdanau)](#2-seq2seq-attention-bahdanau)
3. [Luong Attention](#3-luong-attention)
4. [Scaled Dot-Product Attention](#4-scaled-dot-product-attention)
5. [Multi-Head Attention](#5-multi-head-attention)
6. [Self-Attention vs Cross-Attention](#6-self-attention-vs-cross-attention)
7. [Attention Visualizations](#7-attention-visualizations)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. The Attention Problem

### 1.1 The Bottleneck in Seq2Seq

Traditional Encoder-Decoder without attention:
```
Encoder: x_1 → x_2 → ... → x_T → [final hidden state h_T]
                                        ↓
Decoder: [h_T] → y_1 → y_2 → ... → y_S
```

**Problem**: Entire source sequence must be compressed into single vector h_T
- Information bottleneck
- Long sequences lose early information
- All positions equally distant from decoder

### 1.2 The Attention Solution

**Key insight**: Let decoder look at ALL encoder states, not just final one.

```
Encoder states: [h_1, h_2, ..., h_T]
                   ↓    ↓       ↓
Attention:      [α_1, α_2, ..., α_T]  (weights, sum to 1)
                   ↓    ↓       ↓
Context:        c = Σ α_t × h_t
                         ↓
Decoder:              uses c
```

At each decoder step, compute weighted sum of encoder states based on relevance.

### 1.3 Attention as Soft Alignment

Attention weights indicate alignment between:
- Decoder query (what we're looking for)
- Encoder keys (what's available)

Example (translation):
```
Source: "The black cat sat"
Target: "Le chat noir s'assit"

When generating "noir" (black):
  α_1("The") = 0.05
  α_2("black") = 0.85  ← High attention
  α_3("cat") = 0.08
  α_4("sat") = 0.02
```

---

## 2. Seq2Seq Attention (Bahdanau)

### 2.1 Bahdanau Attention (2014)

Also called "additive attention" - uses a feedforward network to compute scores.

```
Given:
  h_j = encoder hidden state at position j
  s_i = decoder hidden state at step i

Score: e_{ij} = v^T × tanh(W_h × h_j + W_s × s_{i-1})
       (alignment model - small neural network)

Attention weights: α_{ij} = softmax_j(e_{ij})

Context vector: c_i = Σ_j α_{ij} × h_j

Decoder update: s_i = f(s_{i-1}, y_{i-1}, c_i)
```

### 2.2 Why "Additive"?

Score computation combines query and key by addition:
```
e = v^T × tanh(W_1 × query + W_2 × key)

The query and key contributions are ADDED inside tanh
```

### 2.3 Complexity

```
Attention computation per step:
  - For each encoder position: O(d) for score computation
  - Total: O(n × d) where n = source length, d = hidden dimension

Total for sequence: O(n × m × d) where m = target length
```

### 2.4 Bahdanau Architecture

```
                    [Context c_i]
                         ↑
                    [Attention α]
                    ↑ ↑ ↑ ↑ ↑
Encoder:   h_1 → h_2 → h_3 → h_4 → h_5
              (bidirectional LSTM)
               ↑    ↑    ↑    ↑    ↑
Input:       x_1  x_2  x_3  x_4  x_5

                    [Context c_i] + [s_{i-1}]
                         ↓
Decoder:           s_i → output y_i
```

---

## 3. Luong Attention

### 3.1 Luong Attention (2015)

Simplified attention with multiple score functions.

```
Score functions:
  1. Dot:      e_{ij} = h_j^T × s_i
  2. General:  e_{ij} = h_j^T × W × s_i
  3. Concat:   e_{ij} = v^T × tanh(W × [h_j; s_i])

Attention: α_{ij} = softmax_j(e_{ij})
Context:   c_i = Σ_j α_{ij} × h_j
```

### 3.2 Luong vs Bahdanau

| Aspect | Bahdanau | Luong |
|--------|----------|-------|
| Score | Additive (MLP) | Multiplicative (dot/bilinear) |
| Decoder input | Previous state s_{i-1} | Current state s_i |
| Context usage | Input to decoder RNN | Concatenated after RNN |
| Complexity | Higher (MLP) | Lower (dot product) |

### 3.3 Luong Architecture Variants

**Global Attention**: Attend to all encoder states
```
Context = weighted sum of ALL encoder states
```

**Local Attention**: Attend to window around predicted position
```
1. Predict aligned position p_t
2. Context = weighted sum of states in [p_t - D, p_t + D]
More efficient for long sequences
```

---

## 4. Scaled Dot-Product Attention

### 4.1 The Transformer Attention

The attention mechanism used in "Attention Is All You Need" (2017):

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

where:
  Q = queries (what we're looking for)
  K = keys (what we match against)
  V = values (what we retrieve)
  d_k = dimension of keys
```

### 4.2 Why Scale by √d_k?

**Problem**: When d_k is large, dot products grow large → softmax saturates.

```
Query q, Key k ∈ R^{d_k}
E[q × k] = 0
Var[q × k] = d_k  (assuming unit variance components)

Large variance → some scores much larger than others
→ softmax concentrates on few positions (gradients vanish)

Solution: Divide by √d_k to maintain unit variance
Var[(q × k) / √d_k] = 1
```

### 4.3 Step-by-Step Computation

```
Input:
  Q: (batch, seq_q, d_k)
  K: (batch, seq_k, d_k)
  V: (batch, seq_v, d_v)  where seq_k = seq_v

Step 1: Compute attention scores
  scores = Q @ K^T
  Shape: (batch, seq_q, seq_k)

Step 2: Scale
  scores = scores / √d_k

Step 3: (Optional) Apply mask
  scores = scores + mask  (mask has -inf where we don't want attention)

Step 4: Softmax
  weights = softmax(scores, dim=-1)
  Shape: (batch, seq_q, seq_k)

Step 5: Apply to values
  output = weights @ V
  Shape: (batch, seq_q, d_v)
```

### 4.4 Masking

**Padding Mask**: Ignore padded positions
```
mask[i, j] = -inf if position j is padding
```

**Causal Mask**: For autoregressive models (GPT)
```
mask[i, j] = -inf if j > i  (can't look at future)

Example (seq_len=4):
       j=0  j=1  j=2  j=3
i=0 [  0,  -∞,  -∞,  -∞ ]
i=1 [  0,   0,  -∞,  -∞ ]
i=2 [  0,   0,   0,  -∞ ]
i=3 [  0,   0,   0,   0 ]
```

---

## 5. Multi-Head Attention

### 5.1 Why Multiple Heads?

Single attention head:
- Can only learn one type of relationship
- Queries everything the same way

Multiple heads:
- Each head learns different attention patterns
- Head 1: syntactic relationships
- Head 2: semantic relationships
- Head 3: positional patterns
- etc.

### 5.2 Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

where head_i = Attention(Q × W^Q_i, K × W^K_i, V × W^V_i)

Dimensions:
  Q, K, V: (batch, seq, d_model)
  W^Q_i, W^K_i: (d_model, d_k)
  W^V_i: (d_model, d_v)
  W^O: (h × d_v, d_model)

Typically: d_k = d_v = d_model / h
```

### 5.3 Parallel Head Computation

Efficient implementation computes all heads in parallel:

```python
# Instead of looping over heads:
# Project Q, K, V to all heads at once
Q_all = Q @ W_Q  # (batch, seq, h * d_k)
K_all = K @ W_K
V_all = V @ W_V

# Reshape to (batch, h, seq, d_k)
Q_heads = Q_all.view(batch, seq, h, d_k).transpose(1, 2)
K_heads = K_all.view(batch, seq, h, d_k).transpose(1, 2)
V_heads = V_all.view(batch, seq, h, d_v).transpose(1, 2)

# Compute attention for all heads in parallel
# (batch, h, seq_q, d_k) @ (batch, h, d_k, seq_k)
scores = Q_heads @ K_heads.transpose(-2, -1) / sqrt(d_k)
```

### 5.4 Number of Heads

Common configurations:
```
Model          d_model    h (heads)    d_k = d_model/h
BERT-base        768        12              64
BERT-large      1024        16              64
GPT-2 small      768        12              64
GPT-3           12288       96             128
LLaMA-7B        4096        32             128
LLaMA-70B       8192        64             128
```

**Rule of thumb**: d_k = 64 or 128 typically works well.

---

## 6. Self-Attention vs Cross-Attention

### 6.1 Self-Attention

Query, Key, Value all come from SAME sequence:
```
Self-Attention: Attention(X × W_Q, X × W_K, X × W_V)

Each position attends to all positions in same sequence
Used in: encoders, decoder self-attention
```

**Example**: Sentence "The cat sat"
```
Position 1 ("The") attends to: "The", "cat", "sat"
Position 2 ("cat") attends to: "The", "cat", "sat"
Position 3 ("sat") attends to: "The", "cat", "sat"
```

### 6.2 Cross-Attention

Query from one sequence, Key/Value from another:
```
Cross-Attention: Attention(Q_decoder × W_Q, K_encoder × W_K, V_encoder × W_V)

Decoder positions attend to encoder positions
Used in: encoder-decoder models (translation, etc.)
```

**Example**: Translation "cat" → "chat"
```
Decoder position ("chat") attends to: encoder positions ("the", "cat", "sat")
```

### 6.3 Causal (Masked) Self-Attention

Self-attention with causal mask:
```
Position i can only attend to positions 0, 1, ..., i
Cannot look at future positions
Used in: GPT, decoder-only models
```

### 6.4 Comparison Table

| Type | Q Source | K, V Source | Mask | Use Case |
|------|----------|-------------|------|----------|
| Self-Attention | X | X | None/Padding | BERT encoder |
| Causal Self-Attention | X | X | Causal | GPT decoder |
| Cross-Attention | Decoder | Encoder | Padding | T5, translation |

---

## 7. Attention Visualizations

### 7.1 Attention Patterns

Common patterns learned by attention heads:

**1. Position Attention**:
```
Attends to specific relative positions
e.g., "previous word", "two words back"
```

**2. Content-Based Attention**:
```
Attends based on token similarity
e.g., same word, similar words
```

**3. Syntactic Attention**:
```
Attends to syntactically related tokens
e.g., verb to subject, adjective to noun
```

**4. Delimiter Attention**:
```
Special tokens (CLS, SEP) aggregate info
Other tokens attend to special tokens
```

### 7.2 Attention Head Specialization

Research shows different heads specialize:
```
- Some heads attend to previous/next token
- Some heads attend to sentence start
- Some heads capture syntactic dependencies
- Some heads are "attention sinks" (always high on certain positions)
```

### 7.3 Interpretation Caveats

**Warning**: Attention weights ≠ importance
- Attention shows where info flows
- Not necessarily what info is used
- Other interpretability methods needed (probing, ablation)

---

## 8. Interview Questions

### Q1: Explain the difference between additive and multiplicative attention.

**Answer**:

**Additive (Bahdanau)**:
```
score = v^T × tanh(W_1 × query + W_2 × key)

- Uses a small neural network
- Query and key combined by addition
- More parameters (W_1, W_2, v)
- Theoretically more expressive
```

**Multiplicative (Luong/Dot-Product)**:
```
score = query^T × key
or
score = query^T × W × key  (bilinear)

- Direct dot product
- Fewer parameters
- Faster computation
- Works well in practice
```

**Transformer uses scaled dot-product**:
```
score = (query × key^T) / √d_k

Scaling prevents softmax saturation
Standard in modern architectures
```

### Q2: Why do we scale by √d_k in attention?

**Answer**:

**Problem**: Dot products grow with dimension.

For vectors q, k with components ~ N(0, 1):
```
q × k = Σ_i q_i × k_i

E[q × k] = 0
Var[q × k] = d_k  (each term contributes variance 1)
```

When d_k is large (e.g., 512):
- Dot products have high variance
- Some scores become very large
- Softmax concentrates on few positions
- Gradients become very small

**Solution**: Scale by √d_k
```
Var[(q × k) / √d_k] = Var[q × k] / d_k = 1

Maintains reasonable variance regardless of dimension
```

### Q3: What is multi-head attention and why is it useful?

**Answer**:

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

head_i = Attention(Q × W^Q_i, K × W^K_i, V × W^V_i)
```

**Why multiple heads?**:

1. **Diverse representations**: Each head learns different attention patterns
   - Syntactic patterns (subject-verb)
   - Semantic patterns (word similarity)
   - Positional patterns (local context)

2. **Expressiveness**: One head with d-dimensional vectors can only
   represent one attention pattern. h heads can represent h patterns.

3. **Stable training**: Averaging over multiple heads reduces variance

4. **Computational efficiency**: Despite h heads, same total computation
   - Each head: d_k = d_model/h
   - Total: h × (d_model/h)² = d_model²/h (actually similar to single head)

### Q4: Explain self-attention vs cross-attention.

**Answer**:

**Self-Attention**:
```
Q, K, V all derived from SAME input sequence X

Attention(XW_Q, XW_K, XW_V)

Each position attends to all positions in same sequence
Example: Word "cat" attends to "the", "cat", "sat" in same sentence
```

**Cross-Attention**:
```
Q from one sequence, K and V from another

Attention(YW_Q, XW_K, XW_V)

Decoder Y attends to encoder X
Example: French "chat" attends to English "the", "cat", "sat"
```

**Use cases**:
- Self-attention: BERT encoder, GPT decoder
- Cross-attention: Encoder-decoder models (T5, BART, translation)

**In GPT (decoder-only)**:
- Only self-attention, no cross-attention
- Causal mask prevents attending to future

### Q5: What is causal masking and why is it needed?

**Answer**:

**Causal masking** prevents attention to future positions:
```
mask[i, j] = -∞ if j > i, else 0

After softmax:
α[i, j] = 0 if j > i

Position i can only see positions 0, 1, ..., i
```

**Why needed**:

1. **Autoregressive generation**: Model predicts one token at a time
   - At position i, only tokens 0..i-1 are known
   - Must not "cheat" by looking ahead

2. **Training with teacher forcing**:
   - All positions computed in parallel
   - But each position should only use past context
   - Causal mask enforces this constraint

3. **Consistency**: Train and inference use same attention pattern

**Without causal mask**: Model would learn to "cheat" during training
by looking at the answer, making generation at inference impossible.

### Q6: Compare attention mechanisms to RNNs.

**Answer**:

| Aspect | RNN | Attention |
|--------|-----|-----------|
| **Sequential** | Yes (t depends on t-1) | No (all parallel) |
| **Long-range** | Difficult (vanishing gradient) | Easy (direct connection) |
| **Path length** | O(n) steps between positions | O(1) direct attention |
| **Computation** | Sequential | Parallel (GPU friendly) |
| **Memory** | O(1) hidden state | O(n²) attention matrix |

**Why attention replaced RNNs for NLP**:

1. **Parallelization**: RNNs inherently sequential, attention can process
   all positions simultaneously

2. **Long-range dependencies**: Attention provides direct path between
   any two positions, regardless of distance

3. **Gradient flow**: No vanishing gradient through time steps

4. **Training speed**: Much faster due to parallelization

**Trade-off**: Attention uses O(n²) memory for attention matrix,
which is problematic for very long sequences.

---

## 9. Summary

### Quick Reference

| Attention Type | Formula | Complexity |
|---------------|---------|------------|
| Additive (Bahdanau) | v^T tanh(W_q q + W_k k) | O(d) per pair |
| Dot-Product | q^T k | O(d) per pair |
| Scaled Dot-Product | (q^T k) / √d | O(d) per pair |
| Multi-Head | Concat(head_i) W_O | O(d²) total |

### Key Equations

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Causal Mask**:
```
mask[i,j] = 0 if j ≤ i else -∞
```

### Key Takeaways

1. **Attention solves the bottleneck**: No need to compress entire sequence

2. **Scaled dot-product is standard**: √d_k scaling prevents saturation

3. **Multi-head captures diverse patterns**: Each head learns different relationships

4. **Self vs Cross**:
   - Self: sequence attends to itself (encoders, decoders)
   - Cross: one sequence attends to another (encoder-decoder)

5. **Causal masking for autoregressive**: Prevents looking at future

6. **Attention enables transformers**: Foundation for BERT, GPT, all LLMs
