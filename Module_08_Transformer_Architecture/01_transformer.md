# Module 8.1: The Transformer Architecture

## Table of Contents
1. [Overview](#1-overview)
2. [Input Embeddings & Positional Encoding](#2-input-embeddings--positional-encoding)
3. [Encoder Architecture](#3-encoder-architecture)
4. [Decoder Architecture](#4-decoder-architecture)
5. [Full Transformer](#5-full-transformer)
6. [Training Considerations](#6-training-considerations)
7. [Transformer Variants](#7-transformer-variants)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Overview

### 1.1 The Paper

"Attention Is All You Need" (Vaswani et al., 2017):
- Introduced the Transformer architecture
- Replaced RNNs with self-attention
- Enabled parallel computation
- Foundation for BERT, GPT, and all modern LLMs

### 1.2 Key Innovations

```
1. Self-Attention: Each position attends to all positions
2. Multi-Head Attention: Multiple parallel attention heads
3. Positional Encoding: Add position information (no recurrence)
4. Layer Normalization: Stabilize training
5. Residual Connections: Enable gradient flow
```

### 1.3 Architecture Overview

```
Original Transformer (Encoder-Decoder):

Input:  [tokens] → Embedding → + Positional Encoding
                                    ↓
Encoder: [ Multi-Head Self-Attention ]
         [ Add & Norm                ]
         [ Feed Forward              ]  × N layers
         [ Add & Norm                ]
                    ↓
              Encoder Output
                    ↓
Decoder: [ Masked Multi-Head Self-Attention ]
         [ Add & Norm                       ]
         [ Multi-Head Cross-Attention       ] ← (from encoder)
         [ Add & Norm                       ]  × N layers
         [ Feed Forward                     ]
         [ Add & Norm                       ]
                    ↓
Output:  Linear → Softmax → [probabilities]
```

### 1.4 Original Configuration

```
Parameter             Value
d_model              512
d_ff                 2048
num_heads            8
num_layers           6 (encoder) + 6 (decoder)
d_k = d_v            64 (d_model / num_heads)
dropout              0.1
vocab_size           ~37000 (BPE)
max_seq_len          512
```

---

## 2. Input Embeddings & Positional Encoding

### 2.1 Token Embeddings

Convert tokens to dense vectors:
```
token_id → Embedding Matrix (vocab_size × d_model) → embedding vector

Scaling: embeddings × √d_model
(Compensates for small embedding magnitudes)
```

### 2.2 Why Positional Encoding?

**Problem**: Self-attention is permutation-invariant:
```
Attention([A, B, C]) = Attention([C, B, A])

Without position info, "dog bites man" = "man bites dog"
```

**Solution**: Add positional information to embeddings.

### 2.3 Sinusoidal Positional Encoding

The original paper uses sinusoidal functions:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
  pos = position in sequence (0, 1, 2, ...)
  i = dimension index (0, 1, 2, ..., d_model/2)
```

### 2.4 Why Sinusoidal?

**Key property**: PE[pos + k] can be expressed as linear function of PE[pos]:
```
PE[pos + k] = f(PE[pos])  (linear transformation)

This allows the model to learn relative positions
```

**Benefits**:
- No learned parameters
- Generalizes to longer sequences than seen in training
- Encodes both absolute and relative positions

### 2.5 Learned vs Sinusoidal

| Type | Description | Used In |
|------|-------------|---------|
| Sinusoidal | Fixed, computed from formulas | Original Transformer |
| Learned | Trainable embedding matrix | BERT, GPT-2 |
| Rotary (RoPE) | Rotation-based, relative | LLaMA, GPT-NeoX |
| ALiBi | Attention bias, no embedding | BLOOM, MPT |

---

## 3. Encoder Architecture

### 3.1 Encoder Layer

Each encoder layer has two sublayers:
```
Sublayer 1: Multi-Head Self-Attention
Sublayer 2: Position-wise Feed-Forward Network

Each sublayer has:
- Residual connection: output = x + Sublayer(x)
- Layer normalization: LayerNorm(output)
```

### 3.2 Multi-Head Self-Attention (in Encoder)

```
Input X: (batch, seq_len, d_model)

Q = X × W_Q  (queries)
K = X × W_K  (keys)
V = X × W_V  (values)

Attention = softmax(Q × K^T / √d_k) × V
MultiHead = Concat(head_1, ..., head_h) × W_O
```

**No masking in encoder**: All positions can attend to all other positions.

### 3.3 Position-wise Feed-Forward Network

Applied to each position independently:
```
FFN(x) = max(0, x × W_1 + b_1) × W_2 + b_2

or with GELU (modern):
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2

Dimensions:
  Input: d_model
  Hidden: d_ff (typically 4 × d_model)
  Output: d_model
```

### 3.4 Pre-Norm vs Post-Norm

**Post-Norm (Original)**:
```python
x = x + Sublayer(x)
x = LayerNorm(x)
```

**Pre-Norm (Modern)**:
```python
x = x + Sublayer(LayerNorm(x))
```

**Pre-Norm advantages**:
- More stable training
- Easier gradient flow
- Often no warmup needed
- Used in GPT-2, LLaMA, etc.

### 3.5 Complete Encoder Layer

```python
class EncoderLayer:
    def forward(self, x, mask=None):
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x
```

---

## 4. Decoder Architecture

### 4.1 Decoder Layer

Each decoder layer has THREE sublayers:
```
Sublayer 1: Masked Multi-Head Self-Attention
Sublayer 2: Multi-Head Cross-Attention (to encoder)
Sublayer 3: Position-wise Feed-Forward Network
```

### 4.2 Masked Self-Attention

**Causal masking** prevents attending to future positions:
```
mask[i, j] = -∞ if j > i, else 0

After softmax: position i can only attend to positions 0, 1, ..., i
```

**Why needed?**
- Decoder generates tokens autoregressively
- During training (teacher forcing), all positions computed in parallel
- Mask ensures no "cheating" by looking at future tokens

### 4.3 Cross-Attention

Decoder attends to encoder output:
```
Q = Decoder hidden states
K, V = Encoder output

This allows decoder to "look at" the source sequence
```

**In encoder-decoder models**: Cross-attention connects encoder and decoder.
**In decoder-only models (GPT)**: No cross-attention.

### 4.4 Complete Decoder Layer

```python
class DecoderLayer:
    def forward(self, x, encoder_output, self_mask, cross_mask):
        # Masked self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x, x, x, self_mask)  # Causal mask
        x = self.dropout(x)
        x = residual + x

        # Cross-attention to encoder
        residual = x
        x = self.norm2(x)
        x = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.dropout(x)
        x = residual + x

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x
```

---

## 5. Full Transformer

### 5.1 Encoder Stack

```python
class Encoder:
    def forward(self, src, src_mask):
        x = self.embedding(src) * sqrt(d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.final_norm(x)  # Pre-norm requires final norm
```

### 5.2 Decoder Stack

```python
class Decoder:
    def forward(self, tgt, encoder_output, tgt_mask, cross_mask):
        x = self.embedding(tgt) * sqrt(d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, cross_mask)

        return self.final_norm(x)
```

### 5.3 Full Model

```python
class Transformer:
    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, cross_mask)
        logits = self.output_projection(decoder_output)
        return logits
```

### 5.4 Output Layer

**Weight Tying**: Share weights between:
- Input embedding matrix
- Output projection matrix

```python
# Weight tying
self.output_proj.weight = self.embedding.weight

Benefits:
- Fewer parameters
- Better generalization
- Standard practice
```

---

## 6. Training Considerations

### 6.1 Label Smoothing

Soften target distribution:
```
Instead of:
  target = [0, 0, 1, 0, 0]  (one-hot)

Use:
  target = [0.025, 0.025, 0.9, 0.025, 0.025]

Smoothing factor ε = 0.1 (original paper)
```

**Benefits**:
- Prevents overconfidence
- Better calibration
- Slight regularization

### 6.2 Learning Rate Schedule

**Warmup + Decay**:
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

warmup_steps = 4000 (original)

Effect:
- Linear warmup for first 4000 steps
- Then inverse square root decay
```

### 6.3 Dropout

Applied in multiple places:
```
1. After embeddings + positional encoding
2. After attention (on attention weights)
3. After each sublayer
4. Inside FFN (between layers)

Typical dropout rate: 0.1
```

### 6.4 Initialization

```python
# Linear layers
nn.init.xavier_uniform_(linear.weight)

# Embeddings
nn.init.normal_(embedding.weight, std=0.02)

# Layer norm
nn.init.ones_(norm.weight)
nn.init.zeros_(norm.bias)
```

### 6.5 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients, especially early in training.

---

## 7. Transformer Variants

### 7.1 Encoder-Only (BERT)

```
Architecture:
- Only encoder stack
- Bidirectional attention (no masking)
- Output: contextualized representations

Used for:
- Classification
- Named Entity Recognition
- Question Answering
```

### 7.2 Decoder-Only (GPT)

```
Architecture:
- Only decoder stack (without cross-attention)
- Causal masking (autoregressive)
- Output: next token predictions

Used for:
- Language modeling
- Text generation
- Modern LLMs (GPT-3, LLaMA, Claude)
```

### 7.3 Encoder-Decoder (T5, BART)

```
Architecture:
- Full encoder + decoder
- Cross-attention connects them

Used for:
- Translation
- Summarization
- Seq2seq tasks
```

### 7.4 Comparison

| Model | Encoder | Decoder | Attention Type | Best For |
|-------|---------|---------|----------------|----------|
| BERT | Yes | No | Bidirectional | Understanding |
| GPT | No | Yes | Causal | Generation |
| T5 | Yes | Yes | Both | Seq2seq |

### 7.5 Modern Modifications

```
Original → Modern LLMs:
- Post-norm → Pre-norm
- ReLU → GELU/SiLU
- Sinusoidal PE → RoPE
- Separate Q/K/V → Fused QKV
- Standard Attention → GQA/MQA
- LayerNorm → RMSNorm
- FFN → SwiGLU
```

---

## 8. Interview Questions

### Q1: Explain the transformer architecture.

**Answer**:

The transformer consists of:

**Encoder** (N layers, each with):
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections + layer normalization

**Decoder** (N layers, each with):
1. Masked multi-head self-attention (causal)
2. Multi-head cross-attention (to encoder)
3. Position-wise feed-forward network
4. Residual connections + layer normalization

**Key innovations**:
- Self-attention replaces recurrence (parallel computation)
- Positional encoding adds position information
- Multi-head attention captures diverse relationships

### Q2: Why is positional encoding necessary?

**Answer**:

**Problem**: Self-attention is permutation-invariant:
```
Attention("dog bites man") = Attention("man bites dog")
```

Without position info, the model can't distinguish word order.

**Solution**: Add positional encoding to embeddings:
```
input = token_embedding + positional_encoding
```

**Sinusoidal encoding** (original):
- Uses sin/cos functions at different frequencies
- Generalizes to longer sequences
- Encodes relative positions (PE[pos+k] is linear function of PE[pos])

**Modern alternatives**: Learned embeddings, RoPE, ALiBi

### Q3: What is the purpose of multi-head attention?

**Answer**:

**Single head limitation**: One attention head can only learn one type of relationship.

**Multi-head solution**: Run multiple attention heads in parallel, each learning different patterns:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O
head_i = Attention(Q×W_Q_i, K×W_K_i, V×W_V_i)
```

**What heads learn**:
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Positional patterns (previous word)
- Head 3: Semantic similarity
- etc.

**Configuration**: Typically d_k = d_model / num_heads (e.g., 512/8 = 64)

### Q4: Compare Pre-Norm and Post-Norm.

**Answer**:

**Post-Norm (Original)**:
```python
x = LayerNorm(x + Sublayer(x))
```
- Normalize after residual addition
- Used in original transformer
- Harder to train deep models
- Requires careful warmup

**Pre-Norm (Modern)**:
```python
x = x + Sublayer(LayerNorm(x))
```
- Normalize before sublayer
- Used in GPT-2, LLaMA, etc.
- More stable training
- Better gradient flow through residual

**Why Pre-Norm is preferred**:
- Residual path is "clean" (no normalization)
- Gradients flow directly through skip connections
- Can train very deep models without warmup

### Q5: Explain the difference between encoder-only, decoder-only, and encoder-decoder transformers.

**Answer**:

**Encoder-Only (BERT)**:
```
Input: [CLS] The cat sat [SEP]
       ↓
     Encoder (bidirectional attention)
       ↓
Output: Contextualized representations
```
- Bidirectional: each position sees all others
- Good for understanding tasks (classification, NER)
- Not suitable for generation

**Decoder-Only (GPT)**:
```
Input: The cat sat
       ↓
     Decoder (causal attention)
       ↓
Output: Next token probabilities
```
- Causal: each position only sees previous
- Good for generation (language modeling)
- Dominant architecture for LLMs

**Encoder-Decoder (T5)**:
```
Input: Translate: The cat sat
       ↓
     Encoder (bidirectional)
       ↓
     Decoder (causal + cross-attention)
       ↓
Output: Le chat s'est assis
```
- Encoder processes input bidirectionally
- Decoder generates output autoregressively
- Good for seq2seq (translation, summarization)

### Q6: Why do we scale attention by √d_k?

**Answer**:

**Problem**: Dot products grow with dimension.

For q, k with components ~ N(0, 1):
```
Var(q · k) = d_k

When d_k = 512: variance = 512
→ Some scores very large
→ Softmax saturates
→ Gradients vanish
```

**Solution**: Scale by √d_k:
```
Attention = softmax(QK^T / √d_k) × V

Var((q · k) / √d_k) = 1
```

Keeps softmax in well-behaved range regardless of dimension.

---

## 9. Summary

### Key Components

| Component | Purpose |
|-----------|---------|
| Self-Attention | Relate all positions |
| Multi-Head | Diverse attention patterns |
| Positional Encoding | Add position information |
| FFN | Non-linear transformation |
| Layer Norm | Stabilize training |
| Residual | Enable gradient flow |

### Key Equations

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
```

**Position-wise FFN**:
```
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2
```

**Positional Encoding**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Key Takeaways

1. **Self-attention enables parallelism**: No sequential dependency like RNNs
2. **Positional encoding adds position info**: Sinusoidal or learned
3. **Multi-head attention captures diversity**: Multiple attention patterns
4. **Pre-norm is preferred**: Better gradient flow, more stable training
5. **Three main variants**: Encoder-only, decoder-only, encoder-decoder
6. **Foundation for all LLMs**: BERT, GPT, T5, LLaMA all based on transformer
