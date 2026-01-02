# Module 6.1: Recurrent Neural Networks - RNN, LSTM, GRU

## Table of Contents
1. [Why Recurrence?](#1-why-recurrence)
2. [Vanilla RNN](#2-vanilla-rnn)
3. [Vanishing Gradient Problem](#3-vanishing-gradient-problem)
4. [LSTM](#4-lstm)
5. [GRU](#5-gru)
6. [Practical Considerations](#6-practical-considerations)
7. [From RNNs to Transformers](#7-from-rnns-to-transformers)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Why Recurrence?

### 1.1 The Sequence Problem

Standard neural networks:
- Fixed input/output size
- No memory of previous inputs
- Can't handle variable-length sequences

**Sequential data** requires:
- Variable length handling
- Memory of past information
- Order-sensitive processing

### 1.2 Applications

| Task | Input | Output |
|------|-------|--------|
| Language modeling | Words 1...t | Word t+1 |
| Sentiment | Sentence | Positive/Negative |
| Translation | English sequence | French sequence |
| Speech recognition | Audio frames | Text |
| Time series | Past values | Future values |

### 1.3 Key Idea: Weight Sharing

Process sequences by:
1. Using same weights at each time step
2. Maintaining hidden state (memory)
3. Updating state based on current input + previous state

---

## 2. Vanilla RNN

### 2.1 Architecture

```
At each time step t:
  h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
  y_t = W_hy × h_t + b_y

where:
  x_t = input at time t
  h_t = hidden state at time t
  y_t = output at time t
  W_hh = hidden-to-hidden weights
  W_xh = input-to-hidden weights
  W_hy = hidden-to-output weights
```

### 2.2 Unrolled Computation

```
x_0 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3
        ↓            ↓            ↓
       y_0          y_1          y_2

Same weights at each step!
```

### 2.3 Forward Pass

```python
def rnn_forward(x, h_prev, W_xh, W_hh, W_hy):
    # Hidden state
    h = tanh(W_xh @ x + W_hh @ h_prev)
    # Output
    y = W_hy @ h
    return h, y
```

### 2.4 RNN Variants by Task

**Many-to-One** (Classification):
```
x_1 → x_2 → x_3 → x_4 → [h_4] → y
Only final output used
```

**Many-to-Many** (Language Model):
```
x_1 → x_2 → x_3 → x_4
 ↓     ↓     ↓     ↓
y_1   y_2   y_3   y_4
Output at each step
```

**Encoder-Decoder** (Translation):
```
[Encoder: x_1 → x_2 → x_3] → [final_h] → [Decoder: y_1 → y_2 → y_3]
```

---

## 3. Vanishing Gradient Problem

### 3.1 The Problem

Backpropagation through time (BPTT):
```
∂L/∂W = Σ_t ∂L_t/∂W

∂L_t/∂h_1 = ∂L_t/∂h_t × ∂h_t/∂h_{t-1} × ... × ∂h_2/∂h_1

Each ∂h_i/∂h_{i-1} involves W_hh and tanh derivative
```

### 3.2 Why Gradients Vanish

```
∂h_t/∂h_{t-1} = W_hh × diag(1 - h_t²)   (tanh derivative)

If eigenvalues of W_hh < 1:
  Product of many terms → 0

If eigenvalues of W_hh > 1:
  Product of many terms → ∞ (exploding)
```

### 3.3 Consequences

**Vanishing gradients**:
- Early layers don't learn
- Can't capture long-range dependencies
- "Long-term memory" is lost

**Exploding gradients**:
- NaN losses
- Unstable training
- Solution: Gradient clipping

### 3.4 Why This Matters

```
"The cat, which sat on the mat, was happy."
         ↑___________________________↑
         Long-range dependency

RNN must remember "cat" through many steps to predict "was" (singular)
Vanishing gradients make this hard
```

---

## 4. LSTM (Long Short-Term Memory)

### 4.1 Key Insight

Add **gating mechanism** to control information flow:
- **Forget gate**: What to forget from cell state
- **Input gate**: What new info to add
- **Output gate**: What to output

### 4.2 Architecture

```
Cell state c_t: "Long-term memory" (highway)
Hidden state h_t: "Short-term memory" (output)

Gates:
  f_t = σ(W_f × [h_{t-1}, x_t])    # Forget gate
  i_t = σ(W_i × [h_{t-1}, x_t])    # Input gate
  o_t = σ(W_o × [h_{t-1}, x_t])    # Output gate

Cell update:
  c̃_t = tanh(W_c × [h_{t-1}, x_t])  # Candidate cell
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t   # New cell state

Output:
  h_t = o_t ⊙ tanh(c_t)             # Hidden state
```

### 4.3 Gate Intuition

**Forget gate (f_t)**:
```
f_t close to 1: Keep old cell state
f_t close to 0: Forget old cell state

Example: New sentence starts → forget previous subject
```

**Input gate (i_t)**:
```
i_t close to 1: Add new information
i_t close to 0: Ignore new information

Example: Important word → write to memory
```

**Output gate (o_t)**:
```
o_t close to 1: Output cell content
o_t close to 0: Don't output

Example: Processing internal state but not ready to output
```

### 4.4 Why LSTM Solves Vanishing Gradients

**Cell state highway**:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Gradient flows through c_t without multiplication by W!
If f_t ≈ 1: c_t ≈ c_{t-1} (direct copy)

Gradient: ∂c_t/∂c_{t-1} = f_t (not weight matrix!)
```

When f_t ≈ 1, gradient flows unchanged → no vanishing!

### 4.5 LSTM Initialization

**Critical**: Initialize forget gate bias to 1
```python
lstm.bias_ih[hidden_size:2*hidden_size] = 1.0
lstm.bias_hh[hidden_size:2*hidden_size] = 1.0

# This makes f_0 ≈ sigmoid(1) ≈ 0.73
# Starts with "mostly remember" behavior
```

Without this, LSTM may forget everything initially → hard to learn.

---

## 5. GRU (Gated Recurrent Unit)

### 5.1 Simplified Gating

GRU combines forget and input gates:
- **Reset gate**: How much past state to ignore
- **Update gate**: How much to update state

```
r_t = σ(W_r × [h_{t-1}, x_t])    # Reset gate
z_t = σ(W_z × [h_{t-1}, x_t])    # Update gate

h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t])  # Candidate

h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # New hidden state
```

### 5.2 GRU vs LSTM

```
LSTM:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t  (separate cell and hidden)
  h_t = o_t ⊙ tanh(c_t)

GRU:
  h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  (combined)
```

### 5.3 Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | 4 gates | 2 gates |
| States | c (cell) + h (hidden) | h only |
| Memory | ~33% more | Fewer |
| Performance | Slightly better (often) | Similar |
| Training | Slower | Faster |

**Rule of thumb**:
- Try GRU first (faster)
- Use LSTM if GRU doesn't work
- Both outperform vanilla RNN

---

## 6. Practical Considerations

### 6.1 Bidirectional RNNs

Process sequence in both directions:
```
Forward:  x_1 → x_2 → x_3 → x_4
          →h_1  →h_2  →h_3  →h_4

Backward: x_1 ← x_2 ← x_3 ← x_4
          ←h_1  ←h_2  ←h_3  ←h_4

Final: [→h_t, ←h_t] concatenated
```

**Use case**: When full context is available (not autoregressive generation).

### 6.2 Stacking RNN Layers

```
Layer 2: h_1² → h_2² → h_3² → h_4²
            ↑      ↑      ↑      ↑
Layer 1: h_1¹ → h_2¹ → h_3¹ → h_4¹
            ↑      ↑      ↑      ↑
Input:    x_1    x_2    x_3    x_4
```

Deeper networks capture more complex patterns.

### 6.3 Dropout in RNNs

**Correct way**: Dropout on input/output, NOT recurrent connections
```python
# Standard dropout between layers (good)
x = dropout(lstm_layer(x))

# Variational dropout (same mask across time steps)
# Better for recurrent connections
```

### 6.4 Truncated BPTT

For very long sequences, truncate backprop:
```python
# Process in chunks
for chunk in chunks(sequence, chunk_size=100):
    output, hidden = rnn(chunk, hidden.detach())  # Detach!
    loss.backward()
```

**Why detach?**: Limits gradient computation to chunk_size steps.

### 6.5 Sequence Padding and Packing

Efficient batch processing:
```python
# Pack variable-length sequences
packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
output, hidden = lstm(packed)
# Unpack
output, _ = pad_packed_sequence(output)
```

---

## 7. From RNNs to Transformers

### 7.1 RNN Limitations

| Limitation | Explanation |
|------------|-------------|
| **Sequential** | Can't parallelize across time |
| **Long-range** | Even LSTM struggles with 100+ steps |
| **Training speed** | Must process sequentially |
| **Gradient flow** | Still imperfect despite gates |

### 7.2 Attention Solves These

Transformers use attention instead of recurrence:
- **Parallel**: All positions processed simultaneously
- **Direct connections**: Any position attends to any other
- **Constant path length**: O(1) between any two positions

### 7.3 Historical Context

```
1990s: Vanilla RNN
1997: LSTM introduced
2014: GRU introduced
2014: Attention in seq2seq (Bahdanau)
2017: Transformer ("Attention Is All You Need")
2018+: Transformers dominate (BERT, GPT)
```

**Today**: RNNs rarely used for NLP; transformers are standard.

---

## 8. Interview Questions

### Q1: Explain the vanishing gradient problem in RNNs.

**Answer**:

In backpropagation through time:
```
∂L/∂h_1 = ∂L/∂h_T × ∂h_T/∂h_{T-1} × ... × ∂h_2/∂h_1
```

Each term ∂h_t/∂h_{t-1} = W_hh × tanh'(·)

**Problem**: Repeated matrix multiplication:
- If max eigenvalue of W_hh < 1: gradient → 0 (vanishing)
- If max eigenvalue of W_hh > 1: gradient → ∞ (exploding)

**Consequences**:
- Can't learn long-range dependencies
- Early time steps receive no gradient signal

### Q2: How does LSTM solve the vanishing gradient problem?

**Answer**:

LSTM introduces a **cell state highway**:
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

**Key insight**: Gradient through cell state:
```
∂c_t/∂c_{t-1} = f_t  (element-wise, NOT matrix multiplication)
```

When f_t ≈ 1:
- c_t ≈ c_{t-1} (cell state passes through unchanged)
- Gradient flows directly without attenuation

**Additional benefits**:
- Gates learn what to remember/forget
- Can preserve information over many steps
- Forget gate bias = 1 helps initially

### Q3: Compare LSTM and GRU.

**Answer**:

**LSTM**:
```
- 4 gates: forget, input, output, cell candidate
- 2 states: cell (c) and hidden (h)
- More parameters
- Separate memory and output
```

**GRU**:
```
- 2 gates: reset, update
- 1 state: hidden (h) only
- Fewer parameters (~75% of LSTM)
- Combined memory and output
```

**Comparison**:
| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters | More | Fewer |
| Speed | Slower | Faster |
| Performance | Slightly better | Similar |
| Complexity | Higher | Lower |

**Recommendation**: Start with GRU; use LSTM if needed.

### Q4: Why is the forget gate bias initialized to 1?

**Answer**:

**Default initialization** (bias = 0):
- sigmoid(0) = 0.5
- f_t ≈ 0.5 → forgets half the cell state each step
- After 10 steps: 0.5^10 ≈ 0.001 (almost nothing remains)
- LSTM can't remember long-term information

**Bias = 1 initialization**:
- sigmoid(1) ≈ 0.73
- f_t ≈ 0.73 → keeps most of cell state
- Starts in "remember by default" mode
- Network can learn to forget when needed

This is **critical** for learning long-term dependencies.

### Q5: Why have transformers replaced RNNs?

**Answer**:

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| **Parallelization** | Sequential | Fully parallel |
| **Training speed** | Slow | Fast |
| **Long-range** | Difficult | Easy |
| **Path length** | O(n) | O(1) |
| **Gradient flow** | Through gates | Direct |

**RNN bottleneck**: Must process sequentially → can't utilize modern GPUs.

**Transformer advantages**:
- Process all positions simultaneously
- Direct attention between any positions
- Better gradient flow
- Scales better with compute

**Result**: Transformers are now standard for NLP (BERT, GPT, etc.).

---

## 9. Summary

### Quick Reference

| Architecture | Formula | Key Feature |
|--------------|---------|-------------|
| Vanilla RNN | h_t = tanh(Wx + Wh) | Simple, vanishing gradient |
| LSTM | c_t = f⊙c + i⊙c̃ | Cell state highway |
| GRU | h_t = (1-z)⊙h + z⊙h̃ | Fewer parameters |

### Key Equations

**RNN**:
```
h_t = tanh(W_xh × x_t + W_hh × h_{t-1})
```

**LSTM**:
```
f_t = σ(W_f × [h_{t-1}, x_t])
i_t = σ(W_i × [h_{t-1}, x_t])
o_t = σ(W_o × [h_{t-1}, x_t])
c̃_t = tanh(W_c × [h_{t-1}, x_t])
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
h_t = o_t ⊙ tanh(c_t)
```

**GRU**:
```
r_t = σ(W_r × [h_{t-1}, x_t])
z_t = σ(W_z × [h_{t-1}, x_t])
h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t])
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### Key Takeaways

1. **Vanilla RNN**: Simple but vanishing gradients
2. **LSTM**: Cell state highway solves vanishing gradients
3. **GRU**: Simpler LSTM alternative, often works as well
4. **Forget gate bias = 1**: Critical for LSTM
5. **Transformers replaced RNNs**: Parallel, better long-range
