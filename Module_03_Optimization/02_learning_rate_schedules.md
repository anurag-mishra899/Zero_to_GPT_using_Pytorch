# Module 3.2: Learning Rate Schedules

## Table of Contents
1. [Why Learning Rate Schedules?](#1-why-learning-rate-schedules)
2. [Classic Schedules](#2-classic-schedules)
3. [Modern Schedules](#3-modern-schedules)
4. [LLM Training Schedules](#4-llm-training-schedules)
5. [Advanced Techniques](#5-advanced-techniques)
6. [Practical Guidelines](#6-practical-guidelines)
7. [Interview Questions](#7-interview-questions)
8. [Summary](#8-summary)

---

## 1. Why Learning Rate Schedules?

### 1.1 The Learning Rate Dilemma

**High learning rate**:
- Fast initial progress
- Can overshoot optimal solutions
- May cause instability in later training

**Low learning rate**:
- Stable training
- Slow initial progress
- Fine convergence near optimum

**Solution**: Start high, decrease over time!

### 1.2 The Loss Landscape Perspective

```
Early training:
  - Far from optimum
  - Can tolerate large steps
  - Need to escape bad regions quickly

Late training:
  - Near optimum
  - Small steps needed for precision
  - Large steps cause overshooting
```

### 1.3 Types of Schedules

| Schedule Type | Behavior | Use Case |
|--------------|----------|----------|
| Constant | lr stays same | Debugging |
| Step Decay | Discrete drops | CNNs |
| Exponential | Smooth decay | General |
| Cosine | Smooth curve | Transformers |
| Warmup + Decay | Increase then decrease | LLMs |
| Cyclic | Oscillating | Exploration |

---

## 2. Classic Schedules

### 2.1 Step Decay

```
lr = lr_0 × γ^(epoch // step_size)

Parameters:
  lr_0: Initial learning rate
  γ: Decay factor (typically 0.1)
  step_size: Epochs between drops

Example (ImageNet standard):
  lr_0 = 0.1
  γ = 0.1
  step_size = 30

  Epoch 0-29:  lr = 0.1
  Epoch 30-59: lr = 0.01
  Epoch 60-89: lr = 0.001
```

**Pros**:
- Simple to implement
- Well-understood behavior
- Works well for CNNs

**Cons**:
- Abrupt changes can destabilize
- Requires tuning step_size

### 2.2 Multi-Step Decay

```
lr = lr_0 × γ^(number of milestones passed)

milestones = [30, 60, 90]
γ = 0.1

More flexible than fixed step size
```

### 2.3 Exponential Decay

```
lr = lr_0 × γ^epoch

Parameters:
  γ: Decay factor per epoch (e.g., 0.95)

Smooth continuous decay
```

### 2.4 Polynomial Decay

```
lr = lr_0 × (1 - t/T)^power

Parameters:
  T: Total training steps
  power: Decay power (1 = linear, 2 = quadratic)

Linear decay (power=1) is common
```

---

## 3. Modern Schedules

### 3.1 Cosine Annealing

```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))

Parameters:
  lr_max: Maximum (starting) learning rate
  lr_min: Minimum (ending) learning rate
  t: Current step
  T: Total steps
```

**Properties**:
- Smooth, continuous decrease
- Starts slow, speeds up, then slows again
- No sharp transitions
- Standard for transformers

**Shape**:
```
lr_max |.
       |  .
       |    .
       |       .
       |           .
lr_min |______________._______
       0             T/2       T
```

### 3.2 Cosine with Restarts

```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t_i / T_i))

where:
  t_i: Steps since last restart
  T_i: Steps until next restart

T_i can grow: T_1, 2×T_1, 4×T_1, ...
```

**Benefit**: Escapes local minima by periodically increasing lr

### 3.3 OneCycle Learning Rate

```
Phase 1 (warmup): lr increases linearly from lr_min to lr_max
Phase 2 (annealing): lr decreases (cosine) from lr_max to lr_min
Phase 3 (final): lr stays at lr_min or decreases further

Also cycles momentum in opposite direction:
  High lr → Low momentum
  Low lr → High momentum
```

**Benefits**:
- Can use larger max_lr than normal training
- Often trains faster
- Good for limited compute budgets

---

## 4. LLM Training Schedules

### 4.1 Warmup + Cosine Decay (Standard)

The standard LLM schedule:

```
if t < warmup_steps:
    # Linear warmup
    lr = lr_max × (t / warmup_steps)
else:
    # Cosine decay
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_final + 0.5 × (lr_max - lr_final) × (1 + cos(π × progress))
```

**Typical parameters**:
```
lr_max = 3e-4
lr_final = 1e-5 or lr_max / 10
warmup_steps = 2000 (or 1-2% of total)
```

### 4.2 Why Warmup?

**Problem without warmup**:
- Adam's adaptive statistics start at zero
- Early gradients have high variance
- Large updates can destabilize training

**Warmup solution**:
- Start with small lr
- Gradually increase as statistics stabilize
- Typically 1-5% of training

**Warmup duration guidelines**:
```
Small models: 1000-2000 steps
Large models: 2000-5000 steps
Percentage: 1-5% of total training
```

### 4.3 Linear Warmup + Linear Decay

Simpler alternative:

```
if t < warmup_steps:
    lr = lr_max × (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_max × (1 - progress) + lr_final × progress
```

### 4.4 WSD (Warmup-Stable-Decay)

Three-phase schedule:

```
Phase 1 (Warmup): Linear increase to lr_max
Phase 2 (Stable): Keep at lr_max
Phase 3 (Decay): Cosine decay to lr_final

Allows more training at peak learning rate
```

### 4.5 Inverse Square Root (Original Transformer)

From "Attention Is All You Need":

```
lr = d_model^(-0.5) × min(t^(-0.5), t × warmup_steps^(-1.5))

where d_model is model dimension
```

- Increases during warmup
- Then decays as 1/√t
- Less common now (cosine preferred)

---

## 5. Advanced Techniques

### 5.1 Layer-wise Learning Rate Decay

Different learning rates for different layers:

```python
# Higher LR for later layers
lr_mult = {}
for i, layer in enumerate(model.layers):
    lr_mult[layer] = base_lr * (decay_rate ** (num_layers - i - 1))

# Example: 12 layers, decay=0.9, base_lr=1e-4
# Layer 11 (output): 1e-4
# Layer 10: 1e-4 * 0.9 = 9e-5
# Layer 0 (input): 1e-4 * 0.9^11 = 3.1e-5
```

**Use cases**:
- Fine-tuning pretrained models
- Frozen embeddings with trainable heads

### 5.2 Learning Rate Finder

Automatic lr selection:

```
1. Start with very small lr (e.g., 1e-7)
2. Increase lr each batch (multiplicative)
3. Record loss at each lr
4. Plot loss vs lr
5. Choose lr where:
   - Loss is decreasing fastest
   - Before loss starts increasing
```

**Rule of thumb**: Pick lr one order of magnitude below the minimum loss point.

### 5.3 Linear Scaling Rule

For batch size changes:

```
lr_new = lr_base × (batch_new / batch_base)

Example:
  Base: batch=32, lr=0.001
  New: batch=256, lr=0.001 × (256/32) = 0.008
```

**Caveat**: Only works up to a point; very large batches need warmup.

### 5.4 Learning Rate with Gradient Accumulation

When using gradient accumulation:

```
effective_batch = micro_batch × accumulation_steps

# lr should be based on effective batch
lr = base_lr × (effective_batch / reference_batch)
```

---

## 6. Practical Guidelines

### 6.1 Choosing a Schedule

| Task | Recommended Schedule |
|------|---------------------|
| CNN Training | Step decay (0.1x at 30, 60, 90) |
| Transformer Training | Warmup + Cosine |
| LLM Pre-training | Warmup + Cosine |
| Fine-tuning | Constant or linear decay |
| Quick experiments | OneCycle |

### 6.2 Hyperparameter Selection

**Max learning rate**:
```
Adam/AdamW: 1e-4 to 5e-4
SGD: 0.01 to 0.1
Fine-tuning: 1e-5 to 5e-5
```

**Warmup steps**:
```
Pre-training: 1-2% of total steps
Fine-tuning: 5-10% of total steps
Absolute: 1000-5000 steps
```

**Final learning rate**:
```
lr_final = lr_max / 10  (common)
lr_final = 0  (aggressive decay)
```

### 6.3 Common Mistakes

1. **No warmup with Adam**:
   - Adam needs warmup for stable statistics
   - Especially important for transformers

2. **Too short warmup**:
   - Causes instability
   - Increase if seeing spikes early

3. **Wrong total steps**:
   - Cosine decay needs correct total
   - If training longer, lr may go negative!

4. **Batch size changes without lr adjustment**:
   - Larger batch → larger lr
   - Linear scaling rule

### 6.4 Debugging Learning Rate

**Loss spikes**:
- lr too high
- Reduce lr or add warmup

**Loss plateau**:
- lr too low
- Or scheduler decayed too fast

**Training instability**:
- Add or extend warmup
- Reduce max lr

---

## 7. Interview Questions

### Q1: Explain warmup and why it's needed for transformers.

**Answer**:

**What is warmup**:
Linear increase of learning rate from 0 to max over initial steps.

**Why it's needed**:

1. **Adam statistics initialization**:
   - Adam's m and v start at 0
   - Early estimates are unreliable
   - Warmup allows statistics to stabilize

2. **Gradient variance**:
   - Early gradients have high variance
   - Large lr + high variance = instability
   - Warmup reduces impact of early noise

3. **Parameter initialization**:
   - Initial parameters may be in sensitive region
   - Large steps can break careful initialization
   - Warmup allows gentle initial movement

**Typical duration**: 1-5% of total training steps

### Q2: Compare step decay vs cosine annealing.

**Answer**:

**Step Decay**:
```
lr = lr_0 × 0.1^(milestones_passed)
```
- Abrupt lr changes at milestones
- Loss often spikes then recovers
- Requires tuning milestone locations
- Standard for CNN training

**Cosine Annealing**:
```
lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
```
- Smooth, continuous decrease
- No sudden changes
- Only need total steps
- Standard for transformers

**When to use which**:
- CNNs: Step decay is well-studied
- Transformers/LLMs: Cosine is preferred
- Limited epochs: OneCycle can be faster

### Q3: How does the linear scaling rule work?

**Answer**:

```
lr_new = lr_base × (batch_new / batch_base)
```

**Rationale**:
- Larger batch → more accurate gradient estimate
- More accurate gradient → can take larger steps
- Keeps gradient variance per update similar

**Example**:
```
Baseline: batch=32, lr=0.001
Scaled: batch=256, lr=0.001 × 256/32 = 0.008
```

**Limitations**:
- Only works for moderate scaling (up to ~8x)
- Very large batches need longer warmup
- Eventually generalizes worse regardless of lr

### Q4: What learning rate schedule do modern LLMs use?

**Answer**:

**Standard LLM schedule**: Warmup + Cosine Decay

```
Phase 1 (Warmup):
  lr = lr_max × (t / warmup_steps)
  Duration: 2000 steps or 1-2% of training

Phase 2 (Decay):
  progress = (t - warmup) / (total - warmup)
  lr = lr_final + 0.5(lr_max - lr_final)(1 + cos(π × progress))
```

**Typical values**:
```
lr_max = 3e-4 to 6e-4
lr_final = lr_max / 10 to lr_max / 100
warmup = 2000 steps
```

**Why cosine**:
- Smooth optimization trajectory
- No hyperparameter for decay schedule shape
- Works well empirically at scale

---

## 8. Summary

### Quick Reference

| Schedule | Formula | Use Case |
|----------|---------|----------|
| Step | lr × γ^(epoch//step) | CNNs |
| Exponential | lr × γ^epoch | General |
| Cosine | lr_min + 0.5(lr_max-lr_min)(1+cos) | Transformers |
| Warmup | Linear increase | All Adam-based |
| OneCycle | Up then down | Fast training |

### Key Equations

**Cosine Annealing**:
```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
```

**Linear Warmup**:
```
lr = lr_max × (t / warmup_steps)
```

**Warmup + Cosine (LLM Standard)**:
```
if t < warmup:
    lr = lr_max × t / warmup
else:
    progress = (t - warmup) / (total - warmup)
    lr = lr_final + 0.5(lr_max - lr_final)(1 + cos(π × progress))
```

### LLM Training Recipe

```
Schedule: Warmup + Cosine
  warmup_steps = 2000
  lr_max = 3e-4
  lr_final = 3e-5
  total_steps = 100000

Optimizer: AdamW
  β₁ = 0.9
  β₂ = 0.95
  weight_decay = 0.1

Gradient Clipping: max_norm = 1.0
```

### Key Takeaways

1. **Always use warmup** with Adam/AdamW
2. **Cosine decay** is standard for transformers
3. **Linear scaling** for batch size changes
4. **Layer-wise decay** for fine-tuning
5. **Total steps** must be set correctly for cosine
