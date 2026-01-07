# Training Dynamics & Debugging LLMs

## Overview

Training large language models is notoriously challenging. This module covers the practical aspects of monitoring training, diagnosing issues, and debugging common problems that arise during LLM training.

## Table of Contents
1. [The Training Journey](#the-training-journey)
2. [Loss Curves & What They Tell You](#loss-curves--what-they-tell-you)
3. [Gradient Analysis](#gradient-analysis)
4. [Learning Rate Dynamics](#learning-rate-dynamics)
5. [Common Training Problems](#common-training-problems)
6. [Debugging Toolkit](#debugging-toolkit)
7. [Monitoring & Logging](#monitoring--logging)
8. [Distributed Training Issues](#distributed-training-issues)
9. [Interview Questions](#interview-questions)

---

## The Training Journey

### Typical LLM Training Phases

```
Loss
│
│ ████                 Phase 1: Initial rapid descent
│     ████
│         ████         Phase 2: Steady improvement
│             ████
│                 ████ Phase 3: Slow refinement
│                     ████ ████ ████
└──────────────────────────────────────→ Steps
  0    10K   50K   100K  200K  500K
```

### Phase Breakdown

**Phase 1: Memorization (0-10% of training)**
- Loss drops rapidly
- Model learns basic patterns (common words, punctuation)
- Gradient norms are large
- Most "obvious" learning happens here

**Phase 2: Generalization (10-60% of training)**
- Steady loss decrease
- Model learns grammar, semantics
- Validation loss tracks training loss
- Most compute spent here

**Phase 3: Refinement (60-100% of training)**
- Diminishing returns
- Model learns long-range dependencies
- Risk of overfitting increases
- May see validation loss plateau or increase

### Key Metrics to Track

| Metric | What It Shows | Warning Signs |
|--------|--------------|---------------|
| Training Loss | Learning progress | Spikes, plateaus |
| Validation Loss | Generalization | Diverges from train |
| Gradient Norm | Optimization health | Very large/small |
| Learning Rate | Schedule execution | Unexpected values |
| Throughput | Training efficiency | Sudden drops |
| Memory | Resource usage | OOM risks |

---

## Loss Curves & What They Tell You

### Healthy Training Loss

```
        Good Training Curve
Loss
│
│████
│    ████
│        ████
│            ████████
│                    ████████████
└──────────────────────────────→ Steps
```

**Characteristics**:
- Smooth, monotonic decrease
- Occasional small noise (mini-batch variance)
- Gradual flattening toward end

### Problematic Patterns

**1. Loss Spikes**
```
Loss
│
│    █
│   ███
│  █████
│ ███████████████████
└──────────────────→ Steps
       ↑
   Spike here
```

**Causes**:
- Bad data batch (corrupted, very long)
- Learning rate too high
- Numerical instability
- Gradient explosion

**Solutions**:
```python
# Gradient clipping (essential!)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Skip bad batches
if loss.item() > 10 * running_avg_loss:
    print(f"Skipping batch with loss {loss.item()}")
    continue

# Check for NaN/Inf
if torch.isnan(loss) or torch.isinf(loss):
    print("NaN/Inf detected!")
    # Load from checkpoint, reduce LR
```

**2. Loss Plateau**
```
Loss
│████
│    ████████████████████████
│
│
└──────────────────────────→ Steps
```

**Causes**:
- Learning rate too low
- Model capacity reached
- Stuck in local minimum
- Data exhausted (need more diversity)

**Solutions**:
```python
# Try learning rate warmup restart
# Increase model capacity
# Add more/different data
# Reduce regularization temporarily
```

**3. Oscillating Loss**
```
Loss
│  ██  ██  ██  ██  ██
│ ████████████████████
│██  ██  ██  ██  ██
└──────────────────────→ Steps
```

**Causes**:
- Learning rate too high
- Batch size too small
- Conflicting gradients in data

**Solutions**:
```python
# Reduce learning rate
# Increase batch size
# Use gradient accumulation
# Add momentum/reduce beta2 in Adam
```

**4. Train-Val Divergence (Overfitting)**
```
Loss
│████                  ← Train
│    ████
│        ████████████
│
│████████████████████  ← Val (stops improving)
│                ████████ (or increases)
└──────────────────────────→ Steps
```

**Solutions**:
```python
# Early stopping
# Increase dropout
# Add weight decay
# Use data augmentation
# Reduce model size
```

---

## Gradient Analysis

### Why Gradients Matter

Gradients are the "steering wheel" of training. Unhealthy gradients → unhealthy training.

### Gradient Norm Monitoring

```python
def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
```

### Healthy Gradient Norms

| Phase | Expected Norm | Notes |
|-------|--------------|-------|
| Early training | 1-10 | Larger gradients ok |
| Mid training | 0.1-1 | Should stabilize |
| Late training | 0.01-0.1 | Decreasing |

### Gradient Pathologies

**1. Vanishing Gradients**
```
Gradient norm → 0 as training progresses
Early layers receive almost no signal
```

**Diagnosis**:
```python
def check_layer_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: {grad_norm:.6f}")
            if grad_norm < 1e-7:
                print(f"  ⚠️ Vanishing gradient!")
```

**Solutions**:
- Use residual connections
- Layer normalization
- Better initialization
- Gradient checkpointing doesn't help (forward problem)

**2. Exploding Gradients**
```
Gradient norm → ∞
Loss spikes, NaN values appear
```

**Diagnosis**:
```python
def check_for_explosion(model, threshold=100):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.norm() > threshold:
                print(f"⚠️ {name}: grad norm = {param.grad.norm():.2f}")
            if torch.isnan(param.grad).any():
                print(f"❌ {name}: contains NaN!")
```

**Solutions**:
```python
# Gradient clipping (the standard fix)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Or gradient value clipping
torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
```

**3. Dead Neurons**
```
Some neurons always output 0 (ReLU death)
Gradients for those weights are always 0
```

**Diagnosis**:
```python
def check_dead_neurons(activations, threshold=0.01):
    """Check if neurons are consistently zero."""
    dead_ratio = (activations.abs() < threshold).float().mean(dim=0)
    dead_neurons = (dead_ratio > 0.99).sum().item()
    print(f"Dead neurons: {dead_neurons} / {activations.shape[1]}")
```

**Solutions**:
- Use GELU instead of ReLU
- Leaky ReLU
- Proper initialization

---

## Learning Rate Dynamics

### The Learning Rate is Critical

```
             Learning Rate Sweet Spot

Convergence  │           ████
Speed        │         ██████
             │       ████████
             │     ██████████
             │   ████████████████
             │ ██████████████████████
             │██████████████████████████
             └────────────────────────────→ LR
              1e-6    1e-4    1e-2    1

                        ↑
                    Sweet spot
              (task and model dependent)
```

### Learning Rate Schedules

**1. Warmup + Cosine Decay (Most Common)**
```python
def cosine_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr=0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**2. Linear Warmup + Linear Decay**
```python
def linear_with_warmup(step, warmup_steps, total_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return max_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
```

**3. Constant with Warmup**
```python
def constant_with_warmup(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return max_lr
```

### Learning Rate Finder

```python
def lr_finder(model, dataloader, min_lr=1e-7, max_lr=10, num_steps=100):
    """
    Find optimal learning rate by training with exponentially increasing LR.
    Plot loss vs LR, look for steepest descent.
    """
    model_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)

    lr_mult = (max_lr / min_lr) ** (1 / num_steps)
    lrs, losses = [], []

    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break

        loss = train_step(model, batch, optimizer)

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss)

        # Increase LR
        for g in optimizer.param_groups:
            g['lr'] *= lr_mult

        if loss > 4 * losses[0]:  # Diverging
            break

    model.load_state_dict(model_state)
    return lrs, losses
```

---

## Common Training Problems

### 1. Out of Memory (OOM)

**Symptoms**:
- CUDA out of memory error
- Process killed by OS

**Solutions**:
```python
# Reduce batch size
batch_size = 8  # Try smaller

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch).loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
x = checkpoint(self.layer1, x)
```

### 2. Training Instability

**Symptoms**:
- Loss spikes
- NaN values
- Inconsistent results

**Diagnostic Checklist**:
```python
def diagnose_instability(model, batch, loss):
    issues = []

    # Check loss
    if torch.isnan(loss):
        issues.append("NaN loss")
    if loss.item() > 100:
        issues.append(f"Very high loss: {loss.item()}")

    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN gradient in {name}")
            if param.grad.abs().max() > 1000:
                issues.append(f"Large gradient in {name}: {param.grad.abs().max()}")

    # Check weights
    for name, param in model.parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN weights in {name}")

    return issues
```

**Solutions**:
```python
# Lower learning rate
lr = lr * 0.5

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

# Use BF16 instead of FP16
# BF16 has larger dynamic range

# Check data pipeline
# Ensure no NaN/Inf in inputs
assert not torch.isnan(batch['input_ids'].float()).any()
```

### 3. Slow Convergence

**Symptoms**:
- Loss decreases very slowly
- Training takes too long

**Checklist**:
```python
# 1. Learning rate too low?
# Try 10x increase

# 2. Batch size too small?
# Increase if memory allows

# 3. Model initialization bad?
# Use proper init schemes

# 4. Data quality issues?
# Check for duplicates, noise

# 5. Wrong optimizer settings?
# Adam betas, weight decay
```

### 4. Overfitting

**Symptoms**:
- Train loss decreases, val loss increases
- Model memorizes training data

**Solutions**:
```python
# Dropout
dropout = 0.1  # or higher

# Weight decay
optimizer = AdamW(params, weight_decay=0.01)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# Data augmentation
# Add noise, paraphrasing, back-translation
```

---

## Debugging Toolkit

### Essential Debugging Techniques

**1. Overfit on Small Batch**
```python
def overfit_single_batch(model, batch, num_steps=100):
    """
    If model can't overfit one batch, something is broken.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    if loss.item() > 0.1:
        print("⚠️ Warning: Could not overfit single batch!")
    else:
        print("✓ Model can overfit single batch")
```

**2. Gradient Flow Check**
```python
def check_gradient_flow(model):
    """
    Visualize gradient magnitudes across layers.
    """
    layers = []
    grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layers.append(name)
            grads.append(param.grad.abs().mean().item())

    plt.figure(figsize=(15, 5))
    plt.bar(range(len(grads)), grads)
    plt.xticks(range(len(grads)), layers, rotation=45, ha='right')
    plt.ylabel('Mean Gradient Magnitude')
    plt.title('Gradient Flow')
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
```

**3. Activation Statistics**
```python
class ActivationStats:
    """Hook to collect activation statistics."""

    def __init__(self):
        self.stats = {}

    def hook(self, name):
        def fn(module, input, output):
            self.stats[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'dead_frac': (output.abs() < 1e-6).float().mean().item()
            }
        return fn

# Usage
stats = ActivationStats()
for name, module in model.named_modules():
    module.register_forward_hook(stats.hook(name))
```

**4. Weight Distribution**
```python
def plot_weight_distributions(model, save_path='weights.png'):
    """Plot histograms of weight values per layer."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx >= len(axes):
            break
        weights = param.data.cpu().numpy().flatten()
        axes[idx].hist(weights, bins=50, alpha=0.7)
        axes[idx].set_title(name[:20], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
```

---

## Monitoring & Logging

### What to Log

**Every Step**:
- Loss (train)
- Learning rate
- Gradient norm (clipped and unclipped)
- Throughput (samples/sec, tokens/sec)

**Every N Steps**:
- Validation loss
- Generated samples
- Memory usage
- Weight/gradient histograms

**At Checkpoints**:
- Full model state
- Optimizer state
- Training config
- Metrics history

### Logging Best Practices

```python
import wandb
# or tensorboard, mlflow, etc.

wandb.init(project="llm-training", config={
    "model": "gpt-small",
    "lr": 1e-4,
    "batch_size": 32,
    ...
})

for step, batch in enumerate(dataloader):
    loss = train_step(...)

    # Log every step
    wandb.log({
        "train/loss": loss,
        "train/lr": scheduler.get_last_lr()[0],
        "train/grad_norm": grad_norm,
    }, step=step)

    # Log validation periodically
    if step % eval_steps == 0:
        val_loss = evaluate(...)
        wandb.log({"val/loss": val_loss}, step=step)

        # Log generated samples
        samples = generate(...)
        wandb.log({"samples": wandb.Table(...)}, step=step)
```

### Alerting

```python
def check_and_alert(metrics, step):
    """Alert on anomalies."""
    alerts = []

    if metrics['loss'] > 10:
        alerts.append(f"High loss: {metrics['loss']}")

    if metrics['grad_norm'] > 100:
        alerts.append(f"Gradient explosion: {metrics['grad_norm']}")

    if metrics['grad_norm'] < 1e-7:
        alerts.append(f"Vanishing gradients: {metrics['grad_norm']}")

    if torch.isnan(torch.tensor(metrics['loss'])):
        alerts.append("NaN loss detected!")

    if alerts:
        send_alert(f"Training alert at step {step}: {alerts}")
```

---

## Distributed Training Issues

### Common Distributed Problems

**1. Gradient Synchronization Issues**
```python
# Check if all ranks have same gradients
def check_gradient_sync(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            local_grad = param.grad.clone()
            torch.distributed.all_reduce(local_grad)
            local_grad /= torch.distributed.get_world_size()

            diff = (param.grad - local_grad).abs().max()
            if diff > 1e-5:
                print(f"Gradient mismatch in {name}: {diff}")
```

**2. Data Loading Desync**
```python
# Ensure each rank sees different data
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

# Reset sampler each epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Critical!
```

**3. Deadlocks**
```python
# Always ensure all ranks execute same collective ops
if rank == 0:
    data = load_data()  # Only rank 0 loads
    # ❌ This causes deadlock if other ranks wait at all_reduce

# Correct:
if rank == 0:
    data = load_data()
data = broadcast(data, src=0)  # All ranks participate
```

**4. Memory Imbalance**
```python
# Rank 0 often has more memory pressure (logging, etc.)
# Solution: Reduce batch size on rank 0 or offload logging
```

---

## Interview Questions

### Debugging

**Q1: Your training loss spikes every 1000 steps. How do you debug?**

1. **Check data**: Is there a bad batch at that interval? Log batch statistics.
2. **Check LR schedule**: Is there a scheduler step there?
3. **Check gradient norms**: Log pre-clip gradient norms.
4. **Check memory**: Is there a memory spike causing precision issues?
5. **Solution**: Add `torch.autograd.detect_anomaly()` temporarily to pinpoint.

**Q2: Training loss decreases but validation loss increases after epoch 2.**

This is overfitting:
1. Add dropout (start with 0.1)
2. Increase weight decay
3. Use early stopping based on validation loss
4. Reduce model capacity or get more data
5. Add data augmentation

**Q3: Gradients are all zero after the first layer. Diagnosis?**

Vanishing gradients:
1. Check activation functions (avoid sigmoid in deep networks)
2. Check initialization (Xavier/He init)
3. Add residual connections
4. Add layer normalization
5. Check if there's a bug in loss computation

### System Design

**Q4: Design a training monitoring system for a 100B parameter model.**

```
Architecture:
┌─────────────────────────────────────────────────────┐
│                   Training Nodes                     │
│  (Each logs to local buffer, async flush)           │
└───────────────────────┬─────────────────────────────┘
                        │ Async collection
                        ▼
┌─────────────────────────────────────────────────────┐
│               Metrics Aggregator                     │
│  - Combine per-rank metrics                         │
│  - Compute global statistics                        │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   Time Series   │         │   Alert System  │
│    Database     │         │  (PagerDuty)    │
│  (InfluxDB)     │         │                 │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│   Dashboards    │         │   On-call Eng   │
│   (Grafana)     │         │                 │
└─────────────────┘         └─────────────────┘

Key Metrics:
- Per-GPU loss, throughput, memory
- Global loss, gradient norm
- Hardware metrics (GPU temp, util)
- Checkpoint success/failure
```

---

## Summary

| Problem | Symptom | Solution |
|---------|---------|----------|
| Loss spike | Sudden increase | Gradient clip, skip batch |
| Loss plateau | No progress | Increase LR, more data |
| Overfitting | Val loss rises | Dropout, early stopping |
| Vanishing grad | Early layers 0 grad | Residual, LayerNorm |
| Exploding grad | NaN, huge norms | Gradient clipping |
| OOM | Memory error | Smaller batch, mixed precision |

### Key Takeaways

1. **Monitor everything**: Loss, gradients, activations, memory
2. **Start simple**: Overfit one batch first
3. **Gradient clipping is essential** for transformers
4. **Warmup prevents early instability**
5. **Save checkpoints frequently** - training can fail anytime
6. **Log for debugging**: You can't fix what you can't see

---

## References

1. [On the Difficulty of Training RNNs](https://arxiv.org/abs/1211.5063) - Pascanu et al., 2013
2. [An Empirical Study of Training Self-Attention](https://arxiv.org/abs/2202.06897)
3. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
4. [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
5. [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - Micikevicius et al., 2017
