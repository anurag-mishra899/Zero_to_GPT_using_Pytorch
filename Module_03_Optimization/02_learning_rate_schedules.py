"""
Module 3.2: Learning Rate Schedules
Complete implementation of learning rate scheduling strategies.

This module covers:
1. Classic schedules (Step, Exponential, Polynomial)
2. Modern schedules (Cosine, OneCycle)
3. LLM training schedules (Warmup + Cosine)
4. Advanced techniques (Layer-wise decay, LR finder)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Optional, List, Callable
from abc import ABC, abstractmethod

print("=" * 70)
print("MODULE 3.2: LEARNING RATE SCHEDULES")
print("=" * 70)

# =============================================================================
# SECTION 1: BASE SCHEDULER CLASS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: SCHEDULER FUNDAMENTALS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Base Scheduler
# -----------------------------------------------------------------------------
print("\n1.1 Base Scheduler Implementation")
print("-" * 40)


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer, last_step: int = -1):
        self.optimizer = optimizer
        self.step_count = last_step
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    @abstractmethod
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each param group."""
        pass

    def step(self, step: Optional[int] = None):
        """Update learning rate."""
        if step is None:
            self.step_count += 1
        else:
            self.step_count = step

        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


# =============================================================================
# SECTION 2: CLASSIC SCHEDULES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: CLASSIC SCHEDULES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 Step Decay
# -----------------------------------------------------------------------------
print("\n2.1 Step Decay Scheduler")
print("-" * 40)


class StepLR(LRScheduler):
    """
    Step decay scheduler.

    lr = lr_0 × γ^(step // step_size)

    Standard for CNN training (ImageNet).
    """

    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        decay_factor = self.gamma ** (self.step_count // self.step_size)
        return [base_lr * decay_factor for base_lr in self.base_lrs]


# Demonstrate step decay
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

print("Step Decay (step_size=30, gamma=0.1):")
print(f"{'Epoch':>8} {'LR':>12}")
for epoch in range(100):
    if epoch % 30 == 0 or epoch == 99:
        print(f"{epoch:>8} {scheduler.get_last_lr()[0]:>12.6f}")
    scheduler.step()


# -----------------------------------------------------------------------------
# 2.2 Multi-Step Decay
# -----------------------------------------------------------------------------
print("\n\n2.2 Multi-Step Decay Scheduler")
print("-" * 40)


class MultiStepLR(LRScheduler):
    """
    Multi-step decay at specified milestones.

    More flexible than fixed step size.
    """

    def __init__(
        self,
        optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        # Count how many milestones we've passed
        n_passed = sum(1 for m in self.milestones if self.step_count >= m)
        decay_factor = self.gamma ** n_passed
        return [base_lr * decay_factor for base_lr in self.base_lrs]


# Demonstrate multi-step
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

print("Multi-Step Decay (milestones=[30, 60, 80]):")
print(f"{'Epoch':>8} {'LR':>12}")
for epoch in range(100):
    if epoch in [0, 29, 30, 59, 60, 79, 80, 99]:
        print(f"{epoch:>8} {scheduler.get_last_lr()[0]:>12.6f}")
    scheduler.step()


# -----------------------------------------------------------------------------
# 2.3 Exponential Decay
# -----------------------------------------------------------------------------
print("\n\n2.3 Exponential Decay Scheduler")
print("-" * 40)


class ExponentialLR(LRScheduler):
    """
    Exponential decay.

    lr = lr_0 × γ^step
    """

    def __init__(
        self,
        optimizer,
        gamma: float = 0.95,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        return [base_lr * (self.gamma ** self.step_count) for base_lr in self.base_lrs]


# Demonstrate exponential decay
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.95)

print("Exponential Decay (gamma=0.95):")
print(f"{'Epoch':>8} {'LR':>12}")
for epoch in range(50):
    if epoch % 10 == 0:
        print(f"{epoch:>8} {scheduler.get_last_lr()[0]:>12.6f}")
    scheduler.step()


# -----------------------------------------------------------------------------
# 2.4 Polynomial Decay
# -----------------------------------------------------------------------------
print("\n\n2.4 Polynomial Decay Scheduler")
print("-" * 40)


class PolynomialLR(LRScheduler):
    """
    Polynomial decay.

    lr = lr_0 × (1 - t/T)^power

    Linear decay when power=1.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        power: float = 1.0,
        lr_end: float = 0.0,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.total_steps = total_steps
        self.power = power
        self.lr_end = lr_end

    def get_lr(self) -> List[float]:
        if self.step_count >= self.total_steps:
            return [self.lr_end] * len(self.base_lrs)

        progress = self.step_count / self.total_steps
        factor = (1 - progress) ** self.power

        return [
            self.lr_end + (base_lr - self.lr_end) * factor
            for base_lr in self.base_lrs
        ]


# Demonstrate polynomial decay
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = PolynomialLR(optimizer, total_steps=100, power=1.0)

print("Linear Decay (power=1):")
print(f"{'Step':>8} {'LR':>12}")
for step in range(110):
    if step % 20 == 0:
        print(f"{step:>8} {scheduler.get_last_lr()[0]:>12.6f}")
    scheduler.step()


# =============================================================================
# SECTION 3: MODERN SCHEDULES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODERN SCHEDULES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Cosine Annealing
# -----------------------------------------------------------------------------
print("\n3.1 Cosine Annealing Scheduler")
print("-" * 40)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing scheduler.

    lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> List[float]:
        if self.step_count >= self.T_max:
            return [self.eta_min] * len(self.base_lrs)

        progress = self.step_count / self.T_max
        factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


# Demonstrate cosine annealing
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

print("Cosine Annealing (T_max=100, eta_min=0.001):")
print(f"{'Step':>8} {'LR':>12}")
lrs_cosine = []
for step in range(110):
    lr = scheduler.get_last_lr()[0]
    lrs_cosine.append(lr)
    if step % 20 == 0:
        print(f"{step:>8} {lr:>12.6f}")
    scheduler.step()


# -----------------------------------------------------------------------------
# 3.2 Cosine with Warm Restarts
# -----------------------------------------------------------------------------
print("\n\n3.2 Cosine Annealing with Warm Restarts")
print("-" * 40)


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).

    Periodically resets lr to initial value.
    T_mult controls how cycle length grows.
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0

    def get_lr(self) -> List[float]:
        progress = self.T_cur / self.T_i
        factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]

    def step(self, step: Optional[int] = None):
        if step is None:
            self.step_count += 1
            self.T_cur += 1

            # Check if we need to restart
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            self.step_count = step
            # Calculate which cycle we're in
            # (simplified - for exact behavior, track cycles)

        # Update lr
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


# Demonstrate warm restarts
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.001)

print("Cosine with Warm Restarts (T_0=20, T_mult=2):")
print(f"{'Step':>8} {'LR':>12}")
for step in range(100):
    lr = scheduler.get_last_lr()[0]
    if step % 10 == 0 or (step > 0 and abs(lr - 0.1) < 0.01):
        print(f"{step:>8} {lr:>12.6f}")
    scheduler.step()


# -----------------------------------------------------------------------------
# 3.3 OneCycle Learning Rate
# -----------------------------------------------------------------------------
print("\n\n3.3 OneCycle Learning Rate")
print("-" * 40)


class OneCycleLR(LRScheduler):
    """
    OneCycle learning rate policy.

    1. Warmup: lr increases linearly to max_lr
    2. Annealing: lr decreases following cosine
    3. Final: Optional continued decrease

    Also includes momentum cycling (not shown here).
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,  # Fraction for warmup
        div_factor: float = 25.0,  # initial_lr = max_lr / div_factor
        final_div_factor: float = 1e4,  # final_lr = initial_lr / final_div_factor
        last_step: int = -1
    ):
        # Set initial lr before calling parent
        initial_lr = max_lr / div_factor
        for group in optimizer.param_groups:
            group['lr'] = initial_lr

        super().__init__(optimizer, last_step)

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.warmup_steps = int(total_steps * pct_start)

    def get_lr(self) -> List[float]:
        if self.step_count < self.warmup_steps:
            # Warmup phase: linear increase
            progress = self.step_count / self.warmup_steps
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase: cosine decrease
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * progress))

        return [lr] * len(self.base_lrs)


# Demonstrate OneCycle
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=100, pct_start=0.3)

print("OneCycle (max_lr=0.1, pct_start=0.3):")
print(f"{'Step':>8} {'LR':>12} {'Phase':>12}")
for step in range(110):
    lr = scheduler.get_last_lr()[0]
    phase = "Warmup" if step < 30 else "Annealing"
    if step % 10 == 0 or step == 30:
        print(f"{step:>8} {lr:>12.6f} {phase:>12}")
    scheduler.step()


# =============================================================================
# SECTION 4: LLM TRAINING SCHEDULES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: LLM TRAINING SCHEDULES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Warmup + Cosine Decay (Standard LLM Schedule)
# -----------------------------------------------------------------------------
print("\n4.1 Warmup + Cosine Decay (LLM Standard)")
print("-" * 40)


class WarmupCosineScheduler(LRScheduler):
    """
    Warmup + Cosine Decay scheduler.

    Standard for LLM pre-training.

    Phase 1: Linear warmup from 0 to max_lr
    Phase 2: Cosine decay from max_lr to min_lr
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: Optional[float] = None,  # If None, uses optimizer's lr
        min_lr: Optional[float] = None,  # If None, uses max_lr / 10
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr if max_lr is not None else self.base_lrs[0]
        self.min_lr = min_lr if min_lr is not None else self.max_lr / 10

    def get_lr(self) -> List[float]:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.step_count - self.warmup_steps) / decay_steps
            progress = min(1.0, progress)  # Clamp to [0, 1]

            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        return [lr] * len(self.base_lrs)


# Demonstrate LLM schedule
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=2000,
    total_steps=100000,
    max_lr=3e-4,
    min_lr=3e-5
)

print("LLM Schedule (warmup=2000, total=100000):")
print(f"{'Step':>10} {'LR':>15}")
steps_to_show = [0, 500, 1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000]
for step in range(100001):
    if step in steps_to_show:
        print(f"{step:>10} {scheduler.get_last_lr()[0]:>15.8f}")
    scheduler.step()


# -----------------------------------------------------------------------------
# 4.2 Warmup + Linear Decay
# -----------------------------------------------------------------------------
print("\n\n4.2 Warmup + Linear Decay")
print("-" * 40)


class WarmupLinearScheduler(LRScheduler):
    """
    Warmup + Linear Decay scheduler.

    Simpler alternative to cosine.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: Optional[float] = None,
        min_lr: float = 0.0,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr if max_lr is not None else self.base_lrs[0]
        self.min_lr = min_lr

    def get_lr(self) -> List[float]:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.step_count - self.warmup_steps) / decay_steps
            progress = min(1.0, progress)

            lr = self.max_lr - (self.max_lr - self.min_lr) * progress

        return [lr] * len(self.base_lrs)


# Demonstrate
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = WarmupLinearScheduler(optimizer, warmup_steps=1000, total_steps=10000)

print("Warmup + Linear (warmup=1000, total=10000):")
print(f"{'Step':>8} {'LR':>15}")
for step in range(0, 10001, 1000):
    scheduler.step_count = step
    print(f"{step:>8} {scheduler.get_lr()[0]:>15.8f}")


# -----------------------------------------------------------------------------
# 4.3 WSD (Warmup-Stable-Decay)
# -----------------------------------------------------------------------------
print("\n\n4.3 WSD (Warmup-Stable-Decay)")
print("-" * 40)


class WSDScheduler(LRScheduler):
    """
    Warmup-Stable-Decay scheduler.

    Three phases:
    1. Warmup: Linear increase to max_lr
    2. Stable: Keep at max_lr
    3. Decay: Cosine decay to min_lr

    Allows more training at peak learning rate.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        max_lr: Optional[float] = None,
        min_lr: Optional[float] = None,
        last_step: int = -1
    ):
        super().__init__(optimizer, last_step)
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.max_lr = max_lr if max_lr is not None else self.base_lrs[0]
        self.min_lr = min_lr if min_lr is not None else self.max_lr / 10

    def get_lr(self) -> List[float]:
        step = self.step_count

        if step < self.warmup_steps:
            # Phase 1: Warmup
            lr = self.max_lr * (step / self.warmup_steps)
        elif step < self.warmup_steps + self.stable_steps:
            # Phase 2: Stable
            lr = self.max_lr
        else:
            # Phase 3: Decay
            decay_step = step - self.warmup_steps - self.stable_steps
            progress = decay_step / self.decay_steps
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        return [lr] * len(self.base_lrs)


# Demonstrate WSD
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = WSDScheduler(
    optimizer,
    warmup_steps=1000,
    stable_steps=3000,
    decay_steps=6000,
    max_lr=3e-4,
    min_lr=3e-5
)

print("WSD Schedule (warmup=1000, stable=3000, decay=6000):")
print(f"{'Step':>8} {'LR':>15} {'Phase':>10}")
for step in range(0, 10001, 1000):
    scheduler.step_count = step
    lr = scheduler.get_lr()[0]
    if step < 1000:
        phase = "Warmup"
    elif step < 4000:
        phase = "Stable"
    else:
        phase = "Decay"
    print(f"{step:>8} {lr:>15.8f} {phase:>10}")


# =============================================================================
# SECTION 5: ADVANCED TECHNIQUES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: ADVANCED TECHNIQUES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Layer-wise Learning Rate Decay
# -----------------------------------------------------------------------------
print("\n5.1 Layer-wise Learning Rate Decay")
print("-" * 40)


def get_layerwise_lr_decay_params(
    model: nn.Module,
    base_lr: float,
    lr_decay: float = 0.9,
    weight_decay: float = 0.01,
    no_decay_names: List[str] = ['bias', 'LayerNorm', 'layer_norm']
) -> List[dict]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Later layers get higher LR, earlier layers get lower LR.
    Used for fine-tuning pretrained models.
    """
    # Get all named parameters
    named_params = list(model.named_parameters())

    # Simple approach: assign LR based on parameter name depth
    # More sophisticated: actually parse layer indices

    param_groups = []
    no_decay_params = []
    decay_params = []

    for name, param in named_params:
        # Check if this param should have weight decay
        has_decay = not any(nd in name for nd in no_decay_names)

        # Simple depth calculation (count dots in name)
        depth = name.count('.')

        # Calculate layer-specific LR
        layer_lr = base_lr * (lr_decay ** depth)

        if has_decay:
            param_groups.append({
                'params': [param],
                'lr': layer_lr,
                'weight_decay': weight_decay,
                'name': name
            })
        else:
            param_groups.append({
                'params': [param],
                'lr': layer_lr,
                'weight_decay': 0.0,
                'name': name
            })

    return param_groups


# Demonstrate layer-wise decay
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=64, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, 1000)


model = SimpleTransformer()
param_groups = get_layerwise_lr_decay_params(model, base_lr=1e-4, lr_decay=0.9)

print("Layer-wise Learning Rate Decay:")
print(f"{'Layer':>40} {'LR':>15} {'WD':>8}")
print("-" * 65)
for group in param_groups[:10]:  # Show first 10
    print(f"{group['name'][:40]:>40} {group['lr']:>15.8f} {group['weight_decay']:>8.4f}")
print("...")


# -----------------------------------------------------------------------------
# 5.2 Learning Rate Finder
# -----------------------------------------------------------------------------
print("\n\n5.2 Learning Rate Finder")
print("-" * 40)


def lr_finder(
    model: nn.Module,
    train_loader,
    criterion,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_steps: int = 100,
    smooth_factor: float = 0.05
):
    """
    Find optimal learning rate by training with exponentially increasing LR.

    Returns (lrs, losses) for plotting.
    """
    # Save initial state
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = optim.SGD(model.parameters(), lr=start_lr)

    # Calculate multiplicative factor for LR increase
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)

    lrs = []
    losses = []
    smoothed_loss = 0

    model.train()
    data_iter = iter(train_loader)

    for step in range(num_steps):
        # Get batch (cycle if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y = batch

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        # Exponential smoothing
        if step == 0:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = smooth_factor * loss.item() + (1 - smooth_factor) * smoothed_loss

        losses.append(smoothed_loss)

        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        # Stop if loss explodes
        if smoothed_loss > 4 * losses[0]:
            break

    # Restore initial state
    model.load_state_dict(initial_state)

    return lrs, losses


# Simple demonstration (without actual training data)
print("LR Finder Algorithm:")
print("1. Start with very small LR (1e-7)")
print("2. Train one batch, increase LR exponentially")
print("3. Record loss at each LR")
print("4. Plot loss vs LR")
print("5. Choose LR where loss decreases fastest")
print("\nSuggested LR: 1 order of magnitude below minimum loss point")


# =============================================================================
# SECTION 6: PUTTING IT ALL TOGETHER
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: COMPLETE TRAINING EXAMPLE")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Complete LLM Training Setup
# -----------------------------------------------------------------------------
print("\n6.1 Complete LLM Training Configuration")
print("-" * 40)


def create_llm_optimizer_and_scheduler(
    model: nn.Module,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    no_decay_keywords: List[str] = ['bias', 'LayerNorm', 'layer_norm', 'layernorm']
):
    """
    Create optimizer and scheduler for LLM training.

    Standard configuration:
    - AdamW optimizer
    - Warmup + Cosine decay
    - Proper weight decay handling
    """
    # Separate parameters by weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # Create optimizer
    optimizer = optim.AdamW(
        param_groups,
        lr=max_lr,
        betas=betas,
        eps=1e-8
    )

    # Create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr
    )

    return optimizer, scheduler


# Demonstrate
model = SimpleTransformer()
optimizer, scheduler = create_llm_optimizer_and_scheduler(
    model,
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=2000,
    total_steps=100000
)

print("LLM Training Configuration:")
print(f"  Optimizer: AdamW")
print(f"  Max LR: 3e-4")
print(f"  Min LR: 3e-5")
print(f"  Warmup: 2000 steps")
print(f"  Total: 100000 steps")
print(f"  Weight Decay: 0.1 (excluded for biases and LayerNorm)")
print(f"\n  Parameter groups:")
for i, group in enumerate(optimizer.param_groups):
    n_params = sum(p.numel() for p in group['params'])
    print(f"    Group {i}: {n_params:,} params, wd={group['weight_decay']}")


# =============================================================================
# SECTION 7: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. Why warmup?
   - Adam statistics need time to stabilize
   - Early gradients have high variance
   - Prevents instability at training start
   - Typically 1-5% of total steps

2. Cosine vs Step Decay:
   - Cosine: Smooth, no hyperparameter for schedule shape
   - Step: Abrupt changes, needs milestone tuning
   - Cosine preferred for transformers

3. Linear Scaling Rule:
   - lr_new = lr_base × (batch_new / batch_base)
   - Keeps gradient variance similar
   - Only works for moderate scaling

4. Layer-wise Decay:
   - Lower LR for earlier layers
   - Higher LR for later layers
   - Used in fine-tuning pretrained models

5. OneCycle:
   - Warmup then decay
   - Also cycles momentum
   - Can use larger max_lr
   - Good for fast training
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 3.2 SUMMARY: LEARNING RATE SCHEDULES")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────────────┬─────────────────────────┐
│ Schedule        │ Formula                        │ Use Case                │
├─────────────────┼────────────────────────────────┼─────────────────────────┤
│ Step Decay      │ lr × γ^(step//size)            │ CNNs                    │
│ Cosine          │ 0.5(1+cos(πt/T))              │ Transformers            │
│ Warmup+Cosine   │ Linear then cosine             │ LLM Pre-training        │
│ OneCycle        │ Up then down                   │ Fast training           │
│ WSD             │ Warmup→Stable→Decay            │ Extended stable phase   │
└─────────────────┴────────────────────────────────┴─────────────────────────┘

LLM TRAINING RECIPE:

  Schedule: Warmup + Cosine
    warmup_steps = 2000 (or 1-2% of total)
    max_lr = 3e-4
    min_lr = max_lr / 10

  Optimizer: AdamW
    betas = (0.9, 0.95)
    weight_decay = 0.1
    eps = 1e-8

  Gradient Clipping: max_norm = 1.0

KEY TAKEAWAYS:
1. Always use warmup with Adam
2. Cosine decay is standard for transformers
3. Total steps must be set correctly
4. Layer-wise decay for fine-tuning
5. Linear scaling for batch size changes
""")

print("\nModule 3.2 Complete!")
