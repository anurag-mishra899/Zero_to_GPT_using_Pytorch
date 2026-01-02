"""
Module 12.1: Efficient Training Techniques
Implementation of training optimization methods

Covers:
- Mixed precision training (AMP)
- Gradient checkpointing
- Gradient accumulation
- Memory optimization
- Training utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # Device-agnostic AMP (PyTorch 2.0+)
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Callable
import math
import time
from dataclasses import dataclass


# Device detection for Apple Silicon (MPS) / CUDA / CPU
def get_device():
    """Get best available device: MPS > CUDA > CPU"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = get_device()
print(f"Using device: {DEVICE}")

print("=" * 70)
print("Module 12.1: Efficient Training Techniques")
print("=" * 70)


# ===========================================================================
# Section 1: Mixed Precision Training
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: Mixed Precision Training")
print("=" * 70)


class MixedPrecisionTrainer:
    """
    Trainer with automatic mixed precision (AMP).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = True,
        dtype: torch.dtype = torch.float16,
        initial_scale: float = 2**16,
        growth_interval: int = 2000
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.dtype = dtype

        # Gradient scaler for FP16
        if use_amp and dtype == torch.float16:
            self.scaler = GradScaler(
                init_scale=initial_scale,
                growth_interval=growth_interval
            )
        else:
            self.scaler = None

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module
    ) -> dict:
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()

        # Determine device type for autocast (MPS, CUDA, or CPU)
        device_type = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

        if self.use_amp:
            # Forward pass with autocast
            with autocast(device_type=device_type, dtype=self.dtype):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass
            if self.scaler is not None:
                # FP16: scale loss and unscale gradients
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                # Update with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # BF16: no scaling needed
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()
        else:
            # FP32 training
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'scale': self.scaler.get_scale() if self.scaler else 1.0
        }


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


# Test mixed precision training
print("\n--- Testing Mixed Precision Training ---")

model = SimpleModel(256, 512, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Compare FP32 vs FP16 vs BF16
for precision, dtype in [('FP32', None), ('FP16', torch.float16), ('BF16', torch.bfloat16)]:
    model_copy = SimpleModel(256, 512, 10)
    opt_copy = torch.optim.Adam(model_copy.parameters(), lr=1e-3)

    use_amp = dtype is not None
    trainer = MixedPrecisionTrainer(model_copy, opt_copy, use_amp=use_amp, dtype=dtype or torch.float32)

    # Training step
    x = torch.randn(32, 256)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for _ in range(10):
        metrics = trainer.train_step(x, y, criterion)
    elapsed = time.time() - start

    print(f"{precision}: loss={metrics['loss']:.4f}, time={elapsed*100:.1f}ms, scale={metrics['scale']}")


# ===========================================================================
# Section 2: Gradient Checkpointing
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: Gradient Checkpointing")
print("=" * 70)


class CheckpointedTransformerBlock(nn.Module):
    """
    Transformer block with optional gradient checkpointing.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actual forward computation."""
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class CheckpointedModel(nn.Module):
    """Model with configurable checkpointing."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        checkpoint_every: int = 1  # Checkpoint every N layers
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            CheckpointedTransformerBlock(
                d_model, num_heads, d_ff,
                use_checkpoint=(i % checkpoint_every == 0)
            )
            for i in range(num_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def measure_memory_and_time(
    model: nn.Module,
    input_ids: torch.Tensor,
    criterion: nn.Module,
    num_runs: int = 5
) -> dict:
    """Measure memory usage and time for forward + backward."""
    model.train()

    # Warmup
    outputs = model(input_ids)
    loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
    loss.backward()
    model.zero_grad()

    # Reset memory stats (device-agnostic)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Note: MPS doesn't have memory tracking APIs yet

    start = time.time()
    for _ in range(num_runs):
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
        loss.backward()
        model.zero_grad()
    elapsed = (time.time() - start) / num_runs

    # Get peak memory (CUDA only, MPS doesn't support this yet)
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    return {'time_ms': elapsed * 1000, 'memory_mb': peak_memory}


# Compare with and without checkpointing
print("\n--- Comparing With and Without Checkpointing ---")

vocab_size, d_model, num_layers, num_heads, d_ff = 1000, 256, 8, 4, 1024
batch_size, seq_len = 4, 128

# Without checkpointing
model_no_ckpt = CheckpointedModel(vocab_size, d_model, num_layers, num_heads, d_ff, checkpoint_every=999)
# With checkpointing every layer
model_with_ckpt = CheckpointedModel(vocab_size, d_model, num_layers, num_heads, d_ff, checkpoint_every=1)

input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
criterion = nn.CrossEntropyLoss()

# Count parameters
num_params = sum(p.numel() for p in model_no_ckpt.parameters())
print(f"Model parameters: {num_params:,}")

# Measure (CPU only - no GPU memory tracking)
metrics_no_ckpt = measure_memory_and_time(model_no_ckpt, input_ids, criterion)
metrics_with_ckpt = measure_memory_and_time(model_with_ckpt, input_ids, criterion)

print(f"\nWithout checkpointing: time={metrics_no_ckpt['time_ms']:.1f}ms")
print(f"With checkpointing: time={metrics_with_ckpt['time_ms']:.1f}ms")
print(f"Overhead: {(metrics_with_ckpt['time_ms'] / metrics_no_ckpt['time_ms'] - 1) * 100:.1f}%")


# ===========================================================================
# Section 3: Gradient Accumulation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: Gradient Accumulation")
print("=" * 70)


class GradientAccumulationTrainer:
    """
    Trainer with gradient accumulation for simulating larger batch sizes.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        self.accumulated_loss = 0.0

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module
    ) -> Optional[dict]:
        """
        Single micro-batch training step.
        Returns metrics only when gradients are applied.
        """
        # Forward pass
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        # Normalize loss by accumulation steps
        normalized_loss = loss / self.accumulation_steps
        normalized_loss.backward()

        self.accumulated_loss += loss.item()
        self.step_count += 1

        # Update only every accumulation_steps
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # Update
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Return metrics
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0

            return {
                'loss': avg_loss,
                'grad_norm': grad_norm.item(),
                'effective_batch': self.accumulation_steps
            }

        return None  # No update this step


# Test gradient accumulation
print("\n--- Testing Gradient Accumulation ---")

model = SimpleModel(256, 512, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

accumulation_steps = 4
trainer = GradientAccumulationTrainer(model, optimizer, accumulation_steps)

print(f"Accumulation steps: {accumulation_steps}")
print(f"Micro batch: 8, Effective batch: {8 * accumulation_steps}")

# Simulate training
for i in range(12):
    x = torch.randn(8, 256)  # Micro batch
    y = torch.randint(0, 10, (8,))

    metrics = trainer.train_step(x, y, criterion)

    if metrics is not None:
        print(f"Update {i+1}: loss={metrics['loss']:.4f}, grad_norm={metrics['grad_norm']:.4f}")


# ===========================================================================
# Section 4: Learning Rate Scheduling
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: Learning Rate Scheduling")
print("=" * 70)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.

    Used in most LLM training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)

    def step(self):
        """Update learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) scheduler.

    Warmup → Stable (constant LR) → Decay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Warmup
            return self.base_lr * self.current_step / self.warmup_steps
        elif self.current_step < self.warmup_steps + self.stable_steps:
            # Stable
            return self.base_lr
        else:
            # Decay
            decay_progress = (self.current_step - self.warmup_steps - self.stable_steps) / self.decay_steps
            decay_progress = min(decay_progress, 1.0)
            return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - decay_progress))

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


# Visualize schedulers
def visualize_schedulers():
    """Visualize different learning rate schedules."""
    total_steps = 10000
    warmup_steps = 1000

    model = nn.Linear(10, 10)

    # Create schedulers
    opt1 = torch.optim.Adam(model.parameters(), lr=1e-3)
    cosine = WarmupCosineScheduler(opt1, warmup_steps, total_steps)

    opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    wsd = WSDScheduler(opt2, warmup_steps=1000, stable_steps=6000, decay_steps=3000)

    # Collect LR values
    steps = list(range(total_steps))
    cosine_lrs = []
    wsd_lrs = []

    for _ in steps:
        cosine_lrs.append(cosine.step())
        wsd_lrs.append(wsd.step())

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps, cosine_lrs)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='Warmup end')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup + Cosine Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(steps, wsd_lrs)
    plt.axvline(x=1000, color='r', linestyle='--', label='Warmup end')
    plt.axvline(x=7000, color='g', linestyle='--', label='Decay start')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup-Stable-Decay (WSD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/anuragmishra/Documents/Zero_to_GPT/Module_12_Efficient_Training/lr_schedules.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved learning rate schedule visualization")

visualize_schedulers()


# ===========================================================================
# Section 5: Complete Training Loop
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Complete Training Loop")
print("=" * 70)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    vocab_size: int = 1000
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 1024

    # Training
    batch_size: int = 32
    seq_len: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 100
    total_steps: int = 1000

    # Efficiency
    use_amp: bool = True
    accumulation_steps: int = 1
    use_checkpoint: bool = True
    max_grad_norm: float = 1.0


class CompleteTrainer:
    """Complete training implementation with all optimizations."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Model
        self.model = CheckpointedModel(
            config.vocab_size,
            config.d_model,
            config.num_layers,
            config.num_heads,
            config.d_ff,
            checkpoint_every=1 if config.use_checkpoint else 999
        )

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            config.warmup_steps,
            config.total_steps
        )

        # AMP - GradScaler is mainly for CUDA with FP16
        # Note: MPS and CPU work fine without scaling
        self.scaler = GradScaler('cuda') if config.use_amp and torch.cuda.is_available() else None

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.step = 0
        self.accumulated_loss = 0.0

    def train_step(self, batch: torch.Tensor) -> Optional[dict]:
        """Single training step."""
        self.model.train()

        # Determine device type for autocast
        device_type = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

        if self.config.use_amp:
            with autocast(device_type=device_type, dtype=torch.float16):
                logits = self.model(batch)
                loss = self.criterion(
                    logits.view(-1, self.config.vocab_size),
                    batch.view(-1)
                )

            # Normalize and backward
            loss = loss / self.config.accumulation_steps
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()  # MPS/CPU don't need scaling
        else:
            logits = self.model(batch)
            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                batch.view(-1)
            )
            loss = loss / self.config.accumulation_steps
            loss.backward()

        self.accumulated_loss += loss.item() * self.config.accumulation_steps
        self.step += 1

        # Update on accumulation boundary
        if self.step % self.config.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            lr = self.scheduler.step()

            metrics = {
                'loss': self.accumulated_loss / self.config.accumulation_steps,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'lr': lr,
                'scale': self.scaler.get_scale() if self.scaler is not None else 1.0
            }
            self.accumulated_loss = 0.0
            return metrics

        return None


# Test complete training loop
print("\n--- Testing Complete Training Loop ---")

config = TrainingConfig(
    vocab_size=1000,
    d_model=256,
    num_layers=4,
    accumulation_steps=2,
    use_amp=True,
    use_checkpoint=True,
    total_steps=50,
    warmup_steps=10
)

trainer = CompleteTrainer(config)
print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

# Training loop
losses = []
for step in range(100):  # 100 micro-steps = 50 updates
    batch = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
    metrics = trainer.train_step(batch)

    if metrics is not None:
        losses.append(metrics['loss'])
        if len(losses) % 10 == 0:
            print(f"Step {len(losses)}: loss={metrics['loss']:.4f}, "
                  f"lr={metrics['lr']:.6f}, grad_norm={metrics['grad_norm']:.4f}")

print(f"\nFinal loss: {losses[-1]:.4f}")


# ===========================================================================
# Section 6: Memory Estimation
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: Memory Estimation")
print("=" * 70)


def estimate_training_memory(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    precision: str = 'fp32',
    use_checkpoint: bool = False,
    optimizer: str = 'adam'
) -> dict:
    """Estimate GPU memory requirements for training."""

    # Bytes per element
    bytes_per_elem = {'fp32': 4, 'fp16': 2, 'bf16': 2}[precision]

    # Model parameters
    param_memory = num_params * bytes_per_elem
    # FP32 master weights (for mixed precision)
    if precision in ['fp16', 'bf16']:
        param_memory += num_params * 4  # FP32 copy

    # Gradients
    grad_memory = num_params * 4  # Usually FP32

    # Optimizer states
    if optimizer == 'adam':
        optim_memory = num_params * 8  # m and v, each FP32
    elif optimizer == 'sgd':
        optim_memory = 0
    else:
        optim_memory = num_params * 4  # Rough estimate

    # Activations (rough estimate)
    # Each layer: batch × seq × hidden × 2 (forward + attention)
    activation_per_layer = batch_size * seq_len * hidden_dim * 2 * bytes_per_elem

    if use_checkpoint:
        # Only store checkpoint activations
        activation_memory = activation_per_layer * math.sqrt(num_layers)
    else:
        activation_memory = activation_per_layer * num_layers

    total = param_memory + grad_memory + optim_memory + activation_memory

    return {
        'params_gb': param_memory / 1e9,
        'grads_gb': grad_memory / 1e9,
        'optim_gb': optim_memory / 1e9,
        'activations_gb': activation_memory / 1e9,
        'total_gb': total / 1e9
    }


# Memory estimates for different model sizes
print("\nMemory Estimation for Different Models:")
print("-" * 70)

models = [
    ('GPT-2 Small', 117e6, 768, 12),
    ('GPT-2 Large', 774e6, 1280, 36),
    ('LLaMA-7B', 7e9, 4096, 32),
    ('LLaMA-13B', 13e9, 5120, 40),
]

batch_size, seq_len = 4, 2048

print(f"{'Model':<15} {'Params':<15} {'Grads':<10} {'Optim':<10} {'Acts':<10} {'Total':<10}")
print("-" * 70)

for name, params, hidden, layers in models:
    mem = estimate_training_memory(
        int(params), batch_size, seq_len, hidden, layers,
        precision='bf16', use_checkpoint=True, optimizer='adam'
    )
    print(f"{name:<15} {mem['params_gb']:<15.1f} {mem['grads_gb']:<10.1f} "
          f"{mem['optim_gb']:<10.1f} {mem['activations_gb']:<10.1f} {mem['total_gb']:<10.1f}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 12.1 Summary: Efficient Training Techniques")
print("=" * 70)

print("""
Key Takeaways:
==============

1. Mixed Precision Training:
   - Use BF16 (simpler) or FP16 (needs loss scaling)
   - 2x memory reduction, 2-4x speedup
   - PyTorch AMP: autocast() + GradScaler()

2. Gradient Checkpointing:
   - Trade compute for memory
   - ~30% overhead, √N memory reduction
   - checkpoint() function in PyTorch

3. Gradient Accumulation:
   - Simulate large batches
   - Divide loss by accumulation steps
   - Update only every N steps

4. Learning Rate Schedules:
   - Warmup + Cosine decay (standard)
   - WSD for some models
   - Linear scaling rule for batch size

5. Memory Optimization:
   - 8-bit optimizers (bitsandbytes)
   - CPU offloading (DeepSpeed)
   - Activation checkpointing

6. Training Stability:
   - Gradient clipping (max_norm=1.0)
   - Warmup (2000+ steps for LLMs)
   - Monitor loss, gradients, scale

Files created:
- 01_efficient_training.md: Theory and concepts
- 01_efficient_training.py: This implementation file
- lr_schedules.png: LR schedule visualization
""")

print("\nModule 12.1 complete!")
