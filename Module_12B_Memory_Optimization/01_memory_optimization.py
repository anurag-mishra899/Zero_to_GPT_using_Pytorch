"""
Memory Optimization & OOM Handling - Hands-on Implementation
=============================================================

This module covers:
1. GPU memory profiling and monitoring
2. Gradient accumulation implementation
3. Mixed precision training (FP16/BF16)
4. Gradient checkpointing
5. Comprehensive benchmarks with visualizations
6. OOM debugging utilities

Author: Zero to GPT Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, List, Tuple, Callable
import time
import gc
from dataclasses import dataclass, field
import math
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
import warnings


# =============================================================================
# Section 1: Memory Monitoring Utilities
# =============================================================================

class MemoryMonitor:
    """
    Monitor GPU memory usage during training.

    Tracks peak memory, current allocation, and provides profiling utilities.
    """

    def __init__(self):
        self.memory_history = []
        self.timestamps = []
        self.tags = []
        self.start_time = None

    def reset(self):
        """Reset memory monitoring."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.memory_history = []
        self.timestamps = []
        self.tags = []
        self.start_time = time.time()

    def record(self, tag: str = ""):
        """Record current memory state."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
        else:
            allocated = reserved = peak = 0

        elapsed = time.time() - self.start_time if self.start_time else 0

        self.memory_history.append({
            'allocated': allocated,
            'reserved': reserved,
            'peak': peak
        })
        self.timestamps.append(elapsed)
        self.tags.append(tag)

        return allocated, reserved, peak

    def print_current(self, tag: str = ""):
        """Print current memory usage."""
        allocated, reserved, peak = self.record(tag)
        print(f"[{tag}] Allocated: {allocated:.2f} GB, "
              f"Reserved: {reserved:.2f} GB, Peak: {peak:.2f} GB")

    def get_peak_memory(self) -> float:
        """Get peak memory in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return 0

    def plot_memory_timeline(self, save_path: str = 'memory_timeline.png'):
        """Plot memory usage over time."""
        if not self.memory_history:
            print("No memory data recorded")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        allocated = [m['allocated'] for m in self.memory_history]
        reserved = [m['reserved'] for m in self.memory_history]
        peak = [m['peak'] for m in self.memory_history]

        ax.plot(self.timestamps, allocated, label='Allocated', linewidth=2)
        ax.plot(self.timestamps, reserved, label='Reserved', linewidth=2)
        ax.axhline(y=max(peak), color='r', linestyle='--',
                   label=f'Peak: {max(peak):.2f} GB')

        # Add tag annotations
        for i, tag in enumerate(self.tags):
            if tag:
                ax.annotate(tag, (self.timestamps[i], allocated[i]),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, rotation=45)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved memory timeline to {save_path}")


@contextmanager
def memory_profiler(tag: str = ""):
    """Context manager for memory profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        delta = (end_mem - start_mem) / 1e9
        peak = peak_mem / 1e9

        print(f"[{tag}] Memory delta: {delta:+.2f} GB, Peak: {peak:.2f} GB")


def get_model_memory(model: nn.Module) -> Dict[str, float]:
    """Get memory breakdown for a model."""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9

    grad_memory = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters() if p.grad is not None
    ) / 1e9

    return {
        'parameters': param_memory,
        'buffers': buffer_memory,
        'gradients': grad_memory,
        'total': param_memory + buffer_memory + grad_memory
    }


# =============================================================================
# Section 2: Model for Benchmarking
# =============================================================================

class TransformerBlock(nn.Module):
    """Standard transformer block for benchmarking."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class BenchmarkModel(nn.Module):
    """
    Configurable transformer model for memory benchmarking.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.output.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)

        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        # Transformer layers
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return type('Output', (), {'logits': logits, 'loss': loss})()


# =============================================================================
# Section 3: Gradient Accumulation
# =============================================================================

class GradientAccumulationTrainer:
    """
    Trainer with gradient accumulation support.

    Allows training with larger effective batch sizes without increasing memory.
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

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with gradient accumulation.

        Returns metrics dict.
        """
        self.model.train()

        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass
        outputs = self.model(input_ids, labels=labels)

        # Scale loss by accumulation steps
        loss = outputs.loss / self.accumulation_steps
        loss.backward()

        self.step_count += 1

        metrics = {'loss': outputs.loss.item()}

        # Update weights every accumulation_steps
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            metrics['grad_norm'] = grad_norm.item()

            self.optimizer.step()
            self.optimizer.zero_grad()
            metrics['updated'] = True
        else:
            metrics['updated'] = False

        return metrics

    def get_effective_batch_size(self, micro_batch_size: int) -> int:
        """Get effective batch size."""
        return micro_batch_size * self.accumulation_steps


# =============================================================================
# Section 4: Mixed Precision Training
# =============================================================================

class MixedPrecisionTrainer:
    """
    Trainer with automatic mixed precision support.

    Supports both FP16 (with loss scaling) and BF16 (without loss scaling).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        precision: str = 'fp16',  # 'fp32', 'fp16', 'bf16'
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.precision = precision
        self.max_grad_norm = max_grad_norm

        # Setup for FP16 (needs loss scaling)
        self.scaler = GradScaler() if precision == 'fp16' else None

        # Determine autocast dtype
        if precision == 'fp16':
            self.autocast_dtype = torch.float16
        elif precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision."""
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass with autocast
        if self.autocast_dtype is not None:
            with autocast(dtype=self.autocast_dtype):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

        # Backward pass
        if self.scaler is not None:
            # FP16 with loss scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BF16 or FP32 (no scaling needed)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'scale': self.scaler.get_scale() if self.scaler else 1.0
        }


# =============================================================================
# Section 5: Combined Trainer with All Optimizations
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for optimized training."""
    # Batch settings
    batch_size: int = 8
    accumulation_steps: int = 1

    # Precision
    precision: str = 'fp32'  # 'fp32', 'fp16', 'bf16'

    # Checkpointing
    gradient_checkpointing: bool = False

    # Optimization
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0

    # Model settings
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    seq_len: int = 256

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps


class OptimizedTrainer:
    """
    Trainer combining all memory optimization techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device = None
    ):
        self.config = config
        self.device = device or (torch.device('cuda') if torch.cuda.is_available()
                                  else torch.device('cpu'))

        # Move model and setup
        self.model = model.to(self.device)

        # Setup mixed precision
        if config.precision == 'fp16':
            self.autocast_dtype = torch.float16
            self.scaler = GradScaler()
        elif config.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
            self.scaler = None
        else:
            self.autocast_dtype = None
            self.scaler = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        self.step_count = 0
        self.memory_monitor = MemoryMonitor()

    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.autocast_dtype is not None:
            with autocast(dtype=self.autocast_dtype):
                yield
        else:
            yield

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with all optimizations.
        """
        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward with autocast
        with self.autocast_context():
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss / self.config.accumulation_steps

        # Backward with optional scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.step_count += 1
        metrics = {'loss': outputs.loss.item()}

        # Update every accumulation_steps
        if self.step_count % self.config.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                metrics['scale'] = self.scaler.get_scale()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            metrics['grad_norm'] = grad_norm.item()
            metrics['updated'] = True
        else:
            metrics['updated'] = False

        return metrics


# =============================================================================
# Section 6: Benchmarking Framework
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    batch_size: int
    effective_batch_size: int
    peak_memory_gb: float
    avg_step_time_ms: float
    throughput_samples_per_sec: float
    final_loss: float
    precision: str
    checkpointing: bool
    accumulation_steps: int


class MemoryBenchmark:
    """
    Comprehensive memory benchmarking framework.

    Tests different configurations and produces comparison visualizations.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        seq_len: int = 256,
        device: torch.device = None
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.device = device or (torch.device('cuda') if torch.cuda.is_available()
                                  else torch.device('cpu'))

        self.results: List[BenchmarkResult] = []

    def create_model(self, use_checkpointing: bool = False) -> nn.Module:
        """Create model for benchmarking."""
        return BenchmarkModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_seq_len=self.seq_len,
            use_checkpointing=use_checkpointing
        )

    def create_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Create random batch for benchmarking."""
        input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        labels = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        return {'input_ids': input_ids, 'labels': labels}

    def run_single_benchmark(
        self,
        config_name: str,
        batch_size: int,
        precision: str = 'fp32',
        accumulation_steps: int = 1,
        use_checkpointing: bool = False,
        num_steps: int = 20,
        warmup_steps: int = 5
    ) -> Optional[BenchmarkResult]:
        """
        Run a single benchmark configuration.
        """
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            # Create model
            model = self.create_model(use_checkpointing=use_checkpointing)

            # Create config
            config = TrainingConfig(
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                precision=precision,
                gradient_checkpointing=use_checkpointing,
                d_model=self.d_model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                seq_len=self.seq_len
            )

            # Create trainer
            trainer = OptimizedTrainer(model, config, self.device)

            # Warmup
            batch = self.create_batch(batch_size)
            for _ in range(warmup_steps):
                trainer.train_step(batch)

            # Reset timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # Benchmark
            start_time = time.time()
            losses = []

            for step in range(num_steps):
                batch = self.create_batch(batch_size)
                metrics = trainer.train_step(batch)
                losses.append(metrics['loss'])

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start_time

            # Collect metrics
            peak_memory = trainer.memory_monitor.get_peak_memory() if torch.cuda.is_available() else 0
            avg_step_time = (elapsed / num_steps) * 1000  # ms
            throughput = (batch_size * num_steps) / elapsed

            result = BenchmarkResult(
                config_name=config_name,
                batch_size=batch_size,
                effective_batch_size=batch_size * accumulation_steps,
                peak_memory_gb=peak_memory,
                avg_step_time_ms=avg_step_time,
                throughput_samples_per_sec=throughput,
                final_loss=np.mean(losses[-5:]),
                precision=precision,
                checkpointing=use_checkpointing,
                accumulation_steps=accumulation_steps
            )

            self.results.append(result)
            return result

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM: {config_name} with batch_size={batch_size}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            raise

    def run_all_benchmarks(self, base_batch_size: int = 4):
        """Run comprehensive benchmark suite."""
        print("=" * 70)
        print("Running Memory Optimization Benchmarks")
        print("=" * 70)

        configs = [
            # Baseline
            ("FP32 Baseline", base_batch_size, 'fp32', 1, False),

            # Mixed precision
            ("FP16", base_batch_size, 'fp16', 1, False),
            ("BF16", base_batch_size, 'bf16', 1, False),

            # Checkpointing
            ("FP32 + Checkpointing", base_batch_size, 'fp32', 1, True),
            ("FP16 + Checkpointing", base_batch_size, 'fp16', 1, True),

            # Gradient accumulation
            ("FP32 + Accum(4x)", base_batch_size, 'fp32', 4, False),
            ("FP16 + Accum(4x)", base_batch_size, 'fp16', 4, False),

            # Combined
            ("FP16 + Checkpoint + Accum(4x)", base_batch_size, 'fp16', 4, True),

            # Larger batch with optimizations
            ("FP16 (2x batch)", base_batch_size * 2, 'fp16', 1, False),
            ("FP16 + Checkpoint (2x batch)", base_batch_size * 2, 'fp16', 1, True),
        ]

        for name, bs, precision, accum, ckpt in configs:
            print(f"\nRunning: {name}")

            # Check if BF16 is supported
            if precision == 'bf16' and not (torch.cuda.is_available() and
                torch.cuda.is_bf16_supported()):
                print(f"  Skipping {name}: BF16 not supported")
                continue

            result = self.run_single_benchmark(
                config_name=name,
                batch_size=bs,
                precision=precision,
                accumulation_steps=accum,
                use_checkpointing=ckpt
            )

            if result:
                print(f"  Peak Memory: {result.peak_memory_gb:.2f} GB")
                print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
                print(f"  Step Time: {result.avg_step_time_ms:.1f} ms")

    def find_max_batch_size(
        self,
        precision: str = 'fp32',
        use_checkpointing: bool = False,
        start_batch: int = 2,
        max_batch: int = 128
    ) -> int:
        """Find maximum batch size that fits in memory."""
        print(f"\nFinding max batch size for {precision}"
              f"{' + checkpointing' if use_checkpointing else ''}...")

        max_working = 0
        batch_size = start_batch

        while batch_size <= max_batch:
            result = self.run_single_benchmark(
                config_name=f"probe_bs{batch_size}",
                batch_size=batch_size,
                precision=precision,
                use_checkpointing=use_checkpointing,
                num_steps=3,
                warmup_steps=1
            )

            if result is not None:
                max_working = batch_size
                print(f"  Batch size {batch_size}: OK ({result.peak_memory_gb:.1f} GB)")
                batch_size *= 2
            else:
                break

        # Binary search between max_working and batch_size
        low, high = max_working, min(batch_size, max_batch)
        while low < high - 1:
            mid = (low + high) // 2
            result = self.run_single_benchmark(
                config_name=f"probe_bs{mid}",
                batch_size=mid,
                precision=precision,
                use_checkpointing=use_checkpointing,
                num_steps=3,
                warmup_steps=1
            )

            if result is not None:
                low = mid
            else:
                high = mid

        print(f"  Max batch size: {low}")
        return low

    def plot_results(self, save_prefix: str = 'benchmark'):
        """Generate comparison visualizations."""
        if not self.results:
            print("No benchmark results to plot")
            return

        # Filter out probe results
        results = [r for r in self.results if not r.config_name.startswith('probe')]

        if not results:
            print("No non-probe results to plot")
            return

        # 1. Memory comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Memory bar chart
        ax = axes[0, 0]
        names = [r.config_name for r in results]
        memories = [r.peak_memory_gb for r in results]

        bars = ax.barh(range(len(names)), memories)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Peak Memory (GB)')
        ax.set_title('Peak Memory Usage')

        # Color bars by category
        for i, r in enumerate(results):
            if 'Checkpoint' in r.config_name:
                bars[i].set_color('green')
            elif 'FP16' in r.config_name or 'BF16' in r.config_name:
                bars[i].set_color('blue')
            elif 'Accum' in r.config_name:
                bars[i].set_color('orange')

        ax.axvline(x=memories[0], color='red', linestyle='--', alpha=0.5,
                   label='Baseline')
        ax.legend()

        # Throughput comparison
        ax = axes[0, 1]
        throughputs = [r.throughput_samples_per_sec for r in results]

        bars = ax.barh(range(len(names)), throughputs)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Throughput (samples/sec)')
        ax.set_title('Training Throughput')

        # Memory vs Throughput scatter
        ax = axes[1, 0]
        for r in results:
            color = 'red' if 'Baseline' in r.config_name else \
                    'green' if 'Checkpoint' in r.config_name else \
                    'blue' if ('FP16' in r.config_name or 'BF16' in r.config_name) else 'gray'
            ax.scatter(r.peak_memory_gb, r.throughput_samples_per_sec,
                      s=100, c=color, label=r.config_name, alpha=0.7)

        ax.set_xlabel('Peak Memory (GB)')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Memory vs Throughput Trade-off')
        ax.grid(True, alpha=0.3)

        # Summary table
        ax = axes[1, 1]
        ax.axis('off')

        table_data = []
        for r in results:
            table_data.append([
                r.config_name[:25],
                f"{r.peak_memory_gb:.1f}",
                f"{r.throughput_samples_per_sec:.0f}",
                f"{r.effective_batch_size}"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Configuration', 'Memory (GB)', 'Throughput', 'Eff. Batch'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_comparison.png', dpi=150)
        plt.close()
        print(f"Saved {save_prefix}_comparison.png")

        # 2. Memory savings visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        baseline_mem = results[0].peak_memory_gb if results else 1

        savings = [(baseline_mem - r.peak_memory_gb) / baseline_mem * 100
                   for r in results]
        colors = ['green' if s > 0 else 'red' for s in savings]

        bars = ax.barh(range(len(names)), savings, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Memory Savings vs Baseline (%)')
        ax.set_title('Memory Savings Comparison')
        ax.axvline(x=0, color='black', linewidth=0.5)

        for i, (bar, saving) in enumerate(zip(bars, savings)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{saving:.0f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_savings.png', dpi=150)
        plt.close()
        print(f"Saved {save_prefix}_savings.png")


# =============================================================================
# Section 7: OOM Recovery Utilities
# =============================================================================

class OOMHandler:
    """
    Utilities for handling and recovering from OOM errors.
    """

    @staticmethod
    def clear_memory():
        """Aggressively clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def find_safe_batch_size(
        model_fn: Callable,
        batch_fn: Callable,
        train_step_fn: Callable,
        start_batch: int = 32,
        min_batch: int = 1
    ) -> int:
        """
        Find the largest batch size that doesn't OOM.

        Args:
            model_fn: Function that creates the model
            batch_fn: Function that creates a batch given batch_size
            train_step_fn: Function that runs one training step
            start_batch: Initial batch size to try
            min_batch: Minimum acceptable batch size
        """
        batch_size = start_batch

        while batch_size >= min_batch:
            try:
                OOMHandler.clear_memory()

                model = model_fn()
                batch = batch_fn(batch_size)

                # Try forward + backward
                train_step_fn(model, batch)

                print(f"Batch size {batch_size}: OK")
                return batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Batch size {batch_size}: OOM")
                    batch_size //= 2
                    OOMHandler.clear_memory()
                else:
                    raise

        return min_batch

    @staticmethod
    def safe_train_step(
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None,
        max_retries: int = 3
    ) -> Optional[Dict[str, float]]:
        """
        Train step with OOM recovery.

        On OOM, clears cache and retries with smaller effective batch.
        """
        for attempt in range(max_retries):
            try:
                optimizer.zero_grad()

                input_ids = batch['input_ids']
                labels = batch['labels']

                if scaler is not None:
                    with autocast(dtype=torch.float16):
                        outputs = model(input_ids, labels=labels)
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                return {'loss': loss.item()}

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at attempt {attempt + 1}, clearing memory...")
                    OOMHandler.clear_memory()

                    if attempt == max_retries - 1:
                        print("Max retries reached, skipping batch")
                        return None
                else:
                    raise

        return None


# =============================================================================
# Section 8: Demonstrations
# =============================================================================

def demo_memory_monitoring():
    """Demonstrate memory monitoring."""
    print("=" * 60)
    print("Demo: Memory Monitoring")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory demo")
        print("Running CPU-only demonstration...")

    monitor = MemoryMonitor()
    monitor.reset()

    # Create model
    monitor.print_current("Before model")

    model = BenchmarkModel(
        vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512
    )

    if torch.cuda.is_available():
        model = model.cuda()

    monitor.print_current("After model creation")

    # Create batch
    batch_size = 4
    seq_len = 128
    device = next(model.parameters()).device

    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 10000, (batch_size, seq_len), device=device)

    monitor.print_current("After batch creation")

    # Forward pass
    outputs = model(input_ids, labels=labels)
    monitor.print_current("After forward")

    # Backward pass
    outputs.loss.backward()
    monitor.print_current("After backward")

    # Get model memory breakdown
    mem_breakdown = get_model_memory(model)
    print(f"\nModel Memory Breakdown:")
    for key, value in mem_breakdown.items():
        print(f"  {key}: {value:.3f} GB")


def demo_gradient_accumulation():
    """Demonstrate gradient accumulation."""
    print("\n" + "=" * 60)
    print("Demo: Gradient Accumulation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BenchmarkModel(
        vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Compare: standard vs accumulated
    print("\nStandard training (batch_size=8):")
    batch_size = 8
    for step in range(3):
        batch = {
            'input_ids': torch.randint(0, 10000, (batch_size, 128), device=device),
            'labels': torch.randint(0, 10000, (batch_size, 128), device=device)
        }

        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        print(f"  Step {step}: loss = {outputs.loss.item():.4f}")

    print("\nGradient accumulation (batch_size=2, accum_steps=4, effective=8):")
    trainer = GradientAccumulationTrainer(model, optimizer, accumulation_steps=4)

    for step in range(12):  # 12 steps = 3 effective steps
        batch = {
            'input_ids': torch.randint(0, 10000, (2, 128), device=device),
            'labels': torch.randint(0, 10000, (2, 128), device=device)
        }

        metrics = trainer.train_step(batch)
        if metrics['updated']:
            print(f"  Effective step {step // 4}: loss = {metrics['loss']:.4f}")


def demo_mixed_precision():
    """Demonstrate mixed precision training."""
    print("\n" + "=" * 60)
    print("Demo: Mixed Precision Training")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision demo")
        return

    device = torch.device('cuda')

    # Compare precisions
    precisions = ['fp32', 'fp16']
    if torch.cuda.is_bf16_supported():
        precisions.append('bf16')

    for precision in precisions:
        print(f"\n{precision.upper()} Training:")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = BenchmarkModel(
            vocab_size=10000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=512
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = MixedPrecisionTrainer(model, optimizer, precision=precision)

        # Train a few steps
        start_time = time.time()
        for step in range(10):
            batch = {
                'input_ids': torch.randint(0, 10000, (8, 128), device=device),
                'labels': torch.randint(0, 10000, (8, 128), device=device)
            }
            metrics = trainer.train_step(batch)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  Time for 10 steps: {elapsed:.2f}s")
        print(f"  Final loss: {metrics['loss']:.4f}")


def demo_checkpointing():
    """Demonstrate gradient checkpointing."""
    print("\n" + "=" * 60)
    print("Demo: Gradient Checkpointing")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping checkpointing demo")
        return

    device = torch.device('cuda')

    for use_checkpoint in [False, True]:
        tag = "With checkpointing" if use_checkpoint else "Without checkpointing"
        print(f"\n{tag}:")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = BenchmarkModel(
            vocab_size=10000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            d_ff=512,
            use_checkpointing=use_checkpoint
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Train
        start_time = time.time()
        for step in range(5):
            batch = {
                'input_ids': torch.randint(0, 10000, (8, 256), device=device),
                'labels': torch.randint(0, 10000, (8, 256), device=device)
            }

            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  Time for 5 steps: {elapsed:.2f}s")


def demo_full_benchmark():
    """Run full benchmark suite with visualizations."""
    print("\n" + "=" * 60)
    print("Demo: Full Benchmark Suite")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, running minimal CPU benchmark...")

        # CPU-only mini benchmark
        benchmark = MemoryBenchmark(
            vocab_size=5000,
            d_model=128,
            n_layers=2,
            n_heads=2,
            d_ff=256,
            seq_len=64,
            device=torch.device('cpu')
        )

        benchmark.run_single_benchmark(
            "CPU Baseline",
            batch_size=2,
            precision='fp32',
            num_steps=3
        )
        return

    # GPU benchmark
    benchmark = MemoryBenchmark(
        vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        seq_len=128
    )

    # Run benchmarks
    benchmark.run_all_benchmarks(base_batch_size=4)

    # Generate visualizations
    benchmark.plot_results('memory_benchmark')

    # Find max batch sizes
    print("\n" + "-" * 40)
    print("Finding maximum batch sizes:")
    benchmark.find_max_batch_size('fp32', False)
    benchmark.find_max_batch_size('fp16', False)
    benchmark.find_max_batch_size('fp16', True)


if __name__ == "__main__":
    print("Memory Optimization & OOM Handling - Hands-on Implementation")
    print("=" * 70)

    # Run demos
    demo_memory_monitoring()
    demo_gradient_accumulation()
    demo_mixed_precision()
    demo_checkpointing()

    # Full benchmark (uncomment if you have time and GPU)
    # demo_full_benchmark()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
    print("\nTo run full benchmarks with visualizations:")
    print("  Uncomment demo_full_benchmark() and run again")
