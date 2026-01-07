"""
Training Dynamics & Debugging LLMs - Hands-on Implementation
=============================================================

This module covers:
1. Loss curve analysis and diagnostics
2. Gradient monitoring and health checks
3. Learning rate scheduling and finding
4. Debugging tools and techniques
5. Training stability utilities
6. Distributed training helpers

Author: Zero to GPT Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Callable
import math
import copy
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import warnings


# =============================================================================
# Section 1: Loss Tracking and Analysis
# =============================================================================

class LossTracker:
    """
    Track and analyze training loss over time.

    Detects spikes, plateaus, and other anomalies.
    """

    def __init__(
        self,
        window_size: int = 100,
        spike_threshold: float = 3.0,
        plateau_threshold: float = 0.001,
        plateau_patience: int = 500
    ):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience

        self.losses = []
        self.smoothed_losses = []
        self.running_window = deque(maxlen=window_size)

        self.spike_steps = []
        self.plateau_start = None
        self.in_plateau = False

    def update(self, loss: float, step: int) -> Dict[str, any]:
        """
        Update tracker with new loss value.

        Returns dict with analysis results.
        """
        result = {
            'step': step,
            'loss': loss,
            'is_spike': False,
            'is_plateau': False,
            'running_mean': None,
            'running_std': None
        }

        self.losses.append(loss)
        self.running_window.append(loss)

        if len(self.running_window) >= 10:
            running_mean = np.mean(self.running_window)
            running_std = np.std(self.running_window)

            result['running_mean'] = running_mean
            result['running_std'] = running_std

            # Detect spike
            if len(self.running_window) >= self.window_size:
                z_score = (loss - running_mean) / (running_std + 1e-8)
                if z_score > self.spike_threshold:
                    result['is_spike'] = True
                    self.spike_steps.append(step)

            # Detect plateau
            if len(self.losses) > self.plateau_patience:
                recent_losses = self.losses[-self.plateau_patience:]
                loss_range = max(recent_losses) - min(recent_losses)

                if loss_range < self.plateau_threshold * running_mean:
                    if not self.in_plateau:
                        self.plateau_start = step - self.plateau_patience
                        self.in_plateau = True
                    result['is_plateau'] = True
                else:
                    self.in_plateau = False
                    self.plateau_start = None

            self.smoothed_losses.append(running_mean)

        return result

    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics."""
        if not self.losses:
            return {}

        return {
            'total_steps': len(self.losses),
            'final_loss': self.losses[-1],
            'min_loss': min(self.losses),
            'max_loss': max(self.losses),
            'num_spikes': len(self.spike_steps),
            'spike_steps': self.spike_steps,
            'currently_in_plateau': self.in_plateau,
            'plateau_start_step': self.plateau_start
        }

    def plot(self, save_path: str = 'loss_curve.png'):
        """Plot loss curve with annotations."""
        if not self.losses:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot raw loss
        ax.plot(self.losses, alpha=0.3, label='Raw Loss')

        # Plot smoothed loss
        if self.smoothed_losses:
            offset = len(self.losses) - len(self.smoothed_losses)
            x = range(offset, len(self.losses))
            ax.plot(x, self.smoothed_losses, label='Smoothed Loss', linewidth=2)

        # Mark spikes
        for spike_step in self.spike_steps:
            ax.axvline(x=spike_step, color='red', linestyle='--', alpha=0.5)

        # Mark plateau
        if self.plateau_start is not None:
            ax.axvspan(self.plateau_start, len(self.losses),
                      color='yellow', alpha=0.2, label='Plateau')

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved loss curve to {save_path}")


class TrainValTracker:
    """Track training and validation loss for overfitting detection."""

    def __init__(self, patience: int = 5):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def update(self, epoch: int, train_loss: float, val_loss: float) -> Dict[str, bool]:
        """Update with new epoch results."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        result = {
            'is_overfitting': False,
            'should_stop': False,
            'is_best': False
        }

        # Check for overfitting
        if len(self.train_losses) > 1:
            train_improving = train_loss < self.train_losses[-2]
            val_worse = val_loss > self.val_losses[-2]

            if train_improving and val_worse:
                result['is_overfitting'] = True

        # Early stopping check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            result['is_best'] = True
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            result['should_stop'] = True

        return result

    def plot(self, save_path: str = 'train_val_loss.png'):
        """Plot train vs validation loss."""
        if not self.epochs:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
        ax.plot(self.epochs, self.val_losses, label='Val Loss', marker='s')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


# =============================================================================
# Section 2: Gradient Monitoring
# =============================================================================

class GradientMonitor:
    """
    Monitor gradient statistics during training.

    Tracks gradient norms, detects vanishing/exploding gradients.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.history = {
            'total_norm': [],
            'per_layer_norm': [],
            'max_grad': [],
            'min_grad': []
        }

    def compute_gradient_stats(self) -> Dict[str, float]:
        """Compute current gradient statistics."""
        total_norm = 0.0
        per_layer = {}
        max_grad = float('-inf')
        min_grad = float('inf')
        num_nan = 0
        num_inf = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    num_nan += 1
                if torch.isinf(grad).any():
                    num_inf += 1

                param_norm = grad.norm(2).item()
                per_layer[name] = param_norm
                total_norm += param_norm ** 2

                max_grad = max(max_grad, grad.abs().max().item())
                min_grad = min(min_grad, grad.abs().min().item())

        total_norm = total_norm ** 0.5

        stats = {
            'total_norm': total_norm,
            'per_layer_norm': per_layer,
            'max_grad': max_grad,
            'min_grad': min_grad,
            'num_nan': num_nan,
            'num_inf': num_inf
        }

        # Update history
        self.history['total_norm'].append(total_norm)
        self.history['max_grad'].append(max_grad)
        self.history['min_grad'].append(min_grad)

        return stats

    def check_gradient_health(self) -> Dict[str, any]:
        """Check for gradient pathologies."""
        stats = self.compute_gradient_stats()
        issues = []

        # Check for NaN/Inf
        if stats['num_nan'] > 0:
            issues.append(f"NaN gradients in {stats['num_nan']} parameters")

        if stats['num_inf'] > 0:
            issues.append(f"Inf gradients in {stats['num_inf']} parameters")

        # Check for explosion
        if stats['total_norm'] > 100:
            issues.append(f"Exploding gradients: norm = {stats['total_norm']:.2f}")

        # Check for vanishing
        if stats['total_norm'] < 1e-7:
            issues.append(f"Vanishing gradients: norm = {stats['total_norm']:.2e}")

        # Check per-layer
        vanishing_layers = []
        exploding_layers = []
        for name, norm in stats['per_layer_norm'].items():
            if norm < 1e-8:
                vanishing_layers.append(name)
            elif norm > 100:
                exploding_layers.append(name)

        if vanishing_layers:
            issues.append(f"Vanishing in layers: {vanishing_layers[:3]}")
        if exploding_layers:
            issues.append(f"Exploding in layers: {exploding_layers[:3]}")

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }

    def plot_gradient_flow(self, save_path: str = 'gradient_flow.png'):
        """Plot gradient magnitudes across layers."""
        stats = self.compute_gradient_stats()
        per_layer = stats['per_layer_norm']

        if not per_layer:
            return

        names = list(per_layer.keys())
        values = list(per_layer.values())

        # Shorten names for display
        short_names = [n.split('.')[-1] if len(n) > 20 else n for n in names]

        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(len(values)), values)

        # Color code based on magnitude
        for bar, val in zip(bars, values):
            if val < 1e-6:
                bar.set_color('blue')  # Vanishing
            elif val > 10:
                bar.set_color('red')   # Exploding
            else:
                bar.set_color('green') # Healthy

        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Flow (Blue=Vanishing, Red=Exploding, Green=Healthy)')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved gradient flow to {save_path}")

    def plot_gradient_history(self, save_path: str = 'gradient_history.png'):
        """Plot gradient norm over time."""
        if not self.history['total_norm']:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history['total_norm'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


# =============================================================================
# Section 3: Learning Rate Tools
# =============================================================================

class LRScheduler:
    """
    Flexible learning rate scheduler with multiple strategies.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        strategy: str = 'cosine',
        warmup_steps: int = 0,
        total_steps: int = 1000,
        max_lr: float = 1e-4,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

        self.lr_history = []

    def get_lr(self) -> float:
        """Compute current learning rate."""
        step = self.current_step

        # Warmup phase
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps

        # Post-warmup
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)

        if self.strategy == 'cosine':
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress))

        elif self.strategy == 'linear':
            return self.max_lr * (1 - progress) + self.min_lr * progress

        elif self.strategy == 'constant':
            return self.max_lr

        elif self.strategy == 'exponential':
            decay = (self.min_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))
            return self.max_lr * (decay ** (step - self.warmup_steps))

        else:
            return self.max_lr

    def step(self) -> float:
        """Update learning rate and step counter."""
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.lr_history.append(lr)
        self.current_step += 1

        return lr


def lr_finder(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    min_lr: float = 1e-7,
    max_lr: float = 10.0,
    num_steps: int = 100,
    device: torch.device = torch.device('cpu')
) -> Tuple[List[float], List[float]]:
    """
    Find optimal learning rate using the LR range test.

    Trains with exponentially increasing LR, plots loss vs LR.
    Optimal LR is typically where loss is steepest decreasing.

    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        min_lr: Starting learning rate
        max_lr: Ending learning rate
        num_steps: Number of steps to run
        device: Compute device

    Returns:
        (learning_rates, losses)
    """
    # Save model state for restoration
    model_state = copy.deepcopy(model.state_dict())

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)

    # Calculate LR multiplier
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)

    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    data_iter = iter(dataloader)

    for step in range(num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) for k, v in batch.items()}
        else:
            batch = [b.to(device) for b in batch]

        # Forward pass
        optimizer.zero_grad()

        if isinstance(batch, dict):
            outputs = model(**batch)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = criterion(outputs, batch.get('labels'))
        else:
            outputs = model(batch[0])
            loss = criterion(outputs, batch[1])

        # Check for divergence
        if torch.isnan(loss) or loss.item() > 4 * best_loss:
            print(f"Stopping LR finder at step {step} (diverging)")
            break

        best_loss = min(best_loss, loss.item())

        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # Backward
        loss.backward()
        optimizer.step()

        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

    # Restore model
    model.load_state_dict(model_state)

    return lrs, losses


def plot_lr_finder(lrs: List[float], losses: List[float], save_path: str = 'lr_finder.png'):
    """Plot LR finder results with suggested LR."""
    if not lrs or not losses:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate Finder')
    ax.grid(True, alpha=0.3)

    # Find suggested LR (steepest descent)
    smoothed = np.convolve(losses, np.ones(5)/5, mode='valid')
    if len(smoothed) > 1:
        gradients = np.gradient(smoothed)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx + 2]  # Offset for smoothing

        ax.axvline(x=suggested_lr, color='red', linestyle='--',
                  label=f'Suggested LR: {suggested_lr:.2e}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved LR finder plot to {save_path}")


# =============================================================================
# Section 4: Debugging Utilities
# =============================================================================

def overfit_single_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    num_steps: int = 100,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu')
) -> List[float]:
    """
    Test if model can overfit a single batch.

    If it can't, there's likely a bug in the model or training code.

    Returns:
        List of loss values
    """
    model = model.to(device)
    model.train()

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    final_loss = losses[-1]
    if final_loss > 0.1:
        print(f"\n⚠️ Warning: Could not overfit single batch (final loss: {final_loss:.4f})")
        print("Possible issues:")
        print("  - Bug in model forward pass")
        print("  - Bug in loss computation")
        print("  - Learning rate too low")
        print("  - Model capacity too small")
    else:
        print(f"\n✓ Successfully overfit single batch (final loss: {final_loss:.6f})")

    return losses


@torch.no_grad()
def check_model_outputs(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device = torch.device('cpu')
) -> Dict[str, any]:
    """
    Check model outputs for common issues.
    """
    model = model.to(device)
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}

    issues = []
    stats = {}

    # Forward pass
    outputs = model(**batch)

    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        logits = outputs['logits']
    else:
        logits = outputs

    # Check logits
    stats['logits_shape'] = tuple(logits.shape)
    stats['logits_mean'] = logits.mean().item()
    stats['logits_std'] = logits.std().item()
    stats['logits_min'] = logits.min().item()
    stats['logits_max'] = logits.max().item()

    # Check for issues
    if torch.isnan(logits).any():
        issues.append("NaN in logits")

    if torch.isinf(logits).any():
        issues.append("Inf in logits")

    if stats['logits_std'] < 0.01:
        issues.append(f"Very low logits variance: {stats['logits_std']:.4f}")

    if stats['logits_std'] > 100:
        issues.append(f"Very high logits variance: {stats['logits_std']:.4f}")

    # Check probabilities
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values

    stats['max_prob_mean'] = max_probs.mean().item()

    if stats['max_prob_mean'] > 0.99:
        issues.append("Model is overconfident (max prob > 0.99)")

    if stats['max_prob_mean'] < 0.1:
        issues.append("Model is very uncertain (max prob < 0.1)")

    return {
        'issues': issues,
        'healthy': len(issues) == 0,
        'stats': stats
    }


class ActivationHook:
    """
    Hook to collect activation statistics from model layers.
    """

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def register_hooks(self, model: nn.Module, layer_types: Tuple = (nn.Linear, nn.LayerNorm)):
        """Register hooks on specified layer types."""
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'dead_frac': (output.abs() < 1e-6).float().mean().item()
                }
        return hook

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get collected activation statistics."""
        return self.activations

    def check_health(self) -> Dict[str, any]:
        """Check for activation issues."""
        issues = []

        for name, stats in self.activations.items():
            if stats['dead_frac'] > 0.5:
                issues.append(f"{name}: {stats['dead_frac']*100:.1f}% dead neurons")

            if stats['std'] < 0.01:
                issues.append(f"{name}: very low activation variance ({stats['std']:.4f})")

        return {
            'healthy': len(issues) == 0,
            'issues': issues
        }

    def clear(self):
        """Clear collected activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# Section 5: Training Stability Utilities
# =============================================================================

class GradientClipper:
    """
    Gradient clipping with logging.
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_history = []

    def clip(self, model: nn.Module) -> float:
        """Clip gradients and return original norm."""
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type
        )

        pre_clip_norm = pre_clip_norm.item()
        was_clipped = pre_clip_norm > self.max_norm

        self.clip_history.append({
            'pre_clip_norm': pre_clip_norm,
            'was_clipped': was_clipped
        })

        return pre_clip_norm

    def get_clip_stats(self) -> Dict[str, float]:
        """Get clipping statistics."""
        if not self.clip_history:
            return {}

        norms = [h['pre_clip_norm'] for h in self.clip_history]
        clip_fraction = sum(h['was_clipped'] for h in self.clip_history) / len(self.clip_history)

        return {
            'mean_pre_clip_norm': np.mean(norms),
            'max_pre_clip_norm': np.max(norms),
            'clip_fraction': clip_fraction
        }


class NaNDetector:
    """
    Detect NaN values in model parameters and gradients.
    """

    @staticmethod
    def check_parameters(model: nn.Module) -> List[str]:
        """Check for NaN in parameters."""
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        return nan_params

    @staticmethod
    def check_gradients(model: nn.Module) -> List[str]:
        """Check for NaN in gradients."""
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads.append(name)
        return nan_grads

    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check if tensor contains NaN."""
        if torch.isnan(tensor).any():
            print(f"⚠️ NaN detected in {name}")
            return True
        return False


class EarlyStopping:
    """
    Early stopping based on validation loss.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Returns True if should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            return False


# =============================================================================
# Section 6: Comprehensive Training Monitor
# =============================================================================

@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)


class TrainingMonitor:
    """
    Comprehensive training monitor combining all debugging tools.
    """

    def __init__(
        self,
        model: nn.Module,
        log_every: int = 100,
        gradient_clip: float = 1.0
    ):
        self.model = model
        self.log_every = log_every

        # Initialize tools
        self.loss_tracker = LossTracker()
        self.grad_monitor = GradientMonitor(model)
        self.grad_clipper = GradientClipper(max_norm=gradient_clip)
        self.nan_detector = NaNDetector()

        # State
        self.state = TrainingState()

    def pre_backward_check(self, loss: torch.Tensor) -> bool:
        """
        Check loss before backward pass.

        Returns True if safe to proceed, False if should skip.
        """
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn(f"Invalid loss at step {self.state.step}: {loss.item()}")
            return False

        return True

    def post_backward_check(self) -> Dict[str, any]:
        """
        Check gradients after backward pass.
        """
        # Check for NaN gradients
        nan_grads = self.nan_detector.check_gradients(self.model)
        if nan_grads:
            warnings.warn(f"NaN gradients in: {nan_grads}")

        # Get gradient health
        grad_health = self.grad_monitor.check_gradient_health()

        # Clip gradients
        pre_clip_norm = self.grad_clipper.clip(self.model)
        self.state.gradient_norms.append(pre_clip_norm)

        return {
            'nan_gradients': nan_grads,
            'gradient_health': grad_health,
            'pre_clip_norm': pre_clip_norm
        }

    def log_step(
        self,
        loss: float,
        lr: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        Log training step.

        Returns analysis results.
        """
        # Update trackers
        loss_analysis = self.loss_tracker.update(loss, self.state.step)
        self.state.train_losses.append(loss)
        self.state.learning_rates.append(lr)

        result = {
            'step': self.state.step,
            'loss': loss,
            'lr': lr,
            **loss_analysis
        }

        # Periodic detailed logging
        if self.state.step % self.log_every == 0:
            clip_stats = self.grad_clipper.get_clip_stats()
            result['clip_stats'] = clip_stats

            if self.state.step > 0:
                print(f"Step {self.state.step}: "
                      f"loss={loss:.4f}, "
                      f"lr={lr:.2e}, "
                      f"grad_norm={self.state.gradient_norms[-1]:.2f}")

        self.state.step += 1
        return result

    def log_validation(self, val_loss: float) -> Dict[str, any]:
        """Log validation results."""
        self.state.val_losses.append(val_loss)

        is_best = val_loss < self.state.best_val_loss
        if is_best:
            self.state.best_val_loss = val_loss

        return {
            'val_loss': val_loss,
            'is_best': is_best,
            'best_val_loss': self.state.best_val_loss
        }

    def get_summary(self) -> Dict[str, any]:
        """Get training summary."""
        return {
            'total_steps': self.state.step,
            'loss_summary': self.loss_tracker.get_summary(),
            'clip_stats': self.grad_clipper.get_clip_stats(),
            'best_val_loss': self.state.best_val_loss
        }

    def save_plots(self, prefix: str = ''):
        """Save all diagnostic plots."""
        self.loss_tracker.plot(f'{prefix}loss_curve.png')
        self.grad_monitor.plot_gradient_flow(f'{prefix}gradient_flow.png')
        self.grad_monitor.plot_gradient_history(f'{prefix}gradient_history.png')


# =============================================================================
# Section 7: Demonstrations
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for demonstrations."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 128, n_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return type('Output', (), {'logits': logits, 'loss': loss})()


def demo_loss_tracking():
    """Demonstrate loss tracking and analysis."""
    print("=" * 60)
    print("Demo: Loss Tracking")
    print("=" * 60)

    tracker = LossTracker(window_size=50, spike_threshold=2.0)

    # Simulate training with some anomalies
    for step in range(500):
        # Normal decreasing loss
        base_loss = 5.0 * math.exp(-step / 200) + 0.5

        # Add noise
        loss = base_loss + np.random.normal(0, 0.1)

        # Add spike at step 200
        if step == 200:
            loss = base_loss * 3

        # Add plateau at end
        if step > 400:
            loss = 0.6 + np.random.normal(0, 0.01)

        result = tracker.update(loss, step)

        if result['is_spike']:
            print(f"  ⚠️ Spike detected at step {step}")
        if result['is_plateau']:
            print(f"  ⚠️ Plateau detected starting step {tracker.plateau_start}")

    summary = tracker.get_summary()
    print(f"\nSummary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final loss: {summary['final_loss']:.4f}")
    print(f"  Min loss: {summary['min_loss']:.4f}")
    print(f"  Num spikes: {summary['num_spikes']}")

    tracker.plot('demo_loss_curve.png')


def demo_gradient_monitoring():
    """Demonstrate gradient monitoring."""
    print("\n" + "=" * 60)
    print("Demo: Gradient Monitoring")
    print("=" * 60)

    model = SimpleModel()
    monitor = GradientMonitor(model)

    # Create dummy batch
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward + backward
    output = model(input_ids, labels)
    output.loss.backward()

    # Check gradient health
    health = monitor.check_gradient_health()
    print(f"\nGradient Health:")
    print(f"  Healthy: {health['healthy']}")
    print(f"  Total norm: {health['stats']['total_norm']:.4f}")
    if health['issues']:
        print(f"  Issues: {health['issues']}")

    monitor.plot_gradient_flow('demo_gradient_flow.png')


def demo_overfit_batch():
    """Demonstrate single batch overfitting test."""
    print("\n" + "=" * 60)
    print("Demo: Overfit Single Batch")
    print("=" * 60)

    model = SimpleModel()

    batch = {
        'input_ids': torch.randint(0, 1000, (4, 32)),
        'labels': torch.randint(0, 1000, (4, 32))
    }

    losses = overfit_single_batch(model, batch, num_steps=50, lr=1e-3)


def demo_lr_finder():
    """Demonstrate learning rate finder."""
    print("\n" + "=" * 60)
    print("Demo: Learning Rate Finder")
    print("=" * 60)

    model = SimpleModel()

    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 1000, (100, 32)),
        torch.randint(0, 1000, (100, 32))
    )
    dataloader = DataLoader(dataset, batch_size=8)

    # Wrap for the lr_finder
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, labels=None):
            return self.model(input_ids, labels)

    wrapper = ModelWrapper(model)

    def criterion(output, labels):
        return output.loss

    # Find LR
    lrs, losses = lr_finder(
        wrapper,
        dataloader,
        criterion,
        min_lr=1e-6,
        max_lr=1.0,
        num_steps=50
    )

    plot_lr_finder(lrs, losses, 'demo_lr_finder.png')
    print(f"\nLR range: {lrs[0]:.2e} to {lrs[-1]:.2e}")


def demo_training_monitor():
    """Demonstrate comprehensive training monitor."""
    print("\n" + "=" * 60)
    print("Demo: Training Monitor")
    print("=" * 60)

    model = SimpleModel()
    monitor = TrainingMonitor(model, log_every=20, gradient_clip=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate training
    for step in range(100):
        # Create batch
        input_ids = torch.randint(0, 1000, (4, 32))
        labels = torch.randint(0, 1000, (4, 32))

        # Forward
        output = model(input_ids, labels)
        loss = output.loss

        # Pre-backward check
        if not monitor.pre_backward_check(loss):
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Post-backward check
        post_check = monitor.post_backward_check()

        # Optimizer step
        optimizer.step()

        # Log
        monitor.log_step(loss.item(), 1e-4)

    # Get summary
    summary = monitor.get_summary()
    print(f"\nTraining Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final loss: {summary['loss_summary']['final_loss']:.4f}")
    print(f"  Clip fraction: {summary['clip_stats']['clip_fraction']:.2%}")


if __name__ == "__main__":
    print("Training Dynamics & Debugging - Hands-on Implementation")
    print("=" * 60)

    demo_loss_tracking()
    demo_gradient_monitoring()
    demo_overfit_batch()
    demo_lr_finder()
    demo_training_monitor()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
