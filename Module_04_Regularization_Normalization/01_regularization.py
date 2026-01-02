"""
Module 4.1: Regularization Techniques
Complete implementation of regularization methods for deep learning.

This module covers:
1. L1 and L2 regularization
2. Dropout variants
3. Label smoothing
4. DropPath / Stochastic Depth
5. Practical implementation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

print("=" * 70)
print("MODULE 4.1: REGULARIZATION TECHNIQUES")
print("=" * 70)

# =============================================================================
# SECTION 1: L1 AND L2 REGULARIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: L1 AND L2 REGULARIZATION")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Computing Regularization Terms
# -----------------------------------------------------------------------------
print("\n1.1 L1 and L2 Regularization")
print("-" * 40)


def l1_regularization(model: nn.Module, lambda_l1: float = 0.01) -> torch.Tensor:
    """
    Compute L1 regularization term.

    L1 = λ × Σ|w|

    Promotes sparsity (exact zeros).
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm


def l2_regularization(model: nn.Module, lambda_l2: float = 0.01) -> torch.Tensor:
    """
    Compute L2 regularization term.

    L2 = λ × Σw²

    Promotes small weights (but not zeros).
    """
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return lambda_l2 * l2_norm


def elastic_net(model: nn.Module, lambda_l1: float = 0.01, lambda_l2: float = 0.01) -> torch.Tensor:
    """
    Elastic Net: Combination of L1 and L2.

    Elastic = λ₁ × Σ|w| + λ₂ × Σw²
    """
    return l1_regularization(model, lambda_l1) + l2_regularization(model, lambda_l2)


# Demonstrate L1 vs L2 effect
print("Demonstrating L1 vs L2 regularization effect:")

# Create simple model
model_l1 = nn.Linear(10, 1)
model_l2 = nn.Linear(10, 1)

# Copy initial weights
with torch.no_grad():
    initial_weights = torch.randn(1, 10)
    model_l1.weight.copy_(initial_weights)
    model_l2.weight.copy_(initial_weights)
    model_l1.bias.zero_()
    model_l2.bias.zero_()

# Dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# Train with L1
optimizer_l1 = torch.optim.SGD(model_l1.parameters(), lr=0.01)
for _ in range(1000):
    pred = model_l1(X)
    loss = F.mse_loss(pred, y) + l1_regularization(model_l1, 0.1)
    optimizer_l1.zero_grad()
    loss.backward()
    optimizer_l1.step()

# Train with L2
optimizer_l2 = torch.optim.SGD(model_l2.parameters(), lr=0.01)
for _ in range(1000):
    pred = model_l2(X)
    loss = F.mse_loss(pred, y) + l2_regularization(model_l2, 0.1)
    optimizer_l2.zero_grad()
    loss.backward()
    optimizer_l2.step()

# Compare results
w_l1 = model_l1.weight.detach()
w_l2 = model_l2.weight.detach()

print(f"\nL1 weights: {[f'{w:.4f}' for w in w_l1.flatten().tolist()]}")
print(f"L2 weights: {[f'{w:.4f}' for w in w_l2.flatten().tolist()]}")
print(f"\nL1 exact zeros (|w| < 0.01): {(w_l1.abs() < 0.01).sum().item()}")
print(f"L2 exact zeros (|w| < 0.01): {(w_l2.abs() < 0.01).sum().item()}")
print("\nL1 produces more exact zeros (sparsity)!")


# =============================================================================
# SECTION 2: DROPOUT
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: DROPOUT")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 Standard Dropout
# -----------------------------------------------------------------------------
print("\n2.1 Standard Dropout Implementation")
print("-" * 40)


class Dropout(nn.Module):
    """
    Standard dropout layer.

    During training: Zero out random elements with probability p.
    During inference: No changes (using inverted dropout scaling).

    Inverted dropout scales by 1/(1-p) during training so inference
    doesn't need to scale.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        # Create binary mask
        keep_prob = 1 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))

        # Apply mask and scale (inverted dropout)
        return x * mask / keep_prob


# Demonstrate dropout
print("Dropout behavior (p=0.5):")
dropout = Dropout(p=0.5)
x = torch.ones(1, 10)

print(f"Input: {x.tolist()}")

dropout.train()
print("Training mode (5 samples):")
for i in range(5):
    out = dropout(x)
    print(f"  Sample {i+1}: {[f'{v:.1f}' for v in out.flatten().tolist()]}")

dropout.eval()
print(f"Eval mode: {dropout(x).tolist()}")


# -----------------------------------------------------------------------------
# 2.2 Dropout Functional
# -----------------------------------------------------------------------------
print("\n\n2.2 Dropout as Function")
print("-" * 40)


def dropout_functional(
    x: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False
) -> torch.Tensor:
    """
    Functional dropout implementation.

    Same as F.dropout but showing internals.
    """
    if not training or p == 0:
        return x

    keep_prob = 1 - p

    if inplace:
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        x.mul_(mask).div_(keep_prob)
        return x
    else:
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob


# Verify against PyTorch
torch.manual_seed(42)
x = torch.randn(3, 4)
out_custom = dropout_functional(x.clone(), p=0.3, training=True)

torch.manual_seed(42)
out_pytorch = F.dropout(x.clone(), p=0.3, training=True)

# Note: Won't match exactly due to different random state handling
print(f"Custom dropout shape: {out_custom.shape}")
print(f"PyTorch dropout shape: {out_pytorch.shape}")


# -----------------------------------------------------------------------------
# 2.3 Spatial Dropout (for CNNs)
# -----------------------------------------------------------------------------
print("\n\n2.3 Spatial Dropout (for CNNs)")
print("-" * 40)


class SpatialDropout2d(nn.Module):
    """
    Spatial dropout for 2D feature maps.

    Drops entire channels instead of individual elements.
    Better for CNNs where spatial correlation is high.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        """
        if not self.training or self.p == 0:
            return x

        batch, channels, height, width = x.shape
        keep_prob = 1 - self.p

        # Create mask for each channel (shared across spatial dimensions)
        mask = torch.bernoulli(torch.full((batch, channels, 1, 1), keep_prob, device=x.device))

        # Expand mask to match input shape and apply
        return x * mask / keep_prob


# Demonstrate spatial dropout
spatial_dropout = SpatialDropout2d(p=0.5)
x = torch.ones(1, 4, 3, 3)  # (batch=1, channels=4, h=3, w=3)

print(f"Input shape: {x.shape}")
spatial_dropout.train()
out = spatial_dropout(x)
print(f"Output channel sums: {out.sum(dim=(2, 3)).flatten().tolist()}")
print("Entire channels are dropped (0) or kept (scaled by 2)")


# -----------------------------------------------------------------------------
# 2.4 DropPath / Stochastic Depth
# -----------------------------------------------------------------------------
print("\n\n2.4 DropPath / Stochastic Depth")
print("-" * 40)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) for residual networks.

    Randomly drops entire residual blocks during training.
    Used in Vision Transformers and modern ResNets.

    During training: Randomly skip block
    During inference: Scale by survival probability
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch, 1, 1, ...)

        # Random tensor for each sample in batch
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()  # 1 or 0

        # Scale output
        return x * binary_tensor / keep_prob


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Functional version of DropPath."""
    if drop_prob == 0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_tensor = random_tensor.floor()

    return x * binary_tensor / keep_prob


# Usage in a residual block
class ResidualBlockWithDropPath(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)
        return x + self.drop_path(residual)  # Drop path on residual branch


# Demonstrate
block = ResidualBlockWithDropPath(64, drop_path_rate=0.2)
block.train()
x = torch.randn(4, 64)

print("DropPath in residual block:")
print(f"Input norm: {x.norm().item():.4f}")
out = block(x)
print(f"Output norm: {out.norm().item():.4f}")


# =============================================================================
# SECTION 3: LABEL SMOOTHING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: LABEL SMOOTHING")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Label Smoothing Implementation
# -----------------------------------------------------------------------------
print("\n3.1 Label Smoothing")
print("-" * 40)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    y_smooth = (1 - ε) × y_onehot + ε / K

    where:
        ε = smoothing factor
        K = number of classes
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,) class indices
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / num_classes)
            smooth_targets.scatter_(
                1, targets.unsqueeze(1), 1 - self.smoothing + self.smoothing / num_classes
            )

        # Cross-entropy with soft targets
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Alternative implementation using KL divergence
class LabelSmoothingKL(nn.Module):
    """
    Label smoothing using KL divergence formulation.

    Equivalent to above but different computation path.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Two-part loss: confident part + smoothing part
        nll_loss = F.nll_loss(log_probs, targets, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Demonstrate label smoothing
print("Label Smoothing Demonstration:")
logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # Very confident prediction
targets = torch.tensor([0])  # Correct class

ce_standard = F.cross_entropy(logits, targets)
ce_smoothed = LabelSmoothingCrossEntropy(smoothing=0.1)(logits, targets)

print(f"Logits: {logits.tolist()} (very confident)")
print(f"Standard CE loss: {ce_standard.item():.6f}")
print(f"Label smoothed CE (ε=0.1): {ce_smoothed.item():.6f}")
print("Label smoothing increases loss for overconfident predictions!")

# Show smoothed labels
num_classes = 4
smooth_label = torch.zeros(num_classes)
smooth_label.fill_(0.1 / num_classes)
smooth_label[0] = 1 - 0.1 + 0.1 / num_classes
print(f"\nHard label:     [1.0, 0.0, 0.0, 0.0]")
print(f"Smoothed label: {[f'{v:.3f}' for v in smooth_label.tolist()]}")


# =============================================================================
# SECTION 4: REGULARIZATION IN TRANSFORMERS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: REGULARIZATION IN TRANSFORMERS")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Transformer Block with Regularization
# -----------------------------------------------------------------------------
print("\n4.1 Transformer Block with All Regularization")
print("-" * 40)


class TransformerBlockWithRegularization(nn.Module):
    """
    Transformer block showing all regularization positions.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Attention components
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # FFN components
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        # Regularization
        self.attn_dropout = nn.Dropout(attention_dropout)  # On attention weights
        self.proj_dropout = nn.Dropout(dropout)  # After projections
        self.ffn_dropout = nn.Dropout(dropout)  # After FFN
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        head_dim = self.d_model // self.n_heads

        # Project
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # DROPOUT 1: On attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        # DROPOUT 2: After output projection
        output = self.proj_dropout(output)

        return output

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # DROPOUT 3: After FFN
        x = self.ffn_dropout(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with drop path
        attn_out = self.attention(self.ln1(x))
        x = x + self.drop_path(attn_out)  # DROP PATH on residual branch

        # Pre-norm FFN with drop path
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.drop_path(ffn_out)  # DROP PATH on residual branch

        return x


# Show regularization positions
print("Regularization positions in transformer block:")
print("1. Attention dropout: After softmax, before V multiplication")
print("2. Projection dropout: After attention output projection")
print("3. FFN dropout: After FFN computation")
print("4. Drop path: On residual branches (before addition)")

block = TransformerBlockWithRegularization(
    d_model=256, n_heads=4, d_ff=512,
    dropout=0.1, attention_dropout=0.0, drop_path_rate=0.1
)
print(f"\nBlock parameters: {sum(p.numel() for p in block.parameters()):,}")


# -----------------------------------------------------------------------------
# 4.2 Stochastic Depth Schedule
# -----------------------------------------------------------------------------
print("\n\n4.2 Stochastic Depth Schedule")
print("-" * 40)


def get_drop_path_rates(num_layers: int, max_rate: float = 0.1) -> List[float]:
    """
    Linear increase of drop path rate.

    Earlier layers: Lower drop rate (more important)
    Later layers: Higher drop rate
    """
    return [max_rate * i / (num_layers - 1) for i in range(num_layers)]


# Demonstrate
num_layers = 12
rates = get_drop_path_rates(num_layers, max_rate=0.1)
print(f"Drop path rates for {num_layers} layers (max=0.1):")
for i, rate in enumerate(rates):
    print(f"  Layer {i:2d}: {rate:.4f}")


# =============================================================================
# SECTION 5: WEIGHT DECAY IMPLEMENTATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: PROPER WEIGHT DECAY")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Separating Parameters for Weight Decay
# -----------------------------------------------------------------------------
print("\n5.1 Parameter Groups for Weight Decay")
print("-" * 40)


def create_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_keywords: List[str] = ['bias', 'LayerNorm', 'layer_norm', 'layernorm']
) -> List[dict]:
    """
    Separate parameters into groups with and without weight decay.

    No weight decay for:
    - Bias terms (1D tensors)
    - LayerNorm parameters
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if any no_decay keyword is in the name
        if any(kw.lower() in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        elif param.ndim == 1:
            # Also exclude 1D params (typically biases)
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


# Demonstrate
class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)  # weight + bias
        self.ln = nn.LayerNorm(64)  # weight + bias
        self.linear2 = nn.Linear(64, 10)

model = SampleModel()
param_groups = create_param_groups(model, weight_decay=0.1)

print("Parameter groups:")
for i, group in enumerate(param_groups):
    n_params = sum(p.numel() for p in group['params'])
    print(f"  Group {i}: {n_params:,} parameters, weight_decay={group['weight_decay']}")


# =============================================================================
# SECTION 6: COMPLETE REGULARIZATION EXAMPLE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: COMPLETE TRAINING WITH REGULARIZATION")
print("=" * 70)


class RegularizedTrainer:
    """
    Trainer class demonstrating all regularization techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0
    ):
        self.model = model
        self.gradient_clip = gradient_clip

        # Create parameter groups
        param_groups = create_param_groups(model, weight_decay)

        # AdamW optimizer (decoupled weight decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95)
        )

        # Label smoothing loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()

        # Forward
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        # Update
        self.optimizer.step()

        return loss.item()


print("""
Complete regularization setup:
1. ✓ Weight decay (AdamW, 0.1)
2. ✓ Exclude biases and LayerNorm from weight decay
3. ✓ Dropout in model (attention, FFN, residual)
4. ✓ Drop path (stochastic depth)
5. ✓ Label smoothing (ε=0.1)
6. ✓ Gradient clipping (max_norm=1.0)
""")


# =============================================================================
# SECTION 7: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. L1 vs L2 Regularization:
   - L1: Sparsity (exact zeros), |w|
   - L2: Small weights (not zeros), w²
   - L2 preferred in deep learning (smooth)

2. Weight Decay vs L2:
   - Equivalent for SGD
   - Different for Adam (use AdamW!)
   - Exclude biases and LayerNorm

3. Dropout:
   - Inverted dropout: scale by 1/p during training
   - Ensemble interpretation
   - Apply BEFORE residual addition

4. Label Smoothing:
   - y_smooth = (1-ε)y + ε/K
   - Prevents overconfidence
   - Standard ε = 0.1

5. Drop Path:
   - Drop entire residual branches
   - Linear schedule: 0 to max_rate
   - Used in ViT, modern networks

6. Regularization in LLMs:
   - Weight decay: 0.1
   - Dropout: 0.0-0.1
   - Label smoothing: 0.1
   - Gradient clipping: 1.0
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 4.1 SUMMARY: REGULARIZATION")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────────────┬─────────────────────────┐
│ Technique       │ Formula/Rate                   │ Effect                  │
├─────────────────┼────────────────────────────────┼─────────────────────────┤
│ L2/Weight Decay │ λ×Σw² (λ=0.1)                 │ Small weights           │
│ Dropout         │ p=0.1-0.5                      │ Random zeros            │
│ Drop Path       │ Linear 0→0.1                   │ Skip residual           │
│ Label Smooth    │ ε=0.1                          │ Soft targets            │
│ Grad Clip       │ max_norm=1.0                   │ Stable training         │
└─────────────────┴────────────────────────────────┴─────────────────────────┘

LLM REGULARIZATION CONFIG:

  Weight Decay:
    value = 0.1
    exclude = ['bias', 'LayerNorm']

  Dropout:
    attention = 0.0-0.1
    ffn = 0.0-0.1

  Label Smoothing:
    epsilon = 0.1

  Gradient Clipping:
    max_norm = 1.0

KEY TAKEAWAYS:
1. Use AdamW (decoupled weight decay)
2. Exclude biases and norms from weight decay
3. Apply dropout BEFORE residual addition
4. Label smoothing standard in transformers
5. Drop path increases linearly with depth
""")

print("\nModule 4.1 Complete!")
