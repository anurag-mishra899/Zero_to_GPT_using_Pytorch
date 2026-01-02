"""
Module 2.3: Loss Functions for Deep Learning and LLMs
Complete implementation of loss functions from basics to LLM-specific losses.

This module covers:
1. Regression losses (MSE, MAE, Huber)
2. Classification losses (BCE, CE, Focal Loss)
3. Language model losses (Causal LM, Masked LM)
4. Contrastive losses (InfoNCE, Triplet)
5. Advanced LLM losses (DPO, Knowledge Distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

print("=" * 70)
print("MODULE 2.3: LOSS FUNCTIONS FOR DEEP LEARNING AND LLMs")
print("=" * 70)

# =============================================================================
# SECTION 1: REGRESSION LOSSES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: REGRESSION LOSSES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1.1 Mean Squared Error (MSE / L2 Loss)
# -----------------------------------------------------------------------------
print("\n1.1 Mean Squared Error (MSE)")
print("-" * 40)

class MSELoss:
    """
    Mean Squared Error: L = (1/n) Σ (y - ŷ)²

    Properties:
    - Penalizes large errors heavily (squared)
    - Sensitive to outliers
    - Gradient: 2(ŷ - y) / n
    """

    @staticmethod
    def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ((predictions - targets) ** 2).mean()

    @staticmethod
    def forward_with_reduction(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        loss = (predictions - targets) ** 2
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    @staticmethod
    def backward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Gradient w.r.t. predictions"""
        n = predictions.numel()
        return 2 * (predictions - targets) / n


# Demonstrate MSE
predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets = torch.tensor([3.0, -0.5, 2.0, 7.0])

mse_custom = MSELoss.forward(predictions, targets)
mse_pytorch = F.mse_loss(predictions, targets)

print(f"Predictions: {predictions.tolist()}")
print(f"Targets: {targets.tolist()}")
print(f"Per-element squared error: {((predictions - targets) ** 2).tolist()}")
print(f"MSE (custom): {mse_custom.item():.4f}")
print(f"MSE (PyTorch): {mse_pytorch.item():.4f}")


# -----------------------------------------------------------------------------
# 1.2 Mean Absolute Error (MAE / L1 Loss)
# -----------------------------------------------------------------------------
print("\n\n1.2 Mean Absolute Error (MAE)")
print("-" * 40)

class MAELoss:
    """
    Mean Absolute Error: L = (1/n) Σ |y - ŷ|

    Properties:
    - Linear penalty
    - Robust to outliers
    - Gradient: sign(ŷ - y) / n
    """

    @staticmethod
    def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (predictions - targets).abs().mean()

    @staticmethod
    def backward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Gradient w.r.t. predictions"""
        n = predictions.numel()
        return torch.sign(predictions - targets) / n


mae_custom = MAELoss.forward(predictions, targets)
mae_pytorch = F.l1_loss(predictions, targets)

print(f"Per-element absolute error: {(predictions - targets).abs().tolist()}")
print(f"MAE (custom): {mae_custom.item():.4f}")
print(f"MAE (PyTorch): {mae_pytorch.item():.4f}")


# Outlier sensitivity comparison
print("\nOutlier Sensitivity:")
predictions_outlier = torch.tensor([1.0, 1.0, 1.0, 100.0])  # One outlier
targets_normal = torch.tensor([1.0, 1.0, 1.0, 1.0])

mse_outlier = F.mse_loss(predictions_outlier, targets_normal)
mae_outlier = F.l1_loss(predictions_outlier, targets_normal)

print(f"With outlier (100 instead of 1):")
print(f"  MSE: {mse_outlier.item():.2f} (heavily penalized)")
print(f"  MAE: {mae_outlier.item():.2f} (moderate penalty)")


# -----------------------------------------------------------------------------
# 1.3 Huber Loss (Smooth L1)
# -----------------------------------------------------------------------------
print("\n\n1.3 Huber Loss")
print("-" * 40)

class HuberLoss:
    """
    Huber Loss: Combines MSE and MAE

    L = 0.5(y - ŷ)²           if |y - ŷ| ≤ δ
        δ|y - ŷ| - 0.5δ²      otherwise

    Properties:
    - Quadratic for small errors (smooth)
    - Linear for large errors (robust to outliers)
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = predictions - targets
        abs_diff = diff.abs()

        quadratic = 0.5 * diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2

        return torch.where(abs_diff <= self.delta, quadratic, linear).mean()


# Compare all three losses with varying errors
print("Loss comparison for different error magnitudes:")
print(f"{'Error':>10} {'MSE':>10} {'MAE':>10} {'Huber(δ=1)':>12}")
print("-" * 45)

errors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
huber = HuberLoss(delta=1.0)

for error in errors:
    pred = torch.tensor([error])
    target = torch.tensor([0.0])

    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    hub = huber.forward(pred, target).item()

    print(f"{error:>10.1f} {mse:>10.4f} {mae:>10.4f} {hub:>12.4f}")


# =============================================================================
# SECTION 2: CLASSIFICATION LOSSES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: CLASSIFICATION LOSSES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 2.1 Binary Cross-Entropy
# -----------------------------------------------------------------------------
print("\n2.1 Binary Cross-Entropy (BCE)")
print("-" * 40)

class BCELoss:
    """
    Binary Cross-Entropy: L = -[y log(ŷ) + (1-y) log(1-ŷ)]

    Properties:
    - For binary classification
    - Measures divergence between distributions
    - ŷ should be probability (0, 1)
    """

    @staticmethod
    def forward(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Args:
            predictions: Probabilities in (0, 1)
            targets: Binary labels {0, 1}
        """
        # Clip to prevent log(0)
        predictions = predictions.clamp(eps, 1 - eps)
        return -(targets * predictions.log() + (1 - targets) * (1 - predictions).log()).mean()

    @staticmethod
    def backward(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Gradient w.r.t. predictions"""
        predictions = predictions.clamp(eps, 1 - eps)
        return (-targets / predictions + (1 - targets) / (1 - predictions)) / predictions.numel()


# Demonstrate BCE
probs = torch.tensor([0.9, 0.1, 0.8, 0.3])
labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

bce_custom = BCELoss.forward(probs, labels)
bce_pytorch = F.binary_cross_entropy(probs, labels)

print(f"Predictions (probs): {probs.tolist()}")
print(f"Labels: {labels.tolist()}")
print(f"BCE (custom): {bce_custom.item():.4f}")
print(f"BCE (PyTorch): {bce_pytorch.item():.4f}")

# Show loss for different confidence levels
print("\nLoss for different predictions (true label = 1):")
for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    loss = -math.log(p)
    print(f"  P(y=1) = {p:.2f} → BCE = {loss:.4f}")


# -----------------------------------------------------------------------------
# 2.2 BCE with Logits (Numerically Stable)
# -----------------------------------------------------------------------------
print("\n\n2.2 BCE with Logits (Numerically Stable)")
print("-" * 40)

class BCEWithLogitsLoss:
    """
    BCE with Logits: Numerically stable BCE for raw logits.

    L = max(z, 0) - z*y + log(1 + e^(-|z|))

    This is equivalent to:
        -[y * log(σ(z)) + (1-y) * log(1-σ(z))]
    but numerically stable.
    """

    @staticmethod
    def forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Stable formula
        max_val = torch.clamp(-logits, min=0)
        loss = logits - logits * targets + max_val + torch.log(
            torch.exp(-max_val) + torch.exp(-logits - max_val)
        )
        return loss.mean()


# Demonstrate stability
logits = torch.tensor([10.0, -10.0, 0.0, 100.0, -100.0])  # Extreme values
labels = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0])

bce_logits_custom = BCEWithLogitsLoss.forward(logits, labels)
bce_logits_pytorch = F.binary_cross_entropy_with_logits(logits, labels)

print(f"Logits (including extreme): {logits.tolist()}")
print(f"Labels: {labels.tolist()}")
print(f"BCE with Logits (custom): {bce_logits_custom.item():.4f}")
print(f"BCE with Logits (PyTorch): {bce_logits_pytorch.item():.4f}")

# Show why this matters for extreme values
print("\nNumerical stability demonstration:")
extreme_logit = torch.tensor([100.0])
try:
    # Unstable: sigmoid then BCE
    prob = torch.sigmoid(extreme_logit)
    unstable_loss = F.binary_cross_entropy(prob, torch.tensor([1.0]))
    print(f"  Unstable BCE: {unstable_loss.item():.6f}")
except:
    print("  Unstable BCE: FAILED")

# Stable: BCE with logits
stable_loss = F.binary_cross_entropy_with_logits(extreme_logit, torch.tensor([1.0]))
print(f"  Stable BCE with logits: {stable_loss.item():.6f}")


# -----------------------------------------------------------------------------
# 2.3 Categorical Cross-Entropy
# -----------------------------------------------------------------------------
print("\n\n2.3 Categorical Cross-Entropy")
print("-" * 40)

class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification.

    L = -log(softmax(z)_target) = -z_target + log(Σ exp(z))

    Gradient: ∂L/∂z_i = softmax(z)_i - 1_{i=target}
    """

    @staticmethod
    def forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw scores [batch, num_classes]
            targets: Class indices [batch]
        """
        batch_size = logits.size(0)

        # Log-softmax (stable)
        log_softmax = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Gather log probabilities for target classes
        target_log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        return -target_log_probs.mean()

    @staticmethod
    def forward_detailed(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both loss and per-class gradients for understanding."""
        batch_size, num_classes = logits.shape

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, targets)

        # Gradient: softmax - one_hot_target
        one_hot = F.one_hot(targets, num_classes).float()
        gradient = probs - one_hot

        return loss, gradient


# Demonstrate CE
logits = torch.tensor([
    [2.0, 1.0, 0.1],   # Should predict class 0
    [0.1, 2.5, 0.3],   # Should predict class 1
    [0.2, 0.1, 3.0],   # Should predict class 2
])
targets = torch.tensor([0, 1, 2])

ce_custom = CrossEntropyLoss.forward(logits, targets)
ce_pytorch = F.cross_entropy(logits, targets)
loss_detailed, gradients = CrossEntropyLoss.forward_detailed(logits, targets)

print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets.tolist()}")
print(f"Softmax probabilities:\n{F.softmax(logits, dim=-1).numpy()}")
print(f"\nCross-Entropy (custom): {ce_custom.item():.4f}")
print(f"Cross-Entropy (PyTorch): {ce_pytorch.item():.4f}")
print(f"\nGradients w.r.t. logits (softmax - one_hot):")
print(f"{gradients.numpy()}")


# -----------------------------------------------------------------------------
# 2.4 Label Smoothing
# -----------------------------------------------------------------------------
print("\n\n2.4 Cross-Entropy with Label Smoothing")
print("-" * 40)

class LabelSmoothingCrossEntropy:
    """
    Cross-Entropy with Label Smoothing.

    y_smooth = (1 - ε)y + ε/K

    where:
        ε = smoothing factor (e.g., 0.1)
        K = number of classes
        y = original one-hot label
    """

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing + self.smoothing / num_classes)

        # Cross-entropy with soft targets
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()

        return loss


# Compare with and without label smoothing
logits = torch.tensor([[10.0, 0.0, 0.0]])  # Very confident prediction
targets = torch.tensor([0])

ce_standard = F.cross_entropy(logits, targets)
ce_smoothed = LabelSmoothingCrossEntropy(smoothing=0.1).forward(logits, targets)

print(f"Logits: {logits.tolist()} (very confident)")
print(f"Target: class 0")
print(f"\nStandard CE: {ce_standard.item():.6f}")
print(f"Label Smoothed CE (ε=0.1): {ce_smoothed.item():.6f}")
print("\nLabel smoothing increases loss for overconfident predictions!")

# Show smoothed labels
num_classes = 3
smoothing = 0.1
original = torch.tensor([0, 0, 1, 0])  # One-hot for class 2
smoothed = (1 - smoothing) * original.float() + smoothing / num_classes
print(f"\nOriginal one-hot (class 2): {original.tolist()}")
print(f"Smoothed (ε=0.1): {[f'{x:.3f}' for x in smoothed[:4].tolist()]}")


# -----------------------------------------------------------------------------
# 2.5 Focal Loss
# -----------------------------------------------------------------------------
print("\n\n2.5 Focal Loss (For Class Imbalance)")
print("-" * 40)

class FocalLoss:
    """
    Focal Loss: FL(p) = -α_t * (1 - p_t)^γ * log(p_t)

    Properties:
    - Down-weights easy examples
    - Focuses training on hard examples
    - α handles class imbalance
    - γ controls focusing (γ=0 → standard CE)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Get probability of correct class
        p_t = torch.exp(-ce_loss)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        return loss.mean()


# Compare CE and Focal Loss
print("Comparison: CE vs Focal Loss (γ=2)")
print(f"{'Correct Prob':>15} {'CE Loss':>12} {'Focal Loss':>12} {'Ratio':>10}")
print("-" * 55)

focal = FocalLoss(gamma=2.0)
for p_correct in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    # Create logits that give this probability
    logit_correct = math.log(p_correct / (1 - p_correct + 1e-8))
    logits = torch.tensor([[logit_correct, 0.0]])
    targets = torch.tensor([0])

    ce = F.cross_entropy(logits, targets).item()
    fl = focal.forward(logits, targets).item()
    ratio = fl / (ce + 1e-8)

    print(f"{p_correct:>15.2f} {ce:>12.4f} {fl:>12.4f} {ratio:>10.2%}")

print("\nFocal loss down-weights easy examples (high p_correct)!")


# =============================================================================
# SECTION 3: LANGUAGE MODEL LOSSES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: LANGUAGE MODEL LOSSES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3.1 Causal Language Modeling Loss
# -----------------------------------------------------------------------------
print("\n3.1 Causal Language Modeling Loss (GPT-style)")
print("-" * 40)

class CausalLMLoss:
    """
    Causal Language Modeling Loss for next-token prediction.

    L = -Σ_t log P(x_t | x_1, ..., x_{t-1})

    Implementation notes:
    - Shift logits and labels for next-token prediction
    - Ignore padding tokens (typically -100)
    """

    @staticmethod
    def forward(
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        labels: torch.Tensor,  # [batch, seq_len]
        ignore_index: int = -100
    ) -> torch.Tensor:
        # Shift for next-token prediction
        # Logits at position t predict token at position t+1
        shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq-1, vocab]
        shift_labels = labels[..., 1:].contiguous()       # [batch, seq-1]

        # Flatten
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Cross-entropy (ignores -100)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)

        return loss

    @staticmethod
    def forward_detailed(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns loss and per-token losses for analysis."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        # Per-token loss
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction='none'
        ).view(shift_labels.shape)

        # Average loss
        valid_tokens = (shift_labels != ignore_index).float()
        loss = (per_token_loss * valid_tokens).sum() / valid_tokens.sum()

        return loss, per_token_loss


# Demonstrate Causal LM Loss
batch_size = 2
seq_len = 10
vocab_size = 100

# Simulate logits and labels
torch.manual_seed(42)
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

# Add some padding
labels[0, 7:] = -100  # Pad last 3 tokens of first sequence
labels[1, 8:] = -100  # Pad last 2 tokens of second sequence

loss = CausalLMLoss.forward(logits, labels)
loss_detailed, per_token_loss = CausalLMLoss.forward_detailed(logits, labels)

print(f"Batch size: {batch_size}, Seq length: {seq_len}, Vocab size: {vocab_size}")
print(f"Labels shape: {labels.shape}")
print(f"Labels[0]: {labels[0].tolist()}")
print(f"Labels[1]: {labels[1].tolist()}")
print(f"\nCausal LM Loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")

print(f"\nPer-token losses (first sequence):")
print(f"  {[f'{x:.2f}' if x > 0 else 'PAD' for x in per_token_loss[0].tolist()]}")


# -----------------------------------------------------------------------------
# 3.2 Masked Language Modeling Loss
# -----------------------------------------------------------------------------
print("\n\n3.2 Masked Language Modeling Loss (BERT-style)")
print("-" * 40)

class MaskedLMLoss:
    """
    Masked Language Modeling Loss.

    L = -(1/|M|) Σ_{i∈M} log P(x_i | x_masked)

    Only compute loss on masked positions.
    """

    @staticmethod
    def create_masks(
        input_ids: torch.Tensor,
        vocab_size: int,
        mask_token_id: int,
        mask_prob: float = 0.15,
        random_replace_prob: float = 0.1,
        keep_prob: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create BERT-style masks.

        Returns:
            masked_ids: Input with [MASK] tokens
            labels: Original tokens at masked positions, -100 elsewhere
            mask_positions: Boolean mask of masked positions
        """
        # Decide which positions to mask
        mask_positions = torch.rand_like(input_ids.float()) < mask_prob

        # Create labels (-100 for non-masked positions)
        labels = torch.where(mask_positions, input_ids, torch.full_like(input_ids, -100))

        # Apply masking strategy
        rand = torch.rand_like(input_ids.float())
        masked_ids = input_ids.clone()

        # 80%: Replace with [MASK]
        mask_with_mask = mask_positions & (rand < 0.8)
        masked_ids[mask_with_mask] = mask_token_id

        # 10%: Replace with random token
        mask_with_random = mask_positions & (rand >= 0.8) & (rand < 0.9)
        random_tokens = torch.randint(0, vocab_size, masked_ids.shape)
        masked_ids[mask_with_random] = random_tokens[mask_with_random]

        # 10%: Keep original (already done)

        return masked_ids, labels, mask_positions

    @staticmethod
    def forward(
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        labels: torch.Tensor,  # [batch, seq_len] with -100 for non-masked
        ignore_index: int = -100
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)

        # Cross-entropy only on masked positions
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=ignore_index
        )

        return loss


# Demonstrate MLM
seq_len = 20
vocab_size = 1000
mask_token_id = 103  # [MASK] token

input_ids = torch.randint(0, vocab_size, (1, seq_len))
masked_ids, labels, mask_positions = MaskedLMLoss.create_masks(
    input_ids, vocab_size, mask_token_id
)

print(f"Original: {input_ids[0, :10].tolist()}...")
print(f"Masked:   {masked_ids[0, :10].tolist()}...")
print(f"Labels:   {labels[0, :10].tolist()}...")
print(f"\nMask positions: {mask_positions[0].sum().item()} / {seq_len} tokens")
print(f"Mask ratio: {mask_positions.float().mean().item():.2%}")

# Simulate loss computation
logits = torch.randn(1, seq_len, vocab_size)
mlm_loss = MaskedLMLoss.forward(logits, labels)
print(f"\nMLM Loss: {mlm_loss.item():.4f}")


# -----------------------------------------------------------------------------
# 3.3 Perplexity Calculation
# -----------------------------------------------------------------------------
print("\n\n3.3 Perplexity Calculation")
print("-" * 40)

def calculate_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Calculate perplexity from logits and labels.

    Perplexity = exp(cross_entropy_loss)
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )

    return torch.exp(loss)


# Demonstrate perplexity
print("Perplexity interpretation:")
print(f"  Random model (vocab {vocab_size}): {vocab_size}")
print(f"  Good LM: 10-50")
print(f"  State-of-art: < 10")

# Simulate different model qualities
for quality, temp in [("Random", 10.0), ("Poor", 2.0), ("Good", 0.5), ("Excellent", 0.1)]:
    # Create logits that give different perplexities
    # Temperature controls sharpness of distribution
    torch.manual_seed(42)
    logits = torch.randn(1, 50, vocab_size) / temp
    # Make sure correct token has highest logit
    labels = torch.randint(0, vocab_size, (1, 50))
    for i in range(1, 50):
        logits[0, i-1, labels[0, i]] += 5 / temp

    ppl = calculate_perplexity(logits, labels)
    print(f"  {quality} model: perplexity = {ppl.item():.2f}")


# =============================================================================
# SECTION 4: CONTRASTIVE LOSSES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: CONTRASTIVE LOSSES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 4.1 Triplet Loss
# -----------------------------------------------------------------------------
print("\n4.1 Triplet Loss")
print("-" * 40)

class TripletLoss:
    """
    Triplet Loss: L = max(0, D(a, p) - D(a, n) + margin)

    Properties:
    - Pulls anchor-positive pairs together
    - Pushes anchor-negative pairs apart
    - Margin ensures minimum separation
    """

    def __init__(self, margin: float = 1.0, distance: str = 'euclidean'):
        self.margin = margin
        self.distance = distance

    def _compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.distance == 'euclidean':
            return (x1 - x2).pow(2).sum(dim=-1).sqrt()
        elif self.distance == 'cosine':
            return 1 - F.cosine_similarity(x1, x2, dim=-1)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def forward(
        self,
        anchor: torch.Tensor,    # [batch, embed_dim]
        positive: torch.Tensor,  # [batch, embed_dim]
        negative: torch.Tensor   # [batch, embed_dim]
    ) -> torch.Tensor:
        d_pos = self._compute_distance(anchor, positive)
        d_neg = self._compute_distance(anchor, negative)

        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


# Demonstrate triplet loss
embed_dim = 128
anchor = torch.randn(4, embed_dim)
positive = anchor + 0.1 * torch.randn(4, embed_dim)  # Close to anchor
negative = torch.randn(4, embed_dim)  # Random (far from anchor)

triplet = TripletLoss(margin=0.5)
loss = triplet.forward(anchor, positive, negative)

print(f"Embedding dimension: {embed_dim}")
print(f"Distance (anchor, positive): {triplet._compute_distance(anchor, positive).mean().item():.4f}")
print(f"Distance (anchor, negative): {triplet._compute_distance(anchor, negative).mean().item():.4f}")
print(f"Triplet Loss (margin=0.5): {loss.item():.4f}")


# -----------------------------------------------------------------------------
# 4.2 InfoNCE / Contrastive Loss (CLIP, SimCLR)
# -----------------------------------------------------------------------------
print("\n\n4.2 InfoNCE Loss (CLIP, SimCLR)")
print("-" * 40)

class InfoNCELoss:
    """
    InfoNCE Loss: L = -log(exp(sim(z, z⁺)/τ) / Σ exp(sim(z, zⱼ)/τ))

    Used in:
    - CLIP (image-text matching)
    - SimCLR (self-supervised learning)
    - Sentence transformers

    Properties:
    - Treats all other samples in batch as negatives
    - Temperature τ controls hardness
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,  # [batch, embed_dim]
        embeddings_b: torch.Tensor   # [batch, embed_dim]
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss (symmetric, like CLIP).

        Positive pairs: (a_i, b_i) for each i
        Negative pairs: (a_i, b_j) for i != j
        """
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, dim=-1)
        embeddings_b = F.normalize(embeddings_b, dim=-1)

        # Compute similarity matrix
        # logits[i, j] = similarity between a_i and b_j
        logits = embeddings_a @ embeddings_b.T / self.temperature

        # Labels: diagonal is positive (i matches with i)
        batch_size = embeddings_a.size(0)
        labels = torch.arange(batch_size, device=embeddings_a.device)

        # Symmetric loss (both directions)
        loss_a = F.cross_entropy(logits, labels)  # a→b
        loss_b = F.cross_entropy(logits.T, labels)  # b→a

        return (loss_a + loss_b) / 2


# Demonstrate InfoNCE
batch_size = 8
embed_dim = 64

# Create paired embeddings (like image-text pairs)
embeddings_a = torch.randn(batch_size, embed_dim)
# Positive pairs: similar but not identical
embeddings_b = embeddings_a + 0.5 * torch.randn(batch_size, embed_dim)

infonce = InfoNCELoss(temperature=0.07)
loss = infonce.forward(embeddings_a, embeddings_b)

print(f"Batch size: {batch_size}, Embed dim: {embed_dim}")
print(f"InfoNCE Loss (τ=0.07): {loss.item():.4f}")

# Show temperature effect
print("\nTemperature effect:")
for temp in [0.01, 0.07, 0.5, 1.0, 2.0]:
    infonce_temp = InfoNCELoss(temperature=temp)
    loss_temp = infonce_temp.forward(embeddings_a, embeddings_b)
    print(f"  τ={temp}: loss = {loss_temp.item():.4f}")


# =============================================================================
# SECTION 5: ADVANCED LLM LOSSES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: ADVANCED LLM LOSSES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 5.1 Knowledge Distillation Loss
# -----------------------------------------------------------------------------
print("\n5.1 Knowledge Distillation Loss")
print("-" * 40)

class DistillationLoss:
    """
    Knowledge Distillation Loss.

    L = α * L_CE(student, hard_labels) + (1-α) * T² * KL(soft_student || soft_teacher)

    where:
        T = temperature (softens distributions)
        α = weighting factor
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,  # [batch, num_classes]
        teacher_logits: torch.Tensor,  # [batch, num_classes]
        labels: torch.Tensor           # [batch]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss: Combined distillation loss
            hard_loss: CE with true labels
            soft_loss: KL with teacher
        """
        # Hard loss: CE with true labels
        hard_loss = F.cross_entropy(student_logits, labels)

        # Soft loss: KL divergence with teacher
        # Soften both distributions with temperature
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence (multiply by T² to scale gradients appropriately)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss, hard_loss, soft_loss


# Demonstrate distillation
num_classes = 10
batch_size = 4

# Teacher: confident predictions
teacher_logits = torch.randn(batch_size, num_classes)
teacher_logits[range(batch_size), torch.randint(0, num_classes, (batch_size,))] += 5

# Student: less confident
student_logits = torch.randn(batch_size, num_classes)

# True labels
labels = teacher_logits.argmax(dim=-1)

distill = DistillationLoss(temperature=4.0, alpha=0.5)
total_loss, hard_loss, soft_loss = distill.forward(student_logits, teacher_logits, labels)

print(f"Knowledge Distillation (T=4, α=0.5):")
print(f"  Hard loss (CE with labels): {hard_loss.item():.4f}")
print(f"  Soft loss (KL with teacher): {soft_loss.item():.4f}")
print(f"  Total loss: {total_loss.item():.4f}")

# Show temperature effect on teacher distribution
print("\nTemperature effect on teacher distribution:")
example_logits = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1])
for T in [1, 2, 4, 10]:
    soft = F.softmax(example_logits / T, dim=0)
    print(f"  T={T}: {[f'{p:.3f}' for p in soft.tolist()]}")


# -----------------------------------------------------------------------------
# 5.2 DPO Loss (Direct Preference Optimization)
# -----------------------------------------------------------------------------
print("\n\n5.2 DPO Loss (Direct Preference Optimization)")
print("-" * 40)

class DPOLoss:
    """
    Direct Preference Optimization Loss.

    L_DPO = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

    where:
        y_w = preferred (winning) response
        y_l = dispreferred (losing) response
        π_θ = policy (model being trained)
        π_ref = reference policy (original model)
        β = strength parameter (typically 0.1-0.5)

    Key insight: No reward model needed, direct optimization from preferences.
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def forward(
        self,
        policy_chosen_logprobs: torch.Tensor,    # [batch] log P_θ(y_w|x)
        policy_rejected_logprobs: torch.Tensor,  # [batch] log P_θ(y_l|x)
        ref_chosen_logprobs: torch.Tensor,       # [batch] log P_ref(y_w|x)
        ref_rejected_logprobs: torch.Tensor      # [batch] log P_ref(y_l|x)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: DPO loss
            chosen_reward: Implicit reward for chosen response
            rejected_reward: Implicit reward for rejected response
        """
        # Log probability ratios
        pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
        ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs

        # DPO logits
        logits = self.beta * (pi_logratios - ref_logratios)

        # Binary cross-entropy (want to maximize difference for chosen)
        loss = -F.logsigmoid(logits).mean()

        # Implicit rewards (for monitoring)
        chosen_reward = self.beta * (policy_chosen_logprobs - ref_chosen_logprobs)
        rejected_reward = self.beta * (policy_rejected_logprobs - ref_rejected_logprobs)

        return loss, chosen_reward.mean(), rejected_reward.mean()


# Demonstrate DPO
batch_size = 4

# Simulate log probabilities
# Reference model: both responses have similar probability
ref_chosen = torch.tensor([-10.0, -12.0, -11.0, -10.5])
ref_rejected = torch.tensor([-11.0, -11.5, -10.5, -11.0])

# Policy model: should prefer chosen
policy_chosen = ref_chosen + 2.0  # Higher prob for chosen
policy_rejected = ref_rejected - 1.0  # Lower prob for rejected

dpo = DPOLoss(beta=0.1)
loss, chosen_reward, rejected_reward = dpo.forward(
    policy_chosen, policy_rejected, ref_chosen, ref_rejected
)

print(f"DPO Loss (β=0.1):")
print(f"  Loss: {loss.item():.4f}")
print(f"  Chosen reward: {chosen_reward.item():.4f}")
print(f"  Rejected reward: {rejected_reward.item():.4f}")
print(f"  Reward margin: {(chosen_reward - rejected_reward).item():.4f}")


# -----------------------------------------------------------------------------
# 5.3 SFT Loss (Supervised Fine-Tuning)
# -----------------------------------------------------------------------------
print("\n\n5.3 SFT Loss (Instruction Tuning)")
print("-" * 40)

class SFTLoss:
    """
    Supervised Fine-Tuning Loss for instruction following.

    L = -(1/|R|) Σ_{t∈R} log P(x_t | x_{<t}, instruction)

    Only compute loss on response tokens, not instruction/prompt.
    """

    @staticmethod
    def create_labels(
        input_ids: torch.Tensor,
        response_start_positions: torch.Tensor,
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Create labels that mask out instruction tokens.

        Args:
            input_ids: [batch, seq_len]
            response_start_positions: [batch] - where response begins
        """
        labels = input_ids.clone()

        for i, start_pos in enumerate(response_start_positions):
            labels[i, :start_pos] = ignore_index

        return labels

    @staticmethod
    def forward(
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        input_ids: torch.Tensor,  # [batch, seq_len]
        response_start_positions: torch.Tensor,  # [batch]
        ignore_index: int = -100
    ) -> torch.Tensor:
        # Create labels with instruction masked
        labels = SFTLoss.create_labels(input_ids, response_start_positions, ignore_index)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross-entropy (ignores instruction tokens)
        vocab_size = shift_logits.size(-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=ignore_index
        )

        return loss


# Demonstrate SFT
seq_len = 20
vocab_size = 100

# Simulate: [instruction tokens | response tokens]
# Instruction: tokens 0-9, Response: tokens 10-19
input_ids = torch.randint(0, vocab_size, (2, seq_len))
response_start = torch.tensor([10, 8])  # Different instruction lengths

labels = SFTLoss.create_labels(input_ids, response_start)

print(f"Input IDs[0]: {input_ids[0].tolist()}")
print(f"Labels[0]:    {labels[0].tolist()}")
print(f"\nInstruction tokens: masked with -100")
print(f"Response tokens: actual token IDs (loss computed only here)")

# Compute loss
logits = torch.randn(2, seq_len, vocab_size)
sft_loss = SFTLoss.forward(logits, input_ids, response_start)
print(f"\nSFT Loss: {sft_loss.item():.4f}")


# =============================================================================
# SECTION 6: NUMERICAL STABILITY TECHNIQUES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: NUMERICAL STABILITY TECHNIQUES")
print("=" * 70)

# -----------------------------------------------------------------------------
# 6.1 Log-Sum-Exp Trick
# -----------------------------------------------------------------------------
print("\n6.1 Log-Sum-Exp Trick")
print("-" * 40)

def logsumexp_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Naive (unstable) implementation."""
    return torch.log(torch.exp(x).sum(dim=dim))

def logsumexp_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable implementation."""
    max_x = x.max(dim=dim, keepdim=True)[0]
    return max_x.squeeze(dim) + torch.log(torch.exp(x - max_x).sum(dim=dim))


# Demonstrate
x = torch.tensor([1000.0, 1001.0, 1002.0])  # Large values

try:
    naive_result = logsumexp_naive(x)
    print(f"Naive result: {naive_result.item()}")
except:
    print("Naive implementation: OVERFLOW")

stable_result = logsumexp_stable(x)
pytorch_result = torch.logsumexp(x, dim=0)

print(f"Stable result: {stable_result.item():.4f}")
print(f"PyTorch result: {pytorch_result.item():.4f}")


# -----------------------------------------------------------------------------
# 6.2 Numerically Stable Cross-Entropy
# -----------------------------------------------------------------------------
print("\n\n6.2 Numerically Stable Cross-Entropy")
print("-" * 40)

def cross_entropy_unstable(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Unstable: compute softmax then take log."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    return F.nll_loss(log_probs, targets)

def cross_entropy_stable(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stable: compute log_softmax directly."""
    return F.cross_entropy(logits, targets)


# Demonstrate
logits = torch.tensor([[100.0, 0.0, 0.0],
                       [0.0, 100.0, 0.0]])  # Extreme logits
targets = torch.tensor([0, 1])

# Unstable might work here but issues with backprop
unstable = cross_entropy_unstable(logits, targets)
stable = cross_entropy_stable(logits, targets)

print(f"Extreme logits: {logits.tolist()}")
print(f"Unstable CE: {unstable.item():.6f}")
print(f"Stable CE: {stable.item():.6f}")

print("\nAlways use F.cross_entropy() for numerical stability!")


# =============================================================================
# SECTION 7: COMPLETE LOSS LIBRARY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: COMPLETE LOSS FUNCTION LIBRARY")
print("=" * 70)

class LossLibrary:
    """
    Comprehensive collection of loss functions.
    Use as reference for implementation.
    """

    # Regression
    @staticmethod
    def mse(pred, target): return F.mse_loss(pred, target)

    @staticmethod
    def mae(pred, target): return F.l1_loss(pred, target)

    @staticmethod
    def huber(pred, target, delta=1.0): return F.huber_loss(pred, target, delta=delta)

    # Binary Classification
    @staticmethod
    def bce(pred, target): return F.binary_cross_entropy(pred, target)

    @staticmethod
    def bce_logits(logits, target): return F.binary_cross_entropy_with_logits(logits, target)

    # Multi-class Classification
    @staticmethod
    def ce(logits, target): return F.cross_entropy(logits, target)

    @staticmethod
    def nll(log_probs, target): return F.nll_loss(log_probs, target)

    # Language Modeling
    @staticmethod
    def causal_lm(logits, labels, ignore_index=-100):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index
        )

    # Contrastive
    @staticmethod
    def triplet(anchor, pos, neg, margin=1.0):
        return F.triplet_margin_loss(anchor, pos, neg, margin=margin)

    @staticmethod
    def cosine_embedding(x1, x2, target, margin=0.0):
        return F.cosine_embedding_loss(x1, x2, target, margin=margin)

    # KL Divergence
    @staticmethod
    def kl_div(log_pred, target): return F.kl_div(log_pred, target, reduction='batchmean')


# Demonstrate library
print("\nLoss Library Examples:")
pred = torch.randn(4, 10)
target_cls = torch.randint(0, 10, (4,))
target_reg = torch.randn(4, 10)

print(f"MSE: {LossLibrary.mse(pred, target_reg).item():.4f}")
print(f"MAE: {LossLibrary.mae(pred, target_reg).item():.4f}")
print(f"CE: {LossLibrary.ce(pred, target_cls).item():.4f}")


# =============================================================================
# SECTION 8: INTERVIEW PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: INTERVIEW PREPARATION")
print("=" * 70)

print("""
KEY INTERVIEW TOPICS:

1. Why Cross-Entropy for Classification?
   - Matches softmax naturally
   - Clean gradient: softmax - one_hot
   - Probabilistic interpretation (MLE)
   - Convex optimization landscape

2. Label Smoothing:
   - Prevents overconfidence
   - Acts as regularization
   - y_smooth = (1-ε)y + ε/K
   - ε = 0.1 is standard in transformers

3. Perplexity:
   - PPL = exp(cross-entropy)
   - Lower is better
   - Average "branching factor"
   - Only compare same tokenizer!

4. Causal LM Loss:
   - Shift logits and labels
   - L = -Σ log P(x_t | x_{<t})
   - Ignore padding with -100

5. DPO vs RLHF:
   - DPO: Direct from preferences, no reward model
   - RLHF: Train reward model + RL
   - DPO simpler, more stable

6. Numerical Stability:
   - Use log_softmax, not log(softmax)
   - Log-sum-exp trick
   - Use *_with_logits versions
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODULE 2.3 SUMMARY: LOSS FUNCTIONS")
print("=" * 70)

print("""
QUICK REFERENCE:

┌─────────────────┬────────────────────────┬─────────────────────────┐
│ Loss            │ Formula                │ Use Case                │
├─────────────────┼────────────────────────┼─────────────────────────┤
│ MSE             │ Σ(y-ŷ)²               │ Regression              │
│ MAE             │ Σ|y-ŷ|                │ Robust regression       │
│ BCE             │ -[y log ŷ + ...]      │ Binary classification   │
│ CE              │ -Σ yᵢ log ŷᵢ          │ Multi-class             │
│ Causal LM       │ -Σₜ log P(xₜ|x<t)     │ GPT training            │
│ Masked LM       │ -Σᵢ∈M log P(xᵢ|...)   │ BERT training           │
│ InfoNCE         │ -log(exp/Σexp)         │ Contrastive learning    │
│ DPO             │ -log σ(β × Δ)          │ Preference alignment    │
└─────────────────┴────────────────────────┴─────────────────────────┘

KEY TAKEAWAYS:
1. Cross-entropy = standard for classification
2. Label smoothing = always use in transformers (ε=0.1)
3. Perplexity = exp(CE), lower is better
4. Causal LM = shift, then CE
5. DPO = simpler than RLHF, direct optimization
6. Always use numerically stable implementations!

MODERN LLM TRAINING PIPELINE:
  Pre-training → Causal LM loss
  SFT → CE on response tokens only
  Alignment → DPO or RLHF
""")

print("\nModule 2.3 Complete!")
