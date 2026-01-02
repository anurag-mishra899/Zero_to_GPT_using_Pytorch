"""
Module 16: LLM Quantization - Implementation

This module covers:
1. Basic quantization operations
2. Symmetric and asymmetric quantization
3. Per-tensor and per-channel quantization
4. Block-wise quantization
5. GPTQ-style quantization
6. AWQ-style quantization
"""

import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Section 1: Basic Quantization Operations
# ============================================================================

def quantize_symmetric(
    x: torch.Tensor,
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization: maps [-alpha, alpha] to [-2^(b-1)+1, 2^(b-1)-1].

    Args:
        x: Input tensor
        bits: Number of bits

    Returns:
        x_q: Quantized tensor (int)
        scale: Scale factor
    """
    qmax = 2 ** (bits - 1) - 1  # e.g., 127 for 8-bit
    qmin = -qmax

    # Scale based on max absolute value
    alpha = x.abs().max()
    scale = alpha / qmax

    # Quantize
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)

    return x_q, scale


def dequantize_symmetric(
    x_q: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize symmetric quantized tensor."""
    return x_q.float() * scale


def quantize_asymmetric(
    x: torch.Tensor,
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization: maps [min, max] to [0, 2^b - 1].

    Args:
        x: Input tensor
        bits: Number of bits

    Returns:
        x_q: Quantized tensor (uint8)
        scale: Scale factor
        zero_point: Zero point offset
    """
    qmax = 2 ** bits - 1  # e.g., 255 for 8-bit
    qmin = 0

    # Scale and zero point
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / qmax
    zero_point = torch.round(-x_min / scale).clamp(qmin, qmax)

    # Quantize
    x_q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax).to(torch.uint8)

    return x_q, scale, zero_point


def dequantize_asymmetric(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize asymmetric quantized tensor."""
    return (x_q.float() - zero_point) * scale


# ============================================================================
# Section 2: Per-Channel Quantization
# ============================================================================

def quantize_per_channel(
    x: torch.Tensor,
    bits: int = 8,
    axis: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel symmetric quantization.

    Each channel (along axis) gets its own scale.

    Args:
        x: Input tensor
        bits: Number of bits
        axis: Channel axis

    Returns:
        x_q: Quantized tensor
        scales: Scale per channel
    """
    qmax = 2 ** (bits - 1) - 1

    # Move target axis to first position
    x_t = x.transpose(0, axis)
    original_shape = x_t.shape
    x_flat = x_t.reshape(x_t.shape[0], -1)  # (channels, elements)

    # Scale per channel
    alpha = x_flat.abs().max(dim=1).values
    scales = alpha / qmax
    scales = scales.clamp(min=1e-8)  # Avoid division by zero

    # Quantize each channel
    x_q = torch.clamp(
        torch.round(x_flat / scales.unsqueeze(1)),
        -qmax, qmax
    ).to(torch.int8)

    # Reshape back
    x_q = x_q.reshape(original_shape).transpose(0, axis)

    return x_q, scales


def dequantize_per_channel(
    x_q: torch.Tensor,
    scales: torch.Tensor,
    axis: int = 0
) -> torch.Tensor:
    """Dequantize per-channel quantized tensor."""
    # Expand scales to match tensor shape
    shape = [1] * x_q.dim()
    shape[axis] = -1
    scales_expanded = scales.reshape(shape)

    return x_q.float() * scales_expanded


# ============================================================================
# Section 3: Block-wise Quantization (for 4-bit)
# ============================================================================

def quantize_blockwise(
    x: torch.Tensor,
    bits: int = 4,
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block-wise quantization for INT4.

    Divides tensor into blocks, quantizes each independently.

    Args:
        x: Input tensor
        bits: Number of bits
        block_size: Elements per block

    Returns:
        x_q: Quantized tensor (packed if 4-bit)
        scales: Scale per block
        zeros: Zero point per block
    """
    qmax = 2 ** bits - 1

    # Flatten and pad to multiple of block_size
    x_flat = x.flatten()
    n_elements = x_flat.numel()
    n_blocks = (n_elements + block_size - 1) // block_size
    padded_size = n_blocks * block_size

    if n_elements < padded_size:
        x_flat = F.pad(x_flat, (0, padded_size - n_elements))

    # Reshape into blocks
    x_blocks = x_flat.reshape(n_blocks, block_size)

    # Compute scale and zero per block
    block_min = x_blocks.min(dim=1).values
    block_max = x_blocks.max(dim=1).values

    scales = (block_max - block_min) / qmax
    scales = scales.clamp(min=1e-8)
    zeros = torch.round(-block_min / scales).clamp(0, qmax)

    # Quantize
    x_q = torch.clamp(
        torch.round(x_blocks / scales.unsqueeze(1)) + zeros.unsqueeze(1),
        0, qmax
    ).to(torch.uint8)

    # Flatten back (truncate padding)
    x_q = x_q.flatten()[:n_elements].reshape(x.shape)

    return x_q, scales, zeros


def dequantize_blockwise(
    x_q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
    """Dequantize block-wise quantized tensor."""
    original_shape = x_q.shape
    x_flat = x_q.flatten()
    n_elements = x_flat.numel()
    n_blocks = len(scales)

    # Pad if needed
    padded_size = n_blocks * block_size
    if n_elements < padded_size:
        x_flat = F.pad(x_flat.float(), (0, padded_size - n_elements))

    # Reshape into blocks
    x_blocks = x_flat.reshape(n_blocks, block_size)

    # Dequantize
    x_dequant = (x_blocks - zeros.unsqueeze(1)) * scales.unsqueeze(1)

    # Flatten and reshape
    return x_dequant.flatten()[:n_elements].reshape(original_shape)


# ============================================================================
# Section 4: Quantized Linear Layer
# ============================================================================

class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with INT8 weights.

    Weights are stored in INT8 format.
    Computation happens in FP16/FP32 after dequantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        per_channel: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.per_channel = per_channel

        # Quantized weights (stored as int8)
        self.register_buffer(
            'weight_q',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )

        # Scales
        if per_channel:
            self.register_buffer('scales', torch.ones(out_features))
        else:
            self.register_buffer('scales', torch.ones(1))

        # Bias (keep in FP32)
        self.bias = nn.Parameter(torch.zeros(out_features))

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        per_channel: bool = True
    ) -> 'QuantizedLinear':
        """Create quantized layer from float linear layer."""
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            per_channel=per_channel
        )

        # Quantize weights
        if per_channel:
            weight_q, scales = quantize_per_channel(
                linear.weight.data, bits=bits, axis=0
            )
        else:
            weight_q, scales = quantize_symmetric(linear.weight.data, bits=bits)

        quant_linear.weight_q = weight_q
        quant_linear.scales = scales

        if linear.bias is not None:
            quant_linear.bias = nn.Parameter(linear.bias.data.clone())

        return quant_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantized weights."""
        # Dequantize weights
        if self.per_channel:
            weight = dequantize_per_channel(self.weight_q, self.scales, axis=0)
        else:
            weight = dequantize_symmetric(self.weight_q, self.scales)

        # Linear operation
        return F.linear(x, weight.to(x.dtype), self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, bits={self.bits}'


# ============================================================================
# Section 5: INT4 Linear Layer
# ============================================================================

class Int4Linear(nn.Module):
    """
    4-bit quantized linear layer with block-wise quantization.

    Stores weights in 4-bit format with per-block scales.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 128,
        bits: int = 4
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.bits = bits

        # Calculate number of blocks
        total_weights = out_features * in_features
        self.n_blocks = (total_weights + block_size - 1) // block_size

        # Quantized weights (stored as uint8 for simplicity)
        self.register_buffer(
            'weight_q',
            torch.zeros(out_features, in_features, dtype=torch.uint8)
        )

        # Per-block scales and zeros
        self.register_buffer('scales', torch.ones(self.n_blocks))
        self.register_buffer('zeros', torch.zeros(self.n_blocks))

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        block_size: int = 128
    ) -> 'Int4Linear':
        """Create INT4 layer from float linear."""
        int4_linear = cls(
            linear.in_features,
            linear.out_features,
            block_size=block_size
        )

        # Quantize
        weight_q, scales, zeros = quantize_blockwise(
            linear.weight.data,
            bits=4,
            block_size=block_size
        )

        int4_linear.weight_q = weight_q
        int4_linear.scales = scales
        int4_linear.zeros = zeros

        if linear.bias is not None:
            int4_linear.bias = nn.Parameter(linear.bias.data.clone())

        return int4_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dequantized weights."""
        weight = dequantize_blockwise(
            self.weight_q, self.scales, self.zeros, self.block_size
        )
        return F.linear(x, weight.to(x.dtype), self.bias.to(x.dtype))


# ============================================================================
# Section 6: GPTQ-Style Quantization
# ============================================================================

class GPTQQuantizer:
    """
    GPTQ-style quantization using Hessian-based error compensation.

    GPTQ minimizes ||WX - W_qX||² by:
    1. Quantizing columns in order of sensitivity
    2. Compensating errors in remaining columns

    Simplified implementation for demonstration.
    """

    def __init__(
        self,
        bits: int = 4,
        block_size: int = 128,
        damp_percent: float = 0.01
    ):
        self.bits = bits
        self.block_size = block_size
        self.damp_percent = damp_percent

    def quantize_weight(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight matrix using GPTQ algorithm.

        Args:
            weight: Weight matrix (out_features, in_features)
            hessian: Hessian approximation H = X^T X (in_features, in_features)

        Returns:
            weight_q: Quantized weights
            scales: Quantization scales
            zeros: Zero points
        """
        out_features, in_features = weight.shape

        # Add damping to Hessian for numerical stability
        damp = self.damp_percent * torch.diag(hessian).mean()
        hessian = hessian + damp * torch.eye(in_features, device=hessian.device)

        # Compute inverse Hessian
        try:
            hessian_inv = torch.linalg.inv(hessian)
        except RuntimeError:
            # Fallback to pseudo-inverse
            hessian_inv = torch.linalg.pinv(hessian)

        # Working copy of weights
        W = weight.clone()

        # Process in blocks
        qmax = 2 ** self.bits - 1

        for block_start in range(0, in_features, self.block_size):
            block_end = min(block_start + self.block_size, in_features)

            # Process columns in this block
            for col in range(block_start, block_end):
                # Get column
                w_col = W[:, col]

                # Quantize this column
                col_min, col_max = w_col.min(), w_col.max()
                scale = (col_max - col_min) / qmax
                scale = max(scale, 1e-8)
                zero = round(-col_min.item() / scale)

                w_q_col = torch.clamp(
                    torch.round(w_col / scale) + zero,
                    0, qmax
                )

                # Dequantize
                w_dq_col = (w_q_col - zero) * scale

                # Compute error
                error = w_col - w_dq_col

                # Update remaining columns to compensate
                if col < in_features - 1:
                    for j in range(col + 1, block_end):
                        W[:, j] -= error * (hessian_inv[col, j] / hessian_inv[col, col])

                # Store quantized value
                W[:, col] = w_q_col

        # Final quantization of full weight
        weight_q, scales, zeros = quantize_blockwise(
            weight, bits=self.bits, block_size=self.block_size
        )

        return weight_q, scales, zeros

    def collect_hessian(
        self,
        module: nn.Linear,
        calibration_data: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Collect Hessian approximation from calibration data.

        H ≈ X^T X where X is the input activations.
        """
        in_features = module.in_features
        hessian = torch.zeros(in_features, in_features, device=module.weight.device)
        n_samples = 0

        for X in calibration_data:
            # X shape: (batch, seq, features) or (batch, features)
            if X.dim() == 3:
                X = X.reshape(-1, X.shape[-1])  # Flatten batch and seq

            # Accumulate H = X^T X
            hessian += X.T @ X
            n_samples += X.shape[0]

        # Average
        hessian /= n_samples

        return hessian


# ============================================================================
# Section 7: AWQ-Style Quantization
# ============================================================================

class AWQQuantizer:
    """
    AWQ-style quantization using activation-aware scaling.

    AWQ identifies salient channels (high activation) and
    scales them up before quantization to preserve precision.

    Simplified implementation for demonstration.
    """

    def __init__(
        self,
        bits: int = 4,
        block_size: int = 128,
        salient_ratio: float = 0.01  # Top 1% of channels
    ):
        self.bits = bits
        self.block_size = block_size
        self.salient_ratio = salient_ratio

    def find_salient_channels(
        self,
        activations: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find salient channels based on activation magnitudes.

        Args:
            activations: List of activation tensors

        Returns:
            salience: Per-channel salience scores
            salient_mask: Boolean mask for salient channels
        """
        # Compute average activation magnitude per channel
        all_activations = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations])
        salience = all_activations.abs().mean(dim=0)

        # Find top k% salient channels
        k = max(1, int(salience.numel() * self.salient_ratio))
        threshold = salience.topk(k).values[-1]
        salient_mask = salience >= threshold

        return salience, salient_mask

    def compute_scale_factors(
        self,
        weight: torch.Tensor,
        salience: torch.Tensor,
        salient_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-channel scale factors.

        Salient channels get scaled up before quantization.
        """
        in_features = weight.shape[1]
        scale_factors = torch.ones(in_features, device=weight.device)

        # Scale salient channels based on their salience
        # Higher salience = higher scale
        salient_salience = salience[salient_mask]
        if salient_salience.numel() > 0:
            max_salience = salient_salience.max()
            scale_factors[salient_mask] = salience[salient_mask] / max_salience * 2 + 1

        return scale_factors

    def quantize_weight(
        self,
        weight: torch.Tensor,
        activations: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weight with AWQ scaling.

        Args:
            weight: Weight matrix
            activations: Calibration activations

        Returns:
            weight_q: Quantized weights
            scales: Quantization scales
            zeros: Zero points
            channel_scales: Per-channel AWQ scales
        """
        # Find salient channels
        salience, salient_mask = self.find_salient_channels(activations)

        # Compute scale factors
        channel_scales = self.compute_scale_factors(weight, salience, salient_mask)

        # Scale weights before quantization
        weight_scaled = weight * channel_scales.unsqueeze(0)

        # Quantize scaled weights
        weight_q, scales, zeros = quantize_blockwise(
            weight_scaled, bits=self.bits, block_size=self.block_size
        )

        return weight_q, scales, zeros, channel_scales


class AWQLinear(nn.Module):
    """Linear layer with AWQ quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        n_blocks = (out_features * in_features + block_size - 1) // block_size

        self.register_buffer('weight_q', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('scales', torch.ones(n_blocks))
        self.register_buffer('zeros', torch.zeros(n_blocks))
        self.register_buffer('channel_scales', torch.ones(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with AWQ dequantization."""
        # Dequantize
        weight = dequantize_blockwise(
            self.weight_q, self.scales, self.zeros, self.block_size
        )

        # Reverse channel scaling
        weight = weight / self.channel_scales.unsqueeze(0)

        return F.linear(x, weight.to(x.dtype), self.bias.to(x.dtype))


# ============================================================================
# Section 8: Model Quantization Utilities
# ============================================================================

def quantize_model(
    model: nn.Module,
    bits: int = 8,
    method: str = 'symmetric'
) -> nn.Module:
    """
    Quantize all linear layers in a model.

    Args:
        model: PyTorch model
        bits: Quantization bits
        method: 'symmetric' or 'int4'

    Returns:
        Quantized model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if method == 'symmetric' or bits == 8:
                quant_layer = QuantizedLinear.from_float(module, bits=bits)
            else:
                quant_layer = Int4Linear.from_float(module)
            setattr(model, name, quant_layer)
        else:
            quantize_model(module, bits=bits, method=method)

    return model


def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor
) -> Dict[str, float]:
    """Compute quantization error metrics."""
    diff = original - quantized

    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    relative_error = (diff.abs() / (original.abs() + 1e-8)).mean().item()

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'relative_error': relative_error
    }


def estimate_memory(
    num_params: int,
    bits: int,
    block_size: int = 128
) -> Dict[str, float]:
    """
    Estimate memory usage for quantized model.

    Args:
        num_params: Number of parameters
        bits: Quantization bits
        block_size: Block size for block-wise quantization

    Returns:
        Memory estimates in GB
    """
    # Weight memory
    weight_bytes = num_params * bits / 8

    # Scale/zero memory (FP16 per block)
    n_blocks = (num_params + block_size - 1) // block_size
    meta_bytes = n_blocks * 4  # 2 bytes scale + 2 bytes zero

    total_bytes = weight_bytes + meta_bytes

    return {
        'weight_gb': weight_bytes / 1e9,
        'meta_gb': meta_bytes / 1e9,
        'total_gb': total_bytes / 1e9,
        'bits_effective': total_bytes * 8 / num_params
    }


# ============================================================================
# Section 9: Demonstration Functions
# ============================================================================

def demo_basic_quantization():
    """Demonstrate basic quantization operations."""
    print("=" * 60)
    print("Basic Quantization Demo")
    print("=" * 60)

    # Create sample tensor
    x = torch.randn(4, 4) * 2

    print("\nOriginal tensor:")
    print(x)

    # Symmetric quantization
    x_q_sym, scale_sym = quantize_symmetric(x, bits=8)
    x_dq_sym = dequantize_symmetric(x_q_sym, scale_sym)

    print(f"\nSymmetric INT8:")
    print(f"  Scale: {scale_sym:.6f}")
    print(f"  Quantized:\n{x_q_sym}")

    error_sym = compute_quantization_error(x, x_dq_sym)
    print(f"  MSE: {error_sym['mse']:.6f}")

    # Asymmetric quantization
    x_q_asym, scale_asym, zp_asym = quantize_asymmetric(x, bits=8)
    x_dq_asym = dequantize_asymmetric(x_q_asym, scale_asym, zp_asym)

    print(f"\nAsymmetric INT8:")
    print(f"  Scale: {scale_asym:.6f}, Zero point: {zp_asym}")

    error_asym = compute_quantization_error(x, x_dq_asym)
    print(f"  MSE: {error_asym['mse']:.6f}")


def demo_per_channel():
    """Demonstrate per-channel quantization."""
    print("\n" + "=" * 60)
    print("Per-Channel Quantization Demo")
    print("=" * 60)

    # Simulate weight matrix with varying scales per output channel
    weight = torch.randn(4, 8)
    weight[0] *= 0.1  # Small values
    weight[1] *= 10   # Large values

    print("\nWeight matrix (channels have different scales):")
    print(f"  Channel 0 range: [{weight[0].min():.4f}, {weight[0].max():.4f}]")
    print(f"  Channel 1 range: [{weight[1].min():.4f}, {weight[1].max():.4f}]")

    # Per-tensor quantization
    w_q_tensor, scale_tensor = quantize_symmetric(weight, bits=8)
    w_dq_tensor = dequantize_symmetric(w_q_tensor, scale_tensor)

    error_tensor = compute_quantization_error(weight, w_dq_tensor)
    print(f"\nPer-tensor quantization MSE: {error_tensor['mse']:.6f}")

    # Per-channel quantization
    w_q_channel, scales_channel = quantize_per_channel(weight, bits=8, axis=0)
    w_dq_channel = dequantize_per_channel(w_q_channel, scales_channel, axis=0)

    error_channel = compute_quantization_error(weight, w_dq_channel)
    print(f"Per-channel quantization MSE: {error_channel['mse']:.6f}")

    print(f"\nPer-channel {error_tensor['mse']/error_channel['mse']:.1f}x better!")


def demo_int4_quantization():
    """Demonstrate INT4 block-wise quantization."""
    print("\n" + "=" * 60)
    print("INT4 Block-wise Quantization Demo")
    print("=" * 60)

    # Create weight matrix
    weight = torch.randn(256, 256)

    print(f"\nWeight shape: {weight.shape}")
    print(f"Original size: {weight.numel() * 4 / 1024:.1f} KB (FP32)")
    print(f"FP16 size: {weight.numel() * 2 / 1024:.1f} KB")

    # INT8 quantization
    w_q8, scale8 = quantize_symmetric(weight, bits=8)
    w_dq8 = dequantize_symmetric(w_q8, scale8)
    error8 = compute_quantization_error(weight, w_dq8)

    print(f"\nINT8:")
    print(f"  Size: {weight.numel() * 1 / 1024:.1f} KB")
    print(f"  MSE: {error8['mse']:.6f}")

    # INT4 block-wise
    w_q4, scales4, zeros4 = quantize_blockwise(weight, bits=4, block_size=128)
    w_dq4 = dequantize_blockwise(w_q4, scales4, zeros4, block_size=128)
    error4 = compute_quantization_error(weight, w_dq4)

    mem4 = estimate_memory(weight.numel(), bits=4, block_size=128)
    print(f"\nINT4 (block_size=128):")
    print(f"  Size: {mem4['total_gb'] * 1024 * 1024:.1f} KB")
    print(f"  Effective bits: {mem4['bits_effective']:.2f}")
    print(f"  MSE: {error4['mse']:.6f}")


def demo_quantized_linear():
    """Demonstrate quantized linear layer."""
    print("\n" + "=" * 60)
    print("Quantized Linear Layer Demo")
    print("=" * 60)

    # Create float linear layer
    linear = nn.Linear(512, 256)

    # Create quantized versions
    quant8 = QuantizedLinear.from_float(linear, bits=8)
    quant4 = Int4Linear.from_float(linear, block_size=128)

    # Test forward pass
    x = torch.randn(2, 10, 512)

    with torch.no_grad():
        y_float = linear(x)
        y_int8 = quant8(x)
        y_int4 = quant4(x)

    # Compare outputs
    error8 = compute_quantization_error(y_float, y_int8)
    error4 = compute_quantization_error(y_float, y_int4)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y_float.shape}")

    print(f"\nINT8 output error: MSE={error8['mse']:.6f}")
    print(f"INT4 output error: MSE={error4['mse']:.6f}")

    # Memory comparison
    float_mem = linear.weight.numel() * 4
    int8_mem = quant8.weight_q.numel() * 1 + quant8.scales.numel() * 4
    int4_mem = estimate_memory(quant4.weight_q.numel(), 4, 128)['total_gb'] * 1e9

    print(f"\nMemory comparison:")
    print(f"  FP32: {float_mem / 1024:.1f} KB")
    print(f"  INT8: {int8_mem / 1024:.1f} KB ({float_mem/int8_mem:.1f}x reduction)")
    print(f"  INT4: {int4_mem / 1024:.1f} KB ({float_mem/int4_mem:.1f}x reduction)")


def demo_memory_estimation():
    """Demonstrate memory estimation for LLMs."""
    print("\n" + "=" * 60)
    print("LLM Memory Estimation")
    print("=" * 60)

    models = [
        ("LLaMA-7B", 7e9),
        ("LLaMA-13B", 13e9),
        ("LLaMA-70B", 70e9),
    ]

    print(f"\n{'Model':<15} {'FP16':>10} {'INT8':>10} {'INT4':>10}")
    print("-" * 50)

    for name, params in models:
        fp16 = params * 2 / 1e9
        int8 = estimate_memory(int(params), 8)['total_gb']
        int4 = estimate_memory(int(params), 4)['total_gb']

        print(f"{name:<15} {fp16:>9.1f}GB {int8:>9.1f}GB {int4:>9.1f}GB")


def demo_quantization_comparison():
    """Compare different quantization methods."""
    print("\n" + "=" * 60)
    print("Quantization Methods Comparison")
    print("=" * 60)

    # Create test weight
    weight = torch.randn(1024, 1024)

    methods = []

    # FP16 baseline
    methods.append(("FP16 (baseline)", weight, 16))

    # INT8 symmetric
    w_q, s = quantize_symmetric(weight, 8)
    w_dq = dequantize_symmetric(w_q, s)
    methods.append(("INT8 symmetric", w_dq, 8))

    # INT8 per-channel
    w_q, s = quantize_per_channel(weight, 8, 0)
    w_dq = dequantize_per_channel(w_q, s, 0)
    methods.append(("INT8 per-channel", w_dq, 8))

    # INT4 block-wise
    w_q, s, z = quantize_blockwise(weight, 4, 128)
    w_dq = dequantize_blockwise(w_q, s, z, 128)
    methods.append(("INT4 block-128", w_dq, 4.5))

    # INT4 block-wise (smaller blocks)
    w_q, s, z = quantize_blockwise(weight, 4, 32)
    w_dq = dequantize_blockwise(w_q, s, z, 32)
    methods.append(("INT4 block-32", w_dq, 5))

    print(f"\n{'Method':<20} {'Bits':>6} {'MSE':>12} {'MaxErr':>10}")
    print("-" * 50)

    for name, w_dq, bits in methods:
        if name == "FP16 (baseline)":
            print(f"{name:<20} {bits:>6.1f} {'N/A':>12} {'N/A':>10}")
        else:
            error = compute_quantization_error(weight, w_dq)
            print(f"{name:<20} {bits:>6.1f} {error['mse']:>12.6f} {error['max_error']:>10.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("Module 16: LLM Quantization")
    print("=" * 60)

    demo_basic_quantization()
    demo_per_channel()
    demo_int4_quantization()
    demo_quantized_linear()
    demo_memory_estimation()
    demo_quantization_comparison()

    print("\n" + "=" * 60)
    print("Module 16 Complete!")
    print("=" * 60)
    print("\nKey concepts covered:")
    print("1. Symmetric vs asymmetric quantization")
    print("2. Per-tensor vs per-channel quantization")
    print("3. Block-wise quantization for INT4")
    print("4. Quantized linear layers")
    print("5. GPTQ-style Hessian-based quantization")
    print("6. AWQ-style activation-aware quantization")
    print("7. Memory estimation for LLMs")


if __name__ == "__main__":
    main()
