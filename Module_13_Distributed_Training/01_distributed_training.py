"""
Module 13: Distributed Training - Implementation

This module covers:
1. Basic distributed setup
2. DDP implementation
3. FSDP implementation
4. Gradient synchronization utilities
5. Model parallelism concepts
"""

import os
import math
import functools
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Section 1: Distributed Training Basics (Simulation)
# ============================================================================

class DistributedSimulator:
    """
    Simulates distributed training concepts without actual multi-GPU setup.
    Useful for understanding the algorithms.
    """

    def __init__(self, world_size: int = 4):
        self.world_size = world_size

    def all_reduce_sum(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Simulate all-reduce with sum operation.

        All-Reduce: Each GPU starts with different data,
        ends with the sum (or average) of all data.

        Args:
            tensors: List of tensors, one per "GPU"

        Returns:
            Reduced tensor (sum of all inputs)
        """
        print("All-Reduce (Sum) Operation:")
        for i, t in enumerate(tensors):
            print(f"  GPU {i} input: {t.tolist()}")

        result = sum(tensors)
        print(f"  Result on ALL GPUs: {result.tolist()}")
        return result

    def all_reduce_mean(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """All-reduce with mean (used for gradient averaging)."""
        return self.all_reduce_sum(tensors) / self.world_size

    def all_gather(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Simulate all-gather operation.

        All-Gather: Each GPU has a piece, all GPUs end with all pieces.

        Args:
            tensors: List of tensors, one per "GPU"

        Returns:
            List of all tensors (same on each GPU)
        """
        print("All-Gather Operation:")
        for i, t in enumerate(tensors):
            print(f"  GPU {i} has: {t.tolist()}")

        gathered = tensors.copy()
        print(f"  After gather, ALL GPUs have: {[t.tolist() for t in gathered]}")
        return gathered

    def reduce_scatter(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Simulate reduce-scatter operation.

        Reduce-Scatter: Reduce all tensors, then scatter result pieces.
        Each GPU gets 1/N of the reduced result.

        Args:
            tensors: List of tensors (same shape), one per "GPU"

        Returns:
            List of tensor chunks, one per GPU
        """
        print("Reduce-Scatter Operation:")
        for i, t in enumerate(tensors):
            print(f"  GPU {i} input: {t.tolist()}")

        # Sum all tensors
        total = sum(tensors)

        # Scatter - each GPU gets a portion
        chunk_size = total.shape[0] // self.world_size
        results = []
        for i in range(self.world_size):
            start = i * chunk_size
            end = start + chunk_size
            results.append(total[start:end])
            print(f"  GPU {i} receives: {results[-1].tolist()}")

        return results

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> List[torch.Tensor]:
        """
        Simulate broadcast from one GPU to all.

        Args:
            tensor: Tensor to broadcast
            src: Source GPU

        Returns:
            List of tensors (all identical to source)
        """
        print(f"Broadcast from GPU {src}:")
        print(f"  Source tensor: {tensor.tolist()}")
        results = [tensor.clone() for _ in range(self.world_size)]
        print(f"  All GPUs now have: {tensor.tolist()}")
        return results


def demo_collective_operations():
    """Demonstrate collective operations."""
    print("=" * 60)
    print("Collective Communication Operations Demo")
    print("=" * 60)

    sim = DistributedSimulator(world_size=4)

    # All-reduce
    print("\n1. All-Reduce (for gradient synchronization):")
    gradients = [
        torch.tensor([1.0, 2.0]),  # GPU 0's gradient
        torch.tensor([3.0, 4.0]),  # GPU 1's gradient
        torch.tensor([5.0, 6.0]),  # GPU 2's gradient
        torch.tensor([7.0, 8.0]),  # GPU 3's gradient
    ]
    avg_gradient = sim.all_reduce_mean(gradients)
    print(f"  Average gradient: {avg_gradient.tolist()}")

    # All-gather
    print("\n2. All-Gather (for FSDP parameter gathering):")
    param_shards = [
        torch.tensor([1.0]),  # GPU 0's shard
        torch.tensor([2.0]),  # GPU 1's shard
        torch.tensor([3.0]),  # GPU 2's shard
        torch.tensor([4.0]),  # GPU 3's shard
    ]
    sim.all_gather(param_shards)

    # Reduce-scatter
    print("\n3. Reduce-Scatter (for FSDP gradient distribution):")
    full_gradients = [
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.tensor([5.0, 6.0, 7.0, 8.0]),
        torch.tensor([9.0, 10.0, 11.0, 12.0]),
        torch.tensor([13.0, 14.0, 15.0, 16.0]),
    ]
    sim.reduce_scatter(full_gradients)

    # Broadcast
    print("\n4. Broadcast (for parameter initialization):")
    init_params = torch.tensor([0.1, 0.2, 0.3])
    sim.broadcast(init_params, src=0)


# ============================================================================
# Section 2: DDP Simulation
# ============================================================================

class DDPSimulator:
    """
    Simulates Distributed Data Parallel training.

    DDP Algorithm:
    1. Each GPU has full model copy
    2. Each GPU processes different data
    3. All-reduce gradients after backward
    4. All GPUs update identically
    """

    def __init__(self, model: nn.Module, world_size: int = 4):
        self.world_size = world_size
        # Simulate model copies on each GPU
        self.models = [self._clone_model(model) for _ in range(world_size)]
        self.gradient_buckets = [[] for _ in range(world_size)]

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Clone model for each simulated GPU."""
        import copy
        return copy.deepcopy(model)

    def forward_backward(
        self,
        batches: List[torch.Tensor],
        targets: List[torch.Tensor],
        criterion: nn.Module
    ) -> List[float]:
        """
        Simulate forward and backward on all GPUs.

        Args:
            batches: List of input batches, one per GPU
            targets: List of targets, one per GPU
            criterion: Loss function

        Returns:
            List of losses, one per GPU
        """
        losses = []

        for gpu_id, (model, batch, target) in enumerate(
            zip(self.models, batches, targets)
        ):
            # Forward
            output = model(batch)
            loss = criterion(output, target)
            losses.append(loss.item())

            # Backward (computes gradients locally)
            loss.backward()

            print(f"GPU {gpu_id}: Loss = {loss.item():.4f}")

        return losses

    def sync_gradients(self):
        """
        Synchronize gradients across all GPUs using all-reduce.
        This is what DDP does automatically.
        """
        print("\nSynchronizing gradients (All-Reduce)...")

        # For each parameter, average gradients across GPUs
        for param_idx, param_name in enumerate(
            dict(self.models[0].named_parameters()).keys()
        ):
            # Collect gradients from all GPUs
            grads = []
            for model in self.models:
                param = dict(model.named_parameters())[param_name]
                if param.grad is not None:
                    grads.append(param.grad.clone())

            if grads:
                # All-reduce (average)
                avg_grad = sum(grads) / self.world_size

                # Set averaged gradient on all GPUs
                for model in self.models:
                    param = dict(model.named_parameters())[param_name]
                    param.grad = avg_grad.clone()

        print("Gradients synchronized!")

    def step(self, optimizers: List[torch.optim.Optimizer]):
        """Update all models (should result in identical parameters)."""
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

    def verify_sync(self):
        """Verify all models have identical parameters."""
        ref_params = list(self.models[0].parameters())

        for gpu_id, model in enumerate(self.models[1:], 1):
            for p1, p2 in zip(ref_params, model.parameters()):
                if not torch.allclose(p1, p2):
                    print(f"WARNING: GPU {gpu_id} parameters differ!")
                    return False

        print("All model copies are synchronized!")
        return True


def demo_ddp_training():
    """Demonstrate DDP training simulation."""
    print("\n" + "=" * 60)
    print("DDP Training Simulation")
    print("=" * 60)

    # Simple model
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    # Create DDP simulator
    ddp = DDPSimulator(model, world_size=4)

    # Create optimizers for each model copy
    optimizers = [
        torch.optim.SGD(m.parameters(), lr=0.01)
        for m in ddp.models
    ]

    # Simulate different batches on each GPU
    batches = [torch.randn(2, 4) for _ in range(4)]  # Different data!
    targets = [torch.randint(0, 2, (2,)) for _ in range(4)]

    criterion = nn.CrossEntropyLoss()

    print("\nStep 1: Forward + Backward (independent on each GPU)")
    ddp.forward_backward(batches, targets, criterion)

    print("\nStep 2: Gradient Synchronization")
    ddp.sync_gradients()

    print("\nStep 3: Optimizer Step")
    ddp.step(optimizers)

    print("\nStep 4: Verify Synchronization")
    ddp.verify_sync()


# ============================================================================
# Section 3: Gradient Bucketing
# ============================================================================

class GradientBucketing:
    """
    Demonstrates gradient bucketing for efficient all-reduce.

    Instead of all-reducing each gradient tensor separately,
    we group them into buckets for better efficiency.
    """

    def __init__(self, bucket_size_mb: float = 25.0):
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets: List[List[torch.Tensor]] = []
        self.current_bucket: List[torch.Tensor] = []
        self.current_size = 0

    def add_gradient(self, grad: torch.Tensor):
        """Add gradient to current bucket, create new if full."""
        grad_size = grad.numel() * grad.element_size()

        if self.current_size + grad_size > self.bucket_size_bytes:
            # Bucket full, start new one
            if self.current_bucket:
                self.buckets.append(self.current_bucket)
            self.current_bucket = [grad]
            self.current_size = grad_size
        else:
            self.current_bucket.append(grad)
            self.current_size += grad_size

    def finalize(self):
        """Finalize last bucket."""
        if self.current_bucket:
            self.buckets.append(self.current_bucket)
            self.current_bucket = []
            self.current_size = 0

    def get_bucket_stats(self) -> Dict[str, Any]:
        """Get statistics about buckets."""
        return {
            'num_buckets': len(self.buckets),
            'bucket_sizes': [
                sum(g.numel() * g.element_size() for g in b) / (1024 * 1024)
                for b in self.buckets
            ]
        }


def demo_gradient_bucketing():
    """Demonstrate gradient bucketing."""
    print("\n" + "=" * 60)
    print("Gradient Bucketing Demo")
    print("=" * 60)

    # Create a model with various layer sizes
    model = nn.Sequential(
        nn.Linear(1024, 2048),  # ~8MB
        nn.ReLU(),
        nn.Linear(2048, 4096),  # ~32MB
        nn.ReLU(),
        nn.Linear(4096, 1024),  # ~16MB
    )

    # Simulate gradients
    bucketing = GradientBucketing(bucket_size_mb=25.0)

    print("\nAdding gradients to buckets (25MB bucket size):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Simulate gradient
            grad = torch.randn_like(param)
            size_mb = grad.numel() * grad.element_size() / (1024 * 1024)
            print(f"  {name}: {size_mb:.2f} MB")
            bucketing.add_gradient(grad)

    bucketing.finalize()

    stats = bucketing.get_bucket_stats()
    print(f"\nBucket Statistics:")
    print(f"  Number of buckets: {stats['num_buckets']}")
    print(f"  Bucket sizes (MB): {[f'{s:.2f}' for s in stats['bucket_sizes']]}")
    print(f"\nWithout bucketing: {sum(1 for _ in model.parameters())} all-reduce calls")
    print(f"With bucketing: {stats['num_buckets']} all-reduce calls")


# ============================================================================
# Section 4: FSDP Simulation
# ============================================================================

class FSDPSimulator:
    """
    Simulates Fully Sharded Data Parallel training.

    FSDP shards parameters, gradients, and optimizer states
    across GPUs. Parameters are gathered when needed and
    discarded after use.

    Memory per GPU: P/N instead of P (where N = world_size)
    Communication: 3P instead of 2P (more comms, less memory)
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int = 4
    ):
        self.world_size = world_size
        self.model = model

        # Shard parameters across GPUs
        self.param_shards = self._shard_parameters()

        # Track memory usage
        self.peak_memory = [0] * world_size

    def _shard_parameters(self) -> List[Dict[str, torch.Tensor]]:
        """Shard each parameter across GPUs."""
        shards = [{} for _ in range(self.world_size)]

        for name, param in self.model.named_parameters():
            # Flatten parameter
            flat = param.data.flatten()

            # Calculate shard size (pad if necessary)
            shard_size = (flat.numel() + self.world_size - 1) // self.world_size
            padded_size = shard_size * self.world_size

            if flat.numel() < padded_size:
                flat = F.pad(flat, (0, padded_size - flat.numel()))

            # Distribute shards
            for gpu_id in range(self.world_size):
                start = gpu_id * shard_size
                end = start + shard_size
                shards[gpu_id][name] = {
                    'shard': flat[start:end].clone(),
                    'original_shape': param.shape,
                    'original_numel': param.numel()
                }

        return shards

    def all_gather_params(self, gpu_id: int) -> Dict[str, torch.Tensor]:
        """
        Simulate all-gather to reconstruct full parameters.
        Called before forward/backward pass.
        """
        full_params = {}

        for name in self.param_shards[0].keys():
            # Gather shards from all GPUs
            shards = [
                self.param_shards[i][name]['shard']
                for i in range(self.world_size)
            ]

            # Concatenate
            full_flat = torch.cat(shards)

            # Reshape to original
            original_shape = self.param_shards[gpu_id][name]['original_shape']
            original_numel = self.param_shards[gpu_id][name]['original_numel']
            full_params[name] = full_flat[:original_numel].reshape(original_shape)

        return full_params

    def reduce_scatter_grads(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Simulate reduce-scatter for gradient distribution.
        Each GPU ends up with 1/N of the gradients.
        """
        grad_shards = [{} for _ in range(self.world_size)]

        for name, grad in gradients.items():
            flat_grad = grad.flatten()

            # Pad if necessary
            shard_size = (flat_grad.numel() + self.world_size - 1) // self.world_size
            padded_size = shard_size * self.world_size

            if flat_grad.numel() < padded_size:
                flat_grad = F.pad(flat_grad, (0, padded_size - flat_grad.numel()))

            # Each GPU gets its portion
            for gpu_id in range(self.world_size):
                start = gpu_id * shard_size
                end = start + shard_size
                grad_shards[gpu_id][name] = flat_grad[start:end]

        return grad_shards

    def get_memory_stats(self) -> Dict[str, float]:
        """Calculate memory usage comparison."""
        total_params = sum(p.numel() for p in self.model.parameters())
        bytes_per_param = 4  # FP32

        # DDP: Full model + gradients + optimizer (Adam: 2x for m, v)
        ddp_memory = total_params * bytes_per_param * 4  # params + grads + m + v

        # FSDP: 1/N of everything
        fsdp_memory = ddp_memory / self.world_size

        return {
            'total_params': total_params,
            'ddp_memory_mb': ddp_memory / (1024 * 1024),
            'fsdp_memory_mb': fsdp_memory / (1024 * 1024),
            'memory_reduction': self.world_size
        }


def demo_fsdp_simulation():
    """Demonstrate FSDP memory sharding."""
    print("\n" + "=" * 60)
    print("FSDP Simulation")
    print("=" * 60)

    # Create a model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
    )

    fsdp = FSDPSimulator(model, world_size=4)

    print("\nParameter Sharding:")
    for name, param in model.named_parameters():
        total_size = param.numel()
        shard_size = fsdp.param_shards[0][name]['shard'].numel()
        print(f"  {name}: {total_size} total -> {shard_size} per GPU")

    print("\nMemory Statistics:")
    stats = fsdp.get_memory_stats()
    print(f"  Total parameters: {stats['total_params']:,}")
    print(f"  DDP memory per GPU: {stats['ddp_memory_mb']:.2f} MB")
    print(f"  FSDP memory per GPU: {stats['fsdp_memory_mb']:.2f} MB")
    print(f"  Memory reduction: {stats['memory_reduction']}x")

    # Simulate all-gather
    print("\nSimulating All-Gather for forward pass:")
    full_params = fsdp.all_gather_params(gpu_id=0)
    for name, param in full_params.items():
        print(f"  Reconstructed {name}: shape {param.shape}")


# ============================================================================
# Section 5: Model Parallelism
# ============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer for tensor parallelism.

    Splits weight matrix by columns:
    W = [W1 | W2 | ... | Wn]

    Each GPU computes Y_i = X @ W_i
    Output is concatenated: Y = [Y1 | Y2 | ... | Yn]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        gather_output: bool = True
    ):
        super().__init__()

        assert out_features % world_size == 0
        self.world_size = world_size
        self.gather_output = gather_output

        # Each GPU has out_features/world_size columns
        self.out_per_gpu = out_features // world_size

        # Simulate shards for each GPU
        self.weight_shards = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, self.out_per_gpu) * 0.02)
            for _ in range(world_size)
        ])
        self.bias_shards = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_per_gpu))
            for _ in range(world_size)
        ])

    def forward(self, x: torch.Tensor, gpu_id: int = 0) -> torch.Tensor:
        """
        Forward pass for a specific GPU.

        In real implementation, each GPU would only have its shard.
        Here we simulate by indexing into the shard list.
        """
        # Each GPU computes partial output
        partial_output = F.linear(
            x,
            self.weight_shards[gpu_id].t(),
            self.bias_shards[gpu_id]
        )

        if self.gather_output:
            # Simulate all-gather
            all_outputs = [
                F.linear(x, self.weight_shards[i].t(), self.bias_shards[i])
                for i in range(self.world_size)
            ]
            return torch.cat(all_outputs, dim=-1)

        return partial_output


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer for tensor parallelism.

    Splits weight matrix by rows:
    W = [W1; W2; ...; Wn]

    Input is split: X = [X1 | X2 | ... | Xn]
    Each GPU computes Y_i = X_i @ W_i
    Output is summed: Y = sum(Y_i) (all-reduce)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        input_is_parallel: bool = True
    ):
        super().__init__()

        assert in_features % world_size == 0
        self.world_size = world_size
        self.input_is_parallel = input_is_parallel

        # Each GPU has in_features/world_size rows
        self.in_per_gpu = in_features // world_size

        self.weight_shards = nn.ParameterList([
            nn.Parameter(torch.randn(self.in_per_gpu, out_features) * 0.02)
            for _ in range(world_size)
        ])
        # Only one GPU holds bias (rank 0)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, gpu_id: int = 0) -> torch.Tensor:
        """Forward pass with row parallelism."""
        if self.input_is_parallel:
            # Input already split, use local shard
            local_input = x
        else:
            # Split input
            chunks = x.chunk(self.world_size, dim=-1)
            local_input = chunks[gpu_id]

        # Local computation
        local_output = F.linear(local_input, self.weight_shards[gpu_id].t())

        # Simulate all-reduce (sum across GPUs)
        all_outputs = [
            F.linear(
                x.chunk(self.world_size, dim=-1)[i] if not self.input_is_parallel else x,
                self.weight_shards[i].t()
            )
            for i in range(self.world_size)
        ]

        output = sum(all_outputs)

        # Add bias (only on one GPU in practice)
        if gpu_id == 0:
            output = output + self.bias

        return output


class TensorParallelMLP(nn.Module):
    """
    MLP with tensor parallelism.

    Architecture:
    X -> ColumnParallel(4H) -> GeLU -> RowParallel(H) -> Output

    Communication: One all-reduce per MLP block
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int
    ):
        super().__init__()
        self.world_size = world_size

        # Up projection: column parallel (no communication needed)
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size,
            gather_output=False  # Keep split for next layer
        )

        # Down projection: row parallel (needs all-reduce)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, world_size,
            input_is_parallel=True  # Input is already split
        )

    def forward(self, x: torch.Tensor, gpu_id: int = 0) -> torch.Tensor:
        # Up projection (column parallel, no comm)
        h = self.up_proj(x, gpu_id)
        h = F.gelu(h)

        # Down projection (row parallel, all-reduce here)
        output = self.down_proj(h, gpu_id)

        return output


def demo_tensor_parallelism():
    """Demonstrate tensor parallelism concepts."""
    print("\n" + "=" * 60)
    print("Tensor Parallelism Demo")
    print("=" * 60)

    hidden_size = 256
    intermediate_size = 1024
    world_size = 4
    batch_size = 2
    seq_len = 8

    # Create tensor parallel MLP
    tp_mlp = TensorParallelMLP(hidden_size, intermediate_size, world_size)

    print(f"\nMLP Configuration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  World size (GPUs): {world_size}")

    # Calculate memory per GPU
    full_params = hidden_size * intermediate_size * 2  # up + down
    params_per_gpu = full_params // world_size

    print(f"\nParameter Distribution:")
    print(f"  Full MLP parameters: {full_params:,}")
    print(f"  Parameters per GPU: {params_per_gpu:,}")
    print(f"  Memory reduction: {world_size}x")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)

    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")

    # Each GPU computes its portion
    outputs = []
    for gpu_id in range(world_size):
        out = tp_mlp(x, gpu_id)
        outputs.append(out)
        print(f"  GPU {gpu_id} output shape: {out.shape}")

    # Verify all GPUs produce same output (after all-reduce)
    print(f"\nVerifying outputs match across GPUs: {torch.allclose(outputs[0], outputs[1])}")


# ============================================================================
# Section 6: Pipeline Parallelism Simulation
# ============================================================================

@dataclass
class PipelineStage:
    """Represents a pipeline stage (subset of model layers)."""
    stage_id: int
    layers: nn.Module
    device: str = "cpu"  # Simulated device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PipelineParallelism:
    """
    Simulates pipeline parallelism with micro-batching.

    Splits model into sequential stages, each on different GPU.
    Uses micro-batching to reduce pipeline bubble.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        num_microbatches: int = 4
    ):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches

        # Track execution schedule
        self.schedule = []

    def naive_schedule(self) -> List[Tuple[int, str, int]]:
        """
        Generate naive pipeline schedule.
        Returns list of (time_step, operation, microbatch_id)

        Naive: All forwards, then all backwards
        High bubble ratio!
        """
        schedule = []
        time = 0

        # All forwards
        for mb in range(self.num_microbatches):
            for stage in range(self.num_stages):
                schedule.append((time, f"F{stage}", mb))
                time += 1

        # All backwards (reverse order)
        for mb in range(self.num_microbatches):
            for stage in reversed(range(self.num_stages)):
                schedule.append((time, f"B{stage}", mb))
                time += 1

        return schedule

    def gpipe_schedule(self) -> List[List[Tuple[str, int]]]:
        """
        Generate GPipe schedule (per-GPU view).

        Returns: List of operations per GPU
        Each operation is (F/B + stage, microbatch_id)
        """
        schedules = [[] for _ in range(self.num_stages)]

        # Forward passes
        for mb in range(self.num_microbatches):
            for stage in range(self.num_stages):
                # Add delay based on stage
                while len(schedules[stage]) < mb + stage:
                    schedules[stage].append(("idle", -1))
                schedules[stage].append((f"F", mb))

        # Align before backward
        max_len = max(len(s) for s in schedules)
        for s in schedules:
            while len(s) < max_len:
                s.append(("idle", -1))

        # Backward passes (reverse stage order)
        for mb in range(self.num_microbatches):
            for stage in reversed(range(self.num_stages)):
                while len(schedules[stage]) < max_len + mb + (self.num_stages - 1 - stage):
                    schedules[stage].append(("idle", -1))
                schedules[stage].append((f"B", mb))

        return schedules

    def calculate_bubble_ratio(self) -> float:
        """
        Calculate pipeline bubble ratio.

        Bubble = idle time / total time
        For GPipe: bubble = (p-1) / (m + p - 1)
        """
        p = self.num_stages
        m = self.num_microbatches

        # Total time slots
        total_slots = 2 * m  # Forward + backward per microbatch

        # Bubble slots per GPU (warmup + cooldown)
        bubble_slots = p - 1  # Stages - 1

        bubble_ratio = bubble_slots / (m + p - 1)
        return bubble_ratio

    def visualize_schedule(self):
        """Visualize the pipeline schedule."""
        schedules = self.gpipe_schedule()

        print("\nPipeline Schedule Visualization:")
        print("=" * 60)

        # Find max length
        max_len = max(len(s) for s in schedules)

        # Header
        print("Time:  ", end="")
        for t in range(max_len):
            print(f"{t:>4}", end="")
        print()
        print("-" * (7 + max_len * 4))

        # Per GPU schedule
        for gpu, schedule in enumerate(schedules):
            print(f"GPU {gpu}: ", end="")
            for op, mb in schedule:
                if op == "idle":
                    print("  . ", end="")
                else:
                    print(f" {op}{mb} ", end="")
            print()

        print("-" * (7 + max_len * 4))
        print(f"Bubble ratio: {self.calculate_bubble_ratio():.1%}")


def demo_pipeline_parallelism():
    """Demonstrate pipeline parallelism."""
    print("\n" + "=" * 60)
    print("Pipeline Parallelism Demo")
    print("=" * 60)

    # Create simple stages
    stages = [
        PipelineStage(0, nn.Linear(64, 64)),
        PipelineStage(1, nn.Linear(64, 64)),
        PipelineStage(2, nn.Linear(64, 64)),
        PipelineStage(3, nn.Linear(64, 64)),
    ]

    print("\n4 Stages, 4 Micro-batches:")
    pipeline = PipelineParallelism(stages, num_microbatches=4)
    pipeline.visualize_schedule()

    print("\n4 Stages, 8 Micro-batches (better efficiency):")
    pipeline = PipelineParallelism(stages, num_microbatches=8)
    pipeline.visualize_schedule()

    print("\n4 Stages, 16 Micro-batches (even better):")
    pipeline = PipelineParallelism(stages, num_microbatches=16)
    pipeline.visualize_schedule()


# ============================================================================
# Section 7: ZeRO Stages Comparison
# ============================================================================

def calculate_memory_breakdown(
    num_params: int,
    world_size: int,
    dtype_bytes: int = 4,  # FP32
    optimizer: str = "adam"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate memory breakdown for different parallelism strategies.

    Args:
        num_params: Number of model parameters
        world_size: Number of GPUs
        dtype_bytes: Bytes per parameter (4 for FP32, 2 for FP16)
        optimizer: Optimizer type ("adam" or "sgd")

    Returns:
        Memory breakdown in MB for each strategy
    """
    # Optimizer state multiplier
    opt_mult = 2 if optimizer == "adam" else 0  # Adam has m and v

    def to_mb(bytes_val: float) -> float:
        return bytes_val / (1024 * 1024)

    results = {}

    # Single GPU
    results['single_gpu'] = {
        'params': to_mb(num_params * dtype_bytes),
        'gradients': to_mb(num_params * dtype_bytes),
        'optimizer': to_mb(num_params * dtype_bytes * opt_mult),
        'total': to_mb(num_params * dtype_bytes * (2 + opt_mult))
    }

    # DDP (Data Parallel)
    results['ddp'] = {
        'params': to_mb(num_params * dtype_bytes),
        'gradients': to_mb(num_params * dtype_bytes),
        'optimizer': to_mb(num_params * dtype_bytes * opt_mult),
        'total': to_mb(num_params * dtype_bytes * (2 + opt_mult))
    }

    # ZeRO-1 (Optimizer state partitioning)
    results['zero_1'] = {
        'params': to_mb(num_params * dtype_bytes),
        'gradients': to_mb(num_params * dtype_bytes),
        'optimizer': to_mb(num_params * dtype_bytes * opt_mult / world_size),
        'total': to_mb(num_params * dtype_bytes * (2 + opt_mult / world_size))
    }

    # ZeRO-2 (+ Gradient partitioning)
    results['zero_2'] = {
        'params': to_mb(num_params * dtype_bytes),
        'gradients': to_mb(num_params * dtype_bytes / world_size),
        'optimizer': to_mb(num_params * dtype_bytes * opt_mult / world_size),
        'total': to_mb(num_params * dtype_bytes * (1 + (1 + opt_mult) / world_size))
    }

    # ZeRO-3 / FSDP (Full sharding)
    results['zero_3_fsdp'] = {
        'params': to_mb(num_params * dtype_bytes / world_size),
        'gradients': to_mb(num_params * dtype_bytes / world_size),
        'optimizer': to_mb(num_params * dtype_bytes * opt_mult / world_size),
        'total': to_mb(num_params * dtype_bytes * (2 + opt_mult) / world_size)
    }

    return results


def demo_zero_comparison():
    """Compare ZeRO stages memory usage."""
    print("\n" + "=" * 60)
    print("ZeRO Stages Memory Comparison")
    print("=" * 60)

    # 7B parameter model
    num_params = 7_000_000_000
    world_size = 8

    print(f"\nModel: 7B parameters")
    print(f"World size: {world_size} GPUs")
    print(f"Optimizer: Adam (FP32)")

    results = calculate_memory_breakdown(num_params, world_size)

    print("\n" + "-" * 60)
    print(f"{'Strategy':<15} {'Params':>10} {'Grads':>10} {'Optim':>10} {'Total':>12}")
    print("-" * 60)

    for strategy, mem in results.items():
        print(f"{strategy:<15} {mem['params']:>10.1f} {mem['gradients']:>10.1f} "
              f"{mem['optimizer']:>10.1f} {mem['total']:>10.1f} MB")

    print("-" * 60)

    # Show reduction factors
    baseline = results['ddp']['total']
    print("\nMemory reduction vs DDP:")
    for strategy, mem in results.items():
        if strategy != 'single_gpu':
            reduction = baseline / mem['total']
            print(f"  {strategy}: {reduction:.1f}x")


# ============================================================================
# Section 8: DDP Training Template (Real Code)
# ============================================================================

DDP_TEMPLATE = '''
"""
Complete DDP Training Template

Launch with:
    torchrun --nproc_per_node=4 train_ddp.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train(rank, world_size, epochs=10):
    """Main training function."""
    setup(rank, world_size)

    # Create model and move to GPU
    model = YourModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Create dataset with distributed sampler
    dataset = YourDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for shuffling!
        model.train()

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # Gradients auto-synchronized
            optimizer.step()

            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save checkpoint (only on rank 0)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch}.pt')

        dist.barrier()  # Wait for checkpoint save

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
'''


FSDP_TEMPLATE = '''
"""
Complete FSDP Training Template

Launch with:
    torchrun --nproc_per_node=4 train_fsdp.py
"""

import os
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, DistributedSampler


def setup():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def cleanup():
    dist.destroy_process_group()


def get_fsdp_config():
    """Get FSDP configuration."""

    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    # Auto wrap policy (wrap transformer blocks)
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},  # Your transformer block class
    )

    return mp_policy, wrap_policy


def train():
    """Main training function."""
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    # Create model
    model = YourModel()

    # Get FSDP config
    mp_policy, wrap_policy = get_fsdp_config()

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device(),
    )

    # Dataset
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Micro batch size
        sampler=sampler,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()

        for batch in dataloader:
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss

            loss.backward()
            optimizer.step()

        # Save checkpoint with FSDP
        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            save_policy
        ):
            state_dict = model.state_dict()
            if rank == 0:
                torch.save(state_dict, f'checkpoint_epoch_{epoch}.pt')

        dist.barrier()

    cleanup()


if __name__ == '__main__':
    train()
'''


def print_training_templates():
    """Print DDP and FSDP training templates."""
    print("\n" + "=" * 60)
    print("DDP Training Template")
    print("=" * 60)
    print(DDP_TEMPLATE)

    print("\n" + "=" * 60)
    print("FSDP Training Template")
    print("=" * 60)
    print(FSDP_TEMPLATE)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("Module 13: Distributed Training")
    print("=" * 60)

    # 1. Collective operations
    demo_collective_operations()

    # 2. DDP simulation
    demo_ddp_training()

    # 3. Gradient bucketing
    demo_gradient_bucketing()

    # 4. FSDP simulation
    demo_fsdp_simulation()

    # 5. Tensor parallelism
    demo_tensor_parallelism()

    # 6. Pipeline parallelism
    demo_pipeline_parallelism()

    # 7. ZeRO comparison
    demo_zero_comparison()

    # 8. Print templates (commented out to avoid long output)
    # print_training_templates()

    print("\n" + "=" * 60)
    print("Module 13 Complete!")
    print("=" * 60)
    print("\nKey concepts covered:")
    print("1. Collective operations (all-reduce, all-gather, reduce-scatter)")
    print("2. DDP with gradient synchronization")
    print("3. Gradient bucketing for efficiency")
    print("4. FSDP for memory-efficient training")
    print("5. Tensor parallelism (column/row parallel)")
    print("6. Pipeline parallelism with micro-batching")
    print("7. ZeRO optimization stages")
    print("\nTo see training templates, call: print_training_templates()")


if __name__ == "__main__":
    main()
