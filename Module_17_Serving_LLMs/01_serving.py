"""
Module 17: Serving LLMs in Production - Implementation

This module covers:
1. Batching strategies simulation
2. KV-Cache management
3. PagedAttention concepts
4. Continuous batching
5. Request scheduling
6. Performance estimation
"""

import time
import heapq
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Section 1: Data Structures
# ============================================================================

class RequestStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class GenerationRequest:
    """Represents a single generation request."""
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING

    # Generated tokens
    output_tokens: List[int] = field(default_factory=list)

    # Timing
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    # KV-cache info
    kv_cache_blocks: List[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        """Total tokens (prompt + generated)."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        return len(self.output_tokens) >= self.max_new_tokens

    @property
    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def total_latency(self) -> Optional[float]:
        if self.finish_time:
            return self.finish_time - self.arrival_time
        return None


@dataclass
class BatchMetrics:
    """Metrics for a batch of requests."""
    batch_size: int
    total_tokens: int
    prefill_tokens: int
    decode_tokens: int
    latency_ms: float


# ============================================================================
# Section 2: KV-Cache Block Manager (PagedAttention)
# ============================================================================

@dataclass
class KVBlock:
    """A block of KV-cache memory."""
    block_id: int
    num_tokens: int = 0
    max_tokens: int = 16  # Tokens per block
    ref_count: int = 0  # For copy-on-write

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.max_tokens

    @property
    def free_slots(self) -> int:
        return self.max_tokens - self.num_tokens


class BlockManager:
    """
    Manages KV-cache blocks for PagedAttention.

    Implements:
    - Block allocation/deallocation
    - Copy-on-write for beam search
    - Memory usage tracking
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Free and allocated blocks
        self.free_blocks: List[int] = list(range(num_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}  # request_id -> blocks

        # Block metadata
        self.blocks: Dict[int, KVBlock] = {
            i: KVBlock(block_id=i, max_tokens=block_size)
            for i in range(num_blocks)
        }

    def allocate_block(self, request_id: str) -> Optional[int]:
        """Allocate a new block for a request."""
        if not self.free_blocks:
            return None

        block_id = self.free_blocks.pop()
        block = self.blocks[block_id]
        block.num_tokens = 0
        block.ref_count = 1

        if request_id not in self.allocated_blocks:
            self.allocated_blocks[request_id] = []
        self.allocated_blocks[request_id].append(block_id)

        return block_id

    def free_request(self, request_id: str):
        """Free all blocks for a request."""
        if request_id not in self.allocated_blocks:
            return

        for block_id in self.allocated_blocks[request_id]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block_id)
                block.num_tokens = 0

        del self.allocated_blocks[request_id]

    def get_num_required_blocks(self, num_tokens: int) -> int:
        """Calculate blocks needed for given number of tokens."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we can allocate blocks for given tokens."""
        required = self.get_num_required_blocks(num_tokens)
        return len(self.free_blocks) >= required

    def allocate_for_request(self, request_id: str, num_tokens: int) -> bool:
        """Allocate all blocks needed for a request."""
        required = self.get_num_required_blocks(num_tokens)
        if not self.can_allocate(num_tokens):
            return False

        for _ in range(required):
            self.allocate_block(request_id)
        return True

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        allocated = self.num_blocks - len(self.free_blocks)
        bytes_per_block = (
            self.block_size *
            self.num_layers *
            self.num_heads *
            self.head_dim *
            2 *  # K and V
            2    # bytes (FP16)
        )

        return {
            'total_blocks': self.num_blocks,
            'allocated_blocks': allocated,
            'free_blocks': len(self.free_blocks),
            'utilization': allocated / self.num_blocks,
            'memory_per_block_mb': bytes_per_block / 1024 / 1024,
            'total_allocated_mb': allocated * bytes_per_block / 1024 / 1024,
        }


# ============================================================================
# Section 3: Static vs Continuous Batching
# ============================================================================

class StaticBatcher:
    """
    Traditional static batching.

    Waits for batch to fill, processes all together.
    """

    def __init__(self, batch_size: int = 8, timeout_ms: float = 100):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.queue: List[GenerationRequest] = []

    def add_request(self, request: GenerationRequest):
        """Add request to queue."""
        self.queue.append(request)

    def get_batch(self) -> List[GenerationRequest]:
        """Get batch when ready (size or timeout)."""
        if len(self.queue) >= self.batch_size:
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            return batch
        return []

    def process_batch(self, batch: List[GenerationRequest]) -> BatchMetrics:
        """Process a batch (simulation)."""
        if not batch:
            return BatchMetrics(0, 0, 0, 0, 0)

        # Find max length for padding
        max_len = max(r.num_tokens for r in batch)
        total_tokens = max_len * len(batch)  # Padded

        # Simulate processing time
        latency_ms = total_tokens * 0.01  # Simplified

        return BatchMetrics(
            batch_size=len(batch),
            total_tokens=total_tokens,
            prefill_tokens=sum(len(r.prompt_tokens) for r in batch),
            decode_tokens=sum(len(r.output_tokens) for r in batch),
            latency_ms=latency_ms
        )


class ContinuousBatcher:
    """
    Continuous batching (iteration-level batching).

    Processes requests at each decode step, no waiting.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
        block_manager: Optional[BlockManager] = None
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.block_manager = block_manager

        self.waiting_queue: deque = deque()
        self.running_batch: List[GenerationRequest] = []

    def add_request(self, request: GenerationRequest):
        """Add request to waiting queue."""
        request.status = RequestStatus.PENDING
        self.waiting_queue.append(request)

    def schedule_step(self) -> Tuple[List[GenerationRequest], List[GenerationRequest]]:
        """
        Schedule one iteration step.

        Returns:
            prefill_requests: New requests to prefill
            decode_requests: Existing requests to decode
        """
        # Remove finished requests
        self.running_batch = [
            r for r in self.running_batch
            if not r.is_finished and r.status == RequestStatus.RUNNING
        ]

        # Free blocks for finished requests
        if self.block_manager:
            for r in list(self.running_batch):
                if r.is_finished:
                    self.block_manager.free_request(r.request_id)

        # Current batch stats
        current_tokens = sum(r.num_tokens for r in self.running_batch)

        # Add new requests if possible
        prefill_requests = []
        while self.waiting_queue:
            request = self.waiting_queue[0]
            new_tokens = len(request.prompt_tokens) + request.max_new_tokens

            # Check constraints
            if len(self.running_batch) + 1 > self.max_batch_size:
                break
            if current_tokens + new_tokens > self.max_tokens_per_batch:
                break

            # Check memory
            if self.block_manager and not self.block_manager.can_allocate(new_tokens):
                break

            # Add to batch
            self.waiting_queue.popleft()
            request.status = RequestStatus.RUNNING
            self.running_batch.append(request)
            prefill_requests.append(request)

            # Allocate KV-cache blocks
            if self.block_manager:
                self.block_manager.allocate_for_request(request.request_id, new_tokens)

            current_tokens += new_tokens

        return prefill_requests, self.running_batch

    def process_step(
        self,
        prefill_requests: List[GenerationRequest],
        decode_requests: List[GenerationRequest]
    ) -> BatchMetrics:
        """
        Process one decode step (simulation).

        In real implementation, this would call the model.
        """
        now = time.time()

        # Simulate token generation
        for request in decode_requests:
            if not request.is_finished:
                # Generate one token (simulated)
                new_token = random.randint(1, 1000)
                request.output_tokens.append(new_token)

                if request.first_token_time is None:
                    request.first_token_time = now

                if request.is_finished:
                    request.status = RequestStatus.FINISHED
                    request.finish_time = now

                    # Free memory
                    if self.block_manager:
                        self.block_manager.free_request(request.request_id)

        # Calculate metrics
        prefill_tokens = sum(len(r.prompt_tokens) for r in prefill_requests)
        decode_tokens = len([r for r in decode_requests if not r.is_finished])

        return BatchMetrics(
            batch_size=len(decode_requests),
            total_tokens=prefill_tokens + decode_tokens,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            latency_ms=0.5  # Simulated
        )


# ============================================================================
# Section 4: Request Scheduler
# ============================================================================

class SchedulerPolicy(Enum):
    FCFS = "fcfs"  # First Come First Served
    SHORTEST_FIRST = "shortest_first"
    PRIORITY = "priority"


class RequestScheduler:
    """
    Scheduler for LLM serving.

    Manages request queue, prioritization, and preemption.
    """

    def __init__(
        self,
        policy: SchedulerPolicy = SchedulerPolicy.FCFS,
        max_batch_size: int = 32,
        max_batch_tokens: int = 4096,
        block_manager: Optional[BlockManager] = None
    ):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.block_manager = block_manager

        self.waiting: List[GenerationRequest] = []
        self.running: List[GenerationRequest] = []
        self.preempted: List[GenerationRequest] = []

    def add_request(self, request: GenerationRequest):
        """Add new request to scheduler."""
        request.status = RequestStatus.PENDING
        self._insert_waiting(request)

    def _insert_waiting(self, request: GenerationRequest):
        """Insert request into waiting queue based on policy."""
        if self.policy == SchedulerPolicy.FCFS:
            self.waiting.append(request)
        elif self.policy == SchedulerPolicy.SHORTEST_FIRST:
            # Sort by prompt length
            self.waiting.append(request)
            self.waiting.sort(key=lambda r: len(r.prompt_tokens))

    def schedule(self) -> Tuple[List[GenerationRequest], List[GenerationRequest]]:
        """
        Schedule next batch.

        Returns:
            prefill: Requests needing prefill
            decode: Requests in decode phase
        """
        # Move finished to done
        self.running = [r for r in self.running if not r.is_finished]

        current_tokens = sum(r.num_tokens for r in self.running)
        prefill = []

        # Try to add waiting requests
        for request in list(self.waiting):
            new_tokens = len(request.prompt_tokens)

            if len(self.running) + len(prefill) >= self.max_batch_size:
                break
            if current_tokens + new_tokens > self.max_batch_tokens:
                break

            # Check memory
            if self.block_manager:
                needed = len(request.prompt_tokens) + request.max_new_tokens
                if not self.block_manager.can_allocate(needed):
                    continue

            # Add to batch
            self.waiting.remove(request)
            request.status = RequestStatus.RUNNING
            prefill.append(request)
            self.running.append(request)
            current_tokens += new_tokens

        return prefill, self.running

    def preempt_request(self, request: GenerationRequest):
        """Preempt a running request (move to waiting)."""
        if request in self.running:
            self.running.remove(request)
            request.status = RequestStatus.PREEMPTED
            self.preempted.append(request)

            if self.block_manager:
                self.block_manager.free_request(request.request_id)

    def get_stats(self) -> Dict[str, int]:
        """Get scheduler statistics."""
        return {
            'waiting': len(self.waiting),
            'running': len(self.running),
            'preempted': len(self.preempted),
        }


# ============================================================================
# Section 5: Performance Estimation
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration for performance estimation."""
    num_params: int  # Total parameters
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    vocab_size: int
    dtype_bytes: int = 2  # FP16


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gpu_memory_gb: float
    memory_bandwidth_gb_s: float
    compute_tflops: float


def estimate_memory_usage(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int
) -> Dict[str, float]:
    """
    Estimate GPU memory usage.

    Components:
    1. Model weights
    2. KV-cache
    3. Activations
    4. Overhead
    """
    # Model weights
    model_memory = model_config.num_params * model_config.dtype_bytes

    # KV-cache per sequence
    kv_per_token = (
        2 *  # K and V
        model_config.num_layers *
        model_config.num_heads *
        model_config.head_dim *
        model_config.dtype_bytes
    )
    kv_cache = batch_size * seq_len * kv_per_token

    # Activations (rough estimate)
    activation_per_token = (
        model_config.hidden_size *
        model_config.num_layers *
        model_config.dtype_bytes *
        2  # Approximate factor
    )
    activations = batch_size * seq_len * activation_per_token

    # Overhead (10%)
    overhead = (model_memory + kv_cache + activations) * 0.1

    total = model_memory + kv_cache + activations + overhead

    return {
        'model_gb': model_memory / 1e9,
        'kv_cache_gb': kv_cache / 1e9,
        'activations_gb': activations / 1e9,
        'overhead_gb': overhead / 1e9,
        'total_gb': total / 1e9,
    }


def estimate_throughput(
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    batch_size: int,
    context_len: int
) -> Dict[str, float]:
    """
    Estimate throughput for generation.

    Key insight: LLM inference is memory-bound for small batches.
    """
    # Bytes to read per token (model weights)
    bytes_per_token = model_config.num_params * model_config.dtype_bytes

    # Time per token (memory-bound)
    time_per_token_s = bytes_per_token / (hardware_config.memory_bandwidth_gb_s * 1e9)

    # Tokens per second
    tokens_per_second = 1 / time_per_token_s

    # With batching (amortize memory reads)
    effective_tps = tokens_per_second * min(batch_size, 8)  # Diminishing returns

    return {
        'tokens_per_second': effective_tps,
        'time_per_token_ms': 1000 / effective_tps,
        'requests_per_second': effective_tps / context_len,
    }


def estimate_latency(
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    prompt_len: int,
    output_len: int
) -> Dict[str, float]:
    """
    Estimate latency for a single request.

    TTFT: Time to first token (prefill)
    ITL: Inter-token latency (decode)
    E2E: End-to-end latency
    """
    # Prefill: compute bound (process all prompt tokens)
    prefill_flops = 2 * model_config.num_params * prompt_len
    prefill_time = prefill_flops / (hardware_config.compute_tflops * 1e12)

    # Decode: memory bound (one token at a time)
    bytes_per_decode = model_config.num_params * model_config.dtype_bytes
    decode_time_per_token = bytes_per_decode / (hardware_config.memory_bandwidth_gb_s * 1e9)

    total_decode_time = decode_time_per_token * output_len

    return {
        'ttft_ms': prefill_time * 1000,
        'itl_ms': decode_time_per_token * 1000,
        'decode_total_ms': total_decode_time * 1000,
        'e2e_ms': (prefill_time + total_decode_time) * 1000,
    }


# ============================================================================
# Section 6: Speculative Decoding Simulation
# ============================================================================

class SpeculativeDecoder:
    """
    Simulates speculative decoding.

    Draft model generates k tokens quickly.
    Target model verifies all k in parallel.
    Accept matching prefix.
    """

    def __init__(
        self,
        draft_acceptance_rate: float = 0.7,
        draft_tokens_k: int = 5
    ):
        self.acceptance_rate = draft_acceptance_rate
        self.k = draft_tokens_k

    def simulate_step(self) -> Tuple[int, Dict[str, Any]]:
        """
        Simulate one speculative decoding step.

        Returns:
            accepted_tokens: Number of tokens accepted
            metrics: Step metrics
        """
        # Draft generates k tokens
        draft_tokens = self.k

        # Simulate acceptance (binomial)
        accepted = 0
        for i in range(draft_tokens):
            if random.random() < self.acceptance_rate:
                accepted += 1
            else:
                # First rejection, stop accepting
                break

        # Always get at least 1 token (from target model)
        total_tokens = max(accepted, 1)

        return total_tokens, {
            'draft_tokens': draft_tokens,
            'accepted_tokens': accepted,
            'total_tokens': total_tokens,
            'acceptance_rate': accepted / draft_tokens if draft_tokens > 0 else 0,
        }

    def estimate_speedup(self, num_iterations: int = 1000) -> Dict[str, float]:
        """
        Estimate speedup from speculative decoding.

        Assumes draft model is ~10x faster than target.
        """
        total_tokens = 0
        total_iterations = 0

        for _ in range(num_iterations):
            tokens, _ = self.simulate_step()
            total_tokens += tokens
            total_iterations += 1

        avg_tokens_per_step = total_tokens / total_iterations

        # Speedup calculation
        # Without speculation: 1 token per target model call
        # With speculation: avg_tokens_per_step tokens per (draft_k + 1) model calls
        # Assuming draft is 10x faster, effective calls = draft_k/10 + 1

        draft_cost = self.k / 10  # Draft is 10x faster
        target_cost = 1
        total_cost = draft_cost + target_cost

        speedup = avg_tokens_per_step / total_cost

        return {
            'avg_tokens_per_step': avg_tokens_per_step,
            'effective_acceptance_rate': (avg_tokens_per_step - 1) / self.k,
            'estimated_speedup': speedup,
        }


# ============================================================================
# Section 7: Demonstration Functions
# ============================================================================

def demo_block_manager():
    """Demonstrate PagedAttention block management."""
    print("=" * 60)
    print("PagedAttention Block Manager Demo")
    print("=" * 60)

    # Create block manager
    bm = BlockManager(
        num_blocks=100,
        block_size=16,
        num_layers=32,
        num_heads=32,
        head_dim=128
    )

    print(f"\nInitial state:")
    print(f"  Total blocks: {bm.num_blocks}")
    print(f"  Memory per block: {bm.get_memory_usage()['memory_per_block_mb']:.2f} MB")

    # Allocate for some requests
    requests = [
        ("req_1", 100),   # 100 tokens
        ("req_2", 500),   # 500 tokens
        ("req_3", 50),    # 50 tokens
    ]

    for req_id, num_tokens in requests:
        success = bm.allocate_for_request(req_id, num_tokens)
        blocks_needed = bm.get_num_required_blocks(num_tokens)
        print(f"\nAllocating {num_tokens} tokens for {req_id}: "
              f"{blocks_needed} blocks, success={success}")

    usage = bm.get_memory_usage()
    print(f"\nMemory usage:")
    print(f"  Allocated: {usage['allocated_blocks']} / {usage['total_blocks']} blocks")
    print(f"  Utilization: {usage['utilization']:.1%}")
    print(f"  Memory used: {usage['total_allocated_mb']:.2f} MB")

    # Free a request
    print("\nFreeing req_2...")
    bm.free_request("req_2")
    usage = bm.get_memory_usage()
    print(f"  Utilization after free: {usage['utilization']:.1%}")


def demo_continuous_batching():
    """Demonstrate continuous vs static batching."""
    print("\n" + "=" * 60)
    print("Continuous vs Static Batching Demo")
    print("=" * 60)

    # Create requests with varying lengths
    requests = []
    for i in range(20):
        req = GenerationRequest(
            request_id=f"req_{i}",
            prompt_tokens=list(range(random.randint(10, 100))),
            max_new_tokens=random.randint(20, 100)
        )
        requests.append(req)

    # Static batching simulation
    print("\nStatic Batching (batch_size=8):")
    static = StaticBatcher(batch_size=8)
    for req in requests[:8]:
        static.add_request(req)

    batch = static.get_batch()
    metrics = static.process_batch(batch)
    print(f"  Batch size: {metrics.batch_size}")
    print(f"  Total tokens (with padding): {metrics.total_tokens}")
    actual_tokens = sum(len(r.prompt_tokens) for r in batch)
    print(f"  Actual tokens: {actual_tokens}")
    print(f"  Padding overhead: {(metrics.total_tokens - actual_tokens) / metrics.total_tokens:.1%}")

    # Continuous batching simulation
    print("\nContinuous Batching:")
    continuous = ContinuousBatcher(max_batch_size=32, max_tokens_per_batch=2048)

    for req in requests:
        req_copy = GenerationRequest(
            request_id=req.request_id,
            prompt_tokens=req.prompt_tokens.copy(),
            max_new_tokens=req.max_new_tokens
        )
        continuous.add_request(req_copy)

    # Run a few steps
    total_tokens_generated = 0
    for step in range(10):
        prefill, decode = continuous.schedule_step()
        metrics = continuous.process_step(prefill, decode)
        total_tokens_generated += metrics.decode_tokens

    print(f"  Requests processed: {len([r for r in continuous.running_batch if r.is_finished])}")
    print(f"  Total tokens generated: {total_tokens_generated}")
    print(f"  No padding overhead!")


def demo_performance_estimation():
    """Demonstrate performance estimation."""
    print("\n" + "=" * 60)
    print("Performance Estimation Demo")
    print("=" * 60)

    # LLaMA-7B config
    model = ModelConfig(
        num_params=7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        head_dim=128,
        vocab_size=32000
    )

    # A100-80GB
    hardware = HardwareConfig(
        gpu_memory_gb=80,
        memory_bandwidth_gb_s=2000,
        compute_tflops=312
    )

    print(f"\nModel: LLaMA-7B ({model.num_params/1e9:.0f}B params)")
    print(f"Hardware: A100-80GB")

    # Memory estimation
    print("\nMemory Usage (batch=8, seq=2048):")
    mem = estimate_memory_usage(model, batch_size=8, seq_len=2048)
    for k, v in mem.items():
        print(f"  {k}: {v:.2f} GB")

    # Throughput estimation
    print("\nThroughput Estimation:")
    for batch_size in [1, 4, 8, 16]:
        throughput = estimate_throughput(model, hardware, batch_size, context_len=512)
        print(f"  Batch {batch_size}: {throughput['tokens_per_second']:.0f} tokens/s, "
              f"{throughput['requests_per_second']:.2f} req/s")

    # Latency estimation
    print("\nLatency Estimation (prompt=256, output=128):")
    latency = estimate_latency(model, hardware, prompt_len=256, output_len=128)
    print(f"  TTFT: {latency['ttft_ms']:.1f} ms")
    print(f"  ITL: {latency['itl_ms']:.2f} ms")
    print(f"  E2E: {latency['e2e_ms']:.0f} ms")


def demo_speculative_decoding():
    """Demonstrate speculative decoding."""
    print("\n" + "=" * 60)
    print("Speculative Decoding Demo")
    print("=" * 60)

    for acceptance_rate in [0.5, 0.7, 0.9]:
        spec = SpeculativeDecoder(
            draft_acceptance_rate=acceptance_rate,
            draft_tokens_k=5
        )

        results = spec.estimate_speedup(num_iterations=1000)
        print(f"\nAcceptance rate: {acceptance_rate:.0%}")
        print(f"  Avg tokens per step: {results['avg_tokens_per_step']:.2f}")
        print(f"  Estimated speedup: {results['estimated_speedup']:.2f}x")


def demo_memory_vs_batch():
    """Show memory vs batch size trade-off."""
    print("\n" + "=" * 60)
    print("Memory vs Batch Size Trade-off")
    print("=" * 60)

    model = ModelConfig(
        num_params=70_000_000_000,  # 70B
        num_layers=80,
        hidden_size=8192,
        num_heads=64,
        head_dim=128,
        vocab_size=32000
    )

    gpu_memory = 80  # GB

    print(f"\nModel: LLaMA-70B, GPU: A100-80GB")
    print(f"\n{'Batch':<8} {'Seq Len':<10} {'Memory':<12} {'Fits?':<8}")
    print("-" * 40)

    for batch_size in [1, 2, 4, 8, 16]:
        for seq_len in [512, 1024, 2048, 4096]:
            mem = estimate_memory_usage(model, batch_size, seq_len)
            fits = "Yes" if mem['total_gb'] < gpu_memory else "No"
            print(f"{batch_size:<8} {seq_len:<10} {mem['total_gb']:<12.1f} {fits:<8}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("Module 17: Serving LLMs in Production")
    print("=" * 60)

    demo_block_manager()
    demo_continuous_batching()
    demo_performance_estimation()
    demo_speculative_decoding()
    demo_memory_vs_batch()

    print("\n" + "=" * 60)
    print("Module 17 Complete!")
    print("=" * 60)
    print("\nKey concepts covered:")
    print("1. PagedAttention block management")
    print("2. Continuous vs static batching")
    print("3. Request scheduling")
    print("4. Performance estimation (memory, throughput, latency)")
    print("5. Speculative decoding")
    print("6. Memory vs batch size trade-offs")


if __name__ == "__main__":
    main()
