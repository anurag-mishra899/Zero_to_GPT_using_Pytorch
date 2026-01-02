"""
Utility functions for Zero to GPT curriculum.

Provides:
- Device detection (MPS/CUDA/CPU)
- Memory-efficient defaults
- Common helper functions
"""

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Args:
        prefer_gpu: If False, always return CPU

    Returns:
        torch.device
    """
    if not prefer_gpu:
        return torch.device("cpu")

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")

    # Check CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cpu": True,
        "mps": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "recommended": str(get_device()),
    }

    if info["cuda"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)

    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()
    print(f"Available devices:")
    print(f"  CPU: {info['cpu']}")
    print(f"  MPS (Apple Silicon): {info['mps']}")
    print(f"  CUDA: {info['cuda']}")
    if info['cuda']:
        print(f"    Device: {info['cuda_device']}")
    print(f"Recommended: {info['recommended']}")


# Memory-efficient defaults for 24GB RAM Mac
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 256
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8


def get_memory_efficient_config(ram_gb: float = 24.0) -> dict:
    """
    Get memory-efficient configuration based on available RAM.

    Args:
        ram_gb: Available RAM in GB

    Returns:
        Configuration dict
    """
    if ram_gb >= 64:
        return {
            "batch_size": 16,
            "seq_len": 512,
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 8,
        }
    elif ram_gb >= 24:
        return {
            "batch_size": 4,
            "seq_len": 256,
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 8,
        }
    else:
        return {
            "batch_size": 2,
            "seq_len": 128,
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
        }


if __name__ == "__main__":
    print_device_info()
    print(f"\nMemory-efficient config: {get_memory_efficient_config()}")
