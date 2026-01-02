"""
Setup script for Zero to GPT curriculum.

Run this first to install dependencies and verify your environment.

Usage:
    python setup.py

For MacBook Pro M4 Pro (Apple Silicon):
    - Uses MPS (Metal Performance Shaders) backend
    - Optimized for 24GB RAM
"""

import subprocess
import sys


def install_requirements():
    """Install required packages."""
    packages = [
        "torch",
        "torchvision",
        "numpy",
    ]

    print("Installing required packages...")
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "-q"
        ])

    print("\nAll packages installed!")


def verify_installation():
    """Verify PyTorch installation and device availability."""
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")

        # Check available devices
        print("\nDevice availability:")

        # CPU (always available)
        print(f"  CPU: Available")

        # MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  MPS (Apple Silicon): Available")
            device = torch.device("mps")
            # Quick test
            x = torch.randn(100, 100, device=device)
            y = x @ x.T
            print(f"  MPS test: Passed")
        else:
            print(f"  MPS (Apple Silicon): Not available")

        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            print(f"  CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  CUDA: Not available")

        # Recommended device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            recommended = "mps"
        elif torch.cuda.is_available():
            recommended = "cuda"
        else:
            recommended = "cpu"

        print(f"\nRecommended device: {recommended}")

        # Memory info
        import os
        if sys.platform == 'darwin':
            # macOS - get system memory
            try:
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                       capture_output=True, text=True)
                mem_bytes = int(result.stdout.strip())
                mem_gb = mem_bytes / (1024**3)
                print(f"System memory: {mem_gb:.1f} GB")
            except:
                pass

        return True

    except ImportError:
        print("PyTorch not found. Please run: pip install torch")
        return False


def run_quick_test():
    """Run a quick test of core functionality."""
    import torch
    import torch.nn as nn

    print("\n" + "=" * 50)
    print("Running quick functionality test...")
    print("=" * 50)

    # Determine device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Test 1: Basic tensor operations
    print("\n1. Testing tensor operations...")
    x = torch.randn(64, 128, device=device)
    y = torch.randn(128, 64, device=device)
    z = x @ y
    print(f"   Matrix multiply: ({x.shape}) @ ({y.shape}) = {z.shape} ✓")

    # Test 2: Simple neural network
    print("\n2. Testing neural network...")
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64)
    ).to(device)

    input_tensor = torch.randn(4, 128, device=device)
    output = model(input_tensor)
    print(f"   Forward pass: {input_tensor.shape} -> {output.shape} ✓")

    # Test 3: Backward pass
    print("\n3. Testing backward pass...")
    loss = output.sum()
    loss.backward()
    print(f"   Gradients computed ✓")

    # Test 4: Attention-like operation
    print("\n4. Testing attention operation...")
    batch, seq, dim = 2, 16, 64
    q = torch.randn(batch, seq, dim, device=device)
    k = torch.randn(batch, seq, dim, device=device)
    v = torch.randn(batch, seq, dim, device=device)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    print(f"   Attention: Q,K,V {q.shape} -> {out.shape} ✓")

    print("\n" + "=" * 50)
    print("All tests passed! Your system is ready.")
    print("=" * 50)


def main():
    print("=" * 50)
    print("Zero to GPT - Environment Setup")
    print("=" * 50)

    # Check if torch is installed
    try:
        import torch
        print("\nPyTorch already installed.")
    except ImportError:
        print("\nPyTorch not found. Installing...")
        install_requirements()

    # Verify installation
    if verify_installation():
        run_quick_test()

        print("\n" + "=" * 50)
        print("SETUP COMPLETE!")
        print("=" * 50)
        print("\nYou can now run any module, for example:")
        print("  python Module_01_PyTorch_Fundamentals/01_tensors_and_operations.py")
        print("  python Module_07_Attention_Mechanisms/01_attention.py")
        print("  python Module_18_Interview_Prep/01_interview_code.py")
    else:
        print("\nSetup failed. Please install PyTorch manually:")
        print("  pip install torch torchvision")


if __name__ == "__main__":
    main()
