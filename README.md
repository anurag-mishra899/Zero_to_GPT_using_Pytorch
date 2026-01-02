# Zero to GPT using PyTorch

A comprehensive, hands-on curriculum to understand and implement everything from PyTorch fundamentals to modern Large Language Model (LLM) architectures. Each module contains both theory (`.md`) and runnable code (`.py`).

## Overview

This repository provides a structured learning path covering:
- PyTorch fundamentals and deep learning mathematics
- Neural network building blocks (activation functions, loss functions, optimizers)
- Word embeddings and recurrent architectures (RNN, LSTM, GRU)
- Attention mechanisms and the Transformer architecture
- GPT and modern LLM architectures
- Training techniques (distributed training, PEFT, RLHF)
- Inference optimization (quantization, efficient attention, serving)
- Interview preparation for ML/LLM roles

## Prerequisites

- Python 3.10+
- Basic understanding of linear algebra and calculus
- Familiarity with Python programming

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Zero_to_GPT_using_Pytorch.git
cd Zero_to_GPT_using_Pytorch
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Run the setup script

```bash
python setup.py
```

This will:
- Install required packages (PyTorch, torchvision, numpy)
- Verify your installation
- Detect available devices (CPU, CUDA, MPS for Apple Silicon)
- Run quick functionality tests

## Hardware Support

The code is optimized to run on:
- **CPU**: Works everywhere (slower for large models)
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2/M3/M4) - uses Metal Performance Shaders

The setup script automatically detects and recommends the best available device.

## Module Overview

| Module | Topic | Description |
|--------|-------|-------------|
| **01** | PyTorch Fundamentals | Tensors, autograd, nn.Module, DataLoaders |
| **02** | Deep Learning Mathematics | Backpropagation, activation functions, loss functions, weight initialization |
| **03** | Optimization | Optimizers (SGD, Adam, AdamW), learning rate schedules |
| **04** | Regularization & Normalization | Dropout, L1/L2 regularization, BatchNorm, LayerNorm, RMSNorm |
| **05** | Word Representations | Word embeddings, Word2Vec concepts |
| **06** | Recurrent Architectures | RNN, LSTM, GRU implementations |
| **07** | Attention Mechanisms | Self-attention, scaled dot-product attention, multi-head attention |
| **08** | Transformer Architecture | Full encoder-decoder transformer implementation |
| **09** | GPT Architecture | Decoder-only transformers, autoregressive generation |
| **10** | Modern LLM Architectures | LLaMA-style innovations (RoPE, RMSNorm, SwiGLU, GQA) |
| **11** | Efficient Attention | Flash Attention concepts, memory-efficient implementations |
| **12** | Efficient Training | Mixed precision, gradient checkpointing, gradient accumulation |
| **13** | Distributed Training | DDP, FSDP, model parallelism concepts |
| **14** | PEFT | LoRA, QLoRA, parameter-efficient fine-tuning |
| **15** | Alignment & RLHF | Reward modeling, PPO, DPO |
| **16** | Quantization | INT8, INT4, GPTQ, AWQ concepts |
| **17** | Serving LLMs | KV-cache, continuous batching, PagedAttention |
| **18** | Interview Prep | Common questions, system design, coding exercises |

## Running the Code

### Run any module directly

```bash
# PyTorch fundamentals
python Module_01_PyTorch_Fundamentals/01_tensors_and_operations.py

# Attention mechanisms
python Module_07_Attention_Mechanisms/01_attention.py

# GPT implementation
python Module_09_GPT_Architecture/01_gpt.py

# Interview prep code
python Module_18_Interview_Prep/01_interview_code.py
```

### Study approach

Each module contains:
1. **Theory file** (`.md`): Comprehensive explanations, diagrams, and interview questions
2. **Code file** (`.py`): Runnable implementations with comments

**Recommended workflow:**
1. Read the theory (`.md`) file first
2. Run the code (`.py`) and experiment with parameters
3. Review the interview questions at the end of each theory file

## Project Structure

```
Zero_to_GPT_using_Pytorch/
├── setup.py                    # Environment setup and verification
├── utils.py                    # Shared utilities (device detection, configs)
├── Module_01_PyTorch_Fundamentals/
│   ├── 01_tensors_and_operations.md
│   ├── 01_tensors_and_operations.py
│   ├── 02_autograd_computational_graphs.md
│   ├── 02_autograd_computational_graphs.py
│   ├── 03_nn_module_and_parameters.md
│   ├── 03_nn_module_and_parameters.py
│   ├── 04_datasets_and_dataloaders.md
│   └── 04_datasets_and_dataloaders.py
├── Module_02_Deep_Learning_Mathematics/
│   └── ... (backpropagation, activations, loss, initialization)
├── Module_03_Optimization/
│   └── ... (optimizers, learning rate schedules)
├── Module_04_Regularization_Normalization/
│   └── ... (regularization, normalization techniques)
├── Module_05_Word_Representations/
│   └── ... (word embeddings)
├── Module_06_Recurrent_Architectures/
│   └── ... (RNN, LSTM, GRU)
├── Module_07_Attention_Mechanisms/
│   └── ... (attention implementations)
├── Module_08_Transformer_Architecture/
│   └── ... (full transformer)
├── Module_09_GPT_Architecture/
│   └── ... (decoder-only GPT)
├── Module_10_Modern_LLM_Architectures/
│   └── ... (LLaMA-style models)
├── Module_11_Efficient_Attention/
│   └── ... (Flash Attention concepts)
├── Module_12_Efficient_Training/
│   └── ... (mixed precision, checkpointing)
├── Module_13_Distributed_Training/
│   └── ... (DDP, FSDP)
├── Module_14_PEFT/
│   └── ... (LoRA, QLoRA)
├── Module_15_Alignment_RLHF/
│   └── ... (RLHF, DPO)
├── Module_16_Quantization/
│   └── ... (quantization techniques)
├── Module_17_Serving_LLMs/
│   └── ... (inference optimization)
└── Module_18_Interview_Prep/
    └── ... (interview guide and code)
```

## Key Features

- **From-scratch implementations**: All core components (attention, transformers, GPT) implemented from scratch
- **Interview-focused**: Each module includes interview questions with detailed answers
- **Device-agnostic**: Automatically uses the best available hardware (CUDA/MPS/CPU)
- **Memory-efficient defaults**: Configured for systems with 24GB+ RAM
- **Comprehensive coverage**: From basic tensors to production LLM serving

## Memory Requirements

The code uses memory-efficient defaults suitable for:
- **24GB+ RAM**: Full examples run smoothly
- **16GB RAM**: Most examples work; reduce batch sizes if needed
- **8GB RAM**: Basic modules work; skip larger model examples

You can adjust model sizes in `utils.py`:

```python
# Default configuration for 24GB RAM
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 256
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
```

## Target Audience

- ML engineers preparing for LLM-focused interviews
- Developers wanting to understand transformer architectures deeply
- Researchers looking for clean reference implementations
- Students learning deep learning and NLP

## Dependencies

Core dependencies (installed via `setup.py`):
- `torch` - PyTorch deep learning framework
- `torchvision` - For some dataset utilities
- `numpy` - Numerical computing

## Troubleshooting

### CUDA not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### MPS (Apple Silicon) issues
```bash
# Verify MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Out of memory
- Reduce `batch_size` in the scripts
- Reduce `seq_len` or `d_model` parameters
- Use `torch.cuda.empty_cache()` between experiments

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is for educational purposes.

## Acknowledgments

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3 paper)
- "LLaMA: Open and Efficient Foundation Language Models"
- PyTorch documentation and tutorials
