# Zero to GPT using PyTorch

A comprehensive, hands-on curriculum to understand and implement everything from PyTorch fundamentals to modern Large Language Model (LLM) architectures. Each module contains both theory (`.md`) and runnable code (`.py`).

## Overview

This repository provides a structured learning path covering:
- PyTorch fundamentals and deep learning mathematics
- Neural network building blocks (activation functions, loss functions, optimizers)
- **Tokenization** (BPE, WordPiece, SentencePiece) and text preprocessing
- Word embeddings and recurrent architectures (RNN, LSTM, GRU)
- **Language modeling fundamentals** (CLM, MLM, perplexity)
- Attention mechanisms and **positional encodings** (RoPE, ALiBi)
- Transformer architecture, **BERT**, and encoder models
- GPT architecture and **Seq2Seq models** (T5, BART)
- Modern LLM architectures and **decoding strategies**
- Training techniques (distributed training, PEFT, RLHF)
- **Training dynamics & debugging** (loss tracking, gradient analysis)
- Inference optimization (quantization, efficient attention, serving)
- **Evaluation metrics & benchmarks** (BLEU, ROUGE, LLM-as-judge)
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

### Core Modules

| Module | Topic | Description |
|--------|-------|-------------|
| **01** | PyTorch Fundamentals | Tensors, autograd, nn.Module, DataLoaders |
| **02** | Deep Learning Mathematics | Backpropagation, activation functions, loss functions, weight initialization |
| **03** | Optimization | Optimizers (SGD, Adam, AdamW), learning rate schedules |
| **04** | Regularization & Normalization | Dropout, L1/L2 regularization, BatchNorm, LayerNorm, RMSNorm |
| **05** | Word Representations | Word embeddings, Word2Vec concepts |
| **05B** | Tokenization | BPE, WordPiece, SentencePiece, tiktoken implementations |
| **05C** | Text Preprocessing | Data pipelines, datasets, collation, augmentation |
| **06** | Recurrent Architectures | RNN, LSTM, GRU implementations |
| **06B** | Language Modeling Fundamentals | CLM, MLM, perplexity, teacher forcing, training dynamics |
| **07** | Attention Mechanisms | Self-attention, scaled dot-product attention, multi-head attention |
| **07B** | Positional Encodings | Sinusoidal, Learned, RoPE, ALiBi, relative positions |
| **08** | Transformer Architecture | Full encoder-decoder transformer implementation |
| **08B** | BERT & Encoder Models | BERT architecture, MLM, NSP, fine-tuning, variants |
| **09** | GPT Architecture | Decoder-only transformers, autoregressive generation |
| **09B** | Seq2Seq: T5 & BART | Encoder-decoder models, span corruption, denoising |
| **10** | Modern LLM Architectures | LLaMA-style innovations (RoPE, RMSNorm, SwiGLU, GQA) |
| **10B** | Decoding Strategies | Greedy, beam search, top-p/top-k sampling, speculative decoding |
| **11** | Efficient Attention | Flash Attention concepts, memory-efficient implementations |
| **12** | Efficient Training | Mixed precision, gradient checkpointing, gradient accumulation |
| **13** | Distributed Training | DDP, FSDP, model parallelism concepts |
| **14** | PEFT | LoRA, QLoRA, parameter-efficient fine-tuning |
| **15** | Alignment & RLHF | Reward modeling, PPO, DPO |
| **16** | Quantization | INT8, INT4, GPTQ, AWQ concepts |
| **16B** | Training Dynamics & Debugging | Loss analysis, gradient monitoring, debugging tools |
| **17** | Serving LLMs | KV-cache, continuous batching, PagedAttention |
| **18** | Interview Prep | Common questions, system design, coding exercises |
| **18B** | Evaluation Metrics & Benchmarks | BLEU, ROUGE, perplexity, LLM-as-judge, benchmarks |

## Running the Code

### Run any module directly

```bash
# PyTorch fundamentals
python Module_01_PyTorch_Fundamentals/01_tensors_and_operations.py

# Tokenization (BPE, WordPiece, etc.)
python Module_05B_Tokenization/01_tokenization.py

# Language modeling fundamentals
python Module_06B_Language_Modeling_Fundamentals/01_language_modeling.py

# Attention mechanisms
python Module_07_Attention_Mechanisms/01_attention.py

# Positional encodings (RoPE, ALiBi)
python Module_07B_Positional_Encodings/01_positional_encodings.py

# BERT encoder models
python Module_08B_BERT_Encoder_Models/01_bert.py

# GPT implementation
python Module_09_GPT_Architecture/01_gpt.py

# Seq2Seq (T5, BART)
python Module_09B_Seq2Seq_T5_BART/01_seq2seq_t5_bart.py

# Decoding strategies
python Module_10B_Decoding_Strategies/01_decoding_strategies.py

# Training dynamics & debugging
python Module_16B_Training_Dynamics_Debugging/01_training_dynamics.py

# Evaluation metrics
python Module_18B_Evaluation_Metrics_Benchmarks/01_evaluation_metrics.py

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
├── Module_05B_Tokenization/
│   └── ... (BPE, WordPiece, SentencePiece, tiktoken)
├── Module_05C_Text_Preprocessing/
│   └── ... (data pipelines, datasets, augmentation)
├── Module_06_Recurrent_Architectures/
│   └── ... (RNN, LSTM, GRU)
├── Module_06B_Language_Modeling_Fundamentals/
│   └── ... (CLM, MLM, perplexity, teacher forcing)
├── Module_07_Attention_Mechanisms/
│   └── ... (attention implementations)
├── Module_07B_Positional_Encodings/
│   └── ... (sinusoidal, learned, RoPE, ALiBi)
├── Module_08_Transformer_Architecture/
│   └── ... (full transformer)
├── Module_08B_BERT_Encoder_Models/
│   └── ... (BERT architecture, MLM, fine-tuning)
├── Module_09_GPT_Architecture/
│   └── ... (decoder-only GPT)
├── Module_09B_Seq2Seq_T5_BART/
│   └── ... (encoder-decoder, T5, BART)
├── Module_10_Modern_LLM_Architectures/
│   └── ... (LLaMA-style models)
├── Module_10B_Decoding_Strategies/
│   └── ... (greedy, beam search, sampling, speculative)
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
├── Module_16B_Training_Dynamics_Debugging/
│   └── ... (loss tracking, gradient analysis, debugging)
├── Module_17_Serving_LLMs/
│   └── ... (inference optimization)
├── Module_18_Interview_Prep/
│   └── ... (interview guide and code)
└── Module_18B_Evaluation_Metrics_Benchmarks/
    └── ... (BLEU, ROUGE, perplexity, benchmarks)
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
