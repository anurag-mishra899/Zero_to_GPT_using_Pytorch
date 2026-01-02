# Module 1.3: nn.Module, Parameters, and Building Networks

## Table of Contents
1. [Understanding nn.Module](#1-understanding-nnmodule)
2. [Parameters vs Buffers](#2-parameters-vs-buffers)
3. [Building Neural Networks](#3-building-neural-networks)
4. [Layer Types Deep Dive](#4-layer-types-deep-dive)
5. [Weight Initialization](#5-weight-initialization)
6. [Model Inspection and Manipulation](#6-model-inspection-and-manipulation)
7. [Saving and Loading Models](#7-saving-and-loading-models)
8. [Interview Questions](#8-interview-questions)
9. [Summary](#9-summary)

---

## 1. Understanding nn.Module

### 1.1 What is nn.Module?

`nn.Module` is the base class for all neural network modules in PyTorch. It provides:

1. **Parameter management**: Automatically tracks learnable parameters
2. **Device management**: Easy `.to(device)` for all parameters
3. **Train/eval modes**: `.train()` and `.eval()` for dropout, batchnorm
4. **Hooks**: For debugging and modification
5. **Serialization**: `.state_dict()` for saving/loading

### 1.2 Basic Structure

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # MUST call parent __init__
        # Define layers here
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        # Define forward pass
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
```

### 1.3 Why super().__init__() is Critical

```python
class BadModel(nn.Module):
    def __init__(self):
        # FORGOT: super().__init__()
        self.linear = nn.Linear(10, 5)

model = BadModel()
list(model.parameters())  # EMPTY! Parameters not registered
```

### 1.4 The forward() Method

- **Never call `forward()` directly**
- Call the module as a function: `model(x)` not `model.forward(x)`
- Calling as function enables hooks and other features

```python
# CORRECT
output = model(x)

# WRONG (bypasses hooks)
output = model.forward(x)
```

---

## 2. Parameters vs Buffers

### 2.1 Parameters

Parameters are **learnable** - they have `requires_grad=True` and are updated by optimizers.

```python
class MyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter wraps tensor and registers it
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias
```

### 2.2 Buffers

Buffers are **non-learnable** tensors that should move with the model (e.g., running statistics in BatchNorm).

```python
class MyBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Non-learnable buffers (running statistics)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
```

### 2.3 Key Differences

| Aspect | Parameter | Buffer |
|--------|-----------|--------|
| `requires_grad` | True | False |
| Updated by optimizer | Yes | No |
| In `parameters()` | Yes | No |
| In `state_dict()` | Yes | Yes |
| Moves with `.to(device)` | Yes | Yes |

### 2.4 Accessing Parameters and Buffers

```python
model = MyModel()

# All parameters (recursive)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# All buffers
for name, buffer in model.named_buffers():
    print(f"{name}: {buffer.shape}")

# All modules
for name, module in model.named_modules():
    print(f"{name}: {type(module)}")
```

---

## 3. Building Neural Networks

### 3.1 Sequential Container

For simple feed-forward networks:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# With named layers
model = nn.Sequential(
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
)
```

### 3.2 ModuleList

For dynamic lists of modules (e.g., variable depth):

```python
class DynamicNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        # Use ModuleList, NOT Python list!
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)
```

**Warning**: Python lists don't register parameters!

```python
# WRONG - parameters not registered!
self.layers = [nn.Linear(10, 10) for _ in range(5)]

# CORRECT
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

### 3.3 ModuleDict

For named dynamic modules:

```python
class MultiHeadModel(nn.Module):
    def __init__(self, input_size, tasks):
        super().__init__()
        self.shared = nn.Linear(input_size, 64)
        self.heads = nn.ModuleDict({
            task: nn.Linear(64, 1) for task in tasks
        })

    def forward(self, x, task):
        x = torch.relu(self.shared(x))
        return self.heads[task](x)

model = MultiHeadModel(100, ['classification', 'regression'])
```

### 3.4 ParameterList and ParameterDict

For raw parameter collections:

```python
class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, dims):
        super().__init__()
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_embeddings, d))
            for d in dims
        ])
```

---

## 4. Layer Types Deep Dive

### 4.1 Linear (Fully Connected)

```python
# y = xW^T + b
linear = nn.Linear(
    in_features=100,
    out_features=50,
    bias=True  # Default
)

# Shape: (batch, 100) → (batch, 50)
# Parameters: W (50, 100), b (50)
# Total params: 50*100 + 50 = 5050
```

### 4.2 Embedding

Lookup table for discrete inputs (tokens, categories):

```python
embedding = nn.Embedding(
    num_embeddings=10000,  # Vocabulary size
    embedding_dim=256,     # Vector dimension
    padding_idx=0          # Index to zero out
)

# Input: (batch, seq_len) of token indices
# Output: (batch, seq_len, embedding_dim)

tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
embedded = embedding(tokens)  # (2, 3, 256)
```

### 4.3 Dropout

Regularization by randomly zeroing elements:

```python
dropout = nn.Dropout(p=0.5)

# During training: randomly zero 50% of elements, scale rest by 1/(1-p)
# During eval: identity (no dropout)

model.train()  # Dropout active
output_train = dropout(x)

model.eval()  # Dropout disabled
output_eval = dropout(x)
```

### 4.4 Layer Normalization (LLMs use this!)

```python
# Normalizes across features (last dimension)
layer_norm = nn.LayerNorm(
    normalized_shape=256,  # Feature dimension
    eps=1e-5,
    elementwise_affine=True  # Learnable gamma, beta
)

# Input: (batch, seq_len, 256)
# Output: (batch, seq_len, 256) - normalized per position
# Parameters: gamma (256), beta (256)
```

### 4.5 Batch Normalization

```python
# Normalizes across batch dimension
batch_norm = nn.BatchNorm1d(
    num_features=256,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)

# Input: (batch, 256) or (batch, 256, seq_len)
# Buffers: running_mean, running_var
```

### 4.6 Multi-Head Attention (Transformers)

```python
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True  # (batch, seq, embed) instead of (seq, batch, embed)
)

# Query, Key, Value: (batch, seq_len, embed_dim)
attn_output, attn_weights = mha(query, key, value, attn_mask=mask)
```

---

## 5. Weight Initialization

### 5.1 Why Initialization Matters

Poor initialization causes:
- **Vanishing gradients**: Activations shrink to zero
- **Exploding gradients**: Activations explode
- **Dead neurons**: ReLU neurons stuck at zero

### 5.2 Common Initialization Schemes

```python
# Xavier/Glorot (good for tanh, sigmoid)
nn.init.xavier_uniform_(tensor)
nn.init.xavier_normal_(tensor)

# He/Kaiming (good for ReLU)
nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')

# Normal
nn.init.normal_(tensor, mean=0.0, std=0.02)

# Constant
nn.init.constant_(tensor, val=0.0)
nn.init.zeros_(tensor)
nn.init.ones_(tensor)

# Orthogonal (good for RNNs)
nn.init.orthogonal_(tensor)
```

### 5.3 Initialization in Practice

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, 8)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self._init_weights()

    def _init_weights(self):
        # Common transformer initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
```

### 5.4 GPT-Style Initialization

```python
def init_weights(module):
    """GPT-2 style initialization"""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)

model.apply(init_weights)  # Apply to all submodules
```

---

## 6. Model Inspection and Manipulation

### 6.1 Counting Parameters

```python
def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

print(f"Total parameters: {count_parameters(model):,}")
```

### 6.2 Model Summary

```python
def model_summary(model, input_size):
    """Print model summary similar to Keras"""
    from torchinfo import summary  # pip install torchinfo
    return summary(model, input_size=input_size)

# Or manual inspection
for name, param in model.named_parameters():
    print(f"{name:40} {str(param.shape):20} {param.numel():>10,}")
```

### 6.3 Freezing and Unfreezing

```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Freeze by name pattern
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False
```

### 6.4 Getting Intermediate Outputs

```python
# Method 1: Forward hooks
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.layer3.register_forward_hook(get_activation('layer3'))
output = model(x)
print(activation['layer3'].shape)

# Method 2: Modify forward method
class ModelWithIntermediates(nn.Module):
    def forward(self, x, return_intermediates=False):
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x
```

---

## 7. Saving and Loading Models

### 7.1 state_dict (Recommended)

```python
# Save
torch.save(model.state_dict(), 'model_weights.pt')

# Load
model = MyModel()  # Create model architecture first
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()  # Set to evaluation mode
```

### 7.2 Full Model (Not Recommended)

```python
# Save entire model (includes architecture)
torch.save(model, 'full_model.pt')

# Load
model = torch.load('full_model.pt')

# Warning: Pickle-based, prone to issues with code changes
```

### 7.3 Checkpoint with Training State

```python
# Save checkpoint (for resuming training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'scheduler_state_dict': scheduler.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### 7.4 Partial Loading

```python
# Load only matching keys
pretrained_dict = torch.load('pretrained.pt')
model_dict = model.state_dict()

# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# Update current model dict
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

### 7.5 Cross-Device Loading

```python
# Load model trained on GPU to CPU
model.load_state_dict(torch.load('model.pt', map_location='cpu'))

# Load to specific GPU
model.load_state_dict(torch.load('model.pt', map_location='cuda:0'))

# Load to current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model.pt', map_location=device))
```

---

## 8. Interview Questions

### Q1: What's the difference between `nn.Parameter` and `register_buffer`?

**Answer**:
- `nn.Parameter`: Creates a learnable parameter with `requires_grad=True`, included in `model.parameters()`, updated by optimizer
- `register_buffer`: Creates a non-learnable tensor with `requires_grad=False`, NOT in `model.parameters()`, but IS in `state_dict()` and moves with `.to(device)`

Use Parameter for weights/biases, Buffer for things like running statistics in BatchNorm.

### Q2: Why use ModuleList instead of Python list?

**Answer**: Python lists don't register contained modules:
- Parameters won't be tracked
- `.to(device)` won't move them
- `state_dict()` won't include them
- `parameters()` won't iterate over them

Always use `nn.ModuleList`, `nn.ModuleDict`, `nn.ParameterList`, or `nn.ParameterDict`.

### Q3: Explain Xavier vs He initialization.

**Answer**:
- **Xavier (Glorot)**: Designed for sigmoid/tanh. Variance = 2/(fan_in + fan_out). Keeps variance stable through network.
- **He (Kaiming)**: Designed for ReLU. Variance = 2/fan_in. Accounts for ReLU killing half the gradient.

Use Xavier for transformers (GELU/tanh), He for CNNs with ReLU.

### Q4: How would you implement gradient checkpointing?

**Answer**: Trade compute for memory by not saving intermediate activations:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        # This won't save activations; recomputes during backward
        x = checkpoint(self.expensive_layer1, x)
        x = checkpoint(self.expensive_layer2, x)
        return x
```

### Q5: What happens if you forget `super().__init__()`?

**Answer**: The module won't be properly initialized:
- `_modules`, `_parameters`, `_buffers` dicts won't exist
- Parameters won't be registered
- `.to()`, `.parameters()`, `state_dict()` will fail or return empty
- Optimizer won't update any weights

---

## 9. Summary

### Key Concepts

1. **nn.Module**: Base class that handles parameter tracking, device management, serialization
2. **Parameters**: Learnable tensors (`requires_grad=True`)
3. **Buffers**: Non-learnable tensors that move with model
4. **ModuleList/ModuleDict**: Proper containers for dynamic module collections
5. **Initialization**: Xavier for sigmoid/tanh/GELU, He for ReLU

### Best Practices

1. Always call `super().__init__()` in `__init__`
2. Never call `forward()` directly - use `model(x)`
3. Use `ModuleList`/`ModuleDict`, not Python lists/dicts
4. Save `state_dict()`, not entire model
5. Initialize weights appropriately for your activation function
6. Use `.train()` and `.eval()` modes correctly

### Common Patterns

```python
# Creating a model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Define forward pass
        return x

# Training
model.train()
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    predictions = model(x)

# Save/Load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))
```
