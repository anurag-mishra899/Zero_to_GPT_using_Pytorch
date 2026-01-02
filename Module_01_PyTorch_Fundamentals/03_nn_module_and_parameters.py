"""
Module 1.3: nn.Module, Parameters, and Building Networks
========================================================

This module covers:
1. Understanding nn.Module
2. Parameters vs Buffers
3. Building neural networks
4. Layer types
5. Weight initialization
6. Model inspection
7. Saving and loading

Run this file to see all examples in action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("MODULE 1.3: NN.MODULE, PARAMETERS, AND BUILDING NETWORKS")
print("=" * 70)


# =============================================================================
# SECTION 1: BASIC NN.MODULE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: BASIC NN.MODULE")
print("=" * 70)

# 1.1 Simple model structure
print("\n--- 1.1 Basic Model Structure ---")

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # CRITICAL: Must call parent __init__
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel(10, 20, 5)
print(f"Model:\n{model}")

# Test forward pass
x = torch.randn(2, 10)
output = model(x)  # CORRECT: Call as function
print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")

# 1.2 Parameters are automatically tracked
print("\n--- 1.2 Automatic Parameter Tracking ---")
print("Parameters in model:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")


# =============================================================================
# SECTION 2: PARAMETERS VS BUFFERS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: PARAMETERS VS BUFFERS")
print("=" * 70)

# 2.1 Custom layer with both
print("\n--- 2.1 Custom Layer with Parameters and Buffers ---")

class CustomNormLayer(nn.Module):
    """Custom layer demonstrating parameters vs buffers"""

    def __init__(self, num_features):
        super().__init__()
        # PARAMETERS: Learnable, updated by optimizer
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # BUFFERS: Not learnable, but move with model
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches', torch.tensor(0))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
            self.num_batches += 1
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return self.gamma * x_norm + self.beta

layer = CustomNormLayer(5)

print("Parameters (learnable):")
for name, param in layer.named_parameters():
    print(f"  {name}: {param.shape}")

print("\nBuffers (non-learnable):")
for name, buffer in layer.named_buffers():
    print(f"  {name}: {buffer.shape}")

# 2.2 State dict contains both
print("\n--- 2.2 State Dict Contains Both ---")
print("State dict keys:")
for key in layer.state_dict().keys():
    print(f"  {key}")


# =============================================================================
# SECTION 3: MODULE CONTAINERS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODULE CONTAINERS")
print("=" * 70)

# 3.1 Sequential
print("\n--- 3.1 nn.Sequential ---")
sequential_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)
print(f"Sequential model:\n{sequential_model}")

# 3.2 ModuleList
print("\n--- 3.2 nn.ModuleList ---")

class DynamicDepthNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        # ModuleList properly registers all layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.layers.append(nn.Linear(in_dim, hidden_size))
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)

dynamic_model = DynamicDepthNetwork(10, 20, num_layers=4)
print(f"Dynamic model with {len(dynamic_model.layers)} hidden layers:")
print(f"Total parameters: {sum(p.numel() for p in dynamic_model.parameters())}")

# 3.3 ModuleDict
print("\n--- 3.3 nn.ModuleDict ---")

class MultiTaskModel(nn.Module):
    def __init__(self, input_size, task_names):
        super().__init__()
        self.shared = nn.Linear(input_size, 32)
        self.task_heads = nn.ModuleDict({
            name: nn.Linear(32, 1) for name in task_names
        })

    def forward(self, x, task_name):
        shared_features = F.relu(self.shared(x))
        return self.task_heads[task_name](shared_features)

multi_task = MultiTaskModel(10, ['classification', 'regression', 'ranking'])
print("Task heads:")
for name, module in multi_task.task_heads.items():
    print(f"  {name}: {module}")

# 3.4 WRONG: Python list (parameters not registered!)
print("\n--- 3.4 WARNING: Python Lists Don't Register Parameters ---")

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # WRONG: This won't work!
        self.layers = [nn.Linear(10, 10) for _ in range(3)]

bad_model = BadModel()
print(f"Parameters in BadModel: {len(list(bad_model.parameters()))}")
print("WARNING: Parameters not registered with Python list!")


# =============================================================================
# SECTION 4: COMMON LAYER TYPES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: COMMON LAYER TYPES")
print("=" * 70)

# 4.1 Linear
print("\n--- 4.1 nn.Linear ---")
linear = nn.Linear(100, 50, bias=True)
print(f"Linear(100, 50):")
print(f"  Weight shape: {linear.weight.shape}")
print(f"  Bias shape: {linear.bias.shape}")
print(f"  Total params: {linear.weight.numel() + linear.bias.numel()}")

# 4.2 Embedding
print("\n--- 4.2 nn.Embedding ---")
embedding = nn.Embedding(
    num_embeddings=10000,  # Vocabulary size
    embedding_dim=256,
    padding_idx=0  # This index will always be zeros
)
print(f"Embedding(10000, 256):")
print(f"  Weight shape: {embedding.weight.shape}")
print(f"  Total params: {embedding.weight.numel():,}")

# Usage
tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # (batch=2, seq=4)
embedded = embedding(tokens)
print(f"  Input shape: {tokens.shape}")
print(f"  Output shape: {embedded.shape}")

# 4.3 LayerNorm
print("\n--- 4.3 nn.LayerNorm (Used in Transformers) ---")
layer_norm = nn.LayerNorm(256)
x = torch.randn(2, 10, 256)  # (batch, seq_len, features)
output = layer_norm(x)
print(f"LayerNorm(256):")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Normalized per position - mean: {output[0, 0].mean():.4f}, std: {output[0, 0].std():.4f}")

# 4.4 Dropout
print("\n--- 4.4 nn.Dropout ---")
dropout = nn.Dropout(p=0.5)
x = torch.ones(5, 5)

dropout.train()  # Enable dropout
output_train = dropout(x)
print(f"Training mode (50% dropout):\n{output_train}")
print(f"Non-zero elements: {(output_train != 0).sum().item()}/25")

dropout.eval()  # Disable dropout
output_eval = dropout(x)
print(f"\nEval mode (no dropout):\n{output_eval}")

# 4.5 MultiheadAttention
print("\n--- 4.5 nn.MultiheadAttention ---")
mha = nn.MultiheadAttention(
    embed_dim=64,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)
query = key = value = torch.randn(2, 10, 64)  # (batch, seq, embed)
attn_output, attn_weights = mha(query, key, value)
print(f"MultiheadAttention(embed_dim=64, num_heads=8):")
print(f"  Input shape: {query.shape}")
print(f"  Output shape: {attn_output.shape}")
print(f"  Attention weights shape: {attn_weights.shape}")


# =============================================================================
# SECTION 5: WEIGHT INITIALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: WEIGHT INITIALIZATION")
print("=" * 70)

# 5.1 Different initialization schemes
print("\n--- 5.1 Initialization Schemes ---")

# Xavier (good for tanh/sigmoid)
w_xavier = torch.empty(100, 100)
nn.init.xavier_uniform_(w_xavier)
print(f"Xavier uniform: mean={w_xavier.mean():.4f}, std={w_xavier.std():.4f}")

# He/Kaiming (good for ReLU)
w_he = torch.empty(100, 100)
nn.init.kaiming_normal_(w_he, mode='fan_in', nonlinearity='relu')
print(f"He normal: mean={w_he.mean():.4f}, std={w_he.std():.4f}")

# Normal (GPT-style)
w_normal = torch.empty(100, 100)
nn.init.normal_(w_normal, mean=0.0, std=0.02)
print(f"Normal(0, 0.02): mean={w_normal.mean():.4f}, std={w_normal.std():.4f}")

# 5.2 Initialize all layers in a model
print("\n--- 5.2 Initialize Model with .apply() ---")

def init_weights_transformer(module):
    """GPT-2 style initialization"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

block = TransformerBlock(64)
block.apply(init_weights_transformer)
print("Initialized TransformerBlock with GPT-2 style initialization")

# Check a weight
linear_weight = block.ffn[0].weight
print(f"FFN linear weight: mean={linear_weight.mean():.4f}, std={linear_weight.std():.4f}")


# =============================================================================
# SECTION 6: MODEL INSPECTION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: MODEL INSPECTION")
print("=" * 70)

# 6.1 Count parameters
print("\n--- 6.1 Count Parameters ---")

def count_parameters(model, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

model = TransformerBlock(64)
print(f"Trainable parameters: {count_parameters(model):,}")
print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")

# 6.2 Parameter breakdown
print("\n--- 6.2 Parameter Breakdown ---")
print(f"{'Name':<40} {'Shape':<20} {'Params':>10}")
print("-" * 72)
for name, param in model.named_parameters():
    print(f"{name:<40} {str(list(param.shape)):<20} {param.numel():>10,}")

# 6.3 Module tree
print("\n--- 6.3 Module Tree ---")
for name, module in model.named_modules():
    if name:  # Skip empty root name
        print(f"  {name}: {module.__class__.__name__}")

# 6.4 Freezing layers
print("\n--- 6.4 Freezing Layers ---")
# Freeze attention
for param in model.attention.parameters():
    param.requires_grad = False

trainable_after_freeze = count_parameters(model)
print(f"Trainable after freezing attention: {trainable_after_freeze:,}")


# =============================================================================
# SECTION 7: SAVING AND LOADING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: SAVING AND LOADING")
print("=" * 70)

# 7.1 state_dict
print("\n--- 7.1 Understanding state_dict ---")
model = SimpleModel(10, 20, 5)
print("Keys in state_dict:")
for key in model.state_dict().keys():
    print(f"  {key}")

# 7.2 Save and load pattern
print("\n--- 7.2 Save/Load Pattern ---")
print("""
# RECOMMENDED: Save state_dict
torch.save(model.state_dict(), 'model.pt')

# Load
model = SimpleModel(10, 20, 5)  # Create architecture
model.load_state_dict(torch.load('model.pt'))
model.eval()

# For training checkpoints, save more:
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# Resume training:
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
""")

# 7.3 Cross-device loading
print("\n--- 7.3 Cross-Device Loading ---")
print("""
# Load to CPU
state_dict = torch.load('model.pt', map_location='cpu')

# Load to specific GPU
state_dict = torch.load('model.pt', map_location='cuda:0')

# Load to current device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load('model.pt', map_location=device)
""")


# =============================================================================
# SECTION 8: PRACTICAL EXAMPLE - MINI GPT BLOCK
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: PRACTICAL EXAMPLE - MINI GPT BLOCK")
print("=" * 70)

class MiniGPTBlock(nn.Module):
    """A simplified GPT-style transformer block"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        # Pre-norm architecture (like GPT-2)
        # Self-attention
        attn_out, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            attn_mask=mask
        )
        x = x + self.dropout(attn_out)

        # Feed-forward
        x = x + self.ffn(self.ln2(x))

        return x

# Create and inspect
gpt_block = MiniGPTBlock(d_model=256, n_heads=8, d_ff=1024)

print("MiniGPTBlock Architecture:")
print(f"  d_model=256, n_heads=8, d_ff=1024")
print(f"  Total parameters: {count_parameters(gpt_block):,}")

# Test forward pass
x = torch.randn(2, 16, 256)  # (batch, seq_len, d_model)
output = gpt_block(x)
print(f"\nForward pass:")
print(f"  Input: {x.shape}")
print(f"  Output: {output.shape}")


print("\n" + "=" * 70)
print("END OF MODULE 1.3")
print("=" * 70)
