"""
Module 1.2: Autograd and Computational Graphs
==============================================

This module covers:
1. Automatic differentiation basics
2. Computational graphs in PyTorch
3. Gradient computation
4. Controlling gradient flow
5. Common pitfalls and solutions
6. Advanced autograd features

Run this file to see all examples in action.
"""

import torch
import torch.nn as nn

print("=" * 70)
print("MODULE 1.2: AUTOGRAD AND COMPUTATIONAL GRAPHS")
print("=" * 70)


# =============================================================================
# SECTION 1: AUTOGRAD BASICS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: AUTOGRAD BASICS")
print("=" * 70)

# 1.1 Enabling gradient tracking
print("\n--- 1.1 Enabling Gradient Tracking ---")

# Method 1: At creation
x1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"Created with requires_grad=True: {x1.requires_grad}")

# Method 2: After creation
x2 = torch.tensor([1.0, 2.0, 3.0])
x2.requires_grad_(True)
print(f"After .requires_grad_(True): {x2.requires_grad}")

# 1.2 Basic backward pass
print("\n--- 1.2 Basic Backward Pass ---")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

print(f"x = {x.item()}")
print(f"y = x² + 3x + 1 = {y.item()}")

y.backward()  # Compute dy/dx = 2x + 3
print(f"dy/dx = 2x + 3 = 2*2 + 3 = {x.grad.item()}")

# 1.3 Gradient accumulation
print("\n--- 1.3 Gradient Accumulation (CRITICAL!) ---")
x = torch.tensor([2.0], requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(f"After first backward (x²): x.grad = {x.grad.item()}")

# Second backward - gradients ACCUMULATE
y2 = x ** 3
y2.backward()
print(f"After second backward (x³): x.grad = {x.grad.item()}")
print("(4.0 from x² + 12.0 from x³ = 16.0)")

# Zero gradients
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"After zeroing and x²: x.grad = {x.grad.item()}")

print("\nWARNING: Always zero gradients between batches!")

# 1.4 Non-scalar outputs
print("\n--- 1.4 Non-Scalar Outputs ---")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # [1, 4, 9] - not scalar!

print(f"x = {x}")
print(f"y = x² = {y}")

# This would fail:
# y.backward()  # RuntimeError!

# Solution 1: Pass gradient vector (Jacobian-vector product)
y.backward(torch.ones_like(y))
print(f"\nWith gradient=ones_like(y):")
print(f"x.grad = {x.grad}")  # [2, 4, 6] = 2x

# Solution 2: Sum to scalar
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(f"\nWith y.sum().backward():")
print(f"x.grad = {x.grad}")  # Same result

# 1.5 Leaf tensors
print("\n--- 1.5 Leaf Tensors ---")
a = torch.tensor([2.0], requires_grad=True)  # Leaf
b = a * 2  # Not a leaf - created by operation

print(f"a.is_leaf = {a.is_leaf}")
print(f"b.is_leaf = {b.is_leaf}")

c = b.sum()
c.backward()

print(f"a.grad = {a.grad}")  # Has gradient
print(f"b.grad = {b.grad}")  # None - not a leaf!

# Retain gradients for non-leaf
print("\nTo get non-leaf gradients, use retain_grad():")
a = torch.tensor([2.0], requires_grad=True)
b = a * 2
b.retain_grad()  # Tell PyTorch to keep this gradient
c = b.sum()
c.backward()
print(f"b.grad (with retain_grad) = {b.grad}")


# =============================================================================
# SECTION 2: COMPUTATIONAL GRAPH
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: COMPUTATIONAL GRAPH")
print("=" * 70)

# 2.1 Visualizing the graph through grad_fn
print("\n--- 2.1 Computational Graph Structure ---")
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

c = a + b  # AddBackward
d = c * 2  # MulBackward
e = d ** 2  # PowBackward

print("Expression: e = ((a + b) * 2)²")
print(f"\ne.grad_fn = {e.grad_fn}")
print(f"e.grad_fn.next_functions = {e.grad_fn.next_functions}")

# Trace the full graph
print("\nFull graph trace:")
print(f"e: {e.grad_fn}")
print(f"  └─ d: {e.grad_fn.next_functions[0][0]}")
print(f"       └─ c: {e.grad_fn.next_functions[0][0].next_functions[0][0]}")

# 2.2 Manual gradient computation
print("\n--- 2.2 Manual vs Autograd Gradient Check ---")
x = torch.tensor([3.0], requires_grad=True)

# f(x) = x³ + 2x² + x
# f'(x) = 3x² + 4x + 1
y = x ** 3 + 2 * x ** 2 + x
y.backward()

manual_grad = 3 * 3**2 + 4 * 3 + 1  # = 27 + 12 + 1 = 40
print(f"f(x) = x³ + 2x² + x at x=3")
print(f"f'(x) = 3x² + 4x + 1")
print(f"Manual: f'(3) = {manual_grad}")
print(f"Autograd: x.grad = {x.grad.item()}")


# =============================================================================
# SECTION 3: CONTROLLING GRADIENT FLOW
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: CONTROLLING GRADIENT FLOW")
print("=" * 70)

# 3.1 torch.no_grad()
print("\n--- 3.1 torch.no_grad() ---")
x = torch.tensor([1.0], requires_grad=True)

with torch.no_grad():
    y = x * 2
    print(f"Inside no_grad: y.requires_grad = {y.requires_grad}")

# Outside context
y_outside = x * 2
print(f"Outside no_grad: y.requires_grad = {y_outside.requires_grad}")

# 3.2 torch.inference_mode()
print("\n--- 3.2 torch.inference_mode() ---")
x = torch.tensor([1.0], requires_grad=True)

with torch.inference_mode():
    y = x * 2
    print(f"Inside inference_mode: y.requires_grad = {y.requires_grad}")
    # Note: inference_mode is more restrictive but faster than no_grad

# 3.3 detach()
print("\n--- 3.3 .detach() ---")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

y_detached = y.detach()
print(f"y.requires_grad = {y.requires_grad}")
print(f"y_detached.requires_grad = {y_detached.requires_grad}")
print(f"Share memory? {y.data_ptr() == y_detached.data_ptr()}")

# 3.4 Freezing parameters
print("\n--- 3.4 Freezing Parameters ---")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 10)

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = SimpleModel()

# Freeze encoder
print("Before freezing:")
for name, param in model.named_parameters():
    print(f"  {name}: requires_grad = {param.requires_grad}")

for param in model.encoder.parameters():
    param.requires_grad = False

print("\nAfter freezing encoder:")
for name, param in model.named_parameters():
    print(f"  {name}: requires_grad = {param.requires_grad}")


# =============================================================================
# SECTION 4: GRADIENT CLIPPING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: GRADIENT CLIPPING")
print("=" * 70)

# 4.1 Clip by norm
print("\n--- 4.1 Clip by Norm ---")
model = nn.Linear(10, 10)

# Simulate gradients
x = torch.randn(5, 10)
y = torch.randn(5, 10)
loss = ((model(x) - y) ** 2).sum()
loss.backward()

# Check gradient norm before clipping
total_norm_before = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm_before += p.grad.data.norm(2).item() ** 2
total_norm_before = total_norm_before ** 0.5
print(f"Gradient norm before clipping: {total_norm_before:.4f}")

# Clip gradients
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

# Check gradient norm after clipping
total_norm_after = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm_after += p.grad.data.norm(2).item() ** 2
total_norm_after = total_norm_after ** 0.5
print(f"Gradient norm after clipping: {total_norm_after:.4f}")

# 4.2 Clip by value
print("\n--- 4.2 Clip by Value ---")
model = nn.Linear(10, 10)
x = torch.randn(5, 10)
y = torch.randn(5, 10)
loss = ((model(x) - y) ** 2).sum()
loss.backward()

# Check max gradient before
max_grad_before = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f"Max gradient value before: {max_grad_before:.4f}")

# Clip by value
clip_value = 0.5
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

max_grad_after = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
print(f"Max gradient value after: {max_grad_after:.4f}")


# =============================================================================
# SECTION 5: GRADIENT ACCUMULATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: GRADIENT ACCUMULATION")
print("=" * 70)

print("\n--- Gradient Accumulation Pattern ---")
print("""
# Simulate larger batch by accumulating gradients

accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    # Forward
    loss = criterion(model(x), y)

    # Scale loss (important!)
    loss = loss / accumulation_steps

    # Backward (gradients accumulate)
    loss.backward()

    # Update weights every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Handle remaining batches
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
""")

# Demonstration
print("\nDemonstration:")
model = nn.Linear(5, 1)
x = torch.randn(4, 5)  # 4 samples
y = torch.randn(4, 1)

criterion = nn.MSELoss()
accumulation_steps = 2

model.zero_grad()

for i in range(4):
    loss = criterion(model(x[i:i+1]), y[i:i+1])
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        print(f"Step {i+1}: Weight grad norm = {model.weight.grad.norm():.4f}")
        # In real code: optimizer.step() here
        model.zero_grad()


# =============================================================================
# SECTION 6: COMMON PITFALLS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: COMMON PITFALLS")
print("=" * 70)

# 6.1 In-place operations
print("\n--- 6.1 In-Place Operations Break Autograd ---")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
print(f"y = {y}")

try:
    y.add_(1)  # In-place operation
    (y.sum()).backward()
except RuntimeError as e:
    print(f"ERROR: {str(e)[:80]}...")

print("\nSolution: Use out-of-place operations")
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
y = y + 1  # Creates new tensor
(y.sum()).backward()
print(f"x.grad = {x.grad}")

# 6.2 Using .item() for scalars
print("\n--- 6.2 Use .item() for Logging ---")
print("""
# WRONG - accumulates graph in memory!
total_loss = 0
for batch in dataloader:
    loss = criterion(model(batch), targets)
    total_loss += loss  # Keeps entire computation graph!

# CORRECT
total_loss = 0
for batch in dataloader:
    loss = criterion(model(batch), targets)
    total_loss += loss.item()  # Just the scalar value
""")

# 6.3 Double backward
print("\n--- 6.3 Double Backward Requires retain_graph ---")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

y.backward(retain_graph=True)  # First backward
print(f"After first backward: x.grad = {x.grad}")

x.grad.zero_()  # Don't forget to zero!
y.backward()  # Second backward (only works with retain_graph=True)
print(f"After second backward: x.grad = {x.grad}")


# =============================================================================
# SECTION 7: ADVANCED AUTOGRAD
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: ADVANCED AUTOGRAD")
print("=" * 70)

# 7.1 Custom autograd function
print("\n--- 7.1 Custom Autograd Function ---")

class MyReLU(torch.autograd.Function):
    """Custom ReLU implementation"""

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass.
        ctx is a context object to save tensors for backward.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.
        grad_output is the gradient from upstream.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Test
x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = MyReLU.apply(x)
print(f"Input: {x}")
print(f"MyReLU output: {y}")

y.sum().backward()
print(f"Gradients: {x.grad}")
print("(Only positive inputs get gradients)")

# 7.2 Gradient hooks
print("\n--- 7.2 Gradient Hooks ---")

def gradient_hook(grad):
    """Hook function called during backward"""
    print(f"  Hook received gradient: {grad}")
    # Can modify gradient here
    return grad * 2  # Double the gradient

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
hook_handle = x.register_hook(gradient_hook)

y = x ** 2
print("Computing backward for y = x²...")
y.sum().backward()

print(f"Final x.grad (doubled by hook): {x.grad}")
print(f"Expected without hook: {2 * x.detach()}")

# Remove hook
hook_handle.remove()

# 7.3 Higher-order gradients
print("\n--- 7.3 Higher-Order Gradients ---")
x = torch.tensor([3.0], requires_grad=True)

# f(x) = x³
# f'(x) = 3x²
# f''(x) = 6x

y = x ** 3

# First derivative
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"f(x) = x³ at x=3")
print(f"f'(x) = 3x² = 3*9 = {grad1.item()}")

# Second derivative
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"f''(x) = 6x = 6*3 = {grad2.item()}")

# 7.4 torch.autograd.grad()
print("\n--- 7.4 torch.autograd.grad() vs .backward() ---")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Using grad() - returns gradient tensors
grads = torch.autograd.grad(
    outputs=y.sum(),
    inputs=x,
    create_graph=False,
    retain_graph=True,
)
print(f"Using grad(): gradients = {grads[0]}")
print(f"x.grad is still None: {x.grad}")

# Using backward() - populates .grad attribute
y.sum().backward()
print(f"Using backward(): x.grad = {x.grad}")


# =============================================================================
# SECTION 8: PRACTICAL EXAMPLES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: PRACTICAL EXAMPLES")
print("=" * 70)

# 8.1 Gradient penalty (WGAN-GP style)
print("\n--- 8.1 Gradient Penalty ---")

def gradient_penalty(discriminator, real, fake):
    """
    Compute gradient penalty for WGAN-GP.

    Penalizes gradients that deviate from unit norm.
    """
    batch_size = real.size(0)

    # Random interpolation between real and fake
    alpha = torch.rand(batch_size, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    # Forward through discriminator
    d_interpolated = discriminator(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,  # For second-order gradient
        retain_graph=True,
    )[0]

    # Compute penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty

# Demo
discriminator = nn.Linear(10, 1)
real = torch.randn(4, 10)
fake = torch.randn(4, 10)
penalty = gradient_penalty(discriminator, real, fake)
print(f"Gradient penalty: {penalty.item():.4f}")

# 8.2 Computing Jacobian
print("\n--- 8.2 Computing Jacobian ---")

def compute_jacobian(f, x):
    """Compute Jacobian matrix df/dx"""
    x = x.clone().requires_grad_(True)
    y = f(x)

    jacobian = []
    for i in range(y.size(0)):
        grad = torch.autograd.grad(
            y[i], x, retain_graph=True, create_graph=False
        )[0]
        jacobian.append(grad)

    return torch.stack(jacobian)

# Example: f(x) = [x₁², x₁*x₂, x₂²]
def f(x):
    return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

x = torch.tensor([2.0, 3.0])
J = compute_jacobian(f, x)
print(f"f(x) = [x₁², x₁*x₂, x₂²] at x = {x.tolist()}")
print(f"Jacobian:\n{J}")
print("\nExpected:")
print("  [[2*x₁,    0  ],   = [[4, 0],")
print("   [x₂,     x₁ ],      [3, 2],")
print("   [0,     2*x₂]]      [0, 6]]")


# =============================================================================
# SECTION 9: COMPLETE TRAINING LOOP
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: COMPLETE TRAINING LOOP PATTERN")
print("=" * 70)

print("""
def train_epoch(model, dataloader, optimizer, criterion, device,
                clip_grad_norm=None, accumulation_steps=1):
    '''
    Complete training loop with best practices.
    '''
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()  # Zero at start

    for i, (x, y) in enumerate(dataloader):
        # Move to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Scale for accumulation
        loss = loss / accumulation_steps

        # Backward pass (gradients accumulate)
        loss.backward()

        # Track loss (use .item() to avoid memory leak!)
        total_loss += loss.item() * accumulation_steps

        # Update weights every N steps
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping (after backward, before step)
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad_norm
                )

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()  # Decorator form
def validate(model, dataloader, criterion, device):
    '''
    Validation loop - no gradients needed.
    '''
    model.eval()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        total_loss += loss.item()

    return total_loss / len(dataloader)
""")


# =============================================================================
# SECTION 10: DEBUGGING GRADIENTS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10: DEBUGGING GRADIENTS")
print("=" * 70)

print("\n--- Detecting NaN/Inf Gradients ---")

def check_gradients(model):
    """Check for NaN or Inf gradients"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
            print(f"{name}: grad norm = {param.grad.norm():.4f}")

model = nn.Linear(5, 3)
x = torch.randn(2, 5)
y = model(x).sum()
y.backward()

check_gradients(model)

print("\n--- Anomaly Detection ---")
print("""
# Enable anomaly detection (slow but helpful for debugging)
torch.autograd.set_detect_anomaly(True)

# This will give detailed error messages about where
# NaN/Inf gradients originated

# Don't forget to disable after debugging:
torch.autograd.set_detect_anomaly(False)
""")


print("\n" + "=" * 70)
print("END OF MODULE 1.2")
print("=" * 70)
