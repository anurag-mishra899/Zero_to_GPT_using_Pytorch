"""
Module 2.1: Backpropagation Implementation
==========================================

This module covers:
1. Manual forward and backward passes
2. Gradient computation for common layers
3. Building a simple neural network from scratch
4. Numerical gradient checking
5. Visualizing gradient flow

Run this file to see all examples in action.
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("MODULE 2.1: BACKPROPAGATION IMPLEMENTATION")
print("=" * 70)


# =============================================================================
# SECTION 1: BASIC GRADIENT COMPUTATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: BASIC GRADIENT COMPUTATION")
print("=" * 70)

# 1.1 Simple scalar example
print("\n--- 1.1 Scalar Chain Rule ---")

# y = (a + b) * c
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(4.0, requires_grad=True)

# Forward
add_result = a + b  # 5
y = add_result * c  # 20

print(f"a={a.item()}, b={b.item()}, c={c.item()}")
print(f"y = (a + b) * c = {y.item()}")

# Backward
y.backward()

print(f"\nGradients (computed by autograd):")
print(f"  dy/da = {a.grad.item()}")  # c = 4
print(f"  dy/db = {b.grad.item()}")  # c = 4
print(f"  dy/dc = {c.grad.item()}")  # a + b = 5

print(f"\nManual verification:")
print(f"  dy/da = c = {c.item()}")
print(f"  dy/db = c = {c.item()}")
print(f"  dy/dc = a + b = {(a + b).item()}")


# 1.2 Vector gradients
print("\n--- 1.2 Vector Gradients ---")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # [1, 4, 9]
loss = y.sum()  # 14

loss.backward()

print(f"x = {x.tolist()}")
print(f"y = x² = {y.tolist()}")
print(f"loss = sum(y) = {loss.item()}")
print(f"d(loss)/dx = 2x = {x.grad.tolist()}")


# =============================================================================
# SECTION 2: LINEAR LAYER GRADIENTS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: LINEAR LAYER GRADIENTS")
print("=" * 70)

# 2.1 Manual linear layer backward
print("\n--- 2.1 Linear Layer: Y = XW + b ---")

class ManualLinear:
    """Linear layer with manual backward pass"""

    def __init__(self, in_features, out_features):
        # Initialize weights
        self.W = torch.randn(in_features, out_features, requires_grad=True) * 0.01
        self.b = torch.zeros(out_features, requires_grad=True)

        # Cache for backward
        self.cache = None

    def forward(self, X):
        """
        Forward pass: Y = XW + b
        X: (batch, in_features)
        Y: (batch, out_features)
        """
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dY):
        """
        Backward pass
        dY: gradient of loss w.r.t. Y (batch, out_features)
        Returns: gradient w.r.t. X (batch, in_features)
        """
        X = self.cache

        # Gradient w.r.t. weights: dL/dW = X^T @ dY
        self.dW = X.T @ dY

        # Gradient w.r.t. bias: dL/db = sum(dY, dim=0)
        self.db = dY.sum(dim=0)

        # Gradient w.r.t. input: dL/dX = dY @ W^T
        dX = dY @ self.W.T

        return dX

# Test
batch, in_f, out_f = 4, 3, 2
X = torch.randn(batch, in_f)
linear = ManualLinear(in_f, out_f)

# Forward
Y = linear.forward(X)

# Simulate loss = sum(Y)
dY = torch.ones_like(Y)  # dL/dY for loss = sum(Y)

# Manual backward
dX = linear.backward(dY)

print(f"Input X shape: {X.shape}")
print(f"Output Y shape: {Y.shape}")
print(f"dL/dW shape: {linear.dW.shape}")
print(f"dL/db shape: {linear.db.shape}")
print(f"dL/dX shape: {dX.shape}")

# Verify with PyTorch autograd
X_auto = X.clone().requires_grad_(True)
Y_auto = X_auto @ linear.W + linear.b
loss = Y_auto.sum()
loss.backward()

print(f"\nVerification with autograd:")
print(f"  dW matches: {torch.allclose(linear.dW, linear.W.grad)}")
print(f"  db matches: {torch.allclose(linear.db, linear.b.grad)}")
print(f"  dX matches: {torch.allclose(dX, X_auto.grad)}")


# =============================================================================
# SECTION 3: ACTIVATION FUNCTION GRADIENTS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: ACTIVATION FUNCTION GRADIENTS")
print("=" * 70)

# 3.1 ReLU
print("\n--- 3.1 ReLU Backward ---")

class ManualReLU:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X
        return torch.maximum(X, torch.zeros_like(X))

    def backward(self, dY):
        X = self.cache
        # Gradient is 1 where X > 0, else 0
        return dY * (X > 0).float()

X = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu = ManualReLU()
Y = relu.forward(X)
dX = relu.backward(torch.ones_like(Y))

print(f"X:  {X.tolist()}")
print(f"Y = ReLU(X): {Y.tolist()}")
print(f"dY/dX: {dX.tolist()}")
print("Note: Gradient is 0 where input was negative")


# 3.2 Sigmoid
print("\n--- 3.2 Sigmoid Backward ---")

class ManualSigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        Y = 1 / (1 + torch.exp(-X))
        self.cache = Y  # Store output, not input
        return Y

    def backward(self, dY):
        Y = self.cache
        # σ'(x) = σ(x) * (1 - σ(x))
        return dY * Y * (1 - Y)

X = torch.tensor([-2.0, 0.0, 2.0])
sigmoid = ManualSigmoid()
Y = sigmoid.forward(X)
dX = sigmoid.backward(torch.ones_like(Y))

print(f"X: {X.tolist()}")
print(f"σ(X): {[f'{v:.4f}' for v in Y.tolist()]}")
print(f"dσ/dX: {[f'{v:.4f}' for v in dX.tolist()]}")
print("Note: Gradient is maximum (0.25) at X=0, vanishes at extremes")


# 3.3 Softmax + Cross-Entropy
print("\n--- 3.3 Softmax + Cross-Entropy Backward ---")

def softmax(X):
    """Numerically stable softmax"""
    X_max = X.max(dim=-1, keepdim=True).values
    exp_X = torch.exp(X - X_max)
    return exp_X / exp_X.sum(dim=-1, keepdim=True)

def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss with softmax
    logits: (batch, num_classes)
    targets: (batch,) class indices
    """
    probs = softmax(logits)
    batch_size = logits.size(0)
    # Select probability of correct class
    correct_probs = probs[range(batch_size), targets]
    loss = -torch.log(correct_probs).mean()
    return loss, probs

def softmax_cross_entropy_backward(probs, targets):
    """
    Gradient of softmax + cross-entropy w.r.t. logits
    Beautiful result: dL/dz = probs - one_hot(targets)
    """
    batch_size = probs.size(0)
    dlogits = probs.clone()
    dlogits[range(batch_size), targets] -= 1
    dlogits /= batch_size  # Average over batch
    return dlogits

# Test
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])  # Correct classes

loss, probs = cross_entropy_loss(logits, targets)
dlogits = softmax_cross_entropy_backward(probs, targets)

print(f"Logits:\n{logits}")
print(f"Probs (softmax):\n{probs}")
print(f"Targets: {targets.tolist()}")
print(f"Loss: {loss.item():.4f}")
print(f"\ndL/dlogits:\n{dlogits}")

# Verify
logits_v = logits.clone().requires_grad_(True)
loss_v = nn.CrossEntropyLoss()(logits_v, targets)
loss_v.backward()
print(f"\nAutograd verification matches: {torch.allclose(dlogits, logits_v.grad, atol=1e-5)}")


# =============================================================================
# SECTION 4: COMPLETE NEURAL NETWORK FROM SCRATCH
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: COMPLETE NEURAL NETWORK FROM SCRATCH")
print("=" * 70)

class NeuralNetworkFromScratch:
    """
    2-layer neural network with manual forward and backward passes.
    Architecture: Input -> Linear -> ReLU -> Linear -> Softmax -> Loss
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        scale1 = np.sqrt(2.0 / input_size)  # He initialization
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = torch.randn(input_size, hidden_size) * scale1
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * scale2
        self.b2 = torch.zeros(output_size)

        # Gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

        # Cache for backward pass
        self.cache = {}

    def forward(self, X):
        """Forward pass"""
        # Layer 1: Linear
        self.cache['X'] = X
        Z1 = X @ self.W1 + self.b1
        self.cache['Z1'] = Z1

        # Activation: ReLU
        A1 = torch.maximum(Z1, torch.zeros_like(Z1))
        self.cache['A1'] = A1

        # Layer 2: Linear
        Z2 = A1 @ self.W2 + self.b2
        self.cache['Z2'] = Z2

        # Output: Softmax
        probs = softmax(Z2)
        self.cache['probs'] = probs

        return probs

    def compute_loss(self, probs, targets):
        """Cross-entropy loss"""
        batch_size = probs.size(0)
        correct_probs = probs[range(batch_size), targets]
        loss = -torch.log(correct_probs + 1e-8).mean()
        return loss

    def backward(self, targets):
        """Backward pass - compute all gradients"""
        batch_size = self.cache['X'].size(0)

        # Gradient of softmax + cross-entropy
        probs = self.cache['probs']
        dZ2 = probs.clone()
        dZ2[range(batch_size), targets] -= 1
        dZ2 /= batch_size

        # Layer 2 gradients
        A1 = self.cache['A1']
        self.dW2 = A1.T @ dZ2
        self.db2 = dZ2.sum(dim=0)

        # Gradient to A1
        dA1 = dZ2 @ self.W2.T

        # ReLU backward
        Z1 = self.cache['Z1']
        dZ1 = dA1 * (Z1 > 0).float()

        # Layer 1 gradients
        X = self.cache['X']
        self.dW1 = X.T @ dZ1
        self.db1 = dZ1.sum(dim=0)

    def update(self, learning_rate):
        """Gradient descent update"""
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

# Test the network
print("\n--- Training Neural Network from Scratch ---")

# Create simple dataset
torch.manual_seed(42)
X_train = torch.randn(100, 4)
y_train = torch.randint(0, 3, (100,))

# Create network
net = NeuralNetworkFromScratch(input_size=4, hidden_size=10, output_size=3)

# Training loop
print("Training...")
for epoch in range(100):
    # Forward
    probs = net.forward(X_train)
    loss = net.compute_loss(probs, y_train)

    # Backward
    net.backward(y_train)

    # Update
    net.update(learning_rate=0.1)

    if epoch % 20 == 0:
        # Compute accuracy
        predictions = probs.argmax(dim=1)
        accuracy = (predictions == y_train).float().mean()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.2%}")

print("\nTraining complete!")


# =============================================================================
# SECTION 5: NUMERICAL GRADIENT CHECKING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: NUMERICAL GRADIENT CHECKING")
print("=" * 70)

def numerical_gradient_check(net, X, y, eps=1e-5):
    """
    Compare analytical gradients with numerical gradients.
    """
    # Compute analytical gradients
    probs = net.forward(X)
    loss_orig = net.compute_loss(probs, y)
    net.backward(y)

    print("Checking gradients (should all be < 1e-5):")

    # Check W2
    numerical_dW2 = torch.zeros_like(net.W2)
    for i in range(net.W2.shape[0]):
        for j in range(net.W2.shape[1]):
            net.W2[i, j] += eps
            probs = net.forward(X)
            loss_plus = net.compute_loss(probs, y)

            net.W2[i, j] -= 2 * eps
            probs = net.forward(X)
            loss_minus = net.compute_loss(probs, y)

            net.W2[i, j] += eps  # Restore

            numerical_dW2[i, j] = (loss_plus - loss_minus) / (2 * eps)

    rel_error_W2 = torch.abs(net.dW2 - numerical_dW2) / (torch.abs(net.dW2) + torch.abs(numerical_dW2) + 1e-8)
    print(f"  W2 max relative error: {rel_error_W2.max().item():.2e}")

    # Check W1 (subset for speed)
    numerical_dW1 = torch.zeros_like(net.W1)
    for i in range(min(3, net.W1.shape[0])):
        for j in range(min(3, net.W1.shape[1])):
            net.W1[i, j] += eps
            probs = net.forward(X)
            loss_plus = net.compute_loss(probs, y)

            net.W1[i, j] -= 2 * eps
            probs = net.forward(X)
            loss_minus = net.compute_loss(probs, y)

            net.W1[i, j] += eps

            numerical_dW1[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Compare only the subset we computed
    subset_anal = net.dW1[:3, :3]
    subset_num = numerical_dW1[:3, :3]
    rel_error_W1 = torch.abs(subset_anal - subset_num) / (torch.abs(subset_anal) + torch.abs(subset_num) + 1e-8)
    print(f"  W1 max relative error (subset): {rel_error_W1.max().item():.2e}")

# Run gradient check
print("\n--- Numerical Gradient Check ---")
small_X = torch.randn(5, 4)
small_y = torch.randint(0, 3, (5,))
small_net = NeuralNetworkFromScratch(4, 6, 3)
numerical_gradient_check(small_net, small_X, small_y)


# =============================================================================
# SECTION 6: GRADIENT FLOW VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: GRADIENT FLOW ANALYSIS")
print("=" * 70)

def analyze_gradient_flow(model):
    """
    Analyze gradient magnitudes through a PyTorch model.
    Useful for detecting vanishing/exploding gradients.
    """
    print("\nGradient statistics per layer:")
    print("-" * 60)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name:30} | "
                  f"mean: {grad.mean():+.2e} | "
                  f"std: {grad.std():.2e} | "
                  f"max: {grad.abs().max():.2e}")

# Create a deeper network to see gradient flow
class DeepNetwork(nn.Module):
    def __init__(self, depth=5):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Linear(64, 64))

        self.output = nn.Linear(64, 10)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

# Test
deep_net = DeepNetwork(depth=5)
X = torch.randn(32, 64)
y = torch.randint(0, 10, (32,))

# Forward and backward
output = deep_net(X)
loss = nn.CrossEntropyLoss()(output, y)
loss.backward()

analyze_gradient_flow(deep_net)


# =============================================================================
# SECTION 7: BACKPROP THROUGH COMMON OPERATIONS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: BACKPROP THROUGH COMMON OPERATIONS")
print("=" * 70)

print("\n--- 7.1 Matrix Multiplication Gradient ---")

A = torch.randn(3, 4, requires_grad=True)
B = torch.randn(4, 5, requires_grad=True)
C = A @ B  # (3, 5)
loss = C.sum()
loss.backward()

print(f"A shape: {A.shape}, B shape: {B.shape}, C shape: {C.shape}")
print(f"dL/dA shape: {A.grad.shape}")
print(f"dL/dB shape: {B.grad.shape}")

# Manual verification
# dL/dA = dL/dC @ B^T = ones(3,5) @ B^T
# dL/dB = A^T @ dL/dC = A^T @ ones(3,5)
manual_dA = torch.ones(3, 5) @ B.T
manual_dB = A.T @ torch.ones(3, 5)
print(f"Manual dA matches: {torch.allclose(A.grad, manual_dA)}")
print(f"Manual dB matches: {torch.allclose(B.grad, manual_dB)}")


print("\n--- 7.2 Elementwise Operations Gradient ---")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Addition with broadcast
y = x + 10
y.sum().backward()
print(f"d(sum(x+10))/dx = {x.grad.tolist()}")  # [1, 1, 1]

# Multiplication
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x  # x²
y.sum().backward()
print(f"d(sum(x²))/dx = {x.grad.tolist()}")  # [2, 4, 6] = 2x


print("\n--- 7.3 Reduction Operations Gradient ---")

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Sum
y = x.sum()
y.backward()
print(f"d(sum(X))/dX:\n{x.grad}")  # All ones

# Mean
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x.mean()
y.backward()
print(f"d(mean(X))/dX:\n{x.grad}")  # All 1/4

# Max (gradient only flows through max element)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x.max()
y.backward()
print(f"d(max(X))/dX:\n{x.grad}")  # Only [1,1] position has gradient


print("\n" + "=" * 70)
print("END OF MODULE 2.1")
print("=" * 70)
