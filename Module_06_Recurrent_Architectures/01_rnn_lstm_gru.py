"""
Module 6.1: Recurrent Neural Networks - RNN, LSTM, GRU
Complete implementations from scratch and with PyTorch

Covers:
- Vanilla RNN from scratch
- LSTM with gates explained
- GRU simplified architecture
- Bidirectional RNNs
- Stacked RNNs
- Sequence classification and generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import math

print("=" * 70)
print("Module 6.1: Recurrent Neural Networks - RNN, LSTM, GRU")
print("=" * 70)


# ===========================================================================
# Section 1: Vanilla RNN from Scratch
# ===========================================================================
print("\n" + "=" * 70)
print("Section 1: Vanilla RNN from Scratch")
print("=" * 70)


class VanillaRNNCell(nn.Module):
    """
    Single RNN cell implementation.

    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input to hidden weights
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) / np.sqrt(input_size))
        # Hidden to hidden weights
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size))
        # Bias
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)

        Returns:
            h_new: New hidden state (batch_size, hidden_size)
        """
        batch_size = x.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
        h_new = torch.tanh(
            x @ self.W_xh + h_prev @ self.W_hh + self.b_h
        )

        return h_new


class VanillaRNN(nn.Module):
    """
    Full RNN that processes sequences.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of RNN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(VanillaRNNCell(cell_input_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            h_0: Initial hidden state (num_layers, batch_size, hidden_size)

        Returns:
            output: All hidden states (batch_size, seq_len, hidden_size)
            h_n: Final hidden state (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device
            )

        # Process sequence
        h_states = list(h_0)  # Hidden state for each layer
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            for layer_idx, cell in enumerate(self.cells):
                h_states[layer_idx] = cell(x_t, h_states[layer_idx])
                x_t = h_states[layer_idx]  # Output becomes input to next layer

            outputs.append(h_states[-1])  # Collect output from last layer

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        h_n = torch.stack(h_states, dim=0)    # (num_layers, batch_size, hidden_size)

        return output, h_n


# Test Vanilla RNN
print("\n--- Testing Vanilla RNN ---")
batch_size, seq_len, input_size, hidden_size = 4, 10, 8, 16

# Create sample data
x = torch.randn(batch_size, seq_len, input_size)

# Our implementation
rnn_custom = VanillaRNN(input_size, hidden_size, num_layers=2)
output_custom, h_n_custom = rnn_custom(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output_custom.shape}")
print(f"Final hidden shape: {h_n_custom.shape}")

# Compare with PyTorch
rnn_pytorch = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
output_pytorch, h_n_pytorch = rnn_pytorch(x)

print(f"\nPyTorch RNN output shape: {output_pytorch.shape}")
print(f"PyTorch RNN hidden shape: {h_n_pytorch.shape}")
print("Shapes match!" if output_custom.shape == output_pytorch.shape else "Shape mismatch!")


# ===========================================================================
# Section 2: LSTM from Scratch
# ===========================================================================
print("\n" + "=" * 70)
print("Section 2: LSTM from Scratch")
print("=" * 70)


class LSTMCell(nn.Module):
    """
    LSTM cell implementation with all gates.

    Gates:
        f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)    # Forget gate
        i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)    # Input gate
        o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)    # Output gate
        c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c) # Candidate cell

    Cell update:
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

    Output:
        h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weights for efficiency (4 gates at once)
        # Order: input, forget, cell, output (PyTorch convention)
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) / np.sqrt(input_size))
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        # Initialize forget gate bias to 1 (critical for LSTM!)
        self._init_forget_gate_bias()

    def _init_forget_gate_bias(self):
        """Initialize forget gate bias to 1 for better gradient flow."""
        # In PyTorch convention: [input, forget, cell, output]
        # Forget gate is at indices [hidden_size:2*hidden_size]
        with torch.no_grad():
            self.b_ih[self.hidden_size:2*self.hidden_size] = 1.0
            self.b_hh[self.hidden_size:2*self.hidden_size] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch_size, input_size)
            state: Tuple of (h_prev, c_prev), each (batch_size, hidden_size)

        Returns:
            h_new: New hidden state (batch_size, hidden_size)
            (h_new, c_new): New state tuple
        """
        batch_size = x.size(0)

        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = state

        # Compute all gates at once for efficiency
        # gates = W_ih @ x + W_hh @ h + bias
        gates = (x @ self.W_ih.T + self.b_ih) + (h_prev @ self.W_hh.T + self.b_hh)

        # Split into 4 gates
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)

        # Apply activations
        i_t = torch.sigmoid(i_gate)  # Input gate
        f_t = torch.sigmoid(f_gate)  # Forget gate
        c_tilde = torch.tanh(c_gate)  # Candidate cell
        o_t = torch.sigmoid(o_gate)  # Output gate

        # Cell state update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        c_new = f_t * c_prev + i_t * c_tilde

        # Hidden state: h_t = o_t ⊙ tanh(c_t)
        h_new = o_t * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class LSTM(nn.Module):
    """
    Full LSTM that processes sequences.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Create cells for each layer and direction
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(LSTMCell(layer_input_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            state: Tuple of (h_0, c_0), each (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            output: All hidden states (batch_size, seq_len, hidden_size * num_directions)
            (h_n, c_n): Final states
        """
        batch_size, seq_len, _ = x.size()

        # Initialize states
        if state is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=x.device
            )
            c_0 = torch.zeros_like(h_0)
        else:
            h_0, c_0 = state

        # Process each layer
        layer_input = x
        h_n_list = []
        c_n_list = []

        for layer in range(self.num_layers):
            # Forward direction
            forward_cell = self.cells[layer * self.num_directions]
            h_forward = h_0[layer * self.num_directions]
            c_forward = c_0[layer * self.num_directions]

            forward_outputs = []
            for t in range(seq_len):
                h_forward, (h_forward, c_forward) = forward_cell(
                    layer_input[:, t, :], (h_forward, c_forward)
                )
                forward_outputs.append(h_forward)

            h_n_list.append(h_forward)
            c_n_list.append(c_forward)

            if self.bidirectional:
                # Backward direction
                backward_cell = self.cells[layer * self.num_directions + 1]
                h_backward = h_0[layer * self.num_directions + 1]
                c_backward = c_0[layer * self.num_directions + 1]

                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    h_backward, (h_backward, c_backward) = backward_cell(
                        layer_input[:, t, :], (h_backward, c_backward)
                    )
                    backward_outputs.insert(0, h_backward)

                h_n_list.append(h_backward)
                c_n_list.append(c_backward)

                # Concatenate forward and backward
                layer_output = torch.stack([
                    torch.cat([forward_outputs[t], backward_outputs[t]], dim=-1)
                    for t in range(seq_len)
                ], dim=1)
            else:
                layer_output = torch.stack(forward_outputs, dim=1)

            # Apply dropout between layers (not on last layer)
            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        output = layer_output
        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)

        return output, (h_n, c_n)


# Test LSTM
print("\n--- Testing LSTM ---")
batch_size, seq_len, input_size, hidden_size = 4, 10, 8, 16

x = torch.randn(batch_size, seq_len, input_size)

# Our implementation
lstm_custom = LSTM(input_size, hidden_size, num_layers=2)
output_custom, (h_n_custom, c_n_custom) = lstm_custom(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output_custom.shape}")
print(f"Final hidden shape: {h_n_custom.shape}")
print(f"Final cell shape: {c_n_custom.shape}")

# Compare with PyTorch
lstm_pytorch = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
output_pytorch, (h_n_pytorch, c_n_pytorch) = lstm_pytorch(x)

print(f"\nPyTorch LSTM output shape: {output_pytorch.shape}")
print("Shapes match!" if output_custom.shape == output_pytorch.shape else "Shape mismatch!")


# ===========================================================================
# Section 3: GRU from Scratch
# ===========================================================================
print("\n" + "=" * 70)
print("Section 3: GRU from Scratch")
print("=" * 70)


class GRUCell(nn.Module):
    """
    GRU cell implementation.

    Gates:
        r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)    # Reset gate
        z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)    # Update gate
        h̃_t = tanh(W_h @ [r_t ⊙ h_{t-1}, x_t] + b_h)  # Candidate

    Output:
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weights for reset and update gates
        # Order: reset, update, new (PyTorch convention)
        self.W_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size) / np.sqrt(input_size))
        self.W_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) / np.sqrt(hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(3 * hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)

        Returns:
            h_new: New hidden state (batch_size, hidden_size)
        """
        batch_size = x.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Compute gates
        x_gates = x @ self.W_ih.T + self.b_ih
        h_gates = h_prev @ self.W_hh.T + self.b_hh

        # Split
        x_r, x_z, x_n = x_gates.chunk(3, dim=1)
        h_r, h_z, h_n = h_gates.chunk(3, dim=1)

        # Reset and update gates
        r_t = torch.sigmoid(x_r + h_r)  # Reset gate
        z_t = torch.sigmoid(x_z + h_z)  # Update gate

        # Candidate hidden state (with reset applied)
        h_tilde = torch.tanh(x_n + r_t * h_n)

        # Final hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_new = (1 - z_t) * h_prev + z_t * h_tilde

        return h_new


class GRU(nn.Module):
    """
    Full GRU that processes sequences.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Create cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(GRUCell(layer_input_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            h_0: Initial hidden state (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            output: All hidden states (batch_size, seq_len, hidden_size * num_directions)
            h_n: Final hidden state
        """
        batch_size, seq_len, _ = x.size()

        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=x.device
            )

        layer_input = x
        h_n_list = []

        for layer in range(self.num_layers):
            # Forward direction
            forward_cell = self.cells[layer * self.num_directions]
            h_forward = h_0[layer * self.num_directions]

            forward_outputs = []
            for t in range(seq_len):
                h_forward = forward_cell(layer_input[:, t, :], h_forward)
                forward_outputs.append(h_forward)

            h_n_list.append(h_forward)

            if self.bidirectional:
                # Backward direction
                backward_cell = self.cells[layer * self.num_directions + 1]
                h_backward = h_0[layer * self.num_directions + 1]

                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    h_backward = backward_cell(layer_input[:, t, :], h_backward)
                    backward_outputs.insert(0, h_backward)

                h_n_list.append(h_backward)

                layer_output = torch.stack([
                    torch.cat([forward_outputs[t], backward_outputs[t]], dim=-1)
                    for t in range(seq_len)
                ], dim=1)
            else:
                layer_output = torch.stack(forward_outputs, dim=1)

            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        output = layer_output
        h_n = torch.stack(h_n_list, dim=0)

        return output, h_n


# Test GRU
print("\n--- Testing GRU ---")
x = torch.randn(batch_size, seq_len, input_size)

gru_custom = GRU(input_size, hidden_size, num_layers=2)
output_custom, h_n_custom = gru_custom(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output_custom.shape}")
print(f"Final hidden shape: {h_n_custom.shape}")

# Compare with PyTorch
gru_pytorch = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
output_pytorch, h_n_pytorch = gru_pytorch(x)

print(f"\nPyTorch GRU output shape: {output_pytorch.shape}")
print("Shapes match!" if output_custom.shape == output_pytorch.shape else "Shape mismatch!")


# ===========================================================================
# Section 4: Vanishing Gradient Demonstration
# ===========================================================================
print("\n" + "=" * 70)
print("Section 4: Vanishing Gradient Demonstration")
print("=" * 70)


def demonstrate_vanishing_gradient():
    """Show how gradients vanish in vanilla RNN vs LSTM."""

    seq_lengths = [10, 25, 50, 100, 200]
    input_size, hidden_size = 16, 32

    results = {'RNN': [], 'LSTM': [], 'GRU': []}

    for seq_len in seq_lengths:
        # Create input that only has signal at the start
        x = torch.zeros(1, seq_len, input_size)
        x[:, 0, :] = torch.randn(1, input_size)  # Signal only at t=0

        for name, model_class in [('RNN', nn.RNN), ('LSTM', nn.LSTM), ('GRU', nn.GRU)]:
            model = model_class(input_size, hidden_size, batch_first=True)
            model.train()

            x_input = x.clone().requires_grad_(True)

            if name == 'LSTM':
                output, (h_n, c_n) = model(x_input)
            else:
                output, h_n = model(x_input)

            # Use final output
            loss = output[:, -1, :].sum()
            loss.backward()

            # Gradient w.r.t. input at t=0
            grad_magnitude = x_input.grad[:, 0, :].abs().mean().item()
            results[name].append(grad_magnitude)

    # Plot results
    plt.figure(figsize=(10, 6))
    for name, values in results.items():
        plt.plot(seq_lengths, values, 'o-', label=name, linewidth=2, markersize=8)

    plt.xlabel('Sequence Length')
    plt.ylabel('Gradient Magnitude at t=0')
    plt.title('Vanishing Gradient: RNN vs LSTM vs GRU')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/anuragmishra/Documents/Zero_to_GPT/Module_06_Recurrent_Architectures/vanishing_gradient.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nGradient magnitudes at t=0 for different sequence lengths:")
    print(f"{'Length':<10}", end='')
    for name in results.keys():
        print(f"{name:<15}", end='')
    print()

    for i, length in enumerate(seq_lengths):
        print(f"{length:<10}", end='')
        for name in results.keys():
            print(f"{results[name][i]:<15.2e}", end='')
        print()

    print("\nObservation: RNN gradients vanish exponentially with sequence length")
    print("LSTM and GRU maintain better gradient flow due to gating mechanisms")

demonstrate_vanishing_gradient()


# ===========================================================================
# Section 5: Bidirectional RNN
# ===========================================================================
print("\n" + "=" * 70)
print("Section 5: Bidirectional RNN")
print("=" * 70)


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM implementation.

    Processes sequence in both directions:
    - Forward: x_1 -> x_2 -> ... -> x_T
    - Backward: x_T -> x_{T-1} -> ... -> x_1

    Output: [forward_output, backward_output] concatenated
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)

        Returns:
            output: (batch_size, seq_len, 2 * hidden_size)
        """
        # Forward direction
        forward_out, _ = self.forward_lstm(x)

        # Backward direction (reverse, process, reverse back)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, _ = self.backward_lstm(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])

        # Concatenate
        output = torch.cat([forward_out, backward_out], dim=-1)
        return output


# Test bidirectional
print("\n--- Testing Bidirectional LSTM ---")
x = torch.randn(4, 10, 8)

bilstm_custom = BidirectionalLSTM(8, 16)
output_custom = bilstm_custom(x)

# Compare with PyTorch
bilstm_pytorch = nn.LSTM(8, 16, batch_first=True, bidirectional=True)
output_pytorch, _ = bilstm_pytorch(x)

print(f"Input shape: {x.shape}")
print(f"Custom BiLSTM output: {output_custom.shape}")
print(f"PyTorch BiLSTM output: {output_pytorch.shape}")
print("Shapes match!" if output_custom.shape == output_pytorch.shape else "Shape mismatch!")


# ===========================================================================
# Section 6: Sequence Classification
# ===========================================================================
print("\n" + "=" * 70)
print("Section 6: Sequence Classification Example")
print("=" * 70)


class SentimentClassifier(nn.Module):
    """
    Sentiment classification using LSTM.
    Many-to-one architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Token indices (batch_size, seq_len)
            lengths: Actual sequence lengths for each sample

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embed
        embedded = self.embedding(x)  # (batch, seq, embed)

        # LSTM
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, c_n) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (h_n, c_n) = self.lstm(embedded)

        # Use final hidden state (concatenate forward and backward for bidirectional)
        if self.lstm.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden)
            # Last layer forward: h_n[-2]
            # Last layer backward: h_n[-1]
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]

        # Classify
        logits = self.classifier(hidden)
        return logits


# Test sentiment classifier
print("\n--- Testing Sentiment Classifier ---")
vocab_size, embedding_dim, hidden_size, num_classes = 10000, 128, 256, 2

classifier = SentimentClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_classes=num_classes
)

# Sample batch
batch_size, seq_len = 8, 50
x = torch.randint(1, vocab_size, (batch_size, seq_len))
lengths = torch.randint(20, seq_len, (batch_size,))

logits = classifier(x, lengths)
print(f"Input shape: {x.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")


# ===========================================================================
# Section 7: Language Model (Character-level)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 7: Character-Level Language Model")
print("=" * 70)


class CharLSTM(nn.Module):
    """
    Character-level language model using LSTM.
    Many-to-many architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

        # Tie embedding and output weights (optional, but common)
        if embedding_dim == hidden_size:
            self.fc.weight = self.embedding.weight

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tokens (batch_size, seq_len)
            hidden: Previous hidden state

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden: New hidden state
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        generated = start_tokens.clone()
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                # Get next token prediction
                logits, hidden = self(generated[:, -1:], hidden)
                logits = logits[:, -1, :] / temperature

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

        return generated


# Test language model
print("\n--- Testing Character Language Model ---")
vocab_size, embedding_dim, hidden_size = 128, 256, 256  # ASCII + special tokens

char_lm = CharLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=2
)

# Sample batch
x = torch.randint(0, vocab_size, (4, 50))
logits, hidden = char_lm(x)

print(f"Input shape: {x.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Hidden state shapes: h={hidden[0].shape}, c={hidden[1].shape}")

# Generate
start = torch.randint(0, vocab_size, (1, 5))
generated = char_lm.generate(start, max_length=20)
print(f"Generated sequence shape: {generated.shape}")


# ===========================================================================
# Section 8: Sequence-to-Sequence (Encoder-Decoder)
# ===========================================================================
print("\n" + "=" * 70)
print("Section 8: Sequence-to-Sequence (Encoder-Decoder)")
print("=" * 70)


class Seq2SeqEncoder(nn.Module):
    """Encoder for seq2seq model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Project bidirectional hidden states to decoder size
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: Source sequence (batch_size, src_len)
            src_lengths: Actual lengths

        Returns:
            encoder_outputs: (batch_size, src_len, hidden_size * 2)
            hidden: (h_n, c_n) projected for decoder
        """
        embedded = self.embedding(src)

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (h_n, c_n) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (h_n, c_n) = self.lstm(embedded)

        # Reshape hidden states: (num_layers * 2, batch, hidden) -> (num_layers, batch, hidden * 2)
        batch_size = src.size(0)
        num_layers = self.lstm.num_layers

        h_n = h_n.view(num_layers, 2, batch_size, -1)
        h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=-1)
        h_n = self.hidden_projection(h_n)

        c_n = c_n.view(num_layers, 2, batch_size, -1)
        c_n = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=-1)
        c_n = self.cell_projection(c_n)

        return outputs, (h_n, c_n)


class Seq2SeqDecoder(nn.Module):
    """Decoder for seq2seq model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            tgt: Target sequence (batch_size, tgt_len)
            hidden: Initial hidden state from encoder

        Returns:
            logits: (batch_size, tgt_len, vocab_size)
            hidden: Final hidden state
        """
        embedded = self.embedding(tgt)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.encoder = Seq2SeqEncoder(
            src_vocab_size, embedding_dim, hidden_size, num_layers, dropout
        )
        self.decoder = Seq2SeqDecoder(
            tgt_vocab_size, embedding_dim, hidden_size, num_layers, dropout
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_lengths: Source lengths

        Returns:
            logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        logits, _ = self.decoder(tgt, hidden)
        return logits


# Test Seq2Seq
print("\n--- Testing Seq2Seq Model ---")
src_vocab, tgt_vocab = 5000, 8000
embedding_dim, hidden_size = 256, 512

model = Seq2Seq(
    src_vocab_size=src_vocab,
    tgt_vocab_size=tgt_vocab,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
)

src = torch.randint(1, src_vocab, (4, 20))
tgt = torch.randint(1, tgt_vocab, (4, 15))

logits = model(src, tgt)
print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===========================================================================
# Section 9: Truncated BPTT
# ===========================================================================
print("\n" + "=" * 70)
print("Section 9: Truncated Backpropagation Through Time")
print("=" * 70)


class TruncatedBPTTTrainer:
    """
    Trainer that uses truncated backpropagation through time
    for training on long sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 100
    ):
        self.model = model
        self.chunk_size = chunk_size

    def train_epoch(
        self,
        data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Train on a long sequence using truncated BPTT.

        Args:
            data: Long sequence (batch_size, total_length)
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0
        num_chunks = 0
        hidden = None

        # Process sequence in chunks
        for start in range(0, data.size(1) - 1, self.chunk_size):
            end = min(start + self.chunk_size, data.size(1) - 1)

            # Get chunk
            x_chunk = data[:, start:end]
            y_chunk = data[:, start+1:end+1]

            # Detach hidden state from previous computation graph
            # This is the "truncation" - gradients don't flow beyond chunk boundary
            if hidden is not None:
                if isinstance(hidden, tuple):
                    hidden = tuple(h.detach() for h in hidden)
                else:
                    hidden = hidden.detach()

            # Forward pass
            optimizer.zero_grad()
            logits, hidden = self.model(x_chunk, hidden)

            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y_chunk.reshape(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_chunks += 1

        return total_loss / num_chunks


# Demonstrate truncated BPTT
print("\n--- Demonstrating Truncated BPTT ---")
vocab_size = 1000
model = CharLSTM(vocab_size, 128, 256)
trainer = TruncatedBPTTTrainer(model, chunk_size=50)

# Create long sequence
long_sequence = torch.randint(0, vocab_size, (2, 500))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train one epoch
loss = trainer.train_epoch(long_sequence, optimizer, criterion)
print(f"Sequence length: {long_sequence.size(1)}")
print(f"Chunk size: {trainer.chunk_size}")
print(f"Number of chunks: {(long_sequence.size(1) - 1) // trainer.chunk_size}")
print(f"Average loss: {loss:.4f}")


# ===========================================================================
# Section 10: Packed Sequences for Variable-Length Batches
# ===========================================================================
print("\n" + "=" * 70)
print("Section 10: Packed Sequences for Variable-Length Batches")
print("=" * 70)


def demonstrate_packed_sequences():
    """Show how to efficiently process variable-length sequences."""

    # Create variable-length sequences
    sequences = [
        torch.randn(15, 8),  # Length 15
        torch.randn(10, 8),  # Length 10
        torch.randn(20, 8),  # Length 20
        torch.randn(5, 8),   # Length 5
    ]
    lengths = torch.tensor([15, 10, 20, 5])

    # Pad sequences
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    print(f"Padded shape: {padded.shape}")  # (4, 20, 8)

    # Create LSTM
    lstm = nn.LSTM(8, 16, batch_first=True)

    # Method 1: Naive (processes padding too)
    output_naive, _ = lstm(padded)
    print(f"Naive output shape: {output_naive.shape}")

    # Method 2: Using pack_padded_sequence
    packed = nn.utils.rnn.pack_padded_sequence(
        padded, lengths, batch_first=True, enforce_sorted=False
    )
    print(f"Packed data shape: {packed.data.shape}")  # Only non-padded elements
    print(f"Packed batch_sizes: {packed.batch_sizes}")

    output_packed, (h_n, c_n) = lstm(packed)

    # Unpack
    output_unpacked, output_lengths = nn.utils.rnn.pad_packed_sequence(
        output_packed, batch_first=True
    )
    print(f"Unpacked output shape: {output_unpacked.shape}")

    # Get actual last hidden state for each sequence
    batch_size = len(lengths)
    last_outputs = []
    for i in range(batch_size):
        actual_length = lengths[i].item()
        last_outputs.append(output_unpacked[i, actual_length - 1, :])

    last_hidden = torch.stack(last_outputs)
    print(f"Last hidden states shape: {last_hidden.shape}")

    print("\nPacked sequences are more efficient:")
    print(f"  - Naive: processes {padded.numel()} elements")
    print(f"  - Packed: processes {packed.data.numel()} elements")
    print(f"  - Savings: {1 - packed.data.numel()/padded.numel():.1%}")

demonstrate_packed_sequences()


# ===========================================================================
# Section 11: Comparison Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Section 11: RNN vs LSTM vs GRU Comparison")
print("=" * 70)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compare_architectures():
    """Compare RNN, LSTM, and GRU."""

    input_size, hidden_size = 64, 128

    models = {
        'RNN': nn.RNN(input_size, hidden_size, batch_first=True),
        'LSTM': nn.LSTM(input_size, hidden_size, batch_first=True),
        'GRU': nn.GRU(input_size, hidden_size, batch_first=True)
    }

    print("\nParameter Counts:")
    print("-" * 40)
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters")

    # Theoretical parameter counts
    print("\nTheoretical Formula:")
    print("-" * 40)

    # RNN: W_ih (input_size * hidden_size) + W_hh (hidden_size * hidden_size) + biases
    rnn_params = input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size
    print(f"RNN: (I×H + H×H + 2H) = {rnn_params:,}")

    # LSTM: 4 gates, each with W_ih, W_hh, and biases
    lstm_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    print(f"LSTM: 4×(I×H + H×H + 2H) = {lstm_params:,}")

    # GRU: 3 gates
    gru_params = 3 * (input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    print(f"GRU: 3×(I×H + H×H + 2H) = {gru_params:,}")

    # Speed comparison
    print("\nSpeed Comparison (forward pass):")
    print("-" * 40)

    import time

    x = torch.randn(32, 100, input_size)

    for name, model in models.items():
        model.eval()

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Time
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        elapsed = time.time() - start

        print(f"{name}: {elapsed:.3f}s for 100 iterations")

compare_architectures()


# ===========================================================================
# Section 12: Best Practices Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Section 12: Best Practices Summary")
print("=" * 70)

print("""
RNN/LSTM/GRU Best Practices:
============================

1. Architecture Selection:
   - Start with GRU (faster, fewer parameters)
   - Use LSTM if GRU underperforms
   - Avoid vanilla RNN (vanishing gradients)

2. LSTM Initialization:
   - Set forget gate bias to 1 (critical!)
   - Use orthogonal initialization for recurrent weights
   - Xavier/Glorot for input weights

3. Training:
   - Use gradient clipping (max_norm=1.0 to 5.0)
   - Use truncated BPTT for long sequences
   - Learning rate: 0.001 to 0.01 typical

4. Regularization:
   - Dropout between layers, NOT on recurrent connections
   - Weight decay: 1e-5 to 1e-4
   - Consider variational dropout

5. Variable Length Sequences:
   - Use pack_padded_sequence for efficiency
   - Mask loss for padded positions
   - Get correct last hidden state per sequence

6. Bidirectional:
   - Use when full context is available
   - NOT for autoregressive generation
   - Good for classification/encoding

7. Modern Alternatives:
   - Transformers dominate for NLP
   - RNNs still useful for:
     * Real-time processing
     * Very long sequences (with linear attention)
     * Limited memory devices
""")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("Module 6.1 Summary: RNN, LSTM, GRU")
print("=" * 70)

print("""
Key Takeaways:
==============

1. Vanilla RNN:
   - h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1})
   - Simple but vanishing gradient problem
   - Can't learn long-range dependencies

2. LSTM (Long Short-Term Memory):
   - Cell state highway: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
   - Forget, Input, Output gates
   - Solves vanishing gradient via additive updates
   - Initialize forget gate bias = 1

3. GRU (Gated Recurrent Unit):
   - Simplified LSTM with 2 gates (reset, update)
   - h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
   - ~75% parameters of LSTM, similar performance

4. Why LSTM/GRU Work:
   - Additive cell updates (not multiplicative)
   - Gradient: ∂c_t/∂c_{t-1} = f_t (not weight matrix)
   - When f_t ≈ 1, gradient flows unchanged

5. Practical Notes:
   - GRU is often good enough and faster
   - Bidirectional for full context (not generation)
   - Truncated BPTT for long sequences
   - Pack sequences for variable lengths

6. Historical Context:
   - RNNs largely replaced by Transformers for NLP
   - Attention mechanism solves long-range better
   - But RNNs still useful in some applications

Files created:
- 01_rnn_lstm_gru.md: Theory and concepts
- 01_rnn_lstm_gru.py: This implementation file
- vanishing_gradient.png: Visualization
""")

print("\nModule 6.1 complete!")
