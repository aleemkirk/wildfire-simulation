"""
LSTM Models for Wildfire Burned Area Prediction

This module implements LSTM-based architectures for predicting wildfire spread
over time. The models are designed to handle temporal sequences of environmental
features and predict burned areas across a spatial grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell

    Combines convolutional operations with LSTM to preserve spatial structure
    while modeling temporal dependencies.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        """
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden state channels
            kernel_size: Size of convolutional kernel
            bias: Whether to use bias in convolutions
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Combined convolution for all gates (i, f, o, g)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor, h_c: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            h_c: Tuple of (hidden state, cell state) or None

        Returns:
            h_next: Next hidden state [B, hidden_dim, H, W]
            c_next: Next cell state [B, hidden_dim, H, W]
        """
        if h_c is None:
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3),
                          device=x.device, dtype=x.dtype)
            c = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3),
                          device=x.device, dtype=x.dtype)
        else:
            h, c = h_c

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        gates = self.conv(combined)

        # Split into individual gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate

        # Update cell and hidden states
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device):
        """Initialize hidden and cell states."""
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM

    Stacks multiple ConvLSTM layers for deeper temporal modeling.
    """
    def __init__(self, input_dim: int, hidden_dims: list, kernel_sizes: list,
                 num_layers: int, batch_first: bool = True, bias: bool = True,
                 return_all_layers: bool = False):
        """
        Args:
            input_dim: Number of input channels
            hidden_dims: List of hidden dimensions for each layer
            kernel_sizes: List of kernel sizes for each layer
            num_layers: Number of LSTM layers
            batch_first: If True, input shape is [B, T, C, H, W]
            bias: Whether to use bias in convolutions
            return_all_layers: If True, return outputs from all layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create ConvLSTM cells for each layer
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_sizes[i],
                bias=bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x: torch.Tensor, hidden_state: Optional[list] = None):
        """
        Args:
            x: Input tensor [B, T, C, H, W] if batch_first else [T, B, C, H, W]
            hidden_state: List of (h, c) tuples for each layer or None

        Returns:
            layer_output_list: List of outputs from each layer
            last_state_list: List of final (h, c) states from each layer
        """
        if not self.batch_first:
            # Convert to batch_first format
            x = x.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = x.size()

        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = []
            for i in range(self.num_layers):
                hidden_state.append(
                    self.cell_list[i].init_hidden(b, (h, w), x.device)
                )

        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  # [B, T, C, H, W]
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]

        return layer_output_list, last_state_list


class WildfireLSTM(nn.Module):
    """
    LSTM model for wildfire burned area prediction

    Architecture:
        - Initial feature encoding with Conv2D
        - ConvLSTM for temporal modeling
        - Decoder with Conv2D layers
        - Output layer for burned area prediction
    """
    def __init__(self,
                 input_channels: int,
                 hidden_dims: list = [64, 64, 32],
                 kernel_sizes: list = [3, 3, 3],
                 output_channels: int = 1,
                 dropout: float = 0.1):
        """
        Args:
            input_channels: Number of input feature channels
            hidden_dims: List of hidden dimensions for each ConvLSTM layer
            kernel_sizes: List of kernel sizes for each ConvLSTM layer
            output_channels: Number of output channels (1 for burned area)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Input encoder: Reduce channel dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # ConvLSTM layers
        self.convlstm = ConvLSTM(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Decoder: Predict burned area from LSTM outputs
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            output: Predicted burned areas [B, T, 1, H, W]
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Encode each timestep
        encoded = []
        for t in range(seq_len):
            enc = self.encoder(x[:, t])  # [B, hidden_dim, H, W]
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)  # [B, T, hidden_dim, H, W]

        # Process with ConvLSTM
        layer_output_list, _ = self.convlstm(encoded)
        lstm_out = layer_output_list[0]  # [B, T, hidden_dim, H, W]

        # Decode each timestep
        output = []
        for t in range(seq_len):
            dec = self.decoder(lstm_out[:, t])  # [B, 1, H, W]
            output.append(dec)
        output = torch.stack(output, dim=1)  # [B, T, 1, H, W]

        return output


class BidirectionalWildfireLSTM(nn.Module):
    """
    Bidirectional LSTM for wildfire prediction

    Processes sequences in both forward and backward directions to capture
    temporal context from both past and future timesteps.
    """
    def __init__(self,
                 input_channels: int,
                 hidden_dims: list = [64, 64, 32],
                 kernel_sizes: list = [3, 3, 3],
                 output_channels: int = 1,
                 dropout: float = 0.1):
        """
        Args:
            input_channels: Number of input feature channels
            hidden_dims: List of hidden dimensions for each ConvLSTM layer
            kernel_sizes: List of kernel sizes for each ConvLSTM layer
            output_channels: Number of output channels (1 for burned area)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dims = hidden_dims

        # Input encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Forward ConvLSTM
        self.convlstm_forward = ConvLSTM(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=len(hidden_dims),
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Backward ConvLSTM
        self.convlstm_backward = ConvLSTM(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=len(hidden_dims),
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        # Decoder (processes concatenated forward and backward features)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1] * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            output: Predicted burned areas [B, T, 1, H, W]
        """
        batch_size, seq_len, channels, height, width = x.size()

        # Encode each timestep
        encoded = []
        for t in range(seq_len):
            enc = self.encoder(x[:, t])
            encoded.append(enc)
        encoded = torch.stack(encoded, dim=1)  # [B, T, hidden_dim, H, W]

        # Forward LSTM
        forward_out, _ = self.convlstm_forward(encoded)
        forward_out = forward_out[0]  # [B, T, hidden_dim, H, W]

        # Backward LSTM (reverse temporal dimension)
        encoded_reversed = torch.flip(encoded, dims=[1])
        backward_out, _ = self.convlstm_backward(encoded_reversed)
        backward_out = backward_out[0]  # [B, T, hidden_dim, H, W]
        backward_out = torch.flip(backward_out, dims=[1])  # Reverse back

        # Concatenate forward and backward features
        combined = torch.cat([forward_out, backward_out], dim=2)  # [B, T, 2*hidden_dim, H, W]

        # Decode each timestep
        output = []
        for t in range(seq_len):
            dec = self.decoder(combined[:, t])
            output.append(dec)
        output = torch.stack(output, dim=1)  # [B, T, 1, H, W]

        return output


def test_wildfire_lstm():
    """Test the WildfireLSTM model with dummy data."""
    print("Testing WildfireLSTM...")

    # Create dummy input matching the dataset structure
    batch_size = 2
    seq_len = 10
    input_channels = 34  # Number of features in the dataset
    height = 64
    width = 64

    x = torch.randn(batch_size, seq_len, input_channels, height, width)

    # Create model
    model = WildfireLSTM(
        input_channels=input_channels,
        hidden_dims=[64, 64, 32],
        kernel_sizes=[3, 3, 3],
        output_channels=1
    )

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, seq_len, 1, height, width), \
        f"Expected output shape {(batch_size, seq_len, 1, height, width)}, got {output.shape}"

    print("✓ WildfireLSTM test passed!\n")


def test_bidirectional_lstm():
    """Test the BidirectionalWildfireLSTM model."""
    print("Testing BidirectionalWildfireLSTM...")

    batch_size = 2
    seq_len = 10
    input_channels = 34
    height = 64
    width = 64

    x = torch.randn(batch_size, seq_len, input_channels, height, width)

    model = BidirectionalWildfireLSTM(
        input_channels=input_channels,
        hidden_dims=[64, 64, 32],
        kernel_sizes=[3, 3, 3],
        output_channels=1
    )

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, seq_len, 1, height, width), \
        f"Expected output shape {(batch_size, seq_len, 1, height, width)}, got {output.shape}"

    print("✓ BidirectionalWildfireLSTM test passed!\n")


if __name__ == "__main__":
    test_wildfire_lstm()
    test_bidirectional_lstm()
    print("All tests passed!")
