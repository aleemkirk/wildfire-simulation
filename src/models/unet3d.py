"""
3D U-Net Model for Wildfire Spread Prediction

This module implements a 3D U-Net architecture for spatiotemporal wildfire prediction.
The model takes in a sequence of environmental features and predicts fire spread
over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DoubleConv3D(nn.Module):
    """
    Double 3D Convolution block: Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Downsampling block: MaxPool3D -> DoubleConv3D
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    Upsampling block: ConvTranspose3D -> Concatenate -> DoubleConv3D
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled feature map from previous layer
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch (if input size is not divisible by 2^depth)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for Wildfire Spread Prediction

    Architecture:
        Input: [B, C_in, T, H, W] - Batch, Channels, Time, Height, Width
        Output: [B, C_out, T, H, W] - Predicted fire spread probabilities
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 depth: int = 4):
        """
        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output channels (1 for binary fire/no-fire)
            base_channels: Number of channels in first layer (doubles each level)
            depth: Depth of U-Net (number of downsampling/upsampling levels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Initial convolution
        self.inc = DoubleConv3D(in_channels, base_channels)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            self.down_blocks.append(Down3D(in_ch, out_ch))

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = base_channels * (2 ** (depth - i))
            out_ch = base_channels * (2 ** (depth - i - 1))
            self.up_blocks.append(Up3D(in_ch, out_ch))

        # Output convolution
        self.outc = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            out: Output tensor [B, C_out, T, H, W]
        """
        # Initial convolution
        x1 = self.inc(x)

        # Encoder with skip connections
        skip_connections = [x1]
        x_down = x1
        for down in self.down_blocks:
            x_down = down(x_down)
            skip_connections.append(x_down)

        # Decoder with skip connections
        x_up = skip_connections[-1]
        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 2)]
            x_up = up(x_up, skip)

        # Output
        out = self.outc(x_up)

        return out


class UNet3DWithAttention(UNet3D):
    """
    3D U-Net with Attention Gates

    Attention gates help the model focus on relevant spatial-temporal regions
    for better fire spread prediction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add attention gates (to be implemented if needed)
        # This is a placeholder for future enhancement
        pass


def test_unet3d():
    """Test the U-Net3D model with dummy data."""
    # Create dummy input
    batch_size = 2
    in_channels = 30  # Number of features
    time_steps = 10
    height = 64
    width = 64

    x = torch.randn(batch_size, in_channels, time_steps, height, width)

    # Create model
    model = UNet3D(in_channels=in_channels, out_channels=1, base_channels=32, depth=3)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert output.shape == (batch_size, 1, time_steps, height, width), \
        f"Expected output shape {(batch_size, 1, time_steps, height, width)}, got {output.shape}"

    print("✓ Model test passed!")


if __name__ == "__main__":
    test_unet3d()
