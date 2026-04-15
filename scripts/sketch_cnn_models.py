"""Define CNN architectures for rasterized QuickDraw sketch classification."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv-BN-ReLU convenience block used by the deeper CNN."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a reusable conv-batchnorm-ReLU feature block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional feature block to one tensor batch."""
        return self.block(x)


class ResidualBlock(nn.Module):
    """A basic ResNet-style block for sketch classification."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Create a residual block with an optional downsampling shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two residual convolutions and add the shortcut path."""
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class QuickDrawCNN(nn.Module):
    """A compact LeNet-style CNN for 28x28 grayscale QuickDraw images."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_size: int = 28,
        conv_channels: tuple[int, int] = (32, 64),
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        """Build the compact LeNet-style baseline sketch classifier."""
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")

        c1, c2 = conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        flattened_dim = c2 * math.floor(input_size / 4) * math.floor(input_size / 4)
        if flattened_dim <= 0:
            raise ValueError(
                f"input_size={input_size} becomes too small after pooling; expected at least 4."
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the baseline sketch classifier."""
        x = self.features(x)
        return self.classifier(x)


class QuickDrawDeepCNN(nn.Module):
    """
    A stronger 3-stage CNN with batch normalization and adaptive pooling.

    This is intended as a drop-in upgrade over the original LeNet-style model:
    - deeper feature extractor
    - batch normalization for stabler optimization
    - adaptive average pooling to reduce parameter count in the classifier
    """

    def __init__(
        self,
        num_classes: int,
        *,
        input_size: int = 28,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        """Build the deeper non-residual CNN used as a stronger baseline."""
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if len(conv_channels) != 3:
            raise ValueError(
                "QuickDrawDeepCNN expects exactly 3 conv channel values, "
                f"got {conv_channels}."
            )

        c1, c2, c3 = conv_channels
        self.features = nn.Sequential(
            ConvBlock(1, c1),
            ConvBlock(c1, c1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout * 0.25),

            ConvBlock(c1, c2),
            ConvBlock(c2, c2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout * 0.25),

            ConvBlock(c2, c3),
            ConvBlock(c3, c3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the deeper sketch CNN."""
        x = self.features(x)
        return self.classifier(x)


class QuickDrawResNet(nn.Module):
    """
    A small ResNet-style CNN for QuickDraw sketches.

    Compared with the plain deep CNN, this architecture adds residual
    connections so we can increase depth without making optimization as brittle.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        input_size: int = 28,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        hidden_dim: int = 512,
        dropout: float = 0.35,
    ) -> None:
        """Build the ResNet-style sketch classifier with three feature stages."""
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}.")
        if len(conv_channels) != 3:
            raise ValueError(
                "QuickDrawResNet expects exactly 3 conv channel values, "
                f"got {conv_channels}."
            )

        c1, c2, c3 = conv_channels
        block_dropout = dropout * 0.15

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c1, stride=1, dropout=block_dropout),
            ResidualBlock(c1, c1, stride=1, dropout=block_dropout),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, dropout=block_dropout),
            ResidualBlock(c2, c2, stride=1, dropout=block_dropout),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, dropout=block_dropout),
            ResidualBlock(c3, c3, stride=1, dropout=block_dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the ResNet-style sketch classifier."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)
